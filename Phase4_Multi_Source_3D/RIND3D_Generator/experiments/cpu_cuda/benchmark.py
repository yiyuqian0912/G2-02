"""Frozen controls and bounded complete-field experiments. Never publishes READY."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind, make_leaf
from rind.storage import CanonicalBundle
from rind3d.geometry import load_geometry, save_geometry, centres, BudgetExhausted
from rind3d.pack import RINDPack, RINDPackWriter, atomic_json, sha256_file
from rind3d.stream import stream_response, scene_identity
from experiments.shadow_volumes.renderer import strings, shadow_planes, native_stream, geometry_split
from experiments.shadow_volumes.cases import SOURCES, rotation
from experiments.shadow_volumes import _shadow_native as old
from .cpu import GeometryGroup

ROOT = Path(os.environ.get('RIND_EFFICIENCY_OUTPUT','outputs/cpu-cuda-efficiency'))
OLD = Path('outputs/shadow-volume-experiment')
NAMES = ('box','rotated_box','thin_slab','prism','wedge','poly8','poly16','poly32','multiscale4','multiscale12','multiscale24','corridor','cluster')


def frozen(name):
    path = ROOT/'cases'/name
    if not path.exists():
        path.mkdir(parents=True)
        if name != 'multiscale24':
            for f in ('geometry.npz','case.json'): shutil.copy2(OLD/'cases'/name/f, path/f)
        else:
            # Additional frozen stress case, not substituted for any old case.
            rng = np.random.default_rng(20261101); b = CanonicalBundle(3); roots=[]; objects=[]
            for i in range(24):
                c = rng.uniform(.22,.78,3); size = (.018,.035,.075,.14)[i%4]
                matrix = rotation(rng)@np.diag(rng.uniform(.7,1.2,3)*size)
                leaf = make_leaf('affine_box_v1',3,center=c,matrix=matrix); b.add_leaf(leaf); roots.append(leaf.content_hash)
                objects.append({'center':c.tolist(),'matrix':matrix.tolist(),'declared_planes':6})
            node = CSGNode(3,CSGOpcode.UNION,tuple(roots)); b.add_node(node); b.add_geometry(Geometry(3,RootKind.REF,node.content_hash))
            save_geometry(path/'geometry.npz',b)
            atomic_json(path/'case.json',{'family':name,'geometry_id':b.geometries[0].content_hash.hex(),'sources':SOURCES.tolist(),'objects':objects,'new_frozen_stress_case':True})
    return load_geometry(path/'geometry.npz')


class Control:
    def __init__(self, shared, source, mode):
        self.shared=shared; self.source=np.array(source,dtype='<f8',copy=True); self.source.setflags(write=False)
        self.source_id=hashlib.sha256(shared.geometry.content_hash+self.source.tobytes()).hexdigest()
        self.work=Counter(); self.mode=mode; self.solid=None
        shared.bind(self.source)  # complete exact free-source validation, outside field timer
        t=time.perf_counter(); planes=[shadow_planes(b,source) for b in shared.bodies]
        self.native=old.Model(old.Geometry(strings(shared.bodies),shared.old),strings(planes),source.tolist())
        self.setup_seconds=time.perf_counter()-t
        if mode=='B_native' and not shared.old: raise ValueError('B_native has no general convex support; not emulated')
    def __getattr__(self,name): return getattr(self.shared,name)
    @property
    def metrics(self): return dict(self.work)
    def validate_source(self,source,budget=None):
        if not np.array_equal(source,self.source):raise ValueError('Source identity mismatch')
        # The caller's fixed source was independently validated by GeometryGroup.
        # Do not build a new geometry/source model inside each field timer.
    def report(self):
        return {'method':self.mode,'work':dict(self.work),'source_setup_seconds':self.setup_seconds,
                'production_ready':False,'canonical_initialization_seconds':self.shared.initialization_seconds}
    def render(self,source,origin,size,n,*,budget=None,leaf_side=8,old_mode=False):
        if not np.array_equal(source,self.source): raise ValueError('Source identity mismatch')
        if budget: budget.check()
        y,m,stats=self.native.raster(list(origin),size,n,leaf_side,self.mode=='B_native'); self.work.update(stats)
        if budget: budget.check(stats['point_rays']+stats['solid_points'])
        for which,a in enumerate((y,m)):
            bad=np.flatnonzero(a.ravel()<0)
            if len(bad):
                p=centres(origin,size,n)[bad]; base=self.shared.canonical
                a.ravel()[bad]=base.response(source,p) if which==0 else base.contains(p)
        assert not np.any(y&m)
        return y.astype(bool),m.astype(bool),stats['certificate_covered_voxels']


def gpu_stream(a,path,n,seconds):
    from rind3d.information import InformationIndex
    identity=scene_identity(a,a.source); path.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter(); g=(n+63)//64; side=min(64,n); origins=[(x*64,y*64,z*64) for z in range(g) for y in range(g) for x in range(g)]
    writers=[RINDPackWriter(path/(f+'.rindb'),(n,)*3,identity=identity,policy=a.policy_hash) for f in ('response','solid')]
    first=None; compression=0.; committed=0; before=dict(a.runtime.timings); error=None; minimum_free=a.runtime.memory()['free_bytes']
    try:
        for start in range(0,len(origins),a.group.batch):
            if time.perf_counter()-t>seconds: raise BudgetExhausted('time')
            values=a.packed_batch(origins[start:start+a.group.batch],side,n)
            minimum_free=min(minimum_free,a.runtime.memory()['free_bytes'])
            # A blocking oracle call can exceed a soft per-field work budget.
            # Do not commit a successful field after discovering that overrun.
            # The outer process watchdog bounds a still-blocking call.
            if time.perf_counter()-t>seconds:raise BudgetExhausted('time')
            for j,pair in enumerate(values):
                tick=time.perf_counter()
                for w,raw in zip(writers,pair):w.write(start+j,raw)
                compression+=time.perf_counter()-tick; committed+=1
                if first is None:first=time.perf_counter()-t
            for w in writers:w.checkpoint()
            atomic_json(path/'progress.json',{'complete':False,'committed':committed,'total':len(origins),'seconds':time.perf_counter()-t})
        for w in writers:w.finish()
    except RuntimeError as exc:
        error=exc
        for w in writers:w.checkpoint()
    result={'complete':error is None,'bricks':committed,'first_brick_seconds':first,'compression_seconds':compression,
            'runtime_timings':{k:v-before[k] for k,v in a.runtime.timings.items()},'field_wall_seconds':time.perf_counter()-t,
            'minimum_device_free_bytes':minimum_free,'device_total_bytes':a.runtime.memory()['total_bytes']}
    if error: result.update(reason=str(error),error_type=type(error).__name__)
    else:
        tick=time.perf_counter()
        with RINDPack(path/'response.rindb') as y,RINDPack(path/'solid.rindb') as m:
            index=InformationIndex.build(y,m); size=index.save(path/'information.rindi')
        result['index_seconds']=time.perf_counter()-tick; result['index_bytes']=size
    return result


def field(a,path,n,method,seconds=120):
    path=Path(path)
    if (path/'result.json').exists():
        previous=json.loads((path/'result.json').read_text())
        expected=(a.shared.geometry_id,a.source_id,n,method)
        actual=(previous.get('geometry_id'),previous.get('source_id'),previous.get('resolution'),previous.get('method'))
        if actual!=expected:raise ValueError('Cached experiment geometry/source/resolution/method identity mismatch')
        if method=='GPU' and previous.get('acceleration',{}).get('fallback_backend','canonical_cgal')!=a.fallback_backend:
            raise ValueError('Cached GPU fallback-backend identity mismatch')
        if previous.get('complete'):
            for name in ('response','solid'):
                with RINDPack(path/(name+'.rindb'),verify_file=True) as pack:
                    if pack.identity!=scene_identity(a,a.source) or pack.policy!=a.policy_hash or pack.shape!=(n,)*3:
                        raise ValueError('Cached RINDPack identity mismatch')
        return previous
    if path.exists(): raise FileExistsError(f'Unfinished prior run preserved at {path}; use a new path')
    t=time.perf_counter(); cpu=time.process_time(); work=dict(a.work); canonical_before=a.shared.initialization_seconds
    row={'method':method,'resolution':n,'geometry_id':a.shared.geometry_id,'source_id':a.source_id,
         'source':a.source.tolist(),'budget_seconds':seconds,'production_ready':False}
    try:
        if method=='GPU':
            value=gpu_stream(a,path,n,seconds); row.update(value)
        else:
            cfg={'resolution':n,'workers':2,'queue_size':4,'leaf_side':8,'generation_seconds':seconds,
                 'generation_points':100_000_000_000,'build_information_index':True,'field_acceleration':'baseline'}
            first=[None]; original=RINDPackWriter.write
            def write(obj,*args):
                r=original(obj,*args)
                if first[0] is None and obj.path.name=='solid.rindb':first[0]=time.perf_counter()-t
                return r
            RINDPackWriter.write=write
            try:
                with native_stream():value=stream_response(a,a.source,path,cfg)
            finally:RINDPackWriter.write=original
            row.update(complete=True,generation=value,first_brick_seconds=first[0])
    except RuntimeError as e: row.update(complete=False,error_type=type(e).__name__,reason=str(e))
    row.update(wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,
               geometry_setup_seconds=a.shared.setup_seconds,source_setup_seconds=a.setup_seconds,
               canonical_setup_this_field=a.shared.initialization_seconds-canonical_before,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               work={k:v-work.get(k,0) for k,v in a.work.items()},acceleration=a.report())
    if method=='GPU':row['runtime']=a.runtime.info
    if row['complete']:
        row['field_bytes']={f:(path/(f+'.rindb')).stat().st_size for f in ('response','solid')}
    path.mkdir(parents=True,exist_ok=True); atomic_json(path/'result.json',row)
    return row


def compare(p,q):
    for name in ('response','solid'):
        with RINDPack(Path(p)/(name+'.rindb'),verify_file=True) as a,RINDPack(Path(q)/(name+'.rindb'),verify_file=True) as b:
            assert a.identity==b.identity and a.shape==b.shape and a.policy==b.policy
            g=a.grid[0]
            for z in range(g):
                for y in range(g):
                    for x in range(g):
                        if a.read_brick_packed((x,y,z))!=b.read_brick_packed((x,y,z)):raise AssertionError(f'{name} voxel mismatch {x,y,z}')


def small(name,gpu=False):
    out=ROOT/('small-gpu' if gpu else 'small-cpu')/name
    if (out/'result.json').exists():return
    out.mkdir(parents=True,exist_ok=True); shared=GeometryGroup(frozen(name)); rows=[]; references=[]
    controls={'CPU_clip':shared.bind(SOURCES[0]),'CPU_projected':shared.bind(SOURCES[0],projected=True),'C_shadow':Control(shared,SOURCES[0],'C_shadow')}
    if shared.old: controls['B_native']=Control(shared,SOURCES[0],'B_native')
    if gpu:
        from .cuda_runtime import Runtime
        from .gpu import GPUGroup
        runtime=Runtime(ROOT/'cuda'); group=GPUGroup(shared,runtime); a=group.bind(SOURCES[0]); active={'GPU':a}
    else:active=controls
    for n in (32,64,128):
        for method,a in active.items():
            row=field(a,out/f'{method}-{n}',n,method); rows.append(row)
            assert row['complete'],row
        p=out/f'{next(iter(active))}-{n}'
        for method in active:compare(p,out/f'{method}-{n}')
        if gpu:
            compare(p,ROOT/'small-cpu'/name/f'CPU_clip-{n}')
        # Entire field independently rerendered by the UNCHANGED old parametric
        # interval/GMP implementation, not by reading previously computed truth.
        t=time.perf_counter(); ry=controls['C_shadow'].native; count=0
        with RINDPack(p/'response.rindb') as y,RINDPack(p/'solid.rindb') as m:
            for z in range(0,n,16):
                for v in range(0,n,16):
                    for x in range(0,n,16):
                        yy,mm,stats=ry.points(centres((x,v,z),16,n),True)
                        assert np.array_equal(y.read_window(x,v,z,16).ravel(),yy)
                        assert np.array_equal(m.read_window(x,v,z,16).ravel(),mm)
                        count+=16**3
        references.append({'resolution':n,'all_voxels':count,'passed':True,'seconds':time.perf_counter()-t,
                           'reference':'Unchanged independent canonical-halfspace parametric clipping with GMP fallback; not exhaustive CGAL replay'})
        atomic_json(out/'progress.json',{'rows':rows,'independent_references':references})
    if gpu: a.close(); group.close(); runtime.close()
    atomic_json(out/'result.json',{'family':name,'geometry_id':shared.geometry_id,'small_correctness_passed':True,'rows':rows,'independent_references':references,'production_ready':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['small','full']);p.add_argument('--case',default='box',choices=NAMES)
    p.add_argument('--method',default='CPU_clip',choices=['CPU_clip','CPU_projected','C_shadow','B_native','GPU']);p.add_argument('--gpu',action='store_true');p.add_argument('--seconds',type=float,default=120)
    a=p.parse_args()
    if a.stage=='small':small(a.case,a.gpu);return
    gate=ROOT/('small-gpu' if a.method=='GPU' else 'small-cpu')/a.case/'result.json'
    assert json.loads(gate.read_text())['small_correctness_passed'],'Small correctness gate closed'
    group=GeometryGroup(frozen(a.case))
    if a.method=='GPU':
        from .cuda_runtime import Runtime
        from .gpu import GPUGroup
        with Runtime(ROOT/'cuda') as r:
            g=GPUGroup(group,r);m=g.bind(SOURCES[0])
            try:field(m,ROOT/'full'/a.case/a.method,1024,a.method,a.seconds)
            finally:m.close();g.close()
    else:
        m=group.bind(SOURCES[0],projected=a.method=='CPU_projected') if a.method.startswith('CPU') else Control(group,SOURCES[0],a.method)
        field(m,ROOT/'full'/a.case/a.method,1024,a.method,a.seconds)


if __name__=='__main__':main()
