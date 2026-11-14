"""Sequential, bounded controls and 1/2/4/8-source geometry groups."""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
from rind3d.geometry import centres
from rind3d.pack import RINDPack, atomic_json
from rind3d.stream import scene_identity
from experiments.shadow_volumes.cases import SOURCES
from experiments.shadow_volumes.renderer import geometry_split
from .benchmark import ROOT,OLD,NAMES,frozen,Control,field,compare
from .cpu import GeometryGroup

GROUP_SOURCES=np.vstack([SOURCES,[[.09,.87,.91],[.88,.90,.11],[.12,.14,.89],[.91,.89,.88],[.45,.07,.92]]]).astype('<f8')


class SharedMask:
    def __init__(self,shared,source,path):
        self.geometry_id=shared.geometry_id;self.policy_hash=shared.policy_hash;self.lock=threading.Lock()
        self.pack=RINDPack(path,verify_file=True)
        if self.pack.identity!=scene_identity(shared,source) or self.pack.policy!=shared.policy_hash:
            self.pack.close();raise ValueError('Source-bound mask identity mismatch')
        self.shape=self.pack.shape
    def read_window(self,*args):
        with self.lock:return self.pack.read_window(*args)
    def close(self):self.pack.close()


def stratified(name,runtime=None):
    from .gpu import GPUGroup
    path=ROOT/'stratified'/name
    if (path/'result.json').exists():return
    assert json.loads((ROOT/'small-cpu'/name/'result.json').read_text())['small_correctness_passed']
    if runtime:assert json.loads((ROOT/'small-gpu'/name/'result.json').read_text())['small_correctness_passed']
    path.mkdir(parents=True,exist_ok=True);g=GeometryGroup(frozen(name));source=SOURCES[0]
    methods={'CPU_clip':g.bind(source),'CPU_projected':g.bind(source,projected=True),'C_shadow':Control(g,source,'C_shadow')}
    if g.old:methods['B_native']=Control(g,source,'B_native')
    rng=np.random.default_rng(902);origins=set()
    for x in (0,8):
        for y in (0,8):
            for z in (0,8):origins.add(tuple((np.array([x,y,z])+rng.integers(0,8,3))*64))
    case=json.loads((ROOT/'cases'/name/'case.json').read_text())
    for obj in case['objects'][::max(1,len(case['objects'])//8)]:
        i=np.clip((np.array(obj['center'])*16).astype(int),0,15);origins.add(tuple(i*64))
    # Additional unaligned ROI exercises the same half-open sample-centre domain.
    origins.add((391,477,503));origins=sorted(origins)
    rows=[];ref=methods['C_shadow'].native
    if runtime:gg=GPUGroup(g,runtime);gpu=gg.bind(source)
    for origin in origins:
        points=centres(origin,64,1024);t=time.perf_counter();ry,rm,_=ref.points(points,True);reference=time.perf_counter()-t
        row={'origin':list(map(int,origin)),'independent_reference_seconds':reference,'methods':{}}
        for name2,a in methods.items():
            t=time.perf_counter();y,m,c=a.render(source,origin,64,1024)
            assert np.array_equal(y.ravel(),ry) and np.array_equal(m.ravel(),rm)
            row['methods'][name2]={'seconds':time.perf_counter()-t,'certificate_covered_voxels':c,'passed':True}
        if runtime:
            t=time.perf_counter();pair=gpu.packed_batch([origin],64,1024)[0]
            assert all(np.array_equal(np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little'),v) for raw,v in zip(pair,(ry,rm)))
            row['methods']['GPU']={'seconds':time.perf_counter()-t,'passed':True}
        rows.append(row);atomic_json(path/'progress.json',rows)
    if runtime:gpu.close();gg.close()
    atomic_json(path/'result.json',{'family':name,'passed':True,'bricks':rows,'checked_voxels':len(rows)*64**3,'complete_field':False})


def full_all(runtime):
    from .gpu import GPUGroup
    # Same-source, same-geometry methods run sequentially, never concurrently.
    for name in NAMES:
        stratified(name,runtime)
        g=GeometryGroup(frozen(name))
        methods=['B_native'] if g.old else []
        methods+=['C_shadow','CPU_clip','CPU_projected','GPU']
        for method in methods:
            out=ROOT/'full'/name/method
            if (out/'result.json').exists():continue
            print('START full',name,method,flush=True)
            if method=='GPU':
                gg=GPUGroup(g,runtime);a=gg.bind(SOURCES[0])
                try:row=field(a,out,1024,method,120)
                finally:a.close();gg.close()
            else:
                a=g.bind(SOURCES[0],projected=method=='CPU_projected') if method.startswith('CPU') else Control(g,SOURCES[0],method)
                row=field(a,out,1024,method,120)
            if row['complete']:
                compare(out,out)
                reference=ROOT/'full'/name/'C_shadow'
                if (reference/'result.json').exists() and json.loads((reference/'result.json').read_text())['complete']:
                    compare(out,reference)
            print('END full',name,method,row['complete'],round(row['wall_seconds'],3),flush=True)
        records=[json.loads((ROOT/'full'/name/m/'result.json').read_text()) for m in methods]
        # Include the baseline-first case in final comparisons too.
        complete=[m for m,r in zip(methods,records) if r['complete']]
        for method in complete:compare(ROOT/'full'/name/complete[0],ROOT/'full'/name/method)
        atomic_json(ROOT/'full'/name/'comparison.json',{'passed':True,'complete_methods':complete,'all_response_and_mask_bits_equal':True})


def reuse(name,n,method,runtime=None):
    from .gpu import GPUGroup
    path=ROOT/f'reuse-{n}'/name/method
    if (path/'result.json').exists():return
    path.mkdir(parents=True,exist_ok=True);bundle=frozen(name);cold=[];warm=[]
    for rank,source in enumerate(GROUP_SOURCES):
        t=time.perf_counter();cpu=time.process_time();g=GeometryGroup(bundle)
        if method=='GPU':
            gg=GPUGroup(g,runtime);a=gg.bind(source)
        else:a=g.bind(source,projected=method=='CPU_projected')
        row=field(a,path/f'cold-{rank}',n,method)
        if method=='GPU':a.close();gg.close()
        row['group_step_wall_seconds']=time.perf_counter()-t;row['group_step_cpu_seconds']=time.process_time()-cpu;cold.append(row)
        assert row['complete'];print('REUSE cold',n,name,method,rank,flush=True)
    t=time.perf_counter();cpu=time.process_time();g=GeometryGroup(bundle)
    if method=='GPU':gg=GPUGroup(g,runtime)
    setup=time.perf_counter()-t;setup_cpu=time.process_time()-cpu;mask=None
    for rank,source in enumerate(GROUP_SOURCES):
        t=time.perf_counter();cpu=time.process_time()
        a=gg.bind(source,reuse_mask=True) if method=='GPU' else g.bind(source,solid=mask,projected=method=='CPU_projected')
        row=field(a,path/f'warm-{rank}',n,method)
        if method=='GPU':a.close()
        elif rank==0:mask=SharedMask(g,source,path/'warm-0/solid.rindb')
        row['group_step_wall_seconds']=time.perf_counter()-t;row['group_step_cpu_seconds']=time.process_time()-cpu;warm.append(row)
        assert row['complete'];print('REUSE warm',n,name,method,rank,flush=True)
    if mask:mask.close()
    if method=='GPU':gg.close()
    validation_t=time.perf_counter()
    for rank in range(8):compare(path/f'cold-{rank}',path/f'warm-{rank}')
    # Each source-bound file is a distinct file and has a distinct header identity.
    inodes=[(path/f'warm-{rank}/solid.rindb').stat().st_ino for rank in range(8)];assert len(set(inodes))==8
    ids=[]
    for rank in range(8):
        with RINDPack(path/f'warm-{rank}/solid.rindb') as p:ids.append(p.identity)
    assert len(set(ids))==8
    prefixes=[]
    for count in (1,2,4,8):
        cw=sum(r['group_step_wall_seconds'] for r in cold[:count]);cc=sum(r['group_step_cpu_seconds'] for r in cold[:count])
        ww=setup+sum(r['group_step_wall_seconds'] for r in warm[:count]);wc=setup_cpu+sum(r['group_step_cpu_seconds'] for r in warm[:count])
        prefixes.append({'sources':count,'cold_total_wall_seconds':cw,'shared_total_wall_seconds':ww,'cold_total_cpu_seconds':cc,
                         'shared_total_cpu_seconds':wc,'shared_wall_per_source':ww/count,'shared_cpu_per_source':wc/count,
                         'cold_wall_per_source':cw/count,'cold_cpu_per_source':cc/count})
    records=[{'geometry_id':g.geometry_id,'source_id':r['source_id'],'source':source.tolist(),'split':geometry_split(g.geometry_id)} for r,source in zip(warm,GROUP_SOURCES)]
    assert len({r['split'] for r in records})==1
    report={'family':name,'resolution':n,'method':method,'cold':cold,'shared':warm,'prefixes':prefixes,'source_records':records,
            'source_independent_geometry_setup_once_seconds':setup,'cross_method_comparison_seconds_excluded':time.perf_counter()-validation_t,
            'split_group':'geometry_id','rindpack_headers_source_bound':True,'no_hard_links':True,
            'runtime_context_compile_shared_by_both_controls':method=='GPU','all_sources_complete':True,'production_ready':False}
    atomic_json(path/'result.json',report)


def main():
    from .cuda_runtime import Runtime
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['full','reuse']);p.add_argument('--case',default='rotated_box');p.add_argument('--resolution',type=int,default=1024);p.add_argument('--method',default='GPU');args=p.parse_args()
    with Runtime(ROOT/'cuda') as r:
        if args.stage=='full':full_all(r)
        else:reuse(args.case,args.resolution,args.method,r)


if __name__=='__main__':main()
