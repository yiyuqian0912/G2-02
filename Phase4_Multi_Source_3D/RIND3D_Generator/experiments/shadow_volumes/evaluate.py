"""Gated stratified bricks, complete-field pilots and multi-source reuse."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from rind3d.geometry import Oracle,load_geometry,centres
from rind3d.pack import RINDPack,atomic_json
from .benchmark import ROOT,freeze,run_field,compare
from .cases import SOURCES
from .renderer import SharedGeometry,SharedSolid,geometry_split


def gate(name):
    v=json.loads((ROOT/'small'/name/'result.json').read_text())
    if not v.get('small_correctness_passed') or len(v['independent_full_references'])!=3 or any(not r['passed'] for r in v['independent_full_references']):raise ValueError('Smaller-field gate closed')
    return v


def stratified(name):
    gate(name);out=ROOT/'stratified'/name;out.mkdir(parents=True,exist_ok=False);case,bundle=freeze(name)
    start=time.perf_counter();base=Oracle(bundle);shared=SharedGeometry(base);a=shared.bind(SOURCES[0]);rng=np.random.default_rng(902)
    # One random brick in each of the 2x2x2 spatial octants, plus central
    # obstacle neighborhoods; all are voxel-by-voxel independent comparisons.
    origins=set()
    for x in (0,8):
        for y in (0,8):
            for z in (0,8):origins.add(tuple((np.array([x,y,z])+rng.integers(0,8,3))*64))
    meta=json.loads((case/'case.json').read_text())
    for obj in meta['objects']:
        center=np.array(obj['center']);i=np.minimum(15,(center*16).astype(int))
        origins.add(tuple(i*64));origins.add(tuple(np.clip(i+[2,1,-1],0,15)*64))
    records=[]
    for origin in sorted(origins):
        t=time.perf_counter();y,m,c=a.render(SOURCES[0],origin,64,1024);dt=time.perf_counter()-t
        p=centres(origin,64,1024);t=time.perf_counter();ry,rm,stats=a.native.points(p,True);reference_dt=time.perf_counter()-t
        dy=int(np.sum(y.ravel()!=ry));dm=int(np.sum(m.ravel()!=rm));assert not (dy or dm)
        control=None
        if shared.old:
            t=time.perf_counter();by,bm,_=a.render(SOURCES[0],origin,64,1024,old_mode=True);control=time.perf_counter()-t
            assert np.array_equal(y,by) and np.array_equal(m,bm)
        records.append({'origin':list(map(int,origin)),'C_seconds':dt,'B_native_seconds':control,'reference_seconds':reference_dt,'response_differences':dy,'solid_differences':dm,'certified_voxels':c,'independent_reference_exact_clips':stats['reference_exact_clips']})
        atomic_json(out/'progress.json',records)
    atomic_json(out/'result.json',{'family':name,'resolution':1024,'stratified_bricks':records,'complete_field':False,'checked_voxels':len(records)*64**3,'passed':True,'wall_seconds':time.perf_counter()-start,'production_ready':False})


def full(name,method,seconds=60):
    gate(name)
    if not json.loads((ROOT/'stratified'/name/'result.json').read_text()).get('passed'):raise ValueError('Stratified correctness gate closed')
    path=ROOT/'full'/name/(f'{method}-source0'+(f'-budget{int(seconds)}' if seconds!=60 else ''))
    if (path/'result.json').exists():raise FileExistsError('Preserve prior fields; use a fresh RIND_SHADOW_OUTPUT')
    case,bundle=freeze(name);t=time.perf_counter();cpu=time.process_time();base=Oracle(bundle);init=time.perf_counter()-t;shared=SharedGeometry(base);a=shared.bind(SOURCES[0])
    row=run_field(base,a,path,1024,method,seconds)
    row.update(geometry_init_seconds=init,total_wall_including_setup=time.perf_counter()-t,total_cpu_including_setup=time.process_time()-cpu,split=geometry_split(shared.geometry_id),cold_setup_measured=True)
    atomic_json(path/'result.json',row)


def reuse(name,n=1024,variant=''):
    gate(name);case,bundle=freeze(name);root=ROOT/f'reuse-{n}'/(name+('-'+variant if variant else ''));root.mkdir(parents=True,exist_ok=False)
    cold=[]
    for rank,source in enumerate(SOURCES):
        t=time.perf_counter();cpu=time.process_time();base=Oracle(bundle);shared=SharedGeometry(base);a=shared.bind(source)
        row=run_field(base,a,root/f'cold-source{rank}',n,'C',60)
        row.update(total_wall_including_setup=time.perf_counter()-t,total_cpu_including_setup=time.process_time()-cpu)
        cold.append(row)
    t=time.perf_counter();cpu=time.process_time();base=Oracle(bundle);shared=SharedGeometry(base);geometry_setup=time.perf_counter()-t;warm=[];mask=None
    try:
        for rank,source in enumerate(SOURCES):
            step=time.perf_counter();a=shared.bind(source,solid=mask)
            row=run_field(base,a,root/f'shared-source{rank}',n,'D',60);row['step_wall_with_shadow_setup']=time.perf_counter()-step;warm.append(row)
            if rank==0:
                if not row['complete']:break
                mask=SharedSolid(shared,source,root/'shared-source0/solid.rindb')
    finally:
        if mask:mask.close()
    total=time.perf_counter()-t;totalcpu=time.process_time()-cpu
    verification_t=time.perf_counter();verification_cpu=time.process_time()
    for rank,row in enumerate(warm):
        if row['complete'] and cold[rank]['complete']:compare(root/f'cold-source{rank}',root/f'shared-source{rank}')
    sources=[{'geometry_id':shared.geometry_id,'source_id':a['source_id'],'source_rank':i,'split':geometry_split(shared.geometry_id)} for i,a in enumerate(warm)]
    assert len({r['source_id'] for r in sources})==len(sources) and len({r['split'] for r in sources})==1
    report={'family':name,'resolution':n,'cold_sources':cold,'shared_sources':warm,'geometry_id':shared.geometry_id,'source_records':sources,'split_group':'geometry_id','geometry_setup_once_seconds':geometry_setup,'shared_total_wall':total,'shared_total_cpu':totalcpu,'cold_total_wall':sum(r['total_wall_including_setup'] for r in cold),'cold_total_cpu':sum(r['total_cpu_including_setup'] for r in cold),'complete':len(warm)==3 and all(r['complete'] for r in cold+warm),'storage_note':'Current RINDPack source-bound headers require separate mask files; reused mask values are reserialized without changing format','production_ready':False,'cross_method_verification_wall':time.perf_counter()-verification_t,'cross_method_verification_cpu':time.process_time()-verification_cpu,'verification_excluded_from_generation_totals':True}
    atomic_json(root/'result.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stratified','full','reuse']);p.add_argument('--case',required=True);p.add_argument('--method',default='C',choices=['A','B','B_native','C']);p.add_argument('--resolution',type=int,default=1024);p.add_argument('--seconds',type=float,default=60);p.add_argument('--variant',default='');a=p.parse_args()
    if a.stage=='stratified':stratified(a.case)
    elif a.stage=='full':full(a.case,a.method,a.seconds)
    else:reuse(a.case,a.resolution,a.variant)


if __name__=='__main__':main()
