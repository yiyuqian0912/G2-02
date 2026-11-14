"""Frozen, bounded A/B/C controls; full fields require sealing and verification."""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
from rind3d.geometry import Oracle,load_geometry,save_geometry,Budget,BudgetExhausted,NumericalUnresolved,Aborted,centres
from rind3d.stream import stream_response
from rind3d.pack import RINDPack,atomic_json
from .cases import NAMES,SOURCES,make_case
from .renderer import SharedGeometry,native_stream,geometry_split
from . import _shadow_native

ROOT=Path(os.environ.get('RIND_SHADOW_OUTPUT','outputs/shadow-volume-experiment'))
NATIVE_SHA256=hashlib.sha256(Path(_shadow_native.__file__).read_bytes()).hexdigest()
CFG={'workers':2,'queue_size':4,'leaf_side':8,'generation_seconds':30,'generation_points':10_000_000_000,'field_acceleration':'baseline','build_information_index':True}


def freeze(name):
    out=ROOT/'cases'/name;out.mkdir(parents=True,exist_ok=True)
    if not (out/'case.json').exists():
        b,meta=make_case(name);save_geometry(out/'geometry.npz',b);meta.update(geometry_id=b.geometries[0].content_hash.hex(),split=geometry_split(b.geometries[0].content_hash.hex()),split_group='geometry_id')
        atomic_json(out/'case.json',meta)
    return out,load_geometry(out/'geometry.npz')


def run_field(base,a,path,n,method,seconds=30):
    if (path/'result.json').exists():return json.loads((path/'result.json').read_text())
    path.mkdir(parents=True,exist_ok=False);cfg={**CFG,'resolution':n,'generation_seconds':seconds};t=time.perf_counter();cpu=time.process_time();first=[None];bricks=[];times=[]
    from rind3d import stream
    writer=stream.RINDPackWriter.write
    def write(obj,index,raw):
        value=writer(obj,index,raw)
        if obj.path.name=='solid.rindb':
            now=time.perf_counter()-t
            if first[0] is None:first[0]=now
            times.append(now-(bricks[-1][1] if bricks else 0));bricks.append((int(index),now))
        return value
    stream.RINDPackWriter.write=write;old_filter=base.filter
    before=dict(a.work)
    row={'method':method,'resolution':n,'source_id':a.source_id,'geometry_id':a.shared.geometry_id,'is_dataset':False,'production_ready':False,'seconds_budget':seconds,'native_sha256':NATIVE_SHA256}
    try:
        if method=='A':base.filter=None
        active=a if method in ('C','C_python','B_native','D') else base
        if method in ('C','B_native','D'):
            with native_stream(old_mode=method=='B_native'):value=stream_response(active,a.source,path,cfg)
        else:value=stream_response(active,a.source,path,cfg)
        with RINDPack(path/'response.rindb',verify_file=True) as response,RINDPack(path/'solid.rindb',verify_file=True) as solid:
            assert response.verify_all()==solid.verify_all()==((n+63)//64)**3
        row.update(status='COMPLETE',complete=True,generation=value)
    except (RuntimeError,ValueError) as exc:
        row.update(status='BUDGET_INCOMPLETE' if isinstance(exc,BudgetExhausted) else 'NUMERICAL_UNRESOLVED' if isinstance(exc,NumericalUnresolved) else 'REFERENCE_ABORTED' if isinstance(exc,Aborted) else 'ERROR',complete=False,reason=str(exc),error_type=type(exc).__name__)
    finally:base.filter=old_filter;stream.RINDPackWriter.write=writer
    row.update(wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,first_brick_seconds=first[0],committed_brick_pairs=len(bricks),brick_seconds=times,acceleration=a.report())
    row['work_this_field']={k:v-before.get(k,0) for k,v in a.work.items()}
    atomic_json(path/'result.json',row);return row


def reference_field(a,path,n):
    t=time.perf_counter();cpu=time.process_time();diff_y=diff_m=exact=checked=0
    with RINDPack(path/'response.rindb') as y,RINDPack(path/'solid.rindb') as m:
        for z in range(0,n,16):
            for v in range(0,n,16):
                for x in range(0,n,16):
                    p=centres((x,v,z),16,n);ry,rm,stats=a.native.points(p,True);exact+=stats['reference_exact_clips']
                    diff_y+=int(np.sum(y.read_window(x,v,z,16).ravel()!=ry));diff_m+=int(np.sum(m.read_window(x,v,z,16).ravel()!=rm));checked+=len(p)
    result={'reference':'independent interval+GMP parametric clipping of canonical halfspaces; NOT exhaustive CGAL replay','resolution':n,'checked_voxels':checked,'response_differences':diff_y,'solid_differences':diff_m,'exact_clip_fallbacks':exact,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'passed':not (diff_y or diff_m)}
    atomic_json(path/'independent-reference.json',result)
    if not result['passed']:raise AssertionError('Independent reference mismatch')
    return result


def compare(a,b):
    count=0
    for field in ('response','solid'):
        with RINDPack(a/(field+'.rindb')) as x,RINDPack(b/(field+'.rindb')) as y:
            for bz in range(x.grid[0]):
                for by in range(x.grid[1]):
                    for bx in range(x.grid[2]):count+=int(np.unpackbits(np.frombuffer(x.read_brick_packed((bx,by,bz)),dtype=np.uint8)^np.frombuffer(y.read_brick_packed((bx,by,bz)),dtype=np.uint8)).sum())
    if count:raise AssertionError('Field bit mismatch')
    return count


def small(name):
    case,bundle=freeze(name);out=ROOT/'small'/name;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cpu=time.process_time()
    t=time.perf_counter();base=Oracle(bundle);init=time.perf_counter()-t;shared=SharedGeometry(base);a=shared.bind(SOURCES[0]);rows=[];refs=[]
    for n in (32,64,128):
        row=run_field(base,a,out/f'C-{n}',n,'C',60);rows.append(row)
        if not row['complete']:raise RuntimeError('New small field did not complete')
        refs.append(reference_field(a,out/f'C-{n}',n))
        row=run_field(base,a,out/f'C_python-{n}',n,'C_python',30);rows.append(row)
        if row['complete']:compare(out/f'C-{n}',out/f'C_python-{n}')
        if base.filter is not None:
            for method in ('B','B_native'):
                row=run_field(base,a,out/f'{method}-{n}',n,method,30);rows.append(row)
                if row['complete']:compare(out/f'C-{n}',out/f'{method}-{n}')
    # Direct existing canonical backend checks, separately from the all-voxel independent clipping reference.
    rng=np.random.default_rng(4242);points=rng.uniform(.03,.97,(24,3));t=time.perf_counter()
    r=base.reference.segment_intersect(np.repeat(SOURCES[:1],len(points),axis=0),points);m=base.reference.classify_points(points)
    assert r.valid.all() and m.valid.all();assert np.array_equal(a.response(SOURCES[0],points),r.value==0) and np.array_equal(a.contains(points),m.value!=0)
    canonical={'queries':len(points),'response_differences':0,'solid_differences':0,'wall_seconds':time.perf_counter()-t,'exhaustive_full_field':False}
    for n in (32,64,128):
        row=run_field(base,a,out/f'A-{n}',n,'A',5);rows.append(row)
        if row['complete']:compare(out/f'C-{n}',out/f'A-{n}')
    result={'family':name,'geometry_id':shared.geometry_id,'geometry_init_seconds':init,'canonical_capability':base.capability,'shadow':a.report(),'rows':rows,'independent_full_references':refs,'canonical_checks':canonical,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'small_correctness_passed':True,'production_ready':False}
    atomic_json(out/'result.json',result);print(name,'small PASS',round(result['wall_seconds'],2),flush=True)


def run_jobs(names,stage,workers):
    running=[];todo=list(names);done=[]
    while todo or running:
        while todo and len(running)<workers:
            name=todo.pop(0);out=ROOT/stage/name
            if (out/'result.json').exists():done.append(name);continue
            log=ROOT/f'{stage}-{name}.log';handle=log.open('w');p=subprocess.Popen([sys.executable,'-m',__package__+'.benchmark','--one',stage,'--case',name],stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            running.append((name,p,handle,time.perf_counter()))
        for item in running[:]:
            name,p,handle,t=item;code=p.poll()
            if code is None and time.perf_counter()-t>600:p.kill();code=p.wait()
            if code is None:continue
            handle.close();running.remove(item);done.append({'case':name,'exit_code':code});print(done[-1],flush=True)
        atomic_json(ROOT/f'{stage}-progress.json',{'done':done,'remaining':todo,'running':[v[0] for v in running]})
        if running:time.sleep(.5)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('--one',choices=['small']);p.add_argument('--case',choices=NAMES);p.add_argument('--cases',nargs='+',default=list(NAMES));p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if args.one:small(args.case)
    else:run_jobs(args.cases,'small',args.workers)


if __name__=='__main__':main()
