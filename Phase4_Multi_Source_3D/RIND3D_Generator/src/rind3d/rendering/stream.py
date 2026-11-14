"""Transactional packed-brick streaming shared by certified CPU and CUDA.

The writer, file format, CRC/SHA, partial payloads and journal protocol are the
existing RINDPack implementation. Only physical brick scheduling is new.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import time
from rind3d.geometry import Budget,BudgetExhausted,Aborted
from rind3d.pack import RINDPack,RINDPackWriter,atomic_json,atomic_text,sha256_file

def open_writers(root,n,identity,policy,resume):
    stores=[]
    for name in ('response','solid'):
        p=root/(name+'.rindb')
        if p.exists():
            if not resume:raise FileExistsError(p)
            sidecar=p.with_name(p.name+'.sha256')
            with RINDPack(p,verify_file=sidecar.exists()) as pack:
                if pack.shape!=(n,)*3 or pack.identity!=identity or pack.policy!=policy:raise ValueError('Completed field resume identity mismatch')
                if not sidecar.exists():pack.verify_all();atomic_text(sidecar,sha256_file(p)+'  '+p.name+'\n')
            p.with_name(p.name+'.journal.json').unlink(missing_ok=True);stores.append(None)
        else:stores.append(RINDPackWriter(p,(n,)*3,identity=identity,policy=policy,resume=resume))
    return stores

def render_stream(renderer,root,cfg,*,resume=False):
    from rind3d.stream import scene_identity
    root=Path(root);root.mkdir(parents=True,exist_ok=True);n=int(cfg['resolution'])
    if not 1<=n<=1024 or n&(n-1):raise ValueError('Power-of-two field resolution at most 1024 required')
    identity=scene_identity(renderer,renderer.source);renderer.validate_source(renderer.source)
    stores=open_writers(root,n,identity,renderer.policy_hash,resume);g=(n+63)//64;count=g**3
    done=[set(range(count)) if s is None else s.committed for s in stores]
    missing=[i for i in range(count) if any(i not in d for d in done)]
    interval=int(cfg.get('checkpoint_bricks',8));workers=int(cfg['workers']);queue=int(cfg['queue_size'])
    if interval<1 or workers<1 or queue<workers:raise ValueError('Invalid stream checkpoint/workers/queue')
    budget=Budget(cfg['generation_seconds'],cfg['generation_points']);start=time.perf_counter();cpu=time.process_time()
    before=dict(renderer.metrics);first=None;compression=0.;committed=0;times=[]
    runtime=getattr(renderer,'runtime',None);runtime_before=dict(runtime.timings) if runtime else None
    def origin(i):
        z,rem=divmod(i,g*g);y,x=divmod(rem,g);return x*64,y*64,z*64
    def checkpoint():
        for s in stores:
            if s is not None and not s.path.exists():s.checkpoint()
        atomic_json(root/'stream_progress.json',{'complete':False,'identity':identity,'backend':renderer.backend,
                    'committed_this_run':committed,'total_bricks':count,'missing_at_start':len(missing)})
    def commit(i,pair):
        nonlocal first,compression,committed
        budget.check();tick=time.perf_counter()
        if len(pair)!=2 or any(len(raw)!=32768 for raw in pair):raise ValueError('Renderer returned invalid packed brick')
        for s,d,raw in zip(stores,done,pair):
            if s is not None and i not in d:s.write(i,raw);d.add(i)
        compression+=time.perf_counter()-tick;committed+=1
        if first is None:first=time.perf_counter()-start
        if committed%interval==0:checkpoint()
        if cfg.get('stop_after_bricks') is not None and committed>=int(cfg['stop_after_bricks']):
            raise Aborted('Explicit stream interruption requested')
    try:
        if renderer.backend=='cuda_certified':
            batch=renderer.device.batch
            for pos in range(0,len(missing),batch):
                budget.check();ids=missing[pos:pos+batch];tick=time.perf_counter()
                pairs=renderer.packed_batch([origin(i) for i in ids],min(n,64),n,budget=budget)
                times.append(time.perf_counter()-tick)
                for i,pair in zip(ids,pairs):commit(i,pair)
        else:
            def job(i):
                tick=time.perf_counter();pair=renderer.packed_batch([origin(i)],min(n,64),n,budget=budget)[0]
                return pair,time.perf_counter()-tick
            with ThreadPoolExecutor(max_workers=workers) as executor:
                todo=iter(missing);pending={}
                def submit():
                    try:i=next(todo)
                    except StopIteration:return
                    pending[executor.submit(job,i)]=i
                for _ in range(min(queue,len(missing))):submit()
                while pending:
                    completed,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in completed:
                        i=pending.pop(future);pair,dt=future.result();times.append(dt);commit(i,pair);submit()
        if any(len(d)!=count for d in done):raise RuntimeError('Incomplete packed field coverage')
        for s in stores:
            if s is not None:s.finish()
    finally:
        for s in stores:
            if s is not None:
                if s.path.exists():s.journal.unlink(missing_ok=True)
                else:s.checkpoint()
    information=None
    if cfg.get('build_information_index',True):
        from rind3d.information import InformationIndex
        with RINDPack(root/'response.rindb',verify_file=True) as y,RINDPack(root/'solid.rindb',verify_file=True) as m:
            index=InformationIndex.build(y,m);size=index.save(root/'information.rindi');information={**index.build_report,'persistent_bytes':size}
    result={'complete':True,'identity':identity,'resolution':n,'bricks':count,'bricks_computed_this_run':len(missing),
            'backend':renderer.backend,'renderer':renderer.report(),'numerical_policy':renderer.provenance,
            'numerical_policy_sha256':renderer.policy_hash,'wall_seconds':time.perf_counter()-start,
            'cpu_seconds':time.process_time()-cpu,'first_brick_seconds':first,'compression_seconds':compression,
            'physical_point_calls':budget.points,'worker_or_batch_seconds':times,'checkpoint_bricks':interval,
            'work':{k:v-before.get(k,0) for k,v in renderer.metrics.items()},'information_index':information,
            'field_bytes':{name:(root/(name+'.rindb')).stat().st_size for name in ('response','solid')}}
    if runtime:result['cuda_timings']={k:v-runtime_before[k] for k,v in runtime.timings.items()};result['cuda_memory']=runtime.memory()
    atomic_json(root/'stream_progress.json',result);return result
