"""Bounded brick jobs and restartable, one-file-per-field streaming output."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import hashlib,time
import numpy as np
from .geometry import Budget,render_region,POLICY_HASH
from .pack import RINDPack,RINDPackWriter,atomic_json,atomic_text,sha256_file

def scene_identity(oracle,source):
    return hashlib.sha256(oracle.geometry.content_hash+np.asarray(source,dtype='<f8').tobytes()+bytes.fromhex(oracle.policy_hash)).hexdigest()

def stream_response(oracle,source,root,cfg,*,resume=False,renderer=None):
    if renderer is not None:
        if not np.array_equal(np.asarray(source).reshape(-1),renderer.source):raise ValueError('Stream renderer source mismatch')
        from .rendering.stream import render_stream
        return render_stream(renderer,root,cfg,resume=resume)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    n=int(cfg['resolution'])
    if not 1<=n<=1024 or n&(n-1):raise ValueError('Streaming grid resolution must be a power of two, at most 1024')
    g=(n+63)//64;count=g**3;identity=scene_identity(oracle,source)
    oracle.validate_source(source)
    stores=[]
    for name in ['response','solid']:
        p=root/(name+'.rindb')
        if p.exists():
            if not resume:raise FileExistsError(p)
            sidecar=p.with_name(p.name+'.sha256')
            with RINDPack(p,verify_file=sidecar.exists()) as r:
                if r.shape!=(n,)*3 or r.identity!=identity or r.policy!=oracle.policy_hash:raise ValueError('Completed field identity mismatch')
                if not sidecar.exists():
                    # Recover a crash after the field rename, before SHA publish.
                    r.verify_all();atomic_text(sidecar,sha256_file(p)+'  '+p.name+'\n')
            p.with_name(p.name+'.journal.json').unlink(missing_ok=True)
            stores.append(None)
        else:stores.append(RINDPackWriter(p,(n,)*3,identity=identity,policy=oracle.policy_hash,resume=resume))
    done=[set(range(count)) if s is None else s.committed for s in stores]
    missing=[i for i in range(count) if any(i not in d for d in done)]
    mode=cfg.get('field_acceleration','baseline');acceleration_setup=0.
    if mode not in ('baseline','certified_function'):raise ValueError('Unknown field_acceleration mode')
    if mode=='certified_function' and missing:
        from .field_acceleration import FieldOracle,field_evaluator
        t=time.perf_counter()
        if not isinstance(oracle,FieldOracle):oracle=field_evaluator(oracle,n,cfg.get('field_acceleration_settings'))
        acceleration_setup=time.perf_counter()-t
    budget=Budget(cfg['generation_seconds'],cfg['generation_points'])
    wall=time.perf_counter();compression=0.;worker_seconds=0.;certified=0
    def job(i):
        t=time.perf_counter();bz,rem=divmod(i,g*g);by,bx=divmod(rem,g)
        origin=(bx*64,by*64,bz*64);side=min(64,n-min(origin))
        y,m,c=render_region(oracle,source,origin,side,n,budget=budget,leaf_side=cfg['leaf_side'])
        arrays=[]
        for a in (y,m):
            padded=np.zeros((64,)*3,dtype=bool);padded[:side,:side,:side]=a
            arrays.append(np.packbits(padded,bitorder='little').tobytes())
        return arrays,c,time.perf_counter()-t
    workers=int(cfg['workers']);queue=int(cfg['queue_size'])
    if workers<1 or queue<workers:raise ValueError('Bounded queue must be >= positive worker count')
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            todo=iter(missing);pending={}
            def submit():
                try:i=next(todo)
                except StopIteration:return False
                pending[executor.submit(job,i)]=i;return True
            for _ in range(min(queue,len(missing))):submit()
            finished=0
            while pending:
                completed,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    i=pending.pop(future);arrays,c,dt=future.result();certified+=c;worker_seconds+=dt
                    t=time.perf_counter()
                    for s,d,raw in zip(stores,done,arrays):
                        if s is not None and i not in d:s.write(i,raw);d.add(i)
                    compression+=time.perf_counter()-t;finished+=1
                    if finished%8==0:
                        for s in stores:
                            if s is not None:s.checkpoint()
                        atomic_json(root/'stream_progress.json',{'identity':identity,'resolution':n,'done':finished,'total_missing':len(missing),'complete':False})
                    submit()
        for s in stores:
            if s is not None:s.finish()
    finally:
        # A failed job leaves only staging/partial files and explicit journals.
        for s in stores:
            if s is not None:
                try:
                    if s.path.exists():s.journal.unlink(missing_ok=True)
                    else:s.checkpoint()
                except (RuntimeError,ValueError):pass
    information=None
    if cfg.get('build_information_index',True):
        # One packed pass over the sealed fields. No additional physics queries.
        from .information import InformationIndex
        with RINDPack(root/'response.rindb') as response,RINDPack(root/'solid.rindb') as solid:
            index=InformationIndex.build(response,solid);stored_bytes=index.save(root/'information.rindi')
            information={**index.build_report,'persistent_bytes':stored_bytes,'format':'RIND-information-index-v1'}
    result={'complete':True,'identity':identity,'resolution':n,'bricks':count,'bricks_computed_this_run':len(missing),
            'numerical_policy':oracle.provenance,'numerical_policy_sha256':oracle.policy_hash,
            'wall_seconds':time.perf_counter()-wall,'render_worker_seconds':worker_seconds,'compression_seconds':compression,
            'certificate_covered_voxels':certified,'physical_point_calls':budget.points,'oracle_work':dict(oracle.metrics),
            'field_bytes':{name:(root/(name+'.rindb')).stat().st_size for name in ('response','solid')},'information_index':information,
            'field_acceleration':oracle.report() if hasattr(oracle,'report') else {'enabled':False,'requested_mode':mode},
            'field_acceleration_setup_seconds':acceleration_setup}
    atomic_json(root/'stream_progress.json',result);return result
