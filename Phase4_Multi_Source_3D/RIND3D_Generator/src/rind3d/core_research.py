"""Bounded single-GPU / isolated CPU Core pilot, not mass production.

Workers never create a CUDA context. Soft preemption happens only BETWEEN
physical bricks. A hard watchdog terminates an isolated process, preserves the
last atomic journal, and retries the SAME identity under a declared bound.
"""
from collections import Counter,deque,OrderedDict
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,multiprocessing as mp,os,queue,resource,time,traceback,threading,signal
import numpy as np
from .pack import RINDPack,atomic_json,sha256_file
from .pipeline import parquet_write,write_candidate_files,verify_scene
from .pilot import config_hash
from .geometry import load_geometry,save_geometry
from .windows import Window
from .rendering.context import GeometryContext
from .rendering.backends import BackendFactory
from .rendering.polyhedra import geometry_split
from .projection_research import ProjectionSource
from .lite_acceptance import geometry_acceptance,sample_source
from .lite_certification import validate_certificate
from .quality_certification import VERSION,validate as validate_quality
from .physical_evidence import identity as evidence_identity,bricks_for,load_evidence,VERSION as EVIDENCE_VERSION
from .lite_pipeline import append_ledger as _append_ledger,manifest,catalogs,preflight,loader_check,Telemetry
from .information import InformationIndex
from .counts import CountIndex
from .stratified_windows import select,DatasetBalance,settings as sampling_settings
from .stream import stream_response,scene_identity
from .search import CandidateBank
from .window_contract import WINDOW_PROTOCOL

_worker_cache=OrderedDict()

def append_ledger(root,value):_append_ledger(root,dict(value,event_unix_seconds=time.time()))

@contextmanager
def producer_lock(root):
    """An advisory process lock prevents two coordinators writing one identity."""
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.producer.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Core producer owns this output directory') from None
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def checkpoint(path,state):
    state.pop('journal_sha256',None);state['journal_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest();atomic_json(path,state)

def _windows(desc):return [Window(tuple(w['origin']),w['side'],witnesses=tuple(map(tuple,w['witnesses']))) for w in desc['windows']]

def verify_quantum(desc,soft_seconds=30):
    """Resume identity-bound physical work; return COMPLETE or INCOMPLETE."""
    scene=Path(desc['stage']);cfg=desc['cfg'];start=time.perf_counter();cpu=time.process_time()
    key=(str(scene),desc['geometry_id'],desc['verifier'],sha256_file(scene/'geometry.npz'),sha256_file(scene/'driver.f64'),json.dumps(cfg['numerical_policy'],sort_keys=True))
    if key not in _worker_cache:
        ctx=GeometryContext(load_geometry(scene/'geometry.npz'),cfg['numerical_policy']);p=np.fromfile(scene/'driver.f64',dtype='<f8')
        b=ProjectionSource(ctx,p) if desc['verifier']=='silhouette' else ctx.bind(p,independent=True)
        _worker_cache[key]=(ctx,p,[ctx.bind(p),b]);
        while len(_worker_cache)>4:_worker_cache.popitem(last=False)
    else:_worker_cache.move_to_end(key)
    ctx,p,models=_worker_cache[key];windows=_windows(desc);coords=sorted({b for w in windows for b in bricks_for(w)})
    if ctx.geometry_id!=desc['geometry_id'] or ('source' in desc and not np.array_equal(p,desc['source'])):raise ValueError('Physical job geometry/source identity differs')
    n=cfg['resolution'];ek=evidence_identity(ctx,p,n,models[0].identity,models[1].identity,sha256_file(scene/'response.rindb'),sha256_file(scene/'solid.rindb'))
    journal=scene/'physical-work.private.json';proof=scene/'physical_evidence.private.json'
    if proof.exists():
        _,r=load_evidence(proof,ek,windows);return {'status':'COMPLETE','evidence_seconds':r['wall_seconds'],'cached_complete_evidence':True}
    if journal.exists():
        state=json.loads(journal.read_text())
        checksum=state.pop('journal_sha256',None)
        if checksum!=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest():raise ValueError('Physical journal checksum mismatch')
        if state['key']!=ek or state['required']!=[list(b) for b in coords]:raise ValueError('Physical journal identity/coverage mismatch')
    else:state={'key':ek,'required':[list(b) for b in coords],'done':[[],[]],'metrics':[{'wall_seconds':0.,'cpu_seconds':0.,'work':{}},{'wall_seconds':0.,'cpu_seconds':0.,'work':{}}],
                'quantums':0,'setup_seconds':0.,'status':'INCOMPLETE','cpu_only_worker':True}
    state['quantums']+=1;processed=0;state['setup_seconds']+=time.perf_counter()-start
    with RINDPack(scene/'response.rindb',verify_file=True,cache_bricks=0) as y,RINDPack(scene/'solid.rindb',verify_file=True,cache_bricks=0) as m:
        if y.shape!=(n,)*3 or m.shape!=y.shape or y.policy!=ctx.policy_hash or m.policy!=y.policy or y.identity!=scene_identity(ctx,p) or m.identity!=y.identity:raise ValueError('Physical field shape/policy/identity differs')
        for backend,model in enumerate(models):
            before=dict(model.work);t=time.perf_counter();c=time.process_time();position=len(state['done'][backend]);stop=False
            if position>len(coords):raise ValueError('Physical journal count overflow')
            for i,b in enumerate(coords[:position]):
                if state['done'][backend][i]['coordinate']!=list(b):raise ValueError('Physical journal ordering mismatch')
            for b in coords[position:]:
                if processed and time.perf_counter()-start>=soft_seconds:stop=True;break
                pair=model.packed_batch([tuple(k*64 for k in b)],min(n,64),n)[0]
                if pair!=(y.read_brick_packed(b),m.read_brick_packed(b)):raise ValueError(f'Physical backend {backend} mismatch at {b}')
                digest=hashlib.sha256(b''.join(pair)).hexdigest();state['done'][backend].append({'coordinate':list(b),'digest':digest});processed+=1
                if processed%64==0:checkpoint(journal,state)
            metrics=state['metrics'][backend];metrics['wall_seconds']+=time.perf_counter()-t;metrics['cpu_seconds']+=time.process_time()-c
            for k,v in model.work.items():
                if k.startswith('max_'):metrics['work'][k]=max(metrics['work'].get(k,0),v)
                else:metrics['work'][k]=metrics['work'].get(k,0)+v-before.get(k,0)
            if stop:break
    checkpoint(journal,state)
    complete=all(len(d)==len(coords) for d in state['done'])
    if complete:
        rows=[]
        for a,b in zip(*state['done']):
            if a['coordinate']!=b['coordinate'] or a['digest']!=b['digest']:raise ValueError('Independent physical replay differs')
            rows.append({'coordinate':a['coordinate'],'primary_sha256':a['digest'],'independent_sha256':b['digest']})
        stages=[dict(s,stage=name,rendered_bricks=len(coords),all_response_and_solid_bits_matched=True) for name,s in zip(('primary','independent'),state['metrics'])]
        record={'version':EVIDENCE_VERSION,'key':ek,'complete':True,'private':True,'stages':stages,'bricks':rows,
                'wall_seconds':sum(s['wall_seconds'] for s in stages)+state['setup_seconds'],'quantums':state['quantums'],
                'hashes_are_not_physical_verification':True,'cooperative_soft_seconds':soft_seconds,'setup_seconds':state['setup_seconds']}
        atomic_json(proof,record);load_evidence(proof,ek,windows)
    return {'status':'COMPLETE' if complete else 'INCOMPLETE','quantum_wall_seconds':time.perf_counter()-start,'quantum_cpu_seconds':time.process_time()-cpu,
            'processed_bricks':processed,'done_counts':list(map(len,state['done'])),'required_bricks':len(coords),'soft_threshold_exceeded':time.perf_counter()-start>=soft_seconds}

def worker_loop(number,incoming,outgoing,parent_pid):
    # A killed coordinator releases its producer lock. Old workers must die
    # before they can conflict with a resumed coordinator. The kernel signal
    # also works while native GMP holds the GIL. Check the startup race where
    # the parent died before PR_SET_PDEATHSIG was installed.
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True);libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong];libc.prctl.restype=ctypes.c_int
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot install worker parent-death signal')
    if os.getppid()!=parent_pid:os._exit(125)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    while True:
        task=incoming.get()
        if task is None:return
        desc,soft,hard=task
        # Kernel-delivered default termination does not require the Python GIL
        # or interruptibility of an ongoing GMP/CGAL call. Only this isolated
        # worker dies; sealed fields and prior atomic checkpoints are retained.
        signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,hard)
        try:
            try:r=verify_quantum(desc,soft);message=(number,desc['slot'],r,None)
            except BaseException as exc:message=(number,desc['slot'],None,{'reason':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()})
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        outgoing.put(message)

class Worker:
    def __init__(self,ctx,number,outgoing):self.ctx=ctx;self.number=number;self.outgoing=outgoing;self.start()
    def start(self):
        self.incoming=self.ctx.Queue(1);self.process=self.ctx.Process(target=worker_loop,args=(self.number,self.incoming,self.outgoing,os.getpid()));self.process.start();self.job=None;self.started=0.
    def assign(self,desc,soft,hard=300):
        if self.job is not None:raise RuntimeError('Worker concurrent identity collision')
        self.job=desc;self.started=time.perf_counter();self.incoming.put((desc,soft,hard))
    def kill(self):
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        if self.process.is_alive():self.process.kill();self.process.join()
        self.incoming.close();self.incoming.cancel_join_thread()
    def close(self):
        if self.process.is_alive():self.incoming.put(None);self.process.join(timeout=5)
        self.kill()

class CoreTelemetry(Telemetry):
    def __init__(self,root):super().__init__(root);self.resources=[]
    def run(self):
        gpu=threading.Thread(target=super().run,daemon=True);gpu.start()
        try:
            import psutil
            parent=psutil.Process();previous={};last=time.perf_counter()
            while not self.stop.is_set():
                now=time.perf_counter();rss=pss=cpu=0.;current={};pids=[]
                for p in [parent,*parent.children(recursive=True)]:
                    try:
                        memory=p.memory_full_info();rss+=memory.rss;pss+=getattr(memory,'pss',0);t=p.cpu_times();total=t.user+t.system
                        if p.pid in previous:cpu+=max(0,total-previous[p.pid])
                        current[p.pid]=total;pids.append(p.pid)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                row={'monotonic_seconds':now,'family_rss_bytes':int(rss),'family_pss_bytes':int(pss),'cpu_cores_used':cpu/max(now-last,1e-9),'processes':pids}
                self.resources.append(row)
                with (self.root/'process-resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                previous=current;last=now;self.stop.wait(1.)
        finally:gpu.join(timeout=3)
    def report(self):
        r=super().report();r.update(max_family_rss_bytes=max((v['family_rss_bytes'] for v in self.resources),default=0),max_family_pss_bytes=max((v['family_pss_bytes'] for v in self.resources),default=0),
                                   mean_cpu_cores_used=float(np.mean([v['cpu_cores_used'] for v in self.resources[1:]])) if len(self.resources)>1 else None);return r

def prepare(cfg,root,factory,slot,balance,verifier,*,resume=False):
    started=time.perf_counter();cpu=time.process_time();family=cfg['families'][slot%len(cfg['families'])];t=started;c=cpu;stages={}
    if cfg.get('geometry_mixture')=='fast-v1':from .fast_geometry import generate
    else:from .lite_generators import generate
    bundle,recipe=generate(cfg['seed'],slot,family,0);gid=bundle.geometries[0].content_hash.hex();stage=root/'.staging'/f'{slot:05d}-0-{gid[:16]}';stage.mkdir(parents=True,exist_ok=True)
    append_ledger(root,{'event':'attempt_started','seed':cfg['seed'],'slot':slot,'attempt':0,'family':family,'geometry_id':gid,'recipe_version':recipe['version']})
    if recipe['proposal_failures']:raise ValueError('Geometry world fitting proposal failed')
    ctx=factory.context(bundle,cfg['numerical_policy']);acceptance=geometry_acceptance(ctx,family,cfg['seed'],slot)
    atomic_json(stage/'recipe.private.json',recipe);atomic_json(stage/'geometry_acceptance.private.json',acceptance);save_geometry(stage/'geometry.npz',bundle)
    stages['geometry_generation_and_acceptance']={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c};t=time.perf_counter();c=time.process_time()
    truth,sreport=sample_source(ctx,cfg['seed'],slot,limit=cfg.get('source_attempts',256));truth.astype('<f8').tofile(stage/'driver.f64');atomic_json(stage/'sources.private.json',{'truth':sreport,'bank':[sreport['clearance']]})
    sid=sreport['canonical_scene_id'];renderer=factory.source(bundle,truth,cfg['renderer_backend'],cfg['numerical_policy']);source_id=renderer.source_id
    append_ledger(root,{'event':'source_accepted','slot':slot,'attempt':0,'scene_id':sid,'geometry_id':gid,'source_id':source_id,'source':truth.tolist(),'rejections':sreport['rejections']})
    stages['source_sampling_and_renderer_setup']={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c};t=time.perf_counter();c=time.process_time()
    try:field=stream_response(ctx,truth,stage,cfg,resume=resume,renderer=renderer)
    finally:renderer.close()
    stages['complete_field_and_index']={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c};t=time.perf_counter();c=time.process_time()
    with RINDPack(stage/'response.rindb',verify_file=True) as y,RINDPack(stage/'solid.rindb',verify_file=True) as m:
        info=InformationIndex.load(stage/'information.rindi',y,m);index_path=stage/'counts.rindc'
        index=CountIndex.load(index_path,y,m) if resume and index_path.exists() else CountIndex.build(y,m)
        if not index_path.exists():index.save(index_path)
        selected=select(info,index,truth,seed=cfg['seed'],scene=slot,balance=balance,observation=cfg.get('observation_sampling'),natural_settings=cfg['window_selection'])
        index_build=index.build_report
    if not selected.report['complete'] or selected.report['placement']['occupied_octants']<2:raise ValueError('Window quota/coverage gate failed')
    atomic_json(stage/'selection.private.json',selected.report);parquet_write(stage/'window_selection.parquet',selected.provenance);parquet_write(stage/'uniform_control.parquet',selected.controls)
    stages['window_selection_and_exact_counts']={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c}
    desc={'slot':slot,'family':family,'scene_id':sid,'source_id':source_id,'geometry_id':gid,'source':truth.tolist(),'stage':str(stage.resolve()),'cfg':cfg,'verifier':verifier,
          'field':field,'selection':selected.report,'windows':selected.provenance,'count_index':index_build,'stages':stages,'started_wall':time.time(),
          'prepare_wall_seconds':time.perf_counter()-started,'prepare_cpu_seconds':time.process_time()-cpu,
          'ratio_bins':dict(Counter(r['ratio_bin'] for r in selected.provenance)),'mixture_components':dict(Counter(r['mixture_component'] for r in selected.provenance))}
    atomic_json(root/'jobs'/f'{slot:05d}.json',desc);return desc

def finish(desc,root,certificate,quality):
    stage=Path(desc['stage']);cfg=desc['cfg'];start=time.perf_counter();cpu=time.process_time();windows=_windows(desc)
    evidence=json.loads((stage/'physical_evidence.private.json').read_text());ctx=GeometryContext(load_geometry(stage/'geometry.npz'),cfg['numerical_policy']);truth=np.array(desc['source'],dtype='<f8')
    # Required truth references reuse real complete A/B physical evidence.
    physical,_=load_evidence(stage/'physical_evidence.private.json',evidence['key'],windows)
    reports=[];bank=CandidateBank();index=bank.add(truth)
    for w in windows:
        physical.verify_window(w,ctx,truth)
        reports.append({'status':'COMPLETE','actual_candidates':1,'all_grid_exact':True,'independent_validation_passed':True,
                        'search_termination_reason':'core_optional_search_disabled','declared_search_seconds':0.,'unique_candidates_evaluated':0,'uniqueness_proven':False})
    dtype,counts=write_candidate_files(stage,windows,bank,[[index] for _ in windows],reports)
    atomic_json(stage/'candidate_search.private.json',{'mode':'core','optional_search_seconds':0,'candidate_counts':{'1':256},'count_one_is_not_uniqueness':True,'reports':reports})
    atomic_json(stage/'scientific_certification.json',certificate);atomic_json(stage/'quality_certification.json',quality)
    # Preserve resumable journals outside the immutable scene once complete.
    if (stage/'physical-work.private.json').exists():os.replace(stage/'physical-work.private.json',root/'work-journals'/(desc['scene_id']+'.json'))
    hashes={p.name:sha256_file(p) for p in stage.iterdir() if p.is_file() and p.name!='verification.json'}
    meta={'dataset_profile':'rind3d-lite-v0.1','profile_version':'0.1','status':'READY_LITE','contract_version':2,'window_protocol':WINDOW_PROTOCOL,'resolution':1024,
          'max_candidates':8,'true_source_included':True,'min_source_separation':32/1024,'candidate_id_dtype':dtype,'candidate_storage':'pool-offsets-v2',
          'scene_id':desc['scene_id'],'source_id':desc['source_id'],'geometry_id':desc['geometry_id'],'geometry_hash':desc['geometry_id'],
          'split':geometry_split(desc['geometry_id'],0),'split_group':desc['geometry_id'],'split_seed':0,'families':[desc['family']],
          'config_sha256':config_hash(cfg),'numerical_policy':cfg['numerical_policy'],'numerical_policy_sha256':ctx.policy_hash,'candidate_domain':'world_outside_window',
          'equivalence_status':'grid_exact','continuous_certified':False,'generation':desc['field'],'physical_backends':[evidence['key']['primary_backend'],evidence['key']['independent_backend']],
          'scientific_identity':certificate['scientific_identity']['sha256'],'hashes':hashes,'production_revision':VERSION,'quality_identity':quality['identity']['sha256'],
          'production_mode':'core','optional_search_budget_seconds':0,'observation_sampling':cfg.get('observation_sampling',{}).get('mode','natural')}
    atomic_json(stage/'verification.json',meta)
    try:
        checked=verify_scene(stage,full_index=True);loaded=preflight(root,stage,checked,cfg)
    except Exception:
        meta['status']='INCOMPLETE';atomic_json(stage/'verification.json',meta);raise
    stages=dict(desc['stages'],shared_independent_physical_verification={'wall_seconds':evidence['wall_seconds'],'cpu_seconds':sum(s['cpu_seconds'] for s in evidence['stages'])},
                core_reference_and_publication={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu})
    result={'slot':desc['slot'],'family':desc['family'],'scene_id':desc['scene_id'],'geometry_id':desc['geometry_id'],'source':desc['source'],'status':'READY_LITE',
            'stages':stages,'field':desc['field'],'selection':desc['selection'],'physical':evidence['stages'],'candidate_counts':{'1':256},'additional_candidates':0,
            'prepare_wall_seconds':desc['prepare_wall_seconds'],'latency_from_prepared_seconds':time.time()-desc['started_wall'],'loader':loaded,'scene_bytes':checked['scene_row']['bytes']}
    atomic_json(root/'scene-reports'/(desc['scene_id']+'.json'),result);target=root/'scenes'/desc['scene_id']
    if target.exists():raise FileExistsError('Scientific identity already published')
    os.replace(stage,target);append_ledger(root,{'event':'READY_LITE','slot':desc['slot'],'scene_id':desc['scene_id'],'geometry_id':desc['geometry_id'],'latency_from_prepared_seconds':result['latency_from_prepared_seconds']})
    return checked,result

def run(cfg,output,scene_count,**kwargs):
    with producer_lock(output):return _run(cfg,output,scene_count,**kwargs)

def _run(cfg,output,scene_count,*,workers=2,resume=False,pilot_report=None,sequential=False,verifier='silhouette',soft_seconds=30,hard_seconds=300):
    from .cli import validate_config
    validate_config(cfg);sampling_settings(cfg.get('observation_sampling'))
    if cfg.get('ambiguity',{}).get('enabled') is not False or cfg['window_search_seconds']!=0:raise ValueError('Core requires explicit disabled optional search')
    if workers not in (1,2,4) or scene_count not in (10,100,300) or verifier not in ('original','silhouette'):raise ValueError('Only bounded declared research pilots are supported')
    if hard_seconds<=soft_seconds or soft_seconds<=0:raise ValueError('Separate positive soft/hard watchdogs required')
    quality=validate_quality(json.loads(Path(cfg['quality_certificate']).read_text()));certificate=validate_certificate(json.loads(Path(cfg['scientific_certificate']).read_text()))
    if scene_count>10:
        prior=json.loads(Path(pilot_report).read_text()) if pilot_report else {}
        required=10 if scene_count==100 else 100
        if not prior.get('complete') or prior.get('requested_scenes')!=required or prior.get('ready_scenes')!=required or prior.get('config_sha256')!=config_hash(cfg) or prior.get('quality_identity')!=quality['identity']['sha256']:raise ValueError('Prior bounded acceptance pilot gate closed')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for d in ('scenes','.staging','jobs','scene-reports','work-journals'): (root/d).mkdir(exist_ok=True)
    mf=manifest(cfg);mf.update(production_revision=VERSION,production_mode='core',quality_identity=quality['identity']['sha256'])
    if (root/'manifest.json').exists():
        if not resume or json.loads((root/'manifest.json').read_text())!=json.loads(json.dumps(mf)):raise ValueError('Resume manifest/config mismatch')
    else:atomic_json(root/'manifest.json',mf)
    previous=json.loads((root/'report.json').read_text()) if resume and (root/'report.json').exists() else {}
    prior_wall=previous.get('wall_seconds_cumulative',previous.get('wall_seconds_this_run',0.))
    balance=DatasetBalance();rows=[];results=[];failures=list(previous.get('failures',[]));fast=deque();slow=deque();attempted=set();dispatches=0;watchdogs=Counter();contexts=mp.get_context('spawn');outgoing=contexts.Queue(workers*2);pool=[]
    started=time.perf_counter();cpu=time.process_time();child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    # Recover atomic promotions and every prepared identity before new work.
    for slot in range(scene_count):
        path=root/'jobs'/f'{slot:05d}.json'
        if not path.exists():continue
        desc=json.loads(path.read_text());validate_config(desc['cfg']);attempted.add(slot)
        if desc['cfg']!=cfg or desc['verifier']!=verifier:raise ValueError('Prepared identity/config differs on resume')
        for k,v in desc['ratio_bins'].items():balance.bins[int(k)]+=v
        balance.components.update(desc['mixture_components'])
        published=root/'scenes'/desc['scene_id']
        if published.exists():rows.append(verify_scene(published));results.append(json.loads((root/'scene-reports'/(desc['scene_id']+'.json')).read_text()))
        else:fast.append(desc)
    restored_ready=len(results)
    def failure(desc,reason):
        f={'slot':desc['slot'],'family':desc['family'],'scene_id':desc['scene_id'],'geometry_id':desc['geometry_id'],'status':'INCOMPLETE','reason':reason};failures.append(f);append_ledger(root,{'event':'incomplete_attempt',**f})
    def collect(block=False):
        try:number,slot,result,error=outgoing.get(timeout=.2 if block else 0)
        except queue.Empty:return False
        worker=pool[number];desc=worker.job
        if desc is None or desc['slot']!=slot:raise RuntimeError('Worker returned wrong scientific identity')
        worker.job=None
        if error:failure(desc,error['reason']);atomic_json(root/'jobs'/f'{slot:05d}-failure.json',error)
        elif result['status']=='COMPLETE':
            try:
                checked,record=finish(desc,root,certificate,quality);rows.append(checked);results.append(record);catalogs(root,rows)
            except Exception as exc:failure(desc,type(exc).__name__+': '+str(exc));(root/'publication-failures.log').open('a').write(traceback.format_exc())
        else:slow.append(desc);append_ledger(root,{'event':'soft_yield','slot':slot,'scene_id':desc['scene_id'],**result})
        return True
    def schedule():
        nonlocal dispatches
        for worker in pool:
            if worker.job is None and (fast or slow):
                # Fair slow queue: at least every fourth dispatch, no speed skips.
                take_slow=bool(slow) and (not fast or dispatches%4==3);desc=(slow if take_slow else fast).popleft();dispatches+=1;worker.assign(desc,soft_seconds,hard_seconds)
    def monitor():
        for worker in pool:
            if worker.job is not None and (not worker.process.is_alive() or time.perf_counter()-worker.started>hard_seconds):
                desc=worker.job;slot=desc['slot'];exitcode=worker.process.exitcode;timed_out=exitcode==-signal.SIGALRM or time.perf_counter()-worker.started>hard_seconds;worker.kill();watchdogs[slot]+=1
                append_ledger(root,{'event':'ABORTED_WATCHDOG' if timed_out else 'ABORTED_WORKER','slot':slot,'scene_id':desc['scene_id'],'physical_rejection':False,'retry':watchdogs[slot],'exitcode':exitcode})
                if watchdogs[slot]<=1:slow.append(desc)
                else:failure(desc,'Hard watchdog exhausted same-identity retry')
                worker.start()
    def save_report(complete=False,factory=None,telemetry=None):
        elapsed=time.perf_counter()-started
        report={'format':VERSION,'complete':complete,'requested_scenes':scene_count,'ready_scenes':len(results),'config_sha256':config_hash(cfg),'quality_identity':quality['identity']['sha256'],
                'scenes':results,'failures':failures,'families':cfg['families'],'worker_count':workers,'sequential':sequential,'verifier':verifier,'soft_seconds':soft_seconds,'hard_seconds':hard_seconds,
                'wall_seconds_this_run':elapsed,'wall_seconds_cumulative':prior_wall+elapsed,'parent_cpu_seconds_this_run':time.process_time()-cpu,
                'restored_ready_scenes':restored_ready,'new_ready_scenes_this_run':len(results)-restored_ready,'ready_per_hour_this_run':3600*(len(results)-restored_ready)/elapsed,
                'ready_per_hour_cumulative':3600*len(results)/(prior_wall+elapsed),'dataset_balance':balance.json(),
                'max_inflight_scenes':1 if sequential else workers+2,'queue_bound':2,'single_gpu_producer':True,'worker_cuda_contexts':0,'dispatches':dispatches,'watchdogs':dict(watchdogs),'mass_production':False,
                'factory':factory.report() if factory else None,'telemetry':telemetry.report() if telemetry else None}
        atomic_json(root/'report.json',report);return report
    try:
        pool=[Worker(contexts,i,outgoing) for i in range(workers)]
        with CoreTelemetry(root) as telemetry,BackendFactory(cuda_batch=cfg.get('cuda_batch',32),cuda_allowed=certificate.get('cuda_certified',False)) as factory:
            for slot in range(scene_count):
                if slot in attempted:continue
                while len(fast)+len(slow)+sum(w.job is not None for w in pool)>=(1 if sequential else workers+2):
                    changed=collect(True);monitor();schedule()
                    if changed:save_report(factory=factory,telemetry=telemetry)
                try:fast.append(prepare(cfg,root,factory,slot,balance,verifier,resume=resume))
                except Exception as exc:
                    failure({'slot':slot,'family':cfg['families'][slot%len(cfg['families'])],'scene_id':None,'geometry_id':None},type(exc).__name__+': '+str(exc));(root/'preparation-failures.log').open('a').write(traceback.format_exc())
                attempted.add(slot)
                while collect():pass
                monitor();schedule();save_report(factory=factory,telemetry=telemetry)
            while fast or slow or any(w.job is not None for w in pool):
                changed=collect(True);monitor();schedule()
                if changed:save_report(factory=factory,telemetry=telemetry)
            if rows:catalogs(root,rows)
            report=save_report(len(results)==scene_count,factory,telemetry)
            if results:
                report['loader']=loader_check(root,workers=cfg.get('loader_workers',2));atomic_json(root/'report.json',report)
    finally:
        for worker in pool:worker.close()
        outgoing.close();outgoing.cancel_join_thread()
    child=resource.getrusage(resource.RUSAGE_CHILDREN);report['children_cpu_seconds_this_run']=child.ru_utime+child.ru_stime-child_before.ru_utime-child_before.ru_stime
    report['wall_seconds_this_run']=time.perf_counter()-started;report['wall_seconds_cumulative']=prior_wall+report['wall_seconds_this_run']
    report['ready_per_hour_this_run']=3600*(len(results)-restored_ready)/report['wall_seconds_this_run'];report['ready_per_hour_cumulative']=3600*len(results)/report['wall_seconds_cumulative'];atomic_json(root/'report.json',report);return report
