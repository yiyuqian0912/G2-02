"""Durable cumulative Core campaign. Only its coordinator publishes scenes.

Scientific jobs hold shared writer leases. Recovery takes the exclusive lease
before changing ownership, so a lingering native/driver call cannot race it.
"""
from collections import Counter,deque
from pathlib import Path
import ctypes,hashlib,json,multiprocessing as mp,os,queue,resource,secrets,signal,time,traceback,uuid
import numpy as np
from rind3d_reader.catalog import Catalog,FORMAT
from rind3d_reader.durable import write_json,seal,read_sealed,lock,fsync_dir,digest
from .campaign_contract import VERSION,build_identity,validate_certificate
from .pack import sha256_file
from .lite_certification import validate_certificate as original_certificate
from .quality_certification import validate as quality_certificate

def event(root,kind,**data):
 value={'kind':kind,'utc_unix':time.time(),**data};raw=json.dumps(seal(value),sort_keys=True,separators=(',',':')).encode()+b'\n';path=Path(root)/'events.private.jsonl';new=not path.exists()
 fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
 try:
  view=memoryview(raw)
  while view:view=view[os.write(fd,view):]
  os.fsync(fd)
 finally:os.close(fd)
 if new:fsync_dir(path.parent)
def check_events(root):
 path=Path(root)/'events.private.jsonl'
 if not path.exists():return 0
 raw=path.read_bytes();position=0;count=0
 for line in raw.splitlines(keepends=True):
  if not line.endswith(b'\n'):
   write_json(path.with_name('torn-log-'+uuid.uuid4().hex+'.json'),{'offset':position,'tail_hex':line.hex()})
   with path.open('r+b') as f:f.truncate(position);f.flush();os.fsync(f.fileno())
   break
  row=json.loads(line)
  if digest(row['payload'])!=row['sha256']:raise ValueError('Interior event-log corruption')
  count+=1;position+=len(line)
 return count
def attempt_path(root,n):return Path(root)/'attempts'/f'{n:012d}.json'
def attempt_read(root,n):return read_sealed(attempt_path(root,n))
def attempt_write(root,state):write_json(attempt_path(root,state['attempt_id']),seal(state))

def next_geometry_trial(states,limit):
 """Only a proved source rejection advances the source trial, not geometry."""
 if not states:return 0,0
 last=states[max(states)]
 g=last.get('geometry_ordinal',last['attempt_id']);trial=last.get('source_trial',0)
 if last['status']=='REJECTED_SOURCE':
  if trial+1>=limit:raise RuntimeError('Certified source-resampling budget exhausted; geometry remains incomplete')
  return g,trial+1
 return g+1,0

def session_totals(root):
 root=Path(root);complete={p.name.split('.')[0]:read_sealed(p) for p in (root/'sessions').glob('*.complete.json')}
 unfinished=[p for p in (root/'sessions').glob('*.start.json') if p.name.split('.')[0] not in complete]
 wall=sum(s['active_wall_seconds'] for s in complete.values());cpu=sum(s['total_cpu_seconds'] for s in complete.values())
 for p in unfinished:
  progress=p.with_name(p.name.replace('.start.','.progress.'))
  if progress.exists():
   r=read_sealed(progress);wall+=r['active_wall_seconds'];cpu+=r.get('total_cpu_seconds_checkpoint',r.get('parent_cpu_seconds',0))
 return {'active_generation_wall_seconds':wall,'total_cpu_seconds':cpu,'completed_sessions':len(complete),'unfinalized_sessions':len(unfinished),'runtime_accounting':'checkpoint_lower_bound_with_unmeasured_crash_tails' if unfinished else 'complete_process_rusage'}

def live_child_cpu(pool):
 total=0.;ticks=os.sysconf('SC_CLK_TCK')
 for child in pool:
  try:
   stat=Path(f'/proc/{child.process.pid}/stat').read_text().rsplit(')',1)[1].split()
   total+=(int(stat[11])+int(stat[12]))/ticks
  except (FileNotFoundError,ProcessLookupError):pass
 return total

def create(root,cfg,target_total,certificate_path,*,master_seed=None,diagnostic=False):
 from .cli import validate_config
 root=Path(root);validate_config(cfg)
 if root.exists() and any(root.iterdir()):raise FileExistsError('Create requires a fresh campaign directory')
 if not 1<=target_total<=10000 or (diagnostic and target_total>20):raise ValueError('Invalid cumulative target')
 if cfg.get('geometry_mixture')!='original-v01' or cfg['observation_sampling']['mode'] not in ('mixed','natural'):raise ValueError('Release distribution must retain all certified original families')
 from .lite_generators import FAMILIES,DISTRIBUTION
 if set(cfg['families'])!=set(FAMILIES) or len(cfg['families'])!=9:raise ValueError('First release requires all nine certified families exactly once')
 if cfg['resolution']!=1024 or cfg['dataset_profile']!='rind3d-lite-v0.1' or cfg['production_windows']!=256 or cfg['window_search_seconds']!=0 or cfg['ambiguity']['enabled']:raise ValueError('Campaign requires the exact zero-search Lite Core profile')
 if type(cfg.get('source_field_attempts')) is not int or not 1<=cfg['source_field_attempts']<=64:raise ValueError('Declare a source-field trial limit in 1..64')
 cert=json.loads(Path(certificate_path).read_text());validate_certificate(cert,allow_diagnostic=diagnostic)
 if cert['configuration_sha256']!=digest(cfg):raise ValueError('Campaign certificate/config mismatch')
 master=master_seed if master_seed is not None else secrets.token_hex(32)
 if len(bytes.fromhex(master))!=32:raise ValueError('Master seed must be 32 private bytes, hex encoded')
 identity=digest({'version':VERSION,'master_commitment':hashlib.sha256(bytes.fromhex(master)).hexdigest(),'config':digest(cfg)})
 root.mkdir(parents=True,exist_ok=True,mode=0o700);root.chmod(0o700)
 for directory in ('scenes','.staging','attempts','jobs','scene-reports','work-journals','leases','sessions','phase-markers','targets','certificates'):(root/directory).mkdir(parents=True,exist_ok=True)
 write_json(root/'certificates'/'original.json',original_certificate(json.loads(Path(cfg['scientific_certificate']).read_text())))
 write_json(root/'certificates'/'quality.json',quality_certificate(json.loads(Path(cfg['quality_certificate']).read_text())))
 write_json(root/'configuration.private.json',seal(cfg));write_json(root/'master.private.json',seal({'master_seed':master}));write_json(root/'certification.private.json',cert)
 manifest={'format':'RIND3D','version':2,'catalog_format':FORMAT,'campaign_version':VERSION,'campaign_id':identity,'dataset_profile':'rind3d-lite-v0.1','profile_version':'0.1','privacy_policy':'restricted-production-v1','configuration_sha256':digest(cfg),'build_identity':cert['build_identity'],'diagnostic_only':diagnostic,'created_unix':time.time(),'generator_distribution':DISTRIBUTION,'family_probabilities':{f:1/9 for f in cfg['families']},'split_protocol':'sha256(0/geometry_id/geometry-group-v1), little-endian first 8 bytes mod 100; 80/10/10'}
 write_json(root/'manifest.json',manifest,replace=False);write_json(root/'targets'/'000000000000.json',seal({'target_total':target_total,'created_unix':time.time()}),replace=False)
 with Catalog(root,create=True):pass
 event(root,'CAMPAIGN_CREATED',campaign_id=identity,target_total=target_total);return status(root)

def target(root):
 rows=[read_sealed(p) for p in sorted((Path(root)/'targets').glob('*.json'))]
 values=[r['target_total'] for r in rows]
 if not values or values!=sorted(set(values)):raise ValueError('Conflicting cumulative target history')
 return values[-1]
def status(root):
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text());c=Catalog(root);states=[read_sealed(p) for p in (root/'attempts').glob('*.json')]
 return {'campaign_id':manifest['campaign_id'],'target_total':target(root),'ready_scenes':len(c),'windows':len(c)*256,'unique_geometries':len(c.geometry_ids),'complete':len(c)>=target(root),'attempt_states':dict(Counter(r['status'] for r in states)),**session_totals(root),'calendar_seconds':time.time()-manifest['created_unix'],'stop_requested':(root/'pause.request').exists()}
def extend(root,total):
 root=Path(root)
 with lock(root/'.coordinator.lock'):
  old=target(root)
  limit=20 if json.loads((root/'manifest.json').read_text())['diagnostic_only'] else 10000
  if not old<total<=limit:raise ValueError('Target extension must strictly increase within its campaign role')
  n=len(list((root/'targets').glob('*.json')));write_json(root/'targets'/f'{n:012d}.json',seal({'target_total':total,'created_unix':time.time()}),replace=False);event(root,'TARGET_EXTENDED',previous=old,target_total=total)
 return status(root)
def pause(root):write_json(Path(root)/'pause.request',seal({'requested_unix':time.time()}));return {'pause_requested':True}

def child_policy(parent):
 libc=ctypes.CDLL(None,use_errno=True);libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong];libc.prctl.restype=ctypes.c_int
 signal.signal(signal.SIGTERM,signal.SIG_DFL);signal.signal(signal.SIGINT,signal.SIG_IGN)
 if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot install parent-death fence')
 if os.getppid()!=parent:os._exit(125)
def child_loop(role,number,root,incoming,outgoing,parent,session,cfg,master,cuda):
 child_policy(parent)
 if role=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
 factory=None
 if role=='gpu':
  from .rendering.backends import BackendFactory
  factory=BackendFactory(cuda_batch=cfg['cuda_batch'],cuda_allowed=cuda)
 try:
  while True:
   job=incoming.get()
   if job is None:return
   signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,cfg['physical_verification_seconds'] if role=='cpu' else max(300,cfg['generation_seconds']+120))
   try:
    with lock(Path(root)/'.writers.lock',shared=True),lock(Path(root)/'leases'/f"{job['attempt_id']:012d}.lock"):
     owner=read_sealed(Path(root)/'owner.private.json')
     if owner['session_id']!=session:raise RuntimeError('Stale writer ownership')
     if role=='gpu':
      from .campaign_scene import prepare
      result=prepare(root,cfg,master,job['attempt_id'],job['balance'],factory,job['geometry_ordinal'],job['source_trial'])
     else:
      from .campaign_verify import verify_quantum
      result=verify_quantum(job['desc'],cfg.get('soft_seconds',30))
    message=(role,number,job['attempt_id'],job['dispatch_id'],result,None)
   except BaseException as exc:
    from .source_eligibility import SourceIneligible
    message=(role,number,job['attempt_id'],job['dispatch_id'],None,{'reason':type(exc).__name__+': '+str(exc),'certified_source_rejection':isinstance(exc,SourceIneligible),'traceback':traceback.format_exc()})
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   outgoing.put(message)
 finally:
  if factory:factory.close()

class Child:
 def __init__(self,ctx,role,number,root,outgoing,session,cfg,master,cuda):
  self.ctx=ctx;self.args=(role,number,str(root),outgoing,os.getpid(),session,cfg,master,cuda);self.role=role;self.number=number;self.start()
 def start(self):
  role,number,root,outgoing,parent,session,cfg,master,cuda=self.args;self.incoming=self.ctx.Queue(1);self.process=self.ctx.Process(target=child_loop,args=(role,number,root,self.incoming,outgoing,parent,session,cfg,master,cuda));self.process.start();self.job=None;self.started=0
 def assign(self,job):
  if self.job is not None:raise RuntimeError('Concurrent job collision')
  self.job=dict(job,dispatch_id=uuid.uuid4().hex);self.started=time.monotonic();self.incoming.put(self.job)
 def close(self):
  if self.process.is_alive() and self.job is None:self.incoming.put(None);self.process.join(5)
  if self.process.is_alive():self.process.terminate();self.process.join(5)
  if self.process.is_alive():self.process.kill();self.process.join(5)
  self.incoming.close();self.incoming.cancel_join_thread()

def recover(root,catalog,*,full=True):
 root=Path(root);check_events(root);recovered=[]
 for row in catalog.rows():catalog.verify_scene(row,full=full)
 for path in sorted((root/'scenes').iterdir()):
  if not path.is_dir() or path.is_symlink():raise ValueError('Unsafe published scene path')
  if path.name in catalog.by_id:continue
  payload=read_sealed(path/'scene.seal.json');row=dict(payload['catalog_record'],seal_sha256=sha256_file(path/'scene.seal.json'));catalog.verify_scene(row,full=True);catalog.commit(row);event(root,'RECOVERED_CATALOG_REGISTRATION',attempt_id=row['attempt_id'],scene_id=row['scene_id']);recovered.append(row['scene_id'])
 for row in catalog.rows():
  state=attempt_read(root,row['attempt_id'])
  if state.get('status')!='READY' or state.get('scene_id')!=row['scene_id']:state.update(status='READY',scene_id=row['scene_id']);attempt_write(root,state)
  report=root/'scene-reports'/f"{row['attempt_id']:012d}.json"
  if report.exists():
   metrics=read_sealed(report)
   if not metrics.get('catalog_committed',True):metrics.update(scene_rename_durable=True,catalog_committed=True,catalog_registration_recovered=True);write_json(report,seal(metrics))
 return recovered

def run(root,*,max_new_scenes=100,max_runtime=None,workers=4,cpu_only=False,diagnostic=False,fault_phase=None):
 from .campaign_scene import finish
 from .core_research import CoreTelemetry
 root=Path(root)
 if max_new_scenes<0 or workers not in (1,2,4) or (max_runtime is not None and max_runtime<=0):raise ValueError('Invalid session bounds')
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
 with lock(root/'.coordinator.lock'):
  # An orphan in an uninterruptible native call still holds its shared lease.
  # Refuse recovery rather than racing it after only checking a stale PID.
  with lock(root/'.writers.lock'):pass
  manifest=json.loads((root/'manifest.json').read_text());cfg=read_sealed(root/'configuration.private.json');master=read_sealed(root/'master.private.json')['master_seed'];cert=validate_certificate(json.loads((root/'certification.private.json').read_text()),allow_diagnostic=diagnostic)
  if bool(manifest['diagnostic_only'])!=bool(diagnostic) or (diagnostic and target(root)>20) or manifest['configuration_sha256']!=digest(cfg) or cert['configuration_sha256']!=digest(cfg):raise ValueError('Campaign frozen configuration/role mismatch')
  if manifest['build_identity']!=cert['build_identity']:raise ValueError('Campaign manifest/build certificate conflict')
  original=original_certificate(json.loads((root/'certificates'/'original.json').read_text()));quality=quality_certificate(json.loads((root/'certificates'/'quality.json').read_text()))
  sid=uuid.uuid4().hex;started=time.monotonic();cpu=time.process_time();children_before=resource.getrusage(resource.RUSAGE_CHILDREN);stopped={'reason':None};old_handlers={}
  def stop(signum,frame):stopped['reason']='SIGINT' if signum==signal.SIGINT else 'SIGTERM'
  for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,stop)
  (root/'pause.request').unlink(missing_ok=True);write_json(root/'owner.private.json',seal({'session_id':sid,'pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_unix':time.time()}));event(root,'SESSION_STARTED',session_id=sid,max_new_scenes=max_new_scenes,max_runtime=max_runtime,cpu_only=cpu_only,verification_workers=workers,producer_count=1,queue_bound=workers+2,child_native_threads=1)
  write_json(root/'sessions'/(sid+'.start.json'),seal({'session_id':sid,'started_unix':time.time(),'pid':os.getpid()}),replace=False)
  pool=[];outgoing=None;session_result=None
  try:
   with Catalog(root,create=True) as catalog:
    recover(root,catalog,full=True);initial=len(catalog);wanted=min(max_new_scenes,max(0,target(root)-initial));states={r['attempt_id']:r for p in (root/'attempts').glob('*.json') for r in [read_sealed(p)]};prepared={};balance={'bins':[0]*10,'components':{}}
    for p in sorted((root/'jobs').glob('*.json')):
     desc=read_sealed(p)
     if desc['cfg']!=cfg:raise ValueError('Prepared job frozen configuration changed')
     desc['stage']=str((root/'.staging'/f"{desc['attempt_id']:012d}").resolve())
     if desc['balance_before']!=balance:raise ValueError('Prepared sampling history is discontinuous')
     balance=desc['balance_after'];desc['primary_leaf']=cert['primary_leaf']
     if desc['scene_id'] not in catalog.by_id:prepared[desc['attempt_id']]=desc
    next_id=max(states,default=-1)+1;selected=set();failed=set();slow=deque();fast=deque(sorted(prepared));retries=Counter();dispatches=0
    context=mp.get_context('spawn');outgoing=context.Queue(workers*2+4)
    if wanted:
     producer=Child(context,'gpu',0,root,outgoing,sid,cfg,master,original.get('cuda_certified',False) and not cpu_only);pool=[producer]+[Child(context,'cpu',i,root,outgoing,sid,cfg,master,False) for i in range(workers)]
    else:producer=None
    def set_state(n,**values):
     states[n].update(values);attempt_write(root,states[n])
    def can_select(n):return n in selected or len(selected)<wanted
    def fault(phase,desc):
     if phase!=fault_phase:return
     write_json(root/'phase-markers'/'coordinator-fault.json',{'phase':phase,'pid':os.getpid(),'attempt_id':desc['attempt_id']})
     # Explicit diagnostic hook only; the test driver kills this coordinator.
     if not diagnostic:raise ValueError('Fault injection is diagnostic-only')
     while True:time.sleep(.05)
    def publish(desc):
     row,metrics=finish(desc,root,original,quality,cert,fault=fault);fault('before_catalog_commit',desc);commit=catalog.commit(row);fault('after_catalog_commit',desc);metrics.update(catalog_committed=True,catalog_commit=commit);write_json(root/'scene-reports'/f"{desc['attempt_id']:012d}.json",seal(metrics));set_state(desc['attempt_id'],status='READY',scene_id=row['scene_id']);event(root,'READY_LITE',session_id=sid,attempt_id=row['attempt_id'],scene_id=row['scene_id'],catalog_commit=commit)
    pause_started=None;last_progress=0
    with CoreTelemetry(root) as telemetry:
     while True:
      now=time.monotonic()
      if max_runtime is not None and now-started>=max_runtime:stopped['reason']=stopped['reason'] or 'runtime_limit'
      if (root/'pause.request').exists():stopped['reason']=stopped['reason'] or 'pause_request'
      if stopped['reason'] and pause_started is None:pause_started=now;event(root,'PAUSE_STARTED',session_id=sid,reason=stopped['reason'])
      try:role,number,n,dispatch_id,result,error=outgoing.get(timeout=.1) if outgoing else (None,None,None,None,None,None)
      except queue.Empty:role=None
      if role:
       child=producer if role=='gpu' else pool[number+1]
       if child.job is None or child.job['attempt_id']!=n or child.job['dispatch_id']!=dispatch_id:
        event(root,'STALE_WORKER_RESULT_IGNORED',session_id=sid,attempt_id=n,dispatch_id=dispatch_id);continue
       child.job=None
       if error and error.get('certified_source_rejection'):
        proof=root/'.staging'/f'{n:012d}'/'source-rejection.private.json'
        if not json.loads(proof.read_text())['independent_packed_recheck']:raise ValueError('Missing source rejection proof')
        selected.discard(n);set_state(n,status='REJECTED_SOURCE',reason=error['reason'],proof_sha256=sha256_file(proof));event(root,'CERTIFIED_SOURCE_REJECTION',session_id=sid,attempt_id=n,**error)
       elif error:
        failed.add(n);set_state(n,status='INCOMPLETE',reason=error['reason']);event(root,'ATTEMPT_INCOMPLETE',session_id=sid,attempt_id=n,physical_rejection=False,**error)
       elif role=='gpu':
        prepared[n]=result;prepared[n]['primary_leaf']=cert['primary_leaf'];balance=result['balance_after'];set_state(n,status='PREPARED',scene_id=result['scene_id'],geometry_id=result['geometry_id']);fast.append(n);event(root,'PREPARED',session_id=sid,attempt_id=n,scene_id=result['scene_id'])
       elif result['status']=='COMPLETE':
        try:publish(prepared[n])
        except Exception as exc:
         failed.add(n);set_state(n,status='INCOMPLETE',reason=type(exc).__name__+': '+str(exc));event(root,'PUBLICATION_INCOMPLETE',session_id=sid,attempt_id=n,physical_rejection=False,reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
       else:
        set_state(n,status='PREPARED');slow.append(n);event(root,'SOFT_YIELD',session_id=sid,attempt_id=n,**result)
      for child in pool:
       if child.job is not None and not child.process.is_alive():
        n=child.job['attempt_id'];exitcode=child.process.exitcode;child.close();retries[n]+=1;event(root,'ABORTED_WORKER',session_id=sid,attempt_id=n,exitcode=exitcode,physical_rejection=False);set_state(n,status='INCOMPLETE',reason=f'worker exit {exitcode}');child.start()
        if retries[n]>1:failed.add(n)
        elif child.role=='cpu':slow.append(n)
      if stopped['reason']:
       # A signal can arrive after the loop-top check (including during
       # outgoing.get or publication); initialize the pause clock here too.
       if pause_started is None:
        pause_started=time.monotonic();event(root,'PAUSE_STARTED',session_id=sid,reason=stopped['reason'])
       if not any(c.job for c in pool):break
       if now-pause_started>=cfg.get('pause_grace_seconds',60):
        for child in pool:
         if child.job is not None:
          n=child.job['attempt_id'];event(root,'PAUSED_INCOMPLETE',session_id=sid,attempt_id=n,physical_rejection=False);set_state(n,status='INCOMPLETE',reason='pause grace elapsed');child.close();child.job=None
        break
       continue
      # CPU jobs are bounded; slow work receives at least every fourth dispatch.
      for child in pool[1:]:
       if stopped['reason']:break
       if child.job is not None:continue
       choices=(slow if slow and (not fast or dispatches%4==3) else fast)
       found=None
       for _ in range(len(choices)):
        n=choices.popleft()
        if n in failed or states[n]['status']=='READY':continue
        if can_select(n):found=n;break
        choices.append(n)
       if found is not None:
        selected.add(found);set_state(found,status='VERIFYING');child.assign({'attempt_id':found,'desc':prepared[found]});dispatches+=1
      # One producer; no new attempt after a failed preparation because the
      # dataset-level sampling balance must retain its deterministic prefix.
      if not stopped['reason'] and producer and producer.job is None and not any(n in failed and n not in prepared for n in failed):
       n=next((i for i in sorted(states) if states[i]['status'] not in ('READY','REJECTED_SOURCE') and i not in prepared and i not in failed and can_select(i)),None)
       pending_count=sum(r['status'] not in ('READY','REJECTED_SOURCE') for r in states.values())
       inflight=sum(r['status'] not in ('READY','REJECTED_SOURCE') for i,r in states.items() if i in selected)
       if n is None and len(selected)<wanted and len(catalog)+pending_count<target(root) and pending_count<workers+2:
        try:g,trial=next_geometry_trial(states,cfg.get('source_field_attempts',64))
        except RuntimeError as exc:stopped['reason']=str(exc);continue
        n=next_id;next_id+=1;states[n]={'attempt_id':n,'geometry_ordinal':g,'source_trial':trial,'status':'ALLOCATED','allocated_unix':time.time()};attempt_write(root,states[n]);event(root,'ATTEMPT_ALLOCATED',session_id=sid,**states[n])
       if n is not None and inflight<workers+2:
        selected.add(n);set_state(n,status='PREPARING');producer.assign({'attempt_id':n,'balance':balance,'geometry_ordinal':states[n]['geometry_ordinal'],'source_trial':states[n]['source_trial']})
      active=any(c.job for c in pool)
      if not active and (len(catalog)-initial>=wanted or len(selected)>=wanted or failed or len(catalog)>=target(root)):break
      progress={'session_id':sid,'ready_scenes':len(catalog),'target_total':target(root),'active_wall_seconds':time.monotonic()-started,'parent_cpu_seconds':time.process_time()-cpu,'attempts_selected':len(selected),'failed_attempts':sorted(failed),'telemetry':telemetry.report()}
      if now-last_progress>=5:
       used=resource.getrusage(resource.RUSAGE_CHILDREN);progress['total_cpu_seconds_checkpoint']=time.process_time()-cpu+used.ru_utime+used.ru_stime-children_before.ru_utime-children_before.ru_stime+live_child_cpu(pool)
       write_json(root/'sessions'/(sid+'.progress.json'),seal(progress));last_progress=now
     session_result={'session_id':sid,'verification_workers':workers,'cpu_only':cpu_only,'ready_before':initial,'ready_after':len(catalog),'new_ready':len(catalog)-initial,'reason':stopped['reason'] or ('target_reached' if len(catalog)>=target(root) else 'session_scene_limit' if not failed else 'incomplete_attempts'),'failed_attempts':sorted(failed),'telemetry':telemetry.report()}
  finally:
   for child in pool:child.close()
   if outgoing is not None:outgoing.close();outgoing.cancel_join_thread()
   for sig,handler in old_handlers.items():signal.signal(sig,handler)
  child_cpu=resource.getrusage(resource.RUSAGE_CHILDREN);session_result.update(active_wall_seconds=time.monotonic()-started,total_cpu_seconds=time.process_time()-cpu+child_cpu.ru_utime+child_cpu.ru_stime-children_before.ru_utime-children_before.ru_stime,finished_unix=time.time())
  write_json(root/'sessions'/(sid+'.complete.json'),seal(session_result),replace=False);event(root,'SESSION_FINISHED',**session_result);return status(root)|{'last_session':session_result}

def verify(root,*,full=True,physical=False,recover_catalog=False):
 from .pipeline import verify_scene
 root=Path(root)
 with lock(root/'.coordinator.lock'),lock(root/'.writers.lock'),Catalog(root,create=recover_catalog) as catalog:
  m=json.loads((root/'manifest.json').read_text());cfg=read_sealed(root/'configuration.private.json')
  if m['configuration_sha256']!=digest(cfg):raise ValueError('Campaign configuration integrity failure')
  restored=recover(root,catalog,full=full) if recover_catalog else []
  for row in catalog.rows():
   catalog.verify_scene(row,full=full)
   if physical:verify_scene(root/'scenes'/row['scene_id'],full_index=True,physical=True)
  return {'verified_scenes':len(catalog),'windows':len(catalog)*256,'full_sha':full,'physical_replayed':physical,'restored_catalog':restored}
