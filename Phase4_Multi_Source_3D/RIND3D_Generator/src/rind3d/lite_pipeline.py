"""Bounded, resumable Lite production pilot; no implicit mass-production entry.

Every attempted geometry/source identity is retained in the append-only ledger.
The on-disk scene is promoted only after scientific and ordinary loader checks.
"""
from collections import Counter
import hashlib,json,os,resource,shutil,subprocess,time,threading
from pathlib import Path
import numpy as np
from .pack import RINDPack,atomic_json,sha256_file
from .pipeline import parquet_write,write_candidate_files
from .pilot import config_hash
from .profiles import LITE,validate_manifest_profile
from .window_contract import WINDOW_SIDES,WINDOW_PROTOCOL
from .geometry import save_geometry,load_geometry,Aborted,BudgetExhausted,NumericalUnresolved
from .lite_generators import generate,DISTRIBUTION
from .lite_acceptance import geometry_acceptance,sample_source,DRIVER_RADIUS
from .lite_certification import certify,validate_certificate,scientific_identity
from .rendering.backends import BackendFactory
from .rendering.polyhedra import geometry_split
from .stream import stream_response
from .physical_evidence import verify_physical_bricks
from .information import InformationIndex
from .stored_windows import select_stored_windows
from .search import CandidateBank
from .lite_search import source_pool,CandidateVerifier,discover
from .lite_publication import REQUIRED,verify_lite_scene

def append_ledger(root,value):
    with (root/'attempts.jsonl').open('a') as f:
        f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

class Telemetry:
    """One-second resource samples; GPU availability is recorded, never assumed."""
    def __init__(self,root):self.root=root;self.stop=threading.Event();self.samples=[]
    def run(self):
        try:
            import psutil
            process=psutil.Process()
        except ImportError:process=None
        while not self.stop.is_set():
            rss=process.memory_info().rss if process else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            row={'monotonic_seconds':time.perf_counter(),'rss_bytes':rss}
            try:
                output=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                if output.returncode==0:
                    values=output.stdout.strip().splitlines()[0].split(',')
                    for k,v in zip(('temperature_C','utilization_percent','global_used_MiB','power_W'),values):
                        try:row[k]=float(v.strip())
                        except ValueError:row[k]=None
                else:row['gpu_unavailable']=output.stderr.strip()[:200]
            except (OSError,subprocess.TimeoutExpired) as exc:row['gpu_unavailable']=type(exc).__name__
            self.samples.append(row)
            with (self.root/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.stop.wait(1.)
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=3)
    def report(self):
        return {'sample_interval_seconds':1,'samples':len(self.samples),
                'max_rss_bytes':max((r['rss_bytes'] for r in self.samples),default=None),
                'max_temperature_C':max((r['temperature_C'] for r in self.samples if r.get('temperature_C') is not None),default=None),
                'max_global_gpu_used_MiB':max((r['global_used_MiB'] for r in self.samples if r.get('global_used_MiB') is not None),default=None),
                'note':'GPU global memory includes other desktop applications; owned CUDA allocations are reported separately'}

def manifest(cfg):
    return {'format':'RIND3D','version':2,'dataset_profile':LITE,'profile_version':'0.1',
            'response_storage':'RINDPack-v1','production_windows':256,'max_candidates':8,
            'window_sides':list(WINDOW_SIDES),'window_protocol':WINDOW_PROTOCOL,
            'split_group':'geometry_id','config':cfg,'config_sha256':config_hash(cfg),
            'private_labels_default':False,'mass_production_authorized':False}

def catalogs(root,rows):
    parquet_write(root/'scenes.parquet',[r['scene_row'] for r in rows])
    parquet_write(root/'windows.parquet',[w for r in rows for w in r['window_rows']])
    parquet_write(root/'splits.parquet',[{'scene_id':r['scene_row']['scene_id'],'geometry_id':r['scene_row']['geometry_id'],'split':r['scene_row']['split']} for r in rows])

def loader_check(root,*,workers=0,all_scenes=True):
    from .data import RINDDataset,make_dataloader
    checked=[]
    with RINDDataset(root,split=None) as ds:
        if not len(ds):raise ValueError('READY_LITE loader returned no observations')
        if all_scenes:
            for sid in ds.scenes:
                index=next(i for i,r in enumerate(ds.records) if r[0]==sid)
                sample=ds[index]
                if set(sample)!={'scene_id','window_id','window_origin','window_size','structure','observed_response'}:raise ValueError('Private metadata leaked into default model input')
                candidates=ds.private_oracle().get_candidate_sources(sid,sample['window_id'])
                truth=ds.private_oracle().get_true_source(sid)
                if not 1<=len(candidates)<=8 or not any(np.array_equal(p,truth) for p in candidates):raise ValueError('Private references inaccessible')
                checked.append(sid)
        iterator=iter(make_dataloader(ds,batch_size=1,num_workers=workers,seed=7))
        batch=next(iterator)
        if 'candidate_sources' in batch:raise ValueError('Private source leakage in normal DataLoader')
        del iterator
    with RINDDataset(root,split=None,view='candidate_conditioned') as private:
        iterator=iter(make_dataloader(private,batch_size=1,num_workers=workers,seed=7));batch=next(iterator)
        if 'candidate_sources' not in batch or not 1<=len(batch['candidate_sources'][0])<=8:raise ValueError('Private DataLoader references inaccessible')
        del iterator
    return {'passed':True,'num_workers':workers,'normal_and_private_pytorch_dataloaders':True,'default_private_isolation':True,'scenes_opened':checked}

def preflight(root,stage,checked,cfg):
    check=root/'.loader-check';check.mkdir(exist_ok=True);(check/'scenes').mkdir(exist_ok=True)
    sid=checked['scene_row']['scene_id'];link=check/'scenes'/sid
    if link.exists() or link.is_symlink():link.unlink()
    link.symlink_to(stage.resolve(),target_is_directory=True)
    try:
        atomic_json(check/'manifest.json',manifest(cfg));catalogs(check,[checked]);return loader_check(check)
    finally:link.unlink(missing_ok=True)

def diversity(context,recipe,source,windows,scene):
    components=context.geometry_proof()['components'];anchors=np.array([o['center'] for o in recipe['objects']])
    rng=np.random.default_rng(530);points=rng.random((512,3));free=~context.contains(points);points=points[free]
    hits,entry=context.bind(source).native.first_hits(np.ascontiguousarray(points))
    lengths=np.linalg.norm(points-source,axis=1);shadow=hits>=0;nonlocal_distance=(1-entry[shadow])*lengths[shadow]
    balances=[]
    # Exact stored counts; reading these windows is already needed by discovery.
    with RINDPack(scene/'response.rindb') as y,RINDPack(scene/'solid.rindb') as m:
        for w in windows[::16]:
            a=y.read_window(*w.origin,w.size);b=m.read_window(*w.origin,w.size);f=int((~b).sum())
            balances.append(float((a&~b).sum()/f))
        occupancy={'solid':sum(r[7] for r in m.index)/1024**3,'lit':sum(r[7] for r in y.index)/1024**3}
    return {'obstacles':len(components),'actual_faces':[r['actual_faces'] for r in components],
            'declared_planes':[r['declared_planes'] for r in components],
            'exact_bounding_boxes':[[r['lower'],r['upper']] for r in components],
            'half_parameters':[o['half_parameters'] for o in recipe['objects']],
            'orientations':[o['orientation'] for o in recipe['objects']],
            'anchor_octants':len(set(map(tuple,(anchors>=.5).astype(int)))),
            'occupancy':{**occupancy,'shadow_free':1-occupancy['solid']-occupancy['lit']},
            'window_lit_free_balance_stratified_sample':balances,
            'nonlocal_occlusion_probe':{'exact_queries':len(points),'blocked':int(shadow.sum()),
                 'first_hit_component_counts':dict(Counter(map(str,hits[shadow]))),
                 'occluder_receiver_distances':nonlocal_distance.tolist(),
                 'distance_over_0_125':int((nonlocal_distance>.125).sum()),
                 'note':'Exact GMP first-hit identities; distances are binary64 descriptive statistics only'}}

def one_scene(cfg,root,factory,certificate,slot,family,attempt,*,resume=False):
    start=time.perf_counter();cpu=time.process_time();stages={}
    def stage_time(name,t,c):stages[name]={'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c};atomic_json(root/'current-stage.json',{'slot':slot,'family':family,'stage':name,'stages':stages})
    t=time.perf_counter();c=time.process_time();bundle,recipe=generate(cfg['seed'],slot,family,attempt)
    gid=bundle.geometries[0].content_hash.hex();stage=root/'.staging'/f'{slot:05d}-{attempt}-{gid[:16]}';stage.mkdir(parents=True,exist_ok=True)
    append_ledger(root,{'event':'attempt_started','slot':slot,'attempt':attempt,'family':family,'geometry_id':gid,'staging':str(stage.relative_to(root))})
    atomic_json(stage/'recipe.private.json',recipe)
    if recipe['proposal_failures']:raise ValueError('Geometry recipe world-fitting proposal failed')
    context=factory.context(bundle,cfg['numerical_policy']);acceptance=geometry_acceptance(context,family,cfg['seed'],slot,attempt)
    atomic_json(stage/'geometry_acceptance.private.json',acceptance);save_geometry(stage/'geometry.npz',bundle)
    stage_time('geometry_generation_and_acceptance',t,c)
    t=time.perf_counter();c=time.process_time();truth,source_report=sample_source(context,cfg['seed'],slot,attempt,limit=cfg.get('source_attempts',256))
    truth.astype('<f8').tofile(stage/'driver.f64');sid=source_report['canonical_scene_id'];renderer=factory.source(bundle,truth,cfg['renderer_backend'],cfg['numerical_policy'])
    append_ledger(root,{'event':'source_accepted','slot':slot,'attempt':attempt,'geometry_id':gid,'source':truth.tolist(),'source_id':renderer.source_id,'scene_id':sid,'rejections':source_report['rejections']})
    stage_time('source_sampling_and_renderer_setup',t,c)
    t=time.perf_counter();c=time.process_time()
    try:field=stream_response(context,truth,stage,cfg,resume=resume,renderer=renderer)
    finally:renderer.close()
    stage_time('complete_field_and_index',t,c)
    t=time.perf_counter();c=time.process_time()
    with RINDPack(stage/'response.rindb',verify_file=True) as y,RINDPack(stage/'solid.rindb',verify_file=True) as m:
        index=InformationIndex.load(stage/'information.rindi',y,m)
        selected=select_stored_windows(index,truth,256,seed=cfg['seed'],scene=slot,settings=cfg.get('window_selection'),output=stage/'selection-audit')
    atomic_json(stage/'selection.private.json',selected.report)
    if not selected.report['complete']:raise ValueError('Informative window quota incomplete')
    if selected.report['placement']['occupied_octants']<2:raise ValueError('Window coverage gate failed')
    windows=selected.windows;parquet_write(stage/'window_selection.parquet',selected.provenance);parquet_write(stage/'uniform_control.parquet',selected.controls)
    stage_time('window_selection',t,c)
    t=time.perf_counter();c=time.process_time();physical,evidence_report=verify_physical_bricks(stage,windows,context,truth,resume=resume,seconds=cfg.get('physical_verification_seconds',300));stage_time('shared_independent_physical_verification',t,c)
    t=time.perf_counter();c=time.process_time();pool,pool_proofs,pool_report=source_pool(context,truth,cfg,slot)
    bank=CandidateBank();ids=[];reports=[];verifier=CandidateVerifier(context)
    source_proofs={truth.astype('<f8').tobytes():source_report['clearance']}
    source_proofs.update({p.astype('<f8').tobytes():q for p,q in zip(pool,pool_proofs)})
    with RINDPack(stage/'response.rindb') as y,RINDPack(stage/'solid.rindb') as m:
        for wid,w in enumerate(windows):
            observed=y.read_window(*w.origin,w.size);mask=m.read_window(*w.origin,w.size)
            points,proofs,report=discover(context,truth,w,observed,mask,pool,cfg,slot,wid,physical,verifier)
            ids.append([bank.add(p) for p in points]);reports.append(report)
            if wid%32==0:atomic_json(stage/'candidate-progress.json',{'complete':False,'windows':wid+1,'candidates':dict(Counter(len(row) for row in ids))})
    dtype,counts=write_candidate_files(stage,windows,bank,ids,reports)
    atomic_json(stage/'sources.private.json',{'truth':source_report,'bank':[source_proofs[p.astype('<f8').tobytes()] for p in bank.array]})
    atomic_json(stage/'candidate_search.private.json',{'pool':pool_report,'reports':reports,'candidate_counts':dict(Counter(map(str,counts))),
                'count_one_is_not_uniqueness':True,'strict_full_window_and_independent':True,'verification_cache':dict(verifier.stats)})
    stage_time('candidate_search_and_independent_checks',t,c)
    t=time.perf_counter();c=time.process_time();div=diversity(context,recipe,truth,windows,stage);atomic_json(stage/'diversity.private.json',div);stage_time('diversity_audit',t,c)
    atomic_json(stage/'scientific_certification.json',certificate)
    hashes={p.name:sha256_file(p) for p in stage.iterdir() if p.is_file() and p.name!='verification.json'}
    meta={'dataset_profile':LITE,'profile_version':'0.1','status':'READY_LITE','contract_version':2,
          'window_protocol':WINDOW_PROTOCOL,'resolution':1024,'max_candidates':8,'true_source_included':True,
          'min_source_separation':32/1024,'candidate_id_dtype':dtype,'candidate_storage':'pool-offsets-v2',
          'scene_id':sid,'geometry_id':gid,'geometry_hash':gid,'source_id':renderer.source_id,
          'split':geometry_split(gid,0),'split_group':gid,'split_seed':0,'families':[family],
          'config_sha256':config_hash(cfg),'numerical_policy':cfg['numerical_policy'],'numerical_policy_sha256':context.policy_hash,
          'candidate_domain':cfg['candidate_domain'],'equivalence_status':'grid_exact','continuous_certified':False,
          'generation':field,'physical_backends':[evidence_report['key']['primary_backend'],evidence_report['key']['independent_backend']],
          'scientific_identity':certificate['scientific_identity']['sha256'],'hashes':hashes}
    atomic_json(stage/'verification.json',meta)
    checked=verify_lite_scene(stage,full_index=True)
    preflight_result=preflight(root,stage,checked,cfg);stage_time('publication_validation_and_loader',t,c)
    # Scientific files and metadata are immutable after this point. Operational
    # timing is stored outside the scene so its catalog byte count stays exact.
    target=root/'scenes'/sid
    if target.exists():raise FileExistsError(target)
    result={'slot':slot,'attempt':attempt,'family':family,'scene_id':sid,'geometry_id':gid,'source':truth.tolist(),
            'status':'READY_LITE','stages':stages,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'field':field,'selection':selected.report,'physical':evidence_report['stages'],
            'candidate_counts':dict(Counter(map(str,counts))),'additional_candidates':int((counts-1).sum()),
            'candidate_pool':pool_report,'source_rejections':source_report['rejections'],'diversity':div,'loader':preflight_result,
            'scene_bytes':checked['scene_row']['bytes'],'peak_rss_kib_process_so_far':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (root/'scene-reports').mkdir(exist_ok=True);atomic_json(root/'scene-reports'/(sid+'.json'),result)
    os.replace(stage,target)
    append_ledger(root,{'event':'READY_LITE','slot':slot,'attempt':attempt,'scene_id':sid,'geometry_id':gid,'wall_seconds':result['wall_seconds']})
    return checked,result

def generate_lite(cfg,output,scene_count,*,pilot_report=None,resume=False,certificate_path=None):
    from .cli import validate_config
    from .profiles import profile_for
    validate_config(cfg)
    if profile_for(cfg).name!=LITE:raise ValueError('generate_lite requires the explicit Lite profile')
    if not 1<=scene_count<=100:raise ValueError('This release is limited to bounded pilots of at most 100 scenes; mass production is not enabled')
    if scene_count>10:
        if not pilot_report:raise ValueError('A complete 10-scene acceptance pilot is required before 100 scenes')
        prior=json.loads(Path(pilot_report).read_text())
        if not prior.get('complete') or prior.get('requested_scenes')!=10 or prior.get('ready_scenes')!=10 or prior.get('config_sha256')!=config_hash(cfg) or prior.get('scientific_identity')!=scientific_identity()['sha256'] or set(prior.get('families',[]))!=set(cfg['families']):raise ValueError('Ten-scene scientific pilot gate is closed')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);(root/'scenes').mkdir(exist_ok=True);(root/'.staging').mkdir(exist_ok=True)
    mf=manifest(cfg);validate_manifest_profile(mf)
    if (root/'manifest.json').exists():
        if not resume:raise FileExistsError(root/'manifest.json')
        if json.loads((root/'manifest.json').read_text())!=mf:raise ValueError('Resume configuration/manifest mismatch')
    else:atomic_json(root/'manifest.json',mf)
    certificate_path=Path(certificate_path or cfg.get('scientific_certificate','outputs/lite-v01-production/scientific-certification.json'))
    certificate=validate_certificate(json.loads(certificate_path.read_text()))
    if cfg['renderer_backend']=='cuda_certified' and not certificate.get('cuda_certified'):raise ValueError('CUDA scientific fixture gate is closed')
    rows=[];results=[];failures=[];start=time.perf_counter();cpu=time.process_time();restored={}
    if (root/'report.json').exists():
        old=json.loads((root/'report.json').read_text());restored={r['slot']:r for r in old.get('scenes',[])};failures=list(old.get('failures',[]))
    # Recover a crash between atomic scene promotion and catalog publication.
    for path in (root/'scenes').iterdir():
        if path.is_dir():
            meta=json.loads((path/'verification.json').read_text())
            if meta['config_sha256']!=config_hash(cfg):raise ValueError('Published resume scene uses another config')
            result=json.loads((root/'scene-reports'/(path.name+'.json')).read_text())
            if result['slot'] in restored and restored[result['slot']]['scene_id']!=path.name:raise ValueError('Duplicate geometry slot during resume')
            restored[result['slot']]=result
    child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    with Telemetry(root) as telemetry,BackendFactory(cuda_batch=cfg.get('cuda_batch',32),cuda_allowed=certificate.get('cuda_certified',False)) as factory:
        for slot in range(scene_count):
            if slot in restored:
                result=restored[slot];rows.append(verify_lite_scene(root/'scenes'/result['scene_id']));results.append(result);continue
            if time.perf_counter()-start>cfg.get('pilot_seconds',7200):
                failures.append({'slot':slot,'status':'NOT_ATTEMPTED','reason':'declared pilot wall budget exhausted'});break
            family=cfg['families'][slot%len(cfg['families'])]
            try:
                checked,result=one_scene(cfg,root,factory,certificate,slot,family,0,resume=resume);rows.append(checked);results.append(result);catalogs(root,rows)
            except Exception as exc:
                failure={'slot':slot,'family':family,'status':'ABORTED' if isinstance(exc,(Aborted,BudgetExhausted)) else 'UNRESOLVED' if isinstance(exc,NumericalUnresolved) else 'FAILED',
                         'reason':type(exc).__name__+': '+str(exc),'elapsed_pilot_seconds':time.perf_counter()-start}
                failures.append(failure);append_ledger(root,{'event':'attempt_failed',**failure})
                # A READY record in .staging is only a provisional transaction
                # used for the standard-loader preflight. Roll it back on any
                # validation/loader failure; only atomic promotion publishes.
                for partial in (root/'.staging').glob(f'{slot:05d}-*'):
                    verification=partial/'verification.json'
                    if verification.exists():
                        value=json.loads(verification.read_text());value.update(status='INCOMPLETE',publication_failure=failure['reason']);atomic_json(verification,value)
                import traceback
                with (root/'failures.log').open('a') as f:traceback.print_exc(file=f)
            report={'format':'rind3d-lite-production-pilot-v0.1','complete':False,'requested_scenes':scene_count,
                    'ready_scenes':len(results),'config_sha256':config_hash(cfg),'scientific_identity':certificate['scientific_identity']['sha256'],
                    'families':cfg['families'],'scenes':results,'failures':failures,'wall_seconds_this_run':time.perf_counter()-start,
                    'cpu_seconds_this_run':time.process_time()-cpu,'factory':factory.report(),'mass_production':False}
            atomic_json(root/'report.json',report)
        # Catalog recovery also handles a prior interrupted atomic scene promotion.
        if rows:catalogs(root,rows)
        report={'format':'rind3d-lite-production-pilot-v0.1','complete':len(results)==scene_count,
                'requested_scenes':scene_count,'ready_scenes':len(results),'config_sha256':config_hash(cfg),
                'scientific_identity':certificate['scientific_identity']['sha256'],'families':cfg['families'],
                'scenes':results,'failures':failures,'wall_seconds_this_run':time.perf_counter()-start,
                'cpu_seconds_this_run':time.process_time()-cpu,'factory':factory.report(),'mass_production':False,
                'telemetry':telemetry.report()}
        if results:
            try:report['loader']=loader_check(root,workers=cfg.get('loader_workers',2))
            except Exception as exc:
                report['complete']=False;report['loader']={'passed':False,'reason':str(exc)};raise
            finally:atomic_json(root/'report.json',report)
        else:atomic_json(root/'report.json',report)
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        report['wall_seconds_this_run']=time.perf_counter()-start;report['cpu_seconds_this_run']=time.process_time()-cpu
        report['children_cpu_seconds_this_run']=children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime
        report['telemetry']=telemetry.report();atomic_json(root/'report.json',report)
    return report
