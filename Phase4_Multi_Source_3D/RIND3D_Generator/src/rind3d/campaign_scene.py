"""Single-scene Core work; scientific gates reuse the original implementations."""
from collections import Counter
from pathlib import Path
import json,os,time
import numpy as np
from .pack import RINDPack,sha256_file
from .geometry import save_geometry,load_geometry
from .lite_generators import generate
from .lite_acceptance import geometry_acceptance,sample_source
from .rendering.context import GeometryContext
from .rendering.polyhedra import geometry_split
from .stream import stream_response
from .information import InformationIndex
from .counts import CountIndex
from .stratified_windows import select,DatasetBalance
from .pipeline import write_candidate_files,parquet_write,verify_scene
from .search import CandidateBank
from .core_research import _windows
from .physical_evidence import load_evidence
from .quality_certification import VERSION as CORE_VERSION
from .window_contract import WINDOW_PROTOCOL
from .campaign_contract import derived_seed,VERSION
from rind3d_reader.catalog import FORMAT
from rind3d_reader.durable import write_json,seal,read_sealed,fsync_dir

def prepare(root,cfg,master,attempt,balance,factory,geometry_ordinal=None,source_trial=0):
 geometry_ordinal=attempt if geometry_ordinal is None else geometry_ordinal
 root=Path(root);stage=root/'.staging'/f'{attempt:012d}';stage.mkdir(parents=True,exist_ok=True);fsync_dir(stage.parent);start=time.perf_counter();cpu=time.process_time();stages={};family=cfg['families'][geometry_ordinal%len(cfg['families'])]
 gseed=derived_seed(master,geometry_ordinal,'geometry');sseed=derived_seed(master,attempt,'source');wseed=derived_seed(master,attempt,'window');tick=time.perf_counter();c=time.process_time()
 input_path=stage/'input.private.json'
 if input_path.exists():
  inputs=read_sealed(input_path)
  if inputs['attempt_id']!=attempt or inputs['family']!=family or inputs['seeds']!=[gseed,sseed,wseed] or inputs['balance_before']!=balance:raise ValueError('Attempt input/seed/balance conflict')
  bundle=load_geometry(stage/'geometry.npz');truth=np.fromfile(stage/'driver.f64',dtype='<f8');ctx=factory.context(bundle,cfg['numerical_policy'])
 else:
  bundle,recipe=generate(gseed,geometry_ordinal,family,0)
  if recipe['proposal_failures']:raise ValueError('Geometry proposal could not fit declared world domain')
  ctx=factory.context(bundle,cfg['numerical_policy']);accepted=geometry_acceptance(ctx,family,gseed,geometry_ordinal);save_geometry(stage/'geometry.npz',bundle)
  write_json(stage/'recipe.private.json',recipe);write_json(stage/'geometry_acceptance.private.json',accepted);truth,source=sample_source(ctx,sseed,attempt,limit=cfg['source_attempts'])
  with (stage/'driver.f64').open('wb') as f:f.write(truth.astype('<f8').tobytes());f.flush();os.fsync(f.fileno())
  write_json(stage/'sources.private.json',{'truth':source,'bank':[source['clearance']]});inputs={'attempt_id':attempt,'family':family,'seeds':[gseed,sseed,wseed],'balance_before':balance,'geometry_id':ctx.geometry_id,'scene_id':source['canonical_scene_id'],'source':truth.tolist()};write_json(input_path,seal(inputs));fsync_dir(stage)
 stages['geometry_and_source']={'wall_seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-c};tick=time.perf_counter();c=time.process_time()
 base=factory.source(bundle,truth,cfg['renderer_backend'],cfg['numerical_policy'])
 class ProgressRenderer:
  def __init__(self):self.batches=0
  def __getattr__(self,name):return getattr(base,name)
  def packed_batch(self,*args,**kwargs):
   self.batches+=1
   if self.batches<=2:fsync_dir(stage)
   if self.batches==2:write_json(root/'phase-markers'/f'{attempt:012d}-rendering.json',{'attempt_id':attempt,'pid':os.getpid(),'stage':'field_rendering_after_first_batch'})
   return base.packed_batch(*args,**kwargs)
 renderer=ProgressRenderer()
 try:field=stream_response(ctx,truth,stage,cfg,resume=True,renderer=renderer);source_id=renderer.source_id
 finally:renderer.close()
 stages['complete_field_and_index']={'wall_seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-c};tick=time.perf_counter();c=time.process_time()
 b=DatasetBalance();b.bins[:]=balance['bins'];b.components.update(balance['components'])
 with RINDPack(stage/'response.rindb',verify_file=True) as y,RINDPack(stage/'solid.rindb',verify_file=True) as m:
  info=InformationIndex.load(stage/'information.rindi',y,m);p=stage/'counts.rindc';counts=CountIndex.load(p,y,m) if p.exists() else CountIndex.build(y,m)
  if not p.exists():counts.save(p)
  from .source_eligibility import certificate as eligibility_certificate,recheck,SourceIneligible
  rejection=eligibility_certificate(counts)
  if rejection is not None:
   recheck(y,m,rejection);write_json(stage/'source-rejection.private.json',{'proof':rejection,'independent_packed_recheck':True,'geometry_ordinal':geometry_ordinal,'source_trial':source_trial,'field':field,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'scene_id':inputs['scene_id']});raise SourceIneligible(rejection['reason'])
  selection=select(info,counts,truth,seed=wseed,scene=attempt,balance=b,observation=cfg['observation_sampling'],natural_settings=cfg['window_selection'])
 write_json(stage/'selection.private.json',selection.report)
 if not selection.report['complete'] or selection.report['placement']['occupied_octants']<2:raise RuntimeError('INCOMPLETE window construction; no infeasibility proof, not physical rejection')
 write_json(stage/'selection.private.json',selection.report);parquet_write(stage/'window_selection.parquet',selection.provenance);parquet_write(stage/'uniform_control.parquet',selection.controls)
 stages['window_selection']={'wall_seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-c}
 desc={'attempt_id':attempt,'slot':attempt,'family':family,'stage':str(stage.resolve()),'cfg':cfg,'geometry_id':ctx.geometry_id,'scene_id':inputs['scene_id'],'source_id':source_id,'source':truth.tolist(),'windows':selection.provenance,'selection':selection.report,'field':field,'balance_before':balance,'balance_after':b.json(),'stages':stages,'prepare_wall_seconds':time.perf_counter()-start,'prepare_cpu_seconds':time.process_time()-cpu,'verifier':'silhouette'}
 desc.update(geometry_ordinal=geometry_ordinal,source_trial=source_trial)
 for p in stage.iterdir():
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
 fsync_dir(stage)
 write_json(root/'jobs'/f'{attempt:012d}.json',seal(desc));return desc

def finish(desc,root,certificate,quality,campaign_certificate,*,fault=None):
 from rind3d_reader.data import RINDDataset,make_dataloader
 root=Path(root);stage=Path(desc['stage']);cfg=desc['cfg'];start=time.perf_counter();cpu=time.process_time();windows=_windows(desc);evidence=read_json= json.loads((stage/'physical_evidence.private.json').read_text());ctx=GeometryContext(load_geometry(stage/'geometry.npz'),cfg['numerical_policy']);truth=np.asarray(desc['source'],dtype='<f8');physical,_=load_evidence(stage/'physical_evidence.private.json',evidence['key'],windows)
 for w in windows:physical.verify_window(w,ctx,truth)
 bank=CandidateBank();key=bank.add(truth);reports=[{'status':'COMPLETE','actual_candidates':1,'all_grid_exact':True,'independent_validation_passed':True,'search_termination_reason':'core_optional_search_disabled','uniqueness_proven':False} for _ in windows]
 dtype,counts=write_candidate_files(stage,windows,bank,[[key] for _ in windows],reports)
 write_json(stage/'candidate_search.private.json',{'mode':'core','optional_search_seconds':0,'candidate_counts':{'1':256},'count_one_is_not_uniqueness':True,'reports':reports})
 for name,value in (('scientific_certification.json',certificate),('quality_certification.json',quality),('campaign_certification.private.json',campaign_certificate)):write_json(stage/name,value)
 journal=stage/'physical-work.private.json'
 if journal.exists():os.replace(journal,root/'work-journals'/f"{desc['attempt_id']:012d}.json");fsync_dir(root/'work-journals')
 hashes={p.name:(p.with_name(p.name+'.sha256').read_text().split()[0] if p.name in ('response.rindb','solid.rindb') else sha256_file(p)) for p in stage.iterdir() if p.is_file() and p.name not in ('verification.json','scene.seal.json')}
 meta={'dataset_profile':'rind3d-lite-v0.1','profile_version':'0.1','status':'READY_LITE','contract_version':2,'window_protocol':WINDOW_PROTOCOL,'resolution':1024,'max_candidates':8,'true_source_included':True,'min_source_separation':32/1024,'candidate_id_dtype':dtype,'candidate_storage':'pool-offsets-v2','scene_id':desc['scene_id'],'source_id':desc['source_id'],'geometry_id':ctx.geometry_id,'geometry_hash':ctx.geometry_id,'split':geometry_split(ctx.geometry_id,0),'split_group':ctx.geometry_id,'split_seed':0,'families':[desc['family']],'config_sha256':campaign_certificate['configuration_sha256'],'numerical_policy':cfg['numerical_policy'],'numerical_policy_sha256':ctx.policy_hash,'candidate_domain':'world_outside_window','equivalence_status':'grid_exact','continuous_certified':False,'generation':desc['field'],'physical_backends':[evidence['key']['primary_backend'],evidence['key']['independent_backend']],'scientific_identity':certificate['scientific_identity']['sha256'],'hashes':hashes,'production_revision':CORE_VERSION,'quality_identity':quality['identity']['sha256'],'production_mode':'core','optional_search_budget_seconds':0,'observation_sampling':cfg['observation_sampling']['mode'],'campaign_version':VERSION,'campaign_build_identity':campaign_certificate['build_identity']['sha256'],'primary_leaf':campaign_certificate['primary_leaf']}
 write_json(stage/'verification.json',meta);checked=verify_scene(stage,full_index=True)
 for view in ('model','candidate_conditioned'):
  with RINDDataset._from_checked_scene(stage,checked['scene_row'],view=view) as d:
   batch=next(iter(make_dataloader(d,batch_size=1,num_workers=0)))
   if ('candidate_sources' in batch)!=(view=='candidate_conditioned'):raise ValueError('Private label isolation failed')
 row=dict(checked['scene_row'],format=FORMAT,attempt_id=desc['attempt_id']);row.pop('bytes',None)
 files={p.name:{'sha256':hashes[p.name] if p.name in hashes else sha256_file(p),'bytes':p.stat().st_size} for p in stage.iterdir() if p.is_file() and p.name!='scene.seal.json'}
 write_json(stage/'scene.seal.json',seal({k:row[k] for k in ('scene_id','geometry_id','window_count','status','split')}|{'catalog_record':row,'files':files}));row['seal_sha256']=sha256_file(stage/'scene.seal.json')
 target=root/'scenes'/desc['scene_id']
 if target.exists():raise FileExistsError('Scene already durably published; recover catalog instead')
 metrics={'attempt_id':desc['attempt_id'],'scene_id':desc['scene_id'],'geometry_id':ctx.geometry_id,'family':desc['family'],'stages':desc['stages']|{'independent_physical_verification':{'wall_seconds':evidence['wall_seconds'],'cpu_seconds':sum(s['cpu_seconds'] for s in evidence['stages'])},'core_gate_and_publication':{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu}},'physical':evidence['stages'],'field':desc['field'],'selection':desc['selection'],'scientific_bytes':sum(v['bytes'] for v in files.values())+(stage/'scene.seal.json').stat().st_size,'scene_rename_durable':False,'catalog_committed':False,'publication_time_is_lower_bound':True}
 report_path=root/'scene-reports'/f"{desc['attempt_id']:012d}.json";write_json(report_path,seal(metrics))
 for p in stage.iterdir():
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
 fsync_dir(stage);os.rename(stage,target);fsync_dir(target.parent);fsync_dir(stage.parent)
 metrics['stages']['core_gate_and_publication']={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu};metrics.update(scene_rename_durable=True,publication_time_is_lower_bound=False);write_json(report_path,seal(metrics))
 if fault:fault('after_scene_rename',desc)
 return row,metrics
