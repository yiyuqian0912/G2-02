"""Whitelist export: the mutable restricted campaign is never a public archive."""
import json,re,shutil,time
from collections import Counter
from pathlib import Path
import numpy as np
from rind3d_reader.catalog import Catalog,FORMAT
from rind3d_reader.durable import read_sealed,write_json,write_bytes,seal,digest,fsync_dir
from rind3d_reader.releases import seal_release,verify_release,manifest as release_manifest,sha_file
from .campaign import target

COPY_FILES=('geometry.npz','response.rindb','response.rindb.sha256','solid.rindb','solid.rindb.sha256','information.rindi','information.rindi.sha256','counts.rindc','counts.rindc.sha256','window_records.bin')

def audit_geometry(path):
 with np.load(path,allow_pickle=False) as f:
  for key in f.files:
   if not key.startswith(('geometry/','geometry_ref/','geometry_codecs/','geometry_csg/','scene_core/')):raise ValueError('Unrecognized canonical geometry field: '+key)
   a=f[key]
   if a.dtype.hasobject:raise ValueError('Object-valued geometry forbidden')
   if key.startswith('scene_core/') and a.size and not (key.endswith('offsets') and a.size==1 and int(a.flat[0])==0):raise ValueError('Geometry embeds private scene/source data')

def export_release(root,output,name,*,count=None,parent=None,delta=False,diagnostic=False):
 root=Path(root);output=Path(output)
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name):raise ValueError('Unsafe release name')
 if output.exists():raise FileExistsError('Immutable release destination must be new')
 campaign=json.loads((root/'manifest.json').read_text());cfg=read_sealed(root/'configuration.private.json');cert=json.loads((root/'certification.private.json').read_text())
 if campaign.get('diagnostic_only') and not diagnostic:raise ValueError('Diagnostic campaign cannot export a public production release')
 if not campaign.get('diagnostic_only') and diagnostic:raise ValueError('Wrong release diagnostic role')
 c=Catalog(root);rows=c.rows();count=len(rows) if count is None else count
 if not 0<count<=len(rows):raise ValueError('Release count must be a completed READY prefix')
 rows=rows[:count]
 for p in (root/'attempts').glob('*.json'):
  state=read_sealed(p)
  if state['attempt_id']<=rows[-1]['attempt_id'] and state['status'] not in ('READY','REJECTED_SOURCE'):raise ValueError('Export requires a completed attempt prefix, excluding certified source rejections')
 if name=='RIND3D-Lite-Core-1K-v0.1' and (count!=1000 or diagnostic):raise ValueError('Named 1K release requires 1000 production READY scenes')
 base=None
 if parent:
  verify_release(parent,full=True);base=release_manifest(parent)
  if base['scientific_configuration_sha256']!=campaign['configuration_sha256']:raise ValueError('Parent distribution differs')
  if base['scientific_build_identity']!=cert['build_identity']:raise ValueError('Build transition requires an explicitly versioned partition, not an implicit delta')
  base_ids=[r['scene_id'] for r in base['scene_records']]
  if [r['scene_id'] for r in rows[:len(base_ids)]]!=base_ids or len(rows)<=len(base_ids):raise ValueError('Release must strictly extend the parent READY prefix')
 elif delta:raise ValueError('Delta requires a verified parent')
 skip=len(base['scene_records']) if delta else 0
 output.mkdir(parents=True);(output/'scenes').mkdir();records=[]
 with Catalog(output,create=True) as exported:
  for i,row in enumerate(rows):
   if i<skip:records.append(base['scene_records'][i]);continue
   original=c.verify_scene(row,full=True);src=root/'scenes'/row['scene_id'];dst=output/'scenes'/row['scene_id'];dst.mkdir()
   audit_geometry(src/'geometry.npz')
   meta=json.loads((src/'verification.json').read_text())
   if meta.get('campaign_build_identity')!=cert['build_identity']['sha256'] or meta.get('config_sha256')!=campaign['configuration_sha256']:raise ValueError('Mixed scientific build/configuration inside release prefix')
   if meta.get('status')!='READY_LITE' or meta.get('production_mode')!='core' or meta.get('optional_search_budget_seconds')!=0:raise ValueError('Not a verified Core scene')
   if row['candidate_count_min']!=1 or row['candidate_count_max']!=1:raise ValueError('Core release requires exactly one private reference')
   for file in COPY_FILES:
    if file not in original['files']:raise ValueError('Unsealed export input')
    shutil.copyfile(src/file,dst/file)
    if sha_file(dst/file)!=original['files'][file]['sha256']:raise ValueError('Scientific copy mismatch')
   public={k:row[k] for k in ('scene_id','geometry_id','split','families','window_count','status')}
   # The private scene seal also commits operational journals/timings. Keep it
   # in the restricted campaign; public scientific identity depends only on
   # canonical scene/geometry, numerical policy and immutable scientific files.
   scientific_commitment=digest({'scene_id':row['scene_id'],'geometry_id':row['geometry_id'],'numerical_policy':meta['numerical_policy'],'scientific_files':{name:original['files'][name]['sha256'] for name in COPY_FILES}})
   public.update(resolution=1024,window_sides=[64,128,256],size_quotas={'64':85,'128':85,'256':86},reference_policy='one_independently_verified_private_truth; no uniqueness claim',privacy_policy='full-field-training-v1',scientific_scene_commitment=scientific_commitment,scientific_identity=meta['scientific_identity'],quality_identity=meta['quality_identity'],numerical_policy=meta['numerical_policy'])
   write_json(dst/'public_scene.json',public)
   # Public accepted-scene order never discloses private failed-attempt IDs.
   r={**{k:row[k] for k in ('scene_id','geometry_id','window_count','status','split','families')},'scene_order':i,'format':FORMAT}
   files={p.name:{'sha256':sha_file(p),'bytes':p.stat().st_size} for p in dst.iterdir() if p.is_file()}
   write_json(dst/'scene.seal.json',seal({k:r[k] for k in ('scene_id','geometry_id','window_count','status','split')}|{'catalog_record':r,'files':files}));r['seal_sha256']=sha_file(dst/'scene.seal.json');exported.commit(r);records.append(r)
 (output/'catalog'/'.writer.lock').unlink(missing_ok=True)
 if base and not delta and records[:len(base['scene_records'])]!=base['scene_records']:raise ValueError('Parent scientific records changed')
 repository=Path(__file__).resolve().parents[2]
 import importlib.util
 spec=importlib.util.spec_from_file_location('rind3d_reader_release_build',repository/'reader'/'build_backend.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.build_wheel(output/'sdk')
 docs=repository/'docs'/'public'
 for file in ('README.md','DATASET_CARD.md','FORMAT.md','PRODUCTION.md','RELEASES.md'):
  if not (docs/file).is_file():raise FileNotFoundError('Missing release documentation: '+file)
  shutil.copyfile(docs/file,output/file)
 if (docs/'REPORT.md').exists():shutil.copyfile(docs/'REPORT.md',output/'REPORT.md')
 example_root=Path(__file__).resolve().parents[2]/'examples'/'public_reader'
 if example_root.exists():
  (output/'examples').mkdir()
  for file in ('basic.py','torch_batches.py','visualize.py'):shutil.copyfile(example_root/file,output/'examples'/file)
 (output/'splits').mkdir()
 for split in ('train','val','test'):write_bytes(output/'splits'/f'{split}.txt',''.join(r['scene_id']+'\n' for r in records if r['split']==split).encode())
 payload={'format':'RIND3D','version':2,'name':name,'kind':'delta' if delta else 'full','catalog_format':FORMAT,'privacy_policy':'full-field-training-v1','diagnostic_only':diagnostic,'license':'PENDING — no redistribution license granted','citation':'PENDING — authors to be supplied','scene_count':count,'unique_geometry_count':count,'window_count':256*count,'split_counts':dict(Counter(r['split'] for r in records)),'family_counts':dict(Counter(f for r in records for f in r['families'])),'scene_records':records,'included_scene_ids':[r['scene_id'] for r in records[skip:]],'parent_release_id':base['release_id'] if base else None,'scientific_configuration_sha256':campaign['configuration_sha256'],'scientific_build_identity':cert['build_identity'],'numerical_contract':'canonical-binary64; exact closed-segment contact; independently certified A/B; UNKNOWN/ABORTED fail closed','distribution':{'generator':campaign['generator_distribution'],'family_probabilities':campaign['family_probabilities'],'family_schedule':'balanced cyclic over geometry ordinal','source_policy':'uniform FREE clearance-qualified sources; certified global coverage-ineligible sources resampled, same geometry retained; unproved failures incomplete','source_field_attempts':cfg['source_field_attempts'],'source_clearance_hex':cfg['driver_clearance_hex'],'window_sampling':cfg['observation_sampling'],'window_selection':cfg['window_selection'],'numerical_policy':cfg['numerical_policy']},'split_protocol':campaign['split_protocol'],'reader_compatibility':'rind3d-reader>=0.1,<0.2; Python>=3.12,<3.14','format_version':'RINDPack-v1; canonical-geometry-SoA-v1; immutable-scene-catalog-v1','public_task':'Full canonical geometry, full response/solid targets and local response windows are public. Not a blind hidden-field benchmark. Sources, candidates, witnesses and production recipes withheld.'}
 payload.update(dataset_profile='rind3d-lite-v0.1',profile_version='0.1',resolution=1024,windows_per_scene=256,window_sides=[64,128,256],size_quotas={'64':85,'128':85,'256':86},references={'stored_private_per_window':1,'lite_maximum':8,'additional_search_budget_seconds':0,'no_uniqueness_claim':True},publication_status='LOCAL_VERIFIED_CANDIDATE_LICENSE_PENDING',physical_verification_scope='Two independent certified implementations compare response and solid bits for complete brick coverage of every observation window; public readers verify integrity, restricted evidence retains hidden sources.')
 seal_release(output,payload);result=verify_release(output,full=True);fsync_dir(output);fsync_dir(output.parent);return result
