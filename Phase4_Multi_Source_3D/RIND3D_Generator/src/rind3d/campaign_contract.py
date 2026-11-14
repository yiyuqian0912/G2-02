"""Frozen campaign identity, independent of cumulative target and sessions."""
import hashlib,hmac,importlib,importlib.metadata,json,sys,platform
from pathlib import Path
from .lite_certification import scientific_identity
from .quality_certification import identity as quality_identity

VERSION='rind3d-core-campaign-v2'
def canonical(v):
 from rind3d_reader.durable import canonical as encode
 return encode(v)
def sha(v):return hashlib.sha256(canonical(v)).hexdigest()
def derived_seed(master,attempt,tag):
 if attempt<0 or tag not in ('geometry','source','window'):raise ValueError('Invalid seed derivation domain')
 return int.from_bytes(hmac.new(bytes.fromhex(master),f'{VERSION}:{attempt}:{tag}'.encode(),hashlib.sha256).digest()[:16],'little')
def build_identity():
 names=('rind3d.campaign_contract','rind3d.campaign_scene','rind3d.campaign','rind3d.campaign_verify','rind3d.source_eligibility','rind3d_reader.catalog','rind3d_reader.durable','rind3d_reader.data','rind3d_reader.pack','rind3d_reader.counts')
 values={name:hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in names}
 # Include transitive producer gates/helpers, especially pipeline.verify_scene.
 # Release transport/CLI packaging is separately checksummed in each snapshot.
 base=Path(__file__).parent
 for path in sorted(base.rglob('*.py')):
  if path.name in ('release_export.py','campaign_cli.py'):continue
  values['producer-source/'+path.relative_to(base).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
 values['scientific']=scientific_identity()['sha256'];values['quality']=quality_identity()['sha256']
 runtime={'python':sys.version,'abi':sys.implementation.cache_tag,'machine':platform.machine(),'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','pyarrow','zstandard','torch','rind-dataset')}}
 lock=Path(__file__).resolve().parents[2]/'uv.lock'
 runtime['producer_lock_sha256']=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
 payload={'files':values,'runtime':runtime}
 return {**payload,'sha256':sha(payload)}
def validate_certificate(record,*,current=True,allow_diagnostic=False):
 if record.get('diagnostic_only'):
  if not allow_diagnostic:raise ValueError('Diagnostic campaign cannot authorize production')
 elif not record.get('complete') or not record.get('recovery_cuda') or not record.get('recovery_cpu') or not record.get('regression_passed'):raise ValueError('Campaign production certification gate closed')
 if record.get('version') not in (VERSION,'rind3d-core-campaign-v1') or (current and record.get('version')!=VERSION):raise ValueError('Unsupported campaign protocol')
 if record.get('primary_leaf') not in (2,4,8):raise ValueError('Uncertified primary traversal')
 if current and record.get('build_identity')!=build_identity():raise ValueError('Frozen campaign build changed')
 return record
def check_scene(path,meta):
 p=Path(path)/'campaign_certification.private.json';record=validate_certificate(json.loads(p.read_text()),current=False,allow_diagnostic=True)
 if meta.get('campaign_version')!=record['version'] or meta.get('campaign_build_identity')!=record['build_identity']['sha256'] or meta.get('primary_leaf')!=record['primary_leaf']:raise ValueError('Scene/campaign certification mismatch')
 return record
