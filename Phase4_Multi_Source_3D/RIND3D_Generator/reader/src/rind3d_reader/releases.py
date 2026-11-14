"""Immutable full/delta snapshots and bounded, deterministic archive transport.

Checksums provide integrity and parent identity, not an external signature.
Physical certification remains in the restricted production evidence.
"""
import hashlib,json,os,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath
from .durable import digest,write_json,write_bytes,fsync_dir,read_sealed
from .catalog import Catalog

VERSION='rind3d-public-release-v1'
DOC_FILES=('README.md','DATASET_CARD.md','FORMAT.md','PRODUCTION.md','RELEASES.md','REPORT.md')
PUBLIC_FILES={'geometry.npz','response.rindb','response.rindb.sha256','solid.rindb','solid.rindb.sha256','information.rindi','information.rindi.sha256','counts.rindc','counts.rindc.sha256','window_records.bin','public_scene.json','scene.seal.json'}
SCIENTIFIC_FILES=PUBLIC_FILES-{'public_scene.json','scene.seal.json'}

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def safe_path(name):
 p=PurePosixPath(name)
 if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:raise ValueError('Unsafe release path')
 return p

def manifest(root):
 r=json.loads((Path(root)/'manifest.json').read_text());payload={k:v for k,v in r.items() if k!='release_id'}
 if r.get('release_format')!=VERSION or r.get('release_id')!=digest(payload) or r.get('privacy_policy')!='full-field-training-v1':raise ValueError('Invalid public release manifest')
 if r['kind'] not in ('full','delta','composed'):raise ValueError('Unknown release kind')
 return r

def verify_release(root,*,full=True,crc=False):
 root=Path(root);r=manifest(root);expected=set(r['files'])|{'manifest.json','checksums.sha256'}
 actual=set()
 for p in root.rglob('*'):
  if p.is_symlink():raise ValueError('Release symlink forbidden')
  if p.is_file():actual.add(p.relative_to(root).as_posix())
 if actual!=expected:raise ValueError('Unexpected or missing release files')
 lines={}
 for line in (root/'checksums.sha256').read_text().splitlines():
  checksum,name=line.split('  ',1);safe_path(name)
  if name in lines:raise ValueError('Duplicate checksum path')
  lines[name]=checksum
 if lines!={k:v['sha256'] for k,v in r['files'].items()}|{'manifest.json':sha_file(root/'manifest.json')}:raise ValueError('Checksum list/manifest conflict')
 for name,info in r['files'].items():
  safe_path(name);p=root/name
  parts=PurePosixPath(name).parts
  if not (name in DOC_FILES or parts[0] in ('scenes','splits','examples','sdk') or (len(parts)==3 and parts[:2]==('catalog','scenes'))):raise ValueError('Unknown public artifact path: '+name)
  if p.stat().st_size!=info['bytes'] or (full and sha_file(p)!=info['sha256']):raise ValueError('Release file integrity failure: '+name)
  if name.startswith('scenes/') and (len(PurePosixPath(name).parts)!=3 or p.name not in PUBLIC_FILES):raise ValueError('Private/unknown scene artifact in public release')
 c=Catalog(root)
 if r['kind']=='delta':
  ids=set(r['included_scene_ids'])
  if c.rows()!=[row for row in r['scene_records'] if row['scene_id'] in ids]:raise ValueError('Delta catalog identity mismatch')
  for row in c.rows():c.verify_scene(row,full=full)
 else:
  if len(c)!=r['scene_count'] or len(c.geometry_ids)!=r['scene_count']:raise ValueError('Release scene count mismatch')
  if c.rows()!=r['scene_records']:raise ValueError('Release catalog identity mismatch')
  for row in c.rows():c.verify_scene(row,full=full)
  for split in ('train','val','test'):
   ids=(root/'splits'/f'{split}.txt').read_text().splitlines()
   if ids!=[v['scene_id'] for v in c.rows() if v['split']==split]:raise ValueError('Split assignment mismatch')
 for row in c.rows():
  path=root/'scenes'/row['scene_id'];public=json.loads((path/'public_scene.json').read_text())
  if any(public.get(k)!=row[k] for k in ('scene_id','geometry_id','window_count','status','split')):raise ValueError('Public scientific scene/catalog conflict')
  if 'scientific_scene_commitment' in public:
   seal=read_sealed(path/'scene.seal.json');expected_commitment=digest({'scene_id':row['scene_id'],'geometry_id':row['geometry_id'],'numerical_policy':public['numerical_policy'],'scientific_files':{name:seal['files'][name]['sha256'] for name in SCIENTIFIC_FILES}})
   if public['scientific_scene_commitment']!=expected_commitment:raise ValueError('Scientific scene commitment mismatch')
 if crc:
  from .pack import RINDPack
  for row in c.rows():
   for field in ('response','solid'):
    with RINDPack(root/'scenes'/row['scene_id']/(field+'.rindb'),verify_file=full) as pack:pack.verify_all()
 return {'release_id':r['release_id'],'kind':r['kind'],'scenes':r['scene_count'],'included_scenes':len(r['included_scene_ids']),'windows':r['window_count'],'verified_files':len(r['files']),'full_sha':full,'crc':crc}

def seal_release(root,payload):
 root=Path(root);files={}
 for p in sorted(root.rglob('*')):
  if p.is_symlink():raise ValueError('No release symlinks')
  if p.is_file():
   name=p.relative_to(root).as_posix()
   if name in ('manifest.json','checksums.sha256'):raise FileExistsError('Release already sealed')
   files[name]={'sha256':sha_file(p),'bytes':p.stat().st_size}
 r={**payload,'release_format':VERSION,'files':files};r['release_id']=digest(r);write_json(root/'manifest.json',r,replace=False)
 checks={k:v['sha256'] for k,v in files.items()}|{'manifest.json':sha_file(root/'manifest.json')}
 write_bytes(root/'checksums.sha256',''.join(f'{v}  {k}\n' for k,v in sorted(checks.items())).encode(),replace=False)
 return r

def pack_release(root,archive):
 import zstandard as zstd
 root=Path(root);archive=Path(archive)
 verify_release(root,full=True)
 if archive.exists() or root in archive.resolve().parents:raise FileExistsError('Archive must be new and outside snapshot')
 archive.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+archive.name+'.',dir=archive.parent);os.close(fd)
 try:
  with open(tmp,'wb') as raw:
   with zstd.ZstdCompressor(level=6,threads=0).stream_writer(raw,closefd=False) as compressed:
    with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
     for p in [root]+sorted(root.rglob('*')):
      if p.is_symlink():raise ValueError('Unexpected archive symlink')
      name=(Path(root.name)/p.relative_to(root)).as_posix();info=tarfile.TarInfo(name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
      if p.is_dir():info.type=tarfile.DIRTYPE;info.mode=0o755;tar.addfile(info)
      elif p.is_file():
       info.size=p.stat().st_size;info.mode=0o644
       with p.open('rb') as f:tar.addfile(info,f)
      else:raise ValueError('Special archive file forbidden')
   raw.flush();os.fsync(raw.fileno())
  os.link(tmp,archive);fsync_dir(archive.parent)
 finally:Path(tmp).unlink(missing_ok=True)
 checksum=sha_file(archive);write_bytes(archive.with_name(archive.name+'.sha256'),f'{checksum}  {archive.name}\n'.encode(),replace=False)
 return {'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum}

def extract_release(archive,destination,*,expected_sha256=None):
 import zstandard as zstd
 archive=Path(archive);destination=Path(destination)
 if expected_sha256 is None:expected_sha256=archive.with_name(archive.name+'.sha256').read_text().split()[0]
 if sha_file(archive)!=expected_sha256:raise ValueError('Archive checksum mismatch')
 if destination.exists():raise FileExistsError('Extraction requires a fresh destination')
 destination.mkdir(parents=True);top=None;seen=set()
 try:
  with archive.open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as expanded,tarfile.open(fileobj=expanded,mode='r|') as tar:
   for entry in tar:
    p=safe_path(entry.name.rstrip('/'))
    if entry.name in seen or not (entry.isdir() or entry.isfile()):raise ValueError('Duplicate/link/special archive entry')
    seen.add(entry.name)
    if top is None:top=p.parts[0]
    if p.parts[0]!=top:raise ValueError('Multiple archive roots')
    target=destination/str(p)
    if entry.isdir():target.mkdir(parents=True,exist_ok=True)
    else:
     target.parent.mkdir(parents=True,exist_ok=True)
     with tar.extractfile(entry) as source,target.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
  if top is None:raise ValueError('Empty archive')
  result=verify_release(destination/top,full=True)
 except BaseException:
  # Retain the isolated failed extraction for inspection; never mark it valid.
  raise
 return {'root':str(destination/top),**result}

def apply_delta(base,delta,output):
 """Compose into a new directory; immutable base and delta are never edited."""
 base=Path(base);delta=Path(delta);output=Path(output)
 verify_release(base,full=True);verify_release(delta,full=True);a=manifest(base);b=manifest(delta)
 if b['kind']!='delta' or b['parent_release_id']!=a['release_id']:raise ValueError('Delta parent identity mismatch')
 if a['scientific_configuration_sha256']!=b['scientific_configuration_sha256']:raise ValueError('Incompatible scientific distribution')
 if a['scientific_build_identity']!=b['scientific_build_identity']:raise ValueError('Incompatible scientific build transition')
 if b['scene_records'][:len(a['scene_records'])]!=a['scene_records']:raise ValueError('Delta modifies existing scene records or splits')
 if output.exists():raise FileExistsError('Composition requires a fresh output')
 output.mkdir(parents=True)
 try:
  for source in (base,delta):
   for p in sorted(source.rglob('*')):
    if not p.is_file():continue
    name=p.relative_to(source).as_posix()
    if name in ('manifest.json','checksums.sha256') or name.startswith(('splits/','sdk/','examples/')) or name in DOC_FILES:continue
    target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('Delta refuses conflicting overwrite: '+name)
    shutil.copyfile(p,target)
  for name in DOC_FILES:
   if (delta/name).exists():shutil.copyfile(delta/name,output/name)
  for name in ('sdk','examples'):
   source=delta/name if (delta/name).exists() else base/name
   if source.exists():shutil.copytree(source,output/name)
  (output/'splits').mkdir()
  for split in ('train','val','test'):write_bytes(output/'splits'/f'{split}.txt',''.join(r['scene_id']+'\n' for r in b['scene_records'] if r['split']==split).encode())
  payload={k:v for k,v in b.items() if k not in ('release_id','files','release_format')};payload.update(kind='composed',included_scene_ids=[r['scene_id'] for r in b['scene_records']],composed_from=[a['release_id'],b['release_id']])
  seal_release(output,payload);return verify_release(output,full=True)
 except BaseException:raise
