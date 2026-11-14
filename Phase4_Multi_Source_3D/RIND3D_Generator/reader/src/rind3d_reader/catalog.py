"""Immutable per-scene catalog, O(1) commits and O(N) open/reconstruction."""
import json,re,time
from pathlib import Path
from .durable import canonical,digest,write_bytes,fsync_dir,lock,read_sealed

FORMAT='rind3d-immutable-scene-catalog-v1'
HEX=re.compile(r'^[0-9a-f]{64}$')
def order_id(row):return row.get('scene_order',row.get('attempt_id'))

class Catalog:
 def __init__(self,root,*,create=False,synthetic=False):
  self.root=Path(root);self.directory=self.root/'catalog'/'scenes';self.synthetic=synthetic;self.by_id={};self.by_attempt={};self.geometry_ids={}
  self._owner=None
  if create:
   self.directory.mkdir(parents=True,exist_ok=True);fsync_dir(self.directory.parent)
   owner=lock(self.root/'catalog'/'.writer.lock');owner.__enter__();self._owner=owner
  if not self.directory.is_dir():raise ValueError('Missing immutable scene catalog')
  for path in sorted(self.directory.glob('*.json')):
   raw=path.read_bytes();parts=path.name.split('.')
   if len(parts)!=3 or not HEX.fullmatch(parts[0]) or not HEX.fullmatch(parts[1]):raise ValueError('Malformed catalog filename')
   import hashlib
   if hashlib.sha256(raw).hexdigest()!=parts[1]:raise ValueError('Catalog record checksum mismatch: '+path.name)
   row=json.loads(raw);self._validate(row)
   if row['scene_id']!=parts[0]:raise ValueError('Catalog identity/name mismatch')
   self._remember(row,path.name)
 def _validate(self,row):
  if row.get('format')!=FORMAT or row.get('status')!=('SYNTHETIC' if self.synthetic else 'READY_LITE'):raise ValueError('Non-READY or unsupported catalog record')
  if not all(HEX.fullmatch(row.get(k,'')) for k in ('scene_id','geometry_id','seal_sha256')):raise ValueError('Invalid catalog identity')
  if ('scene_order' in row and 'attempt_id' in row) or type(order_id(row)) is not int or order_id(row)<0 or row.get('window_count')!=256:raise ValueError('Invalid catalog order/count')
  if row.get('split') not in ('train','val','test'):raise ValueError('Invalid split')
 def _remember(self,row,name):
  sid=row['scene_id'];attempt=order_id(row);gid=row['geometry_id']
  if sid in self.by_id or attempt in self.by_attempt or gid in self.geometry_ids:raise ValueError('Duplicate/conflicting scene, attempt or geometry identity')
  self.by_id[sid]=(row,name);self.by_attempt[attempt]=sid;self.geometry_ids[gid]=sid
 def __len__(self):return len(self.by_id)
 def rows(self):return [self.by_id[self.by_attempt[k]][0] for k in sorted(self.by_attempt)]
 def get(self,sid):return self.by_id[sid][0]
 def commit(self,row):
  import hashlib
  if self._owner is None:raise RuntimeError('Catalog was not opened for exclusive writing')
  self._validate(row);sid=row['scene_id']
  if sid in self.by_id:
   if self.get(sid)!=row:raise ValueError('Conflicting immutable catalog update')
   return {'committed':False,'idempotent':True}
  if order_id(row) in self.by_attempt or row['geometry_id'] in self.geometry_ids:raise ValueError('Order/geometry already committed')
  if not self.synthetic:self.verify_scene(row,durable=True)
  raw=canonical(row)+b'\n';name=f'{sid}.{hashlib.sha256(raw).hexdigest()}.json';start=time.perf_counter()
  write_bytes(self.directory/name,raw,replace=False);self._remember(row,name)
  return {'committed':True,'wall_seconds':time.perf_counter()-start,'bytes':len(raw)}
 def verify_scene(self,row,*,full=False,durable=False):
  import hashlib,os
  path=self.root/'scenes'/row['scene_id'];seal_path=path/'scene.seal.json'
  if path.is_symlink() or not path.is_dir():raise ValueError('Unsafe/missing scene seal directory')
  if seal_path.is_symlink() or not seal_path.is_file() or hashlib.sha256(seal_path.read_bytes()).hexdigest()!=row['seal_sha256']:raise ValueError('Missing/corrupt scene seal')
  seal=read_sealed(seal_path)
  if any(seal.get(k)!=row[k] for k in ('scene_id','geometry_id','window_count','status','split')):raise ValueError('Catalog/scene seal conflict')
  for name,info in seal['files'].items():
   p=path/name
   if p.parent!=path or p.is_symlink() or not p.is_file() or p.stat().st_size!=info['bytes']:raise ValueError('Missing/unsafe/changed sealed file: '+name)
   if full:
    h=hashlib.sha256()
    with p.open('rb') as f:
     for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=info['sha256']:raise ValueError('Scientific file checksum mismatch: '+name)
   if durable:
    with p.open('rb') as f:os.fsync(f.fileno())
  if durable:
   with seal_path.open('rb') as f:os.fsync(f.fileno())
   fsync_dir(path);fsync_dir(path.parent)
  return seal
 def verify_records(self):
  other=Catalog(self.root,synthetic=self.synthetic)
  if other.rows()!=self.rows():raise ValueError('Catalog changed during integrity verification')
  return len(other)
 def close(self):
  if getattr(self,'_owner',None) is not None:self._owner.__exit__(None,None,None);self._owner=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def __del__(self):self.close()
