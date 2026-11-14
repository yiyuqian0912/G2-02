"""CPU reader with lazy scientific files and compact per-scene window indexes."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,os
import numpy as np
from .pack import RINDPack,PackedWindow
from .catalog import Catalog,FORMAT
from .counts import CountIndex

class Scene:
 def __init__(self,dataset,row):self.dataset=dataset;self.metadata=dict(row);self.scene_id=row['scene_id']
 def roi(self,origin,side,field='response',*,packed=True):
  if field not in ('response','solid'):raise ValueError('Unknown public field')
  return self.dataset._open(self.scene_id)[field].read_window(*origin,side,packed=packed)
 def window(self,window_id):
  item=self.dataset._open(self.scene_id);r=item['windows'][window_id];return self.roi(tuple(map(int,r[:3])),int(r[3]))
 def counts(self,origin,side):
  item=self.dataset._open(self.scene_id)
  if 'counts' not in item:
   path=item['path']/'counts.rindc';item['counts']=CountIndex.load(path,item['response'],item['solid']) if path.exists() else CountIndex.build(item['response'],item['solid'])
  return item['counts'].query(origin,side)

class RINDDataset:
 @classmethod
 def _from_checked_scene(cls,path,row,*,view='model'):
  """Producer preflight after mandatory scientific verification; not a gate."""
  d=object.__new__(cls);d.root=Path(path).parent;d._paths={row['scene_id']:Path(path)};d.manifest={'privacy_policy':'restricted-production-v1'};d.public=False;d.view=view;d.response_format='packed';d.cache_scenes=1;d.cache_bricks=16;d.cache=OrderedDict();d._pid=os.getpid();d.catalog=None;d.scenes={row['scene_id']:row};d.rows=[row];d.windows={row['scene_id']:np.arange(row['window_count'],dtype=np.uint16)};d.offsets=np.array([0,row['window_count']],dtype=np.int64);return d
 def _path(self,sid):return getattr(self,'_paths',{}).get(sid,self.root/'scenes'/sid)
 def __init__(self,root,split='train',*,window_sizes=None,view='model',response_format='packed',cache_scenes=2,cache_bricks=16,include_families=None,exclude_families=None,family_match='subset',**legacy_options):
  self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text());self.view=view;self.response_format=response_format;self.cache_scenes=cache_scenes;self.cache_bricks=cache_bricks;self.cache=OrderedDict();self._pid=os.getpid();self.catalog=None
  if split not in (None,'train','val','test') or view not in ('model','candidate_conditioned') or response_format not in ('packed','bool','torch.bool') or cache_scenes<1:raise ValueError('Invalid reader option')
  self.public=self.manifest.get('privacy_policy')=='full-field-training-v1'
  if self.manifest.get('release_format')=='rind3d-public-release-v1' and self.manifest.get('kind')=='delta':raise ValueError('Apply the delta to its verified parent before opening the complete dataset')
  if self.manifest.get('release_format')=='rind3d-public-release-v1' and not self.public:raise ValueError('Public release privacy policy changed')
  for name,value in legacy_options.items():
   if name=='require_ready' and value is True:continue
   if name=='allow_shortfall' and value is False:continue
   if name=='equivalent_count' and value in (None,1):continue
   if name=='occupancy_range' and value is None:continue
   if name=='legacy_diagnostics' and value is False:continue
   raise ValueError('Unsupported reader option: '+name)
  if self.public and view!='model':raise PermissionError('Private candidate labels are not in this public release')
  sides=set((64,128,256) if window_sizes is None else window_sizes)
  if not sides or not sides<={64,128,256}:raise ValueError('Exact window sides must be 64/128/256')
  if self.manifest.get('catalog_format')==FORMAT:
   self.catalog=Catalog(self.root);rows=self.catalog.rows()
  elif self.manifest.get('format')=='RIND3D':
   import pyarrow.parquet as pq
   rows=pq.read_table(self.root/'scenes.parquet').to_pylist()
   for i,r in enumerate(rows):r['attempt_id']=i
  else:raise ValueError('Unsupported dataset/catalog manifest')
  self.scenes={};self.rows=[];self.windows={};self.offsets=[0]
  for row in rows:
   if row.get('status') not in ('READY_LITE','READY'):raise ValueError('Non-READY catalog scene')
   if split is not None and row['split']!=split:continue
   families=set(row.get('families',[]));allowed=set(include_families or [])
   if allowed:
    match={'any':bool(families&allowed),'all':allowed<=families,'subset':families<=allowed,'exact':families==allowed}
    if family_match not in match or not match[family_match]:continue
   if exclude_families and families&set(exclude_families):continue
   self.scenes[row['scene_id']]=row;self.rows.append(row)
   # Lite order is fixed by the canonical selector; avoid reading 10K window
   # files during ordinary open. A side-filter requires only its small index.
   if sides=={64,128,256}:ids=np.arange(row['window_count'],dtype=np.uint16)
   else:
    w=np.fromfile(self.root/'scenes'/row['scene_id']/'window_records.bin',dtype='<u2').reshape(-1,4);ids=np.flatnonzero(np.isin(w[:,3],list(sides))).astype(np.uint16)
   self.windows[row['scene_id']]=ids;self.offsets.append(self.offsets[-1]+len(ids))
  self.offsets=np.asarray(self.offsets,dtype=np.int64)
 def __len__(self):return int(self.offsets[-1])
 def metadata(self):return dict(self.manifest)
 def iter_scenes(self):
  for row in self.rows:yield Scene(self,row)
 def scene(self,scene_id):return Scene(self,self.scenes[scene_id])
 def _open(self,sid):
  if self._pid!=os.getpid():self.close();self._pid=os.getpid()
  if sid in self.cache:
   item=self.cache.pop(sid);self.cache[sid]=item;return item
  row=self.scenes[sid];path=self._path(sid)
  if self.catalog:self.catalog.verify_scene(row)
  with np.load(path/'geometry.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files if not k.startswith(('scene_','physics_'))}
  for a in arrays.values():a.setflags(write=False)
  windows=np.fromfile(path/'window_records.bin',dtype='<u2')
  if windows.size!=row['window_count']*4:raise ValueError('Truncated window index')
  windows=windows.reshape(-1,4)
  if np.any(~np.isin(windows[:,3],[64,128,256])) or np.any(windows[:,:3]+windows[:,3,None]>1024):raise ValueError('Invalid canonical window coordinates')
  item={'path':path,'windows':windows,'structure':{'format':'canonical-geometry-SoA-v1','arrays':arrays},'response':RINDPack(path/'response.rindb',cache_bricks=self.cache_bricks),'solid':RINDPack(path/'solid.rindb',cache_bricks=self.cache_bricks)}
  self.cache[sid]=item
  while len(self.cache)>self.cache_scenes:
   _,old=self.cache.popitem(last=False);old['response'].close();old['solid'].close()
  return item
 def __getitem__(self,index):
  if index<0:index+=len(self)
  if not 0<=index<len(self):raise IndexError(index)
  scene_index=int(np.searchsorted(self.offsets,index,side='right')-1);row=self.rows[scene_index];sid=row['scene_id'];wid=int(self.windows[sid][index-int(self.offsets[scene_index])]);item=self._open(sid);w=item['windows'][wid];side=int(w[3]);y=item['response'].read_window(*map(int,w[:3]),side,packed=self.response_format=='packed')
  if self.response_format=='torch.bool':
   import torch
   y=torch.from_numpy(y)
  sample={'scene_id':sid,'window_id':wid,'window_origin':w[:3].astype(np.int64),'window_size':side,'structure':item['structure'],'observed_response':y}
  if self.view=='candidate_conditioned':sample['candidate_sources']=self.private_oracle().get_candidate_sources(sid,wid)
  return sample
 def private_oracle(self):
  if self.public:raise PermissionError('Public artifact policy withholds sources and oracle evidence')
  return PrivateOracle(self)
 def verify(self,*,full=True,crc=True):
  for row in self.rows:
   if self.catalog:self.catalog.verify_scene(row,full=full)
   elif full:
    path=self._path(row['scene_id']);meta=json.loads((path/'verification.json').read_text())
    for name,expected in meta['hashes'].items():
     if Path(name).name!=name:raise ValueError('Unsafe scientific file name')
     h=hashlib.sha256()
     with (path/name).open('rb') as f:
      for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
     if h.hexdigest()!=expected:raise ValueError('Legacy scientific SHA mismatch: '+name)
   if crc:
    with RINDPack(self.root/'scenes'/row['scene_id']/'response.rindb',verify_file=full) as y,RINDPack(self.root/'scenes'/row['scene_id']/'solid.rindb',verify_file=full) as m:y.verify_all();m.verify_all()
  return {'scenes':len(self.rows),'windows':len(self),'full_sha':full,'all_brick_crc':crc}
 def close(self):
  for item in self.cache.values():item['response'].close();item['solid'].close()
  self.cache.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def __getstate__(self):
  state=self.__dict__.copy();state['cache']=OrderedDict();return state

class PrivateOracle:
 def __init__(self,dataset):self.dataset=dataset
 def get_true_source(self,sid):return np.fromfile(self.dataset._path(sid)/'driver.f64',dtype='<f8').copy()
 def get_candidate_sources(self,sid,wid):
  path=self.dataset._path(sid);bank=np.fromfile(path/'candidate_bank.bin',dtype='<f8').reshape(-1,3);offset=np.fromfile(path/'candidate_offsets.bin',dtype='<u8');count=np.fromfile(path/'candidate_counts.bin',dtype='<u2');width=(path/'candidate_ids.bin').stat().st_size//int(offset[-1]);ids=np.fromfile(path/'candidate_ids.bin',dtype=f'<u{width}');return bank[ids[int(offset[wid]):int(offset[wid])+int(count[wid])]].copy()
 def get_window_counts(self,sid,wid):
  w=self.dataset._open(sid)['windows'][wid];return self.dataset.scene(sid).counts(tuple(map(int,w[:3])),int(w[3]))

def collate(examples):
 packed=isinstance(examples[0]['observed_response'],PackedWindow)
 result={'scene_id':[r['scene_id'] for r in examples],'window_id':np.asarray([r['window_id'] for r in examples]),'window_origin':np.stack([r['window_origin'] for r in examples]),'window_size':examples[0]['window_size'],'structure':[r['structure'] for r in examples],'observed_response':np.stack([r['observed_response'].data if packed else r['observed_response'] for r in examples]),'response_format':'packed-LSB-X' if packed else 'bool'}
 if 'candidate_sources' in examples[0]:result['candidate_sources']=[r['candidate_sources'] for r in examples]
 return result

class ObservationLoader:
 def __init__(self,loader):self.loader=loader
 def __len__(self):return len(self.loader)
 def __iter__(self):
  import torch
  for batch in self.loader:
   for name in ('window_origin','window_id','observed_response'):batch[name]=torch.from_numpy(batch[name])
   yield batch

def make_dataloader(dataset,batch_size=4,num_workers=0,seed=0):
 import torch
 batches={64:[],128:[],256:[]};rng=np.random.default_rng(seed)
 for j,row in enumerate(dataset.rows):
  path=dataset._path(row['scene_id'])/'window_records.bin';sizes=np.fromfile(path,dtype='<u2').reshape(-1,4)[:,3]
  for i,wid in enumerate(dataset.windows[row['scene_id']]):batches[int(sizes[wid])].append(int(dataset.offsets[j])+i)
 groups=[]
 for ids in batches.values():rng.shuffle(ids);groups.extend(ids[i:i+batch_size] for i in range(0,len(ids),batch_size))
 rng.shuffle(groups);kwargs={'multiprocessing_context':'spawn','prefetch_factor':1} if num_workers else {}
 return ObservationLoader(torch.utils.data.DataLoader(dataset,batch_sampler=groups,collate_fn=collate,num_workers=num_workers,**kwargs))
