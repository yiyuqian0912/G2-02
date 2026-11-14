import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from rind3d_reader.catalog import Catalog,FORMAT
from rind3d_reader.durable import write_json,seal

def test_config_integer_keys_survive_durable_roundtrip(tmp_path):
 from rind3d_reader.durable import read_sealed,digest
 value={'sides':{64:85,128:85,256:86}};write_json(tmp_path/'config',seal(value));loaded=read_sealed(tmp_path/'config')
 assert digest(loaded)==digest(value) and loaded['sides']['64']==85

def row(i,status='READY_LITE'):
 return {'format':FORMAT,'status':status,'attempt_id':i,'scene_id':hashlib.sha256(f'scene{i}'.encode()).hexdigest(),'geometry_id':hashlib.sha256(f'geometry{i}'.encode()).hexdigest(),'window_count':256,'split':'train','families':['rotated_boxes'],'seal_sha256':'0'*64}

def make_scene(root,r):
 p=root/'scenes'/r['scene_id'];p.mkdir(parents=True);(p/'a').write_bytes(b'checked')
 info={'a':{'bytes':7,'sha256':hashlib.sha256(b'checked').hexdigest()}}
 write_json(p/'scene.seal.json',seal({k:r[k] for k in ('scene_id','geometry_id','window_count','status','split')}|{'files':info}))
 r['seal_sha256']=hashlib.sha256((p/'scene.seal.json').read_bytes()).hexdigest();return p

def test_atomic_catalog_requires_complete_scene_and_exclusive_owner(tmp_path):
 r=row(0)
 with Catalog(tmp_path,create=True) as c:
  with pytest.raises(ValueError,match='seal'):c.commit(r)
  with pytest.raises(RuntimeError,match='owner'):Catalog(tmp_path,create=True)
  p=make_scene(tmp_path,r);assert c.commit(r)['committed'];assert c.commit(r)['idempotent']
  bad=dict(r,split='val')
  with pytest.raises(ValueError,match='Conflicting'):c.commit(bad)
  assert c.verify_scene(r,full=True)['status']=='READY_LITE'
  (p/'a').write_bytes(b'changed')
  with pytest.raises(ValueError,match='checksum'):c.verify_scene(r,full=True)
 with Catalog(tmp_path) as c:
  assert len(c)==1 and c.get(r['scene_id'])==r
  with pytest.raises(RuntimeError,match='exclusive'):c.commit(row(1))

def test_catalog_tampering_conflicts_and_no_full_rewrite(tmp_path):
 with Catalog(tmp_path,create=True,synthetic=True) as c:
  c.commit(row(0,'SYNTHETIC'));old=next(c.directory.glob('*.json'));before=old.read_bytes();stamp=old.stat().st_mtime_ns
  c.commit(row(1,'SYNTHETIC'));assert old.read_bytes()==before and old.stat().st_mtime_ns==stamp
  with pytest.raises(ValueError,match='geometry'):c.commit(dict(row(2,'SYNTHETIC'),geometry_id=row(0)['geometry_id']))
 old.write_bytes(before+b' ')
 with pytest.raises(ValueError,match='checksum'):Catalog(tmp_path,synthetic=True)
 with pytest.raises(ValueError):Catalog(tmp_path)

def test_reader_only_counts_match_native_unaligned_and_rare(tmp_path):
 from tests.test_exact_counts import write
 from rind3d.pack import RINDPack as Original
 from rind3d.counts import CountIndex as Native
 from rind3d_reader.pack import RINDPack
 from rind3d_reader.counts import CountIndex
 rng=np.random.default_rng(702);m=rng.random((128,)*3)<.2;y=(rng.random((128,)*3)<.6)&~m
 write(tmp_path/'y',y);write(tmp_path/'m',m)
 with Original(tmp_path/'y') as a,Original(tmp_path/'m') as b:Native.build(a,b).save(tmp_path/'counts')
 with RINDPack(tmp_path/'y') as a,RINDPack(tmp_path/'m') as b:
  c=CountIndex.load(tmp_path/'counts',a,b)
  for side,origin in ((64,(1,7,13)),(128,(0,0,0)),(1,(127,127,127)),(32,(16,16,16)),(65,(63,63,63))):
   x,v,z=origin;sl=(slice(z,z+side),slice(v,v+side),slice(x,x+side));r=c.query(origin,side)
   assert (r.lit,r.shadow,r.solid)==(np.count_nonzero(y[sl]),np.count_nonzero(~y[sl]&~m[sl]),np.count_nonzero(m[sl]))
