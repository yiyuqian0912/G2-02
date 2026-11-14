import hashlib,json,pickle
from pathlib import Path
import numpy as np
import pytest
from rind3d.campaign_contract import derived_seed
from rind3d.campaign import event,check_events,recover
from rind3d_reader.catalog import Catalog
from rind3d_reader.durable import write_json,seal,digest,read_sealed
from tests.test_release_catalog import row,make_scene

def test_seed_domains_are_independent_of_targets_and_scheduling():
 master='29'*32
 expected={n:{tag:derived_seed(master,n,tag) for tag in ('geometry','source','window')} for n in range(30)}
 for target in (1000,2000,5000,10000):
  for n in reversed(range(30)):assert {tag:derived_seed(master,n,tag) for tag in expected[n]}==expected[n]
 assert all(len(set(v.values()))==3 for v in expected.values())
 assert len({v['geometry'] for v in expected.values()})==30

def test_log_torn_tail_recovery_and_interior_corruption(tmp_path):
 event(tmp_path,'A',attempt=0);event(tmp_path,'B',attempt=1)
 with (tmp_path/'events.private.jsonl').open('ab') as f:f.write(b'{"partial"')
 assert check_events(tmp_path)==2
 assert list(tmp_path.glob('torn-log-*.json'))
 p=tmp_path/'events.private.jsonl';raw=p.read_text().replace('"A"','"C"');p.write_text(raw)
 with pytest.raises(ValueError,match='corruption'):check_events(tmp_path)

def test_completed_scene_awaiting_registration_is_recovered_idempotently(tmp_path):
 (tmp_path/'attempts').mkdir();r=row(0);make_scene(tmp_path,r);p=tmp_path/'scenes'/r['scene_id']/'scene.seal.json';s=read_sealed(p);s['catalog_record']={k:v for k,v in r.items() if k!='seal_sha256'};write_json(p,seal(s));write_json(tmp_path/'attempts/000000000000.json',seal({'attempt_id':0,'status':'VERIFIED'}))
 with Catalog(tmp_path,create=True) as catalog:
  assert recover(tmp_path,catalog)==[r['scene_id']]
  assert recover(tmp_path,catalog)==[] and len(catalog)==1
  assert read_sealed(tmp_path/'attempts/000000000000.json')['status']=='READY'

def test_legacy_dataset_pickle_construction_has_no_required_root():
 from rind3d.data import RINDDataset
 d=object.__new__(RINDDataset);d.cache={};d.root=Path('/unopened')
 assert pickle.loads(pickle.dumps(d)).root==d.root


def test_source_resampling_keeps_geometry_and_does_not_resample_timeouts():
 from rind3d.campaign import next_geometry_trial
 states={0:{'attempt_id':0,'geometry_ordinal':0,'source_trial':0,'status':'REJECTED_SOURCE'}}
 assert next_geometry_trial(states,64)==(0,1)
 states[1]={'attempt_id':1,'geometry_ordinal':0,'source_trial':1,'status':'PREPARED'}
 assert next_geometry_trial(states,64)==(1,0)
 states[1]['status']='REJECTED_SOURCE'
 with pytest.raises(RuntimeError,match='exhausted'):next_geometry_trial(states,2)
 # Incomplete attempts are retried by the dispatcher; they are never certified rejections.
 assert states[0]['geometry_ordinal']==states[1]['geometry_ordinal']

def test_unfinished_session_metrics_are_explicit_lower_bounds(tmp_path):
 from rind3d.campaign import session_totals
 (tmp_path/'sessions').mkdir()
 write_json(tmp_path/'sessions/a.start.json',seal({'session_id':'a'}))
 write_json(tmp_path/'sessions/a.progress.json',seal({'active_wall_seconds':10,'total_cpu_seconds_checkpoint':22}))
 r=session_totals(tmp_path)
 assert r['unfinalized_sessions']==1 and r['total_cpu_seconds']==22
 assert 'lower_bound' in r['runtime_accounting']
 write_json(tmp_path/'sessions/a.complete.json',seal({'active_wall_seconds':15,'total_cpu_seconds':30}))
 assert session_totals(tmp_path)['total_cpu_seconds']==30
