import json,shutil
from pathlib import Path
import numpy as np
import pytest
from rind3d.cli import load_config
from rind3d.geometry import Aborted,BudgetExhausted,NumericalUnresolved
from rind3d.search import CandidateBank,search_candidates,diverse_order,separated
from rind3d.pipeline import write_candidate_files,validate_records,generate_dataset
from rind3d.legacy.windows import Window as LegacyWindow
from rind3d.windows import Window
from test_gates import AnalyticScreenOracle


def case(**changes):
    cfg=load_config('configs/pilot.yaml')
    cfg.update(resolution=128,global_proposals=0,window_search_seconds=30,verification_seconds=30)
    cfg.update(changes)
    w=Window((48,16,16),64,0,((48,16,16),(79,16,16)))
    return cfg,np.array([.1,.1,.1]),w


@pytest.mark.parametrize('kind',['point','time'])
def test_search_point_budget_keeps_independently_verified_truth(kind):
    cfg,truth,w=case(**({'window_search_points':0} if kind=='point' else {'window_search_seconds':1e-9}))
    points,proofs,r=search_candidates(AnalyticScreenOracle(),truth,w,np.empty((0,3)),cfg,0,0)
    assert r['status']=='COMPLETE' and r['actual_candidates']==1
    assert r['search_termination_reason']=='search_'+kind+'_budget'
    assert r['all_grid_exact'] and r['independent_validation_passed'] and len(proofs)==1
    assert np.array_equal(points[0],truth)


def test_search_stops_at_64_with_full_proofs_and_no_near_duplicates():
    cfg,truth,w=case()
    pool=np.vstack([np.random.default_rng(2).uniform(.01,.99,(1000,3)),np.tile(truth,(50,1)),truth+[.001,0,0]])
    points,proofs,r=search_candidates(AnalyticScreenOracle(),truth,w,pool,cfg,0,0)
    assert len(points)==64 and r['status']=='COMPLETE' and r['search_termination_reason']=='max_candidates'
    assert r['unique_proposals']==1002 and r['reached_max_candidates']
    assert len(np.unique(points,axis=0))==64
    assert all(e['status']==e['independent_status']=='grid_exact' and e['covered_voxels']==w.size**3 for e in proofs)
    for i,p in enumerate(points):assert all(separated(p,q,32/1024) for q in points[i+1:])


def test_nearby_proposals_are_skipped_before_screening():
    cfg,truth,w=case()
    pool=truth+np.arange(10)[:,None]*.00001
    points,_,r=search_candidates(AnalyticScreenOracle(),truth,w,pool,cfg,0,0)
    assert len(points)==1 and r['skipped_near_accepted']==10 and r['full_verifications']==0


def test_mandatory_verification_budget_cannot_be_ignored():
    cfg,truth,w=case(verification_points=0)
    points,_,r=search_candidates(AnalyticScreenOracle(),truth,w,[],cfg,0,0)
    assert r['status']=='ABORTED' and len(points)==0 and not r['all_grid_exact']


def test_unfinished_independent_rechecks_are_discarded(monkeypatch):
    import rind3d.search as module
    cfg,truth,w=case(max_candidates=8)
    seen={};original=module.verify_grid_exact;checked=0
    def verify(oracle,s,p,*args,**kwargs):
        nonlocal checked
        key=np.asarray(p).tobytes();seen[key]=seen.get(key,0)+1
        if not np.array_equal(p,truth) and seen[key]==2:
            checked+=1
            if checked==3:raise BudgetExhausted('point')
        return original(oracle,s,p,*args,**kwargs)
    monkeypatch.setattr(module,'verify_grid_exact',verify)
    pool=np.random.default_rng(3).uniform(.01,.99,(100,3))
    points,proofs,r=search_candidates(AnalyticScreenOracle(),truth,w,pool,cfg,0,0)
    assert r['status']=='COMPLETE' and len(points)==3 and len(proofs)==3
    assert r['first_pass_candidates']==8 and r['discarded_unrechecked']==5
    assert r['recheck_termination_reason']=='verification_point_budget'


@pytest.mark.parametrize('exception',[Aborted('Canonical oracle aborted'),NumericalUnresolved('Unknown geometry')])
def test_canonical_failures_do_not_become_search_exhaustion(exception):
    class Failure(AnalyticScreenOracle):
        calls=0
        def contains(self,points,budget=None):
            self.calls+=1
            if self.calls>2:raise exception
            return super().contains(points,budget)
    cfg,truth,w=case()
    _,_,r=search_candidates(Failure(),truth,w,np.empty((0,3)),cfg,0,0)
    assert r['status']==('ABORTED' if isinstance(exception,Aborted) else 'UNRESOLVED')
    assert r['actual_candidates']==1  # safe diagnostic retained; failure still blocks READY


def test_diverse_global_order_is_deterministic_and_deduplicates():
    pool=np.random.default_rng(1).uniform(.01,.99,(4096,3));pool=np.vstack([pool,pool[:100]])
    a=diverse_order(pool,1,2,3);b=diverse_order(pool,1,2,3)
    assert np.array_equal(a,b) and len(a)==4096
    assert len({tuple((p*4).astype(int)) for p in a[:64]})==64


def make_records(root,number=4096):
    bank=CandidateBank()
    for p in ([.1,.1,.1],[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]):bank.add(p)
    from rind3d.window_contract import PRODUCTION_SIDE_COUNTS
    windows=[Window((400,(i%64)*8,(i//64)*8),side)
             for side,count in PRODUCTION_SIDE_COUNTS.items() for i in range(count)][:number]
    refs=[list(range(i%4+1)) for i in range(number)]
    dtype,counts=write_candidate_files(root,windows,bank,refs,[{}]*number)
    meta={'resolution':1024,'candidate_id_dtype':dtype,'max_candidates':64,'true_source_included':True,
          'candidate_storage':'pool-offsets-v2','contract_version':2,'min_source_separation':32/1024}
    return meta,counts


def test_4096_variable_counts_pass_ready_quantity_gate_without_padding(tmp_path):
    meta,counts=make_records(tmp_path)
    _,actual,bank,rows=validate_records(tmp_path,meta)
    assert counts.min()==1 and counts.max()==4 and np.array_equal(counts,actual)
    assert (tmp_path/'candidate_ids.bin').stat().st_size==int(counts.sum())*2
    assert (tmp_path/'candidate_offsets.bin').stat().st_size==4097*8
    assert [len(r) for r in rows[:4]]==[1,2,3,4]


def test_production_exact_size_quota_is_checked_without_generating_fields(tmp_path):
    meta,_=make_records(tmp_path)
    p=tmp_path/'window_records.bin';records=np.fromfile(p,dtype='<u2').reshape(-1,4)
    records[0,0]+=1  # keep the record unique so this specifically exercises quota validation
    records[0,3]=128  # allowed power, but the exact 1365/1365/1366 quotas are violated
    records.tofile(p)
    with pytest.raises(ValueError,match='quotas'):validate_records(tmp_path,meta)


@pytest.mark.parametrize('corruption',['offset_count','offset_order','offset_length','id','zero_count','too_many','duplicate'])
def test_variable_storage_corruption_fails(tmp_path,corruption):
    meta,counts=make_records(tmp_path)
    if corruption.startswith('offset'):
        p=tmp_path/'candidate_offsets.bin';v=np.fromfile(p,dtype='<u8')
        if corruption=='offset_count':v[1]+=1
        elif corruption=='offset_order':v[3]=0
        else:v=v[:-1]
        v.tofile(p)
    elif corruption=='id':
        p=tmp_path/'candidate_ids.bin';v=np.fromfile(p,dtype='<u2');v[0]=4;v.tofile(p)
    elif corruption in ('zero_count','too_many'):
        p=tmp_path/'candidate_counts.bin';v=np.fromfile(p,dtype='<u2');v[0]=0 if corruption=='zero_count' else 65;v.tofile(p)
    else:
        p=tmp_path/'candidate_ids.bin';v=np.fromfile(p,dtype='<u2');v[2]=v[1];v.tofile(p)
    with pytest.raises(ValueError):validate_records(tmp_path,meta)


def test_true_source_cannot_be_disabled():
    cfg,truth,w=case(include_true_source=False)
    with pytest.raises(ValueError,match='true source'):search_candidates(AnalyticScreenOracle(),truth,w,[],cfg,0,0)


def test_function_geometry_still_cannot_publish(tmp_path,monkeypatch):
    import rind3d.pipeline as module
    cfg=load_config('configs/pilot.yaml');cfg['families']=[['random_field']]
    monkeypatch.setattr(module,'gate_pilot',lambda *args:None)  # isolate the second, geometry-specific gate
    pilot=tmp_path/'pilot.json';pilot.write_text('{}')
    with pytest.raises(NumericalUnresolved,match='READY capable'):generate_dataset(cfg,tmp_path/'blocked',1,pilot)
    meta=json.loads((tmp_path/'blocked/.staging/scene_000000/verification.json').read_text())
    assert meta['status']=='UNRESOLVED' and not (tmp_path/'blocked/scenes/scene_000000').exists()
