import copy,json
import numpy as np
import pytest
from rind3d.cli import load_config
from rind3d.pack import atomic_json
from rind3d.pilot import config_hash
from rind3d.pipeline import gate_pilot,generate_dataset,write_candidate_files,validate_records,verify_scene
from rind3d.search import CandidateBank,search_candidates
from rind3d.geometry import NumericalUnresolved
from rind3d.legacy.windows import Window as LegacyWindow
from rind3d.windows import Window


def test_failed_pilot_cannot_start_generation(tmp_path):
    cfg=load_config('configs/pilot.yaml');p=tmp_path/'pilot.json'
    atomic_json(p,{'format':'rind3d-pilot-v2','config_sha256':config_hash(cfg),'publication_feasible':False})
    output=tmp_path/'mass'
    with pytest.raises(ValueError,match='CLOSED'):generate_dataset(cfg,output,1000,p)
    assert not output.exists()
    data=json.loads(p.read_text());data['publication_feasible']=True;atomic_json(p,data)
    with pytest.raises(ValueError,match='measured sample'):gate_pilot(cfg,p)
    wrong=copy.deepcopy(cfg);wrong['seed']+=1
    with pytest.raises(ValueError,match='matching'):gate_pilot(wrong,p)


class AnalyticScreenOracle:
    """Transparent synthetic evaluator: every free source has the same step map."""
    def __init__(self,unresolved=False):self.evaluated_sources=set();self.unresolved=unresolved
    def contains(self,points,budget=None):
        if self.unresolved:raise NumericalUnresolved('Deliberately uncertain query')
        if budget:budget.check(len(points))
        return np.zeros(len(points),dtype=bool)
    def response(self,s,p,budget=None):
        s=np.asarray(s).reshape(-1,3);p=np.asarray(p).reshape(-1,3)
        n=max(len(s),len(p))
        if budget:budget.check(n)
        self.evaluated_sources.update(v.astype('<f8').tobytes() for v in s)
        return np.broadcast_to(p[:,0]<.5,(n,)).copy()
    def certificate(self,s,lo,hi,budget=None):
        if budget:budget.check()
        return (-1,0)


def test_mathematically_infeasible_separation_is_not_filled_with_duplicates():
    cfg=load_config('configs/pilot.yaml')
    # No two points in [0,1]^3 can be distance 2 apart. This is a failure test,
    # never a changed production separation or a claim about real inverse sets.
    cfg.update(resolution=128,min_source_separation=2.,global_proposals=0,window_search_seconds=10)
    truth=np.array([.1,.1,.1]);w=Window((48,16,16),64,0,((48,16,16),(79,16,16)))
    pool=np.random.default_rng(1).uniform(.01,.99,(256,3))
    positions,evidence,report=search_candidates(AnalyticScreenOracle(),truth,w,pool,cfg,0,0)
    assert report['status']=='COMPLETE' and report['actual_candidates']==1
    assert positions.shape==(1,3) and report['independent_validation_passed']
    assert report['all_grid_exact'] and report['sample_only']==0
    _,_,unresolved=search_candidates(AnalyticScreenOracle(True),truth,w,pool,cfg,0,0)
    assert unresolved['status']=='UNRESOLVED' and not unresolved['independent_validation_passed'] and unresolved['actual_candidates']==0


def test_bank_upgrades_index_width_without_discarding_coordinates():
    bank=CandidateBank()
    for i in range(65535):bank.add([.1+i/1_000_000,.2,.3])
    assert bank.index_dtype.str=='<u2'
    i=bank.add([.9,.8,.7]);assert i==65535 and bank.index_dtype.str=='<u4'
    assert np.array_equal(bank.array[i],[.9,.8,.7])


def test_candidate_domain_switch_is_explicit():
    cfg=load_config('configs/pilot.yaml')
    cfg.update(resolution=128,global_proposals=0,max_candidates=2,window_search_seconds=10)
    truth=np.array([.1,.1,.1]);inside=np.array([[.5,.25,.25]])
    w=Window((48,16,16),64,0,((48,16,16),(79,16,16)))
    points,_,report=search_candidates(AnalyticScreenOracle(),truth,w,inside,cfg,0,0)
    assert report['status']=='COMPLETE' and len(points)==2
    cfg['candidate_domain']='world_outside_window'
    points,_,report=search_candidates(AnalyticScreenOracle(),truth,w,inside,cfg,0,0)
    assert report['status']=='COMPLETE' and len(points)==1


def test_partial_counts_and_publication_are_fail_closed(tmp_path):
    bank=CandidateBank();bank.add([.1,.2,.3]);w=Window((64,0,0),64)
    dtype,counts=write_candidate_files(tmp_path,[w],bank,[[0]],[{'all_grid_exact':True,'independent_validation_passed':False}])
    meta={'resolution':128,'candidate_id_dtype':dtype,'min_source_separation':.03125}
    records,actual,_,ids=validate_records(tmp_path,meta,strict=False)
    assert counts.tolist()==actual.tolist()==[1] and ids[0].tolist()==[0]
    assert (tmp_path/'candidate_ids.bin').stat().st_size==2
    assert np.fromfile(tmp_path/'candidate_offsets.bin',dtype='<u8').tolist()==[0,1]
    with pytest.raises(ValueError,match='READY counts'):validate_records(tmp_path,meta)
    for state in ('ABORTED','UNRESOLVED','INCOMPLETE'):
        atomic_json(tmp_path/'verification.json',{'status':state})
        with pytest.raises(ValueError,match='not READY'):verify_scene(tmp_path)
    for status in ('READY','READY_SHORTFALL'):
        m=dict(meta,status=status,allow_shortfall=True,equivalence_status='grid_exact',continuous_certified=False,hashes={})
        atomic_json(tmp_path/'verification.json',m)
        with pytest.raises(ValueError,match='Missing canonical'):verify_scene(tmp_path)
