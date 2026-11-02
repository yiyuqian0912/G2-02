# Explicit legacy four-side diagnostic regression tests.
"""Distribution and correctness tests for the isolated C origin selector."""
import hashlib,json,sys
from collections import Counter
from itertools import product
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import chisquare
from rind3d.geometry import Oracle
from rind3d.benchmark import fixture_box
from rind3d.legacy.config import load_config
from rind3d.legacy.windows import Window,excludes,sample_windows
from rind3d.pilot import source_fingerprint

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from full_origin_selector import sample_integer_volume,pair_origin_candidates,select_full_volume
from experimental_boundary_guided import OriginBox,legal_origin_region,guided_sample
from boundary_experiment_support import Meter,final_validators
from origin_audit_meter import AuditMeter
from origin_distribution_features import diagnostic_coordinates,witness_features
from test_boundary_guided import AnalyticOracle

B_BEFORE=hashlib.sha256(Path('scripts/experimental_boundary_guided.py').read_bytes()).hexdigest()
SCIENCE_BEFORE=source_fingerprint();A_BEFORE=sample_windows


def settings():return json.loads(Path('configs/experimental-origin-distribution.json').read_text())
def b_settings():return json.loads(Path('configs/experimental-boundary-guided.json').read_text())


def test_volume_weights_and_empirical_uniformity_on_brute_force_small_set():
    boxes=[OriginBox((0,0,0),(0,0,0)),OriginBox((2,0,0),(3,1,1))]
    legal=[p for b in boxes for p in product(*(range(a,c+1) for a,c in zip(b.lower,b.upper)))]
    rng=np.random.default_rng(20261030)
    draws=Counter(sample_integer_volume(boxes,1,rng)[0] for _ in range(12000))
    assert set(draws)==set(legal)
    assert chisquare([draws[p] for p in sorted(legal)]).pvalue>.001
    assert abs(draws[(0,0,0)]/12000-1/9)<.015


def test_full_legal_origin_enumeration_and_empirical_draws_agree_with_source_exclusion():
    n=36;source=np.array([.5,.5,.5]);p=np.array([.25,.25,.25]);q=np.array([.5,.5,.5])
    # This pair forces the source inside every window: the exact legal set is empty.
    assert legal_origin_region(p,q,source,32,n)==()
    source=np.array([.99,.99,.99]);boxes=legal_origin_region(p,q,source,32,n)
    brute={o for o in product(range(5),repeat=3) if all(o[i]<=p[i]*n<o[i]+32 and o[i]<=q[i]*n<o[i]+32 for i in range(3)) and not all(o[i]<=source[i]*n<o[i]+32 for i in range(3))}
    observed=sample_integer_volume(boxes,len(brute),np.random.default_rng(1))
    assert set(observed)==brute and len(observed)==len(set(observed))
    counts=Counter(sample_integer_volume(boxes,1,np.random.default_rng(i))[0] for i in range(6000))
    assert chisquare([counts[p] for p in sorted(brute)]).pvalue>.001


def test_box_order_invariance_no_duplicates_and_large_volume_without_enumeration():
    boxes=[OriginBox((0,0,0),(1,1,1)),OriginBox((3,2,1),(4,3,2))]
    a=sample_integer_volume(boxes,12,np.random.default_rng(9));b=sample_integer_volume(boxes[::-1],12,np.random.default_rng(9))
    assert a==b and len(a)==len(set(a))==12
    big=OriginBox((0,0,0),(1023,1023,1023));values=sample_integer_volume([big],32,np.random.default_rng(9))
    assert len(values)==32 and all(big.contains(p) for p in values)
    with pytest.raises(ValueError,match='Overlapping'):sample_integer_volume([boxes[0],boxes[0]],2,np.random.default_rng(1))


@pytest.mark.parametrize('side',[31,33,96,110,127,192,512,32.,True])
def test_c_rejects_illegal_sides(side):
    with pytest.raises(ValueError):pair_origin_candidates({'id':'x','positive':[10]*3,'negative':[11]*3},[.9]*3,side,1024,settings(),1,0,302)


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    cfg=load_config('configs/pilot-extended.yaml');cfg=dict(cfg,global_proposals=0)
    truth=np.array([.01,.02,.03]);root=tmp_path_factory.mktemp('matched-witness')
    m=Meter(AnalyticOracle());b,report=guided_sample(truth,cfg,6,m,b_settings(),stream=302,output=root)
    pairs=json.loads((root/'pairs.json').read_text());m.close()
    c,rows,audit=select_full_volume(b,pairs,truth,cfg,b_settings(),settings(),6,302)
    return cfg,truth,b,pairs,c,rows,audit


def test_c_exact_pair_matching_determinism_and_diversity(cohort):
    cfg,truth,b,pairs,c,rows,audit=cohort
    again,rows2,_=select_full_volume(b,pairs,truth,cfg,b_settings(),settings(),6,302)
    assert c==again and rows==rows2 and len(c)==32 and len({w.record() for w in c})==32
    for side in (32,64,128,256):
        old=[w for w in b if w.size==side];new=[w for w in c if w.size==side]
        assert {w.witnesses for w in old}=={w.witnesses for w in new}
        assert audit[str(side)]['unique_pairs']==audit[str(side)]['spatial_buckets']==8
        points=np.asarray([w.origin for w in new]);d=((points[:,None]-points)**2).sum(axis=-1).astype(float);np.fill_diagonal(d,np.inf)
        assert d.min()>=max(64,side/2)**2
        for w in new:
            assert excludes(w,truth,1024) and min(w.origin)>=0 and max(w.origin)+side<=1024
            assert all(all(o<=p<o+side for p,o in zip(point,w.origin)) for point in w.witnesses)


def test_every_c_window_passes_original_independent_candidate_and_record_checks(cohort,tmp_path):
    cfg,truth,b,pairs,c,rows,audit=cohort
    original=Meter(AnalyticOracle());independent=Meter(AnalyticOracle())
    result=final_validators(c,truth,truth[None],cfg,6,original,independent,tmp_path)
    assert result['pilot_quota_complete'] and result['all_retained_candidates_valid']
    assert not result['production_ready'] and result['candidate_counts']=={'1':32}
    original.close();independent.close()


def test_held_out_stratified_composition_is_not_forced_to_contain_witnesses(cohort):
    w=cohort[4][0];coordinates=diagnostic_coordinates(w,[4,4,2],[1,2,3])
    assert coordinates.shape==(32,3) and np.all(coordinates>=w.origin) and np.all(coordinates<np.asarray(w.origin)+w.size)
    offset=coordinates-np.asarray(w.origin);width=np.array([w.size//4,w.size//4,w.size//2])
    assert len({tuple(x) for x in offset//width})==32
    assert np.array_equal(coordinates,diagnostic_coordinates(w,[4,4,2],[1,2,3]))
    f=witness_features(w,1024)
    assert 0<=f['midpoint_face_min_relative']<=.5


def test_native_filter_work_is_counted_without_changing_physics():
    oracle=Oracle(fixture_box());meter=AuditMeter(oracle)
    p=np.array([[.1,.1,.1],[.5,.5,.5],[.9,.9,.9]])
    with meter.timed('test'):
        mask=meter.contains(p);response=meter.response([.1,.1,.1],p);meter.certificate([.1,.1,.1],[.8]*3,[.9]*3)
    stats=meter.report()['calls']['test']
    assert stats['membership_evaluations']==3 and stats['response_rays']==3 and stats['certificate_calls']==1
    assert stats['oracle_calls']==7
    meter.close();assert np.array_equal(mask,oracle.contains(p)) and np.array_equal(response,oracle.response([.1]*3,p))


def test_a_b_and_scientific_implementations_are_unchanged(cohort):
    assert hashlib.sha256(Path('scripts/experimental_boundary_guided.py').read_bytes()).hexdigest()==B_BEFORE
    assert source_fingerprint()==SCIENCE_BEFORE and sample_windows is A_BEFORE
