# Explicit legacy four-side diagnostic regression tests.
"""Mathematical origin enumeration and scientific isolation of the experiment."""
import hashlib,importlib.util,json,sys
from itertools import product
from pathlib import Path
import numpy as np
import pytest
from rind3d.legacy.config import load_config
from rind3d.geometry import Budget,NumericalUnresolved,BudgetExhausted
from rind3d.legacy.windows import Window,excludes,sample_windows
from rind3d.pilot import source_fingerprint

SCIENTIFIC_BEFORE=source_fingerprint()
UNIFORM_BEFORE=sample_windows
CONFIGS_BEFORE={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('configs').glob('pilot*.yaml')}

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from experimental_boundary_guided import (legal_origin_region,OriginBox,ExactPointCache,make_pairs,
                                         select_diverse,guided_sample,pair_region)
from boundary_experiment_support import Meter,independently_check_window,final_validators
from analyze_boundary_ab import union_origin_count,common_center_regions,nearby_nested_chain


def enumerated(boxes):
    values=[o for box in boxes for o in product(*(range(a,b+1) for a,b in zip(box.lower,box.upper)))]
    assert len(values)==len(set(values))
    assert len(values)==sum(box.count for box in boxes)
    return set(values)


@pytest.mark.parametrize('resolution',[40,64])
@pytest.mark.parametrize('p,q,s',[
    ([.25,.5,.5],[.5,.5,.5],[.5,.5,.5]),
    ([0.,0.,0.],[.25,.25,.25],[.875,.875,.875]),
    ([.5,.5,.5],[.75,.75,.75],[.125,.125,.125]),
    ([1.,.5,.5],[.75,.5,.5],[.125,.125,.125]),
    ([.5,.5,.5],[.5,.5,.5],[.5,.5,.5]),
    ([.25,.25,.25],[.5,.5,.5],[.75,.25,.25]),
])
def test_all_analytic_legal_origins_equal_brute_force_small_worlds(resolution,p,q,s):
    # Deliberately dyadic inputs: integer window faces are exact, including
    # lower equality (included) and upper equality (excluded).
    region=legal_origin_region(p,q,s,32,resolution)
    actual=enumerated(region)
    origins=np.asarray(list(product(range(resolution-31),repeat=3)))
    p=np.asarray(p)*resolution;q=np.asarray(q)*resolution;source=np.asarray(s)*resolution
    inside_p=np.all((p>=origins)&(p<origins+32),axis=1)
    inside_q=np.all((q>=origins)&(q<origins+32),axis=1)
    inside_s=np.all((source>=origins)&(source<origins+32),axis=1)
    expected=set(map(tuple,origins[inside_p&inside_q&~inside_s]))
    assert actual==expected


def test_half_open_lower_face_allowed_upper_face_excluded():
    r=legal_origin_region([.5,.5,.5],[.75,.5,.5],[.1,.1,.1],32,64)
    assert any(b.contains((32,32,32)) for b in r)
    assert not any(b.contains((16,32,32)) for b in r)  # .75*64 == origin_x+32


@pytest.mark.parametrize('side',[31,33,96,110,127,192,512,32.,True])
def test_illegal_window_side_rejected_by_constructive_path(side):
    with pytest.raises(ValueError):legal_origin_region([.3]*3,[.4]*3,[.1]*3,side,1024)


class AnalyticOracle:
    """Synthetic exact step-map protocol. Never used for physical measurements."""
    def __init__(self,solid=False,flip=False):
        self.reference=object();self.evaluated_sources=set();self.solid=solid;self.flip=flip
        self.capability={'ready_capable':False};self.queries=0
    def validate_source(self,s,budget=None):
        if self.contains(np.asarray(s).reshape(1,3),budget=budget)[0]:raise ValueError('Source solid')
    def contains(self,p,budget=None):
        if budget:budget.check(len(p))
        self.queries+=len(p)
        return np.full(len(p),self.solid,bool)
    def response(self,s,p,budget=None):
        if budget:budget.check(max(len(np.asarray(s).reshape(-1,3)),len(p)))
        self.queries+=len(p);self.evaluated_sources.add(np.asarray(s,dtype='<f8').tobytes())
        result=np.asarray(p)[:,0]<.5
        return ~result if self.flip else result
    def certificate(self,s,lo,hi,budget=None):
        if budget:budget.check()
        return (-1,0)


def settings():return json.loads(Path('configs/experimental-boundary-guided.json').read_text())


def test_solid_points_never_enter_opposite_response_pairs():
    oracle=AnalyticOracle(solid=True)
    # A solid receiver with response=0 is legal, but cannot be a free witness.
    c=ExactPointCache([.1]*3,1024,oracle,Budget(10),100)
    c.query(np.array([800,20,30]),'test')
    assert c.points[(800,20,30)]['label']=='solid'
    assert make_pairs(c,settings())==[]
    with pytest.raises(NumericalUnresolved):c.query(np.array([20,20,30]),'test')


def test_cache_is_exact_and_does_not_label_neighbors_or_repeat_queries():
    o=AnalyticOracle();c=ExactPointCache([.1]*3,1024,o,Budget(10),100)
    a=c.query(np.array([511,20,30]),'test')
    assert a['label']=='free_lit' and o.queries==2
    assert c.query(np.array([511,20,30]),'test')==a and o.queries==2
    assert (512,20,30) not in c.points
    assert c.query(np.array([512,20,30]),'test')['label']=='free_shadow'
    assert o.queries==4
    with pytest.raises(ValueError):c.query(np.array([.5,.5,.5]),'test')


@pytest.fixture(scope='module')
def synthetic_guided(tmp_path_factory):
    cfg=load_config('configs/pilot-extended.yaml');cfg=dict(cfg,global_proposals=0)
    source=np.array([.01,.02,.03]);outputs=[]
    for repeat in range(2):
        root=tmp_path_factory.mktemp(f'guided-{repeat}')
        meter=Meter(AnalyticOracle())
        windows,report=guided_sample(source,cfg,6,meter,settings(),stream=302,output=root)
        meter.close();outputs.append((windows,report,root))
    return source,cfg,outputs


def test_seeded_diversity_is_deterministic_and_windows_are_not_tiny_shifts(synthetic_guided):
    source,cfg,outputs=synthetic_guided
    (windows,report,root),(again,_,_)=outputs
    assert windows==again and len(windows)==32 and report['complete_quota']
    assert len({w.record() for w in windows})==32
    rows=[json.loads(line) for line in (root/'proposals.jsonl').read_text().splitlines()]
    for side in (32,64,128,256):
        part=[w for w in windows if w.size==side];points=np.array([w.origin for w in part])
        selected=[r for r in rows if r['side']==side]
        assert len(part)==8 and len({r['pair_id'] for r in selected})==8
        assert len({tuple(r['bucket']) for r in selected})==8
        d=np.linalg.norm(points[:,None]-points,axis=-1);np.fill_diagonal(d,np.inf)
        assert d.min()>=max(64,side/2)
        for w in part:
            assert excludes(w,source,1024)
            assert min(w.origin)>=0 and max(w.origin)+side<=1024


def test_exact_witness_labels_are_independently_requeried(synthetic_guided):
    source,cfg,outputs=synthetic_guided;w=outputs[0][0][0]
    independent=AnalyticOracle()
    report=independently_check_window(w,source,cfg,independent)
    assert report['status']=='PASS' and independent.queries==5  # source, two masks, two rays
    with pytest.raises(NumericalUnresolved):independently_check_window(w,source,cfg,AnalyticOracle(flip=True))
    with pytest.raises((ValueError,NumericalUnresolved)):independently_check_window(w,source,cfg,AnalyticOracle(solid=True))
    invalid=Window(w.origin,w.size,witnesses=(w.witnesses[0],tuple(o+w.size for o in w.origin)))
    with pytest.raises(ValueError,match='outside half-open'):independently_check_window(invalid,source,cfg,AnalyticOracle())


def test_guided_windows_pass_unchanged_candidate_storage_and_publication_gates(synthetic_guided,tmp_path):
    source,cfg,outputs=synthetic_guided
    first=Meter(AnalyticOracle());independent=Meter(AnalyticOracle())
    result=final_validators(outputs[0][0],source,source[None],cfg,6,first,independent,tmp_path)
    assert result['pilot_quota_complete'] and result['all_retained_candidates_valid']
    assert result['candidate_counts']=={'1':32}
    assert not result['production_ready'] and result['publication_restriction_preserved']
    checks=json.loads((tmp_path/'final-checks.json').read_text())
    assert all(r['candidate_report']['all_grid_exact'] and r['candidate_report']['independent_validation_passed'] for r in checks)
    assert all(r['evidence'][0]['covered_voxels']==r['side']**3 for r in checks)
    first.close();independent.close()


def test_production_sampler_and_scientific_files_are_byte_for_byte_unchanged():
    assert source_fingerprint()==SCIENTIFIC_BEFORE
    # Sampler is still its original module/function, not an experimental alias.
    assert sample_windows.__module__=='rind3d.legacy.windows'
    assert sample_windows is UNIFORM_BEFORE
    for file,digest in CONFIGS_BEFORE.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest


def test_union_of_origin_boxes_counts_distinct_origins_without_overlap_inflation():
    boxes=[OriginBox((0,0,0),(3,3,3)),OriginBox((2,2,2),(5,5,5)),OriginBox((0,0,0),(3,3,3))]
    brute={o for b in boxes for o in product(*(range(a,c+1) for a,c in zip(b.lower,b.upper)))}
    assert union_origin_count(boxes)==len(brute)==120


def test_exact_common_center_and_world_clipped_nested_chains():
    source=[.95,.95,.95]
    interior={'positive':[400,400,400],'negative':[410,410,410]}
    common=common_center_regions(interior,source,1024)
    assert common
    for center in (common[0].lower,common[0].upper):
        for side in (32,64,128,256):
            o=tuple(v-side//2 for v in center)
            assert any(b.contains(o) for b in pair_region(interior,source,side,1024))
    edge={'positive':[5,5,5],'negative':[12,12,12]}
    assert not common_center_regions(edge,source,1024)
    chain=nearby_nested_chain(edge,source,1024)
    assert chain and len(chain)==4
    for i,r in enumerate(chain):
        assert any(b.contains(r['origin']) for b in pair_region(edge,source,r['side'],1024))
        if i:
            prev=chain[i-1]
            assert np.all(np.array(r['origin'])<=prev['origin'])
            assert np.all(np.array(r['origin'])+r['side']>=np.array(prev['origin'])+prev['side'])
