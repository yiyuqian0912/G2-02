"""Continuous union proofs; synthetic boxes are not new B2 ground truth."""
from fractions import Fraction as Q
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_angular.api import AngularOracle,extension
from experiments.gaussian_geometry.b2_angular.fixtures import SOURCE,RECEIVER,INNER_LO,INNER_HI,union_scene,exact_hit
from experiments.gaussian_geometry.b2_native.api import NativeOracle
from experiments.gaussian_geometry.b2_regions.api import RegionOracle
from experiments.gaussian_geometry.models import Kernel,KernelScene,Threshold
from experiments.gaussian_geometry.queries import FieldOracle

@pytest.fixture(scope='module',autouse=True)
def built():
    try:extension()
    except ImportError as e:pytest.skip(str(e))


@pytest.mark.parametrize('independent',[False,True])
def test_decisive_continuous_union_with_different_blocker_depths(independent):
    m=extension();lo,hi=RECEIVER
    for a,b in zip(INNER_LO,INNER_HI):
        v=m.screen(SOURCE,lo,hi,[a],[b],independent)
        assert v['visibility']==-1
        assert not all(exact_hit(SOURCE,p,a,b) for p in product(*zip(lo,hi)))
    assert m.screen(SOURCE,lo,hi,INNER_LO,INNER_HI,independent,fusion=False)['visibility']==-1
    v=m.screen(SOURCE,lo,hi,INNER_LO,INNER_HI,independent)
    assert v['visibility']==0 and v['distinct_blockers']==2 and v['first_hit_upper']<.75
    # A separate exact universal proof: U is contained in [-1/6,1/6]x[-1/24,1/24].
    # Negative and positive halves are covered at rho=1/4 and rho=7/16.
    # Checking affine extrema at corners proves the full rectangles, not samples.
    for urange,rho,a,b in zip(((Q(-1,6),Q(0)),(Q(0),Q(1,6))),(Q(1,4),Q(7,16)),INNER_LO,INNER_HI):
        for u,vv in product(urange,(Q(-1,24),Q(1,24))):
            point=(Q(1,8)+rho,Q(1,2)+rho*u,Q(1,2)+rho*vv)
            assert all(Q(float(x))<=y<=Q(float(z)) for x,y,z in zip(a,point,b))


@pytest.mark.parametrize('independent',[False,True])
def test_tiny_angular_gap_is_not_padded(independent):
    a=np.array(INNER_LO);b=np.array(INNER_HI);b[0,1]=.5-2**-30;a[1,1]=.5+2**-30
    v=extension().screen(SOURCE,*RECEIVER,a,b,independent)
    assert v['visibility']==-1
    assert not any(exact_hit(SOURCE,[.9,.5,.5],x,y) for x,y in zip(a,b))


@pytest.mark.parametrize('independent',[False,True])
def test_projection_without_depth_never_shadows_front_receivers(independent):
    v=extension().screen(SOURCE,[.25,.49,.49],[.3,.51,.51],INNER_LO,INNER_HI,independent)
    assert v['visibility']==-1


def test_cross_chart_and_source_direction_are_explicit_unknown():
    m=extension()
    assert m.chart([.125]*3,[.75,.75,.13],[.9,.9,.15])[0]==-1
    assert m.chart(SOURCE,SOURCE,SOURCE)[0]==-1
    assert m.screen([.125]*3,[.75,.75,.13],[.9,.9,.15],INNER_LO,INNER_HI)['visibility']==-1
    for axis,sign in product(range(3),(-1,1)):
        s=np.full(3,.5);r=s.copy();r[axis]+=.25*sign
        assert m.chart(s,r,r)[0]==2*axis+(sign<0)
    assert m.screen(SOURCE,*RECEIVER,[],[])['visibility']==-1
    with pytest.raises(ValueError):m.screen([float('nan'),0,0],*RECEIVER,[],[])


@pytest.mark.parametrize('independent',[False,True])
def test_near_source_thin_boxes_and_chart_edges_use_exact_closed_contacts(independent):
    m=extension();source=np.array([.5]*3)
    lo=np.array([.5+2**-38,.5-2**-35,.5-2**-35]);hi=lo+np.array([2**-39,2**-34,2**-34])
    a=np.array([.75,.5-2**-20,.5-2**-20]);b=np.array([.875,.5+2**-20,.5+2**-20])
    row=m.screen(source,a,b,[lo],[hi],independent,12,4096)
    if row['visibility']==0:
        assert all(exact_hit(source,p,lo,hi) for p in product(*zip(a,b)))
    # Closed source contact shadows even a ray directed away from the box.
    point=lo.copy();receiver=point-np.array([.125,0,0])
    assert exact_hit(point,receiver,lo,hi)
    assert m.screen(point,receiver,receiver,[lo],[hi],independent)['visibility']==0


@pytest.mark.parametrize('independent',[False,True])
def test_closed_contact_and_overlapping_but_incomplete_shadows(independent):
    m=extension();box=([.375,.4375,.4375],[.5,.5625,.5625])
    receiver=[.875,.3125,.3125]  # hits lower edge at x=.375
    assert exact_hit(SOURCE,receiver,*box)
    # Interval proof may be UNKNOWN, but cannot label it visible. Fallback is HIT.
    assert m.screen(SOURCE,receiver,receiver,[box[0]],[box[1]],independent)['visibility']!=1
    a=np.array(INNER_LO);b=np.array(INNER_HI);a[:,2]=.49;b[:,2]=.51
    assert m.screen(SOURCE,*RECEIVER,a,b,independent)['visibility']==-1


@pytest.mark.parametrize('independent',[False,True])
def test_union_boxes_are_independently_full_sum_solid(independent):
    s=union_scene();native=NativeOracle(s,independent=independent)
    assert native.source_clearance(SOURCE).status=='FREE'
    assert native.certificate(SOURCE,*RECEIVER)==(-1,0)
    assert RegionOracle(s,independent=independent,mode=4).region(SOURCE,*RECEIVER)[0]==(-1,0)
    for a,b in zip(INNER_LO,INNER_HI):assert native.classify_box(a,b).status=='SOLID'
    for kernel in s.kernels:
        assert NativeOracle(KernelScene('gaussian_sum',[kernel],s.threshold),independent=independent).classify_box([0]*3,[1]*3).status=='FREE'
    for p in product(*zip(*RECEIVER)):
        assert native.membership(p).status=='FREE'
        assert native.segment(SOURCE,p).status=='HIT'
    assert extension().screen(SOURCE,*RECEIVER,INNER_LO,INNER_HI,independent)['visibility']==0


@pytest.mark.parametrize('independent',[False,True])
def test_new_methods_match_unchanged_scalar_and_region_fields(independent):
    s=union_scene();old=NativeOracle(s,independent=independent).field(SOURCE,16,30,scalar=True)
    assert old['complete']
    for method in (0,1,2,3,4):
        result=AngularOracle(s,independent=independent,method=method,cover_depth=3,cover_nodes=1000,
                             adaptive_nodes=128,angular_nodes=1024,trace=True).field(SOURCE,16,30)
        assert result['complete'] and result['valid'].all()
        for k in ('response','solid'):np.testing.assert_array_equal(result[k],old[k])
        assert not AngularOracle.ready_capable
    result=AngularOracle(s,independent=independent,angular_boxes=1,cover_depth=3).field(SOURCE,8,30)
    assert result['complete'] and result['work'].get('angular_inner_boxes',0)<=1


@pytest.mark.parametrize('independent',[False,True])
def test_source_inside_unknown_outer_box_and_budget_remain_fail_closed(independent):
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3)*.125)],Threshold('exp',-.5))
    source=[.625+2**-10,.503,.503];native=NativeOracle(s,independent=independent)
    assert native.source_clearance(source).status=='FREE'
    assert native.classify_box([.625,.5,.5],[.75,.625,.625]).status=='UNKNOWN'
    old=native.field(source,8,30)
    for method in (0,2,3,4):
        v=AngularOracle(s,independent=independent,method=method,cover_depth=3,angular_nodes=6,adaptive_nodes=0,empty_prefix=True).field(source,8,30)
        assert v['complete'];np.testing.assert_array_equal(v['response'],old['response']);np.testing.assert_array_equal(v['solid'],old['solid'])
    v=AngularOracle(s,independent=independent).field(source,8,1e-9)
    assert not v['complete'] and not v['valid'].any()


@pytest.mark.parametrize('independent',[False,True])
def test_physical_closed_source_contact_is_not_relabelled_visible(independent):
    # Not an admissible dataset source: an oracle contact regression only.
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3)*.125)],Threshold('exp',-.5))
    source=[.625,.5,.5];old=NativeOracle(s,independent=independent).field(source,8,30)
    assert old['complete'] and not old['response'].any()
    for method in (0,2,3,4):
        result=AngularOracle(s,independent=independent,method=method,cover_depth=3,empty_prefix=True,angular_nodes=512).field(source,8,30)
        assert result['complete'] and not result['response'].any()
        np.testing.assert_array_equal(result['solid'],old['solid'])


@pytest.mark.parametrize('independent',[False,True])
def test_unaligned_tile_and_deterministic_proofs(independent):
    s=union_scene();old=NativeOracle(s,independent=independent).field(SOURCE,16,30)
    options=dict(independent=independent,cover_depth=3,angular_nodes=512,adaptive_nodes=64,method=4)
    a=AngularOracle(s,**options).tile(SOURCE,16,[3,1,7],8,30)
    b=AngularOracle(s,**options).tile(SOURCE,16,[3,1,7],8,30)
    assert a['complete'] and a['work']==b['work']
    for k in ('response','solid'):np.testing.assert_array_equal(a[k],old[k][7:15,1:9,3:11]);np.testing.assert_array_equal(a[k],b[k])
    with pytest.raises(ValueError):AngularOracle(s).tile(SOURCE,16,[9,0,0],8)
    with pytest.raises(TypeError):AngularOracle(s).tile(SOURCE,16,[0.,0.,0.],8)
    with pytest.raises(ValueError):AngularOracle(s,angular_depth=19).field(SOURCE,8)
    with pytest.raises(ValueError):AngularOracle(s).field(SOURCE,1024)
    with pytest.raises(ValueError):AngularOracle(s,prefix_builds=-1).field(SOURCE,8)
    # A new call/source gets a new map, never the preceding source's decisions.
    changed=[.125,.55,.5]
    a=AngularOracle(s,**options).field(changed,8,30)
    b=NativeOracle(s,independent=independent).field(changed,8,30)
    assert a['complete'] and b['complete']
    for k in ('response','solid'):np.testing.assert_array_equal(a[k],b[k])


def test_random_certified_union_labels_agree_with_independent_exact_clipping():
    rng=np.random.default_rng(20261106);m=extension()
    for _ in range(40):
        source=rng.uniform(.05,.2,3);lo=rng.uniform(.3,.8,(5,3));hi=np.minimum(1,lo+rng.uniform(.03,.2,(5,3)))
        receiver_lo=rng.uniform(.5,.85,3);receiver_hi=np.minimum(.99,receiver_lo+rng.uniform(.001,.08,3))
        points=np.r_[np.array(list(product(*zip(receiver_lo,receiver_hi)))),rng.uniform(receiver_lo,receiver_hi,(32,3))]
        for independent in (False,True):
            v=m.screen(source,receiver_lo,receiver_hi,lo,hi,independent,7,2048)
            if v['visibility']==0:assert all(any(exact_hit(source,p,a,b) for a,b in zip(lo,hi)) for p in points)


def test_union_composition_against_exact_continuous_rectangle_arrangement():
    # Independent finite arrangement: all projected box faces are known exact
    # rational breakpoints. Endpoints and one representative per open cell
    # establish every stratum, not an arbitrary sampling of angular pixels.
    rng=np.random.default_rng(84361);m=extension();certified=0
    arrangements=[[(Q(-1,2),Q(0),Q(-1,2),Q(0)),(Q(0),Q(1,2),Q(-1,2),Q(0)),
                   (Q(-1,2),Q(0),Q(0),Q(1,2)),(Q(0),Q(1,2),Q(0),Q(1,2))]]
    grid=[Q(i,8) for i in range(-4,5)]
    for _ in range(32):
        rectangles=[]
        for _ in range(4):
            u=sorted(rng.choice(len(grid),2,replace=False));v=sorted(rng.choice(len(grid),2,replace=False))
            rectangles.append((grid[u[0]],grid[u[1]],grid[v[0]],grid[v[1]]))
        arrangements.append(rectangles)
    target=(Q(-1,4),Q(1,4))
    def strata(edges):
        values=sorted(set([*target,*[x for x in edges if target[0]<=x<=target[1]]]))
        return values+[(a+b)/2 for a,b in zip(values,values[1:])]
    for rectangles in arrangements:
        us=strata([x for r in rectangles for x in r[:2]]);vs=strata([x for r in rectangles for x in r[2:]])
        truth=all(any(a<=u<=b and c<=v<=d for a,b,c,d in rectangles) for u,v in product(us,vs))
        lo=[];hi=[]
        for i,(a,b,c,d) in enumerate(rectangles):
            depth=Q(2+i%3,8)
            lo.append([float(Q(1,8)+depth),float(Q(1,2)+depth*a),float(Q(1,2)+depth*c)])
            hi.append([float(Q(1,8)+depth),float(Q(1,2)+depth*b),float(Q(1,2)+depth*d)])
        result=m.screen(SOURCE,[.875,.3125,.3125],[.875,.6875,.6875],lo,hi,True,8,4096)
        if result['visibility']==0:assert truth;certified+=1
        if not truth:assert result['visibility']==-1
    assert certified>0


def test_frozen_native_and_previous_region_numerical_sources_are_unchanged():
    root=Path(__file__).parents[1]
    acceptance=json.loads((root/'results/b2-regions-acceptance-v1.json').read_text())
    for name,h in acceptance['native_build']['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
