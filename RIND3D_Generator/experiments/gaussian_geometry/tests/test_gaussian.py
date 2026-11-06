"""Research proofs/regressions. No READY publication or frozen-data mutation."""
from fractions import Fraction
import json

from flint import arb,ctx
import numpy as np
import pytest

from experiments.gaussian_geometry.models import Kernel,KernelScene,ConvexScene,SpectralScene,Threshold,record,save,load
from experiments.gaussian_geometry.queries import FieldOracle,Limits,make_oracle,qrange,qbox,rational,endpoint
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.generators import DEFAULT,generate,grid_points


def sphere(kind='ellipsoid_union',matrix=None):
    k=Kernel([.5]*3,np.eye(3)*.125 if matrix is None else matrix)
    return KernelScene(kind,[k],Threshold('exp',-.5))


@pytest.mark.parametrize('kind',['ellipsoid_union','gaussian_sum'])
def test_exact_boundary_endpoint_tangency_and_nextafter(kind):
    s=sphere(kind);a=FieldOracle(s);b=Reference(s)
    for p,wanted in [([.625,.5,.5],'SOLID'),([np.nextafter(.625,np.inf),.5,.5],'FREE'),
                     ([np.nextafter(.625,-np.inf),.5,.5],'SOLID'),([.5]*3,'SOLID')]:
        assert a.membership(p).status==b.membership(p).status==wanted
    for start,stop,wanted in [([0,.625,.5],[1,.625,.5],'HIT'),
        ([0,np.nextafter(.625,np.inf),.5],[1,np.nextafter(.625,np.inf),.5],'MISS'),
        ([.625,.5,.5],[1,.5,.5],'HIT'),([.5]*3,[.5]*3,'HIT'),
        ([0,0,0],[0,0,0],'MISS'),([.5]*3,[1,1,1],'HIT')]:
        assert a.segment(start,stop).status==b.segment(start,stop).status==wanted
        assert a.segment(stop,start).status==wanted


@pytest.mark.parametrize('matrix',[
    np.array([[.125,.125,0],[-.125,.125,0],[0,0,.0625]]),
    np.diag([.125,.0625,2**-30]),
    np.array([[.125,.03125,0],[0,.0625,0],[0,0,.125]])])
def test_thin_rotated_sheared_ellipsoids_independent(matrix):
    s=sphere(matrix=matrix);a=FieldOracle(s);b=Reference(s)
    rng=np.random.default_rng(912)
    for _ in range(120):
        p,t=rng.random((2,3))
        assert a.membership(p).status==b.membership(p).status
        assert a.segment(p,t).status==b.segment(p,t).status
    assert a.membership(np.array([.5]*3)+matrix[:,0]).status=='SOLID'


def test_sum_is_not_union_and_can_fuse_components():
    kernels=[Kernel([.5+d,.5,.5],np.eye(3)*.08) for d in (-.10,.10)]
    union=FieldOracle(KernelScene('ellipsoid_union',kernels,Threshold('exp',-.5)))
    summed=FieldOracle(KernelScene('gaussian_sum',kernels,Threshold('exp',-.5)))
    assert union.membership([.5]*3).status=='FREE'
    assert summed.membership([.5]*3).status=='SOLID'
    assert Reference(summed.scene).membership([.5]*3).status=='SOLID'


def test_small_tails_are_not_discarded():
    k=Kernel([.5]*3,np.eye(3)*.1)
    p=[.8,.5,.5]
    # Two individually sub-threshold contributions jointly cross the threshold.
    threshold=Threshold('binary64',.02)
    assert FieldOracle(KernelScene('gaussian_sum',[k],threshold)).membership(p).status=='FREE'
    assert FieldOracle(KernelScene('gaussian_sum',[k,k],threshold)).membership(p).status=='SOLID'


def test_exact_common_center_equality_is_closed():
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3),.25),Kernel([.5]*3,np.eye(3),.75)],Threshold('binary64',1.))
    assert FieldOracle(s).membership([.5]*3).status==Reference(s).membership([.5]*3).status=='SOLID'


def test_budget_exhaustion_is_not_physical_rejection():
    s=KernelScene('gaussian_sum',[Kernel([.4,.5,.5],np.eye(3)*.09),Kernel([.6,.5,.5],np.eye(3)*.09)],Threshold('binary64',.7))
    o=FieldOracle(s,Limits(max_nodes=0))
    d=o.segment([0,.5,.5],[1,.5,.5])
    assert d.status=='BUDGET_EXHAUSTED' and not d.certified


def test_global_containment_and_boundary_contact_failure():
    assert FieldOracle(sphere()).containment(.02)['status']=='CERTIFIED'
    s=KernelScene('gaussian_sum',[Kernel([.01,.5,.5],np.eye(3)*.15)],Threshold('exp',-.5))
    assert FieldOracle(s).containment(.02)['status']=='UNKNOWN'
    # A positive sum is bounded, but world clipping is NOT containment proof.
    assert FieldOracle(s).membership([0,.5,.5]).status=='SOLID'


def test_containment_outer_support_and_acceleration_do_not_change_decisions():
    s=generate('gaussian_sum',144,6,.02)
    fast=FieldOracle(s);slow=FieldOracle(s,use_support=False,use_proxies=False)
    ref=Reference(s,max_nodes=4096,seconds=5.)
    assert fast.containment(DEFAULT['world_margin'])['status']=='CERTIFIED'
    rng=np.random.default_rng(5)
    for _ in range(30):
        a,b=rng.random((2,3))
        assert fast.membership(a).status==slow.membership(a).status==ref.membership(a).status
        x,y,z=fast.segment(a,b),slow.segment(a,b),ref.segment(a,b)
        assert x.certified and y.certified and z.certified
        assert x.status==y.status==z.status


def test_exact_interval_bounds_enclose_direct_point_queries():
    s=generate('gaussian_sum',88,3,.02);oracle=FieldOracle(s);ref=Reference(s)
    lo,hi=[.3,.25,.4],[.4,.35,.5]
    for k in s.kernels:
        l,h=qbox(k,tuple(map(rational,lo)),tuple(map(rational,hi)))
        for p in np.random.default_rng(7).uniform(lo,hi,(30,3)):
            assert l<=k.quadratic(p)<=h
    with ctx.workprec(256):
        bound=oracle._score_box(tuple(map(rational,lo)),tuple(map(rational,hi)))
        for p in np.random.default_rng(8).uniform(lo,hi,(30,3)):
            low,high=ref.bounds(tuple(map(Fraction,p)),tuple(map(Fraction,p)))
            from flint import fmpq
            assert endpoint(bound,False)<=fmpq(low.numerator,low.denominator)
            assert endpoint(bound,True)>=fmpq(high.numerator,high.denominator)


@pytest.mark.parametrize('model',['gaussian_convex','ellipsoid_union','gaussian_sum','spectral_grf'])
def test_determinism_lossless_roundtrip_and_closed_publication(tmp_path,model):
    a=generate(model,13,5,.02);b=generate(model,13,5,.02)
    assert record(a)==record(b)
    path=tmp_path/'geometry.json';save(path,a);copy=load(path)
    assert record(copy)==record(a)
    assert record(copy)['ready_capable'] is False
    np.testing.assert_array_equal(copy.approx_values(grid_points(6)),a.approx_values(grid_points(6)))
    if model=='gaussian_convex':
        oracle=make_oracle(a);assert oracle.containment(DEFAULT['world_margin'])['status']=='CERTIFIED'
    else:
        assert FieldOracle(a).containment(DEFAULT['world_margin'])['status']=='CERTIFIED'
    raw=json.loads(path.read_text());raw['parameters']['metadata']['seed']=999
    path.write_text(json.dumps(raw))
    with pytest.raises((ValueError,AssertionError)):load(path)


def test_spectral_field_is_continuous_finite_and_tapered_not_grid_truth():
    s=generate('spectral_grf',11,0,.05);a=FieldOracle(s);b=Reference(s)
    assert a.containment(DEFAULT['world_margin'])['status']=='CERTIFIED'
    for p in ([0,0,0],[1,1,1],[0,.5,.5],[.5,.5,.5],[.47,.53,.44]):
        assert a.membership(p).status==b.membership(p).status
    for i in range(6):
        p=np.random.default_rng(30+i).random((2,3))
        x,y=a.segment(*p),b.segment(*p)
        if x.certified and y.certified:assert x.status==y.status


def test_existing_convex_primary_independent_and_source_clearance():
    s=generate('gaussian_convex',41,2,.02);a=make_oracle(s)
    source=np.array([.03125]*3);assert a.source_clearance(source).status=='FREE'
    one,two=a.context.bind(source),a.context.bind(source,independent=True)
    points=np.random.default_rng(56).random((500,3))
    np.testing.assert_array_equal(one.response(source,points),two.response(source,points))
    ya,ma,_=one.render((0,0,0),32,32)
    yb,mb,_=two.render((0,0,0),32,32)
    np.testing.assert_array_equal(ya,yb);np.testing.assert_array_equal(ma,mb)
    assert not np.any(ya&ma)


def test_invalid_parameters_and_precision_context_restore():
    with pytest.raises(ValueError):Kernel([.5]*3,np.zeros((3,3)))
    with pytest.raises(ValueError):Kernel([.5]*3,np.eye(3),-1.)
    with pytest.raises(ValueError):Threshold('binary64',0.)
    before=ctx.prec;s=generate('gaussian_sum',77,0,.01);o=FieldOracle(s)
    o.membership([.5]*3);o.segment([0,0,0],[1,1,1]);assert ctx.prec==before
    with pytest.raises(ValueError):o.membership([np.nan,.5,.5])


def test_spectral_closed_polynomial_tangency_and_endpoint():
    # Coefficients zero reduce the continuous taper to an exact polynomial.
    s=SpectralScene([[1,0,0]],[0],[0],[.5]*3,[.25]*3,Threshold('binary64',9/16),bias=1.)
    a,b=FieldOracle(s),Reference(s)
    assert a.membership([.625,.5,.5]).status==b.membership([.625,.5,.5]).status=='SOLID'
    assert a.membership([np.nextafter(.625,np.inf),.5,.5]).status=='FREE'
    assert a.segment([.625,0,.5],[.625,1,.5]).status==b.segment([.625,0,.5],[.625,1,.5]).status=='HIT'


def test_clearance_touch_is_not_accepted_and_small_field_pair_is_complete():
    from experiments.gaussian_geometry.render import small_field
    s=sphere('gaussian_sum');a=FieldOracle(s)
    assert a.source_clearance([.625+2**-26,.5,.5]).status!='FREE'
    first=small_field(a,[.03125]*3,8,15.)
    second=small_field(a,[.03125]*3,8,15.,independent=Reference(s))
    assert first['complete'] and second['complete']
    np.testing.assert_array_equal(first['response'],second['response'])
    np.testing.assert_array_equal(first['solid'],second['solid'])
    with pytest.raises(ValueError):small_field(a,[.03125]*3,1024)


def test_reference_extremely_small_tails_have_positive_bounded_remainder():
    from experiments.gaussian_geometry.reference import exp_interval
    a,b=exp_interval(Fraction(-10**9),Fraction(-10**8),192)
    assert a==0 and b==Fraction(1,1<<4096)
    assert b>0


def test_native_full_128_field_is_assembled_from_64_bricks():
    from experiments.gaussian_geometry.render import small_field
    scene=ConvexScene([{'center':np.array([.5]*3),'matrix':np.eye(3)*.125}])
    oracle=make_oracle(scene);source=[.03125]*3
    a=small_field(oracle,source,128,15.)
    b=small_field(oracle,source,128,15.,independent=True)
    assert a['complete'] and b['complete'] and a['native_max_brick_side']==64
    assert a['solid'].sum()==32**3
    np.testing.assert_array_equal(a['response'],b['response']);np.testing.assert_array_equal(a['solid'],b['solid'])


def test_different_corner_occluders_do_not_certify_a_union_shadow():
    from itertools import product
    kernels=[Kernel([.45,y,z],np.eye(3)*.045) for y,z in product((.34,.66),repeat=2)]
    o=FieldOracle(KernelScene('ellipsoid_union',kernels))
    source=[.05,.5,.5];lo=[.75,.2,.2];hi=[.85,.8,.8]
    assert all(o.segment(source,p).status=='HIT' for p in product(*zip(lo,hi)))
    assert o.segment(source,[.8,.5,.5]).status=='MISS'
    assert o.certificate(source,lo,hi)[0]==-1


def test_narrow_gaussian_intersection_is_not_fixed_step_marching():
    k=Kernel([.543123,.5,.5],np.eye(3)*2**-28)
    s=KernelScene('gaussian_sum',[k,k],Threshold('binary64',1.))
    assert FieldOracle(s).segment([0,.5,.5],[1,.5,.5]).status=='HIT'
    assert Reference(s,seconds=3.).segment([0,.5,.5],[1,.5,.5]).status=='HIT'


def test_unresolved_transcendental_tangency_is_never_miss():
    k=Kernel([.5]*3,np.eye(3)*.125,.5)
    s=KernelScene('gaussian_sum',[k,k],Threshold('exp',-.5))
    o=FieldOracle(s,Limits(max_nodes=4,max_depth=2,seconds=.1))
    # The algebraically reducible duplicate sum is not symbolically merged by
    # this prototype. It must preserve uncertainty instead of assuming FREE.
    assert o.membership([.625,.5,.5]).status in ('SOLID','UNKNOWN')
    assert o.segment([0,.625,.5],[1,.625,.5]).status in ('HIT','UNKNOWN','BUDGET_EXHAUSTED')


def test_nonlocal_attribution_counters_do_not_starve_later_window_sizes():
    from experiments.gaussian_geometry.study import window_diagnostic
    from experiments.gaussian_geometry.queries import Decision

    class CountingOracle:
        """Algorithmic accounting fixture; not a physical geometry oracle."""
        calls=0

        def membership(self,p):return Decision('FREE','synthetic_fixture')

        def segment(self,start,stop):
            if list(start)==[0.,0.,0.]:
                self.calls+=1
                return Decision('HIT' if self.calls%2 else 'MISS','synthetic_fixture')
            return Decision('MISS','synthetic_suffix_fixture')

    result=window_diagnostic(CountingOracle(),np.zeros(3),19,seconds=10.)
    counts=result['counts']
    for side in (64,128,256):
        assert counts[f'side_{side}_proposed']==8
        assert counts[f'side_{side}_informative']==8
        assert counts[f'side_{side}_shadow_attribution_queries']==32
        assert counts[f'side_{side}_attribution_external_blocking_proved_inside_segment_free']==32
    assert counts['external_blocking_proved_inside_segment_free']==96


def test_existing_canonical_ellipsoid_export_and_original_oracle():
    from experiments.gaussian_geometry.scripts.canonical_ellipsoids import to_canonical
    from rind3d.geometry import Oracle
    from rind.geometry import SegmentDecision,Membership
    s=sphere(matrix=np.array([[.125,.125,0],[-.125,.125,0],[0,0,.0625]]))
    bundle=to_canonical(s);reference=Oracle(bundle)
    p=FieldOracle(s)
    rng=np.random.default_rng(451)
    for a,b in rng.random((20,2,3)):
        m=reference.reference.classify_points(a)
        r=reference.reference.segment_intersect(a,b)
        assert m.valid.all() and r.valid.all()
        assert p.membership(a).status==('FREE' if int(m.value[0])==int(Membership.FREE) else 'SOLID')
        assert p.segment(a,b).status==('HIT' if int(r.value[0])==int(SegmentDecision.HIT) else 'MISS')
    with pytest.raises(ValueError):to_canonical(sphere('gaussian_sum'))


def test_per_component_world_fit_keeps_centers_and_respects_world():
    from experiments.gaussian_geometry.scripts.placement_ablation import generate_capped
    old=generate('gaussian_convex',20261105,18,.15)
    new=generate_capped(20261105,18,.15)
    np.testing.assert_array_equal([o['center'] for o in old.objects],[o['center'] for o in new.objects])
    assert [o['kind'] for o in old.objects]==[o['kind'] for o in new.objects]
    assert make_oracle(new).containment(DEFAULT['world_margin'])['status']=='CERTIFIED'
    points=grid_points(32)
    assert np.mean(new.approx_values(points)>=0)>=np.mean(old.approx_values(points)>=0)
