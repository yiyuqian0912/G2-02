"""Independent physical and algorithmic evidence for the isolated B2 engine."""
import ctypes
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.models import Kernel,KernelScene,Threshold,load
from experiments.gaussian_geometry.queries import FieldOracle
from experiments.gaussian_geometry.reference import Reference
from experiments.gaussian_geometry.b2_native.api import NativeOracle


def mixture():
    return KernelScene('gaussian_sum',[Kernel([.4,.5,.5],np.eye(3)*.08),Kernel([.6,.5,.5],np.eye(3)*.08)],Threshold('binary64',.7))


@pytest.fixture(scope='module',autouse=True)
def built():
    from experiments.gaussian_geometry.b2_native.api import extension
    try:extension()
    except ImportError as e:pytest.skip(str(e))


@pytest.mark.parametrize('independent',[False,True])
def test_exact_closed_contact_tangencies_and_nextafter(independent):
    k=Kernel([.5]*3,np.diag([.125,.0625,2**-28]),.5)
    scene=KernelScene('gaussian_sum',[k,k],Threshold('exp',-.5));e=NativeOracle(scene,independent=independent)
    for p,w in [([.625,.5,.5],'SOLID'),([np.nextafter(.625,np.inf),.5,.5],'FREE'),([.5]*3,'SOLID')]:
        assert e.membership(p).status==w
    for a,b,w in [([0,.5625,.5],[1,.5625,.5],'HIT'),([0,np.nextafter(.5625,np.inf),.5],[1,np.nextafter(.5625,np.inf),.5],'MISS'),
                  ([.625,.5,.5],[1,.5,.5],'HIT'),([.5]*3,[.5]*3,'HIT'),([0,0,0],[0,0,0],'MISS')]:
        assert e.segment(a,b).status==e.segment(b,a).status==w
    assert len(scene.kernels)==2 and not e.ready_capable


@pytest.mark.parametrize('independent',[False,True])
def test_invalid_unknown_and_budget_are_distinct(independent):
    e=NativeOracle(mixture(),independent=independent,max_nodes=0,filters=False)
    assert e.segment([0,0,0],[1,1,1]).status=='UNKNOWN'
    assert e.membership([np.nan,0,0]).status=='INVALID'
    assert e.segment([0,0,0],[np.inf,0,0]).status=='INVALID'
    assert e.classify_box([1,0,0],[0,1,1]).status=='INVALID'
    with pytest.raises(ValueError):NativeOracle(mixture(),max_depth=51).segment([0,0,0],[1,1,1])


@pytest.mark.parametrize('seed',[41,92,501])
def test_three_numerical_paths_agree_on_rotated_unequal_kernels(seed):
    rng=np.random.default_rng(seed)
    matrix=np.array([[.08,.025,0],[-.035,.10,.012],[0,.02,.045]])
    s=KernelScene('gaussian_sum',[Kernel([.35,.45,.55],matrix,.8),Kernel([.56,.53,.43],matrix*.4,1.3),Kernel([.62,.40,.60],np.diag([.03,.07,.04]),.01)],Threshold('binary64',.55))
    old=FieldOracle(s);a=NativeOracle(s);b=NativeOracle(s,independent=True);points=rng.random((80,3));source=[.03125,.082,.017]
    pa=a.points(points);pb=b.points(points);ra=a.rays(source,points);rb=b.rays(source,points)
    for p,aa,bb,ar,br in zip(points,pa,pb,ra,rb):
        assert aa.certified and bb.certified and aa.status==bb.status==old.membership(p).status
        assert ar.certified and br.certified and ar.status==br.status==old.segment(source,p).status


def test_modes_do_not_change_geometry_or_certified_labels():
    scene=mixture();pts=np.random.default_rng(301).random((100,3));source=[.03125]*3
    baseline=NativeOracle(scene,coefficients=False,filters=False)
    expected=[d.status for d in baseline.rays(source,pts)]
    for opts in [dict(coefficients=True,filters=False),dict(filters=True),dict(groups=True),dict(fallback=False)]:
        e=NativeOracle(scene,**opts);result=e.rays(source,pts)
        assert [d.status for d in result]==expected
    assert baseline.work['coefficient_builds']>NativeOracle(scene).native.rays(source,pts,NativeOracle(scene).options)[1]['coefficient_builds']


@pytest.mark.parametrize('independent',[False,True])
def test_spatial_bounds_are_conservative_at_unaligned_boxes(independent):
    s=mixture();e=NativeOracle(s,independent=independent);old=FieldOracle(s);rng=np.random.default_rng(955)
    for _ in range(32):
        center=rng.random(3);radius=rng.uniform(.0001,.12);lo=center-radius;hi=center+radius;d=e.classify_box(lo,hi)
        if d.certified:
            for p in rng.uniform(lo,hi,(12,3)):assert old.membership(p).status==d.status


@pytest.mark.parametrize('independent',[False,True])
def test_clearance_is_a_whole_cube_proof(independent):
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3)*.125)],Threshold('exp',-.5));e=NativeOracle(s,independent=independent)
    assert e.source_clearance([.03125]*3).status=='FREE'
    assert e.source_clearance([.625+2**-26,.5,.5]).status=='UNKNOWN'
    assert e.source_clearance([0,0,0]).status=='INVALID'


def test_subnormal_threshold_uncertainty_is_not_a_label():
    s=KernelScene('gaussian_sum',[Kernel([0,0,0],np.eye(3),.25),Kernel([0,0,0],np.eye(3)*2,.75)],Threshold('binary64',1.))
    for independent in (False,True):
        e=NativeOracle(s,independent=independent)
        assert e.membership([0,0,0]).status=='SOLID'
        assert e.membership([2**-600,0,0]).status=='UNKNOWN'
        assert e.segment([-1,0,0],[1,0,0]).status=='HIT'


@pytest.mark.parametrize('independent',[False,True])
def test_extremely_narrow_hit_is_not_fixed_step_marching(independent):
    k=Kernel([.543123,.5,.5],np.eye(3)*2**-28)
    s=KernelScene('gaussian_sum',[k,k],Threshold('binary64',1.));e=NativeOracle(s,independent=independent)
    assert e.segment([0,.5,.5],[1,.5,.5]).status=='HIT'


def test_positive_group_remainders_keep_all_far_tails():
    rng=np.random.default_rng(281);ks=[Kernel(c,np.eye(3)*.015,float(w)) for c,w in zip(rng.uniform(.2,.8,(64,3)),rng.uniform(.4,1.5,64))]
    scene=KernelScene('gaussian_sum',ks,Threshold('binary64',.5));a=NativeOracle(scene,groups=True,filters=False);b=NativeOracle(scene,groups=False,filters=False)
    points=np.r_[rng.uniform(.001,.02,(32,3)),rng.random((16,3))]
    assert [d.status for d in a.points(points)]==[d.status for d in b.points(points)]
    assert a.work['group_remainders']>0 and a.work['group_kernel_terms_retained_as_remainder']>0


def test_full_small_fields_independent_scalar_hierarchy_and_old_arb():
    from experiments.gaussian_geometry.render import small_field
    s=mixture();source=[.03125]*3;old=small_field(FieldOracle(s),source,8,30.)
    outputs=[]
    for independent in (False,True):
        e=NativeOracle(s,independent=independent)
        for scalar in (False,True):
            v=e.field(source,8,30.,scalar=scalar);assert v['complete'];outputs.append(v)
            np.testing.assert_array_equal(v['response'],old['response']);np.testing.assert_array_equal(v['solid'],old['solid'])
    assert outputs[0]['work'].get('region_filled_voxels',0)>0
    with pytest.raises(ValueError):NativeOracle(s).field(source,1024)


def test_receiver_region_shadow_certificate_cannot_mix_corner_occluders():
    from itertools import product
    ks=[Kernel([.45,y,z],np.eye(3)*.045) for y,z in product((.34,.66),repeat=2)]
    s=KernelScene('gaussian_sum',ks,Threshold('exp',-.5));source=[.05,.5,.5];lo=[.75,.2,.2];hi=[.85,.8,.8]
    for independent in (False,True):
        e=NativeOracle(s,independent=independent)
        assert all(e.segment(source,p).status=='HIT' for p in product(*zip(lo,hi)))
        assert e.segment(source,[.8,.5,.5]).status=='MISS'
        assert e.certificate(source,lo,hi)[0]!=0


def test_wrong_rounding_environment_fails_closed():
    libc=ctypes.CDLL(None);saved=libc.fegetround()
    e=NativeOracle(mixture())
    try:
        assert libc.fesetround(0x400)==0  # Linux x86 FE_DOWNWARD.
        with pytest.raises(RuntimeError):e.membership([.5]*3)
    finally:libc.fesetround(saved)


def test_rejects_non_b2_and_invalid_parameters():
    union=KernelScene('ellipsoid_union',[Kernel([.5]*3,np.eye(3)*.1)])
    with pytest.raises(ValueError):NativeOracle(union)
    with pytest.raises(ValueError):NativeOracle(mixture(),nonexistent=True)
    from fractions import Fraction
    with pytest.raises(TypeError):NativeOracle(mixture()).segment([Fraction(1,3)]*3,[.8]*3)
    with pytest.raises(TypeError):NativeOracle(mixture()).source_clearance([Fraction(1,3)]*3)
    with pytest.raises(TypeError):NativeOracle(mixture()).field([Fraction(1,3)]*3,8)


@pytest.mark.parametrize('independent',[False,True])
def test_aggregate_taylor_resolves_recorded_near_boundary_miss(independent):
    # Frozen case 7 / 64³ voxel (23,38,45). Unchanged Arb independently
    # certified MISS in 6197 nodes; the original native variants exhausted 4096.
    from experiments.gaussian_geometry.generators import generate
    s=generate('gaussian_sum',20261105,7,.01);source=np.random.default_rng(20261112).random(3)
    receiver=(np.array([23,38,45])+.5)/64
    baseline=NativeOracle(s,independent=independent,taylor=False,exact_depth=51,max_nodes=512)
    improved=NativeOracle(s,independent=independent,taylor=True,max_nodes=512)
    assert baseline.segment(source,receiver).status=='UNKNOWN'
    result=improved.segment(source,receiver)
    assert result.status=='MISS' and result.nodes<=512
    assert improved.work['reference_taylor_interval_free' if independent else 'taylor_interval_free']>0


@pytest.mark.parametrize('independent',[False,True])
def test_nontrivial_transcendental_tangent_is_never_false_miss(independent):
    # Along this ray both quadratics are .5+z_distance²/.125², and the
    # exact maximum equals exp(-.25). The available interval machinery may
    # remain UNKNOWN at equality, but closed contact can never be MISS.
    s=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.4375,.5625)],Threshold('exp',-.25))
    e=NativeOracle(s,independent=independent,max_nodes=512,max_depth=32)
    assert e.segment([.5,.5625,.03125],[.5,.5625,.96875]).status in ('HIT','UNKNOWN')


def test_reference_tail_remainder_is_positive_and_retried_when_necessary():
    s=KernelScene('gaussian_sum',[Kernel([.5]*3,np.eye(3))],Threshold('binary64',np.nextafter(0.,1.)))
    e=NativeOracle(s,independent=True,tail_bounds=True)
    assert e.membership([1000,1000,1000]).status=='FREE'
    assert e.work['reference_positive_tail_remainders']>0
    assert e.work['reference_exp_endpoints']>0  # coarse remainder alone cannot prove this tiny-threshold case


def test_tighter_global_containment_and_independent_cover_recheck():
    from experiments.gaussian_geometry.b2_native.containment import certify,independently_recheck
    ks=[Kernel([.10,y,.5],np.diag([.05,.03,.03])) for y in (.2,.8)]
    s=KernelScene('gaussian_sum',ks,Threshold('binary64',.5))
    assert FieldOracle(s).containment(.025)['status']=='UNKNOWN'
    proof=certify(s,NativeOracle(s))
    assert proof['status']=='CERTIFIED' and proof['kind']=='closed_outer_support_exterior_cover'
    assert independently_recheck(s,proof,NativeOracle(s,independent=True))['status']=='CERTIFIED'
    import copy
    bad=copy.deepcopy(proof);bad['roots']=bad['roots'][::-1]
    with pytest.raises(ValueError):independently_recheck(s,bad,NativeOracle(s,independent=True))


def test_containment_unknown_and_physical_violation_are_distinct():
    from experiments.gaussian_geometry.b2_native.containment import certify
    s=KernelScene('gaussian_sum',[Kernel([.08,y,.5],np.diag([.05,.03,.03])) for y in (.2,.8)],Threshold('binary64',.5))
    assert NativeOracle(s).membership([.025,.2,.5]).status=='SOLID'
    empty=certify(s,NativeOracle(s),max_nodes=0)
    assert empty['status']=='UNKNOWN'
    checked=certify(s,NativeOracle(s))
    assert checked['status'] in ('OUTSIDE_SOLID_PROVED','UNKNOWN')


def test_parent_region_evidence_reuse_preserves_full_field():
    s=mixture();source=[.03125]*3
    for independent in (False,True):
        a=NativeOracle(s,independent=independent,inherited_regions=True).field(source,16)
        b=NativeOracle(s,independent=independent,inherited_regions=False).field(source,16)
        assert a['complete'] and b['complete']
        for key in ('response','solid'):np.testing.assert_array_equal(a[key],b[key])
        assert a['work'].get('reference_inherited_mask_certificates' if independent else 'inherited_mask_certificates',0)>0


def test_continuous_proxy_bounds_do_not_replace_gaussian_truth():
    from experiments.gaussian_geometry.b2_native.proxy_experiment import build_covers,proxy_context
    s=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.15) for x in (.4,.6)],Threshold('binary64',.5))
    inner,outer,report=build_covers(s,depth=3)
    a,b=proxy_context(inner),proxy_context(outer);assert a is not None and b is not None
    points=np.random.default_rng(5018).random((128,3));source=[.03125]*3;old=FieldOracle(s)
    inside=a.contains(points);outside=b.contains(points)
    yi=a.bind(source,independent=True).response(source,points);yo=b.bind(source,independent=True).response(source,points)
    for p,ii,oo,lit_inner,lit_outer in zip(points,inside,outside,yi,yo):
        actual=old.membership(p).status
        if ii:assert actual=='SOLID'
        if not oo:assert actual=='FREE'
        if lit_outer:assert old.segment(source,p).status=='MISS'
        if not lit_inner:assert old.segment(source,p).status=='HIT'
    assert report['not_a_voxel_truth_representation'] is True


def test_unvisited_proxy_regions_remain_in_the_outer_bound():
    from experiments.gaussian_geometry.b2_native.proxy_experiment import build_covers
    inner,outer,report=build_covers(mixture(),depth=3,max_nodes=0)
    assert not inner and outer==[([0.,0.,0.],[1.,1.,1.])]
    assert report['pending_regions']==1 and report['outer_volume_upper_bound']==1.


def test_greedy_proxy_merge_preserves_the_entire_cell_union():
    from experiments.gaussian_geometry.b2_native.proxy_experiment import rectangles
    mask=np.random.default_rng(2403).random((8,8,8))>.55;out=np.zeros_like(mask)
    for lo,hi in rectangles(mask):
        a=np.rint(np.array(lo)*8).astype(int);b=np.rint(np.array(hi)*8).astype(int)
        out[a[2]:b[2],a[1]:b[1],a[0]:b[0]]=True
    np.testing.assert_array_equal(mask,out)


def test_rebuilt_native_library_requires_a_fresh_python_process():
    from experiments.gaussian_geometry.b2_native.api import extension
    m=extension();saved=m.__b2_binary_sha256
    try:
        m.__b2_binary_sha256='stale'
        with pytest.raises(ImportError):extension()
    finally:m.__b2_binary_sha256=saved


def test_native_parameters_are_a_snapshot_and_proofs_reject_mismatches():
    from experiments.gaussian_geometry.b2_native.containment import certify
    s=mixture();e=NativeOracle(s);before=e.membership([.5]*3).status
    s.threshold=Threshold('binary64',100.)
    assert before==e.membership([.5]*3).status=='SOLID'
    assert NativeOracle(s).membership([.5]*3).status=='FREE'
    with pytest.raises(ValueError):certify(s,e)
