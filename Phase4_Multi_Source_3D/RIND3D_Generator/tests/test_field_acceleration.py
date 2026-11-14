"""Conservative cells/bases/closed-ray proofs against unchanged exact oracles."""
from fractions import Fraction
from itertools import product
import math
import numpy as np
import pytest
from flint import arb,ctx,fmpq
from rind.contracts import Geometry,RootKind,make_leaf,CSGNode,CSGOpcode
from rind.storage import CanonicalBundle
from rind.geometry import Membership
from rind3d.geometry import Oracle,NumericalUnresolved,Budget,BudgetExhausted,centres
from rind3d.field_acceleration import FieldOracle,FunctionBounds,enclosure,exact_fraction,field_evaluator


def bundle(frequency=((9,1,0),(0,7,1),(1,0,6)),phase=(.123,.456,.789),amplitude=(1,.35,.2),threshold=.93,primitive=None):
    b=CanonicalBundle(3);leaf=make_leaf('cosine_field_v1',3,frequency=frequency,phase=phase,amplitude=amplitude,threshold=threshold)
    b.add_leaf(leaf);root=leaf.content_hash
    if primitive is not None:
        b.add_leaf(primitive);node=CSGNode(3,CSGOpcode.UNION,(primitive.content_hash,root));b.add_node(node);root=node.content_hash
    b.add_geometry(Geometry(3,RootKind.REF,root));return b


def fast(b,n=64,**options):return FieldOracle(Oracle(b,{'max_precision_bits':256,'max_subdivisions_per_item':64}),n,{'space_depth':3,'space_nodes':4096,**options})


def test_arb_endpoints_are_outward_checked_exactly():
    with ctx.workprec(128):
        for v in (arb(.1),arb(-.1).sin(),arb(1).cos(),arb('1e-310'),arb(-1).exp(),arb(0)):
            lo,hi=enclosure(v)
            assert Fraction.from_float(lo)<=exact_fraction(v.lower())
            assert Fraction.from_float(hi)>=exact_fraction(v.upper())


def test_certified_grid_and_cells_match_original_function_membership():
    f=fast(bundle());rng=np.random.default_rng(5)
    p=(rng.integers(0,64,(2048,3))+.5)/64
    expected=f.base.contains(p);assert np.array_equal(f.contains(p),expected)
    raw=f.grid.contains(p);assert np.array_equal(raw[raw>=0].astype(bool),expected[raw>=0])
    p=rng.uniform(.001,.999,(1024,3));raw=f.space.contains(p);ref=f.base.contains(p)
    assert np.array_equal(raw[raw>=0].astype(bool),ref[raw>=0])
    assert f.capability['ready_capable'] is False
    assert f.report()['approximate_geometry_used_as_truth'] is False


def test_subcell_thin_solid_is_not_missed_by_unknown_spatial_cells():
    f=fast(bundle(frequency=((1,0,0),),phase=(-.5,),amplitude=(1,),threshold=1-2**-40))
    source=np.array([.25,.4,.4]);p=np.array([[.75,.4,.4]])
    assert f.space.visibility(source,p)[0][0]==-1
    assert f.response(source,p).tolist()==[False]
    assert np.array_equal(f.response(source,p),f.base.response(source,p))
    assert f.work['exact_function_rays']+f.work['ray_proof_hits']>=1


def test_closed_ellipsoid_tangency_and_nextafter_keep_reference_fallback():
    ball=make_leaf('affine_ball_v1',3,center=[.5]*3,matrix=np.diag([.125]*3))
    f=fast(bundle(frequency=((0,0,0),),phase=(0,),amplitude=(0,),threshold=1,primitive=ball))
    source=np.array([.125,.625,.5]);p=np.array([[.875,.625,.5],[.875,np.nextafter(.625,1.),.5],[.875,np.nextafter(.625,0.),.5]])
    actual=f.response(source,p);expected=f.base.response(source,p)
    assert actual.tolist()==[False,True,False] and np.array_equal(actual,expected)
    assert f.work['composite_ray_fallbacks']>=1


def test_exact_regular_boundary_and_unresolved_ray_are_not_guessed():
    f=fast(bundle(frequency=((1,0,0),(0,1,0)),phase=(1,1),amplitude=(1,-1),threshold=0))
    p=np.array([[.5,.5,.5]]);assert f.grid.contains(p)[0]==-1 # not a sampled center, no interpolation
    assert f.contains(p).tolist()==[True] and f.base.contains(p).tolist()==[True]
    source=np.array([.75,.25,.5]);f.base.reference.max_subdivisions=8
    # Closed endpoint is a genuine boundary, but the existing segment prototype
    # cannot certify this endpoint-only contact within its declared budget.
    with pytest.raises(NumericalUnresolved):f.base.response(source,p)
    with pytest.raises(NumericalUnresolved):f.response(source,p)


def test_domain_face_unknown_is_preserved_and_budget_is_explicit():
    f=fast(bundle(frequency=((1,0,0),(0,1,0)),phase=(1,1),amplitude=(1,-1),threshold=0))
    p=np.array([[0.,0.,.5]])
    with pytest.raises(NumericalUnresolved):f.contains(p)
    with pytest.raises(NumericalUnresolved):f.base.contains(p)
    with pytest.raises(BudgetExhausted):f.response([.8,.2,.5],[[.7,.1,.5]],budget=Budget(30,0))


def test_convex_interior_occluder_proves_all_rays_and_both_fields():
    f=fast(bundle(frequency=((8,0,0),),phase=(-4,),amplitude=(1,),threshold=.5))
    source=np.array([.1,.51,.51]);lo=np.array([.8,.56,.56]);hi=np.array([.9,.57,.57])
    assert f.certificate(source,lo,hi)==(0,0)
    p=np.array(list(product(*zip(lo,hi))))
    assert not f.base.response(source,p).any() and not f.base.contains(p).any()
    lo=np.array([.12,.51,.51]);hi=np.array([.13,.52,.52]);assert f.certificate(source,lo,hi)==(1,0)
    assert f.base.response(source,np.array(list(product(*zip(lo,hi))))).all()


def test_membership_cache_checks_contents_instead_of_array_identity():
    f=fast(bundle(frequency=((8,0,0),),phase=(-4,),amplitude=(1,),threshold=.5))
    p=np.array([[.1,.5,.5]]);assert not f.contains(p)[0];assert not f.contains(p)[0]
    p[0,0]=.5;assert f.contains(p)[0]
    assert f.work['membership_batch_cache_hits']>=1


def test_unsupported_and_simple_geometry_use_original_evaluator():
    from rind3d.geometry import generate_geometry
    base=Oracle(generate_geometry(1,0,['smooth_blobs']))
    assert field_evaluator(base,32) is base


def test_stream_integration_keeps_original_pack_index_and_publication_status(tmp_path):
    from rind3d.stream import stream_response
    from rind3d.pack import RINDPack
    from rind3d.information import InformationIndex
    b=bundle(frequency=((8,0,0),),phase=(-4,),amplitude=(1,),threshold=.5)
    source=np.array([.1,.51,.51]);n=32
    cfg={'resolution':n,'generation_seconds':30,'generation_points':10_000_000,'workers':2,'queue_size':4,'leaf_side':8,
         'field_acceleration':'certified_function','field_acceleration_settings':{'space_depth':3,'space_nodes':4096}}
    base=Oracle(b,{'max_precision_bits':256,'max_subdivisions_per_item':4096});result=stream_response(base,source,tmp_path,cfg)
    assert result['complete'] and result['field_acceleration']['ready_capable'] is False
    with RINDPack(tmp_path/'response.rindb') as r,RINDPack(tmp_path/'solid.rindb') as m:
        index=InformationIndex.load(tmp_path/'information.rindi',r,m)
        assert index.global_flags==3
        p=centres((0,0,0),n,n);assert np.array_equal(r.read_window(0,0,0,n).ravel(),base.response(source,p))
        assert np.array_equal(m.read_window(0,0,0,n).ravel(),base.contains(p))
    # Finite sampled-field completion does not change the canonical capability.
    assert base.capability['ready_capable'] is False
