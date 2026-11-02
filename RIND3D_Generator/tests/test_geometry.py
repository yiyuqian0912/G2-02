import numpy as np
import pytest
from rind3d.geometry import Oracle,generate_geometry,save_geometry,load_geometry,render_region,centres,Budget,Aborted

def test_lossless_geometry_and_native_certificates(tmp_path):
    b=generate_geometry(7,0,['csg']);p=tmp_path/'geometry.npz';save_geometry(p,b);b2=load_geometry(p)
    assert b2.geometries[0].content_hash==b.geometries[0].content_hash
    a=Oracle(b);s=np.array([.03,.07,.11]);a.validate_source(s)
    rng=np.random.default_rng(17);q=rng.uniform(0,1,(16,3))
    fast=a.response(s,q)
    reference=a.reference.segment_intersect(np.repeat(s[None],len(q),axis=0),q)
    assert reference.valid.all()
    assert np.array_equal(fast,reference.value==0)
    y,m,_=render_region(a,s,(0,0,0),16,64)
    direct=a.response(s,centres((0,0,0),16,64)).reshape((16,)*3)
    assert np.array_equal(y,direct) and not np.any(y&m)

def test_budget_is_aborted_not_unresolved():
    b=Budget(seconds=1,max_points=0)
    with pytest.raises(Aborted):b.check(1)

def test_convex_shadow_certificate_covers_all_points():
    from rind3d._native import IntervalFilter
    f=IntervalFilter([(0,[.5,.5,.5],[.05,.2,.2])])
    source=[.1,.5,.5];lo=[.7,.45,.45];hi=[.9,.55,.55]
    assert f.certificate(source,lo,hi)[0]==0
    q=np.random.default_rng(0).uniform(lo,hi,(1000,3))
    assert not f.visibility(np.array([source]),q).any()


def test_closed_boundaries_tangency_and_nextafter_fallback():
    from rind3d.benchmark import fixture_box
    oracle=Oracle(fixture_box());source=np.array([.125,.5,.5])
    tangent=np.array([.5,.875,.5])
    points=np.stack([tangent,tangent.copy(),tangent.copy(),[.375,.5,.5]])
    points[1,1]=np.nextafter(.875,1.);points[2,1]=np.nextafter(.875,0.)
    actual=oracle.response(source,points)
    reference=oracle.reference.segment_intersect(np.repeat(source[None],len(points),axis=0),points)
    assert reference.valid.all() and np.array_equal(actual,reference.value==0)
    assert actual.tolist()==[False,True,False,False]
    assert oracle.metrics['fallback_points']>0  # exact tangency is not guessed
    boundary=np.array([[.375,.5,.5],[np.nextafter(.375,0.),.5,.5],[np.nextafter(.375,1.),.5,.5]])
    assert oracle.contains(boundary).tolist()==[True,False,True]
    with pytest.raises(ValueError,match='certified free'):oracle.validate_source(boundary[0])


def test_ellipsoid_tangency_preserves_exact_reference():
    from rind.contracts import Geometry,RootKind,make_leaf
    from rind.storage import CanonicalBundle
    bundle=CanonicalBundle(3);leaf=make_leaf('affine_ball_v1',3,center=[.5]*3,matrix=np.diag([.125]*3))
    bundle.add_leaf(leaf);bundle.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash));oracle=Oracle(bundle)
    source=np.array([.125,.625,.5]);points=np.array([[.875,.625,.5],[.875,np.nextafter(.625,1.),.5],[.875,np.nextafter(.625,0.),.5]])
    actual=oracle.response(source,points)
    reference=oracle.reference.segment_intersect(np.repeat(source[None],len(points),axis=0),points)
    assert reference.valid.all() and np.array_equal(actual,reference.value==0)
    assert actual.tolist()==[False,True,False]
