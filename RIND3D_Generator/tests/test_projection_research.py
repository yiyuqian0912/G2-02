import numpy as np
import pytest
from fractions import Fraction as Q
from tests.test_lite_production import box
from rind3d.rendering.context import GeometryContext
from rind3d.lite_generators import FAMILIES,generate
from rind3d.lite_acceptance import sample_source
from rind3d.lite_certification import small_field
from rind3d.projection_research import ProjectionSource,topology,silhouette_planes
from rind3d import _research
from rind3d.rendering.polyhedra import rational_rows

def test_analytic_plane_enclosure_is_rigorous_not_a_tolerance():
    rng=np.random.default_rng(617)
    rows=[tuple(Q(float(x)) for x in row) for row in rng.uniform(-1,1,(256,4))]
    rows.extend([(Q(1,3),Q(-1,7),Q(1,11),Q(-1,2)),(Q(2)**-1074,Q(0),Q(0),-Q(2)**-1074),(Q(100),Q(-7),Q(0),Q(-2))])
    for row in rows:
        a=rng.random(3);b=rng.random(3);lo=np.minimum(a,b);hi=np.maximum(a,b);l,h=_research.projection_interval(list(map(str,row)),lo,hi)
        exact_lo=row[3]+sum(row[k]*Q(float(lo[k] if row[k]>=0 else hi[k])) for k in range(3))
        exact_hi=row[3]+sum(row[k]*Q(float(hi[k] if row[k]>=0 else lo[k])) for k in range(3))
        assert Q(l)<=exact_lo<=exact_hi<=Q(h)

@pytest.mark.parametrize('family',FAMILIES)
def test_b2_complete_fields_all_families(family):
    ctx=GeometryContext(generate(20261031,FAMILIES.index(family),family)[0]);p,_=sample_source(ctx,20261031,FAMILIES.index(family));b=ProjectionSource(ctx,p)
    for n in (32,64,128):
        a,m=small_field(ctx.bind(p,independent=True),n);y,s=small_field(b,n);clip,solid=small_field(ctx.bind(p),n)
        assert np.array_equal(a,y) and np.array_equal(m,s)
        assert np.array_equal(clip,y) and np.array_equal(solid,s)
    assert sum(b.plane_report['retained_planes'])<=sum(b.plane_report['original_fme_planes'])

def test_projection_contacts_and_unaligned_regions():
    ctx=GeometryContext(box());p=np.array([.0625,.5,.5]);b=ProjectionSource(ctx,p)
    x=np.array([[.375,.5,.5],[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1),.5],[.375,.375,.375]])
    assert np.array_equal(b.response(p,x),ctx.bind(p,independent=True).response(p,x))
    assert b.response(p,x)[:4].tolist()==[False,False,False,True]
    for o,s in [((3,7,13),51),((0,0,0),64),((64,64,64),64)]:
        a,m,_=ctx.bind(p).render(o,s,128);y,z,_=b.render(o,s,128);assert np.array_equal(a,y) and np.array_equal(m,z)
    with pytest.raises(ValueError):b.certificate(p,[-.01,0,0],[.1,.1,.1])
    with pytest.raises(ValueError):ProjectionSource(ctx,[.5,.5,.5])

def test_thin_and_nearly_coplanar():
    for half in ((.2,.2,2**-40),(.125,.125,.125)):
        ctx=GeometryContext(box(half=half));p=np.array([.1,.1,.5]);b=ProjectionSource(ctx,p)
        x=np.random.default_rng(2).random((4096,3));x[:4]=[[.5,.5,.5],[.5,.5,.5+2**-40],[.5,.5,np.nextafter(.5+2**-40,1)],[.5,.7,.5]]
        assert np.array_equal(b.response(p,x),ctx.bind(p,independent=True).response(p,x))

def test_exact_first_hit_and_external_counterfactual():
    # Free receiver behind a box wholly outside W => externally necessary.
    ctx=GeometryContext(box((.375,.5,.5),(.0625,.125,.125)));p=[.0625,.5,.5]
    m=_research.Nonlocal(rational_rows(ctx.bodies),p)
    r=m.analyze(np.array([[.875,.5,.5]]),[768,384,384],256,1024)[0]
    assert r['first_location']=='OUTSIDE' and r['external_necessary'] and r['local_blocking']=='none'
    # Same ray, window contains the obstacle: first hit inside.
    r=m.analyze(np.array([[.49,.5,.5]]),[256,384,384],256,1024)[0]
    assert r['first_location']=='INSIDE' and not r['external_necessary']
    # Exact first contact on W lower face must remain a boundary case.
    r=m.analyze(np.array([[.49,.5,.5]]),[320,384,384],256,1024)[0]
    assert r['first_location']=='BOUNDARY' and not r['external_necessary']
    with pytest.raises(ValueError,match='FREE'):m.analyze(np.array([[.375,.5,.5]]),[256,384,384],256,1024)
    with pytest.raises(ValueError,match='half-open'):m.analyze(np.array([[.5,.5,.5]]),[256,384,384],256,1024)

def test_first_external_blocker_does_not_prove_external_necessity():
    from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind
    from rind.storage import CanonicalBundle
    a=box((.375,.5,.5),(.0625,.125,.125));b=box((.8125,.5,.5),(.015625,.125,.125));bundle=CanonicalBundle(3)
    leaves=[a.references[0],b.references[0]]
    for leaf in leaves:bundle.add_leaf(leaf)
    node=CSGNode(3,CSGOpcode.UNION,tuple(v.content_hash for v in leaves));bundle.add_node(node);bundle.add_geometry(Geometry(3,RootKind.REF,node.content_hash))
    ctx=GeometryContext(bundle);model=_research.Nonlocal(rational_rows(ctx.bodies),[.0625,.5,.5])
    r=model.analyze(np.array([[.9375,.5,.5]]),[768,384,384],256,1024)[0]
    assert r['first_location']=='OUTSIDE' and r['local_blocking']=='inside' and not r['external_necessary']
