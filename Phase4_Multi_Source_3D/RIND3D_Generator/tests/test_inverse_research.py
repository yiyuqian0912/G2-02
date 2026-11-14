import numpy as np
import pytest
from tests.test_lite_production import box
from rind3d.rendering.context import GeometryContext
from rind3d.windows import Window,excludes
from rind3d.inverse_research import InverseConstraints,explore,local_proposals
from rind3d.lite_generators import generate
from rind3d.lite_acceptance import sample_source

def fixture():
    ctx=GeometryContext(box());truth=np.array([.1,.2,.3]);x=np.array([[.8,.5,.5],[.8,.9,.1],[.8,.55,.6],[.7,.1,.8]])
    labels=ctx.bind(truth).response(truth,x);return ctx,truth,x,labels

def test_inverse_symmetry_union_complements_and_contacts():
    ctx,p,x,labels=fixture();constraints=InverseConstraints(ctx,x,labels);q=np.random.default_rng(81).uniform(.01,.99,(64,3))
    expected=[]
    for s in q:
        expected.append(False if ctx.contains(s[None])[0] else np.array_equal(ctx.bind(s).response(s,x),labels))
    assert np.array_equal(constraints.screen(q),expected)
    with pytest.raises(ValueError,match='FREE'):InverseConstraints(ctx,[[.5,.5,.5]],[False])

def test_outer_cover_never_drops_matching_exhaustive_grid_points():
    ctx,p,x,labels=fixture();constraints=InverseConstraints(ctx,x,labels);w=Window((768,384,384),128)
    _,r=explore(constraints,p,w,seconds=10,max_regions=73,max_depth=3,separation=0)
    grid=np.array(np.meshgrid(*([np.linspace(.03125,.96875,9)]*3),indexing='ij')).reshape(3,-1).T
    good=grid[constraints.screen(grid)]
    for q in good:
        if not excludes(w,q,1024):continue
        assert any(np.all(q>=b['lower']) and np.all(q<=b['upper']) for b in r['outer_cells'])
    assert 0<=r['outer_cover_volume_upper_bound']<=1 and not r['uniqueness_proven']

def test_budget_unknown_cells_retained_and_deterministic():
    ctx,p,x,y=fixture();c=InverseConstraints(ctx,x,y);w=Window((768,384,384),128)
    a,r=explore(c,p,w,seconds=0,max_regions=1)
    assert r['outer_cover_volume_upper_bound']==1 and r['pending_unresolved_cells']==1
    kw=dict(seconds=10,max_regions=80,max_depth=2,seed=4)
    a,r=explore(c,p,w,**kw);b,s=explore(c,p,w,**kw)
    assert np.array_equal(a,b) and r['outer_cells']==s['outer_cells']

def test_finite_labels_allow_local_perturbations_without_uniqueness():
    ctx,p,x,y=fixture();near=p+np.array([[1e-9,0,0],[-1e-9,0,0],[0,1e-9,0]])
    assert all(np.array_equal(ctx.bind(q).response(q,x),y) for q in near)
    assert all(np.linalg.norm(q-p)<32/1024 for q in near)
    q=local_proposals(p,seed=3);assert np.all(np.linalg.norm(q-p,axis=1)>=32/1024-1e-16)

def test_one_two_three_independent_or_dependent_inverse_planes_are_not_equalities():
    """Actual inverse-polyhedron normals can span 3D with nonunique sources.

    These are exact planes in the known inverse geometry, not observed exact
    boundary equations. Strict finite binary classifications impose sides of
    those planes. Rank alone must never be used as an identifiability test.
    """
    from fractions import Fraction as Q
    from rind3d.projection_research import value
    ctx,p,x,labels=fixture();inverse=InverseConstraints(ctx,x,labels)
    truth=tuple(Q(float(v)) for v in p)
    normals=[tuple(row[:3]) for model in inverse.models for body in model.planes
             for row in body if value(row,truth)!=0 and any(row[:3])]
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    a=normals[0];b=next(v for v in normals if any(cross(a,v)))
    c=next(v for v in normals if sum(i*j for i,j in zip(cross(a,b),v))!=0)
    dependent=tuple(i+j for i,j in zip(a,b))
    assert sum(i*j for i,j in zip(cross(a,b),dependent))==0
    assert sum(i*j for i,j in zip(cross(a,b),c))!=0
    for rows,rank in (((a,),1),((a,b),2),((a,b,c),3),((a,b,dependent),2)):
        # Diagnostic rank has no authority to publish or reject a source.
        assert np.linalg.matrix_rank(np.asarray(rows,dtype=float))==rank
        for q in p+np.r_[np.eye(3),-np.eye(3)]*2**-35:
            assert inverse.screen(q[None])[0]
            for independent in (False,True):
                assert np.array_equal(ctx.bind(q,independent=independent).response(q,x),labels)

@pytest.mark.parametrize('family',['thin_slabs','barriers','multiscale_convex'])
def test_multiple_and_rotated_inverse_cones_against_independent_clipping(family):
    ctx=GeometryContext(generate(718,2,family)[0]);truth,_=sample_source(ctx,718,2);rng=np.random.default_rng(64)
    x=rng.uniform(.01,.99,(64,3));x=x[~ctx.contains(x)][:16];labels=ctx.bind(truth).response(truth,x)
    inverse=InverseConstraints(ctx,x,labels);p=rng.uniform(.01,.99,(32,3));expected=[]
    for q in p:expected.append(False if ctx.contains(q[None])[0] else np.array_equal(ctx.bind(q).response(q,x),labels))
    assert np.array_equal(inverse.screen(p),expected)

def test_disconnected_inverse_union_is_not_convexified():
    from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind
    from rind.storage import CanonicalBundle
    parts=[box((x,.5,.5),(.015625,.015625,.08)) for x in (.3125,.6875)];bundle=CanonicalBundle(3);leaves=[p.references[0] for p in parts]
    for leaf in leaves:bundle.add_leaf(leaf)
    node=CSGNode(3,CSGOpcode.UNION,tuple(v.content_hash for v in leaves));bundle.add_node(node);bundle.add_geometry(Geometry(3,RootKind.REF,node.content_hash));ctx=GeometryContext(bundle)
    # A dark receiver between two occluders has two disjoint inverse cones.
    # A nearby lit receiver also agrees with both sources. The separating slab
    # is certified visible at the dark receiver, proving this is a union rather
    # than one convex feasible set. These are sparse constraints, not a claim
    # about an entire production observation window.
    x=np.array([[.5,.5,.5],[.5,.57,.5]]);labels=np.array([False,True]);inverse=InverseConstraints(ctx,x,labels)
    p=np.array([[.1,.5,.5],[.9,.5,.5]]);assert inverse.screen(p).tolist()==[True,True]
    assert not inverse.possible(np.array([.4,.01,.01]),np.array([.6,.99,.99]))
    for q in p:assert np.array_equal(ctx.bind(q).response(q,x),labels)
