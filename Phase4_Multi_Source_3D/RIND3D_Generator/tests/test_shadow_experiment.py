"""Isolated shadow proof tests. Optional extension is built by the experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
pytest.importorskip('experiments.shadow_volumes._shadow_native')
from experiments.shadow_volumes.renderer import extract,shadow_planes,SharedGeometry,geometry_split
from experiments.shadow_volumes.cases import make_case,SOURCES
from rind3d.geometry import Oracle,centres


@pytest.fixture(scope='module')
def box():
    bundle,_=make_case('box');base=Oracle(bundle);return SharedGeometry(base)


def test_shadow_matches_independent_clipping_on_random_points(box):
    for source in SOURCES:
        a=box.bind(source);p=np.random.default_rng(3).uniform(0,1,(1000,3))
        y,m,_=a.native.points(p,False);ry,rm,_=a.native.points(p,True)
        assert np.array_equal(y[y>=0],ry[y>=0]) and np.array_equal(m[m>=0],rm[m>=0])


def test_closed_face_edge_vertex_tangency_and_nearby(box):
    source=np.array([.125,.5,.5]);a=box.bind(source)
    # Exact front-face contact, silhouette edge, vertex, and nextafter sides.
    p=np.array([[.375,.5,.5],[.625,.75,.5],[.625,.75,.75],[.1,.5,.5]])
    p=np.r_[p,[[.625,np.nextafter(.75,1.),.5],[.625,np.nextafter(.75,0.),.5]]]
    r=box.base.reference.segment_intersect(np.repeat(source[None],len(p),axis=0),p)
    assert r.valid.all() and np.array_equal(a.response(source,p),r.value==0)


def test_source_on_extended_face_and_source_contact(box):
    s=np.array([.125,.625,.5]);a=box.bind(s)
    p=np.array([[.8,.625,.5],[.8,np.nextafter(.625,1.),.5]])
    r=box.base.reference.segment_intersect(np.repeat(s[None],len(p),axis=0),p)
    assert r.valid.all() and np.array_equal(a.response(s,p),r.value==0)
    with pytest.raises(ValueError):box.bind([.375,.5,.5])


def test_arbitrary_unaligned_region_native_vs_independent(box):
    a=box.bind(SOURCES[0]);origin=(7,11,3);side=33;n=64
    y,m,_=a.render(SOURCES[0],origin,side,n)
    ry,rm,_=a.native.points(centres(origin,side,n),True)
    assert np.array_equal(y.ravel(),ry) and np.array_equal(m.ravel(),rm)


def test_original_batching_control_same_outputs(box):
    a=box.bind(SOURCES[0]);y,m,_=a.render(SOURCES[0],(0,0,0),32,32)
    b,c,_=a.render(SOURCES[0],(0,0,0),32,32,old_mode=True)
    assert np.array_equal(y,b) and np.array_equal(m,c)


def test_cache_source_keys_and_grouped_splits(box):
    a,b=[box.bind(s) for s in SOURCES[:2]]
    assert a.source_id!=b.source_id and a.shared.native is b.shared.native
    assert geometry_split(a.shared.geometry_id)==geometry_split(b.shared.geometry_id)
    with pytest.raises(ValueError):a.render(SOURCES[1],(0,0,0),4,32)
    source=SOURCES[0].copy();frozen=box.bind(source);expected=frozen.source.copy();source[:]=SOURCES[1]
    assert np.array_equal(frozen.source,expected) and not frozen.source.flags.writeable
    with pytest.raises(ValueError):frozen.render(source,(0,0,0),4,32)


def test_union_corners_do_not_hide_a_lit_gap():
    from rind.contracts import make_leaf,Geometry,RootKind,CSGNode,CSGOpcode
    from rind.storage import CanonicalBundle
    bundle=CanonicalBundle(3);ids=[]
    for y in (.35,.65):
        leaf=make_leaf('affine_box_v1',3,center=[.4,y,.5],matrix=np.diag([.05,.07,.3]));bundle.add_leaf(leaf);ids.append(leaf.content_hash)
    node=CSGNode(3,CSGOpcode.UNION,tuple(ids));bundle.add_node(node);bundle.add_geometry(Geometry(3,RootKind.REF,node.content_hash))
    a=SharedGeometry(Oracle(bundle)).bind([.1,.5,.5]);assert a.certificate([.1,.5,.5],[.7,.1,.45],[.9,.9,.55])[0]==-1
    assert a.response([.1,.5,.5],[[.8,.5,.5]])[0]


def test_shared_mask_identity_and_outputs(box,tmp_path):
    from experiments.shadow_volumes.renderer import native_stream,SharedSolid
    from rind3d.stream import stream_response
    a=box.bind(SOURCES[0]);cfg={'resolution':32,'workers':1,'queue_size':1,'leaf_side':8,'generation_seconds':30,'generation_points':1000000,'field_acceleration':'baseline','build_information_index':False}
    with native_stream():stream_response(a,SOURCES[0],tmp_path,cfg)
    mask=SharedSolid(box,SOURCES[0],tmp_path/'solid.rindb')
    try:
        reused=box.bind(SOURCES[1],solid=mask);plain=box.bind(SOURCES[1]);y,m,_=reused.render(SOURCES[1],(0,0,0),32,32);ry,rm,_=plain.render(SOURCES[1],(0,0,0),32,32)
        assert np.array_equal(y,ry) and np.array_equal(m,rm) and reused.work['solid_points']==0
        with pytest.raises(ValueError):SharedSolid(box,SOURCES[1],tmp_path/'solid.rindb')
    finally:mask.close()


def test_world_face_only_component_not_mistaken_for_canonical_solid():
    from rind.contracts import make_leaf,Geometry,RootKind
    from rind.storage import CanonicalBundle
    from rind3d.geometry import Unsupported
    from experiments.shadow_volumes.renderer import optional_adapter
    b=CanonicalBundle(3);leaf=make_leaf('affine_box_v1',3,center=[1.125,.5,.5],matrix=np.diag([.125]*3));b.add_leaf(leaf);b.add_geometry(Geometry(3,RootKind.REF,leaf.content_hash))
    base=Oracle(b)
    with pytest.raises(Unsupported):SharedGeometry(base)
    assert optional_adapter(base,SOURCES[0]) is base


@pytest.mark.parametrize('family',['rotated_box','thin_slab','prism','wedge','poly8'])
def test_new_families_canonical_point_checks(family):
    bundle,_=make_case(family);base=Oracle(bundle);a=SharedGeometry(base).bind(SOURCES[0])
    p=np.random.default_rng(11).uniform(.1,.9,(8,3));r=base.reference.segment_intersect(np.repeat(SOURCES[:1],len(p),axis=0),p)
    m=base.reference.classify_points(p)
    assert r.valid.all() and m.valid.all()
    assert np.array_equal(a.response(SOURCES[0],p),r.value==0)
    assert np.array_equal(a.contains(p),m.value!=0)
