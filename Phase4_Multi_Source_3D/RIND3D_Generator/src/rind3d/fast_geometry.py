"""Versioned complexity mixture; selection depends on parameters, never timings.

This declared distribution takes the first <=8 eligible components of an
original randomized recipe, excluding polyhedra with >12 declared planes.
It changes geometry distribution and identity explicitly; hard cases remain in
the frozen baseline. The original canonical geometry acceptance still applies.
"""
from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind
from rind.storage import CanonicalBundle
from .lite_generators import generate as original_generate

VERSION='lite-fast-convex-mixture-v1'
FAMILIES=('thin_slabs','rotated_boxes','barriers','corridors','multiscale_convex','occluder_clusters')

def generate(seed,slot,family,attempt=0):
    original,recipe=original_generate(seed,slot,family,attempt)
    eligible=[o for o in recipe['objects'] if o['declared_planes']<=12];objects=eligible[:8]
    if not objects:raise ValueError('Declared fast geometry filter left no components')
    refs={r.content_hash.hex():r for r in original.references};b=CanonicalBundle(3);roots=[]
    for o in objects:
        leaf=refs[o['leaf_id']];b.add_leaf(leaf);roots.append(leaf.content_hash)
    if len(roots)>1:
        node=CSGNode(3,CSGOpcode.UNION,tuple(roots));b.add_node(node);root=node.content_hash
    else:root=roots[0]
    b.add_geometry(Geometry(3,RootKind.REF,root))
    recipe=dict(recipe,version=VERSION,parent_geometry_id=recipe['geometry_id'],geometry_id=b.geometries[0].content_hash.hex(),objects=objects,
                declared_component_cap=8,declared_plane_cap=12,removed_component_ids=[o['leaf_id'] for o in recipe['objects'] if o not in objects],
                no_render_time_selection=True)
    return b,recipe
