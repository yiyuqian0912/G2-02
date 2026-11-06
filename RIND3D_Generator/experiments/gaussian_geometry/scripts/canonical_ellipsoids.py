"""Lossless B1 export using the EXISTING affine_ball_v1 canonical codec.

No sum-field mesh approximation, no new production registration or certificate.
"""
from rind.contracts import CSGNode,CSGOpcode,Geometry,RootKind,make_leaf
from rind.storage import CanonicalBundle,pack_arrays,_decode_bundle


def to_canonical(scene):
    if scene.kind!='ellipsoid_union':
        raise ValueError('A thresholded Gaussian sum/GRF cannot be replaced by an ellipsoid union')
    bundle=CanonicalBundle(3);roots=[]
    for k in scene.kernels:
        leaf=make_leaf('affine_ball_v1',3,center=k.center,matrix=k.matrix)
        bundle.add_leaf(leaf);roots.append(leaf.content_hash)
    if len(roots)>1:
        node=CSGNode(3,CSGOpcode.UNION,tuple(roots));bundle.add_node(node);root=node.content_hash
    else:root=roots[0]
    bundle.add_geometry(Geometry(3,RootKind.REF,root))
    return _decode_bundle(pack_arrays(bundle),3)
