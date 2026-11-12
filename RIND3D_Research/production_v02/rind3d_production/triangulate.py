"""Direct face triangulation of canonical convex components (acceleration representation only).

The canonical geometry is a top-level union of convex components, each an affine box (centre c, matrix M:
x = c + M u, |u|_inf <= 1) or an affine convex polytope ({u : N u <= b}). Triangulation needs no voxels and
no marching cubes: boxes give 8 vertices / 12 triangles directly; polytopes are enumerated in their local
frame (scipy halfspace intersection + convex hull, well conditioned because u = O(1)) and mapped to world.

What this representation is and is not:
- vertex coordinates are binary64 then float32 roundings of irrational/rational exact vertices, so the
  mesh is a perturbation of the canonical solid (reported: max plane residual of every vertex);
- each component is a closed, consistently oriented triangle surface, but components overlap: the union
  of surfaces is NOT the boundary of the union. Any-hit occlusion of a segment between two FREE points is
  still the union predicate (a segment that meets a closed convex body with both endpoints outside it
  meets its boundary), but inside/outside by ray parity is invalid for overlapping components; solid
  classification must use per-component membership (here: the certified native predicate).
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from rind.contracts import CSGNode, CodecID, Leaf

BOX_FACES = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                      [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]], np.int64)
BOX_SIGNS = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], np.float64)


def leaves(bundle):
    refs = {x.content_hash: x for x in bundle.references}
    out = []

    def walk(key):
        item = refs[key]
        if isinstance(item, CSGNode):
            for child in item.children:
                walk(child)
            return
        out.append(item)
    walk(bundle.geometries[0].root_hash)
    return out


def _orient(V, F, centre):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b-a, c-a)
    flip = np.einsum("ij,ij->i", n, (a+b+c)/3-centre) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def component_mesh(leaf):
    c = leaf.field("center").as_array().astype(np.float64)
    M = leaf.field("matrix").as_array().astype(np.float64)
    if leaf.codec == CodecID.AFFINE_BOX:
        U = BOX_SIGNS
        F = BOX_FACES
    else:
        N = leaf.field("normal").as_array().astype(np.float64)
        b = leaf.field("plane_offset").as_array().astype(np.float64)
        hs = HalfspaceIntersection(np.c_[N, -b], np.zeros(3))
        U = hs.intersections
        F = ConvexHull(U).simplices.astype(np.int64)
    V = c + U @ M.T
    F = _orient(V, F, c)
    return V, F


def triangulate(bundle):
    Vs, Fs, comp, offset = [], [], [], 0
    for k, leaf in enumerate(leaves(bundle)):
        V, F = component_mesh(leaf)
        Vs.append(V)
        Fs.append(F+offset)
        comp.append(np.full(len(F), k))
        offset += len(V)
    V64 = np.concatenate(Vs)
    F = np.concatenate(Fs).astype(np.int32)
    return {"vertices64": V64, "vertices32": V64.astype(np.float32), "faces": F, "component": np.concatenate(comp),
            "components": len(Vs)}


def residuals(context, mesh):
    """Max over vertices of the distance of the float32 vertex from its nearest bounding plane (world units)."""
    worst = 0.0
    V = mesh["vertices32"].astype(np.float64)
    comp_of_vertex = np.empty(len(V), np.int64)
    for k in range(mesh["components"]):
        comp_of_vertex[np.unique(mesh["faces"][mesh["component"] == k])] = k
    for k, comp in enumerate(context.components):
        rows = np.array([[float(x) for x in r] for r in comp.rows])
        n = rows[:, :3]
        nn = np.linalg.norm(n, axis=1)
        v = V[comp_of_vertex == k]
        d = (v @ n.T + rows[:, 3])/nn
        worst = max(worst, float(np.min(np.abs(d), axis=1).max()))
    return worst
