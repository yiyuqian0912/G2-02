"""Proposals (untrusted) for the facets of the exact possible shadow O = K + cone(K - P).

For a convex obstacle K (clipped to the world cube) and a convex source piece P with
disjoint interiors, the union of the shadows S_K(s) over s in P has closure
O = K + cone(K - P) (XLF_CBC_DESIGN.md, section 4d). Every facet of this Minkowski sum lies on
a plane that separates K and P and contains either an edge of K and a vertex of P, or an edge
of P and a vertex of K (or is a facet of K). Envelope supplies the facets of K and the
all-vertex pair rows; penumbra.directions supplies the facets of cone(K - P); the "mixed"
facets (edge of one, vertex of the other, not a cone facet) were missing, which left the
possible shadow loose wherever P straddles a facet plane of K.

This module only proposes integer rows: a binary64 screen picks the separating candidates,
then each row is built exactly from three homogeneous vertices. The trusted kernel validates
every proposed row exactly (max over K <= 0 <= min over P) before using it; a wrong or
missing proposal can only make the screen weaker, never unsound.
"""
import numpy as np

from poly import primitive


def _edges(poly):
    out = set()
    for f in poly.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            out.add((a, b) if a < b else (b, a))
    return sorted(out)


class Geometry:
    """Float view of one polytope (cached per obstacle; rebuilt per piece)."""

    def __init__(self, poly):
        self.verts = list(poly.vertices)
        index = {v: i for i, v in enumerate(self.verts)}
        self.f = np.array([[v[0] / v[3], v[1] / v[3], v[2] / v[3]] for v in self.verts])
        ed = _edges(poly)
        self.e = np.array([(index[a], index[b]) for a, b in ed], np.int64).reshape(-1, 2)


def _plane3(a, b, c):
    d1 = [b[i] * a[3] - a[i] * b[3] for i in range(3)]
    d2 = [c[i] * a[3] - a[i] * c[3] for i in range(3)]
    n = (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0])
    if not any(n):
        return None
    return primitive((n[0] * a[3], n[1] * a[3], n[2] * a[3], -(n[0] * a[0] + n[1] * a[1] + n[2] * a[2])))


def _separating(EA, EB, V, X, Y, tol=1e-9):
    """Planes through edge (EA[i], EB[i]) and point V[j]; True where they separate X (<= 0)
    from Y (>= 0) in either orientation, up to a relative tolerance (proposal only)."""
    if not len(EA) or not len(V):
        return np.zeros((len(EA), len(V)), bool), np.zeros((len(EA), len(V)), bool)
    d = EB - EA                                    # (E, 3)
    w = V[None, :, :] - EA[:, None, :]             # (E, nV, 3)
    n = np.cross(d[:, None, :], w)                 # (E, nV, 3)
    scale = np.linalg.norm(n, axis=2) * (np.ptp(np.vstack((X, Y)), axis=0).max() + 1e-300)
    hx = np.einsum("evk,xk->evx", n, X) - np.einsum("evk,ek->ev", n, EA)[:, :, None]
    hy = np.einsum("evk,yk->evy", n, Y) - np.einsum("evk,ek->ev", n, EA)[:, :, None]
    t = (tol * scale)[:, :, None]
    ok = scale > 0
    fwd = ok & (hx <= t).all(axis=2) & (hy >= -t).all(axis=2)
    bwd = ok & (hx >= -t).all(axis=2) & (hy <= t).all(axis=2)
    return fwd, bwd


def mixed_candidates(kgeo, pg):
    """Integer rows of candidate mixed facets (unvalidated, unoriented); pg = Geometry(piece)."""
    rows = []
    K, P = kgeo.f, pg.f
    if len(kgeo.e):
        sel = _separating(K[kgeo.e[:, 0]], K[kgeo.e[:, 1]], P, K, P)
        for i, j in zip(*np.nonzero(sel[0] | sel[1])):
            a, b = kgeo.verts[kgeo.e[i, 0]], kgeo.verts[kgeo.e[i, 1]]
            rows.append(_plane3(a, b, pg.verts[j]))
    if len(pg.e):
        sel = _separating(P[pg.e[:, 0]], P[pg.e[:, 1]], K, K, P)
        for i, j in zip(*np.nonzero(sel[0] | sel[1])):
            a, b = pg.verts[pg.e[i, 0]], pg.verts[pg.e[i, 1]]
            rows.append(_plane3(a, b, kgeo.verts[j]))
    return [r for r in dict.fromkeys(rows) if r is not None]
