"""Curved-occluder research families (spheres first). Deterministic from (seed, slot, attempt) keyed streams.

* `spheres`: 3-10 balls, radius ~ log-uniform(0.02, 0.12) (20-123 voxels at N = 1024), centres uniform in the
  world-fitting interval [r + 0.01, 1 - r - 0.01]^3. No polytope.
* `spheres_polyhedra`: the v3 `convex_polyhedra` recipe of the same (seed, slot, attempt) (1-5 convex polytopes,
  rind3d_100k.geometry_v3) plus 2-6 balls from an independent stream.
Balls may overlap each other and polytopes (union semantics). Every ball must be strictly inside the open world and
have a nonempty 6-connected voxel-centre sampling (checked exactly, `ball_sampled_components`)."""
import numpy as np

from rind3d.geometry import keyed_rng
from rind3d.lite_generators import log_uniform

VERSION = "curved-spheres-v0"
CURVED_FAMILIES = ("spheres", "spheres_polyhedra")
STREAM = 9101


def generate_curved(seed, slot, family, attempt=0, thin_min_vox=2.0):
    """Returns (polytope bundle or None, balls float64 (n, 4) [cx, cy, cz, r], recipe)."""
    if family not in CURVED_FAMILIES:
        raise ValueError(family)
    rng = keyed_rng(seed, slot, STREAM + 97 * attempt)
    bundle, prec = None, None
    if family == "spheres":
        n = int(rng.integers(3, 11))
    else:
        from rind3d_100k.geometry_v3 import generate_v3
        bundle, prec = generate_v3(seed, slot, "convex_polyhedra", attempt, thin_min_vox=thin_min_vox)
        n = int(rng.integers(2, 7))
    r = log_uniform(rng, .02, .12, n)
    c = np.stack([rng.uniform(ri + .01, 1 - ri - .01, 3) for ri in r])
    balls = np.c_[c, r].astype("<f8")
    rec = dict(version=VERSION, family=family, seed=int(seed), slot=int(slot), attempt=int(attempt), balls=balls.tolist(),
               polytopes=(prec["objects"] if prec else []), proposal_failures=(prec["proposal_failures"] if prec else []),
               polytope_geometry_id=(prec["geometry_id"] if prec else None))
    return bundle, balls, rec


def ball_sampled_components(c, r, N=1024):
    """Voxel-centre sampling {v : |(v + 1/2)/N - c| <= r} of a closed ball: (n_vox, n_components) with 6-adjacency.
    Columns along z are runs (convexity); 4-adjacent columns are 6-connected iff their runs overlap. Run ends near a
    float bound are re-decided with exact rationals."""
    from fractions import Fraction
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cq = [Fraction(float(v)) for v in c]; rq = Fraction(float(r))
    lo = np.clip(np.floor((np.asarray(c) - r) * N - 1), 0, N - 1).astype(int); hi = np.clip(np.ceil((np.asarray(c) + r) * N + 1), 0, N - 1).astype(int)
    I, J = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij"); I = I.ravel(); J = J.ravel()
    X = (I + .5) / N; Y = (J + .5) / N
    rem = r * r - (X - c[0]) ** 2 - (Y - c[1]) ** 2
    ok = rem >= -1e-12
    h = np.sqrt(np.maximum(rem, 0))
    zlo = np.ceil((c[2] - h) * N - .5 - 1e-9).astype(np.int64); zhi = np.floor((c[2] + h) * N - .5 + 1e-9).astype(np.int64)
    inside = lambda i, j, k: sum((Fraction(2 * v + 1, 2 * N) - cq[t]) ** 2 for t, v in enumerate((i, j, k))) <= rq * rq
    for idx in np.flatnonzero(ok):
        a, b = int(zlo[idx]), int(zhi[idx]); i, j = int(I[idx]), int(J[idx])
        while a <= b and not inside(i, j, a):
            a += 1
        while b >= a and not inside(i, j, b):
            b -= 1
        if a <= b:
            if inside(i, j, a - 1):
                a -= 1
            if inside(i, j, b + 1):
                b += 1
        zlo[idx], zhi[idx] = a, b
    good = ok & (zlo <= zhi)
    n_vox = int(np.sum((zhi - zlo + 1)[good]))
    if not n_vox:
        return 0, 0
    nI, nJ = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
    ix = np.full(nI * nJ, -1, np.int64); g = np.flatnonzero(good); ix[g] = np.arange(len(g))
    rr, cc = [], []
    a = np.arange(nI * nJ).reshape(nI, nJ)
    for di, dj in ((1, 0), (0, 1)):
        p = a[:nI - di, :nJ - dj].ravel(); q = a[di:, dj:].ravel()
        e = good[p] & good[q] & (np.maximum(zlo[p], zlo[q]) <= np.minimum(zhi[p], zhi[q]))
        rr.append(ix[p[e]]); cc.append(ix[q[e]])
    rr = np.concatenate(rr); cc = np.concatenate(cc)
    ncomp, _ = connected_components(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(g), len(g))), directed=False)
    return n_vox, int(ncomp)
