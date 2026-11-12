"""Statistical tier for broad posteriors (Direction B identity).

* **Label.** q_B = exact cell masses of the phase-1 outer bound B = F(X') ⊇ F (TPE engine on a receiver subset X').
* **Identity.** F ⊆ B and both posteriors are uniform, so TV(q_B, q_F) = 1 - vol(F)/vol(B) (continuous). Cell-level TV is
  smaller or equal.
* **Certificate.** n i.i.d. uniform points in B, each tested EXACTLY for membership in F:
  1. legal source: not in solid, certified clearance > 2^-26 (rind3d exact predicates);
  2. outside the closed window box;
  3. every FREE receiver of the window has the observed class for this source: certified CUDA rendering of the whole
     window, compared bit for bit.
  The certificate is issued only if all n points pass. If p = vol(F)/vol(B) < 1 - eps, then
  P(all pass) = p^n < (1 - eps)^n = delta. With eps = 0.05 and n = 90, delta < 0.0099. Otherwise the caller falls back
  to the exact phase 2. The fallback rule is fixed in advance and does not depend on the data in a way that breaks the
  bound: the certificate event is exactly {all n pass}.
* **Sampling.** Each IN piece of B is chosen with probability proportional to its exact volume. A point uniform in the
  convex piece is drawn by Delaunay tetrahedralisation of its vertices.
"""
import time

import numpy as np


def _tetrahedra(U):
    """Tetrahedra covering the convex hull of the points U (a convex piece): Delaunay; a single tetrahedron for 4 points;
    a centroid fan over the hull facets for degenerate inputs."""
    from scipy.spatial import Delaunay, ConvexHull
    if len(U) == 4:
        return U[None, :, :]
    try:
        return U[Delaunay(U, qhull_options="QJ Pp").simplices]
    except Exception:
        h = ConvexHull(U, qhull_options="QJ")
        c = U.mean(0)
        return np.stack([np.broadcast_to(c, (len(h.simplices), 3)), U[h.simplices[:, 0]], U[h.simplices[:, 1]], U[h.simplices[:, 2]]], 1)


def sample_in_B(res, n, rng, return_picks=False):
    """n uniform points in the union of phase-1 IN pieces (records with vertices). Returns (points (n,3), ok)."""
    from scipy.spatial import Delaunay
    vol = res["hi"].astype(float).copy()          # IN pieces (lo = hi) and UNDET pieces (upper bound) are all part of B
    voff, vn, vpool = res["voff"], res["vn"], res["vpool"]
    if np.any((voff < 0) | (vn < 4)):
        return (None, "piece without vertices (vertex pool full)", None) if return_picks else (None, "piece without vertices (vertex pool full)")
    tot = vol.sum()
    if not tot > 0:
        return (None, "empty B", None) if return_picks else (None, "empty B")
    picks = rng.choice(len(vol), size=n, p=vol / tot)
    out = np.empty((n, 3)); cache = {}
    for t, i in enumerate(picks):
        if i not in cache:
            V = vpool[voff[i]:voff[i] + vn[i]]
            ctr = V.mean(0); sc = max(float(np.max(np.abs(V - ctr))), 1e-300)
            U = (V - ctr) / sc                                       # well-conditioned for qhull
            T = _tetrahedra(U)
            vols = np.abs(np.einsum("ij,ij->i", T[:, 1] - T[:, 0], np.cross(T[:, 2] - T[:, 0], T[:, 3] - T[:, 0]))) / 6
            if not vols.sum() > 0:
                vols = np.ones(len(T))          # degenerate (zero-volume) piece: chosen with probability ~0 anyway
            cache[i] = (T, vols / vols.sum(), ctr, sc)
        T, pv, ctr, sc = cache[i]
        k = rng.choice(len(pv), p=pv)
        a = rng.random(3)
        # uniform barycentric in a tetrahedron (Rocchini-Cignoni folding)
        x, y, z = a
        if x + y > 1:
            x, y = 1 - x, 1 - y
        if y + z > 1:
            y, z = 1 - z, 1 - x - y
        elif x + y + z > 1:
            x, z = 1 - y - z, x + y + z - 1
        w0 = 1 - x - y - z
        out[t] = ctr + sc * (w0 * T[k, 0] + x * T[k, 1] + y * T[k, 2] + z * T[k, 3])
    if return_picks:
        return out, "ok", picks
    return out, "ok"


class ExactPointTester:
    """Exact membership of candidate sources s in F (legality, exclusion, every FREE receiver of the window)."""

    def __init__(self, bundle, policy, lit_zyx, solid_zyx, origin, N=1024):
        from rind3d.rendering.backends import BackendFactory
        self.fac = BackendFactory(cuda_batch=32)
        self.bundle = bundle; self.policy = policy; self.ctx = self.fac.context(bundle, policy)
        self.lit = lit_zyx; self.solid = solid_zyx; self.o = np.asarray(origin, np.int64); self.N = int(N)
        self.L = lit_zyx.shape[0]
        n = max(self.L // 64, 1)
        self.offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        self.origins = np.array([self.o + np.array(t) * 64 for t in self.offs], np.int32)
        # expected packed tiles
        self.exp = []
        for (a, b, c) in self.offs:
            sl = (slice(c * 64, (c + 1) * 64), slice(b * 64, (b + 1) * 64), slice(a * 64, (a + 1) * 64))
            self.exp.append((np.packbits(self.lit[sl].ravel(), bitorder="little"), np.packbits(self.solid[sl].ravel(), bitorder="little")))
        self.stats = dict(points=0, bind_s=0.0, render_s=0.0, legal_s=0.0, fail_legal=0, fail_box=0, fail_obs=0)

    def test(self, s):
        from rind3d_1k.physical import legal_source
        self.stats["points"] += 1
        lo = self.o / self.N; hi = (self.o + self.L) / self.N
        if np.all((s >= lo) & (s <= hi)):
            self.stats["fail_box"] += 1; return False
        t = time.perf_counter()
        ok = legal_source(self.ctx, s) is not None
        self.stats["legal_s"] += time.perf_counter() - t
        if not ok:
            self.stats["fail_legal"] += 1; return False
        t = time.perf_counter()
        src = self.fac.source(self.bundle, s, "cuda_certified", self.policy)
        self.stats["bind_s"] += time.perf_counter() - t
        try:
            t = time.perf_counter()
            for i in range(0, len(self.origins), 32):
                out = src.packed_batch(self.origins[i:i + 32], 64, self.N)
                for j, (pl, ps) in enumerate(out):
                    el, es = self.exp[i + j]
                    if not np.array_equal(np.frombuffer(ps, np.uint8), es):
                        raise AssertionError("SOLID mismatch: renderer and stored crop disagree")
                    if not np.array_equal(np.frombuffer(pl, np.uint8), el):
                        self.stats["render_s"] += time.perf_counter() - t
                        self.stats["fail_obs"] += 1
                        return False
            self.stats["render_s"] += time.perf_counter() - t
            return True
        finally:
            src.close()


def certify(res, tester, n=90, eps=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pts, why = sample_in_B(res, n, rng)
    if pts is None:
        return dict(ok=False, reason=why)
    for i, s in enumerate(pts):
        if not tester.test(s):
            return dict(ok=False, reason="sample infeasible", tested=i + 1)
    return dict(ok=True, n=n, eps=eps, delta=(1 - eps) ** n, tv_cert=eps)
