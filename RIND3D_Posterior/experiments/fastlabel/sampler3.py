"""Uniform sampling in the phase-1 outer bound B (same law as stat_tier.sample_in_B: piece ∝ exact volume `hi`,
tetrahedron of the piece's Delaunay tetrahedralisation ∝ volume, Rocchini-Cignoni uniform barycentric), vectorised
per piece, with a per-label triangulation cache (pieces repeat across rounds) and lazy chunks."""
import numpy as np
from stat_tier import _tetrahedra


class BSampler:
    def __init__(self, res, rng, cache):
        self.res = res; self.rng = rng; self.cache = cache
        vol = res["hi"].astype(float)
        voff, vn = res["voff"], res["vn"]
        self.ok = "ok"
        if np.any((voff < 0) | (vn < 4)):
            self.ok = "piece without vertices (vertex pool full)"
        tot = vol.sum()
        if not tot > 0:
            self.ok = "empty B"
        self.p = vol / tot if tot > 0 else None

    def _tri(self, i):
        res = self.res
        V = res["vpool"][res["voff"][i]:res["voff"][i] + res["vn"][i]]
        key = V.tobytes()
        v = self.cache.get(key)
        if v is None:
            ctr = V.mean(0); sc = max(float(np.max(np.abs(V - ctr))), 1e-300)
            T = _tetrahedra((V - ctr) / sc)
            vols = np.abs(np.einsum("ij,ij->i", T[:, 1] - T[:, 0], np.cross(T[:, 2] - T[:, 0], T[:, 3] - T[:, 0]))) / 6
            if not vols.sum() > 0:
                vols = np.ones(len(T))
            v = self.cache[key] = (T, vols / vols.sum(), ctr, sc)
        return v

    def draw(self, n):
        if self.ok != "ok":
            return None
        rng = self.rng
        picks = rng.choice(len(self.p), size=n, p=self.p)
        order = np.argsort(picks, kind="stable")
        u, first, cnt = np.unique(picks[order], return_index=True, return_counts=True)
        out = np.empty((n, 3))
        for i, f, m in zip(u.tolist(), first.tolist(), cnt.tolist()):
            T, pv, ctr, sc = self._tri(i)
            k = rng.choice(len(pv), size=m, p=pv) if len(pv) > 1 else np.zeros(m, np.int64)
            a = rng.random((m, 3)); x, y, z = a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
            s1 = x + y > 1
            x[s1], y[s1] = 1 - x[s1], 1 - y[s1]
            s2 = y + z > 1
            s3 = ~s2 & (x + y + z > 1)
            y2, z2 = 1 - z[s2], 1 - x[s2] - y[s2]
            y[s2], z[s2] = y2, z2
            x3, z3 = 1 - y[s3] - z[s3], x[s3] + y[s3] + z[s3] - 1
            x[s3], z[s3] = x3, z3
            w0 = 1 - x - y - z
            Tk = T[k]
            pts = ctr + sc * (w0[:, None] * Tk[:, 0] + x[:, None] * Tk[:, 1] + y[:, None] * Tk[:, 2] + z[:, None] * Tk[:, 3])
            out[order[f:f + m]] = pts
        return out
