"""Exact, enumerable 2D toy world with the same probabilistic structure as RIND3D 1K (audit experiments E5-E7).

World [0,1]^2 on an n x n cell grid. Geometry G = a random subset of `n_obst` candidate obstacle cells (each present
independently with probability `p_obst`; obstacle = closed square of side `shrink`*h centred in its cell). Sources are
cell centres, uniform over the non-obstacle cells (the legal domain D(G)). A window is a w x w block of receiver cells
(never obstacles, so the local SOLID crop is empty and X = (window index, binary response pattern)). Visibility is
closed-segment / closed-solid: receiver x is lit by s iff the segment [s, x] meets no obstacle square.

Everything is enumerated exactly: 2^n_obst geometries x n^2 source cells x windows.
"""
import itertools
import numpy as np


def seg_hits_square(P, Q, lo, hi):
    """Vectorised Liang-Barsky: does segment P->Q (arrays (..., 2)) meet the closed box [lo, hi]?"""
    d = Q - P
    t0 = np.zeros(P.shape[:-1]); t1 = np.ones(P.shape[:-1])
    ok = np.ones(P.shape[:-1], bool)
    for a in range(2):
        da = d[..., a]; pa = P[..., a]
        par = np.abs(da) < 1e-15
        ok &= ~(par & ((pa < lo[a]) | (pa > hi[a])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[a] - pa) / da; tb = (hi[a] - pa) / da
        tmin = np.where(par, -np.inf, np.minimum(ta, tb)); tmax = np.where(par, np.inf, np.maximum(ta, tb))
        t0 = np.maximum(t0, tmin); t1 = np.minimum(t1, tmax)
    return ok & (t0 <= t1)


class Toy:
    def __init__(self, n=16, w=3, windows=((3, 3), (10, 3), (3, 10), (10, 10)), n_obst=10, p_obst=0.35, shrink=0.9,
                 seed=7, obst_cells=None):
        self.n, self.w, self.h = n, w, 1.0 / n
        self.windows = [tuple(o) for o in windows]
        win_cells = set()
        self.win_idx = []
        for (ox, oy) in self.windows:
            cells = [(ox + i, oy + j) for i in range(w) for j in range(w)]
            self.win_idx.append(np.array([cx * n + cy for cx, cy in cells]))
            win_cells |= set(cells)
        rng = np.random.default_rng(seed)
        if obst_cells is None:
            cand = [(i, j) for i in range(n) for j in range(n) if (i, j) not in win_cells]
            pick = rng.choice(len(cand), n_obst, replace=False)
            obst_cells = [cand[i] for i in pick]
        self.obst = list(obst_cells); self.no = len(self.obst); self.p_obst = p_obst
        self.obst_idx = np.array([cx * n + cy for cx, cy in self.obst])
        c = (np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2) + 0.5) * self.h
        self.centres = c                                   # source cell centres, index cx * n + cy
        self.S = n * n
        # block[o, s, wi, r]: obstacle o blocks segment source s -> receiver r of window wi
        self.block = np.zeros((self.no, self.S, len(self.windows), w * w), bool)
        for wi, idx in enumerate(self.win_idx):
            R = c[idx]
            P = np.repeat(c[:, None, :], len(idx), 1); Q = np.repeat(R[None, :, :], self.S, 0)
            for o, (ox, oy) in enumerate(self.obst):
                ctr = (np.array([ox, oy]) + 0.5) * self.h; half = 0.5 * shrink * self.h
                self.block[o, :, wi, :] = seg_hits_square(P, Q, ctr - half, ctr + half)
        self.in_win = np.zeros((len(self.windows), self.S), bool)
        for wi, idx in enumerate(self.win_idx):
            self.in_win[wi, idx] = True

    def geometries(self):
        """Yield (bits, prior prob, obstacle mask over source cells)."""
        for bits in itertools.product((0, 1), repeat=self.no):
            b = np.array(bits, bool)
            pr = float(np.prod(np.where(b, self.p_obst, 1 - self.p_obst)))
            solid = np.zeros(self.S, bool); solid[self.obst_idx[b]] = True
            yield b, pr, solid

    def responses(self, b):
        """lit[s, wi, r] for geometry bits b (sources inside obstacles are handled by the caller's legal mask)."""
        if b.any():
            return ~np.any(self.block[b], axis=0)
        return np.ones(self.block.shape[1:], bool)

    @staticmethod
    def codes(lit):
        """Pattern code per (s, wi): bits of the w*w receivers."""
        wts = (1 << np.arange(lit.shape[-1]))
        return (lit.astype(np.int64) * wts).sum(-1)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def kl(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return t.sum(-1)
