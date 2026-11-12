"""Selection-law study: how much does the v0.1 window-selection factor lambda_W(s) vary over a window's feasible set?

v0.1 ("mixed" observation sampling, stratified_windows.select): per side sigma, a pool of 8 x 85 accepted proposals
(uniform integer origins from scrambled Sobol, accepted iff informative and source-excluded), then 85 picks mixing
'natural' (any remaining pool member), 'uniform' (the lit-ratio bin with the largest dataset deficit) and 'extreme'
(bins 0/9) components, each pick = best spatial gain among 32 random eligible.  Marginalising the per-scene Sobol
scramble and pick RNG (seeds derived independently of the source, HMAC seed protocol) gives, to leading order,

    lambda_W(s)  ~  k_nat / (N_o P_valid(sigma, s))  +  k_b / (N_o P_b(sigma, s)),

with P_valid the fraction of origins giving a valid window and P_b the fraction valid AND in W's lit-ratio bin b
(b is fixed by O_W).  This is a model [A] (pool truncation, spatial-gain competition and bin-availability effects
are ignored), so the ratios below are model estimates, not certificates.  R(s) (source eligibility, 16^3-block
supports) is not evaluated here; it rejected 3 of 1,003 v0.1 sources and is treated as constant on F_W [A].

Fields are rasterised at 1024^3 per (y, z) row: each component's shadow S_c(s) and solid are convex, so each gives
one x-interval per row (float arithmetic; voxel centres exactly on a boundary may differ from the canonical exact
renderer, which is immaterial for these frequencies).
"""
import numpy as np
from numba import njit, prange

from rpl.engine2 import shadow_rows, MAXROWS

N = 1024


@njit(cache=True)
def _interval(rows, nrow, y, z):
    lo = -np.inf; hi = np.inf
    for q in range(nrow):
        a0 = rows[q, 0]; r = rows[q, 1] * y + rows[q, 2] * z + rows[q, 3]
        if abs(a0) < 1e-300:
            if r > 0:
                return 1.0, 0.0
            continue
        t = -r / a0
        if a0 > 0:
            if t < hi:
                hi = t
        else:
            if t > lo:
                lo = t
    return lo, hi


@njit(parallel=True, cache=True)
def rasterize(s, P, K, adj, aoff, MAXI, dark_runs, ndark, solid_runs, nsolid, bbox):
    """Per row (y, z): dark-FREE runs and SOLID runs as half-open voxel ranges [a, b).  bbox[class] collects the
    min/max voxel coordinates of lit-FREE (class 0) and dark-FREE (class 1) voxels for the eligibility test."""
    C = P.shape[0]
    kmax = P.shape[1]
    SR = np.zeros((C, MAXROWS, 4)); nS = np.zeros(C, np.int64); flag = np.zeros(C, np.int64)
    alpha = np.empty(kmax)
    for c in range(C):
        n, fl = shadow_rows(s, P, K, c, adj, aoff, alpha, SR[c])
        nS[c] = n; flag[c] = fl
    Rs = np.zeros((C, kmax, 4))
    for c in range(C):
        for k in range(K[c]):
            nr = np.sqrt(P[c, k, 0] ** 2 + P[c, k, 1] ** 2 + P[c, k, 2] ** 2)
            for j in range(4):
                Rs[c, k, j] = P[c, k, j] / nr
    for row in prange(N * N):
        z = row // N; yv = row % N
        y = (yv + 0.5) / N; zz = (z + 0.5) / N
        sa = np.empty(C, np.int64); sb = np.empty(C, np.int64); m = 0
        da = np.empty(C, np.int64); db = np.empty(C, np.int64); md = 0
        for c in range(C):
            lo, hi = _interval(Rs[c], K[c], y, zz)
            lo = max(lo, -1.0); hi = min(hi, 2.0)     # unbounded ends: clamp before integer conversion
            if hi >= lo:
                a = int(np.ceil(lo * N - 0.5)); b = int(np.floor(hi * N - 0.5)) + 1
                a = max(a, 0); b = min(b, N)
                if b > a:
                    sa[m] = a; sb[m] = b; m += 1
            if flag[c] == 0 and nS[c] > 0:
                lo, hi = _interval(SR[c], nS[c], y, zz)
                lo = max(lo, -1.0); hi = min(hi, 2.0)
                if hi >= lo:
                    a = int(np.ceil(lo * N - 0.5)); b = int(np.floor(hi * N - 0.5)) + 1
                    a = max(a, 0); b = min(b, N)
                    if b > a:
                        da[md] = a; db[md] = b; md += 1
        # merge solid runs
        o1 = np.argsort(sa[:m])
        ms = 0
        for t in range(m):
            a = sa[o1[t]]; b = sb[o1[t]]
            if ms > 0 and a <= solid_runs[row, ms - 1, 1]:
                if b > solid_runs[row, ms - 1, 1]:
                    solid_runs[row, ms - 1, 1] = b
            else:
                solid_runs[row, ms, 0] = a; solid_runs[row, ms, 1] = b; ms += 1
        nsolid[row] = ms
        # merge shadow runs, then subtract solid
        o2 = np.argsort(da[:md])
        tmpa = np.empty(C, np.int64); tmpb = np.empty(C, np.int64); mt = 0
        for t in range(md):
            a = da[o2[t]]; b = db[o2[t]]
            if mt > 0 and a <= tmpb[mt - 1]:
                if b > tmpb[mt - 1]:
                    tmpb[mt - 1] = b
            else:
                tmpa[mt] = a; tmpb[mt] = b; mt += 1
        nd = 0
        for t in range(mt):
            a = tmpa[t]; b = tmpb[t]
            cur = a
            for u in range(ms):
                sa_ = solid_runs[row, u, 0]; sb_ = solid_runs[row, u, 1]
                if sb_ <= cur or sa_ >= b:
                    continue
                if sa_ > cur and nd < MAXI:
                    dark_runs[row, nd, 0] = cur; dark_runs[row, nd, 1] = sa_; nd += 1
                cur = max(cur, sb_)
                if cur >= b:
                    break
            if cur < b and nd < MAXI:
                dark_runs[row, nd, 0] = cur; dark_runs[row, nd, 1] = b; nd += 1
        ndark[row] = nd


@njit(parallel=True, cache=True)
def window_counts(origins, side, dark_runs, ndark, solid_runs, nsolid, out):
    """out[i] = (lit_free, dark_free, solid) of the window at origins[i] (x, y, z) with the given side."""
    for i in prange(origins.shape[0]):
        ox = origins[i, 0]; oy = origins[i, 1]; oz = origins[i, 2]
        x1 = ox + side
        dk = 0; so = 0
        for z in range(oz, oz + side):
            for y in range(oy, oy + side):
                row = z * N + y
                for u in range(ndark[row]):
                    a = max(dark_runs[row, u, 0], ox); b = min(dark_runs[row, u, 1], x1)
                    if b > a:
                        dk += b - a
                for u in range(nsolid[row]):
                    a = max(solid_runs[row, u, 0], ox); b = min(solid_runs[row, u, 1], x1)
                    if b > a:
                        so += b - a
        tot = side * side * side
        out[i, 0] = tot - dk - so; out[i, 1] = dk; out[i, 2] = so


class Field:
    def __init__(self, geom, s):
        C = geom.C
        self.MAXI = max(2 * C, 4)
        self.dark = np.zeros((N * N, self.MAXI, 2), np.int32)
        self.nd = np.zeros(N * N, np.int32)
        self.solid = np.zeros((N * N, max(C, 1), 2), np.int32)
        self.ns = np.zeros(N * N, np.int32)
        rasterize(np.asarray(s, float), geom.P, geom.K, geom.adj, geom.aoff, self.MAXI, self.dark, self.nd,
                  self.solid, self.ns, np.zeros((2, 6), np.int64))
        self.s = np.asarray(s, float)

    def counts(self, origins, side):
        out = np.zeros((len(origins), 3), np.int64)
        window_counts(np.ascontiguousarray(origins, np.int64), int(side), self.dark, self.nd, self.solid, self.ns, out)
        return out


def valid_and_bin(cnt, origins, side, s):
    lit = cnt[:, 0]; dk = cnt[:, 1]
    inf = (lit > 0) & (dk > 0)
    p = np.asarray(s) * N
    contains = np.all((p >= origins) & (p < origins + side), axis=1)
    valid = inf & ~contains
    free = lit + dk
    b = np.where(free > 0, np.minimum(9, (10 * lit) // np.maximum(free, 1)), -1)
    return valid, b
