"""X-measurable interface facets: lit/dark FREE 6-neighbour pair midpoints -> greedy local-RANSAC planar facets (numba)."""
import numpy as np
from numba import njit

@njit(cache=True)
def _interface(sol, lit, out):
    L = sol.shape[0]; n = 0
    for z in range(L):
        for y in range(L):
            for x in range(L):
                if sol[z, y, x]:
                    continue
                c = lit[z, y, x]
                if x + 1 < L and not sol[z, y, x + 1] and lit[z, y, x + 1] != c:
                    out[n, 0] = x + 1.0; out[n, 1] = y + 0.5; out[n, 2] = z + 0.5; n += 1
                if y + 1 < L and not sol[z, y + 1, x] and lit[z, y + 1, x] != c:
                    out[n, 0] = x + 0.5; out[n, 1] = y + 1.0; out[n, 2] = z + 0.5; n += 1
                if z + 1 < L and not sol[z + 1, y, x] and lit[z + 1, y, x] != c:
                    out[n, 0] = x + 0.5; out[n, 1] = y + 0.5; out[n, 2] = z + 1.0; n += 1
                if n + 3 > out.shape[0]:
                    return n
    return n


def interface_points(sol, lit, cap=4_000_000):
    """sol, lit: ZYX bool crops. Returns (P, 3) XYZ midpoints of lit/dark FREE 6-neighbour pairs, in crop voxel units
    (voxel centres at i + 1/2), ordered by (z, y, x, axis)."""
    out = np.empty((cap, 3), np.float64)
    n = _interface(np.ascontiguousarray(sol), np.ascontiguousarray(lit), out)
    return out[:n].copy()

@njit(cache=True)
def _pca(P, idx, n):
    mu = np.zeros(3)
    for k in range(n): mu += P[idx[k]]
    mu /= n
    C = np.zeros((3, 3))
    for k in range(n):
        d = P[idx[k]] - mu
        C += np.outer(d, d)
    w, V = np.linalg.eigh(C / n)
    return mu, V[:, 0], w

@njit(cache=True)
def ransac_facets(P, block_start, block_order, block_of, seed, max_facets, thr, min_inl, iters, nscore):
    np.random.seed(seed)
    n = P.shape[0]
    alive = np.ones(n, np.bool_)
    out = np.zeros((max_facets, 12))     # mu(3) normal(3) n_inl, rms, eig1, eig2 (in-plane sd), alive_after, quality
    nf = 0
    tmp = np.empty(n, np.int64)
    for f in range(max_facets):
        nalive = alive.sum()
        if nalive < min_inl: break
        alive_idx = np.flatnonzero(alive)
        best_cnt = 0; best_mu = np.zeros(3); best_n = np.zeros(3)
        ns = min(nalive, nscore)                     # hypotheses are scored on a random subsample of the alive points
        sub = alive_idx[np.random.choice(nalive, ns, replace=False)] if ns < nalive else alive_idx
        for it in range(iters):
            a = alive_idx[np.random.randint(nalive)]
            b0 = block_start[block_of[a]]; b1 = block_start[block_of[a] + 1]
            if b1 - b0 < 3: continue
            ok = 0; c = np.empty(2, np.int64)
            for t in range(20):
                j = block_order[b0 + np.random.randint(b1 - b0)]
                if alive[j] and j != a and (ok == 0 or j != c[0]):
                    c[ok] = j; ok += 1
                    if ok == 2: break
            if ok < 2: continue
            nrm = np.cross(P[c[0]] - P[a], P[c[1]] - P[a]); ln = np.sqrt((nrm ** 2).sum())
            if ln < 1e-6: continue
            nrm /= ln
            cnt = 0
            for k in range(ns):
                i = sub[k]
                if abs(((P[i] - P[a]) * nrm).sum()) < thr: cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt; best_mu = P[a].copy(); best_n = nrm.copy()
        if best_cnt * nalive < min_inl * ns: break
        mu = best_mu; nrm = best_n
        for rep in range(3):              # refine: PCA on inliers
            m = 0
            for k in range(nalive):
                i = alive_idx[k]
                if abs(((P[i] - mu) * nrm).sum()) < thr:
                    tmp[m] = i; m += 1
            if m < 3: break
            mu, nrm, w = _pca(P, tmp, m)
        m = 0; ss = 0.0
        for k in range(nalive):
            i = alive_idx[k]
            d = ((P[i] - mu) * nrm).sum()
            if abs(d) < thr:
                alive[i] = False; m += 1; ss += d * d
        if m < min_inl: break
        out[nf, 0:3] = mu; out[nf, 3:6] = nrm; out[nf, 6] = m; out[nf, 7] = np.sqrt(ss / m)
        out[nf, 8] = np.sqrt(max(w[1], 0)); out[nf, 9] = np.sqrt(max(w[2], 0)); out[nf, 10] = alive.sum()
        nf += 1
    return out[:nf]

def facets_of(sol, lit, max_facets=8, thr=0.9, min_frac=0.01, min_abs=24, iters=150, seed=0, cap=40000, nscore=3000):
    P = interface_points(sol, lit)
    if len(P) < min_abs: return np.zeros((0, 12)), len(P)
    if len(P) > cap:
        rng = np.random.default_rng(seed); P = P[np.sort(rng.choice(len(P), cap, replace=False))]
    blk = np.floor(P / 8).astype(np.int64); L = int(np.ceil(P.max() / 8)) + 1
    bid = (blk[:, 0] * L + blk[:, 1]) * L + blk[:, 2]
    order = np.argsort(bid, kind='stable'); ub, inv = np.unique(bid, return_inverse=True)
    counts = np.bincount(inv); start = np.r_[0, np.cumsum(counts)]
    block_of = inv.astype(np.int64)
    # block_order indexes points sorted by block; block_start per compact block id
    F = ransac_facets(P, start.astype(np.int64), order.astype(np.int64), block_of, seed, max_facets, thr,
                      max(min_abs, int(min_frac * len(P))), iters, nscore)
    return F, len(P)
