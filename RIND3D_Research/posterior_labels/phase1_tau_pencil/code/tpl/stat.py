"""Statistical (Monte Carlo) labellers with valid 99% upper confidence bounds on TV, for comparison.

Both draw sources uniformly in a region U and test feasibility exactly by brute force (every FREE receiver,
rpl.synth.feasible_points).  Per-cell feasible-volume intervals come from Clopper-Pearson intervals with a
Bonferroni split of delta over all cells (and both sides), so they hold simultaneously with probability >= 1-delta
over the sampling randomness, for this window, under the stated design (sample sizes fixed before sampling).
The TV certificate for the point-estimate label qhat is
    TV(qhat, q) <= TV(qhat, q_L) + tv_bound(L, U),   q_L = L / sum(L)   (triangle inequality; rpl.bound).

* cover_mc:  U = union of the 128^3 cells not refuted by the deterministic stage-1 cover (F is inside U); n_c
             points per cell (stratified).
* prior_mc:  U = the world; sample splitting: the first half fixes the set S of cells with hits; the second half
             gives intervals for every cell of S and for the total mass outside S (charged as a phantom cell).
"""
import numpy as np
from scipy.stats import beta

from rpl.bound import tv_bound
from rpl.synth import feasible_points

G = 128


def cp(k, n, a):
    lo = np.where(k > 0, beta.ppf(a / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    return lo, hi


def _tv_cert(qhat_cells, qhat, cells, L, U):
    Lt = L.sum()
    if Lt <= 0:
        return 1.0
    qL = dict(zip(cells.tolist(), (L / Lt).tolist()))
    keys = set(qL) | set(qhat_cells.tolist())
    qh = dict(zip(qhat_cells.tolist(), qhat.tolist()))
    d = 0.5 * sum(abs(qh.get(k, 0.0) - qL.get(k, 0.0)) for k in keys)
    return float(min(1.0, d + tv_bound(L, U)))


def cell_ids(S):
    i = np.minimum((S * G).astype(np.int64), G - 1)
    return (i[:, 0] * G + i[:, 1]) * G + i[:, 2]


def cover_mc(geom, win, cells_xyz, n_per_cell, delta=0.01, seed=0, chunk=1 << 20):
    rng = np.random.default_rng(seed)
    X, obs = win.receivers()
    wlo, whi = win.box()
    M = len(cells_xyz)
    h = np.zeros(M, np.int64)
    for a in range(0, M, max(1, chunk // n_per_cell)):
        b = min(M, a + max(1, chunk // n_per_cell))
        base = np.repeat(cells_xyz[a:b], n_per_cell, axis=0) / G
        S = base + rng.random(base.shape) / G
        f = feasible_points(S, X, obs, geom.P, geom.K, wlo, whi)
        h[a:b] = (f.reshape(b - a, n_per_cell) == 1).sum(1)
    vc = 1.0 / G ** 3
    lo, hi = cp(h, n_per_cell, delta / M)
    ids = (cells_xyz[:, 0] * G + cells_xyz[:, 1]) * G + cells_xyz[:, 2]
    L = lo * vc; U = hi * vc
    est = h / n_per_cell * vc
    m = est > 0
    qhat = est[m] / est.sum() if est.sum() > 0 else np.zeros(0)
    return dict(n_samples=int(M * n_per_cell), n_hits=int(h.sum()), cells_cover=int(M),
                cells_hit=int(m.sum()), volume_est=float(est.sum()), L_total=float(L.sum()), U_total=float(U.sum()),
                tv_cert=_tv_cert(ids[m], qhat, ids, L, U) if m.any() else 1.0,
                label=(ids[m], qhat))


def prior_mc(geom, win, n, delta=0.01, seed=0, chunk=1 << 21):
    rng = np.random.default_rng(seed)
    X, obs = win.receivers()
    wlo, whi = win.box()
    ids = []
    for half in range(2):
        out = []
        for a in range(0, n // 2, chunk):
            m = min(chunk, n // 2 - a)
            S = rng.random((m, 3))
            f = feasible_points(S, X, obs, geom.P, geom.K, wlo, whi)
            out.append(cell_ids(S[f == 1]))
        ids.append(np.concatenate(out) if out else np.zeros(0, np.int64))
    n2 = n // 2
    Sset = np.unique(ids[0])
    k_in = np.array([(ids[1] == c).sum() for c in Sset], np.int64) if len(Sset) else np.zeros(0, np.int64)
    k_out = int(len(ids[1]) - k_in.sum())
    a = delta / (len(Sset) + 1)
    lo, hi = cp(k_in, n2, a)
    _, hi_out = cp(np.array([k_out]), n2, a)
    L = np.r_[lo, 0.0]; U = np.r_[hi, hi_out]
    cells = np.r_[Sset, -1]
    est = k_in / n2
    m = est > 0
    qhat = est[m] / est.sum() if est.sum() > 0 else np.zeros(0)
    return dict(n_samples=int(n), n_hits=int(len(ids[0]) + len(ids[1])), cells_hit=int(len(Sset)),
                volume_est=float(len(ids[1]) / n2), tv_cert=_tv_cert(Sset[m], qhat, cells, L, U) if m.any() else 1.0,
                label=(Sset[m], qhat))
