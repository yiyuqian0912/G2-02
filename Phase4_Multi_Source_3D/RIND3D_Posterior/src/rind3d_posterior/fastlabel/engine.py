"""Tau-pencil engine wrappers used by FastLabel (production copy of experiments/twophase/twophase.py: stage1, stage2,
point_items, cell_masses, tv). stage2 with `bf` uses the outer-bound filter copy of the engine (tpl/engine_p2x)."""
import time

import numpy as np
from rind3d_posterior.tpl import engine_p2 as E
from rind3d_posterior.tpl.run_p2 import window_aspects, summarize
from rind3d_posterior.tpl.geom import Geom
from rind3d_posterior.tpl.window import enc

def stage2(geom, win, codes, kls, roots, max_records=1 << 22, ne_cap=1 << 26, nq_cap=1 << 25, max_vertices=1 << 23, bf=None,
           max_nodes_cell=2_000_000, max_nodes_total=5_000_000):
    P, K, clo, chi, adj, aoff = geom.tup()
    wmode, wfront = window_aspects(geom, win)
    rec_cell = np.empty(max_records, np.int64); rec_lo = np.empty(max_records); rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8); rec_voff = np.empty(max_records, np.int64); rec_vn = np.empty(max_records, np.int64)
    vpool = np.empty((max_vertices, 3)); vtop = np.zeros(1, np.int64)
    stats = np.zeros(len(E.STAT_NAMES), np.int64)
    n = len(roots)
    cell_nodes = np.zeros(max(n, 1), np.int64); cell_evals = np.zeros(max(n, 1), np.int64)
    dep_evals = np.zeros(64, np.int64); dep_nodes = np.zeros(64, np.int64)
    fbc = np.zeros(geom.C, np.int64); fs = np.zeros(geom.C, np.int64)
    t0 = time.perf_counter()
    args = (P, K, clo, chi, adj, aoff, win.origin, float(win.N), win.L, codes, kls, np.ascontiguousarray(roots, np.int64), n,
            max_nodes_cell, max_nodes_total, 400, 24, True, True, 2.0, wmode, wfront, 2.0 ** -26,
            rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
            fbc, fs, dep_evals, dep_nodes, ne_cap, nq_cap)
    if bf is None:
        nrec = E.stage2(*args)
    else:
        from rind3d_posterior.tpl import engine_p2x as EX
        nrec = EX.stage2(*args, *bf, True)
    assert nrec <= max_records
    res = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(), kind=rec_kind[:nrec].copy(),
               voff=rec_voff[:nrec].copy(), vn=rec_vn[:nrec].copy(), vpool=vpool[:vtop[0]].copy(),
               stats=dict(zip(E.STAT_NAMES, stats.tolist())), seconds=time.perf_counter() - t0)
    res.update(summarize(res))
    return res


def stage1(geom, win, d1=4, n_boundary=512, n_class=64, n_learn=32, max_tests=640):
    P, K, clo, chi, adj, aoff = geom.tup()
    wlo, whi = win.box()
    SX, SL = win.sentinels(n_boundary, n_class, 0)
    cand = np.empty((1 << (3 * d1), 3), np.int64); s1 = np.zeros(4, np.int64)
    t0 = time.perf_counter()
    nc = E.stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, n_learn, max_tests, cand, s1)
    roots = np.zeros((nc, 4), np.int64); roots[:, 0] = d1; roots[:, 1:] = cand[:nc]
    return roots, time.perf_counter() - t0


def point_items(win, n_bd, n_in, seed=0):
    """Level-0 items only: n_bd random boundary receivers + n_in random interior receivers (no block expansion)."""
    rng = np.random.default_rng(seed)
    bd = win.boundary_mask().reshape(-1)
    c = win.cls.reshape(-1)
    free = c != 2
    ib = np.flatnonzero(bd); ii = np.flatnonzero(free & ~bd)
    ib = rng.choice(ib, min(n_bd, len(ib)), replace=False) if len(ib) else ib
    ii = rng.choice(ii, min(n_in, len(ii)), replace=False) if len(ii) else ii
    idx = np.concatenate([ib, ii])
    v = np.stack(np.unravel_index(idx, win.cls.shape), 1)
    codes = np.array([enc(0, a, b, d) for a, b, d in v], np.int64)
    return codes, c[idx].astype(np.int8)


def cell_masses(cell, lo, hi):
    u, inv = np.unique(cell, return_inverse=True)
    return dict(zip(u.tolist(), np.bincount(inv, weights=0.5 * (lo + hi)).tolist()))


def tv(a, b):
    sa, sb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in set(a) | set(b))
