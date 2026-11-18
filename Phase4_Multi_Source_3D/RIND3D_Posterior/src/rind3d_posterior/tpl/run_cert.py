"""Driver of the certificate-emitting tau-pencil engine (engine_cert): stage 1 + optional Local-first root clips +
stage 2, returning the engine's own records (untrusted) and the proof tree (event stream) for the verifier.

The engine is deterministic; every buffer that can overflow reports the exact size needed and the run is repeated
with a larger buffer (as run_p2.run_window), so the event stream is always complete."""
import time

import numpy as np

from . import engine_cert as E
from .run_p2 import window_aspects, summarize


def stage1_cert(geom, win, d1=4, n_boundary=512, n_class=64, n_learn=32, max_tests=640, sentinel_seed=0):
    """Returns (roots (n, 4), s1rec (m, 6) int64 with sentinel indices replaced by flat window voxel indices, seconds)."""
    P, K, clo, chi, adj, aoff = geom.tup()
    wlo, whi = win.box()
    SX, SL = win.sentinels(n_boundary, n_class, sentinel_seed)
    cand = np.empty((1 << (3 * d1), 3), np.int64)
    s1 = np.zeros(4, np.int64)
    # a complete octree to depth d1 has at most sum 8^d terminal boxes
    s1rec = np.empty((sum(8 ** d for d in range(d1 + 1)), 6), np.int64)
    t0 = time.perf_counter()
    nc, nr = E.stage1_cert(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, n_learn, max_tests, cand, s1, s1rec)
    sec = time.perf_counter() - t0
    roots = np.zeros((nc, 4), np.int64); roots[:, 0] = d1; roots[:, 1:] = cand[:nc]
    rec = s1rec[:nr].copy()
    # sentinel index -> flat window voxel index (receivers are voxel centres (o + v + 1/2)/N)
    L = win.L
    vox = np.rint(SX * win.N - win.origin - 0.5).astype(np.int64)
    assert np.all((vox + win.origin + 0.5) / win.N == SX), "sentinel is not a voxel centre"
    flat = (vox[:, 0] * L + vox[:, 1]) * L + vox[:, 2]
    m = rec[:, 4] == E.S1_REFUTED
    rec[m, 5] = flat[rec[m, 5]]
    return roots, rec, sec


def stage2_cert(geom, win, roots, clip_off=None, clip_rows=None, max_records=1 << 22, ne_cap=1 << 25, nq_cap=1 << 24,
                max_vertices=1 << 23, max_nodes_cell=2_000_000, max_nodes_total=5_000_000, max_events=1 << 20,
                max_depth=400, max_pass=24, rho=2.0, shell_eps=2.0 ** -26):
    """Stage 2 on the given roots (and per-root clip rows). Returns dict(res (engine records + stats), ev, evf)."""
    P, K, clo, chi, adj, aoff = geom.tup()
    wmode, wfront = window_aspects(geom, win)
    codes, kls = win.top_items()
    n = len(roots)
    if clip_off is None:
        clip_off = np.zeros(n + 1, np.int64); clip_rows = np.zeros((1, 4))
    clip_rows = np.ascontiguousarray(clip_rows, float).reshape(-1, 4)
    if len(clip_rows) == 0:
        clip_rows = np.zeros((1, 4))
    z = np.zeros(1, np.int64)
    bf = (z, np.zeros(1, np.int8), z, z, np.zeros((1, 6)))
    while True:
        rec_cell = np.empty(max_records, np.int64); rec_lo = np.empty(max_records); rec_hi = np.empty(max_records)
        rec_kind = np.empty(max_records, np.int8); rec_voff = np.empty(max_records, np.int64); rec_vn = np.empty(max_records, np.int64)
        vpool = np.empty((max_vertices, 3)); vtop = np.zeros(1, np.int64); stats = np.zeros(len(E.STAT_NAMES), np.int64)
        cell_nodes = np.zeros(max(n, 1), np.int64); cell_evals = np.zeros(max(n, 1), np.int64)
        dep_evals = np.zeros(64, np.int64); dep_nodes = np.zeros(64, np.int64)
        fbc = np.zeros(geom.C, np.int64); fs = np.zeros(geom.C, np.int64)
        ev = np.empty((max_events, 8), np.int64); evf = np.empty(max_events); nev = np.zeros(1, np.int64)
        t0 = time.perf_counter()
        nrec = E.stage2(P, K, clo, chi, adj, aoff, win.origin, float(win.N), win.L, codes, kls,
                        np.ascontiguousarray(roots, np.int64), n, max_nodes_cell, max_nodes_total, max_depth, max_pass, True, True,
                        rho, wmode, wfront, shell_eps, rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats,
                        cell_nodes, cell_evals, fbc, fs, dep_evals, dep_nodes, ne_cap, nq_cap, *bf, False,
                        np.asarray(clip_off, np.int64), clip_rows, ev, evf, nev)
        sec = time.perf_counter() - t0
        again = False
        if nrec > max_records:
            max_records = 1 << int(nrec - 1).bit_length(); again = True
        if nev[0] > max_events:
            max_events = 1 << int(nev[0] - 1).bit_length(); again = True
        if not again:
            break
    res = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(), kind=rec_kind[:nrec].copy(),
               stats=dict(zip(E.STAT_NAMES, stats.tolist())), seconds=sec)
    res.update(summarize(res))
    return dict(res=res, ev=ev[:nev[0]].copy(), evf=evf[:nev[0]].copy())
