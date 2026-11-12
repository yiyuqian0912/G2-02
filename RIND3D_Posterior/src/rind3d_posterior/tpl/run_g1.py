"""Python driver for the tau-pencil engine: one window -> per-cell volume intervals and a TV certificate."""
import time

import numpy as np

from ..rpl.bound import tv_bound
from . import engine_g2 as E
from .window import Window

CELLS = 1 << E.CELL_DEPTH


def run_window(geom, win: Window, d1=5, max_tests=640, n_boundary=512, n_class=64, n_learn=32, max_nodes_cell=2_000_000, max_nodes_total=5_000_000,
               max_depth=400, max_pass=24, tau_first=True, dual=True, rho=2.0, use_wmode=True, shell_eps=2.0 ** -26, no_alloc_hull=False, use_gpu=False, gpu_min=1024, gpu_prefix=64, gpu_profile=False, gpu_version=2, gpu_f32=False, gpu_cap=1 << 22, max_records=1 << 22, max_vertices=1 << 23, sentinel_seed=0):
    assert 0 <= d1 <= E.CELL_DEPTH
    P, K, clo, chi, adj, aoff = geom.tup()
    o = win.origin
    wlo, whi = win.box()
    t0 = time.perf_counter()
    SX, SL = win.sentinels(n_boundary, n_class, sentinel_seed)
    cand = np.empty((1 << (3 * d1), 3), np.int64)
    s1 = np.zeros(4, np.int64)
    ncand = E.stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, n_learn, max_tests, cand, s1)
    roots = np.zeros((ncand, 4), np.int64)
    roots[:, 0] = d1
    roots[:, 1:] = cand[:ncand]
    t1 = time.perf_counter()
    codes, kls = win.top_items()
    wmode, wfront = window_aspects(geom, win) if use_wmode else (np.zeros(geom.C, np.bool_), np.zeros(geom.P.shape[:2], np.bool_))
    rec_cell = np.empty(max_records, np.int64)
    rec_lo = np.empty(max_records)
    rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8)
    rec_voff = np.empty(max_records, np.int64)
    rec_vn = np.empty(max_records, np.int64)
    vpool = np.empty((max_vertices, 3))
    vtop = np.zeros(1, np.int64)
    stats = np.zeros(len(E.STAT_NAMES), np.int64)
    cell_nodes = np.zeros(ncand, np.int64)
    cell_evals = np.zeros(ncand, np.int64)
    dep_evals = np.zeros(64, np.int64); dep_nodes = np.zeros(64, np.int64)
    face_by_comp = np.zeros(geom.C, np.int64)
    face_surv = np.zeros(geom.C, np.int64)
    gctx = None
    if use_gpu:
        if gpu_version == 1:
            from .gpu_pass import GpuPass as GP
        elif gpu_version == 2:
            from .gpu_pass2 import GpuPass2 as GP
        else:
            from .gpu_pass3 import GpuPass3 as GP
        gctx = (GP(P, K, clo, chi, o, float(win.N), 1 << 24, f32=gpu_f32, cap=gpu_cap) if gpu_version == 3 else GP(P, K, clo, chi, o, float(win.N), 1 << 24, f32=gpu_f32)) if gpu_version != 1 else GP(P, K, clo, chi, o, float(win.N), 1 << 24)
        gctx.profile = gpu_profile
        E.GPU_CTX[0] = gctx
    nrec = E.stage2(P, K, clo, chi, adj, aoff, o, float(win.N), win.L, codes, kls, roots, ncand,
                    max_nodes_cell, max_nodes_total, max_depth, max_pass, tau_first, dual, rho, wmode, wfront, shell_eps, rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals, face_by_comp, face_surv, dep_evals, dep_nodes, no_alloc_hull, use_gpu, gpu_min, gpu_prefix)
    if gctx is not None:
        import cupy as _cp
        E.GPU_CTX[0] = None
        gstats = dict(segments=gctx.segments, entries=gctx.gpu_entries, tt=gctx.tt.tolist(), fallbacks=getattr(gctx, 'fallbacks', 0), guards=getattr(gctx, 'guards', 0), capfb=getattr(gctx, 'capfb', 0), peak_bytes=int(__import__('cupy').get_default_memory_pool().total_bytes()), n32=getattr(gctx, 'n32', 0), n64=getattr(gctx, 'n64', 0), ne64=getattr(gctx, 'ne64', 0), nrl64=getattr(gctx, 'nrl64', 0))
        del gctx
        gctx = None
        _cp.get_default_memory_pool().free_all_blocks()
    else:
        gstats = None
    t2 = time.perf_counter()
    res = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(),
               kind=rec_kind[:nrec].copy(), voff=rec_voff[:nrec].copy(), vn=rec_vn[:nrec].copy(),
               vpool=vpool[:vtop[0]].copy(), candidates=roots, cell_nodes=cell_nodes, cell_evals=cell_evals,
               stage1=dict(nodes=int(s1[0]), out=int(s1[1]), tests=int(s1[2]), candidates=int(ncand)),
               stats=dict(zip(E.STAT_NAMES, stats.tolist())), wmode=wmode.tolist(), face_by_comp=face_by_comp.tolist(), dep_evals=dep_evals.tolist(), dep_nodes=dep_nodes.tolist(), gpu=gstats, face_surv=face_surv.tolist(), n_items=int(len(codes)),
               seconds=dict(stage1=t1 - t0, stage2=t2 - t1, total=t2 - t0))
    res.update(summarize(res))
    return res


def window_aspects(geom, win, marg=1e-11):
    """Component c is handled with the window-side pencil form if none of its face planes meets the box of the
    window's receiver centres: then every receiver sees the same front faces (strictly)."""
    lo = (win.origin + 0.5) / win.N; hi = (win.origin + win.L - 0.5) / win.N
    wmode = np.zeros(geom.C, np.bool_); wfront = np.zeros(geom.P.shape[:2], np.bool_)
    for c in range(geom.C):
        ok = True
        for i in range(geom.K[c]):
            n = geom.P[c, i, :3]; d = geom.P[c, i, 3]
            mn = d + np.sum(np.where(n >= 0, n * lo, n * hi)); mx = d + np.sum(np.where(n >= 0, n * hi, n * lo))
            if mn > marg:
                wfront[c, i] = True
            elif mx < -marg:
                pass
            else:
                ok = False
        wmode[c] = ok
        if not ok:
            wfront[c] = False
    return wmode, wfront


def cell_intervals(res):
    cells, inv = np.unique(res['cell'], return_inverse=True)
    L = np.zeros(len(cells)); U = np.zeros(len(cells))
    np.add.at(L, inv, res['lo'])
    np.add.at(U, inv, res['hi'])
    return cells, L, U


def summarize(res, truth=None):
    cells, L, U = cell_intervals(res)
    out = dict(L_total=float(L.sum()), U_total=float(U.sum()), undet_total=float((U - L).sum()),
               n_cells_certified=int((L > 0).sum()), n_cells_possible=int((U > 0).sum()),
               tv_bound=float(tv_bound(L, U)) if L.sum() > 0 else 1.0)
    return out


def cell_of(s):
    i = np.minimum((np.asarray(s) * CELLS).astype(np.int64), CELLS - 1)
    return int((i[0] * CELLS + i[1]) * CELLS + i[2])


def label(res):
    """Certified label qhat = L / sum(L) over cells (sparse)."""
    cells, L, U = cell_intervals(res)
    m = L > 0
    return cells[m], L[m] / L.sum()
