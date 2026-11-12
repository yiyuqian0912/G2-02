"""Driver for engine v3 (receiver-side aspect classes).  Same outputs as run2."""
import time
import numpy as np

from rpl.bound import tv_bound
from . import engine_v3 as E3
from . import engine_v2 as E2
from .window import Window, enc
from .run import summarize, cell_of, cell_intervals, label  # noqa: F401

MARG = 1e-11
AMARG = 1e-12


def _box_range(n, d, lo, hi):
    mn = d + np.sum(np.where(n >= 0, n * lo, n * hi)); mx = d + np.sum(np.where(n >= 0, n * hi, n * lo))
    return mn, mx


def receiver_classes(geom, win, max_cross=6):
    """Per component: crossing face planes of the window's receiver box and the receiver-side aspect classes."""
    C, kmax = geom.P.shape[:2]
    lo = (win.origin + 0.5) / win.N; hi = (win.origin + win.L - 0.5) / win.N
    X = None
    info = []
    for c in range(C):
        cross = []; frontfix = np.zeros(kmax, bool)
        for i in range(geom.K[c]):
            mn, mx = _box_range(geom.P[c, i, :3], geom.P[c, i, 3], lo, hi)
            if mn > MARG:
                frontfix[i] = True
            elif mx < -MARG:
                pass
            else:
                cross.append(i)
        ok = len(cross) <= max_cross
        if ok and cross:
            if X is None:
                X, _ = win.receivers()
            for i in cross:
                f = X @ geom.P[c, i, :3] + geom.P[c, i, 3]
                if np.min(np.abs(f)) <= AMARG:
                    ok = False      # a receiver (numerically) on a crossing plane: use the source-side form
                    break
        info.append((ok, cross, frontfix))
    nx = max([len(cr) for ok, cr, _ in info if ok] + [0])
    KC = 1 << nx
    emax = max(int(np.max(np.diff(geom.aoff))), 1)
    wmode = np.zeros(C, np.bool_); nxp = np.zeros(C, np.int64); xp = np.zeros((C, max(nx, 1)), np.int64)
    cvalid = np.zeros((C, KC), np.bool_); cfront = np.zeros((C, KC, kmax), np.bool_)
    cnsil = np.zeros((C, KC), np.int64); csilf = np.zeros((C, KC, emax), np.int64); csilb = np.zeros((C, KC, emax), np.int64)
    ckey = np.zeros((C, KC, emax), np.int64); cnuf = np.zeros((C, KC), np.int64); cuf = np.zeros((C, KC, kmax), np.int64)
    csilfi = np.zeros((C, KC, emax), np.int64); csilbi = np.zeros((C, KC, emax), np.int64)
    for c, (ok, cross, frontfix) in enumerate(info):
        if not ok:
            continue
        wmode[c] = True
        nxp[c] = len(cross)
        xp[c, :len(cross)] = cross
        for kap in range(1 << len(cross)):
            F = frontfix.copy()
            for j, i in enumerate(cross):
                F[i] = bool((kap >> j) & 1)
            if not F[:geom.K[c]].any():
                continue
            cvalid[c, kap] = True
            cfront[c, kap] = F
            n = 0; ufl = []
            for t in range(geom.aoff[c], geom.aoff[c + 1]):
                i, j = geom.adj[t, 1], geom.adj[t, 2]
                if F[i] and not F[j]:
                    f, b = i, j
                elif F[j] and not F[i]:
                    f, b = j, i
                else:
                    continue
                csilf[c, kap, n] = f; csilb[c, kap, n] = b
                ckey[c, kap, n] = 2 * t + (0 if f == geom.adj[t, 1] else 1)
                for side, face in ((0, f), (1, b)):
                    if face not in ufl:
                        ufl.append(face)
                    (csilfi if side == 0 else csilbi)[c, kap, n] = ufl.index(face)
                n += 1
            cnsil[c, kap] = n
            cnuf[c, kap] = len(ufl)
            cuf[c, kap, :len(ufl)] = ufl
    return dict(wmode=wmode, nxp=nxp, xp=xp, cvalid=cvalid, cfront=cfront, cnsil=cnsil, csilf=csilf, csilb=csilb,
                ckey=ckey, cnuf=cnuf, cuf=cuf, csilfi=csilfi, csilbi=csilbi)


def presplit_items(geom, win, codes, kls, cl):
    """Split top items until none straddles a crossing plane of a receiver-side component."""
    planes = [(geom.P[c, i, :3], geom.P[c, i, 3]) for c in range(geom.C) if cl['wmode'][c]
              for i in cl['xp'][c, :cl['nxp'][c]]]
    if not planes:
        return codes, kls, 0
    out_c, out_k = [], []
    stack = list(zip(codes.tolist(), kls.tolist()))
    nsplit = 0
    while stack:
        code, k = stack.pop()
        l = (code >> 48) & 0xFFFF; bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
        s = 1 << l
        lo = (win.origin + np.array([bx, by, bz]) * s + 0.5) / win.N
        hi = (win.origin + (np.array([bx, by, bz]) + 1) * s - 0.5) / win.N
        straddle = False
        for n, d in planes:
            mn, mx = _box_range(n, d, lo, hi)
            if not (mn > AMARG or mx < -AMARG):
                straddle = True
                break
        if straddle and l > 0:
            nsplit += 1
            for c8 in range(8):
                stack.append((int(enc(l - 1, 2 * bx + (c8 & 1), 2 * by + ((c8 >> 1) & 1), 2 * bz + ((c8 >> 2) & 1))), k))
        else:
            out_c.append(code); out_k.append(k)
    return np.array(out_c, np.int64), np.array(out_k, np.int8), nsplit


def run_window(geom, win: Window, d1=4, max_tests=640, n_boundary=512, n_class=64, n_learn=32, max_nodes_cell=2_000_000,
               max_nodes_total=5_000_000, max_depth=400, max_pass=24, tau_first=True, dual=True, rho=2.0,
               use_wmode=True, presplit=False, max_records=1 << 22, max_vertices=1 << 23, sentinel_seed=0):
    P, K, clo, chi, adj, aoff = geom.tup()
    o = win.origin
    wlo, whi = win.box()
    t0 = time.perf_counter()
    SX, SL = win.sentinels(n_boundary, n_class, sentinel_seed)
    cand = np.empty((1 << (3 * d1), 3), np.int64)
    s1 = np.zeros(4, np.int64)
    ncand = E2.stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, n_learn, max_tests, cand, s1)
    roots = np.zeros((ncand, 4), np.int64)
    roots[:, 0] = d1
    roots[:, 1:] = cand[:ncand]
    t1 = time.perf_counter()
    codes, kls = win.top_items()
    cl = receiver_classes(geom, win)
    if not use_wmode:
        cl['wmode'][:] = False
    codes, kls, nsplit = presplit_items(geom, win, codes, kls, cl) if presplit else (codes, kls, 0)
    rec_cell = np.empty(max_records, np.int64); rec_lo = np.empty(max_records); rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8); rec_voff = np.empty(max_records, np.int64)
    rec_vn = np.empty(max_records, np.int64); vpool = np.empty((max_vertices, 3)); vtop = np.zeros(1, np.int64)
    stats = np.zeros(len(E2.STAT_NAMES), np.int64)
    cell_nodes = np.zeros(ncand, np.int64); cell_evals = np.zeros(ncand, np.int64)
    face_by_comp = np.zeros(geom.C, np.int64); face_surv = np.zeros(geom.C, np.int64)
    t2 = time.perf_counter()
    nrec = E3.stage2(P, K, clo, chi, adj, aoff, o, float(win.N), win.L, codes, kls, roots, ncand,
                     max_nodes_cell, max_nodes_total, max_depth, max_pass, tau_first, dual, rho,
                     cl['wmode'], cl['nxp'], cl['xp'], cl['cvalid'], cl['cfront'], cl['cnsil'], cl['csilf'],
                     cl['csilb'], cl['ckey'], cl['cnuf'], cl['cuf'], cl['csilfi'], cl['csilbi'],
                     rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
                     face_by_comp, face_surv)
    t3 = time.perf_counter()
    res = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(),
               kind=rec_kind[:nrec].copy(), voff=rec_voff[:nrec].copy(), vn=rec_vn[:nrec].copy(),
               vpool=vpool[:vtop[0]].copy(), candidates=roots, cell_nodes=cell_nodes, cell_evals=cell_evals,
               stage1=dict(nodes=int(s1[0]), out=int(s1[1]), tests=int(s1[2]), candidates=int(ncand)),
               stats=dict(zip(E2.STAT_NAMES, stats.tolist())), face_by_comp=face_by_comp.tolist(),
               face_surv=face_surv.tolist(), wmode=cl['wmode'].tolist(), nxp=cl['nxp'].tolist(),
               n_items=int(len(codes)), presplit=int(nsplit),
               seconds=dict(stage1=t1 - t0, prep=t2 - t1, stage2=t3 - t2, total=t3 - t0))
    res.update(summarize(res))
    return res
