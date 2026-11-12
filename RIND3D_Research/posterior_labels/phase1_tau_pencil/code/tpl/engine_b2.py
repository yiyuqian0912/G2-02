"""TPE: tau-pencil engine for exact global source-posterior cell masses of one window (numba).

Target (docs/METHOD.md):  F_W = { s in (0,1)^3 \\ B(W) : every FREE receiver x of W is dark  <=>  [s, x] meets the
solid }, masses = Lebesgue volume per 128^3 output cell.

Key representation (docs/METHOD.md, section 3).  Fix a source region R on which every face plane of component c has
a constant sign (a fixed *aspect*: front faces F, back faces B).  For a silhouette edge e = (f in F, b in B) define
    tau_e(p) = f_b(p) / f_f(p).
For a receiver x behind every front face (f_i(x) < 0, i in F - "gated"),
    x in S_c(s)  <=>  tau_e(s) <= tau_e(x)  for every silhouette edge e
(this is Lemma 1 with the bilinear row  f_f(s) f_b(x) - f_b(s) f_f(x) <= 0).  Every receiver of the window is
therefore an *orthant* in the common coordinates (tau_e(s))_e, which gives:
  * a dark receiver that only c can cover over R requires  tau_e(s) <= tau_e(x)  for all e: a set of dark receivers
    collapses EXACTLY to |E_c| half-spaces  tau_e(s) <= min_x tau_e(x);
  * a lit receiver with a single undecided edge e requires  tau_e(s) > tau_e(x): a set of them collapses to one
    half-space per edge;
  * tau_e is quasi-linear, so over a pure receiver block (box of voxel centres, f_f < 0 on it) its extremes are at
    the 8 corner receivers: whole blocks are decided or collapsed without visiting their receivers.
Only lit receivers with >= 2 undecided edges and dark receivers coverable by >= 2 components remain as genuinely
non-convex constraints; regions are split on their planes (BSP) until none remain, and the remaining convex
polytope is IN (exact volume).

Stage 1 (coarse, optional accelerator): octree to the 128^3 cell level, boxes refuted by sentinel receivers with
the validated generic box test (rpl.engine2.receiver_status).  Stage 1 never certifies IN.
Stage 2: per surviving cell, BSP over convex polytopes with the full receiver set (pure pyramid blocks).

Every sign decision uses explicit margins; undecided comparisons never certify (they only cause more splitting).
Records: IN (lower = upper = polytope volume), UNDET (lower 0, upper = region volume).
"""
import numpy as np
from numba import njit

from rpl.engine2 import receiver_status, MAXROWS
from rpl.predicates import plane_box_range, hull_misses_component, AMARG
from .poly import set_box, clip, volume, referenced, copy_poly, EMPTY, OVERFLOW

COVER, NOCOVER, PARTIAL, AMBG = 0, 1, 2, 3
DARK, LIT = 0, 1
IN_REC, UNDET_REC = 0, 2
CELL_DEPTH = 7

STAT_NAMES = ['nodes', 'out', 'in', 'undet', 'face_splits', 'tau_splits', 'window_splits', 'geo_splits',
              'entry_evals', 'max_entries', 'passes', 'clips', 'overflow', 'ambig_evals', 'budget_undet',
              'cells', 'cells_out', 'cells_in_any', 'comp_evals', 'expansions']
S_NODES, S_OUT, S_IN, S_UNDET, S_FACE, S_TAU, S_WIN, S_GEO, S_EVALS, S_MAXE, S_PASS, S_CLIP, S_OVF, S_AMB, \
    S_BUDGET, S_CELLS, S_CELLS_OUT, S_CELLS_IN, S_COMP, S_EXP = range(20)


# ---------------------------------------------------------------------------------------------------- stage 1
@njit(cache=True)
def stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, depth, n_learn, max_tests, cand, stats):
    """Coarse octree to `depth`; returns the number of candidate cells written to cand[:, :3]."""
    kmax = P.shape[1]
    alpha = np.empty(kmax)
    rows = np.empty((MAXROWS, 4))
    cut_idx = np.empty(MAXROWS, np.int64)
    ulo = np.empty(3); uhi = np.empty(3)
    stack = np.empty((8 * (depth + 2), 4), np.int64)
    stack[0, 0] = 0; stack[0, 1] = 0; stack[0, 2] = 0; stack[0, 3] = 0
    sp = 1
    ring = np.full(max(n_learn, 1), -1, np.int64)
    nring = 0
    ns = SX.shape[0]
    lo = np.empty(3); hi = np.empty(3)
    ncand = 0
    while sp > 0:
        sp -= 1
        d = stack[sp, 0]; ix = stack[sp, 1]; iy = stack[sp, 2]; iz = stack[sp, 3]
        stats[0] += 1
        h = 1.0 / (1 << d)
        lo[0] = ix * h; lo[1] = iy * h; lo[2] = iz * h
        hi[0] = lo[0] + h; hi[1] = lo[1] + h; hi[2] = lo[2] + h
        inside = True
        for k in range(3):
            if lo[k] < wlo[k] or hi[k] > whi[k]:
                inside = False
        if inside:
            stats[1] += 1
            continue
        ref = -1
        for t in range(nring):
            r = ring[t]
            stats[2] += 1
            if receiver_status(lo, hi, SX[r], SL[r] == 1, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo,
                               uhi) == 1:
                ref = r
                for u in range(t, 0, -1):
                    ring[u] = ring[u - 1]
                ring[0] = r
                break
        if ref < 0:
            for r in range(min(ns, max_tests)):
                inring = False
                for t in range(nring):
                    if ring[t] == r:
                        inring = True
                        break
                if inring:
                    continue
                stats[2] += 1
                if receiver_status(lo, hi, SX[r], SL[r] == 1, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx,
                                   ulo, uhi) == 1:
                    ref = r
                    if nring < ring.shape[0]:
                        nring += 1
                    for u in range(nring - 1, 0, -1):
                        ring[u] = ring[u - 1]
                    ring[0] = r
                    break
        if ref >= 0:
            stats[1] += 1
            continue
        if d == depth:
            cand[ncand, 0] = ix; cand[ncand, 1] = iy; cand[ncand, 2] = iz
            ncand += 1
            continue
        for c8 in range(8):
            stack[sp, 0] = d + 1
            stack[sp, 1] = 2 * ix + (c8 & 1)
            stack[sp, 2] = 2 * iy + ((c8 >> 1) & 1)
            stack[sp, 3] = 2 * iz + ((c8 >> 2) & 1)
            sp += 1
    return ncand


# ---------------------------------------------------------------------------------------------------- helpers
@njit(cache=True)
def item_box(code, o, N, qlo, qhi):
    l = (code >> 48) & 0xFFFF
    bx = (code >> 32) & 0xFFFF
    by = (code >> 16) & 0xFFFF
    bz = code & 0xFFFF
    s = np.int64(1) << l
    inv = 1.0 / N
    qlo[0] = (o[0] + bx * s + 0.5) * inv
    qlo[1] = (o[1] + by * s + 0.5) * inv
    qlo[2] = (o[2] + bz * s + 0.5) * inv
    qhi[0] = (o[0] + (bx + 1) * s - 0.5) * inv
    qhi[1] = (o[1] + (by + 1) * s - 0.5) * inv
    qhi[2] = (o[2] + (bz + 1) * s - 0.5) * inv
    return l


@njit(cache=True)
def fval(P, c, i, x0, x1, x2):
    return P[c, i, 0] * x0 + P[c, i, 1] * x1 + P[c, i, 2] * x2 + P[c, i, 3]


@njit(cache=True)
def aspect_front(Vr, nr, P, K, c, asp_row, front_row):
    """Fill front_row; returns (crossing face index or -1, number of front faces)."""
    nfr = 0
    for i in range(K[c]):
        a = asp_row[i]
        if a == 1:
            front_row[i] = True
            nfr += 1
            continue
        if a == 2:
            front_row[i] = False
            continue
        mn = np.inf; mx = -np.inf
        for v in range(nr):
            f = fval(P, c, i, Vr[v, 0], Vr[v, 1], Vr[v, 2])
            if f < mn:
                mn = f
            if f > mx:
                mx = f
        if mn > 1e-11:
            front_row[i] = True
            nfr += 1
        elif mx < -1e-11:
            front_row[i] = False
        else:
            return i, nfr
    return -1, nfr


@njit(cache=True)
def sil_edges(adj, aoff, c, front_row, silf, silb, silt):
    n = 0
    for t in range(aoff[c], aoff[c + 1]):
        i = adj[t, 1]; j = adj[t, 2]
        if front_row[i] and not front_row[j]:
            silf[n] = i; silb[n] = j; silt[n] = t; n += 1
        elif front_row[j] and not front_row[i]:
            silf[n] = j; silb[n] = i; silt[n] = t; n += 1
    return n


@njit(cache=True)
def tau_range(Vr, nr, P, c, f, b, Lbt, Ubt, m=1.0):
    """Range over the region of coord(s) = m f_b(s)/f_f(s); m = +1 (source-side aspect, f_f > 0 on R) or
    m = -1 (window-side aspect, f_f < 0 on R).  Widened by a rounding tolerance, then intersected with the
    region's logical bounds."""
    lo = np.inf; hi = -np.inf
    for v in range(nr):
        ff = fval(P, c, f, Vr[v, 0], Vr[v, 1], Vr[v, 2])
        fb = fval(P, c, b, Vr[v, 0], Vr[v, 1], Vr[v, 2])
        if m * ff <= 1e-13:
            if fb <= 0.0:
                lo = -np.inf
            if fb >= 0.0:
                hi = np.inf
            continue
        t = m * fb / ff
        tol = 1e-12 * (1.0 + abs(t)) / abs(ff)
        if t - tol < lo:
            lo = t - tol
        if t + tol > hi:
            hi = t + tol
    if Lbt > lo:
        lo = Lbt
    if Ubt < hi:
        hi = Ubt
    return lo, hi


@njit(cache=True)
def eval_comp(qlo, qhi, ispt, P, K, c, front_row, nsil, silf, silb, tlo, thi, tmin, tmax):
    """Status of receiver box Q w.r.t. component c over the region (fixed aspect).
    COVER: every receiver of Q in S_c(s) for every s in R.  NOCOVER: none, for any s.  PARTIAL (n_und undecided
    edges, last one kund; tmin/tmax filled).  AMBG: gating not decidable (receiver within AMARG of a front plane)."""
    amb = False
    for i in range(K[c]):
        if not front_row[i]:
            continue
        mn, mx = plane_box_range(P[c, i, 0], P[c, i, 1], P[c, i, 2], P[c, i, 3], qlo, qhi)
        if mn > AMARG:
            return NOCOVER, 0, -1
        if mx >= -AMARG:
            amb = True
    if amb:
        return AMBG, 0, -1
    nund = 0
    kund = -1
    for k in range(nsil):
        f = silf[k]; b = silb[k]
        if ispt:
            ff = fval(P, c, f, qlo[0], qlo[1], qlo[2])
            fb = fval(P, c, b, qlo[0], qlo[1], qlo[2])
            tm = fb / ff
            tM = tm
        else:
            tm = np.inf; tM = -np.inf
            for cr in range(8):
                x0 = qhi[0] if cr & 1 else qlo[0]
                x1 = qhi[1] if cr & 2 else qlo[1]
                x2 = qhi[2] if cr & 4 else qlo[2]
                t = fval(P, c, b, x0, x1, x2) / fval(P, c, f, x0, x1, x2)
                if t < tm:
                    tm = t
                if t > tM:
                    tM = t
        tmin[k] = tm; tmax[k] = tM
        if tlo[k] >= tM:
            # tau_e(s) >= tau_e(x) on R: x can be covered only on the plane tau_e(s) = tau_e(x) (measure zero)
            return NOCOVER, 0, -1
        if thi[k] <= tm:
            continue
        nund += 1
        kund = k
    if nund == 0:
        return COVER, 0, -1
    return PARTIAL, nund, kund


@njit(cache=True)
def prep_comp(c, front_row, K, nsil, silf, silb, uf, silfi, silbi):
    """Face list for fast evaluation: front faces first, then the back faces of silhouette edges."""
    n = 0
    for i in range(K[c]):
        if front_row[i]:
            uf[n] = i
            n += 1
    nfr = n
    for k in range(nsil):
        b = silb[k]
        pos = -1
        for u in range(nfr, n):
            if uf[u] == b:
                pos = u
                break
        if pos < 0:
            uf[n] = b
            pos = n
            n += 1
        silbi[k] = pos
        f = silf[k]
        for u in range(nfr):
            if uf[u] == f:
                silfi[k] = u
                break
    return n, nfr


@njit(cache=True)
def prep_comp_w(nsil, silf, silb, uf, silfi, silbi):
    n = 0
    for k in range(nsil):
        for side in range(2):
            fc = silf[k] if side == 0 else silb[k]
            pos = -1
            for u in range(n):
                if uf[u] == fc:
                    pos = u
                    break
            if pos < 0:
                uf[n] = fc
                pos = n
                n += 1
            if side == 0:
                silfi[k] = pos
            else:
                silbi[k] = pos
    return n, 0


@njit(cache=True)
def eval_comp2(qlo, qhi, ispt, P, c, nuf, nfr, uf, nsil, silfi, silbi, tlo, thi, tmin, tmax, fbuf, m=1.0):
    """Same decisions as eval_comp (tested equal), with each face evaluated once per corner."""
    d0 = qhi[0] - qlo[0]; d1 = qhi[1] - qlo[1]; d2 = qhi[2] - qlo[2]
    amb = False
    for u in range(nuf):
        if u == nfr:
            if amb:
                return AMBG, 0, -1
        i = uf[u]
        base = P[c, i, 0] * qlo[0] + P[c, i, 1] * qlo[1] + P[c, i, 2] * qlo[2] + P[c, i, 3]
        if ispt:
            fbuf[u, 0] = base
            mn = base; mx = base
        else:
            a = P[c, i, 0] * d0; b = P[c, i, 1] * d1; g = P[c, i, 2] * d2
            mn = np.inf; mx = -np.inf
            for cr in range(8):
                v = base
                if cr & 1:
                    v += a
                if cr & 2:
                    v += b
                if cr & 4:
                    v += g
                fbuf[u, cr] = v
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        if u < nfr:
            if mn > AMARG:
                return NOCOVER, 0, -1
            if mx >= -AMARG:
                amb = True
    if amb:
        return AMBG, 0, -1
    nc = 1 if ispt else 8
    nund = 0
    kund = -1
    for k in range(nsil):
        fu = silfi[k]; bu = silbi[k]
        tm = np.inf; tM = -np.inf
        for cr in range(nc):
            t = m * fbuf[bu, cr] / fbuf[fu, cr]
            if t < tm:
                tm = t
            if t > tM:
                tM = t
        tmin[k] = tm; tmax[k] = tM
        if tlo[k] >= tM:
            return NOCOVER, 0, -1
        if thi[k] <= tm:
            continue
        nund += 1
        kund = k
    if nund == 0:
        return COVER, 0, -1
    return PARTIAL, nund, kund


@njit(cache=True)
def _receiver_limited(nsil, tlo, thi, tmn, tmx, rho):
    """Dual-tree rule: expand a block only if, on some undecided edge, its own tau spread is comparable to the
    region's (otherwise refining the source region is what can decide it)."""
    for k in range(nsil):
        if thi[k] <= tmn[k] or tlo[k] >= tmx[k]:
            continue
        wr = thi[k] - tlo[k]
        if not np.isfinite(wr) or (tmx[k] - tmn[k]) >= rho * wr:
            return True
    return False


@njit(cache=True)
def _bbox(Vr, nr, lo, hi):
    for k in range(3):
        lo[k] = np.inf; hi[k] = -np.inf
    for v in range(nr):
        for k in range(3):
            if Vr[v, k] < lo[k]:
                lo[k] = Vr[v, k]
            if Vr[v, k] > hi[k]:
                hi[k] = Vr[v, k]


@njit(cache=True)
def _copy_slot(p, q, SV, SFc, SFn, Snv, Snf, Sasp, SLb, SUb, Sdep):
    copy_poly(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], SV[q], SFc[q], SFn[q])
    Snv[q] = Snv[p]; Snf[q] = Snf[p]
    Sasp[q, :, :] = Sasp[p, :, :]
    SLb[q, :] = SLb[p, :]
    SUb[q, :] = SUb[p, :]
    Sdep[q] = Sdep[p]


@njit(cache=True)
def _clip_slot(q, a0, a1, a2, b, SV, SFc, SFn, Snv, Snf):
    st, nv, nf = clip(SV[q], SFc[q], SFn[q], Snv[q], Snf[q], a0, a1, a2, b)
    if st == EMPTY:
        Snf[q] = 0
        Snv[q] = 0
        return 0
    if st == OVERFLOW:
        return 2
    Snv[q] = nv; Snf[q] = nf
    return 1



@njit(cache=True)
def _cell_range(Vr, nr, i0, i1):
    G = 1 << CELL_DEPTH
    for k in range(3):
        mn = np.inf; mx = -np.inf
        for v in range(nr):
            if Vr[v, k] < mn:
                mn = Vr[v, k]
            if Vr[v, k] > mx:
                mx = Vr[v, k]
        a = int(np.floor(mn * G + 1e-9)); b = int(np.floor(mx * G - 1e-9))
        a = min(max(a, 0), G - 1); b = min(max(b, 0), G - 1)
        if b < a:
            b = a
        i0[k] = a; i1[k] = b


@njit(cache=True)
def _dump(Vsrc, nvs, rec_voff, rec_vn, vpool, vtop, nrec):
    if vtop[0] + nvs > vpool.shape[0]:
        rec_voff[nrec] = -1; rec_vn[nrec] = 0
        return
    rec_voff[nrec] = vtop[0]; rec_vn[nrec] = nvs
    for v in range(nvs):
        vpool[vtop[0] + v, 0] = Vsrc[v, 0]; vpool[vtop[0] + v, 1] = Vsrc[v, 1]; vpool[vtop[0] + v, 2] = Vsrc[v, 2]
    vtop[0] += nvs


@njit(cache=True)
def emit_region(p, kind, SV, SFc, SFn, Snv, Snf, scr0, Vr, rec_cell, rec_lo, rec_hi, rec_kind, nrec,
                rec_voff, rec_vn, vpool, vtop):
    """Record region p (IN: lower = upper = volume; UNDET: lower 0) split exactly over the 128^3 cells by a
    recursive BSP on the cell grid (cost ~ number of cells actually met, not of the bounding box)."""
    G = 1 << CELL_DEPTH
    i0 = np.empty(3, np.int64); i1 = np.empty(3, np.int64)
    copy_poly(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], SV[scr0], SFc[scr0], SFn[scr0])
    Snv[scr0] = Snv[p]; Snf[scr0] = Snf[p]
    top = 1
    nscr = SV.shape[0] - scr0
    while top > 0:
        top -= 1
        q = scr0 + top
        if Snf[q] == 0:
            continue
        nr = referenced(SV[q], SFc[q], SFn[q], Snv[q], Snf[q], Vr)
        _cell_range(Vr, nr, i0, i1)
        ax = -1; span = 0
        for k in range(3):
            if i1[k] - i0[k] > span:
                span = i1[k] - i0[k]; ax = k
        if ax < 0 or top + 2 > nscr:
            vol = volume(SV[q], SFc[q], SFn[q], Snf[q])
            if vol <= 0.0:
                continue
            if ax < 0:
                rec_cell[nrec] = (i0[0] * G + i0[1]) * G + i0[2]
                rec_lo[nrec] = vol if kind == IN_REC else 0.0
                rec_hi[nrec] = vol; rec_kind[nrec] = kind
                _dump(Vr, nr, rec_voff, rec_vn, vpool, vtop, nrec)
                nrec += 1
            else:
                # scratch exhausted (not expected): every met cell gets the piece volume as an upper bound
                for ix in range(i0[0], i1[0] + 1):
                    for iy in range(i0[1], i1[1] + 1):
                        for iz in range(i0[2], i1[2] + 1):
                            rec_cell[nrec] = (ix * G + iy) * G + iz; rec_lo[nrec] = 0.0
                            rec_hi[nrec] = min(vol, 1.0 / (G * G * G)); rec_kind[nrec] = UNDET_REC
                            rec_voff[nrec] = -1; rec_vn[nrec] = 0
                            nrec += 1
            continue
        lo_ = 1e300; hi_ = -1e300
        for v in range(nr):
            if Vr[v, ax] < lo_:
                lo_ = Vr[v, ax]
            if Vr[v, ax] > hi_:
                hi_ = Vr[v, ax]
        j = int(np.floor(0.5 * (lo_ + hi_) * G + 0.5))
        if j < i0[ax] + 1:
            j = i0[ax] + 1
        if j > i1[ax]:
            j = i1[ax]
        a0 = 1.0 if ax == 0 else 0.0; a1 = 1.0 if ax == 1 else 0.0; a2 = 1.0 if ax == 2 else 0.0
        copy_poly(SV[q], SFc[q], SFn[q], Snv[q], Snf[q], SV[q + 1], SFc[q + 1], SFn[q + 1])
        Snv[q + 1] = Snv[q]; Snf[q + 1] = Snf[q]
        vol_q = volume(SV[q], SFc[q], SFn[q], Snf[q])
        rA = _clip_slot(q, a0, a1, a2, -j / G, SV, SFc, SFn, Snv, Snf)
        rB = _clip_slot(q + 1, -a0, -a1, -a2, j / G, SV, SFc, SFn, Snv, Snf)
        if rA == 2 or rB == 2:
            for ix in range(i0[0], i1[0] + 1):
                for iy in range(i0[1], i1[1] + 1):
                    for iz in range(i0[2], i1[2] + 1):
                        rec_cell[nrec] = (ix * G + iy) * G + iz; rec_lo[nrec] = 0.0
                        rec_hi[nrec] = min(vol_q, 1.0 / (G * G * G)); rec_kind[nrec] = UNDET_REC
                        rec_voff[nrec] = -1; rec_vn[nrec] = 0
                        nrec += 1
            continue
        top += 2
    return nrec


# ---------------------------------------------------------------------------------------------------- stage 2
@njit(cache=True)
def stage2(P, K, clo, chi, adj, aoff, o, N, L, items_code, items_cls, roots, nroots,
           max_nodes_cell, max_nodes_total, max_depth, max_pass, tau_first, dual, rho, wmode, wfront,
           rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals, face_by_comp, face_surv):
    C = P.shape[0]
    kmax = P.shape[1]
    emax = 1
    for c in range(C):
        if aoff[c + 1] - aoff[c] > emax:
            emax = aoff[c + 1] - aoff[c]
    etot = adj.shape[0]
    NSCR = 64
    S = 2 * max_depth + 8 + NSCR
    SCR = S - NSCR
    MV = 160; MF = 72; MPV = 48
    SV = np.empty((S, MV, 3)); SFc = np.empty((S, MF, MPV), np.int32); SFn = np.empty((S, MF), np.int32)
    Snv = np.zeros(S, np.int64); Snf = np.zeros(S, np.int64)
    Sasp = np.zeros((S, C, kmax), np.int8)
    SLb = np.empty((S, max(etot, 1))); SUb = np.empty((S, max(etot, 1)))
    Sdep = np.zeros(S, np.int64)
    Soff = np.zeros(S, np.int64); Slen = np.zeros(S, np.int64)
    NE = 1 << 22
    Pc = np.empty(NE, np.int64); Pk = np.empty(NE, np.int8); Pm = np.empty(NE, np.int64); Pa = np.empty(NE, np.int64)
    NQ = 1 << 21
    Wc = np.empty(NQ, np.int64); Wk = np.empty(NQ, np.int8); Wm = np.empty(NQ, np.int64); Wa = np.empty(NQ, np.int64)
    Qc = np.empty(NQ, np.int64); Qk = np.empty(NQ, np.int8); Qm = np.empty(NQ, np.int64); Qa = np.empty(NQ, np.int64)
    RING = 64
    ring = np.full(RING, -1, np.int64)
    nring = 0
    inring = np.zeros(max(items_code.shape[0], 1), np.bool_)
    Vr = np.empty((MV, 3))
    front = np.zeros((C, kmax), np.bool_)
    nsil = np.zeros(C, np.int64)
    silf = np.zeros((C, emax), np.int64); silb = np.zeros((C, emax), np.int64); silt = np.zeros((C, emax), np.int64)
    tlo = np.zeros((C, emax)); thi = np.zeros((C, emax))
    tmn = np.zeros((C, emax)); tmx = np.zeros((C, emax))
    Unew = np.zeros((C, emax)); Lnew = np.zeros((C, emax))
    xface = np.zeros(C, np.int64)
    mult = np.ones(C)
    for c in range(C):
        if wmode[c]:
            mult[c] = -1.0
    uf = np.zeros((C, kmax), np.int64); nuf = np.zeros(C, np.int64); nfrc = np.zeros(C, np.int64)
    silfi = np.zeros((C, emax), np.int64); silbi = np.zeros((C, emax), np.int64)
    fbuf = np.zeros((kmax, 8))
    qlo = np.empty(3); qhi = np.empty(3)
    Rlo = np.empty(3); Rhi = np.empty(3)
    wlo = np.empty(3); whi = np.empty(3)
    Wqlo = np.empty(3); Wqhi = np.empty(3)
    for k in range(3):
        wlo[k] = o[k] / N; whi[k] = (o[k] + L) / N
        Wqlo[k] = (o[k] + 0.5) / N; Wqhi[k] = (o[k] + L - 0.5) / N
    allmask = (np.int64(1) << C) - 1
    nitems = items_code.shape[0]
    nrec = 0
    cs = 1.0 / (1 << CELL_DEPTH)
    i0 = np.empty(3, np.int64); i1 = np.empty(3, np.int64)
    G = 1 << CELL_DEPTH
    for ci in range(nroots):
        rd = roots[ci, 0]; ix = roots[ci, 1]; iy = roots[ci, 2]; iz = roots[ci, 3]
        stats[S_CELLS] += 1
        rec_start = nrec
        ev0 = stats[S_EVALS]
        # root region = a box of the stage-1 octree
        hh = 1.0 / (1 << rd)
        qlo[0] = ix * hh; qlo[1] = iy * hh; qlo[2] = iz * hh
        qhi[0] = qlo[0] + hh; qhi[1] = qlo[1] + hh; qhi[2] = qlo[2] + hh
        nv, nf = set_box(SV[0], SFc[0], SFn[0], qlo, qhi)
        Snv[0] = nv; Snf[0] = nf
        Sasp[0, :, :] = 0
        SLb[0, :] = -np.inf; SUb[0, :] = np.inf
        Sdep[0] = 0
        # learned refuters first (top-level items that refuted earlier regions), then the rest
        npush = 0
        for t in range(nring):
            a = ring[t]
            inring[a] = True
            Pc[npush] = items_code[a]; Pk[npush] = items_cls[a]; Pm[npush] = allmask; Pa[npush] = a
            npush += 1
        for t in range(nitems):
            if inring[t]:
                continue
            Pc[npush] = items_code[t]; Pk[npush] = items_cls[t]; Pm[npush] = allmask; Pa[npush] = t
            npush += 1
        for t in range(nring):
            inring[ring[t]] = False
        Soff[0] = 0; Slen[0] = nitems
        ptop = nitems
        sp = 1
        nodes_cell = 0
        while sp > 0:
            sp -= 1
            p = sp
            if Snf[p] == 0:
                ptop = Soff[p]
                continue
            if stats[S_NODES] >= max_nodes_total:
                # declared global budget exhausted: every remaining region is reported undetermined (never dropped)
                nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                   rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                stats[S_UNDET] += 1
                stats[S_BUDGET] += 1
                ptop = Soff[p]
                continue
            stats[S_NODES] += 1
            nodes_cell += 1
            ne = Slen[p]
            off = Soff[p]
            for t in range(ne):
                Wc[t] = Pc[off + t]; Wk[t] = Pk[off + t]; Wm[t] = Pm[off + t]; Wa[t] = Pa[off + t]
            ptop = off
            nr = referenced(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], Vr)
            _bbox(Vr, nr, Rlo, Rhi)
            split_kind = -1   # 0 window, 1 face, 2 tau, 3 geometric
            sa0 = 0.0; sa1 = 0.0; sa2 = 0.0; sb = 0.0
            sc = -1; si = -1; st_t = -1; sv = 0.0
            # ---- window box (sources inside B(W) are excluded by definition)
            inside = True
            disjoint = False
            for k in range(3):
                if Rlo[k] < wlo[k] - 1e-12 or Rhi[k] > whi[k] + 1e-12:
                    inside = False
                if Rhi[k] <= wlo[k] + 1e-12 or Rlo[k] >= whi[k] - 1e-12:
                    disjoint = True
            if inside:
                stats[S_OUT] += 1
                continue
            if not disjoint:
                for k in range(3):
                    if split_kind >= 0:
                        break
                    for side in range(2):
                        v = wlo[k] if side == 0 else whi[k]
                        if Rlo[k] < v - 1e-12 and Rhi[k] > v + 1e-12:
                            split_kind = 0
                            sa0 = 1.0 if k == 0 else 0.0
                            sa1 = 1.0 if k == 1 else 0.0
                            sa2 = 1.0 if k == 2 else 0.0
                            sb = -v
                            break
            # ---- relevance and aspects
            relmask = np.int64(0)
            Xmask = np.int64(0)
            out = False
            if split_kind < 0:
                for c in range(C):
                    if not hull_misses_component(Rlo, Rhi, Wqlo, Wqhi, P, K, c, clo, chi):
                        relmask |= np.int64(1) << c
                for c in range(C):
                    if not (relmask >> c) & 1:
                        continue
                    if wmode[c]:
                        # window-side aspect: c can shadow W from s only if s is behind every window-facing face
                        gx = -1
                        outside = False
                        for i in range(K[c]):
                            if not wfront[c, i]:
                                continue
                            a = Sasp[p, c, i]
                            if a == 2:
                                continue
                            if a == 1:
                                outside = True
                                break
                            mn = np.inf; mx = -np.inf
                            for v in range(nr):
                                fv = fval(P, c, i, Vr[v, 0], Vr[v, 1], Vr[v, 2])
                                if fv < mn:
                                    mn = fv
                                if fv > mx:
                                    mx = fv
                            if mn > 1e-11:
                                outside = True
                                break
                            if mx >= -1e-11 and gx < 0:
                                gx = i
                        if outside:
                            relmask &= ~(np.int64(1) << c)
                        elif gx >= 0:
                            Xmask |= np.int64(1) << c
                            xface[c] = gx
                        continue
                    cross, nfr = aspect_front(Vr, nr, P, K, c, Sasp[p, c], front[c])
                    if cross >= 0:
                        # aspect of c not fixed on R: c is handled by the aspect-free hull test for now and the
                        # region is split on this face plane only if some surviving constraint still needs c
                        Xmask |= np.int64(1) << c
                        xface[c] = cross
                        continue
                    if nfr == 0:
                        out = True   # region inside component c (source in the solid)
                        break
            if out:
                stats[S_OUT] += 1
                continue
            ns = 0
            if split_kind < 0:
                for c in range(C):
                    if not (relmask >> c) & 1 or (Xmask >> c) & 1:
                        nsil[c] = 0
                        continue
                    if wmode[c]:
                        nsil[c] = sil_edges(adj, aoff, c, wfront[c], silf[c], silb[c], silt[c])
                        nuf[c], nfrc[c] = prep_comp_w(nsil[c], silf[c], silb[c], uf[c], silfi[c], silbi[c])
                    else:
                        nsil[c] = sil_edges(adj, aoff, c, front[c], silf[c], silb[c], silt[c])
                        nuf[c], nfrc[c] = prep_comp(c, front[c], K, nsil[c], silf[c], silb[c], uf[c], silfi[c], silbi[c])
                    for k in range(nsil[c]):
                        t = silt[c, k]
                        tlo[c, k], thi[c, k] = tau_range(Vr, nr, P, c, silf[c, k], silb[c, k], SLb[p, t], SUb[p, t],
                                                         mult[c])
                # ---- evaluation passes
                for it in range(max_pass):
                    stats[S_PASS] += 1
                    for c in range(C):
                        if (relmask >> c) & 1:
                            for k in range(nsil[c]):
                                Unew[c, k] = np.inf; Lnew[c, k] = -np.inf
                    nq = ne
                    ns = 0
                    i = 0
                    overflow = False
                    while i < nq:
                        code = Wc[i]; kls = Wk[i]; mask = Wm[i] & relmask; anc = Wa[i]
                        i += 1
                        stats[S_EVALS] += 1
                        lvl = item_box(code, o, N, qlo, qhi)
                        ispt = lvl == 0
                        keep = False
                        newmask = np.int64(0)
                        xunk = False
                        rexp = not dual
                        if kls == DARK:
                            cover = False
                            ncand = 0
                            uc = -1
                            amb = False
                            for c in range(C):
                                if not (mask >> c) & 1:
                                    continue
                                stats[S_COMP] += 1
                                if (Xmask >> c) & 1:
                                    if hull_misses_component(Rlo, Rhi, qlo, qhi, P, K, c, clo, chi):
                                        continue
                                    newmask |= np.int64(1) << c
                                    ncand += 1
                                    amb = True
                                    xunk = True
                                    continue
                                st, nund, kund = eval_comp2(qlo, qhi, ispt, P, c, nuf[c], nfrc[c], uf[c], nsil[c], silfi[c],
                                                            silbi[c], tlo[c], thi[c], tmn[c], tmx[c], fbuf,
                                                            mult[c])
                                if st == COVER:
                                    cover = True
                                    break
                                if st == NOCOVER:
                                    continue
                                if st == AMBG:
                                    amb = True
                                    rexp = True
                                    stats[S_AMB] += 1
                                elif not ispt and not rexp:
                                    rexp = _receiver_limited(nsil[c], tlo[c], thi[c], tmn[c], tmx[c], rho)
                                newmask |= np.int64(1) << c
                                ncand += 1
                                uc = c
                            if cover:
                                continue
                            if ncand == 0:
                                out = True
                                break
                            if ncand == 1 and not amb:
                                for k in range(nsil[uc]):
                                    if tmn[uc, k] < thi[uc, k] and tmn[uc, k] < Unew[uc, k]:
                                        Unew[uc, k] = tmn[uc, k]
                                continue
                            keep = True
                        else:
                            for c in range(C):
                                if not (mask >> c) & 1:
                                    continue
                                stats[S_COMP] += 1
                                if (Xmask >> c) & 1:
                                    if hull_misses_component(Rlo, Rhi, qlo, qhi, P, K, c, clo, chi):
                                        continue
                                    newmask |= np.int64(1) << c
                                    xunk = True
                                    continue
                                st, nund, kund = eval_comp2(qlo, qhi, ispt, P, c, nuf[c], nfrc[c], uf[c], nsil[c], silfi[c],
                                                            silbi[c], tlo[c], thi[c], tmn[c], tmx[c], fbuf,
                                                            mult[c])
                                if st == COVER:
                                    out = True
                                    break
                                if st == NOCOVER:
                                    continue
                                if st == PARTIAL and nund == 1:
                                    if tmx[c, kund] > Lnew[c, kund]:
                                        Lnew[c, kund] = tmx[c, kund]
                                    continue
                                if st == AMBG:
                                    rexp = True
                                    stats[S_AMB] += 1
                                elif not ispt and not rexp:
                                    rexp = _receiver_limited(nsil[c], tlo[c], thi[c], tmn[c], tmx[c], rho)
                                newmask |= np.int64(1) << c
                            if out:
                                break
                            if newmask == 0:
                                continue
                            keep = True
                        if keep:
                            if not ispt and not xunk and rexp:
                                # expand the pure block into its 8 children (same class)
                                if nq + 8 > NQ:
                                    overflow = True
                                    break
                                stats[S_EXP] += 1
                                l1 = lvl - 1
                                bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
                                for c8 in range(8):
                                    Wc[nq] = (np.int64(l1) << 48) | (np.int64(2 * bx + (c8 & 1)) << 32) | \
                                             (np.int64(2 * by + ((c8 >> 1) & 1)) << 16) | np.int64(2 * bz + ((c8 >> 2) & 1))
                                    Wk[nq] = kls
                                    Wm[nq] = newmask
                                    Wa[nq] = anc
                                    nq += 1
                            else:
                                if ns >= NQ:
                                    overflow = True
                                    break
                                Qc[ns] = code; Qk[ns] = kls; Qm[ns] = newmask; Qa[ns] = anc
                                ns += 1
                    if nq > stats[S_MAXE]:
                        stats[S_MAXE] = nq
                    if out:
                        # move the refuting top-level item to the front of the learned ring
                        pos = -1
                        for t in range(nring):
                            if ring[t] == anc:
                                pos = t
                                break
                        if pos < 0:
                            if nring < RING:
                                nring += 1
                            pos = nring - 1
                        for t in range(pos, 0, -1):
                            ring[t] = ring[t - 1]
                        ring[0] = anc
                        break
                    if overflow:
                        stats[S_OVF] += 1
                        break
                    # ---- apply collapsed half-spaces
                    nclip = 0
                    for c in range(C):
                        if out:
                            break
                        if not (relmask >> c) & 1:
                            continue
                        for k in range(nsil[c]):
                            f = silf[c, k]; b = silb[c, k]; t = silt[c, k]
                            u = Unew[c, k]
                            if u < thi[c, k]:
                                if u < tlo[c, k]:
                                    out = True
                                    break
                                mu = mult[c] * u
                                r = _clip_slot(p, P[c, b, 0] - mu * P[c, f, 0], P[c, b, 1] - mu * P[c, f, 1],
                                               P[c, b, 2] - mu * P[c, f, 2], P[c, b, 3] - mu * P[c, f, 3],
                                               SV, SFc, SFn, Snv, Snf)
                                stats[S_CLIP] += 1
                                nclip += 1
                                if r == 0:
                                    out = True
                                    break
                                if r == 2:
                                    overflow = True
                                    break
                                if u < SUb[p, t]:
                                    SUb[p, t] = u
                            lv = Lnew[c, k]
                            if lv > tlo[c, k]:
                                if lv > thi[c, k]:
                                    out = True
                                    break
                                ml = mult[c] * lv
                                r = _clip_slot(p, ml * P[c, f, 0] - P[c, b, 0], ml * P[c, f, 1] - P[c, b, 1],
                                               ml * P[c, f, 2] - P[c, b, 2], ml * P[c, f, 3] - P[c, b, 3],
                                               SV, SFc, SFn, Snv, Snf)
                                stats[S_CLIP] += 1
                                nclip += 1
                                if r == 0:
                                    out = True
                                    break
                                if r == 2:
                                    overflow = True
                                    break
                                if lv > SLb[p, t]:
                                    SLb[p, t] = lv
                        if overflow:
                            break
                    if out or overflow:
                        break
                    if nclip == 0:
                        break
                    # region shrank: refresh tau ranges and re-evaluate the survivors
                    nr = referenced(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], Vr)
                    _bbox(Vr, nr, Rlo, Rhi)
                    for c in range(C):
                        if not (relmask >> c) & 1:
                            continue
                        for k in range(nsil[c]):
                            t = silt[c, k]
                            tlo[c, k], thi[c, k] = tau_range(Vr, nr, P, c, silf[c, k], silb[c, k], SLb[p, t],
                                                             SUb[p, t], mult[c])
                    for t in range(ns):
                        Wc[t] = Qc[t]; Wk[t] = Qk[t]; Wm[t] = Qm[t]; Wa[t] = Qa[t]
                    ne = ns
                if out:
                    stats[S_OUT] += 1
                    continue
                if overflow:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                    stats[S_UNDET] += 1
                    continue
                if ns == 0:
                    nrec = emit_region(p, IN_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                    stats[S_IN] += 1
                    continue
                # ---- a surviving constraint involves a component whose aspect is not fixed: split on its face
                need = np.int64(0)
                for q in range(ns):
                    need |= Qm[q]
                need &= Xmask
                if need != 0:
                    for c in range(C):
                        if (need >> c) & 1:
                            split_kind = 1
                            sc = c; si = xface[c]
                            sa0 = P[c, si, 0]; sa1 = P[c, si, 1]; sa2 = P[c, si, 2]; sb = P[c, si, 3]
                            break
                # ---- choose a split among the remaining (non-convex) constraints
                if split_kind == 1:
                    pass
                elif Sdep[p] >= max_depth or nodes_cell >= max_nodes_cell:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                    stats[S_UNDET] += 1
                    stats[S_BUDGET] += 1
                    continue
                if split_kind != 1:
                    # candidate tau split: the most central undecided pencil plane of a surviving receiver
                    best = 2.0
                    for q in range(ns):
                        code = Qc[q]; mask = Qm[q]
                        if (code >> 48) & 0xFFFF:
                            continue
                        item_box(code, o, N, qlo, qhi)
                        for c in range(C):
                            if not (mask >> c) & 1:
                                continue
                            st, nund, kund = eval_comp2(qlo, qhi, True, P, c, nuf[c], nfrc[c], uf[c], nsil[c], silfi[c], silbi[c],
                                                        tlo[c], thi[c], tmn[c], tmx[c], fbuf, mult[c])
                            if st != PARTIAL:
                                continue
                            for k in range(nsil[c]):
                                v = tmn[c, k]
                                if not (v < thi[c, k] and tlo[c, k] < v):
                                    continue
                                if np.isfinite(tlo[c, k]) and np.isfinite(thi[c, k]) and thi[c, k] > tlo[c, k]:
                                    score = abs((v - tlo[c, k]) / (thi[c, k] - tlo[c, k]) - 0.5)
                                else:
                                    score = 0.45
                                if score < best:
                                    best = score
                                    sc = c; st_t = k; sv = v
                    _cell_range(Vr, nr, i0, i1)
                    gax = -1
                    gext = 0
                    for k in range(3):
                        if i1[k] - i0[k] > gext:
                            gext = i1[k] - i0[k]
                            gax = k
                    use_tau = sc >= 0 and (tau_first or gax < 0)
                    if use_tau:
                        split_kind = 2
                        f = silf[sc, st_t]; b = silb[sc, st_t]
                        msv = mult[sc] * sv
                        sa0 = P[sc, b, 0] - msv * P[sc, f, 0]; sa1 = P[sc, b, 1] - msv * P[sc, f, 1]
                        sa2 = P[sc, b, 2] - msv * P[sc, f, 2]; sb = P[sc, b, 3] - msv * P[sc, f, 3]
                    elif gax >= 0:
                        # region spans several output cells: split on the cell grid (shares the filtered list)
                        mid = 0.5 * (Rlo[gax] + Rhi[gax])
                        j = int(np.floor(mid * G + 0.5))
                        if j < i0[gax] + 1:
                            j = i0[gax] + 1
                        if j > i1[gax]:
                            j = i1[gax]
                        split_kind = 3
                        sa0 = 1.0 if gax == 0 else 0.0
                        sa1 = 1.0 if gax == 1 else 0.0
                        sa2 = 1.0 if gax == 2 else 0.0
                        sb = -j / G
                    else:
                        ext = 0.0
                        ax = 0
                        for k in range(3):
                            if Rhi[k] - Rlo[k] > ext:
                                ext = Rhi[k] - Rlo[k]
                                ax = k
                        if ext < 1e-9:
                            nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo,
                                               rec_hi, rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                            stats[S_UNDET] += 1
                            continue
                        split_kind = 3
                        sa0 = 1.0 if ax == 0 else 0.0
                        sa1 = 1.0 if ax == 1 else 0.0
                        sa2 = 1.0 if ax == 2 else 0.0
                        sb = -0.5 * (Rlo[ax] + Rhi[ax])
            # ---- perform the split: child A {a.x+b <= 0} at slot p, child B {a.x+b >= 0} at slot p+1
            if split_kind < 0:
                continue
            if split_kind == 0:
                stats[S_WIN] += 1
            elif split_kind == 1:
                stats[S_FACE] += 1
                face_by_comp[sc] += 1
                face_surv[sc] += ns
            elif split_kind == 2:
                stats[S_TAU] += 1
            else:
                stats[S_GEO] += 1
            if split_kind == 0:
                # entries were not evaluated: pass the input list through
                ns = ne
                for t in range(ne):
                    Qc[t] = Wc[t]; Qk[t] = Wk[t]; Qm[t] = Wm[t]; Qa[t] = Wa[t]
            if ptop + 2 * ns > NE or p + 2 > SCR - 1:
                stats[S_OVF] += 1
                nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                   rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                stats[S_UNDET] += 1
                continue
            _copy_slot(p, p + 1, SV, SFc, SFn, Snv, Snf, Sasp, SLb, SUb, Sdep)
            rB = _clip_slot(p + 1, -sa0, -sa1, -sa2, -sb, SV, SFc, SFn, Snv, Snf)
            rA = _clip_slot(p, sa0, sa1, sa2, sb, SV, SFc, SFn, Snv, Snf)
            if rA == 2 or rB == 2:
                stats[S_OVF] += 1
                # conservative: the pieces that could not be represented stay undetermined
                if rB != 0:
                    nrec = emit_region(p + 1, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                if rA != 0:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop)
                Snf[p + 1] = 0
                stats[S_UNDET] += 1
                continue
            Sdep[p] += 1
            Sdep[p + 1] += 1
            if split_kind == 1:
                Sasp[p, sc, si] = 2
                Sasp[p + 1, sc, si] = 1
            elif split_kind == 2:
                t = silt[sc, st_t]
                if sv < SUb[p, t]:
                    SUb[p, t] = sv
                if sv > SLb[p + 1, t]:
                    SLb[p + 1, t] = sv
            Soff[p] = ptop; Slen[p] = ns
            for t in range(ns):
                Pc[ptop + t] = Qc[t]; Pk[ptop + t] = Qk[t]; Pm[ptop + t] = Qm[t]; Pa[ptop + t] = Qa[t]
            ptop += ns
            Soff[p + 1] = ptop; Slen[p + 1] = ns
            for t in range(ns):
                Pc[ptop + t] = Qc[t]; Pk[ptop + t] = Qk[t]; Pm[ptop + t] = Qm[t]; Pa[ptop + t] = Qa[t]
            ptop += ns
            sp = p + 2
        cell_nodes[ci] = nodes_cell
        cell_evals[ci] = stats[S_EVALS] - ev0
        if nrec == rec_start:
            stats[S_CELLS_OUT] += 1
        else:
            stats[S_CELLS_IN] += 1
    return nrec
