"""CEPL v2: certified exact-polytope labeller for one window (numba core).

Target (docs/METHOD.md section 2):
    F_W = {s in (0,1)^3 \\ B(W) : for every FREE receiver x in W,  [s, x] meets the solid  <=>  O_W(x) = dark}
Masses are Lebesgue volumes (uniform prior on the legal domain; the 2^-26 clearance shell is bounded separately).

Changes against v1 (code/rpl/engine.py, kept for the record):
  * S_C(r) from front faces + silhouette-edge rows only (topology.py); allocation-free predicates;
  * learned refuters: receivers / blocks that refuted a box are tried first on later boxes (cost only);
  * non-convex constraints are filtered against the convex leaf polytope before inclusion-exclusion;
  * undetermined leaves carry the volume of their convex relaxation as the upper bound.

Leaf record kinds: 0 IN (whole box feasible), 1 EXACT (polytope volume), 2 UNDET (lower 0, upper bound).
Every decision is sound under the filtered-predicate margins of predicates.py; volumes are float64 polytope
volumes (relative error ~1e-13, tests/test_polytope.py).
"""
import numpy as np
from numba import njit

from rpl.polytope import box_polytope, clip, volume, EMPTY, OVERFLOW, MV, MF
from rpl.predicates import MARG, AMARG, plane_box_range, aabb_disjoint, segment_hits

FINE_DEPTH = 7
IN_REC, EXACT_REC, UNDET_REC = 0, 1, 2
EXCL, CONTAINS, CUT, AMBIG = 0, 1, 2, 3
MAXROWS = 2048
EPSV = 1e-13


# ----------------------------------------------------------------------------------------------------------
# shadow polyhedron rows (front faces + silhouette edges)
@njit(cache=True)
def shadow_rows(r, P, K, c, adj, aoff, alpha, out):
    k = K[c]
    nF = 0
    for i in range(k):
        al = P[c, i, 0] * r[0] + P[c, i, 1] * r[1] + P[c, i, 2] * r[2] + P[c, i, 3]
        alpha[i] = al
        if al > AMARG:
            nF += 1
        elif al >= -AMARG:
            return 0, 2
    if nF == 0:
        return 0, 1
    n = 0
    for i in range(k):
        if alpha[i] > 0:
            a0 = P[c, i, 0]; a1 = P[c, i, 1]; a2 = P[c, i, 2]
            nr = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            out[n, 0] = a0 / nr; out[n, 1] = a1 / nr; out[n, 2] = a2 / nr; out[n, 3] = P[c, i, 3] / nr
            n += 1
    for t in range(aoff[c], aoff[c + 1]):
        i = adj[t, 1]
        j = adj[t, 2]
        if alpha[i] > 0 and alpha[j] < 0:
            f = i; b = j
        elif alpha[j] > 0 and alpha[i] < 0:
            f = j; b = i
        else:
            continue
        ci = -alpha[f]
        cj = -alpha[b]
        a0 = cj * P[c, f, 0] - ci * P[c, b, 0]
        a1 = cj * P[c, f, 1] - ci * P[c, b, 1]
        a2 = cj * P[c, f, 2] - ci * P[c, b, 2]
        nr = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        if nr < 1e-300:
            continue
        a0 /= nr; a1 /= nr; a2 /= nr
        out[n, 0] = a0; out[n, 1] = a1; out[n, 2] = a2
        out[n, 3] = -(a0 * r[0] + a1 * r[1] + a2 * r[2])
        n += 1
    return n, 0


@njit(cache=True)
def classify_rows(lo, hi, rows, n, cut_idx):
    ncut = 0
    for q in range(n):
        mn, mx = plane_box_range(rows[q, 0], rows[q, 1], rows[q, 2], rows[q, 3], lo, hi)
        if mn > MARG:
            return EXCL, 0
        if mx >= -MARG:
            cut_idx[ncut] = q
            ncut += 1
    if ncut == 0:
        return CONTAINS, 0
    return CUT, ncut


@njit(cache=True)
def comp_status(lo, hi, r, P, K, c, adj, aoff, alpha, rows, cut_idx):
    n, flag = shadow_rows(r, P, K, c, adj, aoff, alpha, rows)
    if flag == 1:
        return CONTAINS, 0
    if flag == 2:
        return AMBIG, 0
    return classify_rows(lo, hi, rows, n, cut_idx)


@njit(cache=True)
def receiver_status(lo, hi, r, lit, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo, uhi):
    """0 consistent for the whole box, 1 contradiction for the whole box, 2 active."""
    for k in range(3):
        ulo[k] = min(lo[k], r[k])
        uhi[k] = max(hi[k], r[k])
    nact = 0
    for c in range(P.shape[0]):
        if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
            continue
        st, ncut = comp_status(lo, hi, r, P, K, c, adj, aoff, alpha, rows, cut_idx)
        if st == CONTAINS:
            return 1 if lit else 0
        if st != EXCL:
            nact += 1
    if nact == 0:
        return 0 if lit else 1
    return 2


@njit(cache=True)
def hull_misses(lo, hi, rlo, rhi, P, K, c, clo, chi, ulo, uhi):
    for k in range(3):
        ulo[k] = min(lo[k], rlo[k])
        uhi[k] = max(hi[k], rhi[k])
    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
        return True
    for i in range(K[c]):
        mn1, _ = plane_box_range(P[c, i, 0], P[c, i, 1], P[c, i, 2], P[c, i, 3], lo, hi)
        if mn1 <= MARG:
            continue
        mn2, _ = plane_box_range(P[c, i, 0], P[c, i, 1], P[c, i, 2], P[c, i, 3], rlo, rhi)
        if mn2 > MARG:
            return True
    return False


@njit(cache=True)
def all_visible(lo, hi, rlo, rhi, P, K, clo, chi, ulo, uhi):
    for c in range(P.shape[0]):
        if not hull_misses(lo, hi, rlo, rhi, P, K, c, clo, chi, ulo, uhi):
            return False
    return True


@njit(cache=True)
def any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, adj, aoff, alpha, rows, w, ulo, uhi):
    """Some single component meets every segment between box B and receiver box R (Lemma 2)."""
    for k in range(3):
        ulo[k] = min(lo[k], rlo[k])
        uhi[k] = max(hi[k], rhi[k])
    for c in range(P.shape[0]):
        if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
            continue
        ok = True
        for corner in range(8):
            w[0] = rhi[0] if corner & 1 else rlo[0]
            w[1] = rhi[1] if corner & 2 else rlo[1]
            w[2] = rhi[2] if corner & 4 else rlo[2]
            n, flag = shadow_rows(w, P, K, c, adj, aoff, alpha, rows)
            if flag == 1:
                continue
            if flag == 2:
                ok = False
                break
            for q in range(n):
                _, mx = plane_box_range(rows[q, 0], rows[q, 1], rows[q, 2], rows[q, 3], lo, hi)
                if mx >= -MARG:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


# ----------------------------------------------------------------------------------------------------------
# blocks
@njit(cache=True)
def _enc(l, bx, by, bz):
    return (np.int64(l) << 48) | (np.int64(bx) << 32) | (np.int64(by) << 16) | np.int64(bz)


@njit(cache=True)
def _dec(code):
    return (code >> 48) & 0xFFFF, (code >> 32) & 0xFFFF, (code >> 16) & 0xFFFF, code & 0xFFFF


@njit(cache=True)
def _block_code(pyr, off, q, l, bx, by, bz):
    n = 1 << (q - l)
    return pyr[off[l] + (bx * n + by) * n + bz]


@njit(cache=True)
def _block_box(o, l, bx, by, bz, rlo, rhi):
    s = 1 << l
    Ninv = 1.0 / 128.0
    rlo[0] = (o[0] + bx * s + 0.5) * Ninv
    rlo[1] = (o[1] + by * s + 0.5) * Ninv
    rlo[2] = (o[2] + bz * s + 0.5) * Ninv
    rhi[0] = (o[0] + (bx + 1) * s - 0.5) * Ninv
    rhi[1] = (o[1] + (by + 1) * s - 0.5) * Ninv
    rhi[2] = (o[2] + (bz + 1) * s - 0.5) * Ninv


@njit(cache=True)
def _refutes(code, lo, hi, o, pyr, off, q, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo, uhi, r, rlo,
             rhi, w):
    """True if receiver/block `code` certifies that no source in [lo,hi] reproduces its observation."""
    l, bx, by, bz = _dec(code)
    bc = _block_code(pyr, off, q, l, bx, by, bz)
    if bc >= 2:
        return False
    if l == 0:
        Ninv = 1.0 / 128.0
        r[0] = (o[0] + bx + 0.5) * Ninv
        r[1] = (o[1] + by + 0.5) * Ninv
        r[2] = (o[2] + bz + 0.5) * Ninv
        return receiver_status(lo, hi, r, bc == 1, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo,
                               uhi) == 1
    _block_box(o, l, bx, by, bz, rlo, rhi)
    if bc == 1:
        return any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, adj, aoff, alpha, rows, w, ulo, uhi)
    return all_visible(lo, hi, rlo, rhi, P, K, clo, chi, ulo, uhi)


# ----------------------------------------------------------------------------------------------------------
# leaf volume with non-convex filtering + inclusion-exclusion
@njit(cache=True)
def _poly_vs_piece(V, Fc, Fn, nf, planes, a, n, used):
    """0 = polytope certainly disjoint from piece {planes[a:a+n] <= 0} (separating piece plane),
    1 = polytope certainly inside the piece, 2 = undecided."""
    inside_all = True
    for qq in range(a, a + n):
        mn = 1e300
        mx = -1e300
        for f in range(nf):
            for j in range(Fn[f]):
                v = Fc[f, j]
                g = planes[qq, 0] * V[v, 0] + planes[qq, 1] * V[v, 1] + planes[qq, 2] * V[v, 2] + planes[qq, 3]
                if g < mn:
                    mn = g
                if g > mx:
                    mx = g
        if mn > EPSV:
            return 0
        if mx >= -EPSV:
            inside_all = False
    return 1 if inside_all else 2


@njit(cache=True)
def _clip_copy_empty(V, Fc, Fn, nv, nf, V2, Fc2, Fn2, planes, a, n):
    for v in range(nv):
        V2[v, 0] = V[v, 0]; V2[v, 1] = V[v, 1]; V2[v, 2] = V[v, 2]
    for f in range(nf):
        Fn2[f] = Fn[f]
        for j in range(Fn[f]):
            Fc2[f, j] = Fc[f, j]
    nv2 = nv
    nf2 = nf
    for qq in range(a, a + n):
        st, nv2, nf2 = clip(V2, Fc2, Fn2, nv2, nf2, planes[qq, 0], planes[qq, 1], planes[qq, 2], planes[qq, 3])
        if st == EMPTY:
            return True, nv2, nf2
        if st == OVERFLOW:
            return False, -1, -1
    return False, nv2, nf2


@njit(cache=True)
def leaf_volume(lo, hi, conv, ncv, pc_planes, pc_start, pc_len, ct_type, ct_first, ct_npc, nct, max_terms,
                V2, Fc2, Fn2, keep_pc, keep_ct_first, keep_ct_n, keep_ct_type):
    """Exact volume of B n {conv <= 0} n (non-convex constraints).
    Returns (volume, upper, status): status 0 exact, 1 too many inclusion-exclusion terms (upper = convex
    relaxation volume), 2 overflow (upper = convex relaxation volume or box volume)."""
    boxvol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    V, Fc, Fn, nv, nf = box_polytope(lo, hi)
    for k in range(ncv):
        st, nv, nf = clip(V, Fc, Fn, nv, nf, conv[k, 0], conv[k, 1], conv[k, 2], conv[k, 3])
        if st == EMPTY:
            return 0.0, 0.0, 0
        if st == OVERFLOW:
            return -1.0, boxvol, 2
    base = volume(V, Fc, Fn, nv, nf)
    if nct == 0:
        return base, base, 0
    # filter non-convex constraints against the convex polytope
    nk = 0          # kept constraints
    nkp = 0         # kept pieces
    used = np.zeros(1, np.bool_)
    for j in range(nct):
        first = nkp
        contains = False
        for p in range(ct_first[j], ct_first[j] + ct_npc[j]):
            rel = _poly_vs_piece(V, Fc, Fn, nf, pc_planes, pc_start[p], pc_len[p], used)
            if rel == 2:
                emp, _, _ = _clip_copy_empty(V, Fc, Fn, nv, nf, V2, Fc2, Fn2, pc_planes, pc_start[p], pc_len[p])
                if emp:
                    rel = 0
            if rel == 0:
                continue
            if rel == 1:
                contains = True
                break
            keep_pc[nkp] = p
            nkp += 1
        if ct_type[j] == 0:          # SUB: avoid the union of the pieces
            if contains:
                return 0.0, 0.0, 0
            if nkp == first:
                continue
        else:                        # UNI: lie in the union of the pieces
            if contains:
                nkp = first
                continue
            if nkp == first:
                return 0.0, 0.0, 0
        keep_ct_first[nk] = first
        keep_ct_n[nk] = nkp - first
        keep_ct_type[nk] = ct_type[j]
        nk += 1
    if nk == 0:
        return base, base, 0
    T = 1
    for j in range(nk):
        if keep_ct_n[j] > 20:
            return -1.0, base, 1
        T *= (1 << keep_ct_n[j])
        if T > max_terms:
            return -1.0, base, 1
    total = 0.0
    for t in range(T):
        rem = t
        sign = 1.0
        valid = True
        # copy base polytope
        for v in range(nv):
            V2[v, 0] = V[v, 0]; V2[v, 1] = V[v, 1]; V2[v, 2] = V[v, 2]
        for f in range(nf):
            Fn2[f] = Fn[f]
            for jj in range(Fn[f]):
                Fc2[f, jj] = Fc[f, jj]
        nv2 = nv
        nf2 = nf
        empty = False
        for j in range(nk):
            radix = 1 << keep_ct_n[j]
            m = rem % radix
            rem //= radix
            cnt = 0
            for p in range(keep_ct_n[j]):
                if m & (1 << p):
                    cnt += 1
            if keep_ct_type[j] == 1:
                if cnt == 0:
                    valid = False
                    break
                if cnt % 2 == 0:
                    sign = -sign
            elif cnt % 2 == 1:
                sign = -sign
            if empty:
                continue
            for p in range(keep_ct_n[j]):
                if not (m & (1 << p)):
                    continue
                pi = keep_pc[keep_ct_first[j] + p]
                for qq in range(pc_start[pi], pc_start[pi] + pc_len[pi]):
                    st, nv2, nf2 = clip(V2, Fc2, Fn2, nv2, nf2, pc_planes[qq, 0], pc_planes[qq, 1],
                                        pc_planes[qq, 2], pc_planes[qq, 3])
                    if st == EMPTY:
                        empty = True
                        break
                    if st == OVERFLOW:
                        return -1.0, base, 2
                if empty:
                    break
        if not valid or empty:
            continue
        total += sign * volume(V2, Fc2, Fn2, nv2, nf2)
    if total < 0.0:
        total = 0.0
    if total > base:
        total = base
    return total, base, 0


# ----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _cell_index(lo):
    s = 1 << FINE_DEPTH
    ix = int(lo[0] * s + 1e-9)
    iy = int(lo[1] * s + 1e-9)
    iz = int(lo[2] * s + 1e-9)
    return (ix * s + iy) * s + iz


@njit(cache=True)
def _emit(lo, depth, vlo, vhi, kind, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec):
    if depth >= FINE_DEPTH:
        rec_cell[nrec] = _cell_index(lo); rec_lo[nrec] = vlo; rec_hi[nrec] = vhi
        rec_kind[nrec] = kind; rec_depth[nrec] = depth
        return nrec + 1
    n = 1 << (FINE_DEPTH - depth)
    cs = 1.0 / (1 << FINE_DEPTH)
    frac = 1.0 / (n * n * n)
    c0 = np.empty(3)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                c0[0] = lo[0] + ix * cs; c0[1] = lo[1] + iy * cs; c0[2] = lo[2] + iz * cs
                rec_cell[nrec] = _cell_index(c0); rec_lo[nrec] = vlo * frac
                rec_hi[nrec] = vhi * frac; rec_kind[nrec] = kind; rec_depth[nrec] = depth
                nrec += 1
    return nrec


@njit(cache=True)
def label_window(P, K, clo, chi, adj, aoff, o, q, pyr, off, sentinels, d_min, d_max, d_full, kappa, max_terms,
                 max_nodes, max_pending, n_learn, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, stats, dstat,
                 probe_order, probe_k):
    Ninv = 1.0 / 128.0
    L = 1 << q
    wlo = np.empty(3)
    whi = np.empty(3)
    for k in range(3):
        wlo[k] = o[k] * Ninv
        whi[k] = (o[k] + L) * Ninv
    seg = np.empty(max_pending, np.int64)
    work = np.empty(1 << 22, np.int64)
    kmax = P.shape[1]
    alpha = np.empty(kmax)
    rows = np.empty((MAXROWS, 4))
    cut_idx = np.empty(MAXROWS, np.int64)
    ulo = np.empty(3)
    uhi = np.empty(3)
    r = np.empty(3)
    rlo = np.empty(3)
    rhi = np.empty(3)
    wv = np.empty(3)
    lo = np.empty(3)
    hi = np.empty(3)
    # leaf buffers
    MAXCV = 1 << 17
    conv = np.empty((MAXCV, 4))
    MAXPP = 1 << 17
    pc_planes = np.empty((MAXPP, 4))
    MAXPC = 1 << 15
    pc_start = np.empty(MAXPC, np.int64)
    pc_len = np.empty(MAXPC, np.int64)
    ct_type = np.empty(MAXPC, np.int64)
    ct_first = np.empty(MAXPC, np.int64)
    ct_npc = np.empty(MAXPC, np.int64)
    keep_pc = np.empty(MAXPC, np.int64)
    keep_ct_first = np.empty(MAXPC, np.int64)
    keep_ct_n = np.empty(MAXPC, np.int64)
    keep_ct_type = np.empty(MAXPC, np.int64)
    V2 = np.empty((MV, 3))
    Fc2 = np.empty((MF, 128), np.int32)
    Fn2 = np.empty(MF, np.int32)
    learn = np.empty(max(n_learn, 1), np.int64)
    nl = 0
    lp = 0
    # DFS stack
    MAXS = 64 * 8
    st_lo = np.empty((MAXS, 3))
    st_hi = np.empty((MAXS, 3))
    st_depth = np.empty(MAXS, np.int64)
    st_a = np.empty(MAXS, np.int64)
    st_b = np.empty(MAXS, np.int64)
    st_top = np.empty(MAXS, np.int64)
    ns = sentinels.shape[0]
    for t in range(ns):
        seg[t] = sentinels[t]
    seg[ns] = _enc(q, 0, 0, 0)
    st_lo[0, 0] = 0.0; st_lo[0, 1] = 0.0; st_lo[0, 2] = 0.0
    st_hi[0, 0] = 1.0; st_hi[0, 1] = 1.0; st_hi[0, 2] = 1.0
    st_depth[0] = 0
    st_a[0] = 0
    st_b[0] = ns + 1
    st_top[0] = ns + 1
    sp = 1
    nrec = 0
    nodes = 0
    invalid = False
    while sp > 0:
        sp -= 1
        for k in range(3):
            lo[k] = st_lo[sp, k]
            hi[k] = st_hi[sp, k]
        depth = st_depth[sp]
        a = st_a[sp]
        b = st_b[sp]
        top = st_top[sp]
        side = hi[0] - lo[0]
        vol = side * side * side
        if nrec > rec_cell.shape[0] - (1 << 22):
            invalid = True
            break
        nodes += 1
        dstat[depth, 0] += 1
        inside_w = True
        disjoint_w = False
        for k in range(3):
            if lo[k] < wlo[k] or hi[k] > whi[k]:
                inside_w = False
            if hi[k] <= wlo[k] or lo[k] >= whi[k]:
                disjoint_w = True
        if inside_w:
            stats[1] += 1
            continue
        if nodes > max_nodes:
            nrec = _emit(lo, depth, 0.0, vol, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
            stats[2] += 1
            continue
        # learned refuters first
        refuted = False
        for t in range(min(nl, n_learn)):
            stats[13] += 1
            if _refutes(learn[t], lo, hi, o, pyr, off, q, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx,
                        ulo, uhi, r, rlo, rhi, wv):
                refuted = True
                break
        if refuted:
            stats[5] += 1
            stats[14] += 1
            dstat[depth, 3] += 1
            continue
        if depth >= d_full and probe_k > 0:
            # probe refutation: render the box centre at receivers (boundary first) with early exit; each
            # mismatching receiver is tried as a whole-box refuter (certified by receiver_status).
            for k in range(3):
                wv[k] = 0.5 * (lo[k] + hi[k])
            nm = 0
            for t in range(probe_order.shape[0]):
                code = probe_order[t]
                l, bx, by, bz = _dec(code)
                bc = _block_code(pyr, off, q, 0, bx, by, bz)
                r[0] = (o[0] + bx + 0.5) * Ninv
                r[1] = (o[1] + by + 0.5) * Ninv
                r[2] = (o[2] + bz + 0.5) * Ninv
                y = 1
                und = False
                for c in range(P.shape[0]):
                    hh = segment_hits(wv, r, P, K, c)
                    if hh == 1:
                        y = 0
                        und = False
                        break
                    if hh == -1:
                        und = True
                stats[13] += 1
                if und or y == (1 if bc == 1 else 0):
                    continue
                nm += 1
                if receiver_status(lo, hi, r, bc == 1, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo,
                                   uhi) == 1:
                    refuted = True
                    if n_learn > 0:
                        learn[lp] = code
                        lp = (lp + 1) % n_learn
                        if nl < n_learn:
                            nl += 1
                    break
                if nm >= probe_k:
                    break
            if refuted:
                stats[5] += 1
                stats[14] += 1
                dstat[depth, 3] += 1
                continue
        out0 = top
        w = top
        nwork = 0
        out_flag = False
        overflow = False
        force = depth >= d_full
        n_single = 0
        idx = a
        refuter = np.int64(-1)
        while True:
            if idx < b:
                code = seg[idx]
                idx += 1
            elif nwork > 0:
                nwork -= 1
                code = work[nwork]
            else:
                break
            l, bx, by, bz = _dec(code)
            bc = _block_code(pyr, off, q, l, bx, by, bz)
            if bc == 2:
                continue
            if l == 0:
                r[0] = (o[0] + bx + 0.5) * Ninv
                r[1] = (o[1] + by + 0.5) * Ninv
                r[2] = (o[2] + bz + 0.5) * Ninv
                stats[3] += 1
                dstat[depth, 1] += 1
                rs = receiver_status(lo, hi, r, bc == 1, P, K, clo, chi, adj, aoff, alpha, rows, cut_idx, ulo,
                                     uhi)
                if rs == 0:
                    continue
                if rs == 1:
                    out_flag = True
                    refuter = code
                    break
                if w >= max_pending:
                    overflow = True
                    break
                seg[w] = code
                w += 1
                n_single += 1
                continue
            if depth < d_full:
                # sentinel-only stage: blocks are carried untested (cost only; never changes the result)
                if w >= max_pending:
                    overflow = True
                    break
                seg[w] = code
                w += 1
                continue
            _block_box(o, l, bx, by, bz, rlo, rhi)
            stats[4] += 1
            dstat[depth, 2] += 1
            if bc == 1:
                if all_visible(lo, hi, rlo, rhi, P, K, clo, chi, ulo, uhi):
                    continue
                if any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, adj, aoff, alpha, rows, wv, ulo, uhi):
                    out_flag = True
                    refuter = code
                    break
            elif bc == 0:
                if any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, adj, aoff, alpha, rows, wv, ulo, uhi):
                    continue
                if all_visible(lo, hi, rlo, rhi, P, K, clo, chi, ulo, uhi):
                    out_flag = True
                    refuter = code
                    break
            bside = (1 << l) * Ninv
            if force or bside > kappa * side:
                if nwork + 8 > work.shape[0]:
                    overflow = True
                    break
                for cx in range(2):
                    for cy in range(2):
                        for cz in range(2):
                            work[nwork] = _enc(l - 1, 2 * bx + cx, 2 * by + cy, 2 * bz + cz)
                            nwork += 1
            else:
                if w >= max_pending:
                    overflow = True
                    break
                seg[w] = code
                w += 1
        if out_flag:
            stats[5] += 1
            dstat[depth, 3] += 1
            if n_learn > 0:
                learn[lp] = refuter
                lp = (lp + 1) % n_learn
                if nl < n_learn:
                    nl += 1
            continue
        if overflow:
            stats[9] += 1
            nrec = _emit(lo, depth, 0.0, vol, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
            continue
        nkept = w - out0
        dstat[depth, 4] += nkept
        win_partial = not disjoint_w
        if nkept == 0 and not win_partial:
            nrec = _emit(lo, depth, vol, vol, IN_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
            stats[6] += 1
            continue
        all_single = n_single == nkept
        leaf_upper = vol
        if depth >= d_min and depth >= d_full and all_single:
            ncv = 0
            npc = 0
            npp = 0
            nct = 0
            bad = False
            for t in range(out0, w):
                l, bx, by, bz = _dec(seg[t])
                bc = _block_code(pyr, off, q, l, bx, by, bz)
                r[0] = (o[0] + bx + 0.5) * Ninv
                r[1] = (o[1] + by + 0.5) * Ninv
                r[2] = (o[2] + bz + 0.5) * Ninv
                lit = bc == 1
                for k in range(3):
                    ulo[k] = min(lo[k], r[k])
                    uhi[k] = max(hi[k], r[k])
                first_piece = npc
                for c in range(P.shape[0]):
                    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
                        continue
                    stc, ncut = comp_status(lo, hi, r, P, K, c, adj, aoff, alpha, rows, cut_idx)
                    if stc == AMBIG:
                        bad = True
                        stats[10] += 1
                        break
                    if stc != CUT:
                        continue
                    if npc >= MAXPC or npp + ncut > MAXPP:
                        bad = True
                        stats[9] += 1
                        break
                    pc_start[npc] = npp
                    pc_len[npc] = ncut
                    for u in range(ncut):
                        for k in range(4):
                            pc_planes[npp, k] = rows[cut_idx[u], k]
                        npp += 1
                    npc += 1
                if bad:
                    break
                npieces = npc - first_piece
                if npieces == 0:
                    continue
                if lit and npieces == 1 and pc_len[first_piece] == 1:
                    if ncv >= MAXCV:
                        bad = True
                        stats[9] += 1
                        break
                    q0 = pc_start[first_piece]
                    for k in range(4):
                        conv[ncv, k] = -pc_planes[q0, k]
                    ncv += 1
                    npc = first_piece
                    npp = pc_start[first_piece]
                elif (not lit) and npieces == 1:
                    q0 = pc_start[first_piece]
                    if ncv + pc_len[first_piece] > MAXCV:
                        bad = True
                        stats[9] += 1
                        break
                    for u in range(pc_len[first_piece]):
                        for k in range(4):
                            conv[ncv, k] = pc_planes[q0 + u, k]
                        ncv += 1
                    npc = first_piece
                    npp = pc_start[first_piece]
                else:
                    if nct >= MAXPC:
                        bad = True
                        stats[9] += 1
                        break
                    ct_type[nct] = 0 if lit else 1
                    ct_first[nct] = first_piece
                    ct_npc[nct] = npieces
                    nct += 1
            if win_partial and not bad:
                if npc < MAXPC and npp + 6 <= MAXPP and nct < MAXPC:
                    pc_start[npc] = npp
                    pc_len[npc] = 6
                    for k in range(3):
                        for sgn in range(2):
                            for u in range(4):
                                pc_planes[npp, u] = 0.0
                            if sgn == 0:
                                pc_planes[npp, k] = -1.0
                                pc_planes[npp, 3] = wlo[k]
                            else:
                                pc_planes[npp, k] = 1.0
                                pc_planes[npp, 3] = -whi[k]
                            npp += 1
                    ct_type[nct] = 0
                    ct_first[nct] = npc
                    ct_npc[nct] = 1
                    npc += 1
                    nct += 1
                else:
                    bad = True
            if not bad:
                v, ub, stl = leaf_volume(lo, hi, conv, ncv, pc_planes, pc_start, pc_len, ct_type, ct_first,
                                         ct_npc, nct, max_terms, V2, Fc2, Fn2, keep_pc, keep_ct_first, keep_ct_n,
                                         keep_ct_type)
                stats[11] += ncv
                if nct > stats[12]:
                    stats[12] = nct
                if stl == 0:
                    if v > 0.0:
                        nrec = _emit(lo, depth, v, v, EXACT_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth,
                                     nrec)
                    stats[7] += 1
                    continue
                leaf_upper = ub
        if depth >= d_max:
            nrec = _emit(lo, depth, 0.0, leaf_upper, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth,
                         nrec)
            stats[8] += 1
            continue
        if sp + 8 > MAXS:
            invalid = True
            break
        h = side * 0.5
        for cx in range(2):
            for cy in range(2):
                for cz in range(2):
                    st_lo[sp, 0] = lo[0] + cx * h
                    st_lo[sp, 1] = lo[1] + cy * h
                    st_lo[sp, 2] = lo[2] + cz * h
                    st_hi[sp, 0] = st_lo[sp, 0] + h
                    st_hi[sp, 1] = st_lo[sp, 1] + h
                    st_hi[sp, 2] = st_lo[sp, 2] + h
                    st_depth[sp] = depth + 1
                    st_a[sp] = out0
                    st_b[sp] = w
                    st_top[sp] = w
                    sp += 1
    stats[0] = nodes
    stats[15] = 1 if invalid else 0
    return nrec


STAT_NAMES = ["nodes", "inside_window", "budget_unresolved", "receiver_tests", "block_tests", "out_boxes",
              "in_boxes", "exact_leaves", "undet_leaves", "overflow", "ambiguous", "conv_planes_total",
              "max_nonconvex_constraints", "learned_and_probe_tests", "learned_or_probe_refutations", "invalid"]


def build_pyramid(cls):
    L = cls.shape[0]
    q = int(round(np.log2(L)))
    assert 1 << q == L
    has_lit = (cls == 1)
    has_dark = (cls == 0)
    codes = []
    for l in range(q + 1):
        code = np.full(has_lit.shape, 2, np.uint8)
        code[has_dark] = 0
        code[has_lit] = 1
        code[has_lit & has_dark] = 3
        codes.append(code.reshape(-1))
        if l < q:
            n = has_lit.shape[0] // 2
            has_lit = has_lit.reshape(n, 2, n, 2, n, 2).any(axis=(1, 3, 5))
            has_dark = has_dark.reshape(n, 2, n, 2, n, 2).any(axis=(1, 3, 5))
    off = np.zeros(q + 2, np.int64)
    for l in range(q + 1):
        off[l + 1] = off[l] + codes[l].size
    return np.concatenate(codes), off, q


def boundary_mask(cls):
    lit = cls == 1
    dark = cls == 0
    bd = np.zeros_like(lit)
    for ax in range(3):
        n = lit.shape[ax]
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, n - 1)
        b[ax] = slice(1, n)
        a = tuple(a); b = tuple(b)
        x = (lit[a] & dark[b]) | (dark[a] & lit[b])
        bd[a] |= x
        bd[b] |= x
    return bd


def choose_sentinels(cls, n_boundary=512, n_class=64, seed=0):
    rng = np.random.default_rng(seed)
    lit = cls == 1
    dark = cls == 0
    bd = boundary_mask(cls)
    picks = []
    for mask, n in ((bd, n_boundary), (lit & ~bd, n_class), (dark & ~bd, n_class)):
        idx = np.flatnonzero(mask.reshape(-1))
        if len(idx) and n > 0:
            picks.append(rng.choice(idx, min(n, len(idx)), replace=False))
    idx = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    L = cls.shape[0]
    x, y, z = np.unravel_index(idx, (L, L, L))
    return (x.astype(np.int64) << 32) | (y.astype(np.int64) << 16) | z.astype(np.int64)


def probe_order_codes(cls, n_random=4096, seed=1):
    """Boundary FREE receivers (shuffled) followed by random non-boundary FREE receivers, as level-0 codes."""
    rng = np.random.default_rng(seed)
    bd = boundary_mask(cls)
    free = cls != 2
    a = np.flatnonzero(bd.reshape(-1))
    rng.shuffle(a)
    b = np.flatnonzero((free & ~bd).reshape(-1))
    if len(b) > n_random:
        b = rng.choice(b, n_random, replace=False)
    idx = np.concatenate([a, b])
    L = cls.shape[0]
    x, y, z = np.unravel_index(idx, (L, L, L))
    return (x.astype(np.int64) << 32) | (y.astype(np.int64) << 16) | z.astype(np.int64)


def run_window(geom, cls, origin, d_min=FINE_DEPTH, d_max=11, d_full=FINE_DEPTH, kappa=1.0, max_terms=256,
               probe_k=16, max_nodes=5_000_000,
               max_pending=1 << 25, max_records=1 << 24, n_boundary=512, n_class=64, sentinel_seed=0,
               n_learn=32):
    P, K, clo, chi, adj, aoff = geom
    pyr, off, q = build_pyramid(cls)
    sentinels = choose_sentinels(cls, n_boundary, n_class, sentinel_seed)
    probe = probe_order_codes(cls) if probe_k > 0 else np.zeros(0, np.int64)
    o = np.asarray(origin, np.int64)
    rec_cell = np.empty(max_records, np.int64)
    rec_lo = np.empty(max_records)
    rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8)
    rec_depth = np.empty(max_records, np.int8)
    stats = np.zeros(16, np.int64)
    dstat = np.zeros((32, 5), np.int64)
    n = label_window(P, K, clo, chi, adj, aoff, o, q, pyr, off, sentinels, d_min, d_max, d_full, kappa, max_terms,
                     max_nodes, max_pending, n_learn, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, stats, dstat,
                     probe, probe_k)
    return dict(cell=rec_cell[:n].copy(), lo=rec_lo[:n].copy(), hi=rec_hi[:n].copy(), kind=rec_kind[:n].copy(),
                depth=rec_depth[:n].copy(), stats=dict(zip(STAT_NAMES, stats.tolist())), depth_stats=dstat)


def scene_geometry(scene):
    from .topology import adjacency
    P, K, clo, chi = scene.geometry_arrays()
    adj, aoff = adjacency(P, K)
    return P, K, clo, chi, adj, aoff
