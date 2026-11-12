"""CEPL v3: certified exact-polytope labeller with incremental clipping (numba core).

Same target, certificates and record semantics as engine2 (see docs/METHOD.md).  Two stages per box:

  * depth < d_full  (cover stage): only single receivers (sentinels, inherited actives) and learned refuters are
    tested; receiver blocks are carried untested.  Produces a certified outer cover of F_W.
  * depth >= d_full (leaf stage): the box polytope P = B is clipped incrementally by every convex constraint as
    soon as its receiver is classified; P = empty certifies the box OUT (the intersection of certified constraints
    is empty).  Receiver blocks are tested and split down to single receivers.  When all receivers are processed,
    the remaining non-convex constraints are filtered against P and resolved by inclusion-exclusion.

Engine2 is kept unchanged for the record.
"""
import numpy as np
from numba import njit

from rpl.polytope import clip, volume, EMPTY, OVERFLOW, MV, MF, MPV
from rpl.predicates import MARG, plane_box_range, aabb_disjoint, segment_hits
from rpl128.engine2 import (comp_status, receiver_status, all_visible, any_blocked, _enc, _dec, _block_code, _block_box,
                      _refutes, _poly_vs_piece, _clip_copy_empty, _emit, build_pyramid, choose_sentinels,
                      boundary_mask, scene_geometry, EXCL, CONTAINS, CUT, AMBIG, IN_REC, EXACT_REC, UNDET_REC,
                      FINE_DEPTH, MAXROWS)


@njit(cache=True)
def init_box(V, Fc, Fn, lo, hi):
    k = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                V[k, 0] = hi[0] if ix else lo[0]
                V[k, 1] = hi[1] if iy else lo[1]
                V[k, 2] = hi[2] if iz else lo[2]
                k += 1
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for f in range(6):
        for j in range(4):
            Fc[f, j] = faces[f][j]
        Fn[f] = 4
    return 8, 6


@njit(cache=True)
def nonconvex_volume(V, Fc, Fn, nv, nf, pc_planes, pc_start, pc_len, ct_type, ct_first, ct_npc, nct, max_terms,
                     V2, Fc2, Fn2, keep_pc, keep_ct_first, keep_ct_n, keep_ct_type):
    """Volume of P n (non-convex constraints) for the convex polytope P = (V, Fc, Fn).
    Returns (volume, status): 0 exact, 1 too many terms, 2 overflow."""
    base = volume(V, Fc, Fn, nv, nf)
    if nct == 0:
        return base, 0
    used = np.zeros(1, np.bool_)
    nk = 0
    nkp = 0
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
        if ct_type[j] == 0:
            if contains:
                return 0.0, 0
            if nkp == first:
                continue
        else:
            if contains:
                nkp = first
                continue
            if nkp == first:
                return 0.0, 0
        keep_ct_first[nk] = first
        keep_ct_n[nk] = nkp - first
        keep_ct_type[nk] = ct_type[j]
        nk += 1
    if nk == 0:
        return base, 0
    T = 1
    for j in range(nk):
        if keep_ct_n[j] > 20:
            return -1.0, 1
        T *= (1 << keep_ct_n[j])
        if T > max_terms:
            return -1.0, 1
    total = 0.0
    for t in range(T):
        rem = t
        sign = 1.0
        valid = True
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
                        return -1.0, 2
                if empty:
                    break
        if not valid or empty:
            continue
        total += sign * volume(V2, Fc2, Fn2, nv2, nf2)
    if total < 0.0:
        total = 0.0
    if total > base:
        total = base
    return total, 0


@njit(cache=True)
def label_window(P, K, clo, chi, adj, aoff, o, q, pyr, off, sentinels, d_min, d_max, d_full, max_terms,
                 max_nodes, max_pending, n_learn, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, stats, dstat,
                 exact_leaves):
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
    V = np.empty((MV, 3))
    Fc = np.empty((MF, MPV), np.int32)
    Fn = np.empty(MF, np.int32)
    V2 = np.empty((MV, 3))
    Fc2 = np.empty((MF, MPV), np.int32)
    Fn2 = np.empty(MF, np.int32)
    learn = np.empty(max(n_learn, 1), np.int64)
    nl = 0
    lp = 0
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
        leafstage = depth >= d_full and depth >= d_min
        nv = 0
        nf = 0
        if leafstage:
            nv, nf = init_box(V, Fc, Fn, lo, hi)
        npc = 0
        npp = 0
        nct = 0
        ambiguous = False
        cbuf_over = False
        out0 = top
        w = top
        nwork = 0
        out_flag = False
        clip_empty = False
        overflow = False
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
                lit = bc == 1
                stats[3] += 1
                dstat[depth, 1] += 1
                for k in range(3):
                    ulo[k] = min(lo[k], r[k])
                    uhi[k] = max(hi[k], r[k])
                # classify all components once, collecting cut pieces
                first_piece = npc
                first_pp = npp
                nact = 0
                status = -1      # -1 undecided yet, 0 consistent, 1 contradiction
                amb_r = False
                for c in range(P.shape[0]):
                    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
                        continue
                    stc, ncut = comp_status(lo, hi, r, P, K, c, adj, aoff, alpha, rows, cut_idx)
                    if stc == CONTAINS:
                        status = 1 if lit else 0
                        break
                    if stc == EXCL:
                        continue
                    nact += 1
                    if stc == AMBIG:
                        amb_r = True
                        continue
                    if leafstage:
                        if npc >= MAXPC or npp + ncut > MAXPP:
                            cbuf_over = True
                            continue
                        pc_start[npc] = npp
                        pc_len[npc] = ncut
                        for u in range(ncut):
                            for kk in range(4):
                                pc_planes[npp, kk] = rows[cut_idx[u], kk]
                            npp += 1
                        npc += 1
                if status == -1 and nact == 0:
                    status = 0 if lit else 1
                if status == 0:
                    npc = first_piece
                    npp = first_pp
                    continue
                if status == 1:
                    out_flag = True
                    refuter = code
                    break
                # active receiver
                if w >= max_pending:
                    overflow = True
                    break
                seg[w] = code
                w += 1
                n_single += 1
                if not leafstage or not exact_leaves:
                    continue
                if amb_r:
                    ambiguous = True
                    npc = first_piece
                    npp = first_pp
                    continue
                npieces = npc - first_piece
                if npieces == 0:
                    continue
                lit_halfspaces = lit
                if lit:
                    for pi in range(first_piece, npc):
                        if pc_len[pi] != 1:
                            lit_halfspaces = False
                if lit_halfspaces or ((not lit) and npieces == 1):
                    # lit: avoid each single-row piece {g <= 0}  ->  keep -g <= 0 (all pieces, convex)
                    # dark: lie in the single convex piece  ->  keep every row g <= 0
                    q0 = pc_start[first_piece]
                    q1 = pc_start[npc - 1] + pc_len[npc - 1]
                    sg = -1.0 if lit else 1.0
                    for u in range(q0, q1):
                        st, nv, nf = clip(V, Fc, Fn, nv, nf, sg * pc_planes[u, 0], sg * pc_planes[u, 1],
                                          sg * pc_planes[u, 2], sg * pc_planes[u, 3])
                        stats[11] += 1
                        if st == EMPTY:
                            clip_empty = True
                            break
                        if st == OVERFLOW:
                            cbuf_over = True
                            break
                    npc = first_piece
                    npp = first_pp
                    if clip_empty:
                        refuter = code
                        break
                else:
                    if nct >= MAXPC:
                        cbuf_over = True
                    else:
                        ct_type[nct] = 0 if lit else 1
                        ct_first[nct] = first_piece
                        ct_npc[nct] = npieces
                        nct += 1
                continue
            # receiver block
            if not leafstage:
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
            if nwork + 8 > work.shape[0]:
                overflow = True
                break
            for cx in range(2):
                for cy in range(2):
                    for cz in range(2):
                        work[nwork] = _enc(l - 1, 2 * bx + cx, 2 * by + cy, 2 * bz + cz)
                        nwork += 1
        if out_flag or clip_empty:
            stats[5] += 1
            if clip_empty:
                stats[6] += 1
            dstat[depth, 3] += 1
            if n_learn > 0 and refuter >= 0 and out_flag:
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
        leaf_upper = vol
        if leafstage:
            win_partial = not disjoint_w
            if win_partial:
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
                    cbuf_over = True
            leaf_upper = volume(V, Fc, Fn, nv, nf)
            if not exact_leaves:
                if nkept == 0 and not win_partial:
                    nrec = _emit(lo, depth, vol, vol, IN_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
                    stats[7] += 1
                    continue
                leaf_upper = vol
                ambiguous = True    # forces split / UNDET below
            if not (ambiguous or cbuf_over):
                v, stl = nonconvex_volume(V, Fc, Fn, nv, nf, pc_planes, pc_start, pc_len, ct_type, ct_first, ct_npc,
                                          nct, max_terms, V2, Fc2, Fn2, keep_pc, keep_ct_first, keep_ct_n,
                                          keep_ct_type)
                if nct > stats[12]:
                    stats[12] = nct
                if stl == 0:
                    if v > 0.0:
                        kind = IN_REC if (nkept == 0 and not win_partial) else EXACT_REC
                        nrec = _emit(lo, depth, v, v, kind, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
                    stats[7] += 1
                    continue
            if ambiguous:
                stats[10] += 1
        if depth >= d_max:
            nrec = _emit(lo, depth, 0.0, leaf_upper, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, nrec)
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
              "out_by_empty_polytope", "exact_leaves", "undet_leaves", "overflow", "ambiguous", "clips",
              "max_nonconvex_constraints", "learned_tests", "learned_refutations", "invalid"]


def run_window(geom, cls, origin, d_min=FINE_DEPTH, d_max=12, d_full=10, max_terms=256, max_nodes=20_000_000,
               max_pending=1 << 25, max_records=1 << 24, n_boundary=512, n_class=64, sentinel_seed=0, n_learn=32,
               exact_leaves=True):
    P, K, clo, chi, adj, aoff = geom
    pyr, off, q = build_pyramid(cls)
    sentinels = choose_sentinels(cls, n_boundary, n_class, sentinel_seed)
    o = np.asarray(origin, np.int64)
    rec_cell = np.empty(max_records, np.int64)
    rec_lo = np.empty(max_records)
    rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8)
    rec_depth = np.empty(max_records, np.int8)
    stats = np.zeros(16, np.int64)
    dstat = np.zeros((32, 5), np.int64)
    n = label_window(P, K, clo, chi, adj, aoff, o, q, pyr, off, sentinels, d_min, d_max, d_full, max_terms,
                     max_nodes, max_pending, n_learn, rec_cell, rec_lo, rec_hi, rec_kind, rec_depth, stats, dstat,
                     exact_leaves)
    return dict(cell=rec_cell[:n].copy(), lo=rec_lo[:n].copy(), hi=rec_hi[:n].copy(), kind=rec_kind[:n].copy(),
                depth=rec_depth[:n].copy(), stats=dict(zip(STAT_NAMES, stats.tolist())), depth_stats=dstat)
