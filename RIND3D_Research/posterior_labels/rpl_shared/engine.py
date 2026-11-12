"""Certified exact-polytope labeller (CEPL) for one window: numba core.

Feasible set (docs/METHOD.md section 2):
    F_W = {s in (0,1)^3 \\ B(W) : for every FREE receiver x in W, [s, x] meets the solid  <=>  O_W(x) = dark}
Masses are Lebesgue volumes (uniform prior); the clearance shell (2^-26) is accounted separately.

Source-space octree over (0,1)^3.  Each node B carries the receiver blocks of W not yet certified for an ancestor.
Block tests (sound, filtered float64; docs/METHOD.md Lemmas 1-3):
  * lit block R:  all segments B-R miss every component      -> consistent, drop R
                  all segments B-R meet one component C       -> contradiction, B is OUT
  * dark block R: all segments B-R meet one component C       -> consistent, drop R
                  all segments B-R miss every component      -> contradiction, B is OUT
  * otherwise split R (or defer it to the children of B).
Single receivers are classified exactly against the H-representation of S_C(r).  When every remaining
constraint of a node is a single receiver, F_W n B is a Boolean combination of convex polytopes and its volume
is computed exactly (convex clip + inclusion-exclusion over the few non-convex constraints).

Leaf record kinds: 0 IN (whole box feasible), 1 EXACT (polytope volume), 2 UNDET (bounds only).
"""
import numpy as np
from numba import njit

from .polytope import box_polytope, clip, volume, EMPTY, OVERFLOW
from .predicates import (MARG, EXCL, CONTAINS, CUT, AMBIG, classify_component, aabb_disjoint,
                         all_visible, any_blocked, plane_box_range, MAXPL)

FINE_DEPTH = 7          # 128^3 output cells
IN_REC, EXACT_REC, UNDET_REC = 0, 1, 2


def build_pyramid(cls):
    """cls[x,y,z] in {0 dark, 1 lit, 2 solid}; returns flat uint8 codes per level (0 dark-only, 1 lit-only,
    2 solid-only, 3 mixed) and level offsets.  Level l has (L >> l)^3 blocks of side 2^l voxels."""
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
def _block_box(o, l, bx, by, bz, Ninv, rlo, rhi):
    s = 1 << l
    rlo[0] = (o[0] + bx * s + 0.5) * Ninv
    rlo[1] = (o[1] + by * s + 0.5) * Ninv
    rlo[2] = (o[2] + bz * s + 0.5) * Ninv
    rhi[0] = (o[0] + (bx + 1) * s - 0.5) * Ninv
    rhi[1] = (o[1] + (by + 1) * s - 0.5) * Ninv
    rhi[2] = (o[2] + (bz + 1) * s - 0.5) * Ninv


@njit(cache=True)
def _receiver_status(lo, hi, r, lit, P, K, clo, chi, buf, cut_idx):
    """Single receiver vs source box.  Returns 0 consistent (drop), 1 contradiction (OUT), 2 active."""
    ulo = np.empty(3)
    uhi = np.empty(3)
    for k in range(3):
        ulo[k] = min(lo[k], r[k])
        uhi[k] = max(hi[k], r[k])
    nact = 0
    for c in range(P.shape[0]):
        if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
            continue
        st, ncut = classify_component(lo, hi, r, P, K, c, buf, cut_idx)
        if st == CONTAINS:
            return 1 if lit else 0
        if st != EXCL:
            nact += 1
    if nact == 0:
        return 0 if lit else 1
    return 2


@njit(cache=True)
def _copy_poly(V, Fc, Fn, nv, nf, V2, Fc2, Fn2):
    for v in range(nv):
        V2[v, 0] = V[v, 0]; V2[v, 1] = V[v, 1]; V2[v, 2] = V[v, 2]
    for f in range(nf):
        Fn2[f] = Fn[f]
        for j in range(Fn[f]):
            Fc2[f, j] = Fc[f, j]


@njit(cache=True)
def _leaf_volume(lo, hi, conv, ncv, pc_planes, pc_start, pc_len, ct_type, ct_first, ct_npc, nct, max_terms):
    """Exact volume of  B n {conv rows <= 0} n (Boolean constraints).
    Constraint j: ct_type 0 = SUB (avoid the union of its pieces), 1 = UNI (lie in the union of its pieces);
    pieces are convex sets {rows <= 0}.  Returns (volume, upper_bound, status) with status 0 exact,
    1 too many terms (volume = -1, upper = convex relaxation), 2 overflow."""
    V, Fc, Fn, nv, nf = box_polytope(lo, hi)
    for k in range(ncv):
        st, nv, nf = clip(V, Fc, Fn, nv, nf, conv[k, 0], conv[k, 1], conv[k, 2], conv[k, 3])
        if st == EMPTY:
            return 0.0, 0.0, 0
        if st == OVERFLOW:
            return -1.0, (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]), 2
    base = volume(V, Fc, Fn, nv, nf)
    if nct == 0:
        return base, base, 0
    # number of terms
    T = 1
    for j in range(nct):
        T *= (1 << ct_npc[j])
        if T > max_terms:
            return -1.0, base, 1
    V2 = np.empty_like(V)
    Fc2 = np.empty_like(Fc)
    Fn2 = np.empty_like(Fn)
    total = 0.0
    sel = np.zeros(nct, np.int64)
    for t in range(T):
        # decode subset choice per constraint (mixed radix 2^npc)
        rem = t
        sign = 1.0
        valid = True
        for j in range(nct):
            m = rem % (1 << ct_npc[j])
            rem //= (1 << ct_npc[j])
            sel[j] = m
            cnt = 0
            for p in range(ct_npc[j]):
                if m & (1 << p):
                    cnt += 1
            if ct_type[j] == 1:
                if cnt == 0:
                    valid = False
                    break
                if cnt % 2 == 0:
                    sign = -sign
            else:
                if cnt % 2 == 1:
                    sign = -sign
        if not valid:
            continue
        _copy_poly(V, Fc, Fn, nv, nf, V2, Fc2, Fn2)
        nv2 = nv
        nf2 = nf
        empty = False
        for j in range(nct):
            for p in range(ct_npc[j]):
                if sel[j] & (1 << p):
                    pi = ct_first[j] + p
                    for q in range(pc_start[pi], pc_start[pi] + pc_len[pi]):
                        st, nv2, nf2 = clip(V2, Fc2, Fn2, nv2, nf2, pc_planes[q, 0], pc_planes[q, 1],
                                            pc_planes[q, 2], pc_planes[q, 3])
                        if st == EMPTY:
                            empty = True
                            break
                        if st == OVERFLOW:
                            return -1.0, base, 2
                    if empty:
                        break
            if empty:
                break
        if not empty:
            total += sign * volume(V2, Fc2, Fn2, nv2, nf2)
    if total < 0.0:
        total = 0.0
    if total > base:
        total = base
    return total, base, 0


@njit(cache=True)
def _cell_index(lo, depth):
    s = 1 << FINE_DEPTH
    ix = int(lo[0] * s + 1e-9)
    iy = int(lo[1] * s + 1e-9)
    iz = int(lo[2] * s + 1e-9)
    return (ix * s + iy) * s + iz


@njit(cache=True)
def _emit(lo, depth, vlo, vhi, kind, rec_cell, rec_lo, rec_hi, rec_kind, nrec):
    """Append a record for box (lo, depth); boxes coarser than the 128^3 grid are spread uniformly over
    their fine cells (exact for IN; an upper bound split evenly for UNDET is still a valid per-cell bound
    only in aggregate, so UNDET coarse boxes are emitted with their full volume on every cell they cover
    divided evenly -- the global undetermined mass U is what the certificate uses)."""
    if depth >= FINE_DEPTH:
        rec_cell[nrec] = _cell_index(lo, depth); rec_lo[nrec] = vlo; rec_hi[nrec] = vhi
        rec_kind[nrec] = kind
        return nrec + 1
    n = 1 << (FINE_DEPTH - depth)
    cs = 1.0 / (1 << FINE_DEPTH)
    frac = 1.0 / (n * n * n)
    c0 = np.empty(3)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                c0[0] = lo[0] + ix * cs; c0[1] = lo[1] + iy * cs; c0[2] = lo[2] + iz * cs
                rec_cell[nrec] = _cell_index(c0, FINE_DEPTH); rec_lo[nrec] = vlo * frac
                rec_hi[nrec] = vhi * frac; rec_kind[nrec] = kind
                nrec += 1
    return nrec


@njit(cache=True)
def label_window(P, K, clo, chi, o, q, pyr, off, d_min, d_max, kappa, max_terms, max_nodes, max_pending,
                 rec_cell, rec_lo, rec_hi, rec_kind, stats, dstat, sentinels):
    """Run CEPL for one window.  Records are appended into rec_* (cell index at depth 7, lower, upper volume,
    kind); returns the number of records.  stats (int64[16]) receives counters (see STAT_NAMES)."""
    Ninv = 1.0 / 1024.0
    L = 1 << q
    wlo = np.empty(3)
    whi = np.empty(3)
    for k in range(3):
        wlo[k] = o[k] * Ninv
        whi[k] = (o[k] + L) * Ninv
    SEG = max_pending
    seg = np.empty(SEG, np.int64)
    work = np.empty(1 << 22, np.int64)
    buf = np.empty((MAXPL, 4))
    cut_idx = np.empty(MAXPL, np.int64)
    # constraint buffers for leaves
    MAXCV = 1 << 16
    conv = np.empty((MAXCV, 4))
    MAXPP = 1 << 14
    pc_planes = np.empty((MAXPP, 4))
    pc_start = np.empty(4096, np.int64)
    pc_len = np.empty(4096, np.int64)
    ct_type = np.empty(1024, np.int64)
    ct_first = np.empty(1024, np.int64)
    ct_npc = np.empty(1024, np.int64)
    # DFS stack
    MAXD = 64
    st_lo = np.empty((MAXD * 8, 3))
    st_hi = np.empty((MAXD * 8, 3))
    st_depth = np.empty(MAXD * 8, np.int64)
    st_a = np.empty(MAXD * 8, np.int64)   # parent's pending segment [a, b)
    st_b = np.empty(MAXD * 8, np.int64)
    sp = 0
    st_lo[0, 0] = 0.0; st_lo[0, 1] = 0.0; st_lo[0, 2] = 0.0
    st_hi[0, 0] = 1.0; st_hi[0, 1] = 1.0; st_hi[0, 2] = 1.0
    st_depth[0] = 0
    # root pending list: sentinel single receivers first (strong exact refutations at every depth), then the
    # whole-window block.  Sentinels are ordinary observed receivers, so this changes cost, never the result.
    ns = sentinels.shape[0]
    for t in range(ns):
        seg[t] = sentinels[t]
    seg[ns] = _enc(q, 0, 0, 0)
    st_a[0] = 0
    st_b[0] = ns + 1
    sp = 1
    top = ns + 1     # first free slot in seg
    nrec = 0
    nodes = 0
    rlo = np.empty(3)
    rhi = np.empty(3)
    r = np.empty(3)
    lo = np.empty(3)
    hi = np.empty(3)
    # owner of each seg region: segments are stacked; a frame's children write above its own segment.
    seg_top_of = np.empty(MAXD * 8, np.int64)
    seg_top_of[0] = ns + 1
    while sp > 0:
        sp -= 1
        for k in range(3):
            lo[k] = st_lo[sp, k]
            hi[k] = st_hi[sp, k]
        depth = st_depth[sp]
        a = st_a[sp]
        b = st_b[sp]
        top = seg_top_of[sp]
        nodes += 1
        dstat[depth, 0] += 1
        side = hi[0] - lo[0]
        vol = side * side * side
        budget_hit = nodes > max_nodes
        # window inclusion
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
        if budget_hit:
            # unresolved: whole box is an upper bound
            nrec = _emit(lo, depth, 0.0, vol, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, nrec)
            stats[2] += 1
            continue
        # ---- process pending blocks ----
        out0 = top
        w = top
        nwork = 0
        out_flag = False
        force = depth >= d_min
        n_single = 0
        idx = a
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
                rs = _receiver_status(lo, hi, r, bc == 1, P, K, clo, chi, buf, cut_idx)
                if rs == 0:
                    continue
                if rs == 1:
                    out_flag = True
                    break
                if w >= SEG:
                    stats[9] += 1
                    out_flag = False
                    budget_hit = True
                    break
                seg[w] = code
                w += 1
                n_single += 1
                continue
            _block_box(o, l, bx, by, bz, Ninv, rlo, rhi)
            stats[4] += 1
            dstat[depth, 2] += 1
            decided = False
            if bc == 1:
                if all_visible(lo, hi, rlo, rhi, P, K, clo, chi):
                    continue
                if any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, buf):
                    out_flag = True
                    break
            elif bc == 0:
                if any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, buf):
                    continue
                if all_visible(lo, hi, rlo, rhi, P, K, clo, chi):
                    out_flag = True
                    break
            bside = (1 << l) * Ninv
            if force or bside > kappa * side:
                if nwork + 8 > work.shape[0]:
                    stats[9] += 1
                    budget_hit = True
                    break
                for cx in range(2):
                    for cy in range(2):
                        for cz in range(2):
                            work[nwork] = _enc(l - 1, 2 * bx + cx, 2 * by + cy, 2 * bz + cz)
                            nwork += 1
            else:
                if w >= SEG:
                    stats[9] += 1
                    budget_hit = True
                    break
                seg[w] = code
                w += 1
        if out_flag:
            stats[5] += 1
            dstat[depth, 3] += 1
            continue
        nkept = w - out0
        dstat[depth, 4] += nkept
        if budget_hit:
            nrec = _emit(lo, depth, 0.0, vol, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, nrec)
            stats[2] += 1
            continue
        # window partial overlap acts as a SUB constraint with one box piece
        win_partial = (not disjoint_w)
        if nkept == 0 and not win_partial:
            # whole box feasible
            nrec = _emit(lo, depth, vol, vol, IN_REC, rec_cell, rec_lo, rec_hi, rec_kind, nrec)
            stats[6] += 1
            continue
        all_single = n_single == nkept
        if depth >= d_min and all_single:
            # ---- exact leaf ----
            ncv = 0
            npc = 0
            npp = 0
            nct = 0
            ambiguous = False
            overflow = False
            for t in range(out0, w):
                l, bx, by, bz = _dec(seg[t])
                bc = _block_code(pyr, off, q, l, bx, by, bz)
                r[0] = (o[0] + bx + 0.5) * Ninv
                r[1] = (o[1] + by + 0.5) * Ninv
                r[2] = (o[2] + bz + 0.5) * Ninv
                lit = bc == 1
                ulo = np.empty(3)
                uhi = np.empty(3)
                for k in range(3):
                    ulo[k] = min(lo[k], r[k])
                    uhi[k] = max(hi[k], r[k])
                first_piece = npc
                for c in range(P.shape[0]):
                    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
                        continue
                    stc, ncut = classify_component(lo, hi, r, P, K, c, buf, cut_idx)
                    if stc == AMBIG:
                        ambiguous = True
                        break
                    if stc != CUT:
                        continue
                    if npc >= pc_start.shape[0] or npp + ncut > pc_planes.shape[0]:
                        overflow = True
                        break
                    pc_start[npc] = npp
                    pc_len[npc] = ncut
                    for u in range(ncut):
                        for k in range(4):
                            pc_planes[npp, k] = buf[cut_idx[u], k]
                        npp += 1
                    npc += 1
                if ambiguous or overflow:
                    break
                npieces = npc - first_piece
                if npieces == 0:
                    continue
                if lit and npieces == 1 and pc_len[first_piece] == 1:
                    # avoid {g <= 0}  <=>  -g <= 0 (boundary has measure zero)
                    if ncv >= MAXCV:
                        overflow = True
                        break
                    q0 = pc_start[first_piece]
                    for k in range(4):
                        conv[ncv, k] = -pc_planes[q0, k]
                    ncv += 1
                    npc = first_piece
                    npp = pc_start[first_piece]
                elif (not lit) and npieces == 1:
                    q0 = pc_start[first_piece]
                    for u in range(pc_len[first_piece]):
                        if ncv >= MAXCV:
                            overflow = True
                            break
                        for k in range(4):
                            conv[ncv, k] = pc_planes[q0 + u, k]
                        ncv += 1
                    npc = first_piece
                    npp = pc_start[first_piece]
                else:
                    if nct >= ct_type.shape[0]:
                        overflow = True
                        break
                    ct_type[nct] = 0 if lit else 1
                    ct_first[nct] = first_piece
                    ct_npc[nct] = npieces
                    nct += 1
            if win_partial and not (ambiguous or overflow):
                if npc < pc_start.shape[0] and npp + 6 <= pc_planes.shape[0] and nct < ct_type.shape[0]:
                    pc_start[npc] = npp
                    pc_len[npc] = 6
                    for k in range(3):
                        for sgn in range(2):
                            for u in range(4):
                                pc_planes[npp, u] = 0.0
                            if sgn == 0:
                                pc_planes[npp, k] = -1.0; pc_planes[npp, 3] = wlo[k]
                            else:
                                pc_planes[npp, k] = 1.0; pc_planes[npp, 3] = -whi[k]
                            npp += 1
                    ct_type[nct] = 0; ct_first[nct] = npc; ct_npc[nct] = 1
                    npc += 1
                    nct += 1
                else:
                    overflow = True
            if ambiguous:
                stats[10] += 1
            if not (ambiguous or overflow):
                v, ub, stl = _leaf_volume(lo, hi, conv, ncv, pc_planes, pc_start, pc_len, ct_type, ct_first,
                                          ct_npc, nct, max_terms)
                if stl == 0:
                    if v > 0.0:
                        rec_cell[nrec] = _cell_index(lo, depth); rec_lo[nrec] = v; rec_hi[nrec] = v
                        rec_kind[nrec] = EXACT_REC; nrec += 1
                    stats[7] += 1
                    stats[11] += ncv
                    if nct > stats[12]:
                        stats[12] = nct
                    continue
                if depth >= d_max:
                    rec_cell[nrec] = _cell_index(lo, depth); rec_lo[nrec] = 0.0; rec_hi[nrec] = ub
                    rec_kind[nrec] = UNDET_REC; nrec += 1
                    stats[8] += 1
                    continue
            elif depth >= d_max:
                rec_cell[nrec] = _cell_index(lo, depth); rec_lo[nrec] = 0.0; rec_hi[nrec] = vol
                rec_kind[nrec] = UNDET_REC; nrec += 1
                stats[8] += 1
                continue
        if depth >= d_max:
            nrec = _emit(lo, depth, 0.0, vol, UNDET_REC, rec_cell, rec_lo, rec_hi, rec_kind, nrec)
            stats[8] += 1
            continue
        # ---- split the source box ----
        if sp + 8 > st_lo.shape[0]:
            stats[9] += 1
            continue
        h = side * 0.5
        for cx in range(2):
            for cy in range(2):
                for cz in range(2):
                    st_lo[sp, 0] = lo[0] + cx * h; st_lo[sp, 1] = lo[1] + cy * h; st_lo[sp, 2] = lo[2] + cz * h
                    st_hi[sp, 0] = st_lo[sp, 0] + h; st_hi[sp, 1] = st_lo[sp, 1] + h
                    st_hi[sp, 2] = st_lo[sp, 2] + h
                    st_depth[sp] = depth + 1
                    st_a[sp] = out0
                    st_b[sp] = w
                    seg_top_of[sp] = w
                    sp += 1
        if nrec > rec_cell.shape[0] - (1 << 22):
            stats[13] = 1   # record buffer exhausted: result INVALID (caller must not emit a label)
            break
    stats[0] = nodes
    return nrec


STAT_NAMES = ["nodes", "inside_window", "budget_unresolved", "receiver_tests", "block_tests", "out_boxes",
              "in_boxes", "exact_leaves", "undet_leaves", "overflow", "ambiguous_leaves", "conv_planes_total",
              "max_nonconvex_constraints", "invalid_record_overflow", "unused14", "unused15"]


def choose_sentinels(cls, n_boundary=512, n_class=64, seed=0):
    """Boundary FREE receivers (6-neighbour of the other FREE class) plus random lit / dark receivers,
    encoded as level-0 block codes.  Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    lit = cls == 1
    dark = cls == 0
    bd = np.zeros_like(lit)
    for ax in range(3):
        for sh in (1, -1):
            nb_l = np.roll(lit, sh, axis=ax)
            nb_d = np.roll(dark, sh, axis=ax)
            edge = np.zeros_like(lit)
            sl = [slice(None)] * 3
            sl[ax] = slice(0, 1) if sh == 1 else slice(-1, None)
            nb_l[tuple(sl)] = False
            nb_d[tuple(sl)] = False
            bd |= (lit & nb_d) | (dark & nb_l)
    picks = []
    for mask, n in ((bd, n_boundary), (lit & ~bd, n_class), (dark & ~bd, n_class)):
        idx = np.flatnonzero(mask.reshape(-1))
        if len(idx):
            picks.append(rng.choice(idx, min(n, len(idx)), replace=False))
    idx = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    L = cls.shape[0]
    x, y, z = np.unravel_index(idx, (L, L, L))
    return (np.int64(0) << 48) | (x.astype(np.int64) << 32) | (y.astype(np.int64) << 16) | z.astype(np.int64)


def run_window(P, K, clo, chi, cls, origin, d_min=FINE_DEPTH, d_max=12, kappa=1.0, max_terms=256,
               max_nodes=5_000_000, max_pending=1 << 24, max_records=1 << 24, n_boundary=512, n_class=64,
               sentinel_seed=0):
    pyr, off, q = build_pyramid(cls)
    sentinels = choose_sentinels(cls, n_boundary, n_class, sentinel_seed)
    o = np.asarray(origin, np.int64)
    rec_cell = np.empty(max_records, np.int64)
    rec_lo = np.empty(max_records)
    rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8)
    stats = np.zeros(16, np.int64)
    dstat = np.zeros((32, 5), np.int64)
    n = label_window(P, K, clo, chi, o, q, pyr, off, d_min, d_max, kappa, max_terms, max_nodes, max_pending,
                     rec_cell, rec_lo, rec_hi, rec_kind, stats, dstat, sentinels)
    return dict(cell=rec_cell[:n].copy(), lo=rec_lo[:n].copy(), hi=rec_hi[:n].copy(), kind=rec_kind[:n].copy(),
                stats=dict(zip(STAT_NAMES, stats.tolist())), depth_stats=dstat)
