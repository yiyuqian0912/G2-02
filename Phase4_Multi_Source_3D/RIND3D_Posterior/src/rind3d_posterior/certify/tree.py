"""Proof-tree traversal of the verifier (numba): re-derives every claim of the labeller's event stream.

Per root box of the stage-1 octree the labeller wrote, in DFS order, a sequence of nodes; a node is zero or more CLIP
events followed by one terminal event (OUT / IN / UNDET / EMPTY / SPLIT_*), and after a SPLIT the B child
({plane >= 0}) subtree precedes the A child ({plane <= 0}) subtree. The verifier keeps its own exact polytope for
every region (certify.poly) and its own list of undecided receiver items (pure blocks of the window), and:

* materialises every oblique plane itself from exact geometry (`_pencil_plane`, `_face_plane`): the labeller only
  names it (component face, or occluder-edge pencil through a receiver). For an exact affine function g with float
  approximation g~ and error bound E on [0, 1]^3, the children are A' = {g~ + E + d <= 0} (g < 0 strictly),
  B' = {-(g~ - E - d) <= 0} (g > 0 strictly), and the slab between them is UNDET (pooled upper volume);
* proves an OUT claim by re-evaluating the named witness item (or, failing that, all items) on the exact region;
* proves an IN claim by evaluating every receiver item on the region with full block expansion: every DARK item must
  have a COVERing component and every LIT item NOCOVER by all components; failure makes the leaf UNDET;
* never trusts a labeller decision: a wrong or unprovable claim only moves volume from 'certified' to 'undetermined'.
Volumes: IN leaves contribute rigorous [lo, hi] per 128^3 cell (exact cell split; the 2^-26 clearance shell of the
legal domain is subtracted from lo through an outer polytope); UNDET leaves contribute [0, hi]; slabs go to a pooled
upper volume that the TV bound treats as maximally uncertain mass."""
import numpy as np
from numba import njit

from .ar import U, INFL, TINY, G2, dn, up
from .poly import (set_box, copy_slot, clip, volume, sos_sign, OK, EMPTY, OVERFLOW, NOOP, MV)
from .check import (COVER, NOCOVER, UNDEC, DARK, LIT, region_face_values, region_tau_table, box_face_values,
                    comp_status, item_box, poly_bbox, face_at_point)

EV_ROOT, EV_RCLIP, EV_ROOT_RESET, EV_SPLIT_AXIS, EV_SPLIT_FACE, EV_SPLIT_PENCIL, EV_CLIP, EV_OUT, EV_IN, EV_UNDET, \
    EV_EMPTY = range(1, 12)
OUT_ITEM, OUT_WINDOW, OUT_SOLID, OUT_PENCIL, OUT_ROOT = range(5)
REC_IN, REC_LO, REC_UNDET = 0, 1, 2
G = 128
SLAB_REL = 2.0 ** -46
EVAL_ONE_SCRATCH = 1 << 16       # last entries of the item arena: scratch of single-witness evaluations
# statistics slots
(ST_NODES, ST_IN_OK, ST_IN_FAIL, ST_OUT_OK, ST_OUT_FAIL, ST_SLABS, ST_UNDET, ST_EMPTY_EXACT, ST_EMPTY_SLIVER,
 ST_OVERFLOW, ST_EVALS, ST_SELF_OUT, ST_CLIP_OUT_OK, ST_CLIP_OUT_FAIL, ST_ARENA, ST_PIECES, ST_SHELL, ST_NPL, ST_APEX,
 ST_APEX_FAIL) = range(20)
NSTATS = 20


@njit(cache=True)
def _add_plane(PL, npl, a0, a1, a2, b):
    n = npl[0]
    if n >= PL.shape[0]:
        return -1
    PL[n, 0] = a0; PL[n, 1] = a1; PL[n, 2] = a2; PL[n, 3] = b
    npl[0] = n + 1
    return n


@njit(cache=True)
def _mr_mul(am, ar, bm, br):
    """(m, r) with |a b - m| <= r for a in am +- ar, b in bm +- br (rigorous, including rounding of am * bm)."""
    m = am * bm
    r = (ar * (abs(bm) + br) + abs(am) * br) * INFL + U * abs(m) * INFL + TINY
    return m, r


@njit(cache=True)
def _pencil_coef(c, f, b, x0, x1, x2, RM, RR, RL, RR2, out_m, out_r):
    """Exact pencil function g(s) = f_f(x) f_b(s) - f_b(x) f_f(s) through the line of faces (f, b) of component c and
    receiver x: binary64 coefficients out_m (4) and radii out_r (4) with |coef - out_m| <= out_r."""
    ffm, ffr = face_at_point(c, f, x0, x1, x2, RM, RL, RR2)
    fbm, fbr = face_at_point(c, b, x0, x1, x2, RM, RL, RR2)
    for t in range(4):
        p, pr = _mr_mul(ffm, ffr, RM[c, b, t], RR[c, b, t])
        q, qr = _mr_mul(fbm, fbr, RM[c, f, t], RR[c, f, t])
        m = p - q
        out_m[t] = m
        out_r[t] = (pr + qr) * INFL + U * abs(m) * INFL + TINY


@njit(cache=True)
def _shift_planes(PL, npl, gm, gr, sigma, rel):
    """For the exact affine function sigma * g (coefficients gm +- gr), add the two parallel binary64 planes
    A: {sigma g~ + E + d <= 0} (subset of {sigma g < 0}) and B: {-(sigma g~ - E - d) <= 0} (subset of {sigma g > 0}),
    E >= sup over [0, 1]^3 of |g - g~|, d = rel * |g~|_1 (rel = 0: inner planes, d only from E). Their complements
    are the OUTER children {sigma g~ - E - d <= 0} (superset of {sigma g <= 0}) and {sigma g~ + E + d >= 0}.
    Returns (ref_A, ref_B) or (-1, -1)."""
    a0 = sigma * gm[0]; a1 = sigma * gm[1]; a2 = sigma * gm[2]; b = sigma * gm[3]
    E = (gr[0] + gr[1] + gr[2] + gr[3]) * INFL + TINY
    d = rel * (abs(a0) + abs(a1) + abs(a2) + abs(b))
    s = up(E + d)
    bA = up(b + s)                  # larger offset => smaller half-space {a.x + bA <= 0}
    bB = dn(b - s)                  # {a.x + bB >= 0} == {-a.x - bB <= 0}
    if not (abs(a0) + abs(a1) + abs(a2) > 0.0):
        return -1, -1
    iA = _add_plane(PL, npl, a0, a1, a2, bA)
    iB = _add_plane(PL, npl, a0, a1, a2, bB)
    if iA < 0 or iB < 0:
        return -1, -1
    return 2 * iA, 2 * iB + 1


@njit(cache=True)
def _vox_point(xflat, L, o, N, out):
    vx = xflat // (L * L); vy = (xflat // L) % L; vz = xflat % L
    out[0] = (o[0] + vx + 0.5) / N; out[1] = (o[1] + vy + 0.5) / N; out[2] = (o[2] + vz + 0.5) / N
    return (vx * L + vy) * L + vz


@njit(cache=True)
def _skip(ev, pos):
    """Position after the subtree starting at pos (structural parse)."""
    need = 1
    n = ev.shape[0]
    while need > 0 and pos < n:
        t = ev[pos, 0]
        pos += 1
        if t == EV_CLIP:
            continue
        if t == EV_SPLIT_AXIS or t == EV_SPLIT_FACE or t == EV_SPLIT_PENCIL:
            need += 1
        else:
            need -= 1
    return pos


@njit(cache=True)
def eval_node(slot, full, AC, AK, AM, ioff, ilen, ooff, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit):
    """Evaluate the item list AC/AK/AM[ioff:ioff+ilen] on the TRUE region of the polytope in `slot` (the outer
    polytope in the slot plus the symbolic facts `fa` and face-aspect overrides `ao` of the path).
    full: 0 normal (undecided blocks expanded on the dual-tree hint, survivors written), 1 IN proof (every undecided
    block expanded, an undecided receiver fails), 2 refutation search (every undecided block expanded down to
    receivers, nothing written, stops at the first refuting item).
    Returns (status, nout): status 0 = survivors written at AC[ooff:ooff+nout], 1 = region refuted (wit[0:2] =
    (code, component or -1)), 2 = undecided receiver in full mode, 3 = queue/arena overflow, 4 = region inside a
    component (outside the legal domain)."""
    C, K, RM, RR, FB, RL, RR2, alo, ahi = geo
    Wlo, Whi, o, N, L, cls_flat = win
    FC, FI, FJ, FX, FS, FVm, FVr, nfa = fa
    AOc, AOi, AOs, nao = ao
    Vm, Vr, nV = polyv
    fvm, fvr, fRlo, fRhi, asp, Tlo, Thi, fqm, fqr, fql, fqh, cflat = scr
    Rlo = np.empty(3); Rhi = np.empty(3)
    poly_bbox(slot, Vm, Vr, nV, Rlo, Rhi)
    relmask = np.int64(0)
    nv = nV[slot]
    for c in range(C):
        rel = True
        for k in range(3):
            lo = min(Rlo[k], Wlo[k]); hi = max(Rhi[k], Whi[k])
            if hi < alo[c, k] or lo > ahi[c, k]:
                rel = False
        if not rel:
            continue
        relmask |= np.int64(1) << c
        region_face_values(slot, c, K, RM, FB, RR, Vm, Vr, nV, fRlo[c], fRhi[c], fvm[c], fvr[c])
        for i in range(K[c]):
            if fRlo[c, i] > 0.0:
                asp[c, i] = 1
            elif fRhi[c, i] < 0.0:
                asp[c, i] = -1
            else:
                asp[c, i] = 0
        for t in range(nao):
            if AOc[t] == c:
                asp[c, AOi[t]] = AOs[t]          # symbolic: the true region is on this side of the face plane
        anyfront = False; fixed = True
        for i in range(K[c]):
            if asp[c, i] == 1:
                anyfront = True
            elif asp[c, i] == 0:
                fixed = False
        if fixed and not anyfront:
            return 4, 0                         # region inside the closed component c (a.e.)
        region_tau_table(nv, K[c], asp[c], fvm[c], fvr[c], Tlo[c], Thi[c])
    qlo = np.empty(3); qhi = np.empty(3)
    NW = WC.shape[0]; NA = AC.shape[0]
    nq = 0
    nout = 0
    i = 0
    while i < ilen + nq:
        if i < ilen:
            code = AC[ioff + i]; kls = AK[ioff + i]; mask = AM[ioff + i] & relmask
        else:
            code = WC[i - ilen]; kls = WK[i - ilen]; mask = WM[i - ilen] & relmask
        i += 1
        stats[ST_EVALS] += 1
        lvl = item_box(code, o, N, qlo, qhi)
        ispt = lvl == 0
        if nfa > 0:
            sz = np.int64(1) << lvl
            bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
            for cr in range(1 if ispt else 8):
                vx = bx * sz + ((sz - 1) if (cr & 1) else 0); vy = by * sz + ((sz - 1) if (cr & 2) else 0)
                vz = bz * sz + ((sz - 1) if (cr & 4) else 0)
                cflat[cr] = (vx * L + vy) * L + vz
        newmask = np.int64(0)
        hint = False
        if kls == DARK:
            cover = False
            for c in range(C):
                if not (mask >> c) & 1:
                    continue
                nc = box_face_values(qlo, qhi, ispt, c, K[c], RM, RL, RR2, fqm, fqr, fql, fqh)
                st, h = comp_status(c, K[c], asp[c], fRlo[c], fRhi[c], Tlo[c], Thi[c], nc, fqm, fqr, fql, fqh, Rlo, Rhi,
                                    qlo, qhi, alo, ahi, nv, fvm[c], fvr[c], cflat, FC, FI, FJ, FX, FS, nfa, FVm, FVr)
                if st == COVER:
                    cover = True
                    break
                if st == UNDEC:
                    newmask |= np.int64(1) << c
                    hint = hint or h
            if cover:
                continue
            if newmask == 0:
                wit[0] = code; wit[1] = -1
                return 1, 0
        else:
            for c in range(C):
                if not (mask >> c) & 1:
                    continue
                nc = box_face_values(qlo, qhi, ispt, c, K[c], RM, RL, RR2, fqm, fqr, fql, fqh)
                st, h = comp_status(c, K[c], asp[c], fRlo[c], fRhi[c], Tlo[c], Thi[c], nc, fqm, fqr, fql, fqh, Rlo, Rhi,
                                    qlo, qhi, alo, ahi, nv, fvm[c], fvr[c], cflat, FC, FI, FJ, FX, FS, nfa, FVm, FVr)
                if st == COVER:
                    wit[0] = code; wit[1] = c
                    return 1, 0
                if st == UNDEC:
                    newmask |= np.int64(1) << c
                    hint = hint or h
            if newmask == 0:
                continue
        if not ispt and (full != 0 or hint):
            if nq + 8 > NW:
                return 3, 0
            l1 = lvl - 1
            bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
            for c8 in range(8):
                WC[nq] = (np.int64(l1) << 48) | (np.int64(2 * bx + (c8 & 1)) << 32) | \
                         (np.int64(2 * by + ((c8 >> 1) & 1)) << 16) | np.int64(2 * bz + ((c8 >> 2) & 1))
                WK[nq] = kls; WM[nq] = newmask
                nq += 1
            continue
        if full == 1:
            return 2, 0
        if full == 2:
            continue                       # search mode: undecided receivers are simply not refuters
        if ooff + nout >= NA:
            return 3, 0
        AC[ooff + nout] = code; AK[ooff + nout] = kls; AM[ooff + nout] = newmask
        nout += 1
    return 0, nout


@njit(cache=True)
def eval_one(slot, code, kls, C, AC, AK, AM, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit):
    """True if the single receiver item (code, observed class kls) refutes the whole true region in `slot`."""
    top = AC.shape[0] - EVAL_ONE_SCRATCH
    AC[top] = code; AK[top] = kls; AM[top] = (np.int64(1) << C) - 1
    st, n = eval_node(slot, 2, AC, AK, AM, top, 1, top + 1, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit)
    return st == 1 or st == 4


@njit(cache=True)
def _in_axis_halfspace(PL, slot, ref, Vp, Vm, Vr, nV):
    """Exact test: every vertex of the polytope satisfies the closed half-space ref (vertices on the plane count)."""
    pid = ref >> 1; sg = -1 if (ref & 1) else 1
    for v in range(nV[slot]):
        if Vp[slot, v, 0] == pid or Vp[slot, v, 1] == pid or Vp[slot, v, 2] == pid:
            continue
        if sos_sign(PL, pid, Vp[slot, v, 0], Vp[slot, v, 1], Vp[slot, v, 2], Vm[slot, v], Vr[slot, v]) * sg > 0:
            return False
    return True


@njit(cache=True)
def _emit(PL, npl, slot, kind, scr0, nscr, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF,
          rec_cell, rec_lo, rec_hi, rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats):
    """Split the polytope in `slot` along the 128^3 grid (exact axis planes) and record per cell
    IN: [vol_lo - shell_hi, vol_hi], UNDET: [0, vol_hi]. Returns (nrec, ok); ok False on scratch exhaustion (the
    caller then records the whole region in the pool)."""
    RC = rec_cell.shape[0]
    copy_slot(slot, scr0, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    top = 1
    lo = np.empty(3); hi = np.empty(3)
    shell_slot = scr0 + nscr - 1
    # exact cell-index bounds per scratch level: [ca, cb] per axis; the initial bounds come from the outward
    # enclosure (a superset of the true range); every split restricts them through the exact grid half-space
    ca = np.zeros((nscr, 3), np.int64); cb = np.zeros((nscr, 3), np.int64)
    poly_bbox(slot, Vm, Vr, nV, lo, hi)
    for k in range(3):
        ca[0, k] = min(max(int(np.floor(lo[k] * G)), 0), G - 1)
        cb[0, k] = min(max(int(np.floor(hi[k] * G)), 0), G - 1)
    while top > 0:
        top -= 1
        q = scr0 + top
        if nF[q] == 0:
            continue
        ax = -1; span = 0
        for k in range(3):
            if cb[top, k] - ca[top, k] > span:
                span = cb[top, k] - ca[top, k]; ax = k
        if ax >= 0:
            if top + 2 > nscr - 1:
                return nrec, False
            j = (ca[top, ax] + cb[top, ax] + 1) // 2           # ca < j <= cb: plane x = j / G
            pid = gid[ax, j]
            copy_slot(q, q + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            for k in range(3):
                ca[top + 1, k] = ca[top, k]; cb[top + 1, k] = cb[top, k]
            cb[top, ax] = j - 1                                # A: x <= j/G
            ca[top + 1, ax] = j                                # B: x >= j/G
            rA = clip(PL, q, 2 * pid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            rB = clip(PL, q + 1, 2 * pid + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            if rA == OVERFLOW or rB == OVERFLOW:
                return nrec, False
            top += 2
            continue
        cid = (ca[top, 0] * G + ca[top, 1]) * G + ca[top, 2]
        vlo, vhi = volume(q, Fn, Fv, Vm, Vr, nF)
        if vhi <= 0.0:
            continue
        rlo = 0.0
        rhi = vhi if kind != REC_LO else 0.0
        if kind == REC_IN or kind == REC_LO:
            sh = _shell_upper(PL, npl, q, shell_slot, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, C, K, RM, RR, alo, ahi, shell_eps,
                              stats)
            rlo = max(dn(vlo - sh), 0.0)
        if nrec < RC:
            rec_cell[nrec] = cid; rec_lo[nrec] = rlo; rec_hi[nrec] = rhi; rec_kind[nrec] = kind
        nrec += 1
        stats[ST_PIECES] += 1
    return nrec, True


@njit(cache=True)
def _shell_upper(PL, npl, q, t, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, C, K, RM, RR, alo, ahi, eps, stats):
    """Upper bound of vol(piece n {s : dist(s, S) <= eps}) <= sum_c vol(piece n {f_i <= eps |n_i| + E_i}) through
    outward binary64 planes; the whole piece volume if a plane cannot be added."""
    lo = np.empty(3); hi = np.empty(3)
    poly_bbox(q, Vm, Vr, nV, lo, hi)
    tot = 0.0
    for c in range(C):
        sep = False
        for k in range(3):
            if hi[k] < alo[c, k] - 2.0 * eps or lo[k] > ahi[c, k] + 2.0 * eps:
                sep = True
        if sep:
            continue
        stats[ST_SHELL] += 1
        copy_slot(q, t, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        empty = False
        for i in range(K[c]):
            a0 = RM[c, i, 0]; a1 = RM[c, i, 1]; a2 = RM[c, i, 2]
            nrm = up(np.sqrt(up(up(up(a0 * a0) + up(a1 * a1)) + up(a2 * a2))))
            E = (RR[c, i, 0] + RR[c, i, 1] + RR[c, i, 2] + RR[c, i, 3]) * INFL + TINY + up(nrm * (RR[c, i, 0] + RR[c, i, 1] + RR[c, i, 2]))
            b = dn(dn(RM[c, i, 3] - up(eps * nrm)) - E)
            pid = _add_plane(PL, npl, a0, a1, a2, b)
            if pid < 0:
                v = volume(q, Fn, Fv, Vm, Vr, nF)
                return v[1]
            r = clip(PL, t, 2 * pid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            if r == EMPTY:
                empty = True
                break
            if r == OVERFLOW:
                v = volume(q, Fn, Fv, Vm, Vr, nF)
                return v[1]
        if not empty:
            v = volume(t, Fn, Fv, Vm, Vr, nF)
            tot = up(tot + v[1])
    return tot


@njit(cache=True)
def _undet(PL, npl, q, scr0, nscr, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi, rec_kind, nrec,
           C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, kind):
    """Record slot q as UNDET (kind REC_UNDET) or IN (REC_IN) per 128^3 cell; pooled if the cell split cannot run."""
    if nF[q] == 0:
        return nrec
    nrec, ok = _emit(PL, npl, q, kind, scr0, nscr, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                     rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats)
    if not ok:
        vv = volume(q, Fn, Fv, Vm, Vr, nF)[1]
        pool[0] = up(pool[0] + vv); pool[4] = up(pool[4] + vv)
    return nrec


@njit(cache=True)
def _dist_up_seg(vm, vr, A, B, er):
    """Rigorous upper bound of the distance from the enclosed point vm +- vr to the exact segment [A, B] (endpoints
    enclosed within +- er): the distance to the point A + t (B - A) of the segment for one t in [0, 1]."""
    d0 = B[0] - A[0]; d1 = B[1] - A[1]; d2 = B[2] - A[2]
    ll = d0 * d0 + d1 * d1 + d2 * d2
    t = 0.0
    if ll > 0.0:
        t = ((vm[0] - A[0]) * d0 + (vm[1] - A[1]) * d1 + (vm[2] - A[2]) * d2) / ll
        t = min(max(t, 0.0), 1.0)
    acc = 0.0
    for k in range(3):
        pk = A[k] + t * (B[k] - A[k])
        # exact point A + t (B - A): within er (endpoints) + rounding of the evaluation
        e = abs(vm[k] - pk) + vr[k] + er + 4.0 * U * (abs(A[k]) + abs(B[k]) + abs(vm[k])) + TINY
        acc += e * e
    return up(np.sqrt(up(acc * INFL)))


@njit(cache=True)
def _near_solid_vertices(slot, Vm, Vr, nV, EC, EA, EB, ER, nE, eps, h, out):
    """Vertices of the polytope whose whole cube [vm - h', vm + h'] (h' ~ h) lies within distance eps of the solid
    (certified through an occluder edge): such cubes contain no legal source. Returns their number (<= len(out))."""
    n = 0
    lim = eps - 2.0 * h * 1.7320508075688772 * INFL
    for v in range(nV[slot]):
        best = np.inf
        for e in range(nE):
            d = _dist_up_seg(Vm[slot, v], Vr[slot, v], EA[e], EB[e], ER[e])
            if d < best:
                best = d
        if best <= lim and n < out.shape[0]:
            out[n] = v
            n += 1
    return n


@njit(cache=True)
def _in_volumes(PL, npl, q, CUR, SPL, nsp, EM0, NEM, gid, P, rec_cell, rec_lo, rec_hi, rec_kind, nrec, geo, shell_eps, stats,
                pool, dbgINV, dbgIN):
    """IN region q: upper bounds from the outer polytope, lower bounds from outer n (inner planes of the path)."""
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = P
    C, K, RM, RR, FB, RL, RR2, alo, ahi = geo
    nrec = _undet(PL, npl, q, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi, rec_kind, nrec,
                  C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
    copy_slot(q, CUR, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    for u in range(nsp):
        rr = clip(PL, CUR, SPL[u], Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        if rr == OVERFLOW:
            nF[CUR] = 0          # no lower bound credited (sound)
        if nF[CUR] == 0:
            break
    if nF[CUR] > 0:
        nrec = _undet(PL, npl, CUR, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi, rec_kind,
                      nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_LO)
        k_ = dbgIN[0]
        if k_ < dbgINV.shape[0]:
            nvv = min(nV[CUR], dbgINV.shape[1])
            for v in range(nvv):
                for kk in range(3):
                    dbgINV[k_, v, kk] = Vm[CUR, v, kk]
            dbgIN[1 + k_] = nvv
            dbgIN[0] = k_ + 1
    return nrec


@njit(cache=True)
def _within_shell(q, Vm, Vr, nV, c, A, B, er, K, RM, FB, eps):
    """Every vertex of slot q lies within eps of component c (distance to the exact edge segment [A, B], or inside the
    closed component): then the whole convex polytope lies in the eps-neighbourhood of c (convex), i.e. outside the
    legal source domain."""
    for v in range(nV[q]):
        inside = True
        for i in range(K[c]):
            m = RM[c, i, 0] * Vm[q, v, 0] + RM[c, i, 1] * Vm[q, v, 1] + RM[c, i, 2] * Vm[q, v, 2] + RM[c, i, 3]
            r = FB[c, i] + (abs(RM[c, i, 0]) * Vr[q, v, 0] + abs(RM[c, i, 1]) * Vr[q, v, 1] + abs(RM[c, i, 2]) * Vr[q, v, 2]) * INFL
            if not (m + r <= 0.0):
                inside = False
                break
        if inside:
            continue
        if not (_dist_up_seg(Vm[q, v], Vr[q, v], A, B, er) <= eps):
            return False
    return True


@njit(cache=True)
def _edge_cut_pieces(PL, npl, s, P0, NP, edges, geo, eps, P, stats):
    """Cut out of slot s the wedge next to every occluder edge it touches: for edge (c, i, j) with a vertex of s on
    both facet planes (|f_i|, |f_j| <= 1e-9), N = s n {f_i >= -eta} n {f_j >= -eta} is dropped when certified to lie
    within eps of the solid (_within_shell); the parts {f_i <= -eta} and {f_i >= -eta, f_j <= -eta} remain. Returns
    the number of pieces in slots P0.., or -1 (nothing cut / overflow)."""
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = P
    C, K, RM, RR, FB, RL, RR2, alo, ahi = geo
    EC, EI, EJ, EA, EB, ER, nE = edges
    copy_slot(s, P0, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    npc = 1
    cut = False
    T = P0 + NP - 1                                   # scratch for N
    for e in range(nE):
        c = EC[e]; i = EI[e]; j = EJ[e]
        touch = False
        for u in range(npc):
            q = P0 + u
            for v in range(nV[q]):
                fi = RM[c, i, 0] * Vm[q, v, 0] + RM[c, i, 1] * Vm[q, v, 1] + RM[c, i, 2] * Vm[q, v, 2] + RM[c, i, 3]
                fj = RM[c, j, 0] * Vm[q, v, 0] + RM[c, j, 1] * Vm[q, v, 1] + RM[c, j, 2] * Vm[q, v, 2] + RM[c, j, 3]
                if abs(fi) <= 1e-9 and abs(fj) <= 1e-9:
                    touch = True
                    break
            if touch:
                break
        if not touch:
            continue
        eta = 2.0 ** -34
        pi = _add_plane(PL, npl, RM[c, i, 0], RM[c, i, 1], RM[c, i, 2], RM[c, i, 3] + eta)   # {f~_i + eta <= 0}
        pj = _add_plane(PL, npl, RM[c, j, 0], RM[c, j, 1], RM[c, j, 2], RM[c, j, 3] + eta)
        if pi < 0 or pj < 0:
            return -1
        nn = npc
        for u in range(npc):
            q = P0 + u
            if nF[q] == 0:
                continue
            # N = q n {f~_i + eta >= 0} n {f~_j + eta >= 0}
            copy_slot(q, T, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            r1 = clip(PL, T, 2 * pi + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            r2 = clip(PL, T, 2 * pj + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF) if nF[T] > 0 else EMPTY
            if r1 == OVERFLOW or r2 == OVERFLOW:
                return -1
            if nF[T] > 0 and not _within_shell(T, Vm, Vr, nV, c, EA[e], EB[e], ER[e], K, RM, FB, eps):
                continue                               # cannot drop N for this piece: leave it whole
            if nn + 1 >= NP - 1:
                return -1
            # piece2 = q n {f~_i + eta >= 0} n {f~_j + eta <= 0}; q := q n {f~_i + eta <= 0}
            t2 = P0 + nn
            copy_slot(q, t2, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            ra = clip(PL, t2, 2 * pi + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            rb = clip(PL, t2, 2 * pj, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF) if nF[t2] > 0 else EMPTY
            rc = clip(PL, q, 2 * pi, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
            if ra == OVERFLOW or rb == OVERFLOW or rc == OVERFLOW:
                return -1
            if nF[t2] > 0:
                nn += 1
            cut = True
        # compact
        k2 = 0
        for u in range(nn):
            if nF[P0 + u] > 0:
                if k2 != u:
                    copy_slot(P0 + u, P0 + k2, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                k2 += 1
        npc = k2
        if npc == 0:
            break
    if not cut:
        return -1
    stats[ST_APEX] += 1
    return npc


@njit(cache=True)
def _check_claim(PL, q, claim_in, wkind, wcode, wcls, wid, ioff, ilen, atop, AC, AK, AM, WC, WK, WM, geo, win, fa, ao, polyv,
                 Vp, scr, stats, wit):
    """Prove a leaf claim on the true region of slot q. claim_in: every item consistent (IN) and the region outside
    the window interior; else OUT: the named witness (wkind 0 item (wcode, wcls), 1 window box, -1 none) or any item
    of the list refutes it. Returns 0 proved, 1 refuted although claimed IN (no mass), 2 failed, 3 overflow."""
    Vm, Vr, nV = polyv
    C = geo[0]
    if claim_in:
        st, nout = eval_node(q, 1, AC, AK, AM, ioff, ilen, atop, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit)
        if st == 3:
            return 3
        if st == 1 or st == 4:
            return 1
        if st != 0:
            return 2
        outwin = (_in_axis_halfspace(PL, q, 2 * wid[0, 0], Vp, Vm, Vr, nV) or _in_axis_halfspace(PL, q, 2 * wid[0, 1] + 1, Vp, Vm, Vr, nV)
                  or _in_axis_halfspace(PL, q, 2 * wid[1, 0], Vp, Vm, Vr, nV) or _in_axis_halfspace(PL, q, 2 * wid[1, 1] + 1, Vp, Vm, Vr, nV)
                  or _in_axis_halfspace(PL, q, 2 * wid[2, 0], Vp, Vm, Vr, nV) or _in_axis_halfspace(PL, q, 2 * wid[2, 1] + 1, Vp, Vm, Vr, nV))
        return 0 if outwin else 2
    ok = False
    if wkind == 0:
        ok = eval_one(q, wcode, np.int8(wcls), C, AC, AK, AM, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit)
    elif wkind == 1:
        ok = (_in_axis_halfspace(PL, q, 2 * wid[0, 0] + 1, Vp, Vm, Vr, nV) and _in_axis_halfspace(PL, q, 2 * wid[0, 1], Vp, Vm, Vr, nV)
              and _in_axis_halfspace(PL, q, 2 * wid[1, 0] + 1, Vp, Vm, Vr, nV) and _in_axis_halfspace(PL, q, 2 * wid[1, 1], Vp, Vm, Vr, nV)
              and _in_axis_halfspace(PL, q, 2 * wid[2, 0] + 1, Vp, Vm, Vr, nV) and _in_axis_halfspace(PL, q, 2 * wid[2, 1], Vp, Vm, Vr, nV))
    if ok:
        return 0
    st, nout = eval_node(q, 2, AC, AK, AM, ioff, ilen, atop, WC, WK, WM, geo, win, fa, ao, polyv, scr, stats, wit)
    if st == 3:
        return 3
    return 0 if (st == 1 or st == 4) else 2


@njit(cache=True)
def _minus_cubes(PL, npl, s, P0, NP, cv, nc, h, P):
    """Decompose (slot s) minus the cubes [Vm[s, cv[t]] -+ h] into convex pieces in slots P0 .. P0 + NP - 1 (the parts
    inside the cubes are dropped: they contain no legal source). Returns the number of pieces, or -1 on overflow."""
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = P
    cen = np.empty((nc, 3))
    for t in range(nc):
        for k in range(3):
            cen[t, k] = Vm[s, cv[t], k]
    copy_slot(s, P0, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    npc = 1
    for t in range(nc):
        # every current piece -> its parts outside cube t (sweep: x<lo, x>hi, y<lo, y>hi, z<lo, z>hi of the rest)
        nnew = 0
        base = npc
        for u in range(npc):
            q = P0 + u
            if nF[q] == 0:
                continue
            for k in range(3):
                for side in range(2):
                    v = dn(cen[t, k] - h) if side == 0 else up(cen[t, k] + h)
                    e0 = 1.0 if k == 0 else 0.0; e1 = 1.0 if k == 1 else 0.0; e2 = 1.0 if k == 2 else 0.0
                    pid = _add_plane(PL, npl, e0, e1, e2, -v)
                    if pid < 0 or base + nnew >= NP:
                        return -1
                    tgt = P0 + base + nnew
                    copy_slot(q, tgt, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    # outside part: side 0: x_k <= v (ref pos), side 1: x_k >= v (ref neg)
                    r = clip(PL, tgt, 2 * pid + (0 if side == 0 else 1), Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    if r == OVERFLOW:
                        return -1
                    if nF[tgt] > 0:
                        nnew += 1
                    # the rest continues inside this slab of the cube
                    r = clip(PL, q, 2 * pid + (1 if side == 0 else 0), Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    if r == OVERFLOW:
                        return -1
                    if nF[q] == 0:
                        break
                if nF[q] == 0:
                    break
            nF[q] = 0           # q is now inside cube t: dropped
        # compact
        k2 = 0
        for u in range(base + nnew):
            if nF[P0 + u] > 0:
                if k2 != u:
                    copy_slot(P0 + u, P0 + k2, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                k2 += 1
        npc = k2
        if npc == 0:
            break
    return npc


@njit(cache=True)
def _refine_claim(PL, npl, q0, P0, NP, budget, dmax, claim_in, wkind, wcode, wcls, wid, ioff, ilen, atop, AC, AK, AM, WC,
                  WK, WM, geo, win, fa, ao, P, scr, stats, wit, CUR, SPL, nsp, EM0, NEM, gid, rec_cell, rec_lo, rec_hi,
                  rec_kind, nrec, shell_eps, pool, dbg):
    """Prove a failed claim piecewise: depth-first bisection of slot q0 on exact axis planes (longest box axis), each
    piece re-proved (IN: lower bound credited from its inner part; OUT: nothing); pieces still failing at the depth or
    budget limit are UNDET. Sound for any split: the pieces tile q0. Returns (nrec, code) with code 3 on overflow."""
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = P
    C, K, RM, RR, FB, RL, RR2, alo, ahi = geo
    polyv = (Vm, Vr, nV)
    dep = np.zeros(NP, np.int64)
    copy_slot(q0, P0, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    top = 1
    used = 0
    lo = np.empty(3); hi = np.empty(3)
    while top > 0:
        top -= 1
        q = P0 + top
        if nF[q] == 0:
            continue
        used += 1
        r = _check_claim(PL, q, claim_in, wkind, wcode, wcls, wid, ioff, ilen, atop, AC, AK, AM, WC, WK, WM, geo, win, fa,
                         ao, polyv, Vp, scr, stats, wit)
        if r == 3:
            return nrec, 3
        if r == 0 and claim_in:
            nrec = _in_volumes(PL, npl, q, CUR, SPL, nsp, EM0, NEM, gid, P, rec_cell, rec_lo, rec_hi, rec_kind, nrec, geo,
                               shell_eps, stats, pool, dbg[4], dbg[5])
            continue
        if r == 0 or r == 1:
            continue
        if used >= budget or dep[top] >= dmax or top + 2 > NP:
            stats[ST_APEX_FAIL] += 1
            dbgV, dbgI, dbgF, dbgA, _d4, _d5 = dbg
            k_ = dbgI[0]
            if k_ < dbgV.shape[0] and dep[top] >= dmax:
                nvv = min(nV[q], dbgV.shape[1])
                for v in range(nvv):
                    for kk in range(3):
                        dbgV[k_, v, kk] = Vm[q, v, kk]
                dbgI[1 + 4 * k_] = -1; dbgI[2 + 4 * k_] = nvv; dbgI[3 + 4 * k_] = int(claim_in); dbgI[4 + 4 * k_] = wcode
                dbgI[0] = k_ + 1
            nrec = _undet(PL, npl, q, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                          rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
            continue
        poly_bbox(q, Vm, Vr, nV, lo, hi)
        ax = 0; ext = hi[0] - lo[0]
        for k in range(1, 3):
            if hi[k] - lo[k] > ext:
                ext = hi[k] - lo[k]; ax = k
        e0 = 1.0 if ax == 0 else 0.0; e1 = 1.0 if ax == 1 else 0.0; e2 = 1.0 if ax == 2 else 0.0
        pid = _add_plane(PL, npl, e0, e1, e2, -(0.5 * (lo[ax] + hi[ax])))
        if pid < 0:
            return nrec, 3
        copy_slot(q, q + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        ra = clip(PL, q, 2 * pid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        rb = clip(PL, q + 1, 2 * pid + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        if ra == OVERFLOW or rb == OVERFLOW:
            nrec = _undet(PL, npl, q, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                          rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
            continue
        d1 = dep[top] + 1
        dep[top] = d1; dep[top + 1] = d1
        top += 2
    stats[ST_APEX] += 1
    return nrec, 0


@njit(cache=True)
def verify_root(PL, npl, gid, wid, ev, pos, rclip_ref, root_d, root_ix, root_iy, root_iz,
                AC, AK, AM, items_code, items_cls, WC, WK, WM, geo, win, shell_eps, P, scr,
                rec_cell, rec_lo, rec_hi, rec_kind, nrec_in, pool, stats, SPL, FST, AOST, dbg, edges):
    """Verify one root subtree starting at ev[pos] (the EV_ROOT event). Returns (pos after the subtree, nrec, code):
    code 0 ok, 1 malformed stream, 3 arena overflow (rerun with larger buffers).

    True regions and their representation. Every named oblique plane (component face, or occluder-edge pencil
    g_x(s) = f_f(x) f_b(s) - f_b(x) f_f(s) through receiver x) splits a true region EXACTLY at g = 0. The verifier
    represents a true child by its OUTER polytope (the plane shifted outward by E >= |g - g~| on [0, 1]^3: a superset,
    where interval facts are proved) and records an INNER plane (shifted inward by E: a subset, where IN volume is
    credited) and a SYMBOLIC FACT (the exact sign of g_x, resp. the face aspect, on the true child). Claims are
    proved for the true leaf (interval facts on the outer polytope + symbolic facts); IN leaves credit the volume of
    outer n (inner planes of the path) as lower bound and of the outer polytope as upper bound; UNDET leaves credit
    the outer volume as upper bound. True leaves tile the root, so these are rigorous per-cell enclosures."""
    C, K, RM, RR, FB, RL, RR2, alo, ahi = geo
    Wlo, Whi, o, N, L, cls_flat = win
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = P
    polyv = (Vm, Vr, nV)
    FC, FI, FJ, FX, FS, FVm, FVr = FST
    AOc, AOi, AOs = AOST
    nrec = nrec_in
    S = Fp.shape[0]
    NSCR = 116
    SMAX = S - NSCR - 1
    scr0 = S - NSCR
    CUR = scr0; T1 = scr0 + 1; PC0 = scr0 + 2; NPC = 36; RF0 = scr0 + 38; NRF = 40; EM0 = scr0 + 78; NEM = NSCR - 78
    RBUDGET = 48; RDMAX = 10          # safety net only: failed claims are re-proved on at most 48 bisection pieces
    EC, EI, EJ, EA, EB, ER, nE = edges
    dbgINV = dbg[4]; dbgIN = dbg[5]
    HC = 2.0 ** -30                    # cube half-width (< 2^-26 / 8)
    cvbuf = np.zeros(4, np.int64)
    wit = np.zeros(2, np.int64)
    xp = np.empty(3)
    gm = np.empty(4); gr = np.empty(4)
    NSP = SPL.shape[0]; NFA = FC.shape[0]; NAO = AOc.shape[0]
    if ev[pos, 0] != EV_ROOT:
        return pos, nrec, 1
    pos += 1
    if root_d > 7:
        return pos, nrec, 1
    sc = 1 << (7 - root_d)
    ids = np.empty((3, 2), np.int64)
    ids[0, 0] = gid[0, root_ix * sc]; ids[0, 1] = gid[0, (root_ix + 1) * sc]
    ids[1, 0] = gid[1, root_iy * sc]; ids[1, 1] = gid[1, (root_iy + 1) * sc]
    ids[2, 0] = gid[2, root_iz * sc]; ids[2, 1] = gid[2, (root_iz + 1) * sc]
    set_box(PL, 0, ids, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
    while pos < ev.shape[0] and (ev[pos, 0] == EV_RCLIP or ev[pos, 0] == EV_ROOT_RESET):
        if ev[pos, 0] == EV_ROOT_RESET:
            set_box(PL, 0, ids, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
        else:
            ref = rclip_ref[ev[pos, 1]]
            if ref >= 0 and nF[0] > 0:
                r = clip(PL, 0, ref, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                if r == OVERFLOW:
                    set_box(PL, 0, ids, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    stats[ST_OVERFLOW] += 1
        pos += 1
    ni = items_code.shape[0]
    if ni > AC.shape[0] // 2:
        return pos, nrec, 3
    allm = (np.int64(1) << C) - 1
    for t in range(ni):
        AC[t] = items_code[t]; AK[t] = items_cls[t]; AM[t] = allm
    fr_off = np.zeros(SMAX + 1, np.int64); fr_len = np.zeros(SMAX + 1, np.int64)
    # path stacks (inner planes SPL, symbolic facts FST, aspect overrides AOST): counts per frame; the pending A child
    # of a split inherits the parent's counts (base) plus its own entries (saved)
    fr_nsp = np.zeros(SMAX + 1, np.int64); fr_nfa = np.zeros(SMAX + 1, np.int64); fr_nao = np.zeros(SMAX + 1, np.int64)
    fr_bsp = np.zeros(SMAX + 1, np.int64); fr_bfa = np.zeros(SMAX + 1, np.int64); fr_bao = np.zeros(SMAX + 1, np.int64)
    sA_sp = np.full(SMAX + 1, -1, np.int64)
    sA_fa = np.zeros((SMAX + 1, 5), np.int64); sA_hasfa = np.zeros(SMAX + 1, np.bool_); sA_fv = np.zeros((SMAX + 1, 4))
    sA_ao = np.zeros((SMAX + 1, 3), np.int64); sA_hasao = np.zeros(SMAX + 1, np.bool_)
    pending = np.zeros(SMAX + 1, np.bool_)
    sp = 0
    fr_off[0] = 0; fr_len[0] = ni
    atop = ni
    while True:
        s = sp
        stats[ST_NODES] += 1
        nsp = fr_nsp[s]; nfa = fr_nfa[s]; nao = fr_nao[s]
        dead = nF[s] == 0
        # ---- CLIP events of this node: removed true side {sigma g_x >= 0} is refuted by the receiver witness x
        while pos < ev.shape[0] and ev[pos, 0] == EV_CLIP:
            if not dead:
                c = ev[pos, 1]; f = ev[pos, 2]; b = ev[pos, 3]; xf = ev[pos, 4]; orn = ev[pos, 5]
                xc = _vox_point(xf, L, o, N, xp)
                _pencil_coef(c, f, b, xp[0], xp[1], xp[2], RM, RR, RL, RR2, gm, gr)
                rK, rO = _shift_planes(PL, npl, gm, gr, float(orn), 0.0)
                if rK < 0 or nsp >= NSP or nfa + 1 >= NFA:
                    stats[ST_OVERFLOW] += 1         # region kept unclipped: sound (the removed side stays in the tree)
                else:
                    # removed side, outer polytope {sigma g~ + E >= 0} = complement of the kept inner plane rK
                    copy_slot(s, T1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    r1 = clip(PL, T1, rK ^ 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    if r1 != EMPTY and nF[T1] > 0:
                        # temporary fact: on the removed true side, B_fb(x, s) has sign +orn  (sigma g >= 0)
                        FC[nfa] = c; FI[nfa] = f; FJ[nfa] = b; FX[nfa] = xc; FS[nfa] = orn
                        FVm[nfa, 0], FVr[nfa, 0] = face_at_point(c, f, xp[0], xp[1], xp[2], RM, RL, RR2)
                        FVm[nfa, 1], FVr[nfa, 1] = face_at_point(c, b, xp[0], xp[1], xp[2], RM, RL, RR2)
                        code = np.int64(((xc // (L * L)) << 32) | (((xc // L) % L) << 16) | (xc % L))
                        fa1 = (FC, FI, FJ, FX, FS, FVm, FVr, nfa + 1); ao1 = (AOc, AOi, AOs, nao)
                        if eval_one(T1, code, np.int8(cls_flat[xc]), C, AC, AK, AM, WC, WK, WM, geo, win, fa1, ao1, polyv,
                                    scr, stats, wit):
                            stats[ST_CLIP_OUT_OK] += 1
                        else:
                            done = False
                            npc = _edge_cut_pieces(PL, npl, T1, PC0, NPC, edges, geo, shell_eps, P, stats)
                            if npc >= 0:
                                if True:
                                    done = True
                                    for u in range(npc):
                                        qq = PC0 + u
                                        if not eval_one(qq, code, np.int8(cls_flat[xc]), C, AC, AK, AM, WC, WK, WM, geo, win,
                                                        fa1, ao1, polyv, scr, stats, wit):
                                            stats[ST_APEX_FAIL] += 1
                                            nrec = _undet(PL, npl, qq, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF,
                                                          rec_cell, rec_lo, rec_hi, rec_kind, nrec, C, K, RM, RR, FB, alo,
                                                          ahi, shell_eps, stats, pool, REC_UNDET)
                            if done:
                                stats[ST_CLIP_OUT_OK] += 1
                            else:
                                stats[ST_CLIP_OUT_FAIL] += 1
                                nrec = _undet(PL, npl, T1, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell,
                                              rec_lo, rec_hi, rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats,
                                              pool, REC_UNDET)
                    # kept side: outer polytope {sigma g~ - E <= 0} = complement of rO; inner plane rK; fact sign -orn
                    r3 = clip(PL, s, rO ^ 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    if r3 == OVERFLOW:
                        stats[ST_OVERFLOW] += 1
                    SPL[nsp] = rK; nsp += 1
                    FC[nfa] = c; FI[nfa] = f; FJ[nfa] = b; FX[nfa] = xc; FS[nfa] = -orn
                    FVm[nfa, 0], FVr[nfa, 0] = face_at_point(c, f, xp[0], xp[1], xp[2], RM, RL, RR2)
                    FVm[nfa, 1], FVr[nfa, 1] = face_at_point(c, b, xp[0], xp[1], xp[2], RM, RL, RR2)
                    nfa += 1
                    if nF[s] == 0:
                        dead = True
            pos += 1
        fr_nsp[s] = nsp; fr_nfa[s] = nfa
        fa = (FC, FI, FJ, FX, FS, FVm, FVr, nfa); ao = (AOc, AOi, AOs, nao)
        if pos >= ev.shape[0]:
            return pos, nrec, 1
        t = ev[pos, 0]
        leaf = True
        if dead:
            pos = _skip(ev, pos)
            stats[ST_EMPTY_EXACT] += 1
        elif t == EV_OUT or t == EV_IN or t == EV_EMPTY or t == EV_UNDET:
            claim_in = t == EV_IN
            wkind = -1; wcode = np.int64(0); wcls = 0
            if t == EV_OUT:
                reason = ev[pos, 1]
                if reason == OUT_ITEM:
                    wcode = ev[pos, 2]; wcls = ev[pos, 3]
                    lvl = (wcode >> 48) & 0xFFFF
                    if lvl == 0:
                        xc = (((wcode >> 32) & 0xFFFF) * L + ((wcode >> 16) & 0xFFFF)) * L + (wcode & 0xFFFF)
                        good = cls_flat[xc] == wcls
                    else:
                        good = _block_class_ok(wcode, wcls, cls_flat, L)
                    if good:
                        wkind = 0
                elif reason == OUT_WINDOW:
                    wkind = 1
                elif reason == OUT_PENCIL:
                    xc = _vox_point(ev[pos, 5], L, o, N, xp)
                    wcode = np.int64(((xc // (L * L)) << 32) | (((xc // L) % L) << 16) | (xc % L)); wcls = cls_flat[xc]
                    wkind = 0
            r = 2
            if t != EV_UNDET:
                r = _check_claim(PL, s, claim_in, wkind, wcode, wcls, wid, fr_off[s], fr_len[s], atop, AC, AK, AM, WC, WK,
                                 WM, geo, win, fa, ao, polyv, Vp, scr, stats, wit)
                if r == 3:
                    return pos, nrec, 3
            if r == 0 and claim_in:
                stats[ST_IN_OK] += 1
                nrec = _in_volumes(PL, npl, s, CUR, SPL, nsp, EM0, NEM, gid, P, rec_cell, rec_lo, rec_hi, rec_kind, nrec,
                                   geo, shell_eps, stats, pool, dbgINV, dbgIN)
            elif r == 0 or r == 1:
                if r == 1:
                    stats[ST_SELF_OUT] += 1
                else:
                    stats[ST_OUT_OK] += 1
            else:
                done = False
                if t != EV_UNDET:
                    # clearance-shell repair: cut cubes around vertices certified within 2^-26 of the solid and
                    # prove the claim on the remaining convex pieces
                    npc = _edge_cut_pieces(PL, npl, s, PC0, NPC, edges, geo, shell_eps, P, stats)
                    if npc >= 0:
                        if True:
                            done = True
                            for u in range(npc):
                                qq = PC0 + u
                                r2 = _check_claim(PL, qq, claim_in, wkind, wcode, wcls, wid, fr_off[s], fr_len[s], atop, AC,
                                                  AK, AM, WC, WK, WM, geo, win, fa, ao, polyv, Vp, scr, stats, wit)
                                if r2 == 3:
                                    return pos, nrec, 3
                                if r2 == 0 and claim_in:
                                    nrec = _in_volumes(PL, npl, qq, CUR, SPL, nsp, EM0, NEM, gid, P, rec_cell, rec_lo,
                                                       rec_hi, rec_kind, nrec, geo, shell_eps, stats, pool, dbgINV, dbgIN)
                                elif r2 == 2:
                                    nrec, rc = _refine_claim(PL, npl, qq, RF0, NRF, RBUDGET, RDMAX, claim_in, wkind, wcode,
                                                             wcls, wid, fr_off[s], fr_len[s], atop, AC, AK, AM, WC, WK, WM,
                                                             geo, win, fa, ao, P, scr, stats, wit, CUR, SPL, nsp, EM0, NEM,
                                                             gid, rec_cell, rec_lo, rec_hi, rec_kind, nrec, shell_eps, pool, dbg)
                                    if rc == 3:
                                        return pos, nrec, 3
                    if not done and RBUDGET > 0:
                        nrec, rc = _refine_claim(PL, npl, s, RF0, NRF, RBUDGET, RDMAX, claim_in, wkind, wcode, wcls, wid,
                                                 fr_off[s], fr_len[s], atop, AC, AK, AM, WC, WK, WM, geo, win, fa, ao, P,
                                                 scr, stats, wit, CUR, SPL, nsp, EM0, NEM, gid, rec_cell, rec_lo, rec_hi,
                                                 rec_kind, nrec, shell_eps, pool, dbg)
                        if rc == 3:
                            return pos, nrec, 3
                        done = True
                if not done:
                    if claim_in:
                        stats[ST_IN_FAIL] += 1
                    elif t == EV_OUT:
                        stats[ST_OUT_FAIL] += 1
                        dbgV, dbgI, dbgF, dbgA, _d4, _d5 = dbg
                        k_ = dbgI[0]
                        if k_ < dbgV.shape[0]:
                            nvv = min(nV[s], dbgV.shape[1])
                            for v in range(nvv):
                                for kk in range(3):
                                    dbgV[k_, v, kk] = Vm[s, v, kk]
                            dbgI[1 + 4 * k_] = pos; dbgI[2 + 4 * k_] = nvv; dbgI[3 + 4 * k_] = nfa; dbgI[4 + 4 * k_] = nao
                            for u in range(min(nfa, dbgF.shape[1])):
                                dbgF[k_, u, 0] = FC[u]; dbgF[k_, u, 1] = FI[u]; dbgF[k_, u, 2] = FJ[u]; dbgF[k_, u, 3] = FX[u]; dbgF[k_, u, 4] = FS[u]
                            for u in range(min(nao, dbgA.shape[1])):
                                dbgA[k_, u, 0] = AOc[u]; dbgA[k_, u, 1] = AOi[u]; dbgA[k_, u, 2] = AOs[u]
                            dbgI[0] = k_ + 1
                    elif t == EV_UNDET:
                        stats[ST_UNDET] += 1
                    else:
                        stats[ST_EMPTY_SLIVER] += 1
                    nrec = _undet(PL, npl, s, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                                  rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
            pos += 1
        elif t == EV_SPLIT_AXIS or t == EV_SPLIT_FACE or t == EV_SPLIT_PENCIL:
            st, nout = eval_node(s, 0, AC, AK, AM, fr_off[s], fr_len[s], atop, WC, WK, WM, geo, win, fa, ao, polyv,
                                 scr, stats, wit)
            if st == 3:
                return pos, nrec, 3
            if st == 1 or st == 4:
                stats[ST_SELF_OUT] += 1
                pos = _skip(ev, pos)
            elif s + 1 > SMAX or nsp + 1 >= NSP or nfa + 1 >= NFA or nao + 1 >= NAO:
                stats[ST_OVERFLOW] += 1
                nrec = _undet(PL, npl, s, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                              rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
                pos = _skip(ev, pos)
            else:
                oA = -1; oB = -1; iA = -1; iB = -1
                fA = False; fcol = np.zeros(5, np.int64); aoF = False; aocol = np.zeros(3, np.int64)
                fvv = np.zeros(4)
                if t == EV_SPLIT_AXIS:
                    pid = ev[pos, 2]
                    oA = 2 * pid; oB = 2 * pid + 1
                elif t == EV_SPLIT_FACE:
                    c = ev[pos, 1]; i = ev[pos, 2]
                    for q in range(4):
                        gm[q] = RM[c, i, q]; gr[q] = RR[c, i, q]
                    rA, rB = _shift_planes(PL, npl, gm, gr, 1.0, 0.0)
                    if rA >= 0:
                        oA = rB ^ 1; oB = rA ^ 1; iA = rA; iB = rB
                        aoF = True; aocol[0] = c; aocol[1] = i      # A: back (-1), B: front (+1)
                else:
                    c = ev[pos, 1]; f = ev[pos, 2]; b = ev[pos, 3]; orn = ev[pos, 5]
                    xc = _vox_point(ev[pos, 4], L, o, N, xp)
                    _pencil_coef(c, f, b, xp[0], xp[1], xp[2], RM, RR, RL, RR2, gm, gr)
                    rA, rB = _shift_planes(PL, npl, gm, gr, float(orn), 0.0)
                    if rA >= 0:
                        oA = rB ^ 1; oB = rA ^ 1; iA = rA; iB = rB
                        fA = True; fcol[0] = c; fcol[1] = f; fcol[2] = b; fcol[3] = xc; fcol[4] = orn
                        fvv[0], fvv[1] = face_at_point(c, f, xp[0], xp[1], xp[2], RM, RL, RR2)
                        fvv[2], fvv[3] = face_at_point(c, b, xp[0], xp[1], xp[2], RM, RL, RR2)
                if oA < 0:
                    stats[ST_OVERFLOW] += 1
                    nrec = _undet(PL, npl, s, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo, rec_hi,
                                  rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
                    pos = _skip(ev, pos)
                else:
                    copy_slot(s, s + 1, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    cB = clip(PL, s + 1, oB, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    cA = clip(PL, s, oA, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF)
                    if cA == OVERFLOW or cB == OVERFLOW:
                        stats[ST_OVERFLOW] += 1
                        nrec = _undet(PL, npl, s, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo,
                                      rec_hi, rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
                        nrec = _undet(PL, npl, s + 1, EM0, NEM, gid, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF, rec_cell, rec_lo,
                                      rec_hi, rec_kind, nrec, C, K, RM, RR, FB, alo, ahi, shell_eps, stats, pool, REC_UNDET)
                        pos = _skip(ev, pos)
                    else:
                        fr_off[s] = atop; fr_len[s] = nout
                        atop += nout
                        fr_off[s + 1] = fr_off[s]; fr_len[s + 1] = nout
                        fr_bsp[s] = nsp; fr_bfa[s] = nfa; fr_bao[s] = nao
                        # B child (processed first): inner plane iB, fact sign +orn, aspect +1
                        n1 = nsp; n2 = nfa; n3 = nao
                        if iB >= 0:
                            SPL[n1] = iB; n1 += 1
                        if fA:
                            FC[n2] = fcol[0]; FI[n2] = fcol[1]; FJ[n2] = fcol[2]; FX[n2] = fcol[3]; FS[n2] = fcol[4]
                            FVm[n2, 0] = fvv[0]; FVr[n2, 0] = fvv[1]; FVm[n2, 1] = fvv[2]; FVr[n2, 1] = fvv[3]
                            n2 += 1
                        if aoF:
                            AOc[n3] = aocol[0]; AOi[n3] = aocol[1]; AOs[n3] = 1; n3 += 1
                        fr_nsp[s + 1] = n1; fr_nfa[s + 1] = n2; fr_nao[s + 1] = n3
                        # A child (pending): inner plane iA, fact sign -orn, aspect -1
                        sA_sp[s] = iA
                        sA_hasfa[s] = fA
                        if fA:
                            sA_fa[s, 0] = fcol[0]; sA_fa[s, 1] = fcol[1]; sA_fa[s, 2] = fcol[2]; sA_fa[s, 3] = fcol[3]
                            sA_fa[s, 4] = -fcol[4]
                            for q in range(4):
                                sA_fv[s, q] = fvv[q]
                        sA_hasao[s] = aoF
                        if aoF:
                            sA_ao[s, 0] = aocol[0]; sA_ao[s, 1] = aocol[1]; sA_ao[s, 2] = -1
                        pending[s] = True
                        sp = s + 1
                        pos += 1
                        leaf = False
        else:
            return pos, nrec, 1
        if leaf:
            q = sp - 1
            while q >= 0 and not pending[q]:
                q -= 1
            if q < 0:
                break
            pending[q] = False
            sp = q
            atop = fr_off[q] + fr_len[q]
            n1 = fr_bsp[q]; n2 = fr_bfa[q]; n3 = fr_bao[q]
            if sA_sp[q] >= 0:
                SPL[n1] = sA_sp[q]; n1 += 1
            if sA_hasfa[q]:
                FC[n2] = sA_fa[q, 0]; FI[n2] = sA_fa[q, 1]; FJ[n2] = sA_fa[q, 2]; FX[n2] = sA_fa[q, 3]; FS[n2] = sA_fa[q, 4]
                FVm[n2, 0] = sA_fv[q, 0]; FVr[n2, 0] = sA_fv[q, 1]; FVm[n2, 1] = sA_fv[q, 2]; FVr[n2, 1] = sA_fv[q, 3]
                n2 += 1
            if sA_hasao[q]:
                AOc[n3] = sA_ao[q, 0]; AOi[n3] = sA_ao[q, 1]; AOs[n3] = sA_ao[q, 2]; n3 += 1
            fr_nsp[q] = n1; fr_nfa[q] = n2; fr_nao[q] = n3
    return pos, nrec, 0


@njit(cache=True)
def _block_class_ok(code, kls, cls_flat, L):
    """Every voxel of the block has the observed class kls (pure block)."""
    l = (code >> 48) & 0xFFFF
    s = np.int64(1) << l
    bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
    for x in range(bx * s, (bx + 1) * s):
        for y in range(by * s, (by + 1) * s):
            for z in range(bz * s, (bz + 1) * s):
                if cls_flat[(x * L + y) * L + z] != kls:
                    return False
    return True
