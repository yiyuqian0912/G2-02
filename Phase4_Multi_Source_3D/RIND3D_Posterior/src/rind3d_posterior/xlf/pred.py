"""XLF trusted predicates (numba): rigorous response classification of a source polytope against window receivers.

Physics (docs/RIND3D/SEMANTICS.md, closed-solid-closed-segment-v1): receiver r is DARK for source s iff the closed
segment [s, r] meets the closed solid S = U_c K_c, K_c = {x : f_i(x) <= 0}. Rows f_i are the canonical rationals of
certify.scene, enclosed componentwise as RM +- RR.

Segment lemma (certify/check.py). For an apex p outside K_c with every f_i(p) strictly signed, Phi(p) = {i : f_i(p) > 0},
    [p, x] meets K_c  <=>  f_i(x) <= 0 (i in Phi(p))  and  B_ij(p, x) = f_i(p) f_j(x) - f_j(p) f_i(x) <= 0
                           for the SILHOUETTE pairs (i in Phi(p), j not in Phi(p), facets i, j adjacent);
every pair (i in Phi, j not in Phi) is a necessary condition. As a function of x this is the shadow volume
C_c(p) = {x : [p, x] meets K_c}: a convex polyhedron, and x in C_c(p) <=> p in C_c(x) (segments are symmetric).

Region lemmas, for a convex source polytope Q with vertex set V (each vertex enclosed; never rounded or snapped):
  (D)  receiver r is dark for every s in Q via c  <=>  V subset of C_c(r)  <=>  r in I_c(Q) = n_{v in V} C_c(v).
  (L)  if some plane g with g <= 0 on K_c has g > 0 on V and at r, no segment from Q to r meets K_c.  Used with
       g = a constraint of C_c(r) (source form), a facet f_i, or g = lam f_i + mu f_j (lam, mu >= 0) for an edge (i, j)
       (receiver form, collective).
  (S)  if exactly the constraints A of C_c(r) straddle Q and all others are <= 0 on V, then C_c(r) n Q = {g <= 0, g in A} n Q.
Every value is an enclosure (mid, rad) computed with a priori binary64 error bounds (certify/ar.py); a sign is used only
when the enclosure excludes zero. Receivers are exact dyadic points (o + t + 1/2)/N."""
import numpy as np
from numba import njit

from ..certify.ar import INFL, TINY, G2, G3, G4, dn, up

DARK, LIT, SOLID = 0, 1, 2            # observation classes of the window (cls)
R_DARK, R_LIT, R_CONV, R_NONCONV = 0, 1, 2, 3     # receiver status on a region (dark-sure, lit-sure, ambiguous kinds)
MA = 8                                # active constraints kept per component
MH = 16                               # halfspaces kept per receiver


# ------------------------------------------------------------------------------------------- rigorous evaluation
@njit(cache=True, inline="always")
def aff(cm, cr, x0, x1, x2, r0, r1, r2):
    """(mid, rad): affine function with coefficients cm +- cr at the point (x0, x1, x2) +- (r0, r1, r2)."""
    p0 = cm[0] * x0; p1 = cm[1] * x1; p2 = cm[2] * x2
    m = p0 + p1 + p2 + cm[3]
    rad = (abs(cm[0]) * r0 + abs(cm[1]) * r1 + abs(cm[2]) * r2
           + cr[0] * (abs(x0) + r0) + cr[1] * (abs(x1) + r1) + cr[2] * (abs(x2) + r2) + cr[3]
           + G4 * (abs(p0) + abs(p1) + abs(p2) + abs(cm[3]))) * INFL + TINY
    return m, rad


@njit(cache=True, inline="always")
def pair_coef(Fi, ri, Fj, rj, qim, qir, qjm, qjr, om, orr):
    """Coefficients (mid, rad) of x -> Fi f_j(x) - Fj f_i(x) for apex values Fi +- ri, Fj +- rj."""
    for t in range(4):
        a = Fi * qjm[t]; b = Fj * qim[t]
        om[t] = a - b
        orr[t] = (abs(Fi) * qjr[t] + ri * (abs(qjm[t]) + qjr[t]) + abs(Fj) * qir[t] + rj * (abs(qim[t]) + qir[t])
                  + G2 * (abs(a) + abs(b))) * INFL + TINY


@njit(cache=True, inline="always")
def comb_coef(lam, mu, qim, qir, qjm, qjr, om, orr):
    """Coefficients of lam f_i + mu f_j (lam, mu >= 0 binary64)."""
    for t in range(4):
        a = lam * qim[t]; b = mu * qjm[t]
        om[t] = a + b
        orr[t] = (lam * qir[t] + mu * qjr[t] + G2 * (abs(a) + abs(b))) * INFL + TINY


@njit(cache=True)
def range_on(cm, cr, Vm, Vr, nv):
    """Rigorous [inf, sup] of the affine function over the polytope with vertex enclosures Vm +- Vr."""
    lo = np.inf; hi = -np.inf
    for v in range(nv):
        m, r = aff(cm, cr, Vm[v, 0], Vm[v, 1], Vm[v, 2], Vr[v, 0], Vr[v, 1], Vr[v, 2])
        a = dn(m - r); b = up(m + r)
        if a < lo:
            lo = a
        if b > hi:
            hi = b
    return lo, hi


# ------------------------------------------------------------------------------------------- relevance
@njit(cache=True)
def relevant_comps(Vm, Vr, nv, wlo, whi, AL, AH, C, out):
    """Components whose bounding box meets the bounding box of Q u window (others meet no segment). Returns count."""
    lo = np.empty(3); hi = np.empty(3)
    for k in range(3):
        lo[k] = wlo[k]; hi[k] = whi[k]
    for v in range(nv):
        for k in range(3):
            a = dn(Vm[v, k] - Vr[v, k]); b = up(Vm[v, k] + Vr[v, k])
            if a < lo[k]:
                lo[k] = a
            if b > hi[k]:
                hi[k] = b
    n = 0
    for c in range(C):
        ok = True
        for k in range(3):
            if AL[c, k] > hi[k] or AH[c, k] < lo[k]:
                ok = False
        if ok:
            out[n] = c; n += 1
    return n


# ------------------------------------------------------------------------------------------- per-receiver test
@njit(cache=True)
def recv_test(Vm, Vr, nv, rx, ry, rz, obs, comps, ncomp, RM, RR, K, ADJ, hs_m, hs_r, hs_key, hs_sense):
    """Status of receiver r = (rx, ry, rz) (exact) on the polytope Q (source form, lemmas D/L/S per component).
    Returns (status, nh, witness_comp). status: R_DARK (dark for all s in Q), R_LIT (lit for all s in Q), R_CONV
    (ambiguous; feasibility of s in Q for this receiver's observation is the intersection of the nh halfspaces
    hs_sense[h] * g_h(s) <= 0 (sense +1: g <= 0; sense -1: g > 0, strict), each g_h = hs_m[h] +- hs_r[h], keyed
    hs_key[h] = (c, i, j, kind) kind 0 gate f_i, 1 pair B_ij(r, .)), R_NONCONV (ambiguous, not convex: hs_* lists the
    straddling constraints, sense 0). obs is the observed class (DARK/LIT), used only to build the halfspaces."""
    fm = np.empty(64); fr = np.empty(64); sg = np.empty(64, np.int64)
    gm = np.empty(4); gr = np.empty(4)
    nonlit = 0; nh = 0; nonconv = False; wit = -1
    for q in range(ncomp):
        c = comps[q]; Kc = K[c]
        unsure = False; nphi = 0
        for i in range(Kc):
            m, r = aff(RM[c, i], RR[c, i], rx, ry, rz, 0.0, 0.0, 0.0)
            fm[i] = m; fr[i] = r
            if dn(m - r) > 0.0:
                sg[i] = 1; nphi += 1
            elif up(m + r) < 0.0:
                sg[i] = -1
            else:
                sg[i] = 0; unsure = True
        if unsure:
            nonlit += 1; nonconv = True; wit = c
            continue
        if nphi == 0:
            return R_DARK, 0, c                 # receiver inside the closed component: every segment meets it
        # constraints of C_c(r): gates then silhouette pairs; classify each on V
        lit = False; nact = 0
        am = np.empty((MA, 4)); arr = np.empty((MA, 4)); ak = np.empty((MA, 4), np.int64)
        overflow = False
        for i in range(Kc):
            if sg[i] != 1:
                continue
            lo, hi = range_on(RM[c, i], RR[c, i], Vm, Vr, nv)
            if lo > 0.0:
                lit = True
                break
            if hi > 0.0:
                if nact < MA:
                    for t in range(4):
                        am[nact, t] = RM[c, i, t]; arr[nact, t] = RR[c, i, t]
                    ak[nact, 0] = c; ak[nact, 1] = i; ak[nact, 2] = -1; ak[nact, 3] = 0
                    nact += 1
                else:
                    overflow = True
        if not lit:
            for i in range(Kc):
                if sg[i] != 1:
                    continue
                for j in range(Kc):
                    if sg[j] != -1 or not ADJ[c, i, j]:
                        continue
                    pair_coef(fm[i], fr[i], fm[j], fr[j], RM[c, i], RR[c, i], RM[c, j], RR[c, j], gm, gr)
                    lo, hi = range_on(gm, gr, Vm, Vr, nv)
                    if lo > 0.0:
                        lit = True
                        break
                    if hi > 0.0:
                        if nact < MA:
                            for t in range(4):
                                am[nact, t] = gm[t]; arr[nact, t] = gr[t]
                            ak[nact, 0] = c; ak[nact, 1] = i; ak[nact, 2] = j; ak[nact, 3] = 1
                            nact += 1
                        else:
                            overflow = True
                if lit:
                    break
        if lit:
            continue                            # lemma L: no segment from Q to r meets K_c
        if nact == 0 and not overflow:
            return R_DARK, 0, c                 # lemma D: Q subset of C_c(r)
        nonlit += 1; wit = c
        if overflow:
            nonconv = True
        # store: dark observation needs all of A (one component only); lit observation needs one violated, so it is
        # convex only with a single straddling constraint per component
        if obs == DARK:
            for a in range(nact):
                if nh < MH:
                    for t in range(4):
                        hs_m[nh, t] = am[a, t]; hs_r[nh, t] = arr[a, t]
                    for t in range(4):
                        hs_key[nh, t] = ak[a, t]
                    hs_sense[nh] = 1; nh += 1
                else:
                    nonconv = True
        else:
            if nact != 1:
                nonconv = True
            for a in range(nact):
                if nh < MH:
                    for t in range(4):
                        hs_m[nh, t] = am[a, t]; hs_r[nh, t] = arr[a, t]
                    for t in range(4):
                        hs_key[nh, t] = ak[a, t]
                    hs_sense[nh] = -1; nh += 1
                else:
                    nonconv = True
    if nonlit == 0:
        return R_LIT, 0, -1
    if obs == DARK and nonlit > 1:
        nonconv = True
    if nonconv:
        for h in range(nh):
            hs_sense[h] = 0
        return R_NONCONV, nh, wit
    return R_CONV, nh, wit


# ------------------------------------------------------------------------------------------- collective rows
@njit(cache=True)
def _prep_comp(c, Vm, Vr, nv, RM, RR, K, ADJ, dm, dr, rm, rr):
    """Receiver-space data of component c for the polytope Q: dark constraints (r in I_c(Q) <=> all <= 0) and lit
    planes (r surely lit via c where any is > 0). Returns (n_dark (-1 if I_c is not available), n_lit)."""
    Kc = K[c]
    fm = np.empty((nv, Kc)); fr = np.empty((nv, Kc)); sg = np.empty((nv, Kc), np.int64)
    unsure = False
    for v in range(nv):
        for i in range(Kc):
            m, r = aff(RM[c, i], RR[c, i], Vm[v, 0], Vm[v, 1], Vm[v, 2], Vr[v, 0], Vr[v, 1], Vr[v, 2])
            fm[v, i] = m; fr[v, i] = r
            if dn(m - r) > 0.0:
                sg[v, i] = 1
            elif up(m + r) < 0.0:
                sg[v, i] = -1
            else:
                sg[v, i] = 0; unsure = True
    gm = np.empty(4); gr = np.empty(4)
    # ---- lit planes: facets with f_i > 0 on V; for every edge (i, j) the two extreme planes lam f_i + mu f_j > 0 on V
    nl = 0
    for i in range(Kc):
        ok = True
        for v in range(nv):
            if dn(fm[v, i] - fr[v, i]) <= 0.0:
                ok = False
                break
        if ok and nl < rm.shape[0]:
            for t in range(4):
                rm[nl, t] = RM[c, i, t]; rr[nl, t] = RR[c, i, t]
            nl += 1
    for i in range(Kc):
        for j in range(i + 1, Kc):
            if not ADJ[c, i, j]:
                continue
            # theta in [0, pi/2] with cos(theta) f_i(v) + sin(theta) f_j(v) > 0 for all v (float estimate)
            tlo = 0.0; thi = np.pi / 2; feas = True
            for v in range(nv):
                a = fm[v, i]; b = fm[v, j]
                if a > 0.0 and b > 0.0:
                    continue
                if a <= 0.0 and b <= 0.0:
                    feas = False
                    break
                if a > 0.0:                    # b <= 0: theta < atan2(a, -b)
                    u = np.arctan2(a, -b)
                    if u < thi:
                        thi = u
                else:                          # a <= 0 < b: theta > atan2(-a, b)
                    u = np.arctan2(-a, b)
                    if u > tlo:
                        tlo = u
            if not feas or thi - tlo <= 1e-12:
                continue
            # the two extreme planes of the pencil (through e, tangent to Q), moved inside by a tiny angle; by the
            # sinusoid argument {r : g_theta(r) > 0 for some theta in [t1, t2]} = {g_t1 > 0} u {g_t2 > 0}
            w = thi - tlo
            for which in range(2):
                for shrink in (1e-10, 1e-7, 1e-4, 2e-2):
                    th = tlo + (shrink * w + 1e-14) if which == 0 else thi - (shrink * w + 1e-14)
                    lam = np.cos(th); mu = np.sin(th)
                    if lam < 0.0:
                        lam = 0.0
                    if mu < 0.0:
                        mu = 0.0
                    comb_coef(lam, mu, RM[c, i], RR[c, i], RM[c, j], RR[c, j], gm, gr)
                    lo, hi = range_on(gm, gr, Vm, Vr, nv)
                    if lo > 0.0:                           # verified: plane > 0 on V (and <= 0 on K_c: lam, mu >= 0)
                        if nl < rm.shape[0]:
                            for t in range(4):
                                rm[nl, t] = gm[t]; rr[nl, t] = gr[t]
                            nl += 1
                        break
    if unsure:
        return -1, nl
    # ---- dark constraints: gates over U Phi(v); pairs B_ij(v, .) for every v with f_i(v) > 0 > f_j(v), (i, j) adjacent
    nd = 0
    for i in range(Kc):
        anyf = False
        for v in range(nv):
            if sg[v, i] == 1:
                anyf = True
                break
        if anyf:
            if nd >= dm.shape[0]:
                return -1, nl
            for t in range(4):
                dm[nd, t] = RM[c, i, t]; dr[nd, t] = RR[c, i, t]
            nd += 1
    for i in range(Kc):
        for j in range(Kc):
            if not ADJ[c, i, j]:
                continue
            for v in range(nv):
                if sg[v, i] == 1 and sg[v, j] == -1:
                    if nd >= dm.shape[0]:
                        return -1, nl
                    pair_coef(fm[v, i], fr[v, i], fm[v, j], fr[v, j], RM[c, i], RR[c, i], RM[c, j], RR[c, j], gm, gr)
                    for t in range(4):
                        dm[nd, t] = gm[t]; dr[nd, t] = gr[t]
                    nd += 1
    return nd, nl


@njit(cache=True, inline="always")
def _ok_le(ah, bh, oz, t, N):
    z = (oz + t + 0.5) / N
    return up(ah + up(bh * z)) <= 0.0


@njit(cache=True, inline="always")
def _ok_gt(al, bl, oz, t, N):
    z = (oz + t + 0.5) / N
    return dn(al + dn(bl * z)) > 0.0


@njit(cache=True, inline="always")
def _tstar(a, b, oz, N, L):
    """Approximate real t at which a + b z_t = 0 (clamped to [-2, L + 1])."""
    x = (-a / b) * N - oz - 0.5
    if not (x == x):
        return 0.0
    if x < -2.0:
        return -2.0
    if x > L + 1.0:
        return L + 1.0
    return x


@njit(cache=True)
def _le_range(cm, cr, x, y, oz, N, L):
    """t-range [lo, hi] of row (x, y) where the affine function is surely <= 0 (empty: lo > hi)."""
    p0 = cm[0] * x; p1 = cm[1] * y
    m = p0 + p1 + cm[3]
    r = (cr[0] * x + cr[1] * y + cr[3] + G3 * (abs(p0) + abs(p1) + abs(cm[3]))) * INFL + TINY
    ah = up(m + r); bh = up(cm[2] + cr[2])
    if bh > 0.0:
        T = int(np.floor(_tstar(ah, bh, oz, N, L)))
        if T > L - 1:
            T = L - 1
        if T < -1:
            T = -1
        while T >= 0 and not _ok_le(ah, bh, oz, T, N):
            T -= 1
        while T + 1 <= L - 1 and _ok_le(ah, bh, oz, T + 1, N):
            T += 1
        return 0, T
    if bh < 0.0:
        T = int(np.ceil(_tstar(ah, bh, oz, N, L)))
        if T < 0:
            T = 0
        if T > L:
            T = L
        while T <= L - 1 and not _ok_le(ah, bh, oz, T, N):
            T += 1
        while T - 1 >= 0 and _ok_le(ah, bh, oz, T - 1, N):
            T -= 1
        return T, L - 1
    if _ok_le(ah, bh, oz, 0, N):
        return 0, L - 1
    return 1, 0


@njit(cache=True)
def _gt_ray(cm, cr, x, y, oz, N, L):
    """Where the affine function is surely > 0 on row (x, y): returns (kind, T): kind 0 none, 1 all, 2 [T, L-1],
    3 [0, T]."""
    p0 = cm[0] * x; p1 = cm[1] * y
    m = p0 + p1 + cm[3]
    r = (cr[0] * x + cr[1] * y + cr[3] + G3 * (abs(p0) + abs(p1) + abs(cm[3]))) * INFL + TINY
    al = dn(m - r); bl = dn(cm[2] - cr[2])
    if bl > 0.0:
        T = int(np.ceil(_tstar(al, bl, oz, N, L)))
        if T < 0:
            T = 0
        if T > L:
            T = L
        while T <= L - 1 and not _ok_gt(al, bl, oz, T, N):
            T += 1
        while T - 1 >= 0 and _ok_gt(al, bl, oz, T - 1, N):
            T -= 1
        return (2, T) if T <= L - 1 else (0, 0)
    if bl < 0.0:
        T = int(np.floor(_tstar(al, bl, oz, N, L)))
        if T > L - 1:
            T = L - 1
        if T < -1:
            T = -1
        while T >= 0 and not _ok_gt(al, bl, oz, T, N):
            T -= 1
        while T + 1 <= L - 1 and _ok_gt(al, bl, oz, T + 1, N):
            T += 1
        return (3, T) if T >= 0 else (0, 0)
    if _ok_gt(al, bl, oz, 0, N):
        return 1, 0
    return 0, 0


SC_OK, SC_OUT, SC_BUG = 0, 1, 2


@njit(cache=True)
def scan_rows(Vm, Vr, nv, rows, nrows, cls, o, N, L, comps, ncomp, RM, RR, K, ADJ, amb, cap, rowflag, wit, rcap):
    """Collective classification of every FREE receiver on the given rows (rows[k] = (X, Y); receivers (X, Y, t),
    t = 0..L-1) for the polytope Q (receiver form: lemma D via I_c(Q), lemma L via verified planes).
    Returns (code, n_amb): code SC_OUT if some receiver's observation is contradicted for every s in Q (wit = X, Y, t,
    kind 0 dark-observed but surely lit / 1 lit-observed but surely dark), SC_OK otherwise (amb[:min(n_amb, cap)] =
    ambiguous receivers (X, Y, t, row k), rowflag[k] = 1 for rows holding one); SC_BUG if sure-dark and sure-lit
    coincide (impossible with correct arithmetic; reported, never used). If at most rcap receivers are ambiguous for
    the collective test, each is re-tested exactly in the source form (recv_test; lemma L there uses the plane through
    the receiver itself, which the receiver form can only approach), and only those still ambiguous are returned."""
    KD = 4096; KL = 512
    nd = np.empty(ncomp, np.int64); nl = np.empty(ncomp, np.int64)
    DM = np.empty((ncomp, KD, 4)); DR = np.empty((ncomp, KD, 4)); LM = np.empty((ncomp, KL, 4)); LR = np.empty((ncomp, KL, 4))
    for q in range(ncomp):
        a, b = _prep_comp(comps[q], Vm, Vr, nv, RM, RR, K, ADJ, DM[q], DR[q], LM[q], LR[q])
        nd[q] = a; nl[q] = b
    ox = o[0]; oy = o[1]; oz = o[2]
    A = np.empty(ncomp, np.int64); B = np.empty(ncomp, np.int64); P = np.empty(ncomp, np.int64); Qh = np.empty(ncomp, np.int64)
    namb = 0
    for k in range(nrows):
        X = rows[k, 0]; Y = rows[k, 1]
        x = (ox + X + 0.5) / N; y = (oy + Y + 0.5) / N
        rowflag[k] = 0
        for q in range(ncomp):
            a = 0; b = L - 1
            if nd[q] < 0:
                a = 1; b = 0
            else:
                for h in range(nd[q]):
                    lo, hi = _le_range(DM[q, h], DR[q, h], x, y, oz, N, L)
                    if lo > a:
                        a = lo
                    if hi < b:
                        b = hi
                    if a > b:
                        break
            A[q] = a; B[q] = b
            p = 0; qq = L - 1
            for h in range(nl[q]):
                kind, T = _gt_ray(LM[q, h], LR[q, h], x, y, oz, N, L)
                if kind == 1:
                    p = 1; qq = 0
                    break
                if kind == 2:
                    if T - 1 < qq:
                        qq = T - 1
                elif kind == 3:
                    if T + 1 > p:
                        p = T + 1
                if p > qq:
                    break
            P[q] = p; Qh[q] = qq
        for t in range(L):
            ob = cls[X, Y, t]
            if ob == SOLID:
                continue
            ds = False; pd = False
            for q in range(ncomp):
                if A[q] <= t and t <= B[q]:
                    ds = True
                if P[q] <= t and t <= Qh[q]:
                    pd = True
            if ds and not pd:
                wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 9
                return SC_BUG, namb
            if ob == DARK:
                if ds:
                    continue
                if not pd:
                    wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 0
                    return SC_OUT, namb
            else:
                if not pd:
                    continue
                if ds:
                    wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 1
                    return SC_OUT, namb
            if namb < cap:
                amb[namb, 0] = X; amb[namb, 1] = Y; amb[namb, 2] = t; amb[namb, 3] = k
            namb += 1
            rowflag[k] = 1
    if namb == 0 or namb > rcap or namb > cap:
        return SC_OK, namb
    hs_m = np.empty((MH, 4)); hs_r = np.empty((MH, 4)); hs_key = np.empty((MH, 4), np.int64); hs_s = np.empty(MH, np.int64)
    for k in range(nrows):
        rowflag[k] = 0
    m = 0
    for a in range(namb):
        X = amb[a, 0]; Y = amb[a, 1]; t = amb[a, 2]
        ob = cls[X, Y, t]
        st, nh, w = recv_test(Vm, Vr, nv, (ox + X + 0.5) / N, (oy + Y + 0.5) / N, (oz + t + 0.5) / N, ob, comps, ncomp,
                              RM, RR, K, ADJ, hs_m, hs_r, hs_key, hs_s)
        if st == R_DARK:
            if ob == DARK:
                continue
            wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 3
            return SC_OUT, m
        if st == R_LIT:
            if ob == LIT:
                continue
            wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 2
            return SC_OUT, m
        for u in range(4):
            amb[m, u] = amb[a, u]
        rowflag[amb[a, 3]] = 1
        m += 1
    return SC_OK, m


@njit(cache=True)
def fine_collect(Vm, Vr, nv, amb, namb, cls, o, N, comps, ncomp, RM, RR, K, ADJ, HM, HR, HK, HS, HPHI, HREC, cap, wit):
    """Source-form test of every ambiguous receiver; collects the halfspaces of the convex ones (HM +- HR, key HK =
    (c, i, j, kind), sense HS, pencil angle HPHI = atan2(-f_j(r), f_i(r)) for pairs, receiver index HREC).
    Returns (code, n_halfspaces, n_convex, first_nonconvex): code 1 = contradiction (wit), 0 otherwise."""
    hs_m = np.empty((MH, 4)); hs_r = np.empty((MH, 4)); hs_key = np.empty((MH, 4), np.int64); hs_s = np.empty(MH, np.int64)
    n = 0; nconv = 0; first_nc = -1
    for a in range(namb):
        X = amb[a, 0]; Y = amb[a, 1]; t = amb[a, 2]
        ob = cls[X, Y, t]
        rx = (o[0] + X + 0.5) / N; ry = (o[1] + Y + 0.5) / N; rz = (o[2] + t + 0.5) / N
        st, nh, w = recv_test(Vm, Vr, nv, rx, ry, rz, ob, comps, ncomp, RM, RR, K, ADJ, hs_m, hs_r, hs_key, hs_s)
        if (st == R_DARK and ob == LIT) or (st == R_LIT and ob == DARK):
            wit[0] = X; wit[1] = Y; wit[2] = t; wit[3] = 2
            return 1, n, nconv, first_nc
        if st == R_NONCONV:
            if first_nc < 0:
                first_nc = a
            continue
        if st != R_CONV:
            continue
        nconv += 1
        for h in range(nh):
            if n >= cap:
                break
            for u in range(4):
                HM[n, u] = hs_m[h, u]; HR[n, u] = hs_r[h, u]; HK[n, u] = hs_key[h, u]
            HS[n] = hs_s[h]; HREC[n] = a
            if hs_key[h, 3] == 1:
                c = hs_key[h, 0]; i = hs_key[h, 1]; j = hs_key[h, 2]
                fi = RM[c, i, 0] * rx + RM[c, i, 1] * ry + RM[c, i, 2] * rz + RM[c, i, 3]
                fj = RM[c, j, 0] * rx + RM[c, j, 1] * ry + RM[c, j, 2] * rz + RM[c, j, 3]
                HPHI[n] = np.arctan2(-fj, fi)
            else:
                HPHI[n] = 0.0
            n += 1
    return 0, n, nconv, first_nc
