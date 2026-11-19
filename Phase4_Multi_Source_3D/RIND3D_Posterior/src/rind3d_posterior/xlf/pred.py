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
    # Only the angular extremes of w_v = (f_i(v), f_j(v)) over V_ij bind: B_ij(v, r) = cross(w_v, w_r) is linear in
    # w_v, and a w_v proven (interval cross products >= 0) inside the cone of the two extremes is their nonnegative
    # combination, so its constraint is implied. Unproven vertices keep their own constraint.
    vs = np.empty(nv, np.int64); keep = np.empty(nv, np.bool_)
    for i in range(Kc):
        for j in range(Kc):
            if not ADJ[c, i, j]:
                continue
            m = 0
            for v in range(nv):
                if sg[v, i] == 1 and sg[v, j] == -1:
                    vs[m] = v; m += 1
            if m == 0:
                continue
            for a in range(m):
                keep[a] = True
            if m > 2:
                amin = 0; amax = 0
                tmin = np.arctan2(fm[vs[0], j], fm[vs[0], i]); tmax = tmin
                for a in range(1, m):
                    th = np.arctan2(fm[vs[a], j], fm[vs[a], i])
                    if th < tmin:
                        tmin = th; amin = a
                    if th > tmax:
                        tmax = th; amax = a
                for a in range(m):
                    if a == amin or a == amax:
                        continue
                    if (_cross_lo(fm[vs[amin], i], fr[vs[amin], i], fm[vs[amin], j], fr[vs[amin], j],
                                  fm[vs[a], i], fr[vs[a], i], fm[vs[a], j], fr[vs[a], j]) >= 0.0 and
                            _cross_lo(fm[vs[a], i], fr[vs[a], i], fm[vs[a], j], fr[vs[a], j],
                                      fm[vs[amax], i], fr[vs[amax], i], fm[vs[amax], j], fr[vs[amax], j]) >= 0.0):
                        keep[a] = False
            for a in range(m):
                if not keep[a]:
                    continue
                v = vs[a]
                if nd >= dm.shape[0]:
                    return -1, nl
                pair_coef(fm[v, i], fr[v, i], fm[v, j], fr[v, j], RM[c, i], RR[c, i], RM[c, j], RR[c, j], gm, gr)
                for t in range(4):
                    dm[nd, t] = gm[t]; dr[nd, t] = gr[t]
                nd += 1
    return nd, nl


@njit(cache=True, inline="always")
def _cross_lo(a0, ra0, a1, ra1, b0, rb0, b1, rb1):
    """Rigorous lower bound of a0 b1 - a1 b0 for a = (a0, a1) +- (ra0, ra1), b likewise."""
    p = a0 * b1; q = a1 * b0
    r = (abs(a0) * rb1 + ra0 * (abs(b1) + rb1) + abs(a1) * rb0 + ra1 * (abs(b0) + rb0) + G2 * (abs(p) + abs(q))) * INFL + TINY
    return dn((p - q) - r)


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


SC_OK, SC_OUT, SC_BUG, SC_MANY = 0, 1, 2, 3


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
    nd, nl, DM, DR, LM, LR = prep_all(Vm, Vr, nv, comps, ncomp, RM, RR, K, ADJ)
    return scan_core(Vm, Vr, nv, rows, nrows, cls, o, N, L, comps, ncomp, RM, RR, K, ADJ, amb, cap, rowflag, wit, rcap,
                     nd, nl, DM, DR, LM, LR)


@njit(cache=True)
def prep_all(Vm, Vr, nv, comps, ncomp, RM, RR, K, ADJ):
    """Receiver-space data of every relevant component for the polytope Q (see _prep_comp); computed once per node."""
    KD = 1024; KL = 256
    nd = np.empty(ncomp, np.int64); nl = np.empty(ncomp, np.int64)
    DM = np.empty((ncomp, KD, 4)); DR = np.empty((ncomp, KD, 4)); LM = np.empty((ncomp, KL, 4)); LR = np.empty((ncomp, KL, 4))
    for q in range(ncomp):
        a, b = _prep_comp(comps[q], Vm, Vr, nv, RM, RR, K, ADJ, DM[q], DR[q], LM[q], LR[q])
        nd[q] = a; nl[q] = b
    return nd, nl, DM, DR, LM, LR


@njit(cache=True)
def scan_core(Vm, Vr, nv, rows, nrows, cls, o, N, L, comps, ncomp, RM, RR, K, ADJ, amb, cap, rowflag, wit, rcap,
              nd, nl, DM, DR, LM, LR):
    """Row pass of scan_rows on prepared component data (prep_all of the same polytope)."""
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


# ------------------------------------------------------------------------------------------- hierarchical scan
P_DARK, P_LIT, P_SOLID, P_MIXED = 0, 1, 2, 3


def pyramid(cls):
    """Pure-class pyramid of the window (tpl.window convention): codes[off[l] + (bx n + by) n + bz], n = L >> l, code
    DARK / LIT / SOLID if every receiver of the 2^l block has that class, MIXED otherwise."""
    L = cls.shape[0]; q = int(round(np.log2(L)))
    d = cls == DARK; li = cls == LIT; s = cls == SOLID
    codes = []
    for l in range(q + 1):
        code = np.full(d.shape, P_MIXED, np.uint8)
        code[d] = P_DARK; code[li] = P_LIT; code[s] = P_SOLID
        codes.append(code.reshape(-1))
        if l < q:
            n = d.shape[0] // 2
            d = d.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
            li = li.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
            s = s.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
    off = np.zeros(q + 2, np.int64)
    for l in range(q + 1):
        off[l + 1] = off[l] + codes[l].size
    return np.concatenate(codes), off, q


@njit(cache=True)
def scan_tree(Vm, Vr, nv, starts, nstart, lb, pyr, poff, q, cls, o, N, L, comps, ncomp, RM, RR, K, ADJ,
              amb, cap, bmark, wit, rcap, nd, nl, DM, DR, LM, LR, stop_level=0, cnt=np.zeros(4, np.int64),
              amb_limit=1 << 62):
    """Collective classification (same lemmas and outputs as scan_core) over receiver BLOCKS: an octree descent of the
    window from the start blocks (starts[i] = (level, bx, by, bz)), carrying per component only the constraints that
    are still undecided on the block (rigorous range of each affine constraint over the block's receiver box: centre
    +- half extent). A block is consistent when its observation is pure and its status is decided for all its
    receivers; contradicted when pure and decided against the observation (witness = a receiver of the block).
    bmark[(bx n + by) n + bz] = 1 (n = L >> lb) for every level-lb block holding an ambiguous receiver (the start
    blocks of the children). Returns (code, n_amb). With stop_level > 0 blocks are not descended below that level
    (a refutation-only pre-pass: SC_OUT is valid, SC_OK carries no completeness information). With stop_level == lb the
    pass is a complete classification at block resolution: every undecided level-lb block is marked (receivers of
    unmarked blocks are all proven consistent); n_amb counts the PURE undecided blocks. If more than amb_limit
    receivers are ambiguous the descent stops and SC_MANY is returned (no contradiction found so far; no marks)."""
    KD = DM.shape[1]; KL = LM.shape[1]
    NLV = q + 2
    aD = np.empty((NLV, ncomp, KD), np.int64); naD = np.empty((NLV, ncomp), np.int64)
    aL = np.empty((NLV, ncomp, KL), np.int64); naL = np.empty((NLV, ncomp), np.int64)
    sD = np.empty((NLV, ncomp), np.int64)      # dark status per comp: 0 undecided, 1 sure-dark on block, 2 impossible
    sL = np.empty((NLV, ncomp), np.int64)      # lit status per comp: 0 undecided, 1 lit-sure on block
    top = q + 1                                 # virtual parent of the start blocks: all constraints active
    for c in range(ncomp):
        if nd[c] < 0:
            naD[top, c] = 0; sD[top, c] = 2
        else:
            for h in range(nd[c]):
                aD[top, c, h] = h
            naD[top, c] = nd[c]; sD[top, c] = 1 if nd[c] == 0 else 0
        for h in range(nl[c]):
            aL[top, c, h] = h
        naL[top, c] = nl[c]; sL[top, c] = 0
    ox = o[0]; oy = o[1]; oz = o[2]
    stk = np.empty((8 * (q + 2) + nstart + 8, 5), np.int64)
    namb = 0
    nb = L >> lb
    for si in range(nstart):
        sp = 0
        stk[0, 0] = starts[si, 0]; stk[0, 1] = starts[si, 1]; stk[0, 2] = starts[si, 2]; stk[0, 3] = starts[si, 3]
        stk[0, 4] = 1
        sp = 1
        while sp > 0:
            sp -= 1
            l = stk[sp, 0]; bx = stk[sp, 1]; by = stk[sp, 2]; bz = stk[sp, 3]; isstart = stk[sp, 4]
            n = L >> l
            code = pyr[poff[l] + (bx * n + by) * n + bz]
            if code == P_SOLID:
                continue
            sz = 1 << l
            x0 = bx * sz; y0 = by * sz; z0 = bz * sz
            cx = (ox + x0 + 0.5 * sz) / N; cy = (oy + y0 + 0.5 * sz) / N; cz = (oz + z0 + 0.5 * sz) / N
            hh = (sz - 1) * 0.5 / N
            par = top if isstart == 1 else l + 1
            cnt[0] += 1
            anyd = False; alllit = True
            for c in range(ncomp):
                # ---- dark constraints
                st = sD[par, c]; m = 0
                if st == 0:
                    cnt[1] += naD[par, c]
                    for u in range(naD[par, c]):
                        h = aD[par, c, u]
                        mm, rr = aff(DM[c, h], DR[c, h], cx, cy, cz, hh, hh, hh)
                        if up(mm + rr) <= 0.0:
                            continue
                        if dn(mm - rr) > 0.0:
                            st = 2; m = 0
                            break
                        aD[l, c, m] = h; m += 1
                    if st == 0 and m == 0:
                        st = 1
                naD[l, c] = m; sD[l, c] = st
                if st == 1:
                    anyd = True
                # ---- lit planes
                sl = sL[par, c]; m = 0
                if sl == 0:
                    cnt[2] += naL[par, c]
                    for u in range(naL[par, c]):
                        h = aL[par, c, u]
                        mm, rr = aff(LM[c, h], LR[c, h], cx, cy, cz, hh, hh, hh)
                        if dn(mm - rr) > 0.0:
                            sl = 1; m = 0
                            break
                        if up(mm + rr) <= 0.0:
                            continue
                        aL[l, c, m] = h; m += 1
                naL[l, c] = m; sL[l, c] = sl
                if sl == 0:
                    alllit = False
            if anyd and alllit:
                wit[0] = x0; wit[1] = y0; wit[2] = z0; wit[3] = 9
                return SC_BUG, namb
            if code == P_DARK:
                if anyd:
                    continue
                if alllit:
                    wit[0] = x0; wit[1] = y0; wit[2] = z0; wit[3] = 0
                    return SC_OUT, namb
            elif code == P_LIT:
                if alllit:
                    continue
                if anyd:
                    wit[0] = x0; wit[1] = y0; wit[2] = z0; wit[3] = 1
                    return SC_OUT, namb
            if stop_level > 0 and l <= stop_level:
                if l == lb and stop_level == lb:      # coarse resolution: mark the undecided level-lb block
                    bmark[(bx * nb + by) * nb + bz] = 1
                    if code != P_MIXED:              # count only PURE undecided blocks: genuinely ambiguous at
                        namb += 1                    # block scale (mixed blocks can only be decided inside)
                continue
            if l == 0:
                if code == P_DARK or code == P_LIT:     # single receiver, undecided
                    if namb < cap:
                        amb[namb, 0] = x0; amb[namb, 1] = y0; amb[namb, 2] = z0
                        amb[namb, 3] = ((x0 >> lb) * nb + (y0 >> lb)) * nb + (z0 >> lb)
                    namb += 1
                    bmark[((x0 >> lb) * nb + (y0 >> lb)) * nb + (z0 >> lb)] = 1
                    if namb > amb_limit:
                        return SC_MANY, namb
                continue
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        stk[sp, 0] = l - 1; stk[sp, 1] = 2 * bx + dx; stk[sp, 2] = 2 * by + dy; stk[sp, 3] = 2 * bz + dz
                        stk[sp, 4] = 0
                        sp += 1
    if namb == 0 or namb > rcap or namb > cap:
        return SC_OK, namb
    hs_m = np.empty((MH, 4)); hs_r = np.empty((MH, 4)); hs_key = np.empty((MH, 4), np.int64); hs_s = np.empty(MH, np.int64)
    for a in range(namb):
        bmark[amb[a, 3]] = 0
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
        bmark[amb[a, 3]] = 1
        m += 1
    return SC_OK, m
