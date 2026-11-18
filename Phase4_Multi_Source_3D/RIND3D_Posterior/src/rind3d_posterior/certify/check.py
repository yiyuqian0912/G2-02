"""Rigorous visibility predicates of the verifier (numba). Independent of the labelling engines.

Physics (docs/RIND3D/SEMANTICS.md, closed-solid-closed-segment-v1): receiver x is response-inactive for source s iff
the closed segment [s, x] meets the closed solid S = U_c C_c, C_c = {y : f_i(y) <= 0, i in faces(c)}.

Lemma (exact segment-polytope intersection, no adjacency or silhouette needed). Let s be outside C with every face
strictly signed at s, F = {i : f_i(s) > 0} (nonempty), B = {j : f_j(s) < 0}. Then
    [s, x] meets C  <=>  f_i(x) <= 0 for all i in F  and  B_ij(s, x) := f_i(s) f_j(x) - f_j(s) f_i(x) <= 0 for all
                         (i, j) in F x B.
Proof: on p(t) = s + t (x - s), f_k(p(t)) = (1 - t) f_k(s) + t f_k(x); the segment meets C iff some t in [0, 1] has
f_k(p(t)) <= 0 for all k; for i in F this is t >= f_i(s)/(f_i(s) - f_i(x)) and needs f_i(x) <= 0 (t <= 1); for j in B
it is t <= -f_j(s)/(f_j(x) - f_j(s)) when f_j(x) > f_j(s) and void otherwise; eliminating t gives exactly the pair
conditions (which hold automatically when f_j(x) <= f_j(s)). The roles of s and x can be exchanged (window form).
Box form: B_ij is affine in s for fixed x and affine in x for fixed s, so its extremes over (polytope R) x (receiver
box Q) are attained at vertex/corner pairs; with f_i(s) > 0 on R and f_i(x) < 0 on Q,
    B_ij <= 0 on R x Q  <=>  max_R tau_ij <= min_Q tau_ij,   tau_ij = f_j / f_i  (linear-fractional, extremes at vertices).
Silhouette lemma. With s outside C (strict aspect), {x : [s, x] meets C} = {f_i(x) <= 0, i in F} n K(s), and the cone
K(s) = s + cone(C - s) equals the intersection of the half-spaces B_ij(s, .) <= 0 over the SILHOUETTE pairs only
(facets i in F, j in B sharing an edge). Proof: each such half-space contains C, hence K(s); conversely every facet
plane of the polyhedral cone K(s) passes through s and supports C along an edge (a vertex gives no 2-face, a facet of
C would contain s), and a supporting plane through s and an edge e exists iff the two facets of e face s oppositely,
i.e. e is a silhouette edge. So COVER needs the gates and the silhouette pairs only (exact adjacency from the exact
vertex/edge enumeration of certify.scene); every pair row is a necessary condition, so NOCOVER may use any of them.
Statuses for (region R, receiver box Q, component c):
    COVER   every receiver of Q is response-inactive for every s in R (c meets every segment),
    NOCOVER c meets no segment from R to Q (strict separations),
    UNDEC   neither is proven.
All quantities are enclosures: face values f_i at binary64 receiver corners within +-FB; at polytope vertices
(enclosed in Vm +- Vr) within +-(FB + sum |a| Vr); quotients by the rigorous mid-radius division `_qdiv`; endpoint
widening `_wid` makes q -+ r' rigorous after binary64 rounding (see ar.py)."""
import numpy as np
from numba import njit

from .ar import U, INFL, TINY, two_prod, dd_add, DD_EPS

COVER, NOCOVER, UNDEC = 0, 1, 2
DARK, LIT, SOLID, MIXED = 0, 1, 2, 3


@njit(cache=True, inline="always")
def _wid(q, r):
    """r' such that fl(q - r') <= q - r and fl(q + r') >= q + r."""
    return (r + 2.0 * U * (abs(q) + r)) * INFL + TINY


@njit(cache=True, inline="always")
def _qdiv(nm, nr, dm, dr):
    """(q, r): |n/d - q| <= r for n in nm +- nr, d in dm +- dr, |dm| > dr (caller checks)."""
    q = nm / dm
    r = (nr + abs(q) * (1.0 + 2.0 * U) * dr) / (abs(dm) - dr) * INFL + U * abs(q) * INFL + TINY
    return q, r


@njit(cache=True)
def region_face_values(slot, c, K, RM, FB, RR, Vm, Vr, nV, fRlo, fRhi, fvm, fvr):
    """Per face i of c: rigorous [fRlo, fRhi] of f_i over the polytope (vertex enclosures), and per vertex the
    mid/radius of f_i (fvm[i, v], fvr[i, v]) for the quotient ranges."""
    nv = nV[slot]
    for i in range(K[c]):
        a0 = RM[c, i, 0]; a1 = RM[c, i, 1]; a2 = RM[c, i, 2]; b = RM[c, i, 3]
        ra = (abs(a0) + RR[c, i, 0]); rb = (abs(a1) + RR[c, i, 1]); rc = (abs(a2) + RR[c, i, 2])
        lo = np.inf; hi = -np.inf
        for v in range(nv):
            m = a0 * Vm[slot, v, 0] + a1 * Vm[slot, v, 1] + a2 * Vm[slot, v, 2] + b
            r = FB[c, i] + (ra * Vr[slot, v, 0] + rb * Vr[slot, v, 1] + rc * Vr[slot, v, 2]) * INFL
            fvm[i, v] = m; fvr[i, v] = r
            w = _wid(m, r)
            if m - w < lo:
                lo = m - w
            if m + w > hi:
                hi = m + w
        fRlo[i] = lo; fRhi[i] = hi


@njit(cache=True)
def region_tau_table(nv, Kc, asp, fvm, fvr, Tlo, Thi):
    """Tlo/Thi[i, j]: rigorous range over the polytope of tau_ij = f_j / f_i for every face i strictly signed on it
    (asp[i] != 0); NaN rows otherwise."""
    for i in range(Kc):
        for j in range(Kc):
            Tlo[i, j] = np.nan; Thi[i, j] = np.nan
        if asp[i] == 0:
            continue
        for j in range(Kc):
            if j == i:
                continue
            lo = np.inf; hi = -np.inf
            ok = True
            for v in range(nv):
                dm = fvm[i, v]; dr = fvr[i, v]
                if not (abs(dm) > dr):
                    ok = False
                    break
                q, r = _qdiv(fvm[j, v], fvr[j, v], dm, dr)
                w = _wid(q, r)
                if q - w < lo:
                    lo = q - w
                if q + w > hi:
                    hi = q + w
            if ok:
                Tlo[i, j] = lo; Thi[i, j] = hi


@njit(cache=True)
def face_at_point(c, i, x0, x1, x2, RM, RL, RR2):
    """(mid, rad) of f_i(x) at a binary64 point x, evaluated in double-double from the double-double row: the error is
    relative to |f_i(x)| (plus ~1e-31), not to the magnitude of the terms."""
    sh = RM[c, i, 3]; sl = RL[c, i, 3]
    mag = abs(sh)
    rad = RR2[c, i, 3]
    for k in range(3):
        xk = x0 if k == 0 else (x1 if k == 1 else x2)
        p, e = two_prod(RM[c, i, k], xk)
        sh, sl = dd_add(sh, sl, p, e)
        q = RL[c, i, k] * xk
        sh, sl = dd_add(sh, sl, q, 0.0)
        mag += abs(p) + abs(q) + abs(sh)
        rad += RR2[c, i, k] * abs(xk) + U * abs(q)
    m = sh + sl
    rad += abs((sh - m) + sl) + 8.0 * DD_EPS * mag + U * abs(m)
    return m, rad * INFL + TINY


@njit(cache=True)
def box_face_values_fast(qlo, qhi, ispt, c, Kc, RM, FB, fqm, fqr, fql, fqh):
    """Binary64 face values at the box corners with the a priori bound FB (row radius + rounding for |x| <= 2): the
    fast path; decisions that it leaves undecided are redone with box_face_values (double-double)."""
    nc = 1 if ispt else 8
    for i in range(Kc):
        a0 = RM[c, i, 0]; a1 = RM[c, i, 1]; a2 = RM[c, i, 2]; b = RM[c, i, 3]; r = FB[c, i]
        lo = np.inf; hi = -np.inf
        for cr in range(nc):
            x0 = qhi[0] if cr & 1 else qlo[0]
            x1 = qhi[1] if cr & 2 else qlo[1]
            x2 = qhi[2] if cr & 4 else qlo[2]
            m = a0 * x0 + a1 * x1 + a2 * x2 + b
            fqm[i, cr] = m; fqr[i, cr] = r
            w = _wid(m, r)
            if m - w < lo:
                lo = m - w
            if m + w > hi:
                hi = m + w
        fql[i] = lo; fqh[i] = hi
    return nc


@njit(cache=True)
def box_face_values(qlo, qhi, ispt, c, Kc, RM, RL, RR2, fqm, fqr, fql, fqh):
    """Face values of c at the corners of the receiver box [qlo, qhi] (binary64 corners), double-double accurate:
    fqm[i, cr] mid, fqr[i, cr] rigorous radius, fql/fqh[i] rigorous min/max over the corners. Returns #corners."""
    nc = 1 if ispt else 8
    for i in range(Kc):
        lo = np.inf; hi = -np.inf
        for cr in range(nc):
            x0 = qhi[0] if cr & 1 else qlo[0]
            x1 = qhi[1] if cr & 2 else qlo[1]
            x2 = qhi[2] if cr & 4 else qlo[2]
            m, r = face_at_point(c, i, x0, x1, x2, RM, RL, RR2)
            fqm[i, cr] = m; fqr[i, cr] = r
            w = _wid(m, r)
            if m - w < lo:
                lo = m - w
            if m + w > hi:
                hi = m + w
        fql[i] = lo; fqh[i] = hi
    return nc


@njit(cache=True)
def _box_tau(nc, i, j, fqm, fqr):
    """Rigorous [lo, hi] of f_j / f_i over the box corners; (nan, nan) if f_i is not separated from 0."""
    lo = np.inf; hi = -np.inf
    for cr in range(nc):
        dm = fqm[i, cr]
        if not (abs(dm) > fqr[i, cr]):
            return np.nan, np.nan
        q, r = _qdiv(fqm[j, cr], fqr[j, cr], dm, fqr[i, cr])
        w = _wid(q, r)
        if q - w < lo:
            lo = q - w
        if q + w > hi:
            hi = q + w
    return lo, hi


@njit(cache=True)
def fact_sign(c, i, j, xr, FC, FI, FJ, FX, FS, nfa):
    """Known sign (+1: >= 0, -1: <= 0, 0 unknown) of B_ij(x, s) = f_i(x) f_j(s) - f_j(x) f_i(s) on the TRUE region for
    receiver x (flat index xr), from the path's symbolic facts (FS[t] = sign of B_{FI FJ}(x_FX, s)). Facts are exact
    statements about the true region (it is defined by the exact pencil plane), not interval computations."""
    for t in range(nfa):
        if FC[t] != c or FX[t] != xr:
            continue
        if FI[t] == i and FJ[t] == j:
            return FS[t]
        if FI[t] == j and FJ[t] == i:
            return -FS[t]
    return 0


@njit(cache=True, inline="always")
def _sgn_iv(m, r):
    """+1 / -1 if the interval m +- r excludes 0, else 0 (unknown)."""
    w = _wid(m, r)
    if m - w > 0.0:
        return 1
    if m + w < 0.0:
        return -1
    return 0


@njit(cache=True)
def derived_sign(c, i, j, xr, xim, xir, xjm, xjr, effi, FC, FI, FJ, FX, FS, FVm, FVr, nfa):
    """Sign of B_ij(x, s) on the true region derived from a fact on another receiver y of the same pencil, through the
    exact identity  f_i(y) B_ij(x, s) = f_i(x) B_ij(y, s) + B_ij(x, y) f_i(s)  (B_ij(x, y) = f_i(x) f_j(y) - f_j(x)
    f_i(y)); effi = sign of f_i on the true region. Returns +1 (>= 0, > 0 a.e.), -1, or 0 (not derivable)."""
    for t in range(nfa):
        if FC[t] != c:
            continue
        if FI[t] == i and FJ[t] == j:
            sB = FS[t]; yim = FVm[t, 0]; yir = FVr[t, 0]; yjm = FVm[t, 1]; yjr = FVr[t, 1]
        elif FI[t] == j and FJ[t] == i:
            sB = -FS[t]; yim = FVm[t, 1]; yir = FVr[t, 1]; yjm = FVm[t, 0]; yjr = FVr[t, 0]
        else:
            continue
        if FX[t] == xr:
            return sB
        sy = _sgn_iv(yim, yir)
        sx = _sgn_iv(xim, xir)
        if sy == 0 or sx == 0 or effi == 0:
            continue
        p = xim * yjm; q = xjm * yim
        bm = p - q
        br = (xir * (abs(yjm) + yjr) + abs(xim) * yjr + xjr * (abs(yim) + yir) + abs(xjm) * yir) * INFL \
            + 2.0 * U * (abs(p) + abs(q)) * INFL + TINY
        sxy = _sgn_iv(bm, br)
        t1 = sx * sB                 # sign of f_i(x) B_ij(y, s)  (a.e. strict: B_ij(y, .) is a nonzero affine function)
        if sxy == 0:
            continue                 # second term of unknown sign
        t2 = sxy * effi              # sign of B_ij(x, y) f_i(s)
        if t1 == t2:
            return t1 * sy
    return 0


@njit(cache=True)
def _pair_flags(c, nc, nv, i, j, fqm, fqr, fvm, fvr, cflat, FC, FI, FJ, FX, FS, nfa, eff, FVm, FVr):
    """Signs of B_ij(x, s) over (box corners) x (polytope vertices), affine in each argument: returns
    (ge, gt, le, lt) = (>= 0, > 0 a.e., <= 0, < 0 a.e.) certified for every corner/vertex pair. A corner that is the
    receiver of a symbolic fact uses the fact's exact sign (non-strict on the fact plane only: measure zero)."""
    ge = True; gt = True; le = True; lt = True
    for cr in range(nc):
        fs = 0
        if nfa > 0:
            fs = derived_sign(c, i, j, cflat[cr], fqm[i, cr], fqr[i, cr], fqm[j, cr], fqr[j, cr], eff[i], FC, FI, FJ, FX, FS,
                              FVm, FVr, nfa)
        if fs > 0:
            le = False; lt = False
            continue
        if fs < 0:
            ge = False; gt = False
            continue
        ai = fqm[i, cr]; aj = fqm[j, cr]; ri = fqr[i, cr]; rj = fqr[j, cr]
        lo = np.inf; hi = -np.inf
        for v in range(nv):
            bj = fvm[j, v]; rbj = fvr[j, v]; bi = fvm[i, v]; rbi = fvr[i, v]
            p = ai * bj; q = aj * bi
            m = p - q
            r = (ri * (abs(bj) + rbj) + abs(ai) * rbj + rj * (abs(bi) + rbi) + abs(aj) * rbi) * INFL \
                + 2.0 * U * (abs(p) + abs(q)) * INFL + TINY
            w = _wid(m, r)
            if m - w < lo:
                lo = m - w
            if m + w > hi:
                hi = m + w
        if not (lo >= 0.0):
            ge = False
        if not (lo > 0.0):
            gt = False
        if not (hi <= 0.0):
            le = False
        if not (hi < 0.0):
            lt = False
        if not (ge or gt or le or lt):
            break
    return ge, gt, le, lt


@njit(cache=True)
def comp_status(c, Kc, eff, fRlo, fRhi, Tlo, Thi, nc, fqm, fqr, fql, fqh, Rlo, Rhi, qlo, qhi, alo, ahi, nv, fvm, fvr,
                cflat, FC, FI, FJ, FX, FS, nfa, FVm, FVr, deep, ADJ):
    """Status of component c for (true region, receiver box): COVER / NOCOVER / UNDEC, plus a hint to expand the
    block. eff[i]: sign of face i on the true region (+1 front, -1 back, 0 unknown), from intervals on the outer
    polytope or from a symbolic face-split fact. Pair conditions are decided by the fast quotient ranges, and when
    those are inconclusive (only when deep: leaf proofs; internal-node pruning uses the fast form only) by the bilinear
    vertex/corner form with the path's symbolic pencil facts."""
    srcform = True
    for i in range(Kc):
        if eff[i] == 0:
            srcform = False
            break
    amb = False
    # ---- source form: aspect of c fixed on the region
    if srcform:
        gated = True
        for i in range(Kc):
            if eff[i] != 1:
                continue
            if fql[i] > 0.0:
                return NOCOVER, False          # s (a.e.) and every x strictly in front of face i
            if not (fqh[i] < 0.0):
                gated = False
        allcov = gated
        for i in range(Kc):
            if eff[i] != 1:
                continue
            for j in range(Kc):
                if eff[j] != -1 or not ADJ[c, i, j]:
                    continue
                if gated:
                    ql, qh = _box_tau(nc, i, j, fqm, fqr)
                    if not (np.isnan(ql) or np.isnan(Tlo[i, j])):
                        if Tlo[i, j] > qh:
                            return NOCOVER, False
                        if Thi[i, j] <= ql:
                            continue
                        if nc > 1 and (qh - ql) >= 2.0 * (Thi[i, j] - Tlo[i, j]):
                            amb = True
                if not deep:
                    allcov = False
                    continue
                # B_ij(s, x) = -B_ij(x, s): condition B_ij(s, x) <= 0  <=>  B_ij(x, s) >= 0
                ge, gt, le, lt = _pair_flags(c, nc, nv, i, j, fqm, fqr, fvm, fvr, cflat, FC, FI, FJ, FX, FS, nfa, eff, FVm, FVr)
                if lt:
                    return NOCOVER, False      # necessary pair condition violated for a.e. s and every x
                if not ge:
                    allcov = False
        if allcov:
            return COVER, False
        if not gated:
            amb = True
    # ---- window form: aspect of c fixed on the receiver box
    qform = True
    for i in range(Kc):
        if not (fql[i] > 0.0 or fqh[i] < 0.0):
            qform = False
            break
    if qform:
        nF = 0
        for i in range(Kc):
            if fql[i] > 0.0:
                nF += 1
        if nF == 0:
            return COVER, False                # every receiver of the box lies in the closed component
        sgated = True
        for i in range(Kc):
            if not (fql[i] > 0.0):
                continue
            if eff[i] == 1:
                return NOCOVER, False          # source (a.e.) and receivers strictly in front of face i
            if eff[i] != -1:
                sgated = False
        allcov = sgated
        for i in range(Kc):
            if not (fql[i] > 0.0):
                continue
            for j in range(Kc):
                if not (fqh[j] < 0.0) or not ADJ[c, i, j]:
                    continue
                if sgated:
                    ql, qh = _box_tau(nc, i, j, fqm, fqr)
                    if not (np.isnan(ql) or np.isnan(Tlo[i, j])):
                        # B_ij(x, s) <= 0  <=>  tau(x) <= tau(s)
                        if ql > Thi[i, j]:
                            return NOCOVER, False
                        if qh <= Tlo[i, j]:
                            continue
                        if nc > 1 and (qh - ql) >= 2.0 * (Thi[i, j] - Tlo[i, j]):
                            amb = True
                if not deep:
                    allcov = False
                    continue
                ge, gt, le, lt = _pair_flags(c, nc, nv, i, j, fqm, fqr, fvm, fvr, cflat, FC, FI, FJ, FX, FS, nfa, eff, FVm, FVr)
                if gt:
                    return NOCOVER, False
                if not le:
                    allcov = False
        if allcov:
            return COVER, False
    # ---- aspect-free separations
    for i in range(Kc):
        if eff[i] == 1 and fql[i] > 0.0:
            return NOCOVER, False
    for k in range(3):
        lo = min(Rlo[k], qlo[k]); hi = max(Rhi[k], qhi[k])
        if hi < alo[c, k] or lo > ahi[c, k]:
            return NOCOVER, False
    return UNDEC, amb            # amb: hint to expand the receiver block (ambiguous gating or receiver-limited)


@njit(cache=True)
def item_box(code, o, N, qlo, qhi):
    """Receiver-centre box of the pure block `code` = (level << 48) | (bx << 32) | (by << 16) | bz (exact)."""
    l = (code >> 48) & 0xFFFF
    s = np.int64(1) << l
    bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
    qlo[0] = (o[0] + bx * s + 0.5) / N; qlo[1] = (o[1] + by * s + 0.5) / N; qlo[2] = (o[2] + bz * s + 0.5) / N
    qhi[0] = (o[0] + (bx + 1) * s - 0.5) / N; qhi[1] = (o[1] + (by + 1) * s - 0.5) / N; qhi[2] = (o[2] + (bz + 1) * s - 0.5) / N
    return l


@njit(cache=True)
def poly_bbox(slot, Vm, Vr, nV, lo, hi):
    for k in range(3):
        lo[k] = np.inf; hi[k] = -np.inf
    for v in range(nV[slot]):
        for k in range(3):
            a = Vm[slot, v, k] - Vr[slot, v, k]; b = Vm[slot, v, k] + Vr[slot, v, k]
            if a < lo[k]:
                lo[k] = a
            if b > hi[k]:
                hi[k] = b
    for k in range(3):
        lo[k] = np.nextafter(lo[k], -np.inf); hi[k] = np.nextafter(hi[k], np.inf)
