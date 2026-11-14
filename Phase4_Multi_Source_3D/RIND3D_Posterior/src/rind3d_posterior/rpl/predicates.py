"""Filtered float64 predicates for source-box / receiver classification (numba).

Geometry: P[c, k] = (n0, n1, n2, d) with C_c = {x : n.x + d <= 0}, K[c] rows used.  Rows are the float64 roundings
of the canonical rational rows (|coef| <= 1, relative rounding <= 2^-53).

Shadow polyhedron of convex C seen from a receiver r outside C (Fourier-Motzkin, docs/METHOD.md Lemma 1):
    S_C(r) = {s : [r, s] meets C}
           = {s : f_i(s) <= 0 for i in F} and {(c_j n_i - c_i n_j).(s - r) <= 0 for i in F, j in B}
where alpha_i = f_i(r), F = {alpha_i > 0}, B = {alpha_i < 0}, c = -alpha.  (If some alpha_i == 0 the
exact representation needs extra rows; such receivers are flagged AMBIGUOUS and never certified.)

Every sign decision uses an absolute safety margin MARG on unit-normalised planes over coordinates in [0, 1];
float evaluation error of these expressions is < 1e-14, so a decision with |value| > MARG is exact.
Undecided comparisons are always resolved in the conservative direction (CUT / UNKNOWN).
"""
import numpy as np
from numba import njit

MARG = 1e-11      # decision margin for normalised planes
AMARG = 1e-12     # margin for alpha_i = f_i(r) classification
MAXPL = 64 * 33   # max shadow planes per (receiver, component): |F| (1 + |B|) with <= 64 rows

# box/plane classification
INSIDE, OUTSIDE, STRADDLE = 0, 1, 2
# receiver-component status for a source box
EXCL, CONTAINS, CUT, AMBIG = 0, 1, 2, 3


@njit(cache=True)
def plane_box_range(a0, a1, a2, b, lo, hi):
    """min and max of a.x + b over the box [lo, hi]."""
    mn = b
    mx = b
    if a0 >= 0:
        mn += a0 * lo[0]; mx += a0 * hi[0]
    else:
        mn += a0 * hi[0]; mx += a0 * lo[0]
    if a1 >= 0:
        mn += a1 * lo[1]; mx += a1 * hi[1]
    else:
        mn += a1 * hi[1]; mx += a1 * lo[1]
    if a2 >= 0:
        mn += a2 * lo[2]; mx += a2 * hi[2]
    else:
        mn += a2 * hi[2]; mx += a2 * lo[2]
    return mn, mx


@njit(cache=True)
def shadow_planes(r, P, K, c, out):
    """Write unit-normalised rows (a0, a1, a2, b) of S_C(r) into out; returns (n, flag).
    flag: 0 = r strictly outside C (rows valid); 1 = r strictly inside C (S = whole space);
          2 = ambiguous (some |alpha| <= AMARG) -> caller must not certify."""
    k = K[c]
    alpha = np.empty(k)
    nF = 0
    nB = 0
    amb = False
    for i in range(k):
        al = P[c, i, 0] * r[0] + P[c, i, 1] * r[1] + P[c, i, 2] * r[2] + P[c, i, 3]
        alpha[i] = al
        if al > AMARG:
            nF += 1
        elif al < -AMARG:
            nB += 1
        else:
            amb = True
    if nF == 0 and not amb:
        return 0, 1
    if amb:
        return 0, 2
    n = 0
    for i in range(k):
        if alpha[i] <= 0:
            continue
        a0 = P[c, i, 0]; a1 = P[c, i, 1]; a2 = P[c, i, 2]; b = P[c, i, 3]
        nr = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        out[n, 0] = a0 / nr; out[n, 1] = a1 / nr; out[n, 2] = a2 / nr; out[n, 3] = b / nr
        n += 1
        ci = -alpha[i]
        for j in range(k):
            if alpha[j] >= 0:
                continue
            cj = -alpha[j]
            a0 = cj * P[c, i, 0] - ci * P[c, j, 0]
            a1 = cj * P[c, i, 1] - ci * P[c, j, 1]
            a2 = cj * P[c, i, 2] - ci * P[c, j, 2]
            nr = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if nr < 1e-300:
                continue
            a0 /= nr; a1 /= nr; a2 /= nr
            out[n, 0] = a0; out[n, 1] = a1; out[n, 2] = a2
            out[n, 3] = -(a0 * r[0] + a1 * r[1] + a2 * r[2])
            n += 1
    return n, 0


@njit(cache=True)
def classify_component(lo, hi, r, P, K, c, buf, cut_idx):
    """Status of source box [lo,hi] w.r.t. S_C(r) for one component.
    Returns (status, ncut): EXCL (box disjoint from S, certified), CONTAINS (box inside S, certified),
    CUT (cut_idx[:ncut] index rows of buf that straddle the box; all other rows contain the box),
    AMBIG (receiver on a face plane: no certificate possible)."""
    n, flag = shadow_planes(r, P, K, c, buf)
    if flag == 1:
        return CONTAINS, 0
    if flag == 2:
        return AMBIG, 0
    ncut = 0
    for q in range(n):
        mn, mx = plane_box_range(buf[q, 0], buf[q, 1], buf[q, 2], buf[q, 3], lo, hi)
        if mn > MARG:
            return EXCL, 0
        if mx >= -MARG:
            cut_idx[ncut] = q
            ncut += 1
    if ncut == 0:
        return CONTAINS, 0
    return CUT, ncut


@njit(cache=True)
def aabb_disjoint(alo, ahi, blo, bhi):
    for k in range(3):
        if ahi[k] < blo[k] or bhi[k] < alo[k]:
            return True
    return False


@njit(cache=True)
def hull_misses_component(lo, hi, rlo, rhi, P, K, c, clo, chi):
    """Certify conv(B u R) does not meet C (B = [lo,hi] source box, R = [rlo,rhi] receiver box).
    Sufficient tests: AABB separation, or one face plane of C strictly positive on both boxes."""
    ulo = np.empty(3)
    uhi = np.empty(3)
    for k in range(3):
        ulo[k] = min(lo[k], rlo[k])
        uhi[k] = max(hi[k], rhi[k])
    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
        return True
    for i in range(K[c]):
        a0 = P[c, i, 0]; a1 = P[c, i, 1]; a2 = P[c, i, 2]; b = P[c, i, 3]
        mn1, _ = plane_box_range(a0, a1, a2, b, lo, hi)
        if mn1 <= MARG:
            continue
        mn2, _ = plane_box_range(a0, a1, a2, b, rlo, rhi)
        if mn2 > MARG:
            return True
    return False


@njit(cache=True)
def all_visible(lo, hi, rlo, rhi, P, K, clo, chi):
    for c in range(P.shape[0]):
        if not hull_misses_component(lo, hi, rlo, rhi, P, K, c, clo, chi):
            return False
    return True


@njit(cache=True)
def blocked_by(lo, hi, rlo, rhi, P, K, c, clo, chi, buf):
    """Certify every segment [s, x], s in B, x in R meets the single convex C
    (Lemma 2: holds iff for every corner w of R, B is inside S_C(w))."""
    # quick necessary condition: C's AABB must meet the AABB of the union
    ulo = np.empty(3)
    uhi = np.empty(3)
    for k in range(3):
        ulo[k] = min(lo[k], rlo[k])
        uhi[k] = max(hi[k], rhi[k])
    if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
        return False
    w = np.empty(3)
    for corner in range(8):
        w[0] = rhi[0] if corner & 1 else rlo[0]
        w[1] = rhi[1] if corner & 2 else rlo[1]
        w[2] = rhi[2] if corner & 4 else rlo[2]
        n, flag = shadow_planes(w, P, K, c, buf)
        if flag == 1:
            continue
        if flag == 2:
            return False
        for q in range(n):
            _, mx = plane_box_range(buf[q, 0], buf[q, 1], buf[q, 2], buf[q, 3], lo, hi)
            if mx >= -MARG:
                return False
    return True


@njit(cache=True)
def any_blocked(lo, hi, rlo, rhi, P, K, clo, chi, buf):
    for c in range(P.shape[0]):
        if blocked_by(lo, hi, rlo, rhi, P, K, c, clo, chi, buf):
            return True
    return False


@njit(cache=True)
def segment_hits(s, r, P, K, c):
    """Float Cyrus-Beck test of the closed segment [s, r] against C: 1 hit, 0 miss, -1 too close to call."""
    tlo = 0.0
    thi = 1.0
    tol = 1e-12
    for i in range(K[c]):
        al = P[c, i, 0] * s[0] + P[c, i, 1] * s[1] + P[c, i, 2] * s[2] + P[c, i, 3]
        be = P[c, i, 0] * (r[0] - s[0]) + P[c, i, 1] * (r[1] - s[1]) + P[c, i, 2] * (r[2] - s[2])
        if abs(be) < 1e-300:
            if al > tol:
                return 0
            if al > -tol:
                return -1
            continue
        t = -al / be
        if be > 0:
            if t < thi:
                thi = t
        else:
            if t > tlo:
                tlo = t
    if tlo < thi - tol:
        return 1
    if tlo > thi + tol:
        return 0
    return -1


@njit(cache=True)
def render_points(s, R, P, K):
    """Float response of source s at receiver points R (n,3): 1 lit, 0 dark, -1 undecided."""
    n = R.shape[0]
    out = np.empty(n, np.int8)
    for q in range(n):
        y = 1
        for c in range(P.shape[0]):
            h = segment_hits(s, R[q], P, K, c)
            if h == 1:
                y = 0
                break
            if h == -1:
                y = -1
        out[q] = y
    return out
