"""Rigorous arithmetic for the certificate verifier (numba, binary64, round-to-nearest-even).

Everything here is independent of the labelling engines. The model assumed is IEEE-754 binary64 with round-to-nearest
and no fused multiply-add contraction (numba's default, fastmath=False); `self_test()` checks the error-free
transformations on adversarial inputs at import time of the verifier and refuses to run if they are not exact.

* Error-free transformations (Dekker / Knuth): two_sum, two_prod (Veltkamp splitting, no FMA needed).
* Shewchuk expansion arithmetic (nonoverlapping, increasing magnitude, zero-eliminated) and the EXACT signs of 3x3
  and 4x4 determinants of binary64 entries (`det3_sign`, `det4_sign`) with a static floating-point filter in front.
* Outward rounding `dn`/`up` (nextafter) and a few rigorous interval helpers.
A priori error bounds: for an expression that is a sum of products evaluated in binary64, every intermediate result
carries at most a relative error u = 2^-53 per operation, so |fl(E) - E| <= gamma_n * E_abs, gamma_n = n u / (1 - n u),
where n is the longest operation path and E_abs the same expression with absolute values (Higham, Accuracy and
Stability of Numerical Algorithms, 2nd ed., Lemma 3.1 / 3.3). Underflow adds at most 2^-1074 per operation; every
bound below adds TINY = 2^-1000 to cover it."""
import numpy as np
from numba import njit

U = 2.0 ** -53
TINY = 2.0 ** -1000
SPLITTER = 134217729.0          # 2^27 + 1


def gamma(n):
    return n * U / (1.0 - n * U)


# safety-inflated constants (bound evaluation itself is rounded: inflate by a few ulps)
G5 = gamma(5) * (1.0 + 2.0 ** -40)
G9 = gamma(9) * (1.0 + 2.0 ** -40)
G4 = gamma(4) * (1.0 + 2.0 ** -40)
G3 = gamma(3) * (1.0 + 2.0 ** -40)
G2 = gamma(2) * (1.0 + 2.0 ** -40)
INFL = 1.0 + 64 * U             # multiply a computed nonnegative error bound by this to cover its own rounding


@njit(cache=True, inline="always")
def dn(x):
    return np.nextafter(x, -np.inf)


@njit(cache=True, inline="always")
def up(x):
    return np.nextafter(x, np.inf)


@njit(cache=True, inline="always")
def two_sum(a, b):
    x = a + b
    bv = x - a
    av = x - bv
    return x, (a - av) + (b - bv)


@njit(cache=True, inline="always")
def fast_two_sum(a, b):            # requires |a| >= |b|
    x = a + b
    return x, b - (x - a)


@njit(cache=True, inline="always")
def _split(a):
    c = SPLITTER * a
    ab = c - a
    hi = c - ab
    return hi, a - hi


@njit(cache=True, inline="always")
def two_prod(a, b):
    x = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e1 = x - ah * bh
    e2 = e1 - al * bh
    e3 = e2 - ah * bl
    return x, al * bl - e3


@njit(cache=True, inline="always")
def dd_add(ah, al, bh, bl):
    """Double-double sum (accurate variant): |result - (a + b)| <= 2^-100 (|a| + |b|) (Joldes, Muller, Popescu 2017)."""
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e += t
    s, e = fast_two_sum(s, e)
    e += f
    return fast_two_sum(s, e)


DD_EPS = 2.0 ** -100


# ------------------------------------------------------------------------------------------- expansions (Shewchuk)
@njit(cache=True)
def exp_sum(e, ne, f, nf, h):
    """h = e + f (fast_expansion_sum_zeroelim). e, f nonoverlapping increasing; returns len(h)."""
    if ne == 0:
        for i in range(nf):
            h[i] = f[i]
        return nf
    if nf == 0:
        for i in range(ne):
            h[i] = e[i]
        return ne
    ei = 0; fi = 0
    enow = e[0]; fnow = f[0]
    if (fnow > enow) == (fnow > -enow):
        Q = enow; ei += 1
    else:
        Q = fnow; fi += 1
    hi = 0
    if ei < ne and fi < nf:
        enow = e[ei]; fnow = f[fi]
        if (fnow > enow) == (fnow > -enow):
            Q, hh = fast_two_sum(enow, Q); ei += 1
        else:
            Q, hh = fast_two_sum(fnow, Q); fi += 1
        if hh != 0.0:
            h[hi] = hh; hi += 1
        while ei < ne and fi < nf:
            enow = e[ei]; fnow = f[fi]
            if (fnow > enow) == (fnow > -enow):
                Q, hh = two_sum(Q, enow); ei += 1
            else:
                Q, hh = two_sum(Q, fnow); fi += 1
            if hh != 0.0:
                h[hi] = hh; hi += 1
    while ei < ne:
        Q, hh = two_sum(Q, e[ei]); ei += 1
        if hh != 0.0:
            h[hi] = hh; hi += 1
    while fi < nf:
        Q, hh = two_sum(Q, f[fi]); fi += 1
        if hh != 0.0:
            h[hi] = hh; hi += 1
    if Q != 0.0 or hi == 0:
        h[hi] = Q; hi += 1
    return hi


@njit(cache=True)
def exp_scale(e, ne, b, h):
    """h = b * e (scale_expansion_zeroelim); returns len(h)."""
    if ne == 0 or b == 0.0:
        h[0] = 0.0
        return 1
    Q, hh = two_prod(e[0], b)
    hi = 0
    if hh != 0.0:
        h[hi] = hh; hi += 1
    for i in range(1, ne):
        p1, p0 = two_prod(e[i], b)
        s, hh = two_sum(Q, p0)
        if hh != 0.0:
            h[hi] = hh; hi += 1
        Q, hh = fast_two_sum(p1, s)
        if hh != 0.0:
            h[hi] = hh; hi += 1
    if Q != 0.0 or hi == 0:
        h[hi] = Q; hi += 1
    return hi


@njit(cache=True, inline="always")
def exp_sign(e, ne):
    v = e[ne - 1]
    return 1 if v > 0 else (-1 if v < 0 else 0)


@njit(cache=True)
def _det2_exp(a, b, c, d, out):
    """out = a d - b c exactly; returns length."""
    x1, x0 = two_prod(a, d)
    y1, y0 = two_prod(b, c)
    e = np.empty(2); f = np.empty(2)
    ne = 0
    if x0 != 0.0:
        e[ne] = x0; ne += 1
    e[ne] = x1; ne += 1
    nf = 0
    if y0 != 0.0:
        f[nf] = -y0; nf += 1
    f[nf] = -y1; nf += 1
    return exp_sum(e, ne, f, nf, out)


@njit(cache=True)
def det3_exp(m00, m01, m02, m10, m11, m12, m20, m21, m22, out):
    """Exact 3x3 determinant as an expansion in out (len >= 32); returns its length."""
    a = np.empty(4); b = np.empty(4); c = np.empty(4)
    na = _det2_exp(m11, m12, m21, m22, a)
    nb = _det2_exp(m10, m12, m20, m22, b)
    nc = _det2_exp(m10, m11, m20, m21, c)
    sa = np.empty(8); sb = np.empty(8); sc = np.empty(8)
    nsa = exp_scale(a, na, m00, sa)
    nsb = exp_scale(b, nb, -m01, sb)
    nsc = exp_scale(c, nc, m02, sc)
    t = np.empty(16)
    nt = exp_sum(sa, nsa, sb, nsb, t)
    return exp_sum(t, nt, sc, nsc, out)


@njit(cache=True)
def det3_float(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """(value, error bound) of the binary64 evaluation of the 3x3 determinant."""
    p1 = m11 * m22; p2 = m12 * m21
    q1 = m10 * m22; q2 = m12 * m20
    r1 = m10 * m21; r2 = m11 * m20
    v = m00 * (p1 - p2) - m01 * (q1 - q2) + m02 * (r1 - r2)
    perm = abs(m00) * (abs(p1) + abs(p2)) + abs(m01) * (abs(q1) + abs(q2)) + abs(m02) * (abs(r1) + abs(r2))
    return v, G5 * perm * INFL + TINY


@njit(cache=True)
def det3_sign(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    v, e = det3_float(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if v > e:
        return 1
    if v < -e:
        return -1
    out = np.empty(32)
    n = det3_exp(m00, m01, m02, m10, m11, m12, m20, m21, m22, out)
    return exp_sign(out, n)


@njit(cache=True)
def det4_rows_sign(R0, R1, R2, R3):
    """Exact sign of det [R0; R1; R2; R3] for 4-vectors (a0, a1, a2, b), by Laplace expansion on the last column:
    det = -b0 D(1,2,3) + b1 D(0,2,3) - b2 D(0,1,3) + b3 D(0,1,2), D = 3x3 determinant of the normals."""
    d123, e123 = det3_float(R1[0], R1[1], R1[2], R2[0], R2[1], R2[2], R3[0], R3[1], R3[2])
    d023, e023 = det3_float(R0[0], R0[1], R0[2], R2[0], R2[1], R2[2], R3[0], R3[1], R3[2])
    d013, e013 = det3_float(R0[0], R0[1], R0[2], R1[0], R1[1], R1[2], R3[0], R3[1], R3[2])
    d012, e012 = det3_float(R0[0], R0[1], R0[2], R1[0], R1[1], R1[2], R2[0], R2[1], R2[2])
    v = -R0[3] * d123 + R1[3] * d023 - R2[3] * d013 + R3[3] * d012
    # |fl(b D) - b D| <= |b| e_D (1 + u) + u |b D|; four-term sum: gamma_3
    err = (abs(R0[3]) * (e123 + abs(d123)) + abs(R1[3]) * (e023 + abs(d023)) + abs(R2[3]) * (e013 + abs(d013))
           + abs(R3[3]) * (e012 + abs(d012))) * G5 * INFL
    err += abs(R0[3]) * e123 + abs(R1[3]) * e023 + abs(R2[3]) * e013 + abs(R3[3]) * e012 + TINY
    err *= INFL
    if v > err:
        return 1
    if v < -err:
        return -1
    D = np.empty((4, 32)); nD = np.zeros(4, np.int64)
    nD[0] = det3_exp(R1[0], R1[1], R1[2], R2[0], R2[1], R2[2], R3[0], R3[1], R3[2], D[0])
    nD[1] = det3_exp(R0[0], R0[1], R0[2], R2[0], R2[1], R2[2], R3[0], R3[1], R3[2], D[1])
    nD[2] = det3_exp(R0[0], R0[1], R0[2], R1[0], R1[1], R1[2], R3[0], R3[1], R3[2], D[2])
    nD[3] = det3_exp(R0[0], R0[1], R0[2], R1[0], R1[1], R1[2], R2[0], R2[1], R2[2], D[3])
    S = np.empty((4, 64)); nS = np.zeros(4, np.int64)
    nS[0] = exp_scale(D[0], nD[0], -R0[3], S[0])
    nS[1] = exp_scale(D[1], nD[1], R1[3], S[1])
    nS[2] = exp_scale(D[2], nD[2], -R2[3], S[2])
    nS[3] = exp_scale(D[3], nD[3], R3[3], S[3])
    t1 = np.empty(128); t2 = np.empty(128); t3 = np.empty(256)
    n1 = exp_sum(S[0], nS[0], S[1], nS[1], t1)
    n2 = exp_sum(S[2], nS[2], S[3], nS[3], t2)
    n3 = exp_sum(t1, n1, t2, n2, t3)
    return exp_sign(t3, n3)


@njit(cache=True)
def exp_approx(e, ne):
    """(a, r) with |sum(e) - a| <= r: a = binary64 sum from small to large (Shewchuk 'estimate'), r rigorous."""
    a = 0.0
    tail = 0.0
    for i in range(ne):
        a += e[i]
    # the exact sum differs from the largest component by less than sum of the others' magnitudes
    for i in range(ne - 1):
        tail += abs(e[i])
    r = abs(a - e[ne - 1]) + tail
    return a, up(up(r * INFL)) + TINY


# ------------------------------------------------------------------------------------------- self test
def self_test():
    """Exactness checks of the error-free transformations (detects FMA contraction or extended precision)."""
    from fractions import Fraction as Q
    rng = np.random.default_rng(12345)
    for _ in range(2000):
        a, b = rng.standard_normal(2) * np.exp2(rng.integers(-30, 30, 2))
        x, y = two_prod(float(a), float(b))
        assert Q(x) + Q(y) == Q(float(a)) * Q(float(b)), "two_prod is not exact (FMA contraction?)"
        x, y = two_sum(float(a), float(b))
        assert Q(x) + Q(y) == Q(float(a)) + Q(float(b)), "two_sum is not exact"
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0 + 2.0 ** -49]])   # det = -3 * 2^-49
    assert det3_sign(*M.reshape(-1)) == -1
    M[2, 2] = 9.0
    assert det3_sign(*M.reshape(-1)) == 0
    return True
