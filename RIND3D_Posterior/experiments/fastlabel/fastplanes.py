"""Exact integer re-implementation of rind3d.rendering.polyhedra.shadow_planes (research tester only).

The original works in Fractions. Here every body row is scaled once to integers by a positive factor, and the source
p (binary64) is written as P / 2^E. Every intermediate row is then a positive multiple of the original intermediate
row, and `normalize` (division by the largest absolute component) removes positive factors. So the output tuple of
Fractions is IDENTICAL to the original, in the same order. `install(check=K)` monkeypatches the CUDA backend and
compares the first K calls with the original."""
from fractions import Fraction
from math import lcm

import numpy as np

_INT = {}


def _int_rows(body):
    key = id(body)
    v = _INT.get(key)
    if v is None or v[0] is not body:
        rows = []
        for r in body:
            m = lcm(*[Fraction(x).denominator for x in r])
            rows.append(tuple(int(Fraction(x) * m) for x in r))
        v = (body, rows)
        _INT[key] = v
    return v[1]


def _dyadic(x):
    f = Fraction(float(x))                 # exact binary64 value
    return f.numerator, f.denominator


def shadow_planes_fast(body, source):
    rows = _int_rows(body)
    nd = [_dyadic(v) for v in source]
    D = max(d for _, d in nd)
    P = [n * (D // d) for n, d in nd]                          # p_k = P_k / D exactly (power-of-two denominators)
    c = [-(r[3] * D + r[0] * P[0] + r[1] * P[1] + r[2] * P[2]) for r in rows]     # D * lambda_i * c_i (original)
    negative = [i for i, v in enumerate(c) if v < 0]; positive = [i for i, v in enumerate(c) if v > 0]
    if not negative:
        raise ValueError('Source must be outside every closed component')
    out = [rows[i] for i in negative]
    for i, v in enumerate(c):
        if not v:
            a = rows[i]
            out.append((a[0] * D, a[1] * D, a[2] * D, -(a[0] * P[0] + a[1] * P[1] + a[2] * P[2])))
    for i in negative:
        for j in positive:
            n = tuple(c[j] * rows[i][k] - c[i] * rows[j][k] for k in range(3))
            row = (n[0] * D, n[1] * D, n[2] * D, -(n[0] * P[0] + n[1] * P[1] + n[2] * P[2]))
            if any(row):
                out.append(row)
    res = []
    for row in out:
        sc = max(abs(v) for v in row)
        res.append(tuple(Fraction(v, sc) for v in row) if sc else tuple(Fraction(v) for v in row))
    return tuple(dict.fromkeys(res))


def install(check=20):
    import rind3d.rendering.cuda_backend as cb
    import rind3d.rendering.polyhedra as ph
    orig = ph.shadow_planes
    state = dict(n=0)

    def patched(body, source):
        fast = shadow_planes_fast(body, source)
        if state["n"] < check:
            state["n"] += 1
            ref = orig(body, source)
            if ref != fast:
                raise AssertionError("fast shadow planes differ from the original")
        return fast
    cb.shadow_planes = patched
    return state


def _enclose_fast(q):
    """Identical to cuda_backend.enclose for a Fraction q: the tightest binary64 interval [lo, hi] containing q.
    float(n / d) is correctly rounded, so at most one nextafter step is needed; comparisons are exact integer
    cross-multiplications."""
    q = Fraction(q)
    n, d = q.numerator, q.denominator
    x = n / d
    if not np.isfinite(x):
        raise ValueError('Nonfinite rational coefficient')
    a, b = x.as_integer_ratio()
    lhs, rhs = a * d, n * b                        # compare x = a/b with q = n/d
    lo = hi = x
    if lhs > rhs:
        lo = float(np.nextafter(x, -np.inf))
    elif lhs < rhs:
        hi = float(np.nextafter(x, np.inf))
    return lo, hi


def encode_fast(bodies):
    rows = []; offsets = [0]
    for body in bodies:
        rows.extend([[v for q in row for v in _enclose_fast(q)] for row in body]); offsets.append(len(rows))
    return np.asarray(rows, dtype='<f8').reshape(-1, 8), np.asarray(offsets, dtype='<i4')


def install_encode(check=20):
    import rind3d.rendering.cuda_backend as cb
    orig = cb.encode
    state = dict(n=0)

    def patched(bodies):
        fast = encode_fast(bodies)
        if state["n"] < check:
            state["n"] += 1
            r, o = orig(bodies)
            if not (np.array_equal(r, fast[0]) and np.array_equal(o, fast[1])):
                raise AssertionError("fast encode differs from the original")
        return fast
    cb.encode = patched
    return state
