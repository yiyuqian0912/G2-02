"""Independent exact rational reference for the XLF mathematical and implementation tests (not a test module).

Everything here uses Python Fractions only (no binary64 decisions) and shares no code with rind3d_posterior.xlf:
* seg_hits(s, x, rows): does the CLOSED segment [s, x] meet the CLOSED polytope {y : a.y + d <= 0}? (1-D clipping of
  the segment parameter, exact).
* pair_formula(p, x, rows, pairs): the gate/pair characterisation of the segment lemma with ZERO-INCLUSIVE gates
  (f_i(p) >= 0 => f_i(x) <= 0) and the given pairs (i, j) with f_i(p) > 0 > f_j(p).
* edge_pairs(rows): row pairs that share at least two distinct exact vertices (the certify.scene ADJ rule),
  computed here independently.
* exact_vertex(PL, ids): exact intersection point of three binary64 plane rows read as rationals.
* box_rows, window_cls: exact rows of axis boxes and the exact observation of a window for a given source."""
from fractions import Fraction as Q
import itertools


def fr(v):
    return v if isinstance(v, Q) else Q(float(v)) if not isinstance(v, int) else Q(v)


def ev(r, x):
    return r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3]


def seg_hits(s, x, rows):
    """Closed segment [s, x] meets the closed polytope {y : ev(r, y) <= 0 for r in rows}."""
    lo, hi = Q(0), Q(1)
    for r in rows:
        fs = ev(r, s); fx = ev(r, x)
        a = fx - fs                      # f(s + t (x - s)) = fs + t a
        if a == 0:
            if fs > 0:
                return False
        elif a > 0:
            hi = min(hi, -fs / a)
        else:
            lo = max(lo, -fs / a)
        if lo > hi:
            return False
    return lo <= hi


def hits_any(s, x, comps):
    return any(seg_hits(s, x, rows) for rows in comps)


def inside(x, rows):
    return all(ev(r, x) <= 0 for r in rows)


def det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def solve3(rows3):
    """Exact common point of three planes (rows a.x + d = 0), None if singular."""
    A = [list(r[:3]) for r in rows3]; b = [-r[3] for r in rows3]
    d = det3(A)
    if d == 0:
        return None
    out = []
    for k in range(3):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][k] = b[i]
        out.append(det3(M) / d)
    return tuple(out)


def exact_vertex(PL, ids):
    return solve3([tuple(Q(float(PL[i, k])) for k in range(4)) for i in ids])


def vertices(rows):
    """Exact vertices of {ev(r, .) <= 0}: dict point -> frozenset of rows tight at it."""
    pts = {}
    for i, j, k in itertools.combinations(range(len(rows)), 3):
        x = solve3([rows[i], rows[j], rows[k]])
        if x is None or x in pts:
            continue
        if inside(x, rows):
            pts[x] = frozenset(t for t, r in enumerate(rows) if ev(r, x) == 0)
    return pts


def edge_pairs(rows):
    """Ordered row pairs (i, j), i != j, tight together at >= 2 distinct vertices (independent of certify.scene)."""
    V = vertices(rows)
    out = set()
    for i, j in itertools.permutations(range(len(rows)), 2):
        if sum(1 for s in V.values() if i in s and j in s) >= 2:
            out.add((i, j))
    return out


def pair_formula(p, x, rows, pairs=None):
    """Gate/pair characterisation of 'closed [p, x] meets the polytope' for p outside it. Zero-inclusive gates:
    f_i(p) >= 0 => f_i(x) <= 0. Pairs: (i, j) in `pairs` (all pairs if None) with f_i(p) > 0 > f_j(p):
    B_ij = f_i(p) f_j(x) - f_j(p) f_i(x) <= 0."""
    a = [ev(r, p) for r in rows]; b = [ev(r, x) for r in rows]
    n = len(rows)
    for i in range(n):
        if a[i] >= 0 and b[i] > 0:
            return False
    it = itertools.permutations(range(n), 2) if pairs is None else pairs
    for i, j in it:
        if a[i] > 0 > a[j] and a[i] * b[j] - a[j] * b[i] > 0:
            return False
    return True


def normalise(rows):
    out = []
    for r in rows:
        r = tuple(fr(v) for v in r)
        m = max(abs(v) for v in r)
        out.append(tuple(v / m for v in r))
    return tuple(out)


def box_rows(lo, hi):
    """Exact rows of the axis box [lo, hi] (Fractions), normalised by max |coefficient|."""
    rows = []
    for k in range(3):
        e = [Q(0)] * 3; e[k] = Q(1)
        rows.append((*e, -fr(hi[k])))
        rows.append((*[-v for v in e], fr(lo[k])))
    return normalise(rows)


class Comp:
    def __init__(self, rows):
        self.rows = rows


def receiver(o, t, N):
    return tuple(Q(2 * (o[k] + t[k]) + 1, 2 * N) for k in range(3))


def window_cls(comps, s, o, N, L):
    """Exact observation [x, y, z] of the window for source s: 2 SOLID (receiver in the closed solid), 0 dark, 1 lit."""
    import numpy as np
    cls = np.zeros((L, L, L), np.uint8)
    for t in itertools.product(range(L), repeat=3):
        r = receiver(o, t, N)
        if any(inside(r, rows) for rows in comps):
            cls[t] = 2
        else:
            cls[t] = 0 if hits_any(s, r, comps) else 1
    return cls
