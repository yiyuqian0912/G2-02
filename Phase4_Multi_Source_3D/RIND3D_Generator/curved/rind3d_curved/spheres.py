"""Exact and certified predicates for closed balls.

Closed segment [s, x] versus closed ball B(c, r), with w = s - c, d = x - s:
    A = d.d,  B = w.d,  C = w.w - r^2,  E = (x - c).(x - c) - r^2 = A + 2B + C.
    f(t) = |s + t d - c|^2 - r^2 = A t^2 + 2 B t + C,  t in [0, 1].
    hit  iff  min f <= 0  iff  C <= 0  or  E <= 0  or  (B < 0 and A + B > 0 and A C - B^2 <= 0).
(The last case is the interior minimum t* = -B/A in (0, 1), with min f = C - B^2/A, multiplied by A > 0.)
The test is division-free. With exact rationals it is exact; with outward-rounded binary64 intervals each sign is
either certified or UNKNOWN, and UNKNOWN cases are recomputed exactly. Tangency (min f = 0) is a hit, as in the
generator's closed-segment policy.

Membership: x in B iff |x - c|^2 - r^2 <= 0. Clearance of a source: dist(s, B) > rho iff |s - c|^2 > (r + rho)^2."""
from fractions import Fraction

import numpy as np

Q = Fraction
INF = np.inf


def _q(v):
    return Q(float(v))


def ball_contains_exact(c, r, x):
    c = [_q(v) for v in c]; x = [_q(v) for v in x]; r = _q(r)
    return sum((x[i] - c[i]) ** 2 for i in range(3)) - r * r <= 0


def segment_hits_ball_exact(c, r, s, x):
    c = [_q(v) for v in c]; s = [_q(v) for v in s]; x = [_q(v) for v in x]; r = _q(r)
    w = [s[i] - c[i] for i in range(3)]; d = [x[i] - s[i] for i in range(3)]
    A = sum(v * v for v in d); B = sum(w[i] * d[i] for i in range(3)); C = sum(v * v for v in w) - r * r
    E = A + 2 * B + C
    return C <= 0 or E <= 0 or (B < 0 and A + B > 0 and A * C - B * B <= 0)


def clearance_exact(c, r, s, rho):
    c = [_q(v) for v in c]; s = [_q(v) for v in s]; r = _q(r); rho = Q(rho)
    return sum((s[i] - c[i]) ** 2 for i in range(3)) > (r + rho) ** 2


# ------------------------------------------------------------------ outward-rounded interval arithmetic (numpy)
def _dn(a):
    return np.nextafter(a, -INF)


def _up(a):
    return np.nextafter(a, INF)


def iadd(a, b):
    return _dn(a[0] + b[0]), _up(a[1] + b[1])


def isub(a, b):
    return _dn(a[0] - b[1]), _up(a[1] - b[0])


def imul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _dn(np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))), _up(np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3])))


def isq(a):
    lo, hi = imul(a, a)
    return np.where((a[0] <= 0) & (a[1] >= 0), 0.0, lo), hi


def ipt(v):
    v = np.asarray(v, float)
    return v, v


def _tri_le0(iv):
    """Three-valued 'value <= 0': 1 certain, 0 certainly false, -1 unknown."""
    return np.where(iv[1] <= 0, 1, np.where(iv[0] > 0, 0, -1))


def _tri_lt0(iv):
    return np.where(iv[1] < 0, 1, np.where(iv[0] >= 0, 0, -1))


def _tri_gt0(iv):
    return np.where(iv[0] > 0, 1, np.where(iv[1] <= 0, 0, -1))


def _and(*ts):
    out = np.ones_like(ts[0])
    for t in ts:
        out = np.where((out == 0) | (t == 0), 0, np.where((out == 1) & (t == 1), 1, -1))
    return out


def _or(*ts):
    out = np.zeros_like(ts[0])
    for t in ts:
        out = np.where((out == 1) | (t == 1), 1, np.where((out == 0) & (t == 0), 0, -1))
    return out


def segment_hits_ball_interval(c, r, s, X):
    """Three-valued hit (1 / 0 / -1 unknown) of [s, X_i] with B(c, r) for receivers X (n, 3), certified intervals."""
    X = np.asarray(X, float)
    s = np.asarray(s, float); c = np.asarray(c, float)
    w = [isub(ipt(s[i]), ipt(c[i])) for i in range(3)]
    d = [isub(ipt(X[:, i]), ipt(s[i])) for i in range(3)]
    A = iadd(iadd(isq(d[0]), isq(d[1])), isq(d[2]))
    B = iadd(iadd(imul(w[0], d[0]), imul(w[1], d[1])), imul(w[2], d[2]))
    rr = imul(ipt(float(r)), ipt(float(r)))
    C = isub(iadd(iadd(isq(w[0]), isq(w[1])), isq(w[2])), rr)
    C = (np.broadcast_to(C[0], A[0].shape), np.broadcast_to(C[1], A[0].shape))
    e = [isub(ipt(X[:, i]), ipt(c[i])) for i in range(3)]
    E = isub(iadd(iadd(isq(e[0]), isq(e[1])), isq(e[2])), rr)
    ApB = iadd(A, B)
    D = isub(imul(A, C), isq(B))
    return _or(_tri_le0(C), _tri_le0(E), _and(_tri_lt0(B), _tri_gt0(ApB), _tri_le0(D)))


def ball_contains_interval(c, r, X):
    X = np.asarray(X, float); c = np.asarray(c, float)
    e = [isub(ipt(X[:, i]), ipt(c[i])) for i in range(3)]
    rr = imul(ipt(float(r)), ipt(float(r)))
    return _tri_le0(isub(iadd(iadd(isq(e[0]), isq(e[1])), isq(e[2])), rr))


class BallSet:
    """A finite union of closed balls with exact decisions. centres (n, 3), radii (n,) binary64."""

    def __init__(self, centres, radii):
        self.c = np.asarray(centres, float).reshape(-1, 3); self.r = np.asarray(radii, float).reshape(-1)
        if np.any(self.r <= 0):
            raise ValueError("radii must be positive")
        self.stats = dict(interval=0, exact=0)

    def __len__(self):
        return len(self.r)

    def contains(self, X):
        """Exact SOLID-by-balls for points X (n, 3)."""
        X = np.asarray(X, float); out = np.zeros(len(X), bool)
        for c, r in zip(self.c, self.r):
            t = ball_contains_interval(c, r, X); self.stats["interval"] += len(X)
            u = np.flatnonzero(t == -1)
            for i in u:
                t[i] = int(ball_contains_exact(c, r, X[i])); self.stats["exact"] += 1
            out |= t == 1
        return out

    def blocked(self, s, X):
        """Exact: closed segment [s, X_i] meets some ball (n,) bool."""
        X = np.asarray(X, float); out = np.zeros(len(X), bool)
        for c, r in zip(self.c, self.r):
            todo = np.flatnonzero(~out)
            if not len(todo):
                break
            t = segment_hits_ball_interval(c, r, s, X[todo]); self.stats["interval"] += len(todo)
            u = np.flatnonzero(t == -1)
            for i in u:
                t[i] = int(segment_hits_ball_exact(c, r, s, X[todo[i]])); self.stats["exact"] += 1
            out[todo[t == 1]] = True
        return out

    def legal_source(self, s, rho=2.0 ** -26):
        """Outside every closed ball with clearance > rho (exact)."""
        return all(clearance_exact(c, r, s, rho) for c, r in zip(self.c, self.r))

    def to_array(self):
        return np.c_[self.c, self.r].astype("<f8")


def ball_misses_hull_exact(c, r, points):
    """Certified: the closed ball B(c, r) misses conv(points) (e.g. a source and the 8 corners of a receiver box).
    A separating direction n = (c - q) / |c - q| is found in float (q ~ the closest hull point, by Frank-Wolfe); the
    bound min_i n.(c - p_i) > r is then verified exactly with rationals for n's binary64 value (any n gives a valid
    lower bound on the distance after scaling by |n|, so the float search only affects whether a proof is found)."""
    P = np.asarray(points, float); c = np.asarray(c, float)
    lam = np.full(len(P), 1.0 / len(P))
    for it in range(64):
        q = lam @ P; g = q - c
        i = int(np.argmin(P @ g)); dvec = P[i] - q
        den = dvec @ dvec
        if den <= 0:
            break
        step = float(np.clip(-(g @ dvec) / den, 0, 1))
        lam *= (1 - step); lam[i] += step
    q = lam @ P; n = c - q
    if not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0:
        return False
    nq = [_q(v) for v in n]; cq = [_q(v) for v in c]; rq = _q(r)
    nn = sum(v * v for v in nq)
    m = min(sum(nq[k] * (cq[k] - _q(p[k])) for k in range(3)) for p in P)     # min over hull vertices of n.(c - p)
    return m > 0 and m * m > rq * rq * nn                                     # dist(c, hull) >= m / |n| > r
