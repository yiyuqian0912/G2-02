"""Independent exact membership oracle for one source point (research reference, not in the TCB).

s in F  iff  s is in the open unit cube, outside the closed window box, at Euclidean
distance > 2^-26 from every obstacle, and for every FREE receiver r the closed segment
[s, r] meets some obstacle exactly when r is inactive.

Visibility: vectorized binary64 parametric clipping with an explicit margin; every
receiver whose decision is within the margin is re-decided with exact rationals.
Distance: exact squared distance to each obstacle polytope from its vertices, edges and
facets (no shared code with the kernel's clearance covers).
Independent of the solver: it shares only the integer obstacle rows and the observation.
"""
from fractions import Fraction
from itertools import product

import numpy as np

from poly import Poly, within

EPS2 = Fraction(1, 1 << 52)  # (2^-26)^2


def body_polys(bodies):
    return [within(Poly.cube(), b) for b in bodies]


def _pt(v):
    return tuple(Fraction(v[i], v[3]) for i in range(3))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def dist2_point_poly(s, poly, rows):
    """Exact squared distance from rational point s to the convex polytope (0 if inside)."""
    if all(sum(r[i] * s[i] for i in range(3)) + r[3] <= 0 for r in rows):
        return Fraction(0)
    best = None
    verts = [_pt(v) for v in poly.vertices]
    for v in verts:
        d = _sub(s, v)
        x = _dot(d, d)
        best = x if best is None or x < best else best
    for face in poly.faces:
        f = [_pt(v) for v in face]
        for a, b in zip(f, f[1:] + f[:1]):
            e = _sub(b, a)
            ee = _dot(e, e)
            if ee == 0:
                continue
            t = _dot(_sub(s, a), e) / ee
            if 0 < t < 1:
                q = tuple(ai + t * ei for ai, ei in zip(a, e))
                d = _sub(s, q)
                best = min(best, _dot(d, d))
    for r in rows:  # facet planes: projection inside the facet polygon
        n = r[:3]
        nn = sum(x * x for x in n)
        val = sum(n[i] * s[i] for i in range(3)) + r[3]
        if val <= 0:
            continue
        q = tuple(s[i] - val * n[i] / nn for i in range(3))
        if all(sum(rr[i] * q[i] for i in range(3)) + rr[3] <= 0 for rr in rows):
            best = min(best, val * val / nn)
    return best


def legal(s, bodies, polys, origin, side, n=1024):
    if not all(0 < x < 1 for x in s):
        return False
    lo = [Fraction(o, n) for o in origin]
    hi = [Fraction(o + side, n) for o in origin]
    if all(lo[i] <= s[i] <= hi[i] for i in range(3)):
        return False
    return all(dist2_point_poly(s, p, b) > EPS2 for p, b in zip(polys, bodies))


def _exact_hit(rows, s, r):
    lo, hi = Fraction(0), Fraction(1)
    for row in rows:
        fs = sum(row[i] * s[i] for i in range(3)) + row[3]
        fr = sum(row[i] * r[i] for i in range(3)) + row[3]
        d = fr - fs
        if d == 0:
            if fs > 0:
                return False
        elif d > 0:
            hi = min(hi, -fs / d)
        else:
            lo = max(lo, -fs / d)
        if lo > hi:
            return False
    return True


class Oracle:
    def __init__(self, bodies, cls, origin, n=1024):
        self.bodies = bodies
        self.polys = body_polys(bodies)
        self.cls = cls
        self.origin = origin
        self.side = cls.shape[0]
        self.n = n
        free = np.argwhere(cls != 2)
        self.free = free
        self.label = cls[tuple(free.T)].astype(np.int8)
        self.R = (np.asarray(origin)[None, :] + free + 0.5) / n
        self.rows = [np.array([[float(x) for x in r] for r in b]) for b in bodies]
        self.solid = np.argwhere(cls == 2)

    def visible_float(self, s):
        """Per FREE receiver: +1 visible, 0 blocked, -1 undecided by the float filter."""
        M = len(self.R)
        hit = np.zeros(M, bool)
        unsure = np.zeros(M, bool)
        for rows in self.rows:
            nrm = np.abs(rows[:, :3]).sum(1) + np.abs(rows[:, 3])
            fs = rows[:, :3] @ s + rows[:, 3]
            fr = self.R @ rows[:, :3].T + rows[:, 3]
            d = fr - fs[None, :]
            tol = 1e-9 * nrm[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -fs[None, :] / d
            pos = d > tol
            neg = d < -tol
            zero = ~pos & ~neg
            lo = np.max(np.where(neg, t, 0.0), axis=1, initial=0.0)
            hi = np.min(np.where(pos, t, 1.0), axis=1, initial=1.0)
            miss_const = np.any(zero & (fs[None, :] > tol), axis=1)
            near_zero = np.any(zero & (np.abs(fs)[None, :] <= tol), axis=1)
            h = (lo <= hi) & ~miss_const
            gap = np.abs(hi - lo)
            u = (gap < 1e-9) | near_zero | (np.abs(lo) < 1e-12) | (np.abs(hi - 1) < 1e-12)
            hit |= h & ~u
            unsure |= u
        out = np.where(hit, 0, 1).astype(np.int8)
        out[unsure & ~hit] = -1
        return out

    def feasible(self, s_exact):
        """Exact decision. Returns (legal, mismatches, exact_rechecks)."""
        if not legal(s_exact, self.bodies, self.polys, self.origin, self.side, self.n):
            return False, None, 0
        s = np.array([float(x) for x in s_exact])
        v = self.visible_float(s)
        rechecks = 0
        for k in np.flatnonzero(v < 0):
            r = tuple(Fraction(2 * (self.origin[i] + int(self.free[k][i])) + 1, 2 * self.n) for i in range(3))
            v[k] = 0 if any(_exact_hit(b, s_exact, r) for b in self.bodies) else 1
            rechecks += 1
        mism = int(np.count_nonzero(v != self.label))
        return True, mism, rechecks
