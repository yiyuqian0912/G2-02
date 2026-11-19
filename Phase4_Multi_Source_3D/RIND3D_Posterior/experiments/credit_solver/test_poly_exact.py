"""Exact differential tests of the polytope primitive (poly.Poly) and its structural check.

Reference: every piece is also tracked as an explicit half-space list (the cube's six rows
plus the signed cut rows). Its vertex set is enumerated independently by solving every
triple of rows exactly (Cramer) and keeping the points that satisfy all rows. Poly.split
must reproduce exactly that vertex set, its face structure must pass polycheck.check, the
two halves must conserve the parent volume exactly, and the fan volume must match an
independent floating-point hull volume. Corrupted face structures must be rejected.

Run from this directory:
  PYTHONPATH=".:../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K" python -m pytest -q test_poly_exact.py
"""
from fractions import Fraction
from itertools import combinations
from math import gcd
from pathlib import Path
import random

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import inputs  # noqa: F401  (research paths)
import polycheck
from poly import Poly, form, primitive, within, without
from solver import cone

CUBE_ROWS = [(-1, 0, 0, 0), (1, 0, 0, -1), (0, -1, 0, 0), (0, 1, 0, -1), (0, 0, -1, 0), (0, 0, 1, -1)]
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3")


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def reference_vertices(rows):
    """All vertices of {x : row(x) <= 0 for all rows}, primitive homogeneous, exact."""
    out = set()
    for r1, r2, r3 in combinations(rows, 3):
        A = [r1[:3], r2[:3], r3[:3]]
        b = [-r1[3], -r2[3], -r3[3]]
        D = det3(A)
        if D == 0:
            continue
        xs = []
        for c in range(3):
            M = [list(row) for row in A]
            for i in range(3):
                M[i][c] = b[i]
            xs.append(det3(M))
        v = (*xs, D) if D > 0 else (*(-x for x in xs), -D)
        if all(form(r, v) <= 0 for r in rows):
            out.add(primitive(v))
    return out


def float_volume(vertices):
    pts = np.array([[v[i] / v[3] for i in range(3)] for v in vertices])
    return ConvexHull(pts).volume


def assert_exact(p, rows):
    polycheck.check(p)
    assert set(p.vertices) == reference_vertices(rows)
    fv = float_volume(p.vertices)
    assert abs(float(p.volume) - fv) <= 1e-9 * max(fv, 1e-30)


def random_row(rng, kind):
    if kind == "small":  # many planes through vertices, edges and faces (degenerate cases)
        while True:
            r = tuple(rng.randint(-3, 3) for _ in range(3)) + (rng.randint(-3, 3),)
            if any(r[:3]):
                return r
    if kind == "dyadic":  # axis planes at dyadic positions (the octree and the search)
        axis, m = rng.randrange(3), rng.randint(1, 40)
        c = Fraction(rng.randrange(1, 1 << m), 1 << m)
        r = [0, 0, 0, -c.numerator]
        r[axis] = c.denominator
        return tuple(r)
    # "big": random wide coefficients (large vertex bit sizes)
    while True:
        r = tuple(rng.randint(-(1 << 60), 1 << 60) for _ in range(3))
        x = [Fraction(rng.randrange(1, 1 << 30), 1 << 30) for _ in range(3)]
        d = -sum(a * b for a, b in zip(r, x))
        den = d.denominator
        r = tuple(a * den for a in r) + (d.numerator,)
        if any(r[:3]):
            return primitive(r)


@pytest.mark.parametrize("kind,seed", [(k, s) for k in ("small", "dyadic", "big") for s in range(4)])
def test_split_matches_halfspace_reference(kind, seed):
    rng = random.Random(1000 * seed + ("small", "dyadic", "big").index(kind))  # deterministic
    pieces = [(Poly.cube(), list(CUBE_ROWS))]
    done = 0
    for _ in range(400):
        if done >= 40:
            break
        i = rng.randrange(len(pieces))
        P, rows = pieces[i]
        if len(rows) > 16:
            continue
        r = random_row(rng, kind)
        neg, pos = P.split(r)
        if neg is None or pos is None:
            # the plane misses the interior: every vertex on one closed side
            vals = [form(r, v) for v in P.vertices]
            assert (max(vals) <= 0) if pos is None else (min(vals) >= 0)
            continue
        assert neg.volume + pos.volume == P.volume
        assert_exact(neg, rows + [r])
        assert_exact(pos, rows + [tuple(-x for x in r)])
        pieces[i:i + 1] = [(neg, rows + [r]), (pos, rows + [tuple(-x for x in r)])]
        done += 1
    assert done >= 10


def real_scene():
    path = ROOT / "scenes" / "0000" / "000030.npz"
    if not path.exists():
        pytest.skip("v3 canary data not available")
    return inputs.load(path, 1, 0)[0]


def test_cone_cuts_match_reference():
    """within/without with exact inverse cones of a real scene (large integers)."""
    obs = real_scene()
    rng = random.Random(7)
    L = obs.side
    tested = 0
    for _ in range(40):
        pos = tuple(rng.randrange(L) for _ in range(3))
        if obs.cls[pos] == 2:
            continue
        r = obs.world(pos)
        body = obs.bodies[rng.randrange(len(obs.bodies))]
        try:
            rows_c = cone(body, r)
        except ValueError:
            continue
        P = Poly.cube()
        inside = within(P, rows_c)
        if inside is not None:
            assert_exact(inside, CUBE_ROWS + list(rows_c))
            tested += 1
        outside = without(P, rows_c)
        total = sum((q.volume for q in outside), Fraction(0)) + (inside.volume if inside is not None else 0)
        assert total == P.volume  # the difference and the intersection partition the cube
        for q in outside:
            polycheck.check(q)
    assert tested >= 5


def test_subnormal_coordinates_use_exact_signs():
    """Vertices with coordinates below 2^-1022 are classified exactly (no float filter)."""
    tiny = Fraction(1, 1 << 1100)
    P = within(Poly.cube(), [(tiny.denominator, 0, 0, -tiny.numerator)])  # x <= 2^-1100
    assert P is not None
    polycheck.check(P)
    assert any(fv is None for fv in P.filter_vertices)
    # a row whose float evaluation at those vertices would be dominated by rounding
    big = (1 << 1000) * 3
    row = (big, 0, 0, -1)  # 3*2^1000 * x - 1 <= 0  <=>  x <= 2^-1000 / 3 (true for all of P)
    assert P.side(row) == -1
    row2 = (big << 200, 0, 0, -1)  # x <= 2^-1200/3: cuts P
    neg, pos = P.split(row2)
    assert neg is not None and pos is not None and neg.volume + pos.volume == P.volume


def corrupt_variants(p):
    faces = [list(f) for f in p.faces]
    yield "drop a face", faces[1:]
    yield "duplicate a face", faces + [faces[0]]
    f = list(faces[0])
    if len(f) >= 4:
        f[1], f[2] = f[2], f[1]
        yield "non-convex order", [f] + faces[1:]
    g = list(faces[0])
    a, b = g[0], g[1]
    mid = primitive(tuple(x * b[3] + y * a[3] for x, y in zip(a, b)))  # midpoint of an edge
    g.insert(1, mid)
    yield "collinear vertex", [g] + faces[1:]
    yield "reversed face only", [list(reversed(faces[0]))] + faces[1:]  # harmless: orientation is free


def test_check_rejects_corrupted_structures():
    rng = random.Random(3)
    P = Poly.cube()
    for _ in range(6):
        neg, pos = P.split(random_row(rng, "small"))
        P = neg if neg is not None and (pos is None or neg.volume >= pos.volume) else pos
    polycheck.check(P)
    for name, faces in corrupt_variants(P):
        Q = Poly([tuple(f) for f in faces])
        if name == "reversed face only":
            polycheck.check(Q)
            continue
        with pytest.raises(ArithmeticError):
            polycheck.check(Q)
