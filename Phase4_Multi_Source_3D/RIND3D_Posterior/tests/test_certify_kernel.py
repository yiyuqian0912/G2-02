"""Exact-kernel tests of the certificate verifier (rind3d_posterior.certify.ar / .poly) against an independent exact
rational reference (Fractions: vertex enumeration over all plane triples, exact facet ordering, cone volumes)."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest

from rind3d_posterior.certify import ar, poly


def exact_volume(rows):
    """Exact volume of {x : a.x + b <= 0 for all rows} (rows of Fractions), assumed bounded. Proportional rows
    (same half-space) are merged so that each facet is counted once."""
    canon = {}
    for r in rows:
        m = max(abs(v) for v in r[:3])
        canon[tuple(v / m for v in r)] = r
    rows = list(canon.values())
    pts = set()
    n = len(rows)
    for i, j, k in itertools.combinations(range(n), 3):
        A = [rows[t][:3] for t in (i, j, k)]; b = [-rows[t][3] for t in (i, j, k)]
        det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        if det == 0:
            continue
        x = []
        for c in range(3):
            M = [list(r) for r in A]
            for r in range(3):
                M[r][c] = b[r]
            dc = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                  + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
            x.append(dc / det)
        if all(r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3] <= 0 for r in rows):
            pts.add(tuple(x))
    pts = list(pts)
    if len(pts) < 4:
        return Q(0)
    c = tuple(sum(p[t] for p in pts) / len(pts) for t in range(3))
    vol = Q(0)
    for r in rows:
        F = [p for p in pts if r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3] == 0]
        if len(F) < 3:
            continue
        g = tuple(sum(p[t] for p in F) / len(F) for t in range(3))
        nrm = r[:3]
        # exact cyclic order around g in the facet plane: sort by (half-plane, cross-product) comparator
        u0 = [F[0][t] - g[t] for t in range(3)]
        if all(v == 0 for v in u0):
            u0 = [F[1][t] - g[t] for t in range(3)]
        w0 = [nrm[1] * u0[2] - nrm[2] * u0[1], nrm[2] * u0[0] - nrm[0] * u0[2], nrm[0] * u0[1] - nrm[1] * u0[0]]

        def key(p):
            d = [p[t] - g[t] for t in range(3)]
            x = sum(d[t] * u0[t] for t in range(3)); y = sum(d[t] * w0[t] for t in range(3))
            return (x, y)

        import functools

        def half(xy):
            x, y = xy
            return 0 if (y > 0 or (y == 0 and x > 0)) else 1

        def cmp(p, q):
            a, b = key(p), key(q)
            ha, hb = half(a), half(b)
            if ha != hb:
                return ha - hb
            cr = a[0] * b[1] - a[1] * b[0]
            return -1 if cr > 0 else (1 if cr < 0 else 0)

        F = sorted(F, key=functools.cmp_to_key(cmp))
        for t in range(1, len(F) - 1):
            a = [F[0][s] - c[s] for s in range(3)]; b = [F[t][s] - c[s] for s in range(3)]; d = [F[t + 1][s] - c[s] for s in range(3)]
            vol += abs(a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) + a[2] * (b[0] * d[1] - b[1] * d[0]))
    return vol / 6


class Kernel:
    def __init__(self, S=8, NP=4096):
        self.PL = np.zeros((NP, 4)); self.n = 0; self.slots = poly.new_slots(S)

    def plane(self, a, b):
        self.PL[self.n] = (*a, b); self.n += 1
        return self.n - 1

    def args(self):
        d = self.slots
        return d["Fp"], d["Fn"], d["Fv"], d["Fe"], d["Vp"], d["Vm"], d["Vr"], d["nV"], d["nF"]

    def box(self, s, lo, hi):
        ids = np.zeros((3, 2), np.int64)
        for k in range(3):
            e = [0.0, 0.0, 0.0]; e[k] = 1.0
            ids[k, 0] = self.plane(e, -lo[k]); ids[k, 1] = self.plane(e, -hi[k])
        poly.set_box(self.PL, s, ids, *self.args())
        rows = []
        for k in range(3):
            e = [Q(0)] * 3; e[k] = Q(1)
            rows.append((*e, -Q(hi[k])))
            e2 = [Q(0)] * 3; e2[k] = Q(-1)
            rows.append((*e2, Q(lo[k])))
        return rows

    def clip(self, s, pid, neg=False):
        return poly.clip(self.PL, s, 2 * pid + int(neg), *self.args())

    def volume(self, s):
        d = self.slots
        return poly.volume(s, d["Fn"], d["Fv"], d["Vm"], d["Vr"], d["nF"])


def exact_row(PL, pid, neg):
    r = [Q(float(v)) for v in PL[pid]]
    return tuple(-v for v in r) if neg else tuple(r)


def test_selftest():
    assert ar.self_test()


def test_box_volume_exact():
    K = Kernel()
    K.box(0, [0.25, 0.5, 0.125], [0.5, 0.5625, 0.25])
    lo, hi = K.volume(0)
    v = Q(1, 4) * Q(1, 16) * Q(1, 8)
    assert Q(lo) <= v <= Q(hi) and Q(hi) - Q(lo) < v * Q(1, 10 ** 12)


@pytest.mark.parametrize("seed", range(40))
def test_random_clips_enclose_exact_volume(seed):
    rng = np.random.default_rng(seed)
    K = Kernel()
    lo = np.sort(rng.random(3))[:1].repeat(3) * 0 + rng.random(3) * 0.5
    hi = lo + 0.05 + rng.random(3) * 0.4
    rows = K.box(0, lo, hi)
    for t in range(int(rng.integers(1, 9))):
        a = rng.standard_normal(3)
        c = lo + rng.random(3) * (hi - lo)
        b = -float(a @ c)
        pid = K.plane(a, b)
        neg = bool(rng.integers(2))
        st = K.clip(0, pid, neg)
        rows.append(exact_row(K.PL, pid, neg))
        if st == poly.EMPTY:
            assert exact_volume(rows) == 0
            return
        assert st in (poly.OK, poly.NOOP)
    vlo, vhi = K.volume(0)
    ev = exact_volume(rows)
    assert Q(vlo) <= ev <= Q(vhi), (vlo, float(ev), vhi)
    assert float(Q(vhi) - Q(vlo)) <= 1e-10 * max(float(ev), 1e-30) + 1e-30


@pytest.mark.parametrize("seed", range(30))
def test_degenerate_planes_through_vertices_and_coincident(seed):
    """Planes through existing vertices / edges, axis planes on box faces, repeated planes: exact SoS combinatorics."""
    rng = np.random.default_rng(1000 + seed)
    K = Kernel()
    rows = K.box(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    for t in range(6):
        kind = rng.integers(4)
        if kind == 0:     # plane through three box corners (exact dyadic)
            P = rng.integers(0, 2, (3, 3)).astype(float)
            a = np.cross(P[1] - P[0], P[2] - P[0])
            if not np.any(a):
                continue
            b = -float(a @ P[0])
        elif kind == 1:   # axis plane at a dyadic position (possibly a box face)
            k = int(rng.integers(3)); a = np.zeros(3); a[k] = 1.0; b = -float(rng.integers(0, 5)) / 4
        elif kind == 2:   # repeat an earlier plane (same coefficients, new id)
            pid = int(rng.integers(K.n)); a = K.PL[pid, :3].copy(); b = float(K.PL[pid, 3])
        else:             # plane through the centre with small integer normal
            a = rng.integers(-2, 3, 3).astype(float)
            if not np.any(a):
                continue
            b = -float(a @ np.array([0.5, 0.5, 0.5]))
        pid = K.plane(a, b); neg = bool(rng.integers(2))
        st = K.clip(0, pid, neg)
        rows.append(exact_row(K.PL, pid, neg))
        if st == poly.EMPTY:
            assert exact_volume(rows) == 0
            return
        assert st in (poly.OK, poly.NOOP)
    vlo, vhi = K.volume(0)
    ev = exact_volume(rows)
    assert Q(vlo) <= ev <= Q(vhi)


@pytest.mark.parametrize("seed", range(20))
def test_split_children_tile_parent(seed):
    rng = np.random.default_rng(2000 + seed)
    K = Kernel()
    K.box(0, [0.1, 0.2, 0.3], [0.6, 0.4, 0.9])
    for t in range(3):
        a = rng.standard_normal(3); pid = K.plane(a, -float(a @ np.array([0.35, 0.3, 0.6])))
        K.clip(0, pid, bool(rng.integers(2)))
    P = K.volume(0)
    a = rng.standard_normal(3); pid = K.plane(a, -float(a @ np.array([0.35, 0.3, 0.6])) + 1e-3 * rng.standard_normal())
    poly.copy_slot(0, 1, *K.args())
    sa = K.clip(0, pid, False); sb = K.clip(1, pid, True)
    A = K.volume(0) if sa != poly.EMPTY else (0.0, 0.0)
    B = K.volume(1) if sb != poly.EMPTY else (0.0, 0.0)
    assert Q(A[0]) + Q(B[0]) <= Q(P[1]) and Q(A[1]) + Q(B[1]) >= Q(P[0])


def test_thin_sliver_volume():
    """Two parallel planes 1e-15 apart inside a cell: the enclosure must contain the exact tiny volume."""
    K = Kernel()
    rows = K.box(0, [0.5, 0.5, 0.5], [0.5078125, 0.5078125, 0.5078125])
    a = np.array([1.0, 0.3, -0.2]); c = 0.5 + 0.0039
    b1 = -float(a @ np.array([c, 0.504, 0.504]))
    p1 = K.plane(a, b1); p2 = K.plane(a, np.nextafter(np.nextafter(b1, 1), 1))     # a.x in [-b1 - 2 ulp, -b1]
    K.clip(0, p1, False); rows.append(exact_row(K.PL, p1, False))
    K.clip(0, p2, True); rows.append(exact_row(K.PL, p2, True))
    vlo, vhi = K.volume(0)
    ev = exact_volume(rows)
    assert ev > 0 and Q(vlo) <= ev <= Q(vhi)


def test_empty_clip_clears_slot_and_cells_tile():
    """Regression (found by the engine cross-check): clipping by a face plane with the opposite orientation must leave
    an EMPTY slot; splitting a box on the 128 grid must give pieces whose volumes sum to the box."""
    K = Kernel()
    ids = np.zeros((3, 2), np.int64)
    for k, (a, b) in enumerate(((10, 11), (20, 21), (30, 32))):
        e = [0.0, 0.0, 0.0]; e[k] = 1.0
        ids[k, 0] = K.plane(e, -a / 128); ids[k, 1] = K.plane(e, -b / 128)
    poly.set_box(K.PL, 0, ids, *K.args())
    poly.copy_slot(0, 1, *K.args())
    st = poly.clip(K.PL, 1, 2 * int(ids[0, 1]) + 1, *K.args())       # {x >= 11/128} on a box with x <= 11/128
    assert st == poly.EMPTY and K.slots["nF"][1] == 0
    assert K.volume(1) == (0.0, 0.0)
    e = [0.0, 0.0, 1.0]; mid = K.plane(e, -31 / 128)
    poly.copy_slot(0, 1, *K.args())
    assert poly.clip(K.PL, 0, 2 * mid, *K.args()) == poly.OK and poly.clip(K.PL, 1, 2 * mid + 1, *K.args()) == poly.OK
    a, b = K.volume(0), K.volume(1)
    v = Q(1, 128 ** 3)
    assert Q(a[0]) <= v <= Q(a[1]) and Q(b[0]) <= v <= Q(b[1])


def test_emit_cell_split_counts_each_cell_once():
    from rind3d_posterior.certify import verify as V, tree
    P = V.Planes(1 << 14)
    gid = np.zeros((3, 129), np.int64)
    for k in range(3):
        for j in range(129):
            gid[k, j] = P.axis_plane(k, j / 128)
    sl = V.Slots(80)
    ids = np.array([[gid[0, 10], gid[0, 12]], [gid[1, 20], gid[1, 21]], [gid[2, 30], gid[2, 33]]])
    poly.set_box(P.PL, 0, ids, *sl.a)
    rc = np.zeros(64, np.int64); rl = np.zeros(64); rh = np.zeros(64); rk = np.zeros(64, np.int8)
    st = np.zeros(tree.NSTATS, np.int64)
    n, ok = tree._emit(P.PL, P.npl, 0, tree.REC_UNDET, 40, 39, gid, *sl.a, rc, rl, rh, rk, 0, 1, np.array([6]),
                       np.zeros((1, 6, 4)), np.zeros((1, 6, 4)), np.zeros((1, 6)), np.full((1, 3), 5.0), np.full((1, 3), 6.0),
                       2.0 ** -26, st)
    assert ok and n == 6 and len(set(rc[:n].tolist())) == 6
    want = {(x * 128 + y) * 128 + z for x in (10, 11) for y in (20,) for z in (30, 31, 32)}
    assert set(rc[:n].tolist()) == want
    assert sum(Q(v) for v in rh[:n]) >= 6 * Q(1, 128 ** 3) > sum(Q(v) for v in rl[:n])
