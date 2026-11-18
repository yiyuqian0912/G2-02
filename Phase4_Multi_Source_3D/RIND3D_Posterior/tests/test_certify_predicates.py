"""Soundness of the verifier's visibility predicates (rind3d_posterior.certify.check) against an exact rational oracle.

For random convex occluders (rotated boxes and random half-space polytopes), random source boxes (exact kernel
polytopes) and random receiver blocks / points, every COVER verdict must give "closed segment meets the closed body" and
every NOCOVER verdict "misses" at exactly evaluated sample pairs (source vertices, interior points; block corners and
interior receivers). UNDEC is always allowed. Also checks the exact identity behind derived pencil facts."""
from fractions import Fraction as Q
import itertools

import numpy as np
import pytest

from rind3d_posterior.certify import poly, check, scene as SC
from rind3d_posterior.certify.verify import Planes, Slots


class _Comp:
    def __init__(self, rows):
        self.rows = rows


def seg_hits(s, x, rows):
    lo, hi = Q(0), Q(1)
    for r in rows:
        fs = r[0] * s[0] + r[1] * s[1] + r[2] * s[2] + r[3]; fx = r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3]
        a = fx - fs
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


def random_body(rng):
    c = rng.uniform(0.35, 0.65, 3)
    if rng.random() < 0.5:
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        h = rng.uniform(0.03, 0.12, 3)
        rows = []
        for k in range(3):
            for sg in (1, -1):
                n = sg * R[:, k]
                rows.append((*n, -float(n @ c) - h[k]))
    else:
        rows = []
        for _ in range(int(rng.integers(5, 9))):
            n = rng.standard_normal(3); n /= np.linalg.norm(n)
            rows.append((*n, -float(n @ c) - rng.uniform(0.04, 0.1)))
    return tuple(tuple(Q(float(v)) for v in r) for r in rows)


def build(rng):
    body = random_body(rng)
    try:
        sc = SC.ExactScene([_Comp(body)])
    except ValueError:
        return None
    return sc, body


@pytest.mark.parametrize("seed", range(60))
def test_comp_status_sound(seed):
    rng = np.random.default_rng(seed)
    b = build(rng)
    if b is None:
        pytest.skip("degenerate body")
    sc, body = b
    # source box (exact dyadic corners) and receiver block on random sides of the body
    P = Planes(1 << 10); sl = Slots(4)
    s0 = rng.uniform(0.02, 0.9, 3); sz = rng.uniform(0.002, 0.08)
    lo = np.round(s0 * 1024) / 1024; hi = lo + np.ceil(sz * 1024) / 1024
    ids = np.array([[P.axis_plane(k, float(lo[k])), P.axis_plane(k, float(hi[k]))] for k in range(3)])
    poly.set_box(P.PL, 0, ids, *sl.a)
    Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF = sl.a
    C = 1; kmax = sc.RM.shape[1]; Kc = int(sc.K[0])
    fvm = np.zeros((kmax, poly.MV)); fvr = np.zeros((kmax, poly.MV)); fRlo = np.zeros(kmax); fRhi = np.zeros(kmax)
    check.region_face_values(0, 0, sc.K, sc.RM, sc.FB, sc.RR, Vm, Vr, nV, fRlo, fRhi, fvm, fvr)
    eff = np.array([1 if fRlo[i] > 0 else (-1 if fRhi[i] < 0 else 0) for i in range(Kc)] + [0] * (kmax - Kc), np.int64)
    Tlo = np.zeros((kmax, kmax)); Thi = np.zeros((kmax, kmax))
    check.region_tau_table(int(nV[0]), Kc, eff, fvm, fvr, Tlo, Thi)
    N = 1024
    xo = rng.integers(0, N - 64, 3); lvl = int(rng.integers(0, 4)); sz_ = 1 << lvl
    qlo = (xo + 0.5) / N; qhi = (xo + sz_ - 0.5) / N
    fqm = np.zeros((kmax, 8)); fqr = np.zeros((kmax, 8)); fql = np.zeros(kmax); fqh = np.zeros(kmax)
    nc = check.box_face_values(qlo, qhi, lvl == 0, 0, Kc, sc.RM, sc.RL, sc.RR2, fqm, fqr, fql, fqh)
    Rlo = np.array(lo); Rhi = np.array(hi)
    z = np.zeros(1, np.int64); zf = np.zeros((1, 2))
    st, _ = check.comp_status(0, Kc, eff, fRlo, fRhi, Tlo, Thi, nc, fqm, fqr, fql, fqh, Rlo, Rhi, qlo, qhi, sc.aabb_lo,
                              sc.aabb_hi, int(nV[0]), fvm, fvr, np.zeros(8, np.int64), z, z, z, z, z, 0, zf, zf)
    if st == check.UNDEC:
        return
    # exact samples: source box corners + random interior points; receiver block corners + random interior receivers
    S = [tuple(Q(float(v)) for v in Vm[0, v_]) for v_ in range(int(nV[0]))]
    for _ in range(12):
        S.append(tuple(Q(float(lo[k])) + Q(float(rng.random())) * (Q(float(hi[k])) - Q(float(lo[k]))) for k in range(3)))
    X = []
    for cr in range(8 if lvl else 1):
        X.append(tuple(Q(float(qhi[k] if cr >> k & 1 else qlo[k])) for k in range(3)))
    for _ in range(4):
        X.append(tuple(Q(int(xo[k]) + int(rng.integers(0, sz_)), N) + Q(1, 2 * N) for k in range(3)))
    for s in S:
        for x in X:
            h = seg_hits(s, x, body)
            if st == check.COVER:
                assert h, ("COVER claimed but the segment misses", s, x)
            else:
                # without symbolic facts every NOCOVER proof is strict on the closed region (no contact either)
                assert not h, ("NOCOVER claimed but the segment meets the body", s, x)


def test_derived_fact_identity():
    """f_i(y) B_ij(x, s) = f_i(x) B_ij(y, s) + B_ij(x, y) f_i(s) for random exact rationals (the derived-sign lemma)."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        fi_x, fj_x, fi_y, fj_y, fi_s, fj_s = (Q(int(v), 97) for v in rng.integers(-500, 500, 6))
        B = lambda ai, aj, bi, bj: ai * bj - aj * bi
        assert fi_y * B(fi_x, fj_x, fi_s, fj_s) == fi_x * B(fi_y, fj_y, fi_s, fj_s) + B(fi_x, fj_x, fi_y, fj_y) * fi_s


def test_face_at_point_double_double():
    """Receiver face values are enclosed with error relative to |f(x)| even under cancellation."""
    rng = np.random.default_rng(9)
    for _ in range(200):
        body = random_body(rng)
        try:
            sc = SC.ExactScene([_Comp(body)])
        except ValueError:
            continue
        x = (rng.integers(0, 1024, 3) + 0.5) / 1024
        for i in range(int(sc.K[0])):
            m, r = check.face_at_point(0, i, x[0], x[1], x[2], sc.RM, sc.RL, sc.RR2)
            ex = sum(body[i][k] * Q(float(x[k])) for k in range(3)) + body[i][3]
            assert abs(Q(m) - ex) <= Q(r)
            assert r <= 1e-15 * abs(float(ex)) + 1e-30
