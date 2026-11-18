"""Soundness of the XLF trusted predicates (rind3d_posterior.xlf.pred) against an exact rational segment oracle.

Random convex occluders, random exact source polytopes (dyadic boxes, optionally cut by a random binary64 plane) and a
small window of exact dyadic receivers. Exact vertices are recomputed in rationals from the plane triples; sample
sources are the vertices and random rational convex combinations of them.
* recv_test: R_DARK => every sample's segment meets the body; R_LIT => none does; R_CONV => for every sample, the
  receiver's observation is reproduced iff all returned halfspaces hold (evaluated EXACTLY from the canonical rows).
* scan_rows: an OUT witness is contradicted at every sample; every receiver not reported ambiguous is reproduced at
  every sample (both with and without the exact re-test of ambiguous receivers)."""
from fractions import Fraction as Q
import itertools

import numpy as np
import pytest

from rind3d_posterior import xlf  # noqa: F401
from rind3d_posterior.certify import poly, scene as SC
from rind3d_posterior.xlf import pred as PR
from rind3d_posterior.xlf.solve import Planes, Slots


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
    c = rng.uniform(0.4, 0.6, 3)
    if rng.random() < 0.5:
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        h = rng.uniform(0.03, 0.1, 3)
        rows = []
        for k in range(3):
            for sg in (1, -1):
                n = sg * R[:, k]
                rows.append((*n, -float(n @ c) - h[k]))
    else:
        rows = []
        for _ in range(int(rng.integers(5, 9))):
            n = rng.standard_normal(3); n /= np.linalg.norm(n)
            rows.append((*n, -float(n @ c) - rng.uniform(0.04, 0.09)))
    m = [max(abs(Q(float(v))) for v in r) for r in rows]
    return tuple(tuple(Q(float(v)) / mm for v in r) for r, mm in zip(rows, m)), c


def exact_vertex(PL, ids):
    A = [[Q(float(PL[i, k])) for k in range(3)] for i in ids]; b = [-Q(float(PL[i, 3])) for i in ids]
    def det(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    d = det(A)
    out = []
    for k in range(3):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][k] = b[i]
        out.append(det(M) / d)
    return out


def setup(seed):
    rng = np.random.default_rng(seed)
    body, c = random_body(rng)
    try:
        sc = SC.ExactScene([_Comp(body)])
    except ValueError:
        return None
    rows = [tuple(r) for r in sc.rows[0]]
    P = Planes(1 << 10); sl = Slots(4)
    # source box on one side of the body, window on another side (segments cross the body region)
    d = rng.standard_normal(3); d /= np.linalg.norm(d)
    s0 = np.clip(c + d * rng.uniform(0.15, 0.3), 0.02, 0.9)
    sz = rng.uniform(0.001, 0.03)
    lo = np.round(s0 * 4096) / 4096; hi = lo + np.ceil(sz * 4096) / 4096
    ids = np.array([[P.axis_plane(k, float(lo[k])), P.axis_plane(k, float(hi[k]))] for k in range(3)], np.int64)
    a = sl.args()
    poly.set_box(P.PL, 0, ids, *a)
    if rng.random() < 0.6:
        n = rng.standard_normal(3); mid = 0.5 * (lo + hi)
        pid = P.add(np.array([n[0], n[1], n[2], -float(n @ mid) + rng.uniform(-0.3, 0.3) * sz]))
        if poly.clip(P.PL, 0, 2 * pid, *a) != poly.OK:
            return None
    N = 512; L = 8
    nv = int(sl.a["nV"][0])
    # window on the shadow boundary of the source-box centre: both classes present among its receivers
    sc_ctr = [Q(float(v)) for v in 0.5 * (lo + hi)]
    o = None
    for _ in range(400):
        w0 = c - d * rng.uniform(0.12, 0.3) + rng.standard_normal(3) * rng.uniform(0.02, 0.12)
        if np.any(w0 < 0.03) or np.any(w0 > 0.96):
            continue
        oo = np.floor(w0 * N).astype(np.int64) - L // 2
        corners = [[Q(2 * int(oo[k] + t) + 1, 2 * N) for k, t in enumerate(cc)] for cc in itertools.product((0, L - 1), repeat=3)]
        hs_ = {seg_hits(sc_ctr, r, rows) for r in corners}
        if len(hs_) == 2:
            o = oo
            break
    if o is None:
        return None
    V = [exact_vertex(P.PL, sl.a["Vp"][0, v]) for v in range(nv)]
    samples = list(V)
    for _ in range(12):
        w = rng.integers(1, 9, nv); tot = int(w.sum())
        samples.append([sum(Q(int(w[v]), tot) * V[v][k] for v in range(nv)) for k in range(3)])
    recv = {}
    for X, Y, Z in itertools.product(range(L), repeat=3):
        r = [Q(2 * int(o[k] + t) + 1, 2 * N) for k, t in zip(range(3), (X, Y, Z))]
        recv[(X, Y, Z)] = r
    return sc, rows, P, sl, nv, samples, recv, o, N, L, rng


def inside(r, rows):
    return all(row[0] * r[0] + row[1] * r[1] + row[2] * r[2] + row[3] <= 0 for row in rows)


@pytest.mark.parametrize("seed", range(40))
def test_recv_test_sound(seed):
    S = setup(seed)
    if S is None:
        pytest.skip("degenerate")
    sc, rows, P, sl, nv, samples, recv, o, N, L, rng = S
    Vm = sl.a["Vm"][0, :nv]; Vr = sl.a["Vr"][0, :nv]
    comps = np.zeros(1, np.int64)
    hm = np.zeros((PR.MH, 4)); hr = np.zeros((PR.MH, 4)); hk = np.zeros((PR.MH, 4), np.int64); hs = np.zeros(PR.MH, np.int64)
    seen = set()
    for key, r in list(recv.items())[::3]:
        if inside(r, rows):
            continue
        hits = [seg_hits(s, r, rows) for s in samples]
        for obs in (PR.DARK, PR.LIT):
            st, nh, w = PR.recv_test(Vm, Vr, nv, float(r[0]), float(r[1]), float(r[2]), obs, comps, 1, sc.RM, sc.RR,
                                     sc.K, sc.ADJ, hm, hr, hk, hs)
            seen.add(st)
            if st == PR.R_DARK:
                assert all(hits)
            elif st == PR.R_LIT:
                assert not any(hits)
            elif st == PR.R_CONV:
                for s, h in zip(samples, hits):
                    ok = True
                    for t in range(nh):
                        c, i, j, kind = (int(v) for v in hk[t])
                        fi = lambda x, i=i: rows[i][0] * x[0] + rows[i][1] * x[1] + rows[i][2] * x[2] + rows[i][3]
                        g = fi(s) if kind == 0 else (fi(r) * (rows[j][0] * s[0] + rows[j][1] * s[1] + rows[j][2] * s[2]
                                                               + rows[j][3]) - (rows[j][0] * r[0] + rows[j][1] * r[1]
                                                               + rows[j][2] * r[2] + rows[j][3]) * fi(s))
                        ok = ok and ((g <= 0) if hs[t] == 1 else (g > 0))
                    assert ok == (h == (obs == PR.DARK)), (key, obs, nh)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("rcap", [0, 512])
def test_scan_rows_sound(seed, rcap):
    S = setup(seed)
    if S is None:
        pytest.skip("degenerate")
    sc, rows, P, sl, nv, samples, recv, o, N, L, rng = S
    Vm = sl.a["Vm"][0, :nv]; Vr = sl.a["Vr"][0, :nv]
    # observation: exact response for a random sample, a few flipped receivers
    s_ref = samples[int(rng.integers(len(samples)))]
    cls = np.zeros((L, L, L), np.uint8)
    for (X, Y, Z), r in recv.items():
        cls[X, Y, Z] = PR.SOLID if inside(r, rows) else (PR.DARK if seg_hits(s_ref, r, rows) else PR.LIT)
    if rng.random() < 0.5:
        for _ in range(3):
            X, Y, Z = (int(v) for v in rng.integers(0, L, 3))
            if cls[X, Y, Z] != PR.SOLID:
                cls[X, Y, Z] = 1 - cls[X, Y, Z]
    gx, gy = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    rws = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)
    amb = np.zeros((L ** 3, 4), np.int64); rowflag = np.zeros(len(rws), np.int64); wit = np.zeros(4, np.int64)
    comps = np.zeros(1, np.int64)
    code, namb = PR.scan_rows(Vm, Vr, nv, rws, len(rws), cls, o, N, L, comps, 1, sc.RM, sc.RR, sc.K, sc.ADJ, amb,
                              len(amb), rowflag, wit, rcap)
    assert code != PR.SC_BUG
    if code == PR.SC_OUT:
        X, Y, Z = (int(v) for v in wit[:3]); r = recv[(X, Y, Z)]
        for s in samples:
            assert seg_hits(s, r, rows) != (cls[X, Y, Z] == PR.DARK)
        return
    ambs = {tuple(int(v) for v in amb[k, :3]) for k in range(namb)}
    for (X, Y, Z), r in recv.items():
        if cls[X, Y, Z] == PR.SOLID or (X, Y, Z) in ambs:
            continue
        for s in samples:
            assert seg_hits(s, r, rows) == (cls[X, Y, Z] == PR.DARK), (X, Y, Z)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("rcap", [0, 512])
@pytest.mark.parametrize("lb", [0, 2])
def test_scan_tree_sound(seed, rcap, lb):
    """Hierarchical block scan: same soundness property as scan_rows, and the marked level-lb blocks cover every
    ambiguous receiver."""
    S = setup(seed)
    if S is None:
        pytest.skip("degenerate")
    sc, rows, P, sl, nv, samples, recv, o, N, L, rng = S
    Vm = sl.a["Vm"][0, :nv]; Vr = sl.a["Vr"][0, :nv]
    s_ref = samples[int(rng.integers(len(samples)))]
    cls = np.zeros((L, L, L), np.uint8)
    for (X, Y, Z), r in recv.items():
        cls[X, Y, Z] = PR.SOLID if inside(r, rows) else (PR.DARK if seg_hits(s_ref, r, rows) else PR.LIT)
    if rng.random() < 0.5:
        for _ in range(3):
            X, Y, Z = (int(v) for v in rng.integers(0, L, 3))
            if cls[X, Y, Z] != PR.SOLID:
                cls[X, Y, Z] = 1 - cls[X, Y, Z]
    pyr, poff, q = PR.pyramid(cls)
    comps = np.zeros(1, np.int64)
    prep = PR.prep_all(Vm, Vr, nv, comps, 1, sc.RM, sc.RR, sc.K, sc.ADJ)
    starts = np.array([[q, 0, 0, 0]], np.int64)
    amb = np.zeros((L ** 3, 4), np.int64); bmark = np.zeros((L >> lb) ** 3, np.int64); wit = np.zeros(4, np.int64)
    code, namb = PR.scan_tree(Vm, Vr, nv, starts, 1, lb, pyr, poff, q, cls, o, N, L, comps, 1, sc.RM, sc.RR, sc.K, sc.ADJ,
                              amb, len(amb), bmark, wit, rcap, *prep)
    assert code != PR.SC_BUG
    if code == PR.SC_OUT:
        X, Y, Z = (int(v) for v in wit[:3]); r = recv[(X, Y, Z)]
        assert cls[X, Y, Z] != PR.SOLID
        for s in samples:
            assert seg_hits(s, r, rows) != (cls[X, Y, Z] == PR.DARK)
        return
    ambs = {tuple(int(v) for v in amb[k, :3]) for k in range(namb)}
    nb = L >> lb
    for a in ambs:
        assert bmark[((a[0] >> lb) * nb + (a[1] >> lb)) * nb + (a[2] >> lb)] == 1
    for (X, Y, Z), r in recv.items():
        if cls[X, Y, Z] == PR.SOLID or (X, Y, Z) in ambs:
            continue
        for s in samples:
            assert seg_hits(s, r, rows) == (cls[X, Y, Z] == PR.DARK), (X, Y, Z)
