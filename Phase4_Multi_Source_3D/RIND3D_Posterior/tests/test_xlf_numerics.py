"""Numerical soundness of the XLF trusted arithmetic against exact rationals (docs/XLF_IMPLEMENTATION_ASSURANCE.md).

Each test checks one a priori error bound or directed-rounding argument of the implementation on adversarial inputs
(cancellation, values at binade boundaries, zero and tiny coefficients, ill-conditioned triples), plus the regression
tests of the defects found by the audit (D1 accumulation, D2 checker inner-plane test, D4 gamma index; D3 is covered
by the many-face volume test). Tests are evidence, not proofs; the proofs are in the document."""
from fractions import Fraction as Q
import itertools
import random

import numpy as np
import pytest

from rind3d_posterior import xlf  # noqa: F401
from rind3d_posterior.certify import poly, tvb
from rind3d_posterior.certify.ar import G3, G4, INFL, TINY, U
from rind3d_posterior.xlf import pred as PR
from rind3d_posterior.xlf import solve as XS
from rind3d_posterior.xlf import check as XC

F = lambda x: Q(float(x))


def adv_float(rng, scale=1.0):
    """Adversarial binary64: near powers of two, tiny, zero, or random."""
    k = rng.random()
    if k < 0.15:
        return 0.0
    sg = -1.0 if rng.random() < 0.5 else 1.0
    if k < 0.45:
        e = rng.randint(-8, 1)
        return sg * float(np.nextafter(2.0 ** e, np.inf if rng.random() < 0.5 else -np.inf)) * scale
    if k < 0.55:
        return sg * rng.random() * 2.0 ** -rng.randint(30, 60) * scale
    return sg * rng.random() * scale


def prod_range(a, ra, b, rb):
    c = [x * y for x in (a - ra, a + ra) for y in (b - rb, b + rb)]
    return min(c), max(c)


# ------------------------------------------------------------------------------------------------ pred.aff
def test_aff_encloses_affine_function_over_interval_coefficients_and_box():
    rng = random.Random(1)
    for _ in range(20000):
        cm = np.array([adv_float(rng) for _ in range(4)]); cr = np.array([abs(adv_float(rng, 2.0 ** -50)) for _ in range(4)])
        x = [abs(adv_float(rng)) for _ in range(3)]; r = [abs(adv_float(rng, 2.0 ** -20)) * (rng.random() < 0.5) for _ in range(3)]
        m, rad = PR.aff(cm, cr, x[0], x[1], x[2], r[0], r[1], r[2])
        lo = F(cm[3]) - F(cr[3]); hi = F(cm[3]) + F(cr[3])
        for k in range(3):
            a, b = prod_range(F(cm[k]), F(cr[k]), F(x[k]), F(r[k]))
            lo += a; hi += b
        assert F(PR.dn(m - rad)) <= lo and hi <= F(PR.up(m + rad))


def test_pair_and_comb_coefficient_enclosures():
    rng = random.Random(2)
    om = np.zeros(4); orr = np.zeros(4)
    for _ in range(5000):
        Fi, Fj = adv_float(rng), adv_float(rng); ri, rj = abs(adv_float(rng, 2.0 ** -45)), abs(adv_float(rng, 2.0 ** -45))
        qim = np.array([adv_float(rng) for _ in range(4)]); qjm = np.array([adv_float(rng) for _ in range(4)])
        qir = np.array([abs(adv_float(rng, 2.0 ** -52)) for _ in range(4)]); qjr = np.array([abs(adv_float(rng, 2.0 ** -52)) for _ in range(4)])
        PR.pair_coef(Fi, ri, Fj, rj, qim, qir, qjm, qjr, om, orr)
        for t in range(4):
            a1, b1 = prod_range(F(Fi), F(ri), F(qjm[t]), F(qjr[t]))
            a2, b2 = prod_range(F(Fj), F(rj), F(qim[t]), F(qir[t]))
            assert F(om[t]) - F(orr[t]) <= a1 - b2 and b1 - a2 <= F(om[t]) + F(orr[t])
        lam, mu = abs(adv_float(rng)), abs(adv_float(rng))
        PR.comb_coef(lam, mu, qim, qir, qjm, qjr, om, orr)
        for t in range(4):
            lo = F(lam) * (F(qim[t]) - F(qir[t])) + F(mu) * (F(qjm[t]) - F(qjr[t]))
            hi = F(lam) * (F(qim[t]) + F(qir[t])) + F(mu) * (F(qjm[t]) + F(qjr[t]))
            assert F(om[t]) - F(orr[t]) <= lo and hi <= F(om[t]) + F(orr[t])


def test_cross_lo_is_a_lower_bound():
    rng = random.Random(3)
    for _ in range(20000):
        v = [adv_float(rng) for _ in range(4)]; r = [abs(adv_float(rng, 2.0 ** -40)) for _ in range(4)]
        lo = PR._cross_lo(v[0], r[0], v[1], r[1], v[2], r[2], v[3], r[3])
        a1, b1 = prod_range(F(v[0]), F(r[0]), F(v[3]), F(r[3]))
        a2, b2 = prod_range(F(v[1]), F(r[1]), F(v[2]), F(r[2]))
        assert F(lo) <= a1 - b2


@pytest.mark.parametrize("seed", range(4))
def test_row_ranges_are_sound(seed):
    """_le_range (surely <= 0) and _gt_ray (surely > 0) on a receiver row against the exact extremes over the
    interval coefficients at every receiver of the row."""
    rng = random.Random(10 + seed)
    N = 1024; L = 32
    for _ in range(600):
        cm = np.array([adv_float(rng) for _ in range(4)]); cr = np.array([abs(adv_float(rng, 2.0 ** -48)) for _ in range(4)])
        oz = rng.randint(0, N - L); X = rng.randint(0, N - 1); Y = rng.randint(0, N - 1)
        x = (X + 0.5) / N; y = (Y + 0.5) / N
        if rng.random() < 0.5:                                  # make the zero crossing fall inside the row
            z = (oz + rng.randint(0, L - 1) + 0.5) / N
            cm[3] = -(cm[0] * x + cm[1] * y + cm[2] * z) * (1 + (rng.random() - 0.5) * 2.0 ** -45)
        lo, hi = PR._le_range(cm, cr, x, y, oz, N, L)
        kind, T = PR._gt_ray(cm, cr, x, y, oz, N, L)
        gt = set()
        if kind == 1:
            gt = set(range(L))
        elif kind == 2:
            gt = set(range(T, L))
        elif kind == 3:
            gt = set(range(0, T + 1))
        for t in range(L):
            r = (F(x), F(y), Q(2 * (oz + t) + 1, 2 * N))
            emax = F(cm[3]) + F(cr[3]) + sum(max(F(cm[k]) * r[k] + F(cr[k]) * r[k], F(cm[k]) * r[k] - F(cr[k]) * r[k]) for k in range(3))
            emin = F(cm[3]) - F(cr[3]) + sum(min(F(cm[k]) * r[k] + F(cr[k]) * r[k], F(cm[k]) * r[k] - F(cr[k]) * r[k]) for k in range(3))
            if lo <= t <= hi:
                assert emax <= 0
            if t in gt:
                assert emin > 0


# ------------------------------------------------------------------------------------------------ D4 gamma index
def test_affine_at_gamma4_bound_and_gamma3_counterexample():
    """D4: three products and three additions need gamma_4. The recorded case violates the former gamma_3 enclosure
    (it is far from zero, so the sos_sign decision it feeds was not affected); the gamma_4 bound holds on it and on
    random adversarial sums."""
    a = [float.fromhex(h) for h in ('0x1.950dd23b6510ap+0', '0x1.64f2b23209fdep+0', '0x1.4f505d82556bap-43')]
    v = [float.fromhex(h) for h in ('0x1.b62c668be6728p-2', '0x1.f13d613cc3bfcp-1', '0x1.08a0e52a07e94p-2')]
    a3 = float.fromhex('-0x1.03b27168f120cp-46')
    s = a[0] * v[0] + a[1] * v[1] + a[2] * v[2] + a3
    ab = abs(a[0] * v[0]) + abs(a[1] * v[1]) + abs(a[2] * v[2]) + abs(a3)
    E = sum(F(a[k]) * F(v[k]) for k in range(3)) + F(a3)
    assert abs(F(s) - E) > F(G3 * ab * INFL + TINY)              # former bound fails as an enclosure
    PL = np.array([[a[0], a[1], a[2], a3]])
    m, rad = poly._affine_at(PL, 0, np.array(v), np.zeros(3))
    assert m == s and abs(F(m) - E) <= F(rad)
    rng = random.Random(5)
    for _ in range(30000):
        a = [1 + rng.random() for _ in range(3)]; v = [0.25 + 0.75 * rng.random() for _ in range(3)]
        a3 = -(a[0] * v[0] + a[1] * v[1] + a[2] * v[2]) * (1 + (rng.random() - 0.5) * 2 ** -50) if rng.random() < 0.3 \
            else (rng.random() - 0.5) * 2 ** -rng.randint(0, 55)
        m, rad = poly._affine_at(np.array([[a[0], a[1], a[2], a3]]), 0, np.array(v), np.zeros(3))
        assert abs(F(m) - (sum(F(a[k]) * F(v[k]) for k in range(3)) + F(a3))) <= F(rad)


# ------------------------------------------------------------------------------------------------ vertex enclosures
def test_vertex_enclosure_ill_conditioned_triples():
    rng = random.Random(6)
    from xlf_exact_ref import exact_vertex
    om = np.zeros(3); orr = np.zeros(3)
    n_exact = 0
    for _ in range(3000):
        base = np.array([rng.uniform(-1, 1) for _ in range(3)]); base /= np.max(np.abs(base))
        P = np.zeros((3, 4))
        for i in range(3):
            d = np.array([rng.uniform(-1, 1) for _ in range(3)]) * 10.0 ** -rng.randint(0, 14)
            n = base + d if i > 0 else base
            P[i, :3] = n; P[i, 3] = rng.uniform(-1, 1)
        if rng.random() < 0.3:
            P[2, :3] = P[1, :3] + np.nextafter(0.0, 1.0) * 0      # exactly parallel normals: singular
        ok = poly.vertex_enclosure(P, 0, 1, 2, om, orr)
        x = exact_vertex(P, (0, 1, 2))
        if x is None:
            assert not ok
            continue
        if not ok:
            continue
        n_exact += 1
        for k in range(3):
            assert F(om[k]) - F(orr[k]) <= x[k] <= F(om[k]) + F(orr[k])
    assert n_exact > 1000


# ------------------------------------------------------------------------------------------------ volumes (D3)
def _exact_volume(rows):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_certify_kernel import exact_volume
    return exact_volume(rows)


@pytest.mark.parametrize("seed", range(6))
def test_volume_enclosure_with_many_faces(seed):
    """A box cut by many planes tangent-ish to an inner sphere: dozens of faces and >100 fan triangles (the regime in
    which the former radius sum was not covered by INFL)."""
    rng = np.random.default_rng(100 + seed)
    from rind3d_posterior.xlf.solve import Planes, Slots
    pls = Planes(256); sl = Slots(4)
    ax = np.array([[pls.axis_plane(k, 0.25), pls.axis_plane(k, 0.75)] for k in range(3)], np.int64)
    s = sl.get(); poly.set_box(pls.PL, s, ax, *sl.args())
    rows = [tuple(Q(float(v)) for v in pls.PL[i]) for i in ax.reshape(-1)]
    rows = [r if (k % 2) else tuple(-v for v in r) for k, r in enumerate(rows)]   # lo faces: -(x - lo) <= 0
    for _ in range(60):
        n = rng.standard_normal(3); n /= np.linalg.norm(n)
        row = np.array([n[0], n[1], n[2], -float(n @ np.full(3, 0.5)) - 0.2 - 0.02 * rng.random()])
        pid = pls.add(row)
        st = poly.clip(pls.PL, s, 2 * pid, *sl.args())
        assert st != poly.OVERFLOW
        rows.append(tuple(Q(float(v)) for v in row))
    nf = int(sl.a["nF"][s]); nt = sum(int(sl.a["Fn"][s, f]) - 2 for f in range(nf))
    assert nt > 64
    lo, hi = poly.volume(s, sl.a["Fn"], sl.a["Fv"], sl.a["Vm"], sl.a["Vr"], sl.a["nF"])
    ev = _exact_volume(rows)
    assert Q(lo) <= ev <= Q(hi)
    assert Q(hi) - Q(lo) < ev * Q(1, 10 ** 10)


# ------------------------------------------------------------------------------------------------ D2 inner plane
class _P:
    pass


def _inner_cex():
    """The exact counterexample of D2: f(x) = -3x + y - 3 * 2^-30 (exact binary64 row, radius 0)."""
    from rind3d_posterior.xlf.solve import Planes, Slots
    cm = np.array([-3.0, 1.0, 0.0, -3 * 2.0 ** -30]); cr = np.zeros(4)
    sc = float(np.max(np.abs(cm))); h = cm / sc
    p = _P(); p.RM = cm.reshape(1, 1, 4); p.RR = cr.reshape(1, 1, 4)
    pls = Planes(64); sl = Slots(4)
    y = 7 / 8; x0 = float(h[1] * y + h[3])
    lo = np.array([x0 - 2 ** -20, y - 2 ** -20, 0.5]); hi = np.array([x0 + 2 ** -20, y + 2 ** -20, 0.5 + 2 ** -20])
    ax = np.array([[pls.axis_plane(k, lo[k]), pls.axis_plane(k, hi[k])] for k in range(3)], np.int64)
    return p, cm, h, pls, sl, ax


def test_checker_rejects_inner_plane_with_insufficient_offset():
    """D2: offset A_3 - h_3 = 1 ulp(h_3) = 2^-82 passed the former float test (E computed = 0 because
    fl(3 * fl(1/3)) == 1), yet half of the vertices of the accepted piece lie OUTSIDE the facet: the OUT_SOLID claim
    was false. The exact test rejects it."""
    from xlf_exact_ref import exact_vertex
    p, cm, h, pls, sl, ax = _inner_cex()
    A3 = float(np.nextafter(h[3], np.inf))
    a = pls.add(np.array([h[0], h[1], h[2], A3]))
    s = sl.get(); poly.set_box(pls.PL, s, ax, *sl.args())
    assert poly.clip(pls.PL, s, 2 * a, *sl.args()) == poly.OK
    g = [F(v) for v in cm]
    nv = int(sl.a["nV"][s])
    outside = [v for v in range(nv) if sum(g[k] * x for k, x in enumerate(exact_vertex(pls.PL, [int(t) for t in sl.a["Vp"][s, v]]))) + g[3] > 0]
    assert len(outside) > 0                                   # the claim "piece inside {f < 0}" is false
    E = float(np.sum(np.abs(cm - 3.0 * h))) * (1 + 2 ** -40) / 3.0
    assert E == 0.0 and A3 - h[3] > E * (1 + 2 ** -30)        # ... and the former test accepted it
    assert not XC._inner_plane_ok(p, pls.PL, sl.a, s, a, 0, 0)


@pytest.mark.parametrize("seed", range(3))
def test_solver_plane_pair_construction_satisfies_the_exact_inner_condition(seed):
    """The solver's own construction (offset max(8E, 2^-49)) is valid: the exact test accepts every A it builds for
    random canonical facet rows (with nonzero enclosure radii) and the slab plane B satisfies the mirror condition."""
    from xlf_exact_ref import box_rows
    rng = np.random.default_rng(seed)
    sv = XS.Solver.__new__(XS.Solver); sv.delta_floor = 2.0 ** -49; sv.PLs = XS.Planes(1024)
    for _ in range(300):
        n = rng.standard_normal(4); n[3] = -abs(n[3]) * rng.random()
        if rng.random() < 0.3:
            n[rng.integers(0, 3)] = 0.0
        exact = [Q(float(v)) + Q(int(rng.integers(-1000, 1000)), 2 ** 70) for v in n]     # canonical rational row
        m = max(abs(v) for v in exact); exact = [v / m for v in exact]
        cm = np.array([float(v) for v in exact]); cr = np.array([float(abs(Q(float(v)) - v)) * (1 + 2 ** -50) for v in exact])
        A, B = sv._plane_pair(cm.copy(), cr.copy())
        sc = Q(float(np.max(np.abs(cm))))
        ar = [F(v) for v in sv.PLs.PL[A]]; br = [F(v) for v in sv.PLs.PL[B]]
        hi = [F(cm[t]) + F(cr[t]) for t in range(4)]; lo = [F(cm[t]) - F(cr[t]) for t in range(4)]
        # {A <= 0} n cube inside {f < 0}
        assert sum(max(Q(0), hi[t] / sc - ar[t]) for t in range(3)) + hi[3] / sc - ar[3] < 0
        # {B >= 0} n cube inside {f > 0}:  inf (f / sc - B) > 0
        assert sum(min(Q(0), lo[t] / sc - br[t]) for t in range(3)) + lo[3] / sc - br[3] > 0


# ------------------------------------------------------------------------------------------------ D1 accumulation
def test_directed_accumulation_of_cell_bounds():
    """D1: a round-to-nearest running sum of per-piece LOWER bounds can exceed their exact sum (and the sum of upper
    bounds can fall below theirs). add_dn / add_up never do."""
    seq = [1.0] + [0.75 * 2.0 ** -52] * 8                     # each addition rounds UP by 0.25 ulp
    naive = 0.0
    for v in seq:
        naive += v
    exact = sum((F(v) for v in seq), Q(0))
    assert F(naive) > exact                                    # the defect: lo_c > sum of the pieces' lower bounds
    lo = 0.0; hi = 0.0
    for v in seq:
        lo = XS.add_dn(lo, v); hi = XS.add_up(hi, v)
    assert F(lo) <= exact <= F(hi)
    rng = random.Random(9)
    for _ in range(3000):
        vs = [abs(adv_float(rng)) * 2.0 ** -rng.randint(0, 60) for _ in range(rng.randint(1, 30))]
        lo = hi = 0.0
        for v in vs:
            lo = XS.add_dn(lo, v); hi = XS.add_up(hi, v)
        ex = sum((F(v) for v in vs), Q(0))
        assert F(lo) <= ex <= F(hi)


# ------------------------------------------------------------------------------------------------ TV bound
def _tv(p, q):
    keys = set(p) | set(q)
    return sum((abs(p.get(k, Q(0)) - q.get(k, Q(0))) for k in keys), Q(0)) / 2


def _exact_sup(lo, x, pool):
    """Exact supremum of TV(lo / Lt, v / |v|) over lo <= v <= lo + x (pool mass anywhere): max over subsets A of
    (x_A + pool) (1 - lo_A / Lt) / (Lt + x_A + pool) (Thm 6.3 of the foundations)."""
    Lt = sum(lo, Q(0)); best = Q(0)
    n = len(lo)
    for m in range(n + 1):
        for A in itertools.combinations(range(n), m):
            xa = sum((x[i] for i in A), Q(0)) + pool; la = sum((lo[i] for i in A), Q(0))
            best = max(best, xa * (1 - la / Lt) / (Lt + xa))
    return best


@pytest.mark.parametrize("seed", range(5))
def test_tv_bound_dominates_every_consistent_posterior(seed):
    rng = random.Random(40 + seed)
    for _ in range(60):
        n = rng.randint(1, 7)
        lo = [abs(adv_float(rng)) * (rng.random() < 0.8) for _ in range(n)]
        if sum(lo) == 0:
            lo[0] = 0.5
        hi = [l + abs(adv_float(rng, 0.3)) * (rng.random() < 0.7) for l in lo]
        pool = abs(adv_float(rng, 0.1)) * (rng.random() < 0.3)
        cells = np.arange(n)
        b, Lt = tvb.tv_bound_exact(cells, np.array(lo), np.array(hi), pool)
        L = [F(v) for v in lo]; H = [F(v) for v in hi]
        qt = {i: L[i] / Lt for i in range(n) if L[i] > 0}
        sup = _exact_sup(L, [H[i] - L[i] for i in range(n)], F(pool))
        assert sup <= b <= 1 - Lt / (Lt + sum((H[i] - L[i] for i in range(n)), Q(0)) + F(pool)) or b == 1
        for _ in range(20):                                    # random exact posteriors inside the enclosures
            v = [L[i] + (H[i] - L[i]) * Q(rng.randint(0, 8), 8) for i in range(n)]
            pm = F(pool) * Q(rng.randint(0, 8), 8); tgt = rng.randrange(n + 1)
            if tgt < n:
                v[tgt] += pm
            else:
                v.append(pm)
            V = sum(v, Q(0))
            q = {i: v[i] / V for i in range(len(v)) if v[i] > 0}
            assert _tv(qt, q) <= sup <= b


def test_tv_bound_single_cell_is_zero():
    b, Lt = tvb.tv_bound_exact(np.array([7]), np.array([1e-20]), np.array([3e-20]), 0.0)
    assert b == 0
