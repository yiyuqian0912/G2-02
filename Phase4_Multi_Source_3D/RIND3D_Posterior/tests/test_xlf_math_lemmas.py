"""Exact (Fraction) checks of the mathematical statements of docs/XLF_MATHEMATICAL_FOUNDATIONS.md, including the
counterexamples to tempting stronger claims. These tests are EVIDENCE for, and regression guards of, the theorems; the
proofs are in the document. Theorem numbers refer to that document.

No binary64 decision is involved except where a test explicitly exercises the implementation (marked IMPL)."""
from fractions import Fraction as Q
import itertools
import random

import numpy as np
import pytest

from xlf_exact_ref import (Comp, box_rows, edge_pairs, ev, hits_any, inside, normalise, pair_formula, seg_hits,
                           vertices, det3)


# ------------------------------------------------------------------------------------------------ random bodies
def rq(rng, lo, hi, den=64):
    return Q(rng.randint(int(lo * den), int(hi * den)), den)


def random_body(rng):
    """Bounded full-dimensional rational polytope: an axis box cut by up to 3 random planes that keep its centre."""
    c = [rq(rng, Q(2, 5), Q(3, 5)) for _ in range(3)]
    h = [rq(rng, Q(1, 20), Q(1, 8), 160) for _ in range(3)]
    rows = list(box_rows([c[k] - h[k] for k in range(3)], [c[k] + h[k] for k in range(3)]))
    for _ in range(rng.randint(0, 3)):
        n = [Q(rng.randint(-8, 8), 8) for _ in range(3)]
        if not any(n):
            continue
        off = sum(n[k] * c[k] for k in range(3))
        m = sum(abs(n[k]) * h[k] for k in range(3))
        rows.append((*n, -off - m * rq(rng, Q(2, 5), Q(9, 10), 40)))   # cuts a corner region, keeps the centre
    return normalise(rows), c


def random_point(rng, lo=0, hi=1, den=97):
    return tuple(Q(rng.randint(lo * den, hi * den), den) for _ in range(3))


def points_on_facet_planes(rows, rng, n):
    """Exact points OUTSIDE the body lying on one (or two) facet planes: beyond a facet vertex inside the facet plane,
    and on the extension of an edge line."""
    V = vertices(rows)
    out = []
    for _ in range(n):
        v, tight = rng.choice(list(V.items()))
        others = [w for w, t in V.items() if w != v and len(t & tight) >= 2]
        if not others:
            continue
        w = rng.choice(others)
        lam = Q(rng.randint(1, 40), 20)
        p = tuple(v[k] + lam * (v[k] - w[k]) for k in range(3))          # on every plane tight at both v and w
        if not inside(p, rows):
            out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ Theorem 2.1 / 2.3
@pytest.mark.parametrize("seed", range(12))
def test_segment_lemma_all_pairs_and_silhouette_reduction(seed):
    """Thm 2.1: zero-inclusive gates + ALL pairs (i in Phi+, j in Phi-) characterise the closed hit, for every apex
    outside the closed body (no aspect condition). Thm 2.3: for a full-dimensional body the pairs may be restricted
    to rows tight together on an edge (>= 2 shared vertices), also with zero facet values at the apex."""
    rng = random.Random(1000 + seed)
    rows, c = random_body(rng)
    adj = edge_pairs(rows)
    apexes = [random_point(rng) for _ in range(12)] + points_on_facet_planes(rows, rng, 12)
    V = list(vertices(rows))
    n_zero = 0
    for p in apexes:
        if inside(p, rows):
            continue
        n_zero += any(ev(r, p) == 0 for r in rows)
        xs = [random_point(rng) for _ in range(10)]
        for k in V[:6]:                                    # points beyond the body along grazing-ish rays from p
            for lam in (Q(1), Q(3, 2), Q(4)):
                xs.append(tuple(p[t] + lam * (k[t] - p[t]) for t in range(3)))
        for x in xs:
            truth = seg_hits(p, x, rows)
            assert pair_formula(p, x, rows) == truth
            assert pair_formula(p, x, rows, adj) == truth
    assert n_zero > 0                                       # zero facet values were exercised


def test_flat_body_apex_in_its_plane_needs_more_than_edge_pairs():
    """Counterexample 2.4: for a 2-dimensional body and an apex IN its plane, gates + edge pairs claim a hit that does
    not exist (the in-plane wedge needs the pair of the two side rows, which share only a vertex). So the silhouette
    reduction needs strict aspect (or full dimension); the all-pairs formula stays exact."""
    h = Q(1, 2)
    rows = normalise([(0, 0, 1, -h), (0, 0, -1, h), (-1, 0, 0, Q(2, 5)), (1, 0, 0, -Q(3, 5)),
                      (0, -1, 0, Q(2, 5)), (0, 1, 0, -Q(3, 5))])          # square [.4,.6]^2 x {1/2}
    p = (Q(1, 5), h, h); x = (Q(4, 5), Q(9, 10), h)
    adj = edge_pairs(rows)
    assert not seg_hits(p, x, rows)
    assert pair_formula(p, x, rows) is False                # all pairs: exact
    assert pair_formula(p, x, rows, adj) is True            # edge pairs only: wrong claim "dark"
    # off the plane (strict aspect) the edge-pair formula is exact again
    rng = random.Random(7)
    for _ in range(300):
        q = random_point(rng)
        if q[2] == h or inside(q, rows):
            continue
        y = random_point(rng)
        assert pair_formula(q, y, rows, adj) == seg_hits(q, y, rows)


def test_flat_body_apex_in_plane_is_never_decided_by_the_code():
    """IMPL: the implementation requires strict signs at every apex (receivers in recv_test, source vertices in
    _prep_comp), so the configuration of the counterexample is reported undecided, never DARK."""
    from rind3d_posterior import xlf  # noqa: F401
    from rind3d_posterior.certify import scene as SC
    from rind3d_posterior.xlf import pred as PR
    h = Q(1, 2)
    rows = normalise([(0, 0, 1, -h), (0, 0, -1, h), (-1, 0, 0, Q(2, 5)), (1, 0, 0, -Q(3, 5)),
                      (0, -1, 0, Q(2, 5)), (0, 1, 0, -Q(3, 5))])
    sc = SC.ExactScene([Comp(rows)])
    # receiver in the plane z = 1/2 (exact dyadic), source box around (4/5, 9/10, 1/2) in the same plane
    rx, ry, rz = 0.203125, 0.5, 0.5
    Vm = np.array([[0.8 + dx, 0.9 + dy, 0.5 + dz] for dx in (-1e-3, 1e-3) for dy in (-1e-3, 1e-3) for dz in (-1e-3, 1e-3)])
    Vr = np.zeros_like(Vm)
    hs = (np.zeros((PR.MH, 4)), np.zeros((PR.MH, 4)), np.zeros((PR.MH, 4), np.int64), np.zeros(PR.MH, np.int64))
    st, nh, w = PR.recv_test(Vm, Vr, 8, rx, ry, rz, PR.DARK, np.array([0]), 1, sc.RM, sc.RR, sc.K, sc.ADJ, *hs)
    assert st != PR.R_DARK
    # a source vertex in the plane of the flat body: the dark constraints of the component are unavailable
    dm = np.zeros((64, 4)); dr = np.zeros((64, 4)); lm = np.zeros((64, 4)); lr = np.zeros((64, 4))
    Vm2 = Vm.copy(); Vm2[0] = [0.203125, 0.5, 0.5]
    nd, nl = PR._prep_comp(0, Vm2, Vr, 8, sc.RM, sc.RR, sc.K, sc.ADJ, dm, dr, lm, lr)
    assert nd == -1


# ------------------------------------------------------------------------------------------------ Lemma D / pencils
@pytest.mark.parametrize("seed", range(6))
def test_guaranteed_shadow_is_decided_by_vertices(seed):
    """Lemma 3.1 (D): a receiver is dark for every source of a convex Q iff it is dark for every vertex of Q
    (C_K(r) is convex). Checked on random rational combinations; the converse direction is trivial."""
    rng = random.Random(50 + seed)
    rows, c = random_body(rng)
    for _ in range(40):
        a = random_point(rng); b = tuple(a[k] + Q(rng.randint(1, 12), 97) for k in range(3))
        V = list(itertools.product(*[(a[k], b[k]) for k in range(3)]))
        if any(inside(v, rows) for v in V):
            continue
        r = random_point(rng)
        if inside(r, rows):
            continue
        allv = all(seg_hits(v, r, rows) for v in V)
        for _ in range(10):
            w = [Q(rng.randint(0, 9)) for _ in V]
            if sum(w) == 0:
                continue
            s = tuple(sum(w[i] * V[i][k] for i in range(len(V))) / sum(w) for k in range(3))
            if allv:
                assert seg_hits(s, r, rows)


def test_pencil_collapse_only_angular_extremes_bind():
    """Lemma 3.3: B_ij(v, r) is linear in w_v = (f_i(v), f_j(v)); a w_v inside the cone of two others is implied."""
    rng = random.Random(3)
    for _ in range(2000):
        ws = [(Q(rng.randint(1, 50), 7), -Q(rng.randint(1, 50), 7)) for _ in range(rng.randint(2, 6))]
        import math
        key = lambda w: math.atan2(float(w[1]), float(w[0]))
        a = min(ws, key=key); b = max(ws, key=key)
        wr = (Q(rng.randint(-50, 50), 7), Q(rng.randint(-50, 50), 7))
        B = lambda w: w[0] * wr[1] - w[1] * wr[0]
        if B(a) <= 0 and B(b) <= 0:
            assert all(B(w) <= 0 for w in ws)


# ------------------------------------------------------------------------------------------------ Lemma L
def test_lit_plane_families_are_incomplete_at_an_occluder_vertex():
    """Remark 3.6: no facet plane, no edge-pencil plane (receiver form) and no constraint of C_K(r) (source form)
    separates the triangle conv(Q u {r}) from the unit cube, although the triangle misses the cube (the separating
    plane x + y + z = 3 supports the cube at a VERTEX only). Hence the implemented lit test is sound but incomplete
    at region level; it is complete for points (Thm 2.1)."""
    rows = box_rows((0, 0, 0), (1, 1, 1))
    y1 = (Q(13, 10), Q(9, 10), Q(9, 10)); y2 = (Q(9, 10), Q(13, 10), Q(9, 10)); r = (Q(9, 10), Q(9, 10), Q(13, 10))
    tri = [y1, y2, r]
    # the triangle misses the cube: x + y + z = 31/10 > 3 on it
    assert all(sum(p) > 3 for p in tri)
    for s in (y1, y2):
        assert not seg_hits(s, r, rows)
    # facets: none positive on all three points
    assert not any(all(ev(f, p) > 0 for p in tri) for f in rows)
    # edge pencils lam f_i + mu f_j (lam, mu >= 0, not both 0), parametrised as (1 - t, t), t in [0, 1]: each point
    # gives an open half-line condition on t; the feasible set is an interval, so it is empty iff no critical value
    # or midpoint between consecutive critical values is feasible (exact).
    for i, j in edge_pairs(rows):
        ab = [(ev(rows[i], p), ev(rows[j], p)) for p in tri]
        crit = {Q(0), Q(1)}
        for a, b in ab:
            if a != b and 0 <= a / (a - b) <= 1:
                crit.add(a / (a - b))
        crit = sorted(crit)
        cand = crit + [(u + v) / 2 for u, v in zip(crit, crit[1:])]
        assert not any(all((1 - t) * a + t * b > 0 for a, b in ab) for t in cand)
    adj = edge_pairs(rows)
    # source form: every constraint of C_K(r) (gates and edge pairs at apex r) is <= 0 somewhere on Q = [y1, y2]
    a = [ev(f, r) for f in rows]
    cons = [rows[i] for i in range(6) if a[i] >= 0]
    for i, j in adj:
        if a[i] > 0 > a[j]:
            cons.append(tuple(a[i] * rows[j][t] - a[j] * rows[i][t] for t in range(4)))
    assert not any(all(ev(g, s) > 0 for s in (y1, y2)) for g in cons)


@pytest.mark.parametrize("seed", range(4))
def test_possible_shadow_identity(seed):
    """Thm 3.5 (Astra L3): for Q disjoint from K, U_{s in Q} C_K(s) = K + cone(K - Q). Checked through the identity
    used in the proof: k0 + mu (k1 - s1) = k + mu (k - s1) with k = (k0 + mu k1) / (1 + mu), which lies in C_K(s1)."""
    rng = random.Random(90 + seed)
    rows, c = random_body(rng)
    V = list(vertices(rows))
    for _ in range(200):
        k0 = rng.choice(V); k1 = rng.choice(V)
        s1 = random_point(rng)
        if inside(s1, rows):
            continue
        mu = Q(rng.randint(0, 30), 7)
        x = tuple(k0[t] + mu * (k1[t] - s1[t]) for t in range(3))
        assert seg_hits(s1, x, rows)


# ------------------------------------------------------------------------------------------------ local-first
def test_prop_A_prime_is_strictly_weaker_than_the_exact_pair_constraint():
    """Counterexample 4.3: Proposition A' admits s = 0 for the interface (l lit, k dark) through the silhouette edge
    {x = 4, y = -1/10} of K = [4, 5] x [-1/10, 1/10]^2, but [s, k] misses K, so s is infeasible; the exact pair
    constraint s in C(k) minus C(l) rejects it."""
    K = box_rows((4, Q(-1, 10), Q(-1, 10)), (5, Q(1, 10), Q(1, 10)))
    s = (Q(0), Q(0), Q(0)); l = (Q(10), Q(1), Q(0)); k = (Q(10), Q(-1), Q(0))
    assert not seg_hits(s, k, K) and not seg_hits(s, l, K)          # both lit: s does not reproduce k dark
    assert not seg_hits(l, k, K)                                       # H1 holds
    # the edge e = {x = 4, y = -1/10, |z| <= 1/10} meets the triangle conv{s, l, k} (plane z = 0) at (4, -1/10, 0)
    p = (Q(4), Q(-1, 10), Q(0))
    # barycentric coordinates of p in (s, l, k): p = a s + b l + c k, a + b + c = 1
    b_ = (Q(4) / 10 + Q(-1, 10)) / 2; c_ = (Q(4) / 10 - Q(-1, 10)) / 2; a_ = 1 - b_ - c_
    assert min(a_, b_, c_) >= 0 and tuple(a_ * s[t] + b_ * l[t] + c_ * k[t] for t in range(3)) == p
    # silhouette condition of A' item 3 at p = k: outward rows of the two facets of e have opposite signs
    g1 = (Q(-1), Q(0), Q(0), Q(4)); g2 = (Q(0), Q(-1), Q(0), Q(-1, 10))
    assert ev(g1, k) * ev(g2, k) <= 0


def test_interface_constraints_alone_do_not_determine_F():
    """Thm 4.7: a window with NO response interface (all lit) still excludes sources whose segment to some receiver
    meets the solid. Interface-only regions are outer bounds; the exact posterior needs every receiver."""
    K = box_rows((Q(2, 5), Q(2, 5), Q(2, 5)), (Q(3, 5), Q(3, 5), Q(3, 5)))
    N, o, L = 16, (12, 7, 7), 2                                         # receivers near x = 0.78, y,z ~ 0.47
    from xlf_exact_ref import receiver, window_cls
    s_true = (Q(9, 10), Q(1, 2), Q(1, 2))                               # same side as the window: all lit
    cls = window_cls([K], s_true, o, N, L)
    assert np.all(cls == 1)
    s_other = (Q(1, 10), Q(1, 2), Q(1, 2))                              # behind the box
    assert any(seg_hits(s_other, receiver(o, t, N), K) for t in itertools.product(range(L), repeat=3))


def test_multiple_explanations_make_the_pair_region_nonconvex():
    """Thm 4.8: with two occluders the feasible set of one interface pair (l lit, k dark) is a union of two wedges:
    s1 and s2 are feasible (k dark via K1 resp. K2, l lit), their midpoint is not."""
    e = Q(1, 20)
    K1 = box_rows((1 - e, Q(-1, 2) - e, -e), (1 + e, Q(-1, 2) + e, e))
    K2 = box_rows((1 - e, Q(1, 2) - e, -e), (1 + e, Q(1, 2) + e, e))
    comps = [K1, K2]
    k = (Q(0), Q(0), Q(0)); l = (Q(0), Q(1), Q(0))
    s1 = (Q(2), Q(-1), Q(0)); s2 = (Q(2), Q(1), Q(0)); m = (Q(2), Q(0), Q(0))
    feas = lambda s: hits_any(s, k, comps) and not hits_any(s, l, comps)
    assert feas(s1) and feas(s2) and not feas(m)
    assert seg_hits(s1, k, K1) and not seg_hits(s1, k, K2)
    assert seg_hits(s2, k, K2) and not seg_hits(s2, k, K1)


def test_three_wedge_localisation_degenerates_exactly_on_the_transversal_quadric():
    """Thm 4.6 / Remark 4.6b: the event planes aff(s u l_i) of three edge lines have linearly dependent normals iff
    s lies on a common transversal of the three lines (an EEE event surface)."""
    rng = random.Random(11)
    def plane_normal(s, P, d):                      # normal of the plane through s containing the line P + t d
        u = tuple(P[t] - s[t] for t in range(3))
        return (u[1] * d[2] - u[2] * d[1], u[2] * d[0] - u[0] * d[2], u[0] * d[1] - u[1] * d[0])
    for _ in range(50):
        s = random_point(rng); m = tuple(Q(rng.randint(1, 9)) for _ in range(3))      # transversal s + t m
        lines = []
        for t in (Q(1), Q(2), Q(-3)):
            P = tuple(s[k] + t * m[k] for k in range(3))
            d = tuple(Q(rng.randint(-9, 9)) for _ in range(3))
            lines.append((P, d))
        Nm = [plane_normal(s, P, d) for P, d in lines]
        if any(not any(n) for n in Nm):
            continue
        assert det3(Nm) == 0
        s2 = tuple(s[k] + Q(rng.randint(1, 9), 13) for k in range(3))                # generic point
        N2 = [plane_normal(s2, P, d) for P, d in lines]
        # generic points are off the quadric (a nonzero determinant is the generic case; equality is measure zero)
        if det3(N2) == 0:
            continue
        assert det3(N2) != 0
