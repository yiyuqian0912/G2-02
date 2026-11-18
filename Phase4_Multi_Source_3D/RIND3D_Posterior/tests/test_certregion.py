"""Certified interface regions (Proposition A', rind3d_posterior.localfirst.certregion; docs/LOCALFIRST_E123.md).

1. The two box/point predicates are SOUND: every "certainly disjoint" answer is checked against an independent exact
   rational reference (hull facets enumerated in Fractions, segment clipped exactly).
2. Proposition A' on exact fixtures: a box, a knife-edge wedge, two near-aligned boxes and a near-coplanar source. The
   feasible set F is sampled densely (uniform, around the true source, and bisection points on the boundary of F
   between feasible and infeasible samples); every feasible sample must lie in R_I and in a box of the octree cover.
   The fixtures are built so that F is a genuine 3D set with boundary, and F's samples include near-degenerate ones.
CPU only (exact GMP predicates through rind3d's native evaluator)."""
import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from rind3d_posterior.localfirst.certregion import (world_edges, admissible_edges, seg_misses_hull, seg_misses_triangle,
                                                    in_region_batch, build_region, MARGIN)
from rind3d_posterior.localfirst.local import interface_edges, edge_free_mask

D100K = Path(__file__).resolve().parents[2] / "RIND3D_Dataset100K"
N = 1024


# ---------------------------------------------------------------------------------------------------- exact references
def _exact_hull_halfspaces(pts):
    """H-representation of conv(pts) in rationals by brute force over point triples (pts: list of Fraction triples)."""
    H = []
    n = len(pts)
    for i, j, k in itertools.combinations(range(n), 3):
        u = [pts[j][t] - pts[i][t] for t in range(3)]; v = [pts[k][t] - pts[i][t] for t in range(3)]
        nrm = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        if nrm == (0, 0, 0):
            continue
        d = sum(nrm[t] * pts[i][t] for t in range(3))
        vals = [sum(nrm[t] * p[t] for t in range(3)) - d for p in pts]
        if all(x <= 0 for x in vals):
            H.append((nrm, d))
        if all(x >= 0 for x in vals):
            H.append((tuple(-c for c in nrm), -d))
        if all(x == 0 for x in vals):
            # coplanar point set: add the in-plane edge halfspaces (normal (p_j - p_i) x n, every point on one side)
            for a_, b_ in itertools.combinations(range(n), 2):
                w = [pts[b_][t] - pts[a_][t] for t in range(3)]
                m = (w[1] * nrm[2] - w[2] * nrm[1], w[2] * nrm[0] - w[0] * nrm[2], w[0] * nrm[1] - w[1] * nrm[0])
                if m == (0, 0, 0):
                    continue
                dm = sum(m[t] * pts[a_][t] for t in range(3))
                vm = [sum(m[t] * p[t] for t in range(3)) - dm for p in pts]
                if all(x <= 0 for x in vm):
                    H.append((m, dm))
                elif all(x >= 0 for x in vm):
                    H.append((tuple(-c for c in m), -dm))
    return H


def _exact_seg_meets_hull(a, b, pts):
    """Exact: does the closed segment [a, b] meet conv(pts)? Handles flat hulls by also clipping with the hull's
    affine hull (if pts are coplanar the triple loop returns the two plane halfspaces)."""
    H = _exact_hull_halfspaces(pts)
    lo, hi = Fraction(0), Fraction(1)
    for nrm, d in H:
        A = sum(nrm[t] * a[t] for t in range(3)) - d; B = sum(nrm[t] * (b[t] - a[t]) for t in range(3))
        if B == 0:
            if A > 0:
                return False
        elif B > 0:
            hi = min(hi, -A / B)
        else:
            lo = max(lo, -A / B)
        if lo > hi:
            return False
    return True


def _F(v):
    return [Fraction(float(x)) for x in v]


@pytest.mark.parametrize("seed", range(4))
def test_seg_misses_hull_is_sound(seed):
    rng = np.random.default_rng(seed); claims = 0; kept_true_disjoint = 0
    for _ in range(400):
        lo = rng.uniform(0, 0.9, 3); hi = lo + rng.uniform(0.001, 0.1, 3)
        l = rng.uniform(0, 1, 3); k = l.copy(); k[rng.integers(3)] += 1 / N
        c = rng.uniform(0, 1, 3); a = c + rng.normal(0, 0.05, 3); b = c + rng.normal(0, 0.05, 3)
        if rng.random() < 0.3:                                   # near-touching configurations
            p = 0.5 * (lo + hi) * rng.random() + (1 - rng.random()) * l
            a = p + rng.normal(0, 1e-3, 3); b = 2 * p - a
        pts = [_F(x) for x in itertools.product(*zip(lo, hi))] + [_F(l), _F(k)]
        exact = _exact_seg_meets_hull(_F(a), _F(b), pts)
        claim = seg_misses_hull(a, b, l, k, lo, hi, MARGIN)
        if claim:
            claims += 1
            assert not exact, "box test excluded a box whose hull meets the edge"
        elif not exact:
            kept_true_disjoint += 1
    assert claims > 100
    assert kept_true_disjoint <= 0.02 * 400                      # tight: undecided only within the margin


@pytest.mark.parametrize("seed", range(4))
def test_seg_misses_triangle_is_sound(seed):
    rng = np.random.default_rng(10 + seed); claims = 0; loose = 0
    for _ in range(600):
        s, l = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3); k = l.copy(); k[rng.integers(3)] += 1 / N
        c = (s + l) / 2 + rng.normal(0, 0.02, 3); a = c + rng.normal(0, 0.05, 3); b = c + rng.normal(0, 0.05, 3)
        exact = _exact_seg_meets_hull(_F(a), _F(b), [_F(s), _F(l), _F(k)])
        claim = seg_misses_triangle(a, b, s, l, k, MARGIN)
        if claim:
            claims += 1
            assert not exact
        elif not exact:
            loose += 1
    assert claims > 300 and loose <= 0.02 * 600


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fac():
    from rind3d.rendering.backends import BackendFactory
    return BackendFactory(cuda_allowed=False)


def _ctx(fac, leaves):
    sys.path.insert(0, str(D100K / "tests"))
    from test_local_geometry_v3 import _ctx as mk
    return mk(fac, leaves)


BOXN = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
WEDGE_N = np.array([[-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [1, 0, 1]], float)


def _leaf(c, M, normals, offsets):
    from rind.contracts import make_leaf
    return make_leaf("convex_halfspaces_v1", 3, center=np.asarray(c, "<f8"), matrix=np.asarray(M, "<f8"),
                     normal=np.asarray(normals, "<f8"), plane_offset=np.asarray(offsets, "<f8"))


def _rot(seed):
    from scipy.spatial.transform import Rotation
    return Rotation.random(random_state=seed).as_matrix()


def _scene(name):
    """(leaves, true source s, window origin o, side L)."""
    if name == "box":
        return [_leaf([.5, .5, .5], _rot(1) @ np.diag([.06, .04, .05]), BOXN, np.ones(6))], np.array([.47, .52, .2]), np.array([500, 520, 700]), 24
    if name == "knife":
        return [_leaf([.5, .5, .5], _rot(2) @ np.diag([.15, .1, .02]), WEDGE_N, [1, 1, 1, 1, .4])], np.array([.45, .55, .18]), np.array([500, 540, 690]), 24
    if name == "two_boxes":
        R = _rot(3)
        return [_leaf([.5, .5, .45], R @ np.diag([.05, .05, .02]), BOXN, np.ones(6)),
                _leaf([.5, .5, .55], R @ np.diag([.0501, .05, .02]) , BOXN, np.ones(6))], np.array([.52, .5, .2]), np.array([495, 505, 760]), 24
    if name == "coplanar":
        # the source lies (to 1e-9) on the extension of a box facet plane x = 0.45
        return [_leaf([.5, .5, .5], np.diag([.05, .06, .04]), BOXN, np.ones(6))], np.array([.45 + 1e-9, .5, .2]), np.array([455, 500, 720]), 24
    raise KeyError(name)


def _place_window(ctx, s, L, seed=0):
    """A window on the shadow boundary: centred on the ray from s through an occluder edge point, beyond the occluder,
    with a direct interface and not containing s."""
    from rind3d_1k.physical import geometry_rows
    P, K = geometry_rows(ctx); EA, EB, _, _ = world_edges(P, K)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        e = rng.integers(len(EA)); v = EA[e] + rng.uniform(0.2, 0.8) * (EB[e] - EA[e])
        t = rng.uniform(1.2, 2.5); q = s + t * (v - s)
        if not np.all((q > 0.05) & (q < 0.95)):
            continue
        o = np.floor(q * N).astype(np.int64) - L // 2
        if np.all((s >= o / N) & (s <= (o + L) / N)):
            continue
        cls, _, _ = _window(ctx, s, o, L)
        lit, _ = interface_edges(cls)
        if len(lit) >= 40 and (cls == 2).mean() < 0.5:
            return o
    raise RuntimeError("no window found")


def _window(ctx, s, o, L):
    g = np.stack(np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 3)      # x, y, z
    X = (o[None] + g + .5) / N
    y, sol, _ = ctx.bind(s).native.points(np.ascontiguousarray(X))
    cls = np.where(sol == 1, 2, np.where(y == 1, 1, 0)).astype(np.uint8).reshape(L, L, L)
    return cls, X, g


def _feasible(ctx, cls, X, Q, o, L):
    free = cls.reshape(-1) != 2; lit = cls.reshape(-1)[free] == 1
    out = np.zeros(len(Q), bool)
    from rind3d_1k.physical import legal_source
    lo, hi = o / N, (o + L) / N
    for j, q in enumerate(Q):
        if np.all((q >= lo) & (q <= hi)) or not np.all((q > 0) & (q < 1)):
            continue
        try:
            if not legal_source(ctx, q):
                continue
        except Exception:
            continue
        y, _, _ = ctx.bind(q).native.points(np.ascontiguousarray(X[free]))
        out[j] = np.array_equal(y == 1, lit)
    return out


@pytest.mark.parametrize("name", ["box", "knife", "two_boxes", "coplanar"])
def test_proposition_A_full_feasible_set_inclusion(fac, name):
    from rind3d_1k.physical import geometry_rows
    leaves, s, _, L = _scene(name)
    ctx = _ctx(fac, leaves)
    o = _place_window(ctx, s, L)
    cls, X, g = _window(ctx, s, o, L)
    lit, dark = interface_edges(cls)
    assert len(lit) > 0, "fixture must have a direct interface"
    rows = [c.rows for c in ctx.components]
    free = edge_free_mask(rows, o, N, lit, dark)
    Lp = (o + lit[free] + .5) / N; Kp = (o + dark[free] + .5) / N
    P, K = geometry_rows(ctx)
    EA, EB, Fr, _ = world_edges(P, K)
    off, idx = admissible_edges(Fr, Lp, Kp)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Lp), min(64, len(Lp)), replace=False)
    o_, i_ = admissible_edges(Fr, Lp[sel], Kp[sel])
    reg = build_region(EA, EB, Lp[sel], Kp[sel], o_, i_, o / N, (o + L) / N, D=7)
    # dense samples of F: uniform, around the true source, and boundary points by bisection
    Q = np.r_[s[None], rng.uniform(0, 1, (600, 3))] 
    Q = np.r_[Q, *[s + rng.normal(0, sc, (400, 3)) for sc in (1e-5, 1e-4, 1e-3, 1e-2)]]
    feas = _feasible(ctx, cls, X, Q, o, L)
    assert feas[0] and feas.sum() >= 5, "fixture F too small to be informative"
    Fp = Q[feas]; Ip = Q[~feas & np.all((Q > 0) & (Q < 1), 1)][:400]
    B = []
    for a in Fp[rng.choice(len(Fp), min(40, len(Fp)), replace=False)]:
        for b in Ip[rng.choice(len(Ip), 3, replace=False)]:
            for _ in range(18):
                m = 0.5 * (a + b)
                if _feasible(ctx, cls, X, m[None], o, L)[0]:
                    a = m
                else:
                    b = m
            B.append(a)
    Fall = np.r_[Fp, np.array(B)]
    assert _feasible(ctx, cls, X, Fall, o, L).all()
    # Proposition A' with ALL free edges, and with the 64-edge subset used by the octree
    assert in_region_batch(Fall, EA, EB, Lp, Kp, off, idx, MARGIN).all()
    assert in_region_batch(Fall, EA, EB, Lp[sel], Kp[sel], o_, i_, MARGIN).all()
    # every feasible sample lies in a kept box of the cover
    h = 1.0 / (1 << reg["dep"]); inside = np.zeros(len(Fall), bool)
    for lo_, hh in zip(reg["lo"], h):
        inside |= np.all((Fall >= lo_ - 1e-12) & (Fall <= lo_ + hh + 1e-12), 1)
    assert inside.all()
    # and the region is informative: it is much smaller than the cube
    assert len(reg["lo"]) < 0.2 * 128 ** 3


@pytest.mark.parametrize("name", ["box", "knife", "two_boxes"])
def test_root_clip_wedges_contain_the_feasible_set(fac, name):
    """The E2 clip rows (unique explanations on a box, wedge half-spaces) contain every feasible sample in that box."""
    from rind3d_1k.physical import geometry_rows
    from rind3d_posterior.localfirst.certregion import box_explanations, root_clips
    leaves, s, _, L = _scene(name)
    ctx = _ctx(fac, leaves); o = _place_window(ctx, s, L)
    cls, X, g = _window(ctx, s, o, L)
    lit, dark = interface_edges(cls)
    free = edge_free_mask([c.rows for c in ctx.components], o, N, lit, dark)
    Lp = (o + lit[free] + .5) / N; Kp = (o + dark[free] + .5) / N
    P, K = geometry_rows(ctx); EA, EB, Fr, _ = world_edges(P, K)
    off, idx = admissible_edges(Fr, Lp, Kp)
    rng = np.random.default_rng(1)
    Q = np.r_[s[None], *[s + rng.normal(0, sc, (300, 3)) for sc in (1e-6, 1e-5, 1e-4, 1e-3)]]
    Fp = Q[_feasible(ctx, cls, X, Q, o, L)]
    assert len(Fp) >= 2
    # boxes of side 1/128 around feasible samples
    blo = np.unique(np.floor(Fp * 128) / 128, axis=0)
    expl = box_explanations(EA, EB, Lp, Kp, off, idx, blo, np.full(len(blo), 1 / 128), MARGIN)
    coff, rows, empty = root_clips(EA, EB, Lp, Kp, expl)
    assert not empty.any()
    assert len(rows) > 0
    for q, lo in enumerate(blo):
        inb = np.all((Fp >= lo - 1e-12) & (Fp <= lo + 1 / 128 + 1e-12), 1)
        R = rows[coff[q]:coff[q + 1]]
        if len(R):
            assert np.all(Fp[inb] @ R[:, :3].T + R[:, 3] <= 0)


def test_octree_explanations_equal_brute_force():
    """explanations_for_roots (one shared octree descent) gives the same per-root explanation summaries as the
    brute-force box_explanations scan (unique edge / several / none), on random edges and boxes."""
    from rind3d_posterior.localfirst.certregion import explanations_for_roots, box_explanations
    rng = np.random.default_rng(7)
    EA = rng.uniform(0.3, 0.7, (40, 3)); EB = EA + rng.normal(0, 0.05, (40, 3))
    Lp = rng.uniform(0.8, 0.9, (24, 3)); Kp = Lp.copy(); Kp[:, 0] += 1 / N
    off = np.arange(0, 24 * 40 + 1, 40).astype(np.int64); idx = np.tile(np.arange(40, dtype=np.int32), 24)
    roots = np.unique(np.c_[np.full(60, 4), rng.integers(0, 12, (60, 3))], axis=0)
    ex, _ = explanations_for_roots(EA, EB, Lp, Kp, off, idx, roots, np.array([.9, .9, .9]), np.array([.95, .95, .95]))
    h = 1.0 / 16; bf = box_explanations(EA, EB, Lp, Kp, off, idx, roots[:, 1:] * h, np.full(len(roots), h), MARGIN)
    # a root excluded by the octree has some interface edge without explanation in the brute force too
    for q in range(len(roots)):
        if np.all(ex[q] == -1):
            assert np.any(bf[q] == -1)
        else:
            assert np.array_equal(ex[q], bf[q])


def test_filter_from_leaves_covers_coarse_leaves():
    """Regression (E2 harness): a cover leaf coarser than a 128^3 cell (test-budget stop) must make every cell it covers
    full in the outer-bound filter; leaves finer than a cell are listed as boxes."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "localfirst"))
    from e12 import _filter_from_leaves
    lo = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5 + 1 / 512, 0.5, 0.5]])
    dep = np.array([5, 7, 9])
    cells, kind, start, ln, boxes = _filter_from_leaves(lo, dep)
    n = 1 << (7 - 5)
    assert np.sum(kind == 2) == n ** 3 + 1                  # the depth-5 leaf (4^3 cells) and the depth-7 leaf
    c9 = (64 * 128 + 64) * 128 + 64
    assert c9 in cells and kind[np.searchsorted(cells, c9)] == 2   # the depth-9 leaf lies in the full depth-7 cell
    keep = np.array([0, 1, 5000])
    cells, kind, start, ln, boxes = _filter_from_leaves(lo, dep, keep_cells=keep)
    assert set(cells.tolist()) <= set(keep.tolist()) and 0 in cells and np.all(kind == 2)


def test_root_clips_numba_equals_python():
    """root_clips_numba writes the same clip rows (as sets per root) as the reference root_clips."""
    from rind3d_posterior.localfirst.certregion import root_clips, root_clips_numba
    rng = np.random.default_rng(11)
    EA = rng.uniform(0.3, 0.7, (30, 3)); EB = EA + rng.normal(0, 0.05, (30, 3))
    Lp = rng.uniform(0.8, 0.9, (50, 3)); Kp = Lp.copy(); Kp[np.arange(50), rng.integers(0, 3, 50)] += 1 / N
    expl = rng.integers(-2, 30, (40, 50)); expl[rng.random((40, 50)) < 0.02] = -1
    o1, r1, e1 = root_clips(EA, EB, Lp, Kp, expl)
    o2, r2, e2 = root_clips_numba(EA, EB, Lp, Kp, expl)
    assert np.array_equal(e1, e2)
    for q in range(len(expl)):
        A = {tuple(np.round(x, 12)) for x in r1[o1[q]:o1[q + 1]]}; B = {tuple(np.round(x, 12)) for x in r2[o2[q]:o2[q + 1]]}
        assert A == B
