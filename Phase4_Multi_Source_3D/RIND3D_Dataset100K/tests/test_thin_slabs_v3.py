"""Contract v3 thin slabs (rind3d_100k.geometry_v3, docs/THIN_SLABS_V3.md): the thickness law, the identity with v0.1 at
T_min = 0, the exact 6-connectivity checker against dense labelling, the infinite-slab theorem and its sharpness, rim
isolation, continuous/discrete consistency and near-surface visibility. CPU only (certified CPU predicates)."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from rind3d.lite_generators import generate as v01_generate
from rind3d_100k.geometry_v3 import generate_v3, sampled_solid_components, thin_slab_report, thin_log2_lo

HERE = Path(__file__).resolve().parent.parent
SIX = ndimage.generate_binary_structure(3, 1)


@pytest.fixture(scope="module")
def fac():
    from rind3d.rendering.backends import BackendFactory
    return BackendFactory(cuda_allowed=False)


def _ctx(fac, bundle):
    from rind3d_1k.physical import POLICY
    return fac.context(bundle, POLICY)


@pytest.mark.parametrize("family", ["thin_slabs", "occluder_clusters", "rotated_boxes", "corridors", "multiscale_convex"])
def test_t0_is_byte_identical_to_v01(family):
    for slot in range(6):
        b0, r0 = v01_generate(20261216, slot, family, 0)
        b1, r1 = generate_v3(20261216, slot, family, 0, thin_min_vox=0)
        assert r0["geometry_id"] == r1["geometry_id"] and r0["objects"] == r1["objects"]


@pytest.mark.parametrize("tmin", [1, 2, 3])
def test_thin_law_minimum_and_paired_draws(tmin):
    lo2 = thin_log2_lo(tmin)
    assert abs(2 ** lo2 * 2 * 1024 - tmin) < 1e-9
    th = []
    for slot in range(40):
        _, r0 = v01_generate(20261216, slot, "thin_slabs", 0)
        _, r1 = generate_v3(20261216, slot, "thin_slabs", 0, thin_min_vox=tmin)
        assert len(r0["objects"]) == len(r1["objects"])
        for a, b in zip(r0["objects"], r1["objects"]):
            assert a["orientation"] == b["orientation"] and a["half_parameters"][:2] == b["half_parameters"][:2]
            t = 2 * b["half_parameters"][2] * 1024
            assert tmin * (1 - 1e-12) <= t <= 64 * (1 + 1e-12)
            th.append(t)
    # log-uniform on [tmin, 64]: the median is the geometric mean (loose check)
    assert 0.5 < np.median(th) / np.sqrt(tmin * 64) < 2.0


def _dense(rows, lo, hi, N=1024):
    R = np.array([[float(v) for v in r] for r in rows])
    ax = [np.arange(lo[a], hi[a] + 1) for a in range(3)]
    X, Y, Z = np.meshgrid(*[(g + .5) / N for g in ax], indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    ins = np.all(P @ R[:, :3].T + R[:, 3] <= 0, axis=1).reshape(X.shape)
    return ins, ndimage.label(ins, structure=SIX)[1]


def _slab_rows(n, d0, T, N=1024):
    """Unbounded slab d0 <= n.(v + 1/2) <= d0 + T (voxel units), as two exact rational rows in world units."""
    n = [Fraction(float(x)) for x in n]
    return [(n[0], n[1], n[2], -Fraction(float(d0 + T)) / N),      # n.x - (d0 + T)/N <= 0
            (-n[0], -n[1], -n[2], Fraction(float(d0)) / N)]        # -n.x + d0/N <= 0


@pytest.mark.parametrize("seed", range(12))
def test_infinite_slab_theorem_thickness_l1_norm_suffices(seed):
    """Theorem 1 of THIN_SLABS_V3.md: T >= ||n||_1 (unit n, voxel units) gives a 6-connected sampling. Checked on a
    40 x 40 column window with the slab away from the top and bottom (only side cuts, which the proof allows)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    n = n[np.argsort(np.abs(n))]                                    # dominant component last (z)
    T = float(np.abs(n).sum())
    lo, hi = np.array([300, 300, 0]), np.array([339, 339, 1023])
    d0 = n @ (np.array([320, 320, 512]) + .5) - T / 2 + rng.uniform(-.5, .5)
    ins, ncomp = _dense(_slab_rows(n, d0, T), lo, hi)
    assert ins.sum() > 0 and ncomp == 1


def test_one_voxel_diagonal_slab_is_totally_disconnected():
    """Sharpness: n = (1,1,1)/sqrt 3, T = 1 voxel, placed so that a single level of x + y + z is inside: no two samples
    are 6-adjacent (every 6-step changes x + y + z by 1)."""
    n = np.ones(3) / np.sqrt(3)
    k = 3 * 320 + 1.5                       # sum of (v + 1/2) over the three axes at level x + y + z = 3*320 ... + 1.5
    d0 = (k - 0.9) / np.sqrt(3)             # [d0, d0 + 1] in n-units contains only the level sum(v+1/2) = k
    ins, ncomp = _dense(_slab_rows(n, d0, 1.0), np.array([300, 300, 300]), np.array([339, 339, 339]))
    assert ins.sum() > 50 and ncomp == ins.sum()


@pytest.mark.parametrize("slot", range(8))
def test_checker_matches_dense_labelling_on_generated_slabs(fac, slot):
    b, rec = generate_v3(20261216, slot, "thin_slabs", 0, thin_min_vox=1)
    if rec["proposal_failures"]:
        pytest.skip("geometry proposal failure")
    ctx = _ctx(fac, b); by = {c.leaf_id: c.rows for c in ctx.components}
    for ob in rec["objects"][:3]:
        rows = [tuple(Fraction(v) for v in r) for r in by[ob["leaf_id"]]]
        rep = sampled_solid_components(rows)
        c = np.array(ob["center"]); e = np.array(ob["conservative_extent"])
        lo = np.clip(np.floor((c - e) * 1024 - 1), 0, 1023).astype(int); hi = np.clip(np.ceil((c + e) * 1024 + 1), 0, 1023).astype(int)
        ins, ncomp = _dense(rows, lo, hi)
        assert rep["n_vox"] == int(ins.sum()) and rep["n_components"] == ncomp


def test_rim_isolation_is_detected_regardless_of_thickness(fac):
    """A 3.8-voxel slab (above the sqrt(3) bound) whose sampling has one isolated rim voxel (seed 20261216, slot 8):
    thickness alone does not guarantee connectivity of a finite rotated slab; the checker rejects it."""
    b, rec = generate_v3(20261216, 8, "thin_slabs", 0, thin_min_vox=2)
    rep = thin_slab_report(_ctx(fac, b), rec, min_thickness_vox=2)
    bad = [r for r in rep if not r["ok"]]
    assert bad and all(r["thickness_vox"] >= np.sqrt(3) and r["n_components"] >= 2 for r in bad)


def test_continuous_and_discrete_solid_agree(fac):
    """The checker's sampled set and the certified SOLID predicate of the same canonical geometry agree exactly on the
    receivers around a slab (including the ones within 0.1 voxel of its surface)."""
    b, rec = generate_v3(20261216, 3, "thin_slabs", 0, thin_min_vox=2)
    ctx = _ctx(fac, b); by = {c.leaf_id: c.rows for c in ctx.components}
    ob = rec["objects"][0]
    rows = [tuple(Fraction(v) for v in r) for r in by[ob["leaf_id"]]]
    R = np.array([[float(v) for v in r] for r in rows])
    rng = np.random.default_rng(0)
    c = np.array(ob["center"]); Rot = np.array(ob["orientation"]); half = np.array(ob["half_parameters"])
    loc = rng.uniform(-1.2, 1.2, (4000, 3)) * half                  # around the slab, many points near its faces
    vox = np.floor((c + loc @ Rot.T) * 1024).astype(np.int64)
    vox = vox[np.all((vox >= 0) & (vox < 1024), axis=1)]
    pts = (vox + .5) / 1024
    mine = np.all(pts @ R[:, :3].T + R[:, 3] <= 0, axis=1)
    other = [ob2 for ob2 in rec["objects"] if ob2 is not ob]
    cert = ctx.contains(np.ascontiguousarray(pts))
    # cert is the union of all bodies; compare where no other body contains the point
    R2 = [np.array([[float(v) for v in r] for r in by[o["leaf_id"]]]) for o in other]
    alone = ~np.any([np.all(pts @ r[:, :3].T + r[:, 3] <= 0, axis=1) for r in R2], axis=0) if R2 else np.ones(len(pts), bool)
    assert mine.any() and np.array_equal(mine[alone], cert[alone])


def _exact_segment_hits(rows, s, x):
    """Closed segment [s, x] meets the closed convex body (exact rationals, Cyrus-Beck)."""
    s = [Fraction(float(v)) for v in s]; x = [Fraction(float(v)) for v in x]
    lo, hi = Fraction(0), Fraction(1)
    for a, b, c, d in rows:
        al = a * s[0] + b * s[1] + c * s[2] + d; be = a * (x[0] - s[0]) + b * (x[1] - s[1]) + c * (x[2] - s[2])
        if be == 0:
            if al > 0:
                return False
        elif be > 0:
            hi = min(hi, -al / be)
        else:
            lo = max(lo, -al / be)
        if lo > hi:
            return False
    return True


def test_near_surface_visibility_matches_exact_segment_test(fac):
    """Receivers within ~1 voxel of a rotated 2+ voxel slab, sources on both sides: the certified CPU visibility of the
    full scene equals the exact rational segment/body test (the scene's slab is the only body on those segments)."""
    b, rec = generate_v3(20261216, 3, "thin_slabs", 0, thin_min_vox=2)
    ctx = _ctx(fac, b); by = {c.leaf_id: c.rows for c in ctx.components}
    bodies = [[tuple(Fraction(v) for v in r) for r in by[o["leaf_id"]]] for o in rec["objects"]]
    ob = rec["objects"][0]; c = np.array(ob["center"]); Rot = np.array(ob["orientation"]); half = np.array(ob["half_parameters"])
    rng = np.random.default_rng(1)
    loc = np.c_[rng.uniform(-.8, .8, (300, 2)) * half[:2], rng.choice([-1, 1], 300) * (half[2] + rng.uniform(0, 1.5, 300) / 1024)]
    vox = np.floor((c + loc @ Rot.T) * 1024).astype(np.int64); X = (vox + .5) / 1024
    from rind3d_1k.physical import legal_source
    for sign in (-1, 1):
        s = c + Rot @ np.array([0, 0, sign * 0.15])
        if legal_source(ctx, s) is None:
            continue
        y, _, _ = ctx.bind(s).native.points(np.ascontiguousarray(X))
        free = ~ctx.contains(np.ascontiguousarray(X))
        for i in np.flatnonzero(free):
            lit_exact = not any(_exact_segment_hits(bd, s, X[i]) for bd in bodies)
            assert bool(y[i] == 1) == lit_exact
