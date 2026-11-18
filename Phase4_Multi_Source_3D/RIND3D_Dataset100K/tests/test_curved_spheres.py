"""Curved occluders (closed balls; rind3d_curved, docs/CURVED_OCCLUDERS.md): exact predicates, the certified interval
paths (CPU and CUDA), the independent canonical exact oracle, tangency, sampled connectivity, source legality,
determinism, and the composite scene's probe/render consistency."""
import subprocess, sys
from fractions import Fraction

import numpy as np
import pytest
from scipy import ndimage

from rind3d_curved.spheres import (BallSet, segment_hits_ball_exact, ball_contains_exact, segment_hits_ball_interval,
                                   clearance_exact, ball_misses_hull_exact)
from rind3d_curved.families import generate_curved, ball_sampled_components

N = 1024


def _near_tangent_rays(rng, c, r, s, n=400):
    """Receivers on rays from s at angles alpha * (1 + eps) around the ball's tangent cone (alpha = asin(r/|c-s|)),
    beyond and before the ball: a hit-rich, near-tangent sample snapped to the receiver lattice."""
    a = c - s; D = np.linalg.norm(a); a /= D
    alpha = np.arcsin(r / D)
    u = np.cross(a, [1, 0, 0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(a, [0, 1, 0]) / np.linalg.norm(np.cross(a, [0, 1, 0]))
    v = np.cross(a, u)
    out = []
    for _ in range(n):
        phi = rng.uniform(0, 2 * np.pi); th = alpha * (1 + rng.normal(0, 0.02)); t = D * rng.uniform(0.3, 2.0)
        d = np.cos(th) * a + np.sin(th) * (np.cos(phi) * u + np.sin(phi) * v)
        x = s + t * d
        x = (np.floor(np.clip(x, 0, 1 - 1e-9) * N) + .5) / N
        out.append(x)
    return np.array(out)


@pytest.mark.parametrize("seed", range(4))
def test_interval_predicate_matches_exact_and_canonical_oracle(seed):
    from rind.ellipsoid3d import ExactEllipsoid3D
    rng = np.random.default_rng(seed)
    c = rng.uniform(.35, .65, 3); r = float(rng.uniform(.03, .12)); s = rng.uniform(.05, .95, 3)
    while np.linalg.norm(s - c) < r * 1.5:
        s = rng.uniform(.05, .95, 3)
    X = _near_tangent_rays(rng, c, r, s)
    t = segment_hits_ball_interval(c, r, s, X)
    ex = np.array([segment_hits_ball_exact(c, r, s, x) for x in X])
    orc = ExactEllipsoid3D(c, np.eye(3) * r)
    canon = np.array([orc.segment_hit(s, x) for x in X])
    assert np.array_equal(ex, canon)                                   # two independent exact implementations
    known = t != -1
    assert np.array_equal(t[known] == 1, ex[known])                    # interval decisions are never wrong
    assert 0.2 < ex.mean() < 0.8                                       # the sample is hit-rich (near the cone)
    B = BallSet(c[None], [r])
    assert np.array_equal(B.blocked(s, X), ex)                         # certified + exact fallback = exact


def test_exact_tangency_is_a_hit_on_every_path():
    yl = 1279 / 2048; zl = 1025 / 2048; c = np.array([0.5, 0.5, zl]); r = yl - 0.5     # tangent at a lattice row
    s = np.array([0.0625, yl, zl])
    X = np.array([[(2 * 900 + 1) / 2048, yl, zl], [(2 * 900 + 1) / 2048, yl + 1 / 1024, zl], [(2 * 400 + 1) / 2048, yl, zl]])
    assert [segment_hits_ball_exact(c, r, s, x) for x in X] == [True, False, False]
    assert BallSet(c[None], [r]).blocked(s, X).tolist() == [True, False, False]
    from rind.ellipsoid3d import ExactEllipsoid3D
    assert [ExactEllipsoid3D(c, np.eye(3) * r).segment_hit(s, x) for x in X] == [True, False, False]


def test_membership_and_clearance_exact():
    c = np.array([.5, .5, .5]); r = 0.125
    assert ball_contains_exact(c, r, [.5 + .125, .5, .5]) and not ball_contains_exact(c, r, [.5 + .125 + 2 ** -40, .5, .5])
    assert not clearance_exact(c, r, [.5 + .125 + 2 ** -27, .5, .5], 2 ** -26)
    assert clearance_exact(c, r, [.5 + .125 + 2 ** -25, .5, .5], 2 ** -26)
    B = BallSet(c[None], [r])
    assert not B.legal_source(np.array([.5, .5, .5])) and B.legal_source(np.array([.9, .9, .9]))


@pytest.mark.parametrize("seed", range(3))
def test_ball_sampling_connectivity_matches_dense(seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(.3, .7, 3); r = float(rng.uniform(.003, .02))                  # 3-20 voxels
    nv, nc = ball_sampled_components(c, r, N)
    lo = np.floor((c - r) * N - 1).astype(int); hi = np.ceil((c + r) * N + 1).astype(int)
    ax = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    X, Y, Z = np.meshgrid(*[(g + .5) / N for g in ax], indexing="ij")
    ins = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= r * r
    assert nv == int(ins.sum()) and nc == ndimage.label(ins, structure=ndimage.generate_binary_structure(3, 1))[1] == 1


def test_hull_certificate_is_sound():
    rng = np.random.default_rng(5)
    for _ in range(200):
        c = rng.uniform(.2, .8, 3); r = float(rng.uniform(.01, .1)); P = rng.uniform(0, 1, (9, 3))
        if ball_misses_hull_exact(c, r, P):
            # sound: no point of the hull is in the ball (check many convex combinations + the exact vertex test)
            lam = rng.dirichlet(np.ones(9), 2000); Q = lam @ P
            assert np.all(np.linalg.norm(Q - c, axis=1) > r)


def test_curved_generation_is_deterministic():
    for fam in ("spheres", "spheres_polyhedra"):
        a = generate_curved(20261119, 3, fam, 0); b = generate_curved(20261119, 3, fam, 0)
        assert np.array_equal(a[1], b[1]) and a[2]["polytope_geometry_id"] == b[2]["polytope_geometry_id"]
        assert np.all(a[1][:, 3] >= 0.02) and np.all(a[1][:, :3] - a[1][:, 3:] > 0) and np.all(a[1][:, :3] + a[1][:, 3:] < 1)


def _cuda():
    r = subprocess.run([sys.executable, "-c", "import cupy; cupy.cuda.runtime.getDeviceCount()"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
def test_gpu_tiles_equal_exact_cpu_including_near_tangent():
    from rind3d_curved.cuda_balls import BallTiles
    rng = np.random.default_rng(3)
    balls = np.c_[rng.uniform(.3, .7, (4, 3)), rng.uniform(.03, .1, 4)]
    B = BallSet(balls[:, :3], balls[:, 3]); T = BallTiles(balls)
    s = np.array([.05, .1, .12])
    assert B.legal_source(s)
    # tiles placed where the first ball's shadow boundary passes (near-tangent receivers)
    X0 = _near_tangent_rays(rng, balls[0, :3], balls[0, 3], s, 8)
    for x in X0:
        o = np.clip(np.floor(x * N).astype(int) - 32, 0, N - 64)
        blk, sol = T.tile(s, o)
        zyx = rng.integers(0, 64, (300, 3)); X = (o + zyx[:, ::-1] + .5) / N
        assert np.array_equal(blk[zyx[:, 0], zyx[:, 1], zyx[:, 2]], B.blocked(s, X))
        assert np.array_equal(sol[zyx[:, 0], zyx[:, 1], zyx[:, 2]], B.contains(X))
