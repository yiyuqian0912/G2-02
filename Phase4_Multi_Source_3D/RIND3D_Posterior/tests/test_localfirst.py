"""Local-first research module (rind3d_posterior.localfirst): X-only interface analysis and the free-edge diagnostic.
The certified pathway itself is the tau-pencil engine (covered by test_regression.py); see
docs/LOCALFIRST_COMPARISON.md for its measured agreement with the reference."""
from fractions import Fraction

import numpy as np
import pytest

from rind3d_posterior.localfirst.local import (interface_edges, plane_components, conditioning, local_band_cells,
                                               select_receivers, edge_free_mask)


def _brute_edges(cls):
    L = cls.shape[0]; out = set()
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    X, Y, Z = x + d[0], y + d[1], z + d[2]
                    if X < L and Y < L and Z < L:
                        a, b = cls[x, y, z], cls[X, Y, Z]
                        if a != 2 and b != 2 and a != b:
                            lit, dark = ((x, y, z), (X, Y, Z)) if a == 1 else ((X, Y, Z), (x, y, z))
                            out.add((lit, dark))
    return out


@pytest.mark.parametrize("seed", range(4))
def test_interface_edges_match_bruteforce(seed):
    rng = np.random.default_rng(seed)
    cls = rng.choice([0, 1, 2], size=(7, 7, 7), p=[.4, .4, .2]).astype(np.uint8)
    li, da = interface_edges(cls)
    got = {(tuple(a), tuple(b)) for a, b in zip(li.tolist(), da.tolist())}
    assert got == _brute_edges(cls)
    assert np.all((cls[tuple(li.T)] == 1) & (cls[tuple(da.T)] == 0)) and np.all(np.abs(li - da).sum(1) == 1)


def test_plane_components_recover_two_planes_and_rank():
    L = 64
    x, y, z = np.meshgrid(*(np.arange(L),) * 3, indexing="ij")
    n1 = np.array([1.0, 0.3, 0.1]); n2 = np.array([0.1, 1.0, -0.4])
    lit = ((x - 20) * n1[0] + (y - 32) * n1[1] + (z - 32) * n1[2] > 0) & ((x - 32) * n2[0] + (y - 40) * n2[1] + (z - 32) * n2[2] < 0)
    cls = lit.astype(np.uint8)
    li, da = interface_edges(cls)
    comps = plane_components(li, da, seed=0)
    assert len(comps) >= 2
    normals = [c["normal"] for c in comps[:2]]
    targets = [n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2)]
    for t in targets:
        assert max(abs(float(n @ t)) for n in normals) > 0.98
    assert conditioning(comps[:2])["rank"] == 2


def test_select_receivers_budget_and_validity():
    rng = np.random.default_rng(1)
    L = 32
    x, y, z = np.meshgrid(*(np.arange(L),) * 3, indexing="ij")
    cls = ((x + 0.4 * y - 0.2 * z) > 14).astype(np.uint8)
    cls[rng.random(cls.shape) < 0.03] = 2
    vox, comps = select_receivers(cls, 64, 32, seed=0)
    assert len(vox) <= 64 + 32 and len(vox) > 32
    assert np.all(cls[tuple(vox.T)] != 2)
    li, da = interface_edges(cls)
    boundary = {tuple(v) for v in np.r_[li, da].tolist()}
    assert sum(tuple(v) in boundary for v in vox.tolist()) >= 32       # most picks are interface endpoints


def _exact_hit(rows, u, v):
    lo, hi = Fraction(0), Fraction(1)
    for r in rows:
        r = [Fraction(a) for a in r]
        A = sum(r[k] * u[k] for k in range(3)) + r[3]; B = sum(r[k] * (v[k] - u[k]) for k in range(3))
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


@pytest.mark.parametrize("seed", range(3))
def test_edge_free_mask_matches_exact(seed):
    """Random thin rotated boxes near a window; every receiver edge tested exactly with rationals."""
    rng = np.random.default_rng(seed); N = 1024
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    half = np.array([0.01, 0.01, rng.uniform(0.0002, 0.002)]); c = np.array([0.5, 0.5, 0.5]) + rng.uniform(-2, 2, 3) / N
    rows = []
    for i in range(3):
        for sgn in (-1, 1):
            n = sgn * Q[:, i]
            rows.append(tuple(Fraction(float(v)) for v in n) + (Fraction(float(-(n @ c) - half[i])),))
    o = np.floor(c * N).astype(int) - 8
    lit = rng.integers(0, 16, (400, 3)); ax = rng.integers(0, 3, 400); dark = lit.copy(); dark[np.arange(400), ax] += 1
    free = edge_free_mask([rows], o, N, lit, dark)
    for i in range(400):
        u = [Fraction(2 * int(o[k] + lit[i, k]) + 1, 2 * N) for k in range(3)]
        v = [Fraction(2 * int(o[k] + dark[i, k]) + 1, 2 * N) for k in range(3)]
        assert bool(free[i]) == (not _exact_hit(rows, u, v))
    assert (~free).sum() > 0                                            # some edges cross the thin box


def test_local_band_is_heuristic_shape():
    comps = [dict(n=100, centroid=np.array([32.0, 32, 32]), normal=np.array([0, 0, 1.0]), extent=40.0)]
    band = local_band_cells(comps, np.array([480, 480, 480]), 1024)
    assert band.dtype == bool and 0 < band.sum() < 128 ** 3
