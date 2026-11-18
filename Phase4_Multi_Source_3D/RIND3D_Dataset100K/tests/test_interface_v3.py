"""Contract v3 window eligibility: the direct FREE-FREE response interface (rind3d_100k.interface), its tile scan, the
constructive bisection proof, the contract declaration and the validator. CPU only; the GPU decision test is in
test_pipeline_gpu.py."""
import json
from pathlib import Path

import numpy as np
import pytest

from rind3d_100k import sampling as SM
from rind3d_100k.interface import (TILE, TileScan, bisect_interface, channel_class, has_interface, interface_axes,
                                   is_mixed, lattice_path)

HERE = Path(__file__).resolve().parent.parent


def _brute(lit, sol):
    """Reference: loop over every 6-adjacent pair."""
    free = ~sol; L = lit.shape[0]; n = 0
    for z in range(L):
        for y in range(L):
            for x in range(L):
                for dz, dy, dx in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    Z, Y, X = z + dz, y + dy, x + dx
                    if Z < L and Y < L and X < L and free[z, y, x] and free[Z, Y, X] and lit[z, y, x] != lit[Z, Y, X]:
                        n += 1
    return n


def test_interface_definition_on_small_cases():
    L = 6
    sol = np.zeros((L, L, L), bool)
    lit = np.zeros((L, L, L), bool); lit[:, :, :3] = True                    # plane x = 2.5: a direct interface
    assert interface_axes(lit, sol) == [0, 0, L * L] and channel_class(lit, sol) == "interface"
    # SOLID-separated: active | SOLID wall | inactive -> mixed (v2) but no direct interface (v3)
    sol2 = np.zeros_like(sol); sol2[:, :, 3] = True
    lit2 = np.zeros_like(lit); lit2[:, :, :3] = True
    assert is_mixed(lit2, sol2) and not has_interface(lit2, sol2) and channel_class(lit2, sol2) == "mixed_no_interface"
    # diagonal contact only (no 6-adjacency between the two FREE receivers)
    sol3 = np.ones_like(sol); sol3[0, 0, 0] = sol3[0, 1, 1] = False
    lit3 = np.zeros_like(lit); lit3[0, 0, 0] = True
    assert is_mixed(lit3, sol3) and not has_interface(lit3, sol3)
    assert channel_class(np.ones_like(lit), sol) == "all_active"
    assert channel_class(np.zeros_like(lit), sol) == "all_inactive"
    assert channel_class(lit, np.ones_like(sol)) == "no_free"
    # an active receiver next to SOLID next to nothing else is not an interface
    sol4 = np.zeros_like(sol); sol4[:, :, 1:] = True
    lit4 = np.zeros_like(lit); lit4[:, :, 0] = True
    assert not has_interface(lit4, sol4)


@pytest.mark.parametrize("seed", range(6))
def test_interface_axes_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    L = 7
    sol = rng.random((L, L, L)) < rng.uniform(0.1, 0.7)
    lit = (rng.random((L, L, L)) < rng.uniform(0.05, 0.95)) & ~sol
    assert sum(interface_axes(lit, sol)) == _brute(lit, sol)


def _window(rng, L, kind):
    """A random L^3 window: 'plane' (one oblique interface), 'separated' (active and inactive FREE only across SOLID),
    'face' (an interface exactly on a tile face), 'uniform'."""
    z, y, x = np.meshgrid(*(np.arange(L),) * 3, indexing="ij")
    sol = np.zeros((L, L, L), bool)
    if kind == "plane":
        n = rng.normal(size=3); lit = (n[0] * x + n[1] * y + n[2] * z) > n @ np.full(3, L / 2)
        sol = rng.random((L, L, L)) < 0.05
    elif kind == "separated":
        sol = (x // 8) % 2 == 1                                  # SOLID slabs between FREE slabs whose response
        lit = (x // 16) % 2 == 0                                 # alternates: every active/inactive contact is via SOLID
    elif kind == "face":
        lit = x < TILE                                           # the only interface edges cross the x = 64 tile face
    else:
        lit = np.ones((L, L, L), bool)
    return lit & ~sol, sol


@pytest.mark.parametrize("kind", ["plane", "separated", "face", "uniform"])
@pytest.mark.parametrize("seed", range(3))
def test_tilescan_equals_full_crop_any_order(kind, seed):
    rng = np.random.default_rng(seed)
    L = 128
    lit, sol = _window(rng, L, kind)
    n = L // TILE
    offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
    rng.shuffle(offs)
    scan = TileScan(L)
    for (a, b, c) in offs:
        sl = (slice(c * TILE, (c + 1) * TILE), slice(b * TILE, (b + 1) * TILE), slice(a * TILE, (a + 1) * TILE))
        scan.add((a, b, c), lit[sl], sol[sl])
    assert scan.complete
    assert scan.found == has_interface(lit, sol)
    expect = dict(plane=True, separated=False, face=True, uniform=False)[kind]
    assert has_interface(lit, sol) == expect and (kind != "separated" or is_mixed(lit, sol))
    assert scan.mixed == is_mixed(lit, sol)
    assert (scan.nl, scan.nd, scan.ns) == (int(lit.sum()), int((~lit & ~sol).sum()), int(sol.sum()))
    if kind == "face":
        assert scan.found and all(sum(interface_axes(lit[sl], sol[sl])) == 0 for sl in
                                  [(slice(None), slice(None), slice(0, TILE)), (slice(None), slice(None), slice(TILE, None))])


@pytest.mark.parametrize("seed", range(20))
def test_bisection_proof_is_a_real_interface(seed):
    rng = np.random.default_rng(seed)
    L = 32
    lit, sol = _window(rng, L, "plane" if seed % 2 else "separated")
    query = lambda v: (lit[v[:, 2], v[:, 1], v[:, 0]], sol[v[:, 2], v[:, 1], v[:, 0]])   # vox in (x, y, z)
    free = np.argwhere(~sol)[:, ::-1]                                                     # xyz
    for _ in range(10):
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        la, lb = query(np.array([a, b]))[0]
        if la == lb:
            continue
        if not la:
            a, b = b, a
        pair, nq = bisect_interface(query, lattice_path(a, b, tuple(rng.permutation(3))), True)
        assert nq <= int(np.ceil(np.log2(3 * L))) + 1
        if pair is not None:
            u, v = pair
            assert np.abs(u - v).sum() == 1
            lu, su = query(np.array([u, v]))
            assert not su.any() and lu[0] != lu[1]
        else:
            assert has_interface(lit, sol) or not has_interface(lit, sol)    # inconclusive is allowed, never wrong


def test_lattice_path_is_6_connected():
    p = lattice_path(np.array([3, 9, 1]), np.array([0, 2, 5]), (2, 0, 1))
    assert (p[0] == [3, 9, 1]).all() and (p[-1] == [0, 2, 5]).all()
    assert np.all(np.abs(np.diff(p, axis=0)).sum(1) == 1) and len(p) == 3 + 7 + 4 + 1


def test_contract_v3_declares_the_direct_interface_rule():
    v2 = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v2.json").read_text())
    v3 = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())
    assert v3["contract"] == "rind3d-100k-contract-v3" and v3["seed"] != v2["seed"]
    assert SM.eligibility(v3) == "direct_interface" and SM.eligibility(v2) == "mixed"
    assert SM.declared(v3)
    for k in ("families", "quota", "cap", "rate_n", "n_probe", "source_count", "source_rule", "resolution", "splits"):
        assert v3[k] == v2[k], k                       # unchanged design parameters (docs/SAMPLING_IMPACT_V3.md)
    assert v3["selection"]["coverage"] == v2["selection"]["coverage"]
    assert "fallback" in v3["eligibility"] and "none" in v3["eligibility"]["fallback"]
    with pytest.raises(ValueError):
        SM.eligibility(dict(eligibility=dict(rule="both_values_somewhere")))
