"""Closed-segment and cache correctness for experimental MISS extensions."""
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_independent.adversarial import fixtures
from experiments.gaussian_geometry.b2_independent.benchmark import LABELS, config, native_engine
from experiments.gaussian_geometry.b2_miss.bench import OUT
from experiments.gaussian_geometry.b2_precision.bench import settings
from experiments.gaussian_geometry.b2_precision.build import extension
from experiments.gaussian_geometry.models import Kernel, KernelScene, Threshold
from experiments.gaussian_geometry.queries import FieldOracle, Limits


@pytest.fixture(scope="module")
def module():
    path = Path(os.environ.get("RIND_B2_MISS_BUILD", OUT / "propagation-native"))
    if not path.exists():
        pytest.skip("Build the isolated B2 MISS experiment first")
    return extension(path)


@pytest.fixture(scope="module")
def geometry():
    # At x=y=33/64 and z=33/64-1/16, both Gaussians equal exp(-1/4).
    c = 33/64
    kernels = [Kernel([c + dx, c, c], np.eye(3) * .125, .5) for dx in (-.0625, .0625)]
    return KernelScene("gaussian_sum", kernels, Threshold("exp", -.25))


def options(module, method="F", mode=0):
    opts, cfg = settings(module, method, config())
    cfg.cover_depth = 2
    cfg.study.miss_mode = mode
    cfg.trace = True
    return opts, cfg


@pytest.mark.parametrize("name,scene,source,receiver,expected", fixtures(),
                         ids=[row[0] for row in fixtures()])
def test_f_and_controls_keep_adversarial_labels(module, name, scene, source, receiver, expected):
    engine = native_engine(module, scene)
    for method in ("A", "D", "F"):
        opts, cfg = options(module, method)
        opts.max_nodes, opts.query_seconds = 2048, .4
        r = engine.ray(source, receiver, opts, cfg)
        assert LABELS[r["status"]] in (expected, "UNKNOWN"), (name, method, r)
        assert r["nodes"] <= opts.max_nodes
    arb = FieldOracle(scene, Limits(max_nodes=8192, max_depth=40, seconds=.5))
    status = arb.segment(source, receiver).status
    assert status in (expected, "UNKNOWN", "BUDGET_EXHAUSTED"), (name, "Arb", status)


def test_axis_proof_checks_closed_neighbor_endpoint(module, geometry):
    c = 33/64
    source, touch = [c, c, c-.375], [c, c, c-.0625]
    seed = [c, c, touch[2]-1/32]
    engine = native_engine(module, geometry)
    session = engine.session(source, 32, *options(module), True)
    proof = session.miss_probe(seed, [3, -3], 1)
    assert proof["status"] == 0 and proof["bounds"]["complete_proof"]
    assert proof["bounds"]["local_3_FREE"] == 0
    assert proof["bounds"]["local_-3_FREE"] == 1
    assert session.miss_probe(seed, [3], 2)["bounds"]["local_3_FREE"] == 0

    arb = FieldOracle(geometry, Limits(max_nodes=2048, seconds=.2))
    assert arb.segment(source, touch).status in ("HIT", "UNKNOWN", "BUDGET_EXHAUSTED")
    assert engine.ray(source, touch, *options(module, "A"))["status"] in (1, -1)
    assert arb.segment(source, [c, c, touch[2]+1/32]).status == "HIT"
    for offset in (0., .5, 1.):
        dest = [c, c, seed[2] - offset/32]
        assert arb.segment(source, dest).status == "MISS"
        assert engine.ray(source, dest, *options(module, "A"))["status"] == 0


def test_source_close_to_boundary_keeps_closed_proof(module, geometry):
    c = 33/64
    source = [c, c, c - 1/16 - 2**-20]
    seed = [c, c, c - 3/32]
    session = native_engine(module, geometry).session(source, 32, *options(module), True)
    bounds = session.miss_probe(seed, [-3, 3], 1)["bounds"]
    assert bounds["complete_proof"] == 1
    assert bounds["local_-3_FREE"] == 1 and bounds["local_3_FREE"] == 0


def test_narrow_rotated_bridge_between_adjacent_rays(module):
    c, grid = 65/128, 64
    matrix = np.array([[2**-18, 2**-20, 0.], [0., 1/32, 1/512], [0., 0., 1/16]])
    scene = KernelScene("gaussian_sum",
                        [Kernel([c+offset, c, c], matrix, .6) for offset in (0., 2**-20)],
                        Threshold("binary64", .9))
    source, blocked = [c, .125, c], [c, c+.125, c]
    seed = [c-1/grid, blocked[1], c]
    engine = native_engine(module, scene)
    session = engine.session(source, grid, *options(module), True)
    proof = session.miss_probe(seed, [1], 1)
    assert proof["status"] == 0 and proof["bounds"]["local_1_FREE"] == 0
    assert engine.ray(source, blocked, *options(module, "A"))["status"] == 1
    assert FieldOracle(scene).segment(source, blocked).status == "HIT"


def test_subnormal_margin_rounds_down_without_extending(module):
    tiny = np.nextafter(0., 1.)
    scene = KernelScene("gaussian_sum",
                        [Kernel([x]*3, np.eye(3) / 64, tiny) for x in (.75, .875)],
                        Threshold("binary64", tiny))
    source, seed = [.125]*3, [25/64]*3
    session = native_engine(module, scene).session(source, 32, *options(module), True)
    proof = session.miss_probe(seed, [-1, 1], 1)
    assert proof["status"] == 0 and proof["bounds"]["complete_proof"] == 1
    assert proof["bounds"]["margin_lower"] == 0
    assert proof["bounds"]["certified_receivers_in_radius4_stencil"] == 1
    assert proof["bounds"]["local_-1_FREE"] == proof["bounds"]["local_1_FREE"] == 0


def test_no_certificate_from_aborted_or_analytic_ray(module, geometry):
    opts, cfg = options(module)
    opts.max_nodes = 0
    c = 33/64
    source = [c, c, .125]
    seed = [c, c, .375 + 1/64]
    session = native_engine(module, geometry).session(source, 32, opts, cfg, True)
    proof = session.miss_probe(seed, [3], 1)
    assert proof["status"] == -1 and proof["bounds"] == {"complete_proof": 0}
    with pytest.raises(ValueError, match="voxel center"):
        session.miss_probe([c, c, .5], [3], 1)
    with pytest.raises(ValueError, match="Invalid bounded"):
        session.miss_probe(seed, [0], 1)

    kernel = geometry.kernels[0]
    analytic = KernelScene("gaussian_sum", [Kernel(kernel.center, kernel.matrix, 1.)],
                           Threshold("exp", -.25))
    session = native_engine(module, analytic).session(source, 32, *options(module), True)
    proof = session.miss_probe(seed, [-3], 1)
    assert proof["status"] == 0 and proof["bounds"] == {"complete_proof": 0}


def test_unaligned_cache_preserves_response_solid_and_exact_indices(module, geometry):
    scene = KernelScene("gaussian_sum", geometry.kernels, Threshold("binary64", .75))
    source, origin, grid, side = [.125, .5, .125], [9, 7, 11], 32, 16
    engine = native_engine(module, scene)
    old = engine.session(source, grid, *options(module, "A"), True).tile(origin, side, 30)
    original = engine.session(source, grid, *options(module), True).tile(origin, side, 30)
    assert old["complete"] and original["complete"]
    assert np.array_equal(old["response"], original["response"])
    hits = 0
    for mode in (1, 2, 3):
        session = engine.session(source, grid, *options(module, mode=mode), True)
        got = session.tile(origin, side, 30)
        again = session.tile(origin, side, 30)
        assert got["complete"] and again["complete"]
        for key in ("solid", "valid", "response"):
            assert np.array_equal(got[key], old[key])
            assert np.array_equal(got[key], again[key])
        marks = got["propagated_seed"]
        assert np.array_equal(marks, again["propagated_seed"])
        used = marks >= 0
        hits += int(used.sum())
        assert np.all(original["evaluation_path"][used] == 6)
        assert np.all(got["response"][used] & ~got["solid"][used])
        assert np.all(marks[used] < side**3)
        assert np.all(original["evaluation_path"].ravel()[marks[used]] == 6)
    assert hits > 0
