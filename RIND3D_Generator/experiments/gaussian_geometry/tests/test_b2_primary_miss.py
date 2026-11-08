"""Numerical and cache checks for the isolated primary and adaptive build."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_independent.adversarial import fixtures
from experiments.gaussian_geometry.b2_independent.benchmark import LABELS, config
from experiments.gaussian_geometry.b2_angular.preservation import sha
from experiments.gaussian_geometry.b2_precision.bench import settings
from experiments.gaussian_geometry.b2_precision.build import extension
from experiments.gaussian_geometry.b2_primary_miss.build import OUT
from experiments.gaussian_geometry.b2_primary_miss.gate import admission, trial
from experiments.gaussian_geometry.models import Kernel, KernelScene, Threshold
from experiments.gaussian_geometry.queries import FieldOracle, Limits


@pytest.fixture(scope="module")
def module():
    path = Path(os.environ.get("RIND_B2_PRIMARY_MISS_BUILD", OUT / "propagation-native"))
    if not path.exists():
        pytest.skip("Build the isolated primary MISS experiment first")
    return extension(path)


@pytest.fixture(scope="module")
def contact():
    c = 33/64
    kernels = [Kernel([c + dx, c, c], np.eye(3) * .125, .5)
               for dx in (-.0625, .0625)]
    return KernelScene("gaussian_sum", kernels, Threshold("exp", -.25))


def engine(module, scene, independent=False):
    return module.Engine([k.center.tolist() for k in scene.kernels],
                         [k.matrix.ravel().tolist() for k in scene.kernels],
                         [k.weight for k in scene.kernels], scene.threshold.value,
                         scene.threshold.kind == "exp", independent)


def options(module, mode=0, independent=False):
    opts, cfg = settings(module, "F" if independent else "A", config())
    cfg.cover_depth = 2
    cfg.study.miss_mode = mode
    return opts, cfg


ADVERSARIES = fixtures()


@pytest.mark.parametrize("name,scene,source,receiver,expected", ADVERSARIES,
                         ids=[row[0] for row in ADVERSARIES])
def test_primary_scalar_control_preserves_adversaries(module, name, scene, source, receiver, expected):
    opts, cfg = options(module)
    opts.max_nodes, opts.query_seconds = 2048, .4
    got = engine(module, scene).ray(source, receiver, opts, cfg)
    assert LABELS[got["status"]] in (expected, "UNKNOWN"), (name, got)
    assert got["nodes"] <= opts.max_nodes


@pytest.mark.parametrize("source_z", [9/64, 29/64 - 2**-20])
def test_closed_contact_and_source_clearance(module, contact, source_z):
    c = 33/64
    source, touch, receiver = [c, c, source_z], [c, c, 29/64], [c, c, 27/64]
    p = engine(module, contact)
    opts, cfg = options(module)
    opts.filters = False  # Exercise a retained partition even for an easy seed.
    session = p.session(source, 32, opts, cfg, True)
    toward = session.primary_probe(receiver, 3, 1)
    away = session.primary_probe(receiver, -3, 1)
    assert toward["status"] == away["status"] == 0
    assert toward["certified_step"] == 0 and away["certified_step"] == 1
    assert p.ray(source, touch, *options(module))["status"] in (1, -1)
    oracle = FieldOracle(contact, Limits(max_nodes=4096, seconds=.5))
    assert oracle.segment(source, [c, c, 31/64]).status == "HIT"
    for d in (0, .5, 1):
        assert oracle.segment(source, [c, c, (27-2*d)/64]).status == "MISS"


def test_primary_uses_its_own_swept_bounds_on_thin_bridge(module):
    c, grid = 65/128, 64
    matrix = np.array([[2**-18, 2**-20, 0.], [0., 1/32, 1/512], [0., 0., 1/16]])
    scene = KernelScene("gaussian_sum",
                        [Kernel([c+dx, c, c], matrix, .6) for dx in (0., 2**-20)],
                        Threshold("binary64", .9))
    source, blocked = [c, .125, c], [c, c+.125, c]
    receiver = [c-1/grid, blocked[1], c]
    p = engine(module, scene)
    proof = p.session(source, grid, *options(module), True).primary_probe(receiver, 1, 1)
    assert proof["status"] == 0 and proof["certified_step"] == 0
    assert p.ray(source, blocked, *options(module))["status"] == 1
    assert FieldOracle(scene).segment(source, blocked).status == "HIT"


def test_subnormal_margin_and_budget_never_enable_cache(module, contact):
    tiny = np.nextafter(0., 1.)
    scene = KernelScene("gaussian_sum",
                        [Kernel([x]*3, np.eye(3)/64, tiny) for x in (.75, .875)],
                        Threshold("binary64", tiny))
    opts, cfg = options(module)
    opts.exact_depth = 0
    opts.filters = False
    # Force exact-sum closure of a strictly subnormal margin.
    near = engine(module, scene)
    r = near.ray([.125]*3, [25/64]*3, opts, cfg)
    assert r["status"] == 0
    assert r["work"]["early_exact_interval_free"] == 1

    opts, cfg = options(module)
    opts.max_nodes = 0
    opts.filters = False
    p = engine(module, contact)
    session = p.session([33/64, 33/64, .125], 32, opts, cfg, True)
    d = session.primary_probe([33/64, 33/64, 27/64], -3, 1)
    assert d["status"] == -1 and d["certified_step"] == 0
    with pytest.raises(ValueError, match="voxel center"):
        session.primary_probe([.5, .5, .5], 1, 1)


def test_exact_leaf_extraction_and_single_shape_exit(module, contact):
    c = 33/64
    source = [c, c, .125]
    strict = KernelScene("gaussian_sum", contact.kernels, Threshold("exp", -.25 + 2**-52))
    p = engine(module, strict)
    opts, cfg = options(module)
    opts.exact_depth = 0
    d = p.session(source, 32, opts, cfg, True).primary_probe([c, c, 29/64], -3, 1)
    assert d["status"] == 0
    assert d["work"].get("early_exact_interval_free", 0) > 0
    assert d["work"]["primary_capture_leaves"] > 0

    k = contact.kernels[0]
    single = KernelScene("gaussian_sum", [Kernel(k.center, k.matrix, 1.)], contact.threshold)
    p = engine(module, single)
    d = p.session(source, 32, *options(module), True).primary_probe([c, c, 27/64], -3, 1)
    assert d["status"] == 0 and d["certified_step"] == 0


def test_primary_and_independent_caches_are_deterministic_on_unaligned_tile(module, contact):
    scene = KernelScene("gaussian_sum", contact.kernels, Threshold("binary64", .75))
    source, origin, grid, side = [.125, .5, .125], [9, 7, 11], 32, 16
    p, f = engine(module, scene), engine(module, scene, True)
    old = p.session(source, grid, *options(module), True).tile(origin, side, 30)
    ref = f.session(source, grid, *options(module, independent=True), True).tile(origin, side, 30)
    hits = 0
    for backend, mode, independent, baseline in ((p, 2, False, old), (p, 3, False, old),
                                                   (p, 4, False, old), (f, 4, True, ref)):
        session = backend.session(source, grid, *options(module, mode, independent), True)
        a = session.tile(origin, side, 30)
        b = session.tile(origin, side, 30)
        assert a["complete"] and b["complete"]
        for key in ("response", "solid", "valid"):
            assert np.array_equal(a[key], old[key]) and np.array_equal(a[key], b[key])
        mark = a["propagated_seed"]
        assert np.array_equal(mark, b["propagated_seed"])
        used = mark >= 0
        hits += int(used.sum())
        assert np.all(baseline["evaluation_path"][used] == 6)
        assert np.all(baseline["evaluation_path"].ravel()[mark[used]] == 6)
        assert np.all(a["response"][used] & ~a["solid"][used])
        if independent:
            with pytest.raises(ValueError, match="Invalid primary axial"):
                session.primary_probe([33/64]*3, 1, 1)
    assert hits > 0


def test_admission_requires_all_original_pairs_and_memory_cap(tmp_path):
    rows = [{"id": str(i), "origin": [i, 0, 0], "complete": True,
             "tile_seconds": i % 2 + 1} for i in range(16)]
    result = {"variant": "P_J", "rows": rows, "setup_seconds": 2.,
              "peak_RSS_MiB": 128., "wall_seconds": 30., "CPU_seconds": 29.}
    checked = admission(result, 360.)
    assert checked["accepted"]
    assert checked["guarded_planning_seconds"] == pytest.approx(4. + 1.2*8*8*2.)
    assert [r["id"] for r in checked["octant_slowest"]] == [str(i) for i in range(1, 16, 2)]
    assert not admission({**result, "rows": rows[:-1]}, 360.)["accepted"]
    assert not admission({**result, "peak_RSS_MiB": 513.}, 360.)["accepted"]
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    (tmp_path / "identity.json").write_text(json.dumps(
        {"primary": "P_J", "plan_SHA256": sha(plan)}))
    (tmp_path / "admission.json").write_text('{"admitted": false}')
    args = SimpleNamespace(worker="primary", phase="remaining", primary="P_J",
                           output=tmp_path, plan=plan)
    with pytest.raises(ValueError, match="Both probe estimates"):
        trial(args, [], [])
