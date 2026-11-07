"""Precision-local coverage and source preparation on the isolated B2 build."""
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_independent.adversarial import fixtures
from experiments.gaussian_geometry.b2_independent.benchmark import LABELS, config, native_engine
from experiments.gaussian_geometry.b2_precision.bench import settings
from experiments.gaussian_geometry.b2_precision.build import OUT, extension
from experiments.gaussian_geometry.models import Kernel, KernelScene, Threshold


@pytest.fixture(scope="module")
def module():
    path = Path(os.environ.get("RIND_B2_PRECISION_BUILD", OUT / "native-final"))
    if not path.exists():
        pytest.skip("Build the isolated B2 precision experiment first")
    return extension(path)


ADVERSARIES = fixtures()


@pytest.mark.parametrize("name,scene,source,receiver,expected", ADVERSARIES,
                         ids=[row[0] for row in ADVERSARIES])
def test_adversarial_physical_labels_and_budget(module, name, scene, source, receiver, expected):
    opts, cfg = settings(module, "E", config())
    opts.max_nodes, opts.query_seconds = 2048, .4
    result = native_engine(module, scene).ray(source, receiver, opts, cfg)
    label = LABELS[result["status"]]
    assert label in (expected, "UNKNOWN")
    if name in ("near_contact_HIT", "near_contact_MISS", "one_ULP_HIT", "one_ULP_MISS",
                "analytic_closed_source_contact", "positive_underflow_remainder"):
        assert label == expected
    assert result["nodes"] <= opts.max_nodes


@pytest.fixture(scope="module")
def mixture():
    return KernelScene("gaussian_sum",
                       [Kernel([x, .5, .5], np.eye(3) * .125, .5)
                        for x in (.4375, .5625)],
                       Threshold("exp", -.125 + 2**-20))


def test_low_proofs_survive_high_arithmetic_and_cover_closed_ray(module, mixture):
    engine = native_engine(module, mixture)
    opts, cfg = settings(module, "E", config())
    r = engine.ray([.125, .5, .125], [.875, .5, .875], opts, cfg)
    assert r["status"] == 0
    work = r["work"]
    assert work["precision_prefix_FREE_retained"] > 0
    assert work["precision_high_visits"] > 0
    assert work["precision_low_FREE"] + work["precision_high_FREE"] == work["precision_subdivisions"] + 1
    assert work.get("precision_high_line_kernels", 0) == 0
    assert work["precision_gaussian_term_reuses"] > 0


def test_unresolved_never_becomes_free_under_node_or_depth_limit(module, mixture):
    engine = native_engine(module, mixture)
    opts, cfg = settings(module, "E", config())
    opts.max_nodes = 0
    r = engine.ray([.125, .5, .125], [.875, .5, .875], opts, cfg)
    assert r["status"] == -1 and r["nodes"] == 0
    assert engine.ray([np.nan, .5, .5], [.875, .5, .875], opts, cfg)["status"] == -2
    opts.max_nodes, opts.max_depth = 1, 0
    r = engine.ray([.125, .5, .125], [.875, .5, .875], opts, cfg)
    assert r["status"] == -1 and r["nodes"] == 1
    assert r["work"]["precision_high_terminal_tests"] == 1
    assert r["work"]["precision_high_line_kernels"] == 2


def test_scoped_source_preparation_across_two_sources_and_unaligned_tile(module, mixture):
    engine = native_engine(module, mixture)
    totals = 0
    for source in ([.125, .5, .125], [.25, .75, .125]):
        fields = []
        for method in ("A", "E", "F"):
            opts, cfg = settings(module, method, config())
            cfg.cover_depth = 3
            fields.append(engine.tile(source, 64, [27, 31, 31], 8, 15, opts, cfg))
        assert all(f["complete"] and f["valid"].all() for f in fields)
        for key in ("response", "solid", "evaluation_path"):
            assert all(np.array_equal(f[key], fields[0][key]) for f in fields[1:])
        assert fields[-1]["work"]["precision_source_offset_preparations"] == 1
        totals += fields[-1]["work"].get("precision_source_offset_kernel_reuses", 0)
    assert totals > 0


@pytest.fixture(scope="module")
def tile_module():
    path = Path(os.environ.get("RIND_B2_TILE_BUILD", OUT / "tile-native"))
    if not path.exists():
        pytest.skip("Build the isolated tile session first")
    return extension(path)


@pytest.mark.parametrize("independent", [False, True])
def test_source_bound_tile_session_matches_unaligned_monolithic_crop(tile_module, mixture, independent):
    scene = KernelScene("gaussian_sum", mixture.kernels, Threshold("binary64", .75))
    source = [.125, .5, .125]
    opts, cfg = settings(tile_module, "F" if independent else "A", config())
    cfg.cover_depth = 3
    engine = tile_module.Engine(
        [k.center.tolist() for k in scene.kernels],
        [k.matrix.ravel().tolist() for k in scene.kernels],
        [k.weight for k in scene.kernels], scene.threshold.value, False, independent,
    )
    session = engine.session(source, 32, opts, cfg, True)
    full = engine.field(source, 32, 20, opts, cfg)
    assert full["complete"] and full["solid"].any()
    assert session.preparation_work()["cover_box_queries"] > 0
    for x, y, z in ((3, 7, 13), (11, 7, 13)):
        tile = session.tile([x, y, z], 8, 20)
        assert tile["complete"] and tile["work"]["tile_reused_cover"] == 1
        assert tile["work"].get("cover_box_queries", 0) == 0
        for key in ("response", "solid", "valid"):
            assert np.array_equal(tile[key], full[key][z:z+8, y:y+8, x:x+8])
    with pytest.raises(ValueError):
        session.tile([25, 7, 13], 8, 20)
    with pytest.raises(ValueError):
        engine.session([.5, .5, .5], 32, opts, cfg, True)
