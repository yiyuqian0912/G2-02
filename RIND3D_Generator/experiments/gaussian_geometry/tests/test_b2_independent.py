"""Independent scalar-search regression tests for the B2-only study build."""
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_independent.adversarial import fixtures
from experiments.gaussian_geometry.b2_independent.benchmark import (
    LABELS, config, native_engine, native_settings,
)
from experiments.gaussian_geometry.b2_independent.build import DEFAULT, extension
from experiments.gaussian_geometry.b2_native.api import NativeOracle
from experiments.gaussian_geometry.models import Kernel, KernelScene, Threshold
from experiments.gaussian_geometry.queries import FieldOracle, Limits


@pytest.fixture(scope="module")
def module():
    directory = Path(os.environ.get("RIND_B2_INDEPENDENT_BUILD", DEFAULT / "native"))
    try:
        return extension(directory)
    except ImportError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("method", ["B", "C", "D", "D128", "D_Newton"])
@pytest.mark.parametrize("name,scene,source,receiver,expected", fixtures(), ids=[r[0] for r in fixtures()])
def test_no_opposite_certified_adversarial_label(module, method, name, scene, source, receiver, expected):
    engine = native_engine(module, scene)
    options, settings = native_settings(module, method, config())
    options.max_nodes = 2048
    options.query_seconds = .4
    result = engine.ray(source, receiver, options, settings)
    actual = LABELS[result["status"]]
    assert actual in (expected, "UNKNOWN")
    if name in ("exact_transcendental_contact", "quartic_degenerate_contact"):
        assert actual != "MISS"
    if name in ("near_contact_MISS", "near_contact_HIT", "analytic_closed_source_contact",
                "zero_length_FREE", "ill_conditioned_mixed_scales_MISS",
                "ill_conditioned_mixed_scales_HIT", "positive_underflow_remainder"):
        assert actual == expected
    assert result["nodes"] <= options.max_nodes


@pytest.fixture(scope="module")
def mixture():
    return KernelScene("gaussian_sum",
                       [Kernel([x, .5, .5], np.eye(3) * .12, .6) for x in (.42, .58)],
                       Threshold("binary64", .88))


@pytest.mark.parametrize("method", ["B", "C", "D", "D_Newton"])
def test_independent_closed_field_matches_original_and_arb(module, method, mixture):
    source = [.05, .5, .5]
    independent = NativeOracle(mixture, independent=True)
    assert independent.source_clearance(source).status == "FREE"
    engine = native_engine(module, mixture)
    opts, settings = native_settings(module, method, config())
    settings.cover_depth = 3
    baseline = independent.field(source, 8, 15)
    result = engine.field(source, 8, 15, opts, settings)
    assert baseline["complete"] and result["complete"] and result["valid"].all()
    for key in ("response", "solid"):
        assert np.array_equal(result[key], baseline[key])
    arb = FieldOracle(mixture)
    for i, j, k in ((1, 3, 5), (7, 4, 4), (0, 0, 7)):
        p = [(k + .5) / 8, (j + .5) / 8, (i + .5) / 8]
        mask = arb.membership(p).status
        assert mask in ("SOLID", "FREE")
        assert result["solid"][i, j, k] == (mask == "SOLID")
        if mask == "FREE":
            hit = arb.segment(source, p).status
            assert hit in ("MISS", "HIT")
            assert result["response"][i, j, k] == (hit == "MISS")


@pytest.mark.parametrize("method", ["B", "C", "D"])
def test_exhausted_and_invalid_are_distinct(module, method, mixture):
    engine = native_engine(module, mixture)
    opts, settings = native_settings(module, method, config())
    opts.max_nodes = 0
    result = engine.ray([.05, .5, .5], [.95, .5, .5], opts, settings)
    assert result["status"] == -1 and result["reason"] == "reference_node_budget"
    assert engine.ray([np.nan, .5, .5], [.95, .5, .5], opts, settings)["status"] == -2


def test_low_precision_reaches_original_fallback_without_extra_nodes(module, mixture):
    engine = native_engine(module, mixture)
    opts, settings = native_settings(module, "B", config())
    settings.study.filter_depth = 0
    opts.max_nodes = 2
    result = engine.ray([.05, .56, .5], [.95, .56, .5], opts, settings)
    assert result["nodes"] <= 2
    assert result["work"]["study_low_nodes"] == 1
    assert result["work"]["study_precision_escalations"] == 1


def test_experiment_rejects_unsupported_precision(module, mixture):
    opts, settings = native_settings(module, "B", config())
    settings.study.filter_bits = 40
    with pytest.raises(ValueError):
        native_engine(module, mixture).ray([.05, .5, .5], [.95, .5, .5], opts, settings)


def test_signed_concavity_proves_strict_miss_near_mixture_maximum(module):
    # Along the diagonal the exact symmetric mixture is
    # exp(-1/8) * exp(-36*(t-.5)^2) * cosh(3*(t-.5)).
    # cosh(z) <= exp(z²/2) puts its maximum at t=.5.
    scene = KernelScene("gaussian_sum", [
        Kernel([x, .5, .5], np.eye(3) * .125, .5) for x in (.4375, .5625)
    ], Threshold("exp", -.125 + 2**-20))
    source, receiver = [.125, .5, .125], [.875, .5, .875]
    engine = native_engine(module, scene)
    opts, settings = native_settings(module, "C", config())
    result = engine.ray(source, receiver, opts, settings)
    assert result["status"] == 0
    assert result["work"].get("study_concave_free", 0) > 0
    assert NativeOracle(scene, independent=True).segment(source, receiver).status == "MISS"
    assert FieldOracle(scene, Limits(max_nodes=65536, max_depth=48, seconds=5)).segment(source, receiver).status == "MISS"


def test_optional_stationary_stage_respects_closed_endpoint(module):
    row = next(r for r in fixtures() if r[0] == "subnormal_weight_exact_closed_end")
    _, scene, source, receiver, _ = row
    engine = native_engine(module, scene)
    opts, settings = native_settings(module, "D_Newton", config())
    result = engine.ray(source, receiver, opts, settings)
    assert result["status"] == 1
    assert result["work"].get("study_stationary_tests", 0) > 0
