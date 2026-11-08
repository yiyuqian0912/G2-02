"""Closed-contact, strict-witness, and typed cache regressions."""
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.gaussian_geometry.b2_hit.freeze import OUT
from experiments.gaussian_geometry.b2_independent.adversarial import fixtures
from experiments.gaussian_geometry.b2_independent.benchmark import LABELS, config
from experiments.gaussian_geometry.b2_precision.bench import settings
from experiments.gaussian_geometry.b2_precision.build import extension
from experiments.gaussian_geometry.models import Kernel, KernelScene, Threshold
from experiments.gaussian_geometry.queries import FieldOracle, Limits


@pytest.fixture(scope="module")
def module():
    build = Path(os.environ.get("RIND_B2_HIT_BUILD", OUT/"live-native"))
    if not build.exists():
        pytest.skip("Build the isolated fixed-witness experiment first")
    return extension(build)


def engine(module, scene, independent=False):
    return module.Engine([k.center.tolist() for k in scene.kernels],
                         [k.matrix.ravel().tolist() for k in scene.kernels],
                         [k.weight for k in scene.kernels], scene.threshold.value,
                         scene.threshold.kind == "exp", independent)


def options(module, mode=0, independent=False):
    opts, cfg = settings(module, "F" if independent else "A", config())
    cfg.cover_depth = 2
    cfg.study.miss_mode = 4
    cfg.study.hit_mode = mode
    return opts, cfg


ADVERSARIES = fixtures()


@pytest.mark.parametrize("name,scene,source,receiver,expected", ADVERSARIES,
                         ids=[row[0] for row in ADVERSARIES])
def test_old_scalar_adversaries(module, name, scene, source, receiver, expected):
    opts, cfg = options(module)
    opts.max_nodes, opts.query_seconds = 2048, .4
    r = engine(module, scene).ray(source, receiver, opts, cfg)
    assert LABELS[r["status"]] in (expected, "UNKNOWN"), (name, r)
    assert r["nodes"] <= opts.max_nodes


@pytest.mark.parametrize("endpoint", [False, True])
def test_exact_closed_contact_does_not_expand(module, endpoint):
    c = 33/64
    scene = KernelScene("gaussian_sum", [Kernel([c]*3, np.eye(3)/8, 1.)],
                        Threshold("exp", -.125))
    source = [17/64,c,c] if endpoint else [17/64,37/64,c]
    target = [29/64,c,c] if endpoint else [49/64,37/64,c]
    p = engine(module, scene)
    s = p.session(source, 32, *options(module), True)
    r = s.hit_audit(target, [[1,1],[-2,4]])
    assert r["status"] == 1 and r["kind"] == "analytic_contact"
    assert r["t"] == (1. if endpoint else .5)
    assert r["margin"] == 0
    assert all(not item["gradient"] and not item["direct"] for item in r["trials"])
    assert FieldOracle(scene).segment(source,target).status == "HIT"


@pytest.mark.parametrize("endpoint", [False, True])
def test_fusion_only_strict_witness(module, endpoint):
    c = 33/64
    scene = KernelScene("gaussian_sum",
                        [Kernel([c+dx,c,c], np.eye(3)/8, .5) for dx in (-1/32,1/32)],
                        Threshold("binary64", .8))
    source, target = [c,c,.125], [c,c,c if endpoint else 57/64]
    r = engine(module, scene).session(source, 32, *options(module), True).hit_audit(target, [[2,1]])
    assert r["status"] == 1 and r["margin"] > 0
    assert (r["t"] == 1.) if endpoint else (0 < r["t"] < 1.)
    assert r["trials"][0]["direct"] and r["trials"][0]["gradient"]
    arb = FieldOracle(scene)
    for y in (c, c+1/64, c+1/32):
        assert arb.segment(source,[c,y,target[2]]).status == "HIT"


def test_one_ulp_and_narrow_adjacent_rays(module):
    c = 65/128
    scene = KernelScene("gaussian_sum", [Kernel([c]*3, np.eye(3)/8, .4)]*2,
                        Threshold("binary64", np.nextafter(.8,0)))
    source, receiver = [c,c,.125], [c,c,c]
    opts, cfg = options(module)
    opts.exact_depth = 0
    p = engine(module, scene)
    r = p.session(source, 64, opts, cfg, True).hit_audit(receiver, [[1,1]])
    assert r["status"] == 1 and 0 < r["margin"] < 1e-14
    assert not r["trials"][0]["gradient"] and not r["trials"][0]["direct"]
    assert p.ray(source, [c+1/64,c,c], opts, cfg)["status"] == 0

    thin = np.array([[2**-18,2**-20,0.],[0.,1/32,1/512],[0.,0.,1/16]])
    scene = KernelScene("gaussian_sum",
                        [Kernel([c+dx,c,c],thin,.6) for dx in (0.,2**-20)],
                        Threshold("binary64",.9))
    source, receiver = [c,.125,c], [c,c+.125,c]
    r = engine(module,scene).session(source,64,*options(module),True).hit_audit(receiver,[[-1,1]])
    assert r["status"] == 1
    assert not r["trials"][0]["direct"] and not r["trials"][0]["gradient"]
    assert FieldOracle(scene).segment(source,[c-1/64,receiver[1],c]).status == "MISS"


def test_subnormal_witness_and_free_zero_length(module):
    c = 33/64
    tiny = np.nextafter(0.,1.)
    scene = KernelScene("gaussian_sum", [Kernel([c]*3,np.eye(3)/8,64*tiny)]*2,
                        Threshold("binary64",96*tiny))
    source = [c,c,9/64]
    session = engine(module,scene).session(source,32,*options(module),True)
    r = session.hit_audit([c,c,c], [[2,1]])
    assert r["status"] == 1
    if r["trials"][0]["direct"] or r["trials"][0]["gradient"]:
        assert FieldOracle(scene).segment(source,[c,c+1/32,c]).status == "HIT"
    zero = session.hit_audit(source, [[1,1]])
    assert zero["status"] == 0 and zero["margin"] == 0


def test_source_close_strict_witness_keeps_closed_t(module):
    c = 33/64
    scene = KernelScene("gaussian_sum", [Kernel([c]*3,np.eye(3)/8,1.)],
                        Threshold("binary64",.5))
    # The exact .5 level is at z=c-(1/8)*sqrt(2*log(2)).
    # This dyadic source is just below it, with clearance exceeding 2^-26.
    source = [c,c,0.368438720703125]
    p = engine(module,scene)
    r = p.session(source,32,*options(module),True).hit_audit([c,c,c], [[1,1]])
    assert r["status"] == 1 and r["kind"] == "inner_ellipsoid" and r["margin"] > 0
    assert r["trials"][0]["direct"]
    assert 0 < r["t"] <= 1
    assert FieldOracle(scene,Limits(seconds=.5)).segment(source,[c+1/32,c,c]).status == "HIT"


def test_unaligned_dual_cache_matches_both_controls(module):
    scene = KernelScene("gaussian_sum",
                        [Kernel([x,.5,.5],np.eye(3)/8,.5) for x in (.453125,.578125)],
                        Threshold("binary64", .75))
    source, origin, grid, side = [.125,.5,.125], [9,7,11], 32, 16
    p, f = engine(module,scene), engine(module,scene,True)
    old = p.session(source,grid,*options(module),True).tile(origin,side,30)
    reference = f.session(source,grid,*options(module,independent=True),True).tile(origin,side,30)
    count = 0
    for mode in (2,3,4,5):
        s = p.session(source,grid,*options(module,mode),True)
        a, b = (s.tile(origin,side,30) for _ in range(2))
        assert a["complete"] and b["complete"]
        for k in ("response","solid","valid"):
            assert np.array_equal(a[k], reference[k]) and np.array_equal(a[k], b[k])
        assert np.array_equal(a["propagated_seed"], old["propagated_seed"])
        h = a["hit_propagated_seed"]
        assert np.array_equal(h,b["hit_propagated_seed"])
        used = h >= 0
        count += int(used.sum())
        assert np.all(old["evaluation_path"][used] == 7)
        assert np.all(old["evaluation_path"].ravel()[h[used]] == 7)
        xyz = np.argwhere(used)
        xyz_seed = np.array(np.unravel_index(h[used], (side,)*3)).T
        assert np.all(np.count_nonzero(xyz-xyz_seed,axis=1) == 1)
        assert np.all(np.abs(xyz-xyz_seed).sum(axis=1) <= 4)
    assert count > 0
    independent = f.session(source,grid,*options(module,independent=True),True)
    with pytest.raises(ValueError, match="primary session"):
        independent.hit_audit([33/64]*3, [[1,1]])


def test_conflicting_certificates_fail_closed(module):
    for first, second in ((0,1),(1,0)):
        with pytest.raises(RuntimeError, match="Contradictory"):
            module._require_compatible_hit_label(first,second)
    module._require_compatible_hit_label(-1,1)
    module._require_compatible_hit_label(1,1)
