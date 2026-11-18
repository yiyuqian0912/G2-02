"""Contract v2 on the GPU: a fixture scene regenerated bit for bit, validated, and its FastLabel fixture labels
reproduced. Skipped without CUDA. Each CUDA runtime runs in its own spawned process (one rind3d runtime per process)."""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _cuda():
    r = subprocess.run([sys.executable, "-c", "from rind3d.rendering.backends import BackendFactory as B; "
                        "import sys; sys.exit(0 if B()._cuda() is not None else 1)"], capture_output=True)
    return r.returncode == 0


needs_cuda = pytest.mark.skipif(not _cuda(), reason="needs CUDA")


def _spawn(fn, *a):
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex:
        return ex.submit(fn, *a).result()


def _physical(root, slot):
    sys.path.insert(0, str(HERE / "scripts"))
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k.controller import plan_epoch
    from rind3d_100k.physical import run_scene
    from rind3d_100k.storage import scene_path, atomic_write
    from rind3d_100k.validate import validate_scene
    from make_fixtures import v2_scene_fixture
    cfg = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v2.json").read_text())
    pl = plan_epoch(cfg, 0, [])[slot]
    rec, payload = run_scene(cfg, slot, pl["k"], pl["c_table"], BackendFactory(), 0)
    atomic_write(scene_path(root, slot), payload)
    v = validate_scene(root, slot, None, physics=128, deep=True)
    return json.loads(json.dumps(v2_scene_fixture(rec, payload), default=str)), v["ok"], v["errors"]


def _labels(container, tasks):
    sys.path.insert(0, str(HERE / "scripts"))
    from make_fixtures import fastlabel_digest
    from rind3d_posterior.fastlabel import label_window_fast, window_from_100k
    ref = json.loads((HERE / "handoff" / "fixtures" / "fastlabel_v2" / "INDEX.json").read_text())
    return [fastlabel_digest(label_window_fast(window_from_100k(container, w, s), **ref["config"])) for w, s in tasks]


@needs_cuda
def test_v2_fixture_scene_and_fastlabel_labels(tmp_path):
    slot = 0
    ref = json.loads((HERE / "handoff" / "fixtures" / "physical_v2" / f"slot_{slot:06d}.json").read_text())
    got, ok, errs = _spawn(_physical, str(tmp_path), slot)
    assert ok, errs
    for kk in ("windows", "crops", "sides", "scene_id"):
        assert got[kk] == ref[kk], kk
    lab = [r for r in json.loads((HERE / "handoff" / "fixtures" / "fastlabel_v2" / "INDEX.json").read_text())["labels"]
           if r["slot"] == slot]
    container = str(tmp_path / "scenes" / "0000" / f"{slot:06d}.npz")
    digests = _spawn(_labels, container, [(r["window"], r["source"]) for r in lab])
    assert digests == [r["digest"] for r in lab]


def _v3_decisions(slot, n_prop):
    """Contract v3: lazy and exact decisions of evaluate_proposal(rule="direct_interface") on a real scene equal the
    direct-interface rule recomputed from complete certified renderings of every channel."""
    import numpy as np
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.families import generate
    from rind3d_1k.physical import POLICY, sample_sources
    from rind3d_100k.local_eval import LocalScene, evaluate_proposal, contains_any, render_accepted
    from rind3d_100k.interface import interface_axes
    cfg = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())
    fac = BackendFactory()
    bundle, _ = generate(cfg["seed"], slot, cfg["families"][slot % 9], 0)
    ctx = fac.context(bundle, POLICY)
    src, _ = sample_sources(ctx, cfg["seed"], slot, 4, 0)
    sc = LocalScene(fac, bundle, src, POLICY, 1024); sc.enable_hull(ctx)
    rng = np.random.default_rng(slot); prng = np.random.default_rng(slot + 1)
    bad = []; n = 0
    try:
        for i in range(n_prop):
            L = (64, 128)[i % 2]
            o = rng.integers(0, 1024 - L + 1, 3)
            if contains_any(src, o, L, 1024):
                continue
            d = evaluate_proposal(sc, src, o, L, 0.0, [1.0] * 4, prng, n_probe=64, need_exact=True, rule="direct_interface")
            lits, sol, counts, m, mask, extra = render_accepted(sc, o, L, "direct_interface")
            if (~sol).any():
                n += 1
                ref = sum(1 << j for j in range(4) if sum(interface_axes(lits[j], sol)) > 0)
                if (d.mask, d.m, d.mixed_mask) != (ref, bin(ref).count("1"), extra["mixed_mask"]) or mask != ref:
                    bad.append(("exact", o.tolist(), L))
                c = sorted(rng.uniform(0.05, 1, 4).tolist()); U = float(rng.random())
                dl = evaluate_proposal(sc, src, o, L, U, c, prng, n_probe=64, rule="direct_interface")
                if bool(dl.accept) != bool(m >= 1 and U < c[m - 1]):
                    bad.append(("lazy", o.tolist(), L))
    finally:
        sc.close()
    return n, bad


@needs_cuda
@pytest.mark.parametrize("slot", [1, 6])
def test_v3_direct_interface_decisions_are_exact(slot):
    n, bad = _spawn(_v3_decisions, slot, 16)
    assert n >= 8 and not bad, bad
