"""FastLabel v3.4 production package (rind3d_1k.fastlabel).

* design2: the two-stage certificate's false-certificate probability is <= its budget for every p < 1 - eps.
* gpuverify vs the original certified renderer: identical FREE-voxel fields (UNKNOWN voxels decided exactly).
* label_task vs certified release labels (scene 000000), determinism of reruns, and the task-file round trip through
  run_task and the canonical loader (in a temporary copy of the scene; the release is never written).
CUDA tests are skipped when the certified CUDA backend is unavailable."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "RIND3D_Posterior" / "src"))
RELEASE = Path(os.environ.get("RIND3D_1K_RELEASE", "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1"))
SCENE = RELEASE / "scenes" / "000000"


def _cuda():
    try:
        from rind3d_1k.fastlabel.pipeline import process_factory
        process_factory()
        return True
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not (SCENE.exists() and _cuda()), reason="needs the 1K release and CUDA")


def test_design2_alpha_monotone():
    from scipy.stats import binom
    from rind3d_1k.fastlabel import design2
    eps = 0.05
    for d in (0.01 / 2, 1e-4, 0.01 / 159802 / 2, 0.01 / 159802 / 2 ** 8):
        a1, a2, alpha = design2(512, 1024, eps, d)
        assert alpha <= d and a1 <= a2

        def p_cert(p):
            pm = binom.pmf(np.arange(513), 512, 1 - p)
            v = binom.cdf(a1, 512, 1 - p) if a1 >= 0 else 0.0
            return v + sum(pm[k] * binom.cdf(a2 - k, 512, 1 - p) for k in range(a1 + 1, a2 + 1))
        vals = [p_cert(p) for p in np.linspace(0.5, 1 - eps, 40)]
        assert max(vals) <= d * (1 + 1e-9) and np.all(np.diff(vals) >= -1e-15)


def _tasks(L_wanted=(64, 128, 256)):
    W = np.load(SCENE / "public" / "windows.npy"); out = []
    for L in L_wanted:
        for r in W:
            if int(r["L"]) != L or not int(r["mask"]):
                continue
            j = next(j for j in range(int(r["k"])) if (int(r["mask"]) >> j) & 1)
            if (SCENE / "labels" / "tasks" / f"w{int(r['window']):03d}_s{j}.npz").exists():
                out.append((int(r["window"]), j, L)); break
    return out


@needs_cuda
def test_verifier_matches_original_renderer():
    from rind3d_1k.fastlabel import engine as T, gpuverify as GV
    from rind3d_1k.fastlabel.pipeline import scene_cache, process_factory
    from rind3d_1k.fastlabel.sampler import BSampler
    from rind3d_1k.fastlabel.window import fast_cls3, LazyWindow2
    from rind3d_1k.physical import POLICY, legal_source
    scn = scene_cache(SCENE); S = scn["S"]; fac = process_factory()
    src0 = json.loads((SCENE / "privileged" / "sources.json").read_text())["sources_hex"]
    checked = 0
    for w, s, L in _tasks((64, 128)):
        cls = fast_cls3(S, w, s); win = LazyWindow2(cls, np.asarray(S.origin(w), np.int64), 1024)
        roots, _ = T.stage1(scn["geom"], win); xc, xk = T.point_items(win, 256, 128, 0)
        res = T.stage2(scn["geom"], win, xc, xk, roots)
        rng = np.random.default_rng(1)
        true = np.array([float.fromhex(v) for v in src0[s]])
        cands = list(BSampler(res, rng, {}).draw(12)) + [true + rng.normal(0, sc, 3) for sc in (1e-8, 1e-6, 1e-4)] + list(rng.random((4, 3)))
        V = GV.WindowVerifier(fac, scn["ctx"], S, w, s, cls)
        o = np.asarray(S.origin(w)); n = max(L // 64, 1)
        offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        free = ~S.crop(w, -1)
        for p in cands:
            if legal_source(scn["ctx"], p) is None or np.all((p >= o / 1024) & (p <= (o + L) / 1024)):
                continue
            src = fac.source(scn["bundle"], p, "cuda_certified", POLICY)        # the original certified renderer
            ref = np.zeros((L, L, L), bool)
            out = src.packed_batch(np.array([o + np.array(t) * 64 for t in offs], np.int32), 64, 1024)
            src.close()
            for (a, b, c), (pl, _) in zip(offs, out):
                ref[c * 64:(c + 1) * 64, b * 64:(b + 1) * 64, a * 64:(a + 1) * 64] = \
                    np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
            V.test_batch(p[None], dump=True)
            d = V.dump.reshape(L, L, L).astype(np.int16)
            unk = np.argwhere((d == 2) & free)
            if len(unk):
                cz, cy, cx = unk.T
                yy, _, _ = scn["ctx"].bind(p).native.points(np.ascontiguousarray((np.stack([cx, cy, cz], 1) + o + 0.5) / 1024), False)
                d[cz, cy, cx] = 1 - yy
            assert np.array_equal(d[free] == 0, ref[free])
            ok_batch = V.test_batch(p[None])[0][0]
            assert ok_batch == bool(np.all(ref[free] == S.crop(w, s)[free]))
            checked += 1
        V.close()
    assert checked >= 10


def _q(cell, lo):
    u, inv = np.unique(cell, return_inverse=True)
    m = np.bincount(inv, weights=lo); return dict(zip(u.tolist(), (m / m.sum()).tolist()))


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


@needs_cuda
def test_label_task_vs_release_and_determinism():
    from rind3d_1k.fastlabel import label_task
    for w, s, L in _tasks():
        lab = label_task(SCENE, w, s)
        assert lab["status"] in ("certified", "certified_stat")
        with np.load(SCENE / "labels" / "tasks" / f"w{w:03d}_s{s}.npz") as z:
            ref = _q(z["cell"], z["lo"]); ref_tv = float(z["tv_bound"])
        assert _tv(_q(lab["cell"], lab["lo"]), ref) <= 0.05 + ref_tv
        again = label_task(SCENE, w, s)
        assert np.array_equal(again["cell"], lab["cell"]) and np.array_equal(again["lo"], lab["lo"])


@needs_cuda
def test_run_task_roundtrip(tmp_path):
    from rind3d_1k.labels import run_task, task_path
    from rind3d_1k.loader import Scene
    d = tmp_path / "000000"; d.mkdir()
    for name in ("scene.json", "public", "privileged"):
        os.symlink(SCENE / name, d / name)
    w, s, L = _tasks((128,))[0]
    meta = run_task((str(d), w, s, L, True), "fastlabel")
    assert meta["engine"] == "fastlabel" and meta["status"] in ("certified", "certified_stat")
    p = task_path(d, w, s); assert p.exists() and not (SCENE / "labels" / "tasks" / p.name).samefile(p)
    lab = Scene(d).posterior(w, s)
    assert lab["status"] == meta["status"] and abs(lab["q"].sum() - 1) < 1e-12 and lab["certificate"]["kind"] in ("statistical", "support", "deterministic")
