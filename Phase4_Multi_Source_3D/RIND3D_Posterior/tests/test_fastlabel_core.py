"""FastLabel core (rind3d_posterior.fastlabel): scene-format independence, adapters, Gate A, release reproduction.

* design2: the two-stage certificate's false-certificate probability is <= its budget for every p < 1 - eps.
* cls_from_packed equals the reference class construction (SceneObs.cls semantics) on random crops.
* window_from_1k decodes the stored crops exactly; window_from_100k on a container in the RIND3D-100K layout
  (rind3d_100k.physical.run_scene keys) built from the same scene gives the identical task record.
* Gate A smoke (CUDA): verifier fields == certified CUDA renderer, batch decisions agree, no false violations.
* label_window_fast reproduces the stored FastLabel release labels bit for bit (arrays, status, certificate) and stays
  within the certified TV of the stored exact tau-pencil labels.
Data tests use the 1K release (RIND3D_1K_RELEASE; train-split scenes 000000, with exact tau-pencil labels, and 000007,
with backfill FastLabel labels) and skip without it or without CUDA; nothing is written to the release.

CUDA tests run in a spawned subprocess: the rind3d CUDA runtime creates its own (non-primary) context and leaves it
current on the calling thread, which invalidates CuPy handles used later in the same process (test_regression GPU pass).
Production keeps FastLabel and CuPy tau-pencil workers in separate processes for the same reason."""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

RELEASE = Path(os.environ.get("RIND3D_1K_RELEASE", os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")
                              + "/release_1k_v1"))
SCENE = RELEASE / "scenes" / "000000"
SCENE_FL = RELEASE / "scenes" / "000007"
has_release = pytest.mark.skipif(not SCENE.exists(), reason="needs the 1K release")


def _cuda():
    r = subprocess.run([sys.executable, "-c", "from rind3d_posterior.fastlabel.pipeline import process_factory; process_factory()"],
                       capture_output=True)
    return r.returncode == 0


needs_cuda = pytest.mark.skipif(not (SCENE.exists() and SCENE_FL.exists() and _cuda()), reason="needs the 1K release and CUDA")


def _spawn(fn, *args):
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def test_design2_alpha_monotone():
    from scipy.stats import binom
    from rind3d_posterior.fastlabel import design2
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


def _ref_cls(lit, sol):
    """Reference: rind3d_1k.labels.SceneObs.cls from bool [z, y, x] crops."""
    c = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(c, (2, 1, 0)))


@pytest.mark.parametrize("L", [64, 128])
def test_cls_from_packed(L):
    from rind3d_posterior.fastlabel.window import cls_from_packed
    rng = np.random.default_rng(L)
    lit = rng.random((L, L, L)) < 0.4; sol = rng.random((L, L, L)) < 0.2
    got = cls_from_packed(np.packbits(lit.ravel()), np.packbits(sol.ravel()), L)
    assert got.dtype == np.uint8 and np.array_equal(got, _ref_cls(lit, sol))


def _windows_1k(scene=SCENE):
    return np.load(Path(scene) / "public" / "windows.npy")


@has_release
def test_window_from_1k_decodes_store():
    import zstandard
    from rind3d_posterior.fastlabel import window_from_1k
    W = _windows_1k(); idx = np.load(SCENE / "public" / "observations_index.npy")
    blob = (SCENE / "public" / "observations.bin").read_bytes(); dz = zstandard.ZstdDecompressor()
    pos = {(int(a), int(b)): (int(o), int(n)) for a, b, o, n in idx}
    src = json.loads((SCENE / "privileged" / "sources.json").read_text())["sources_hex"]
    for w in (0, len(W) // 2, len(W) - 1):
        r = W[w]; L = int(r["L"]); s = int(r["k"]) - 1
        t = window_from_1k(SCENE, w, s)
        crop = lambda j: np.unpackbits(np.frombuffer(dz.decompress(blob[pos[(w, j)][0]:sum(pos[(w, j)])]), np.uint8),
                                       count=L ** 3).astype(bool)
        assert t.L == L and t.N == 1024 and t.origin == (int(r["x"]), int(r["y"]), int(r["z"])) and t.key == (0, w, s)
        assert np.array_equal(np.unpackbits(t.lit_bits, count=L ** 3).astype(bool), crop(s))
        assert np.array_equal(np.unpackbits(t.solid_bits, count=L ** 3).astype(bool), crop(-1))
        assert np.array_equal(t.bind_point, [float.fromhex(v) for v in src[s]])


def _container_100k_from_1k(scene, path):
    """A scene container with the RIND3D-100K keys and dtypes (rind3d_100k.physical.run_scene) holding the 1K scene."""
    W1 = _windows_1k(); idx = np.load(scene / "public" / "observations_index.npy")
    W = np.zeros(len(W1), dtype=[("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"), ("Lp", "<i4"),
                                 ("m", "<i1"), ("mask", "<u1"), ("k", "<i1")])
    for f in ("window", "x", "y", "z", "L", "mask", "k"):
        W[f] = W1[f]
    W["Lp"] = W1["L"]
    g = np.load(scene / "privileged" / "geometry_rows.npz")
    src = json.loads((scene / "privileged" / "sources.json").read_text())["sources_hex"]
    meta = dict(contract="test", slot=int(scene.name), resolution=1024, k=int(W1[0]["k"]))
    np.savez(path, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), windows=W,
             obs=np.frombuffer((scene / "public" / "observations.bin").read_bytes(), np.uint8),
             obs_index=np.asarray(idx, np.int64), geometry=np.frombuffer((scene / "privileged" / "geometry.npz").read_bytes(), np.uint8),
             P=g["P"], K=g["K"], sources_hex=np.array([[float.fromhex(v).hex() for v in q] for q in src]))


@has_release
def test_window_from_100k_layout(tmp_path):
    from rind3d_posterior.fastlabel import window_from_1k, window_from_100k
    c = tmp_path / "scene_000000.npz"; _container_100k_from_1k(SCENE, c)
    W = _windows_1k()
    for w in (0, len(W) - 1):
        for s in range(int(W[w]["k"])):
            a, b = window_from_1k(SCENE, w, s), window_from_100k(c, w, s)
            assert (a.L, a.N, a.origin, a.key) == (b.L, b.N, b.origin, b.key)
            for f in ("lit_bits", "solid_bits", "bind_point", "P", "K"):
                assert np.array_equal(getattr(a, f), getattr(b, f)), f


def _release_tasks(L_wanted=(64, 128, 256), scene=SCENE):
    scene = Path(scene); W = _windows_1k(scene); out = []
    for L in L_wanted:
        for r in W:
            if int(r["L"]) != L or not int(r["mask"]):
                continue
            j = next(j for j in range(int(r["k"])) if (int(r["mask"]) >> j) & 1)
            if (scene / "labels" / "tasks" / f"w{int(r['window']):03d}_s{j}.npz").exists():
                out.append((int(r["window"]), j, L)); break
    return out


def _gate_a_smoke(scene):
    from rind3d_posterior.fastlabel import window_from_1k
    from rind3d_posterior.fastlabel.gate_a import check_window
    recs = []
    for w, s, L in _release_tasks((64, 128), scene):
        rec = check_window(window_from_1k(scene, w, s), seed=w, n_b=12, n_near=4, n_vert=4, n_world=4, n_plane=6)
        recs.append({k: rec[k] for k in ("key", "legal", "field_diff", "batch_disagree", "bad_violation")})
    return recs


@needs_cuda
def test_gate_a_smoke():
    recs = _spawn(_gate_a_smoke, str(SCENE))
    for rec in recs:
        assert rec["field_diff"] == 0 and rec["batch_disagree"] == 0 and rec["bad_violation"] == 0, rec
    assert sum(r["legal"] for r in recs) >= 10


def _q(cell, lo):
    u, inv = np.unique(cell, return_inverse=True)
    m = np.bincount(inv, weights=lo)
    return dict(zip(u.tolist(), (m / m.sum()).tolist()))


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _fastlabel_release_tasks(scene, n_per_L=2):
    """Release labels written by the fastlabel engine."""
    scene = Path(scene); W = _windows_1k(scene); out = {}
    for r in W:
        L = int(r["L"])
        for j in range(int(r["k"])):
            p = scene / "labels" / "tasks" / f"w{int(r['window']):03d}_s{j}.npz"
            if len(out.get(L, [])) < n_per_L and (int(r["mask"]) >> j) & 1 and p.exists():
                with np.load(p) as z:
                    if json.loads(str(z["meta"])).get("engine") == "fastlabel":
                        out.setdefault(L, []).append((int(r["window"]), j))
    return [(w, s, L) for L in sorted(out) for w, s in out[L]]


def _reproduce(scene_fl, scene_exact):
    from rind3d_posterior.fastlabel import label_window_fast, window_from_1k
    bad, n_fl, tvs = [], 0, []
    for w, s, L in _fastlabel_release_tasks(scene_fl):
        n_fl += 1
        lab = label_window_fast(window_from_1k(scene_fl, w, s))
        with np.load(Path(scene_fl) / "labels" / "tasks" / f"w{w:03d}_s{s}.npz") as z:
            m = json.loads(str(z["meta"]))
            for k in ("cell", "lo", "hi", "kind"):
                if np.asarray(lab[k]).astype(z[k].dtype).tobytes() != z[k].tobytes():
                    bad.append((w, s, k))
            if float(lab["tv_bound"]) != float(z["tv_bound"]) or lab["status"] != m["status"] or \
                    json.loads(json.dumps(lab["meta"]["certificate"], default=float)) != m["certificate"]:
                bad.append((w, s, "meta"))
    for w, s, L in _release_tasks(scene=scene_exact):
        with np.load(Path(scene_exact) / "labels" / "tasks" / f"w{w:03d}_s{s}.npz") as z:
            if json.loads(str(z["meta"])).get("engine") == "fastlabel":
                continue
            ref, ref_tv = _q(z["cell"], z["lo"]), float(z["tv_bound"])
        lab = label_window_fast(window_from_1k(scene_exact, w, s))
        tvs.append((_tv(_q(lab["cell"], lab["lo"]), ref), float(lab["tv_bound"]) + ref_tv))
    return bad, n_fl, tvs


@needs_cuda
def test_label_window_fast_reproduces_release():
    """Bit-identical to the stored FastLabel labels; within the certified TV of the stored exact labels."""
    bad, n_fl, tvs = _spawn(_reproduce, str(SCENE_FL), str(SCENE))
    assert n_fl >= 3 and not bad, bad
    assert tvs and all(d <= b for d, b in tvs), tvs
