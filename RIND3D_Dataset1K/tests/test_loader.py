"""Loader (M2): every certified status is accepted by default, the one-hot fallback is explicit and labelled, weights
follow the declared population, provenance travels with every example. Uses a temporary copy of release scene 000000
(read-only source) with synthetic label files of every status; never writes to the release."""
import json, os, shutil, sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rind3d_1k import loader as LD   # noqa: E402

REL = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
pytestmark = pytest.mark.skipif(not (REL / "scenes" / "000000").exists(), reason="release not available")


def _fake_release(tmp):
    src = REL / "scenes" / "000000"
    d = tmp / "rel" / "scenes" / "000000"
    (d / "labels" / "tasks").mkdir(parents=True)
    for name in ("scene.json", "SEAL.json"):
        shutil.copy(src / name, d / name)
    os.symlink(src / "public", d / "public")
    os.symlink(src / "privileged", d / "privileged")
    W = np.load(src / "public" / "windows.npy")
    inf = [(int(r["window"]), j) for r in W for j in range(int(r["k"])) if (int(r["mask"]) >> j) & 1][:5]
    statuses = ["certified", "certified_stat", "certified_loose", "uncertified", None]
    rows = []
    for (w, j), st in zip(inf, statuses):
        if st is None:
            rows.append(dict(window=w, source=j, L=64, informative=True, status="timeout")); continue
        with np.load(src / "labels" / "tasks" / f"w{w:03d}_s{j}.npz") as z:
            cell, lo, hi, kind = z["cell"], z["lo"], z["hi"], z["kind"]
        tv = {"certified": 0.0, "certified_stat": 0.05, "certified_loose": 0.08, "uncertified": 0.5}[st]
        meta = dict(window=w, source=j, L=64, informative=True, tv_bound=tv, status=st, engine="fastlabel",
                    engine_version="test", path="test", config_sha256="x",
                    certificate=dict(kind="statistical", eps=0.05) if st == "certified_stat" else None)
        np.savez_compressed(d / "labels" / "tasks" / f"w{w:03d}_s{j}.npz", cell=cell, lo=lo, hi=hi, kind=kind,
                            tv_bound=tv, meta=json.dumps(meta))
        rows.append(dict(meta, sha256="deadbeef"))
    (d / "labels" / "LABELS.json").write_text(json.dumps(dict(coverage={}, labels=rows)))
    return tmp / "rel", inf, statuses


def _first(ds, n, **kw):
    out = []
    for ex in LD.flatten(ds, with_inputs=False, **kw):
        out.append(ex)
        if len(out) == n:
            break
    return out


def test_population_must_be_declared(tmp_path):
    root, _, _ = _fake_release(tmp_path)
    with pytest.raises(ValueError):
        next(LD.flatten(LD.RIND3D1K(root)))


def test_default_accepts_every_certified_status_and_labels_missing(tmp_path):
    root, inf, statuses = _fake_release(tmp_path)
    ds = LD.RIND3D1K(root)
    rep = {}
    ex = _first(ds, 5, population="B1K", report=rep)
    sup = [e["supervision"] for e in ex]
    assert sup == ["posterior", "posterior", "missing", "missing", "missing"]
    assert ex[1]["status"] == "certified_stat" and ex[1]["certificate"]["kind"] == "statistical"
    assert ex[2]["target"] is None and ex[2]["status"] == "certified_loose"
    assert ex[4]["status"] == "timeout"
    assert abs(ex[0]["target"]["q"].sum() - 1) < 1e-12
    assert ex[0]["provenance"]["sha256"] == "deadbeef" and ex[0]["provenance"]["engine"] == "fastlabel"
    # legacy restriction reproduces the old (biased) default
    ex_old = _first(ds, 5, population="B1K", accept_statistical=False)
    assert [e["supervision"] for e in ex_old][:2] == ["posterior", "missing"]
    # drop is counted, not silent
    rep2 = {}
    n = sum(1 for _ in zip(range(5), LD.flatten(ds, population="B1K", missing="drop", with_inputs=False, report=rep2)))
    assert sum(v[0] for k, v in rep2.items() if k.endswith(":missing")) >= 3 and n == 2   # only 2 accepted labels exist


def test_onehot_fallback_is_explicit_and_correct(tmp_path):
    root, inf, statuses = _fake_release(tmp_path)
    ds = LD.RIND3D1K(root)
    ex = _first(ds, 5, population="Tstar", fallback="onehot")
    assert [e["supervision"] for e in ex] == ["posterior", "posterior", "onehot_fallback", "onehot_fallback",
                                              "onehot_fallback"]
    src = json.loads((REL / "scenes" / "000000" / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    for e in ex:
        c = LD.cell_of(S[e["source"]])
        assert e["privileged"]["source_cell"] == c
        if e["supervision"] == "onehot_fallback":
            assert e["target"]["cells"].tolist() == [c] and e["target"]["q"].tolist() == [1.0]
        else:                                   # true source inside every certified support (E4 convention check)
            assert c in set(e["target"]["cells"].tolist())
    one = _first(ds, 5, population="Tstar", supervision="onehot")
    assert all(e["supervision"] == "onehot" for e in one)


def test_population_weights(tmp_path):
    root, _, _ = _fake_release(tmp_path)
    ds = LD.RIND3D1K(root)
    S = ds.scene(0)
    ex_b = _first(ds, 40, population="B1K")
    ex_t = _first(ds, 40, population="Tstar")
    for b, t in zip(ex_b, ex_t):
        r = S.W[b["window"]]; Lp = int(r["Lp"]); m = int(r["m"])
        om = S.weights(Lp)["combined_rb"]
        assert abs(b["weight"] - om) < 1e-15
        assert abs(t["weight"] - om / (LD.QUOTA[Lp] * S.c(Lp, m))) < 1e-15
    # rb estimator equals the stream's acceptance probability sum_m rate_m c(m) / rate_n
    sd = S.side[64]
    assert abs(S.weights(64)["rb"] - sum(a * b for a, b in zip(sd["rate_m"], S.meta["c_table"]["64"])) / sd["rate_n"]) < 1e-15
    nat = tmp_path / "nat" / "scenes" / "000000"
    nat.mkdir(parents=True)
    (nat / "natural.json").write_text(json.dumps(dict(n_p=8)))
    ex_m = _first(ds, 40, population="Tstar_mis", natural_root=tmp_path / "nat")
    for e in ex_m:
        assert 0 < e["weight"] <= 1 / 8 + 1e-15     # balance-heuristic weights are bounded by 1 / n_p
