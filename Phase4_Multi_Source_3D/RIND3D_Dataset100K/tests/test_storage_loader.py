"""Durable storage, the v2 validator rules and the training loader on a synthetic root (no GPU)."""
import io
import json

import numpy as np
import pytest
import zstandard

from rind3d_100k import loader as LD
from rind3d_100k.storage import (Journal, append_line, atomic_write, journal_index, labels_path, read_jsonl,
                                 scene_path, sha256_file, task_key)
from rind3d_100k.validate import check_sampling_v2


# ------------------------------------------------------------------ storage durability
def test_read_jsonl_torn_tail(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"')
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}')
    assert [r["a"] for r in read_jsonl(p)] == [1, 2, 3]
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)


def test_append_line_and_atomic_write(tmp_path):
    q = tmp_path / "n.jsonl"
    for i in range(3):
        append_line(q, json.dumps(dict(i=i)))
    assert [r["i"] for r in read_jsonl(q)] == [0, 1, 2]
    w = tmp_path / "x" / "f.bin"
    assert atomic_write(w, b"hello") == sha256_file(w)


def test_journal_index_ignores_torn_and_out_of_range_lines(tmp_path):
    j = Journal(tmp_path, "w0")
    j.append("000000:00:0", dict(cell=np.arange(3, dtype=np.int32)), dict(status="certified"))
    loc = j.append("000000:00:1", dict(cell=np.arange(5, dtype=np.int32)), dict(status="certified"))
    j.close()
    with open(tmp_path / "labels" / "journal" / "w0.idx", "a") as f:
        f.write(f"000000:01:0\t{loc[1]}\t99999999\n000000:01:1\t12")
    assert sorted(journal_index(tmp_path)) == ["000000:00:0", "000000:00:1"]


# ------------------------------------------------------------------ synthetic v2 root
L = 8
W_DTYPE = [("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"), ("Lp", "<i4"), ("m", "<i1"),
           ("mask", "<u1"), ("k", "<i1"), ("stream", "<i1")]


def _side(Lp, cov):
    sd = dict(Lp=Lp, L=L, quota=4, rate_n=100, rate_A=50, rate_m=[20, 15, 10, 5], rate_exact=100, P_A_hat=0.5,
              Lambda_hat=0.5, proposals=20, reasons={},
              streams=[dict(name="natural", n_target=3 if cov else 4, c=[1.0] * 4, accepted=3 if cov else 4,
                            proposals=10, reasons={}, fallback=0)])
    if cov:
        sd["streams"].append(dict(name="coverage", n_target=1, c=[0.125, 0.25, 0.5, 1.0], accepted=1, proposals=7, reasons={}))
    return sd


def make_root(tmp_path, k=2, statuses=("certified_stat", "uncertified")):
    """One scene, 12 windows (4 per side; one L64 coverage window), k = 2 sources, labels for every mixed channel."""
    root = tmp_path / "root"; slot = 0
    src = np.array([[0.9, 0.9, 0.9], [0.05, 0.9, 0.5]])
    cctx = zstandard.ZstdCompressor(); blob = io.BytesIO(); index = []; W = []; full = []
    rng = np.random.default_rng(0)
    for w in range(12):
        Lp = (64, 128, 256)[w // 4]; stream = 1 if w == 3 else 0
        sol = np.zeros((L, L, L), bool); sol[0] = True
        lits = []
        for j in range(k):
            lit = rng.random((L, L, L)) < 0.5; lit &= ~sol
            if j == 1 and w % 2:                       # make channel 1 all-dark in odd windows (uninformative)
                lit[:] = False
            lits.append(lit)
        inf = [bool(lit[~sol].any() and (~lit[~sol]).any()) for lit in lits]
        mask = sum(1 << j for j in range(k) if inf[j]); m = sum(inf)
        for j, a in [(-1, sol)] + list(enumerate(lits)):
            b = cctx.compress(np.packbits(a.ravel()).tobytes()); index.append((w, j, blob.tell(), len(b))); blob.write(b)
        W.append((w, 10 * w, 0, 0, L, Lp, m, mask, k, stream))
        c_row = [0.125, 0.25, 0.5, 1.0] if stream == 1 else [1.0] * 4
        full.append(dict(stream=("natural", "coverage")[stream], U=0.01, c_row=c_row, proposal=1,
                         counts=[[int(l.sum()), int((~sol & ~l).sum()), int(sol.sum())] for l in lits]))
    meta = dict(contract="rind3d-100k-contract-v2", scene_id="ab" * 32, slot=slot, resolution=1024, k=k,
                sides=[_side(64, True), _side(128, False), _side(256, False)], windows_full=full,
                sampling=dict(law="declared", k=k, k_from_draw=k,
                              selection=dict(coverage=dict(enabled=True, sides=[64], k=[k], slots=1))))
    p = scene_path(root, slot); p.parent.mkdir(parents=True)
    buf = io.BytesIO()
    np.savez(buf, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), windows=np.array(W, dtype=W_DTYPE),
             obs=np.frombuffer(blob.getvalue(), np.uint8), obs_index=np.array(index, np.int64),
             geometry=np.zeros(1, np.uint8), P=np.zeros((1, 6, 4)), K=np.array([6]),
             sources_hex=np.array([[float(v).hex() for v in s] for s in src]))
    atomic_write(p, buf.getvalue())
    rec = dict(slot=slot, status="accepted", scene_id=meta["scene_id"], k=k, contract=meta["contract"],
               windows=[[L, w_[7], w_[6]] for w_ in W])
    append_line(root / "manifest" / "physical.jsonl", json.dumps(rec))
    # consolidated labels: the true-source cell plus one neighbour per mixed channel
    cells, los, his, offs, metas, status = [], [], [], [0], [], {}
    for w_ in W:
        w, mask = w_[0], w_[7]
        for j in range(k):
            key = task_key(slot, w, j)
            if not (mask >> j) & 1:
                status[key] = "not_computed"; continue
            st = statuses[(w + j) % len(statuses)]
            c0 = LD.cell_of(src[j]); cc = np.array([c0, c0 + 1], np.int32)
            cells.append(cc); los.append(np.array([0.75, 0.25])); his.append(np.array([0.75, 0.25])); offs.append(offs[-1] + 2)
            metas.append(dict(key=key, source=j, status=st, tv_bound=0.05 if st == "certified_stat" else 0.3, n_cells=2,
                              certificate=dict(kind="statistical", eps=0.05, delta_label=1e-9, K=1, a2=16) if st == "certified_stat" else None,
                              engine="fastlabel"))
            status[key] = st
    lb = io.BytesIO()
    np.savez_compressed(lb, cell=np.concatenate(cells), lo=np.concatenate(los), hi=np.concatenate(his),
                        offsets=np.array(offs, np.int64), metas=np.frombuffer(json.dumps(metas).encode(), np.uint8),
                        status=np.frombuffer(json.dumps(status).encode(), np.uint8))
    atomic_write(labels_path(root, slot), lb.getvalue())
    return root, src


def test_flatten_requires_population(tmp_path):
    root, _ = make_root(tmp_path)
    with pytest.raises(ValueError):
        next(LD.flatten(LD.RIND3D100K(root)))


def test_tstar_weights_and_streams(tmp_path):
    root, _ = make_root(tmp_path)
    ds = LD.RIND3D100K(root)
    ex = list(LD.flatten(ds, population="Tstar", estimator="rate"))
    S = ds.scene(0)
    for e in ex:
        m = int(S.S.W[e["window"]]["m"]); Lp = e["Lp"]
        if Lp == 64:
            lc = (20 * 0.125 + 15 * 0.25 + 10 * 0.5 + 5 * 1.0) / 100
            c = [0.125, 0.25, 0.5, 1.0][m - 1]
            assert e["weight"] == pytest.approx(1 / (3 / 0.5 + c / lc))
        else:
            assert e["weight"] == pytest.approx(0.5 / 4)
    assert {e["stream"] for e in ex} == {"natural", "coverage"}
    nat = list(LD.flatten(ds, population="Tstar_natural", estimator="rate"))
    assert all(e["stream"] == "natural" for e in nat) and len(nat) < len(ex)
    assert all(e["weight"] == pytest.approx(0.5 / (3 if e["Lp"] == 64 else 4)) for e in nat)


def test_supervision_statuses_fallback_and_inputs(tmp_path):
    root, src = make_root(tmp_path)
    ds = LD.RIND3D100K(root)
    rep = {}
    ex = list(LD.flatten(ds, population="Tstar", report=rep))
    sup = {e["supervision"] for e in ex}
    assert sup == {"posterior", "missing"}                     # certified_stat accepted, uncertified not
    assert all(e["status"] == "certified_stat" for e in ex if e["supervision"] == "posterior")
    assert all(e["certificate"]["kind"] == "statistical" for e in ex if e["supervision"] == "posterior")
    assert rep["status_not_accepted"] == {"uncertified": sum(e["supervision"] == "missing" for e in ex)}
    fb = list(LD.flatten(ds, population="Tstar", fallback="onehot"))
    for e in fb:
        if e["supervision"] == "onehot_fallback":
            assert e["target"]["cells"].tolist() == [LD.cell_of(src[e["source"]])]
            assert e["privileged"]["source_cell"] == LD.cell_of(src[e["source"]])
    dropped = list(LD.flatten(ds, population="Tstar", missing="drop"))
    assert all(e["supervision"] == "posterior" for e in dropped)
    x = LD.student_inputs(ex[0])
    assert set(x) == {"solid", "response", "L", "origin"} and x["response"].shape == (L, L, L)


def test_validator_sampling_rules(tmp_path):
    root, _ = make_root(tmp_path)
    S = LD.Scene(root, 0)
    errs = []
    check_sampling_v2(S.meta, S.S.W, S.S.k, errs)
    assert errs == []
    bad = json.loads(json.dumps(S.meta)); bad["sides"][0]["streams"][1]["accepted"] = 0
    errs = []; check_sampling_v2(bad, S.S.W, S.S.k, errs)
    assert any("stream counts" in e for e in errs)
    bad = json.loads(json.dumps(S.meta)); bad["sides"][1]["rate_m"] = [1, 0, 0, 0]
    errs = []; check_sampling_v2(bad, S.S.W, S.S.k, errs)
    assert any("rate stream" in e for e in errs)
