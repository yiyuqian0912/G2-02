#!/usr/bin/env python3
"""Build the handoff reference fixtures (run once on the development machine).

* **Label fixtures** (`handoff/fixtures/labels/*.npz`): real 1024^3 tasks from the RIND3D 1K release, train-split scenes
  only. Each stores the geometry rows, origin, local classes (packed) and the reference label digest
  (sha256 of cell, lo, hi, kind, exactly as tests/test_regression.py), tv_bound, n_records and reference CPU time.
* **Physical fixtures** (`handoff/fixtures/physical/slot_<s>.json`): 100K-contract scenes regenerated from scratch.
  Each stores the windows table, a sha256 per stored crop, Lambda_hat per side, and the controller plan used.
* **Controller prior** (`handoff/prior/release_1k_rate_stats.jsonl`): the rate-stream statistics of the independent
  1K release, which the contract uses as prior.

Usage: make_fixtures.py <release_1k_root> [--physical-slots 0:4]"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]
import numpy as np
import zstandard

# (scene, window, source, purpose) - train split scenes 0-3 of release_1k_v1 (checked below)
LABEL_FIXTURES = [
    ("000000", 63, 0, "L256 single-cell, cheap"),
    ("000000", 60, 2, "L256 single-cell"),
    ("000003", 45, 1, "L256 single-cell, 50 candidate roots, heavy refutation"),
    ("000000", 49, 2, "L256 single-cell, heavy (142 s reference)"),
    ("000001", 63, 0, "L256 multi-cell (5 records)"),
    ("000002", 53, 0, "L256 heavy tail (435 s reference), 99 records, max_entries near queue cap"),
    ("000001", 53, 1, "L256 broad thin-sheet posterior (2469 records)"),
    ("000000", 35, 3, "L128 few-cell"),
    ("000000", 17, 1, "L128 broad (11920 records)"),
    ("000001", 30, 2, "L128 heavy (85 s reference)"),
    ("000000", 6, 1, "L64 broad (7922 records), cheap"),
    ("000000", 11, 2, "L64 multi-cell"),
    ("000000", 13, 3, "L64 very broad (287758 records)"),
    ("000001", 2, 2, "L64 thin_slabs"),
]


def digest(cell, lo, hi, kind):
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(cell, np.int64), np.ascontiguousarray(lo), np.ascontiguousarray(hi), np.ascontiguousarray(kind, np.int8)):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def label_fixtures(rel):
    from rind3d_1k.labels import SceneObs
    from rind3d_1k.loader import split_of
    out = HERE / "handoff" / "fixtures" / "labels"; out.mkdir(parents=True, exist_ok=True)
    log = {(r["scene"], r["window"], r["source"]): r for r in map(json.loads, open(rel / "labels_log.jsonl")) if "cpu" in r}
    cctx = zstandard.ZstdCompressor(level=9); index = []
    for sc, w, s, purpose in LABEL_FIXTURES:
        meta = json.loads((rel / "scenes" / sc / "scene.json").read_text())
        assert split_of(meta["scene_id"]) == "train", sc
        S = SceneObs(rel / "scenes" / sc); cls = S.cls(w, s)
        ref = np.load(rel / "scenes" / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz")
        lr = log[(sc, w, s)]
        name = f"{sc}_w{w:02d}_s{s}"
        dg = digest(ref["cell"], ref["lo"], ref["hi"], ref["kind"])
        np.savez(out / f"{name}.npz", P=S.P, K=S.K, origin=S.origin(w), N=np.int64(1024), L=np.int64(cls.shape[0]),
                 lit=np.frombuffer(cctx.compress(np.packbits(cls == 1).tobytes()), np.uint8),
                 solid=np.frombuffer(cctx.compress(np.packbits(cls == 2).tobytes()), np.uint8))
        index.append(dict(name=name, scene=sc, window=w, source=s, family=meta["family"], k=meta["k"], L=int(cls.shape[0]),
                          purpose=purpose, digest=dg, tv_bound=float(ref["tv_bound"]), n_records=int(len(ref["cell"])),
                          L_total=float(ref["lo"].sum()), ref_cpu_s=lr["cpu"], ref_engine=lr["engine"],
                          ref_engine_version=lr["engine_version"], ref_params=lr.get("params")))
        print(name, dg, purpose, flush=True)
    (out / "INDEX.json").write_text(json.dumps(index, indent=1))


def prior(rel):
    out = HERE / "handoff" / "prior"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "release_1k_rate_stats.jsonl", "w") as f:
        for line in open(rel / "manifest.jsonl"):
            r = json.loads(line)
            if r.get("status") != "accepted":
                continue
            f.write(json.dumps(dict(slot=r["slot"], status="accepted", k=r["k"],
                                    sides=[dict(Lp=s["Lp"], rate_m=s["rate_m"], rate_A=s["rate_A"]) for s in r["sides"]])) + "\n")


def physical_fixtures(slots):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k.controller import plan_epoch
    from rind3d_100k.physical import run_scene
    import io
    cfg = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v1.json").read_text())
    pri = HERE / "handoff" / "prior" / "release_1k_rate_stats.jsonl"
    plans = plan_epoch(cfg, 0, [], str(pri))
    out = HERE / "handoff" / "fixtures" / "physical"; out.mkdir(parents=True, exist_ok=True)
    fac = BackendFactory()
    for slot in range(*slots):
        pl = plans[slot]
        rec, payload = run_scene(cfg, slot, pl["k"], pl["c_table"], fac, 0)
        d = dict(slot=slot, k=pl["k"], c_table=pl["c_table"], status=rec["status"])
        if payload is not None:
            z = np.load(io.BytesIO(payload))
            blob = z["obs"].tobytes(); dz = zstandard.ZstdDecompressor()
            # hash the DECOMPRESSED packed bits: zstd output bytes may differ between library versions
            d.update(windows=z["windows"].tolist(), Lambda_hat=rec["Lambda_hat"], scene_id=rec["scene_id"],
                     crops={f"{int(a)}:{int(b)}": hashlib.sha256(dz.decompress(blob[int(c):int(c) + int(n)])).hexdigest()[:16]
                            for a, b, c, n in z["obs_index"]},
                     timing=rec["timing"], work=rec["work"])
        (out / f"slot_{slot:06d}.json").write_text(json.dumps(d, indent=0, default=str))
        print("physical", slot, rec["status"], rec["timing"].get("total"), flush=True)


if __name__ == "__main__":
    rel = Path(sys.argv[1])
    prior(rel)
    label_fixtures(rel)
    ps = (0, 4)
    if "--physical-slots" in sys.argv:
        a, b = sys.argv[sys.argv.index("--physical-slots") + 1].split(":"); ps = (int(a), int(b))
    physical_fixtures(ps)
