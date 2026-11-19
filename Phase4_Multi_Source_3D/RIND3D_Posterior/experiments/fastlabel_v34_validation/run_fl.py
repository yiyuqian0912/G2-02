"""Run FastLabel v3.4 at the v3 production configuration on single tasks (validation only; one process, one GPU context).

Production configuration (contract v3 labels.fastlabel): n_tasks = 4,800,000, family_delta = 0.01, base_seed = 20261118,
eps = 0.05, every other parameter = pipeline.DEFAULTS. Optionally `--validate N`: N extra points from a SEPARATE keyed
stream (SeedSequence(..., 99)) tested by the same exact verifier on the certified B (empirical estimate of
p = |F| / |B|; not part of the certificate). The validation stream never changes the label: the certificate and q_B
are computed before it is drawn.

usage: python run_fl.py OUT_DIR [--validate N] [--root ROOT] KEY [KEY ...]     (KEY = SSSSSS_wWW_sS)
Writes OUT_DIR/{key}_fl.npz (cell, lo, hi as returned) and appends OUT_DIR/fl_runs.jsonl (status, path, certificate,
rounds, validation, geometry sha256)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PROD = dict(n_tasks=4800000, family_delta=0.01, base_seed=20261118, eps=0.05)
ROOT = "/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("keys", nargs="+")
    ap.add_argument("--validate", type=int, default=0); ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    from rind3d_posterior.fastlabel import label_window_fast, window_from_100k, VERSION
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for key in a.keys:
        slot, w, s = int(key[:6]), int(key[8:10]), int(key[12:])
        cont = Path(a.root) / "scenes" / f"{slot // 1000:04d}" / f"{slot:06d}.npz"
        with np.load(cont, allow_pickle=False) as z:
            geo_sha = hashlib.sha256(z["geometry"].tobytes()).hexdigest()
        t0 = time.perf_counter()
        lab = label_window_fast(window_from_100k(cont, w, s), validate_n=a.validate, **PROD)
        wall = time.perf_counter() - t0
        np.savez_compressed(out / f"{key}_fl.npz", cell=np.asarray(lab["cell"], np.int64), lo=np.asarray(lab["lo"], float),
                            hi=np.asarray(lab["hi"], float))
        m = lab["meta"]
        rec = dict(key=key, container=str(cont), geometry_sha256=geo_sha, version=VERSION, status=lab["status"],
                   tv_bound=float(lab["tv_bound"]), path=m["path"], certificate=m.get("certificate"),
                   rounds=[{k: v for k, v in r.items() if not k.endswith("_s")} for r in m.get("rounds", [])],
                   validation=m.get("validation"), n_cells=m["n_cells"], wall=wall, config=PROD)
        with open(out / "fl_runs.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(key, rec["status"], rec["path"], "cells", rec["n_cells"], "wall %.1f" % wall,
              "validation", rec["validation"], flush=True)


if __name__ == "__main__":
    main()
