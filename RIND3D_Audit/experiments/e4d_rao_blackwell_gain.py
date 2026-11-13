"""E4d: how much variance does posterior-field supervision remove relative to one-hot, on the 333 real labels?
For a softmax Student with logits z over cells, the per-example gradient of CE(target, softmax(z)) is p - target.
With target = e_S (one-hot) vs target = q (label), E[e_S | V] = q and the conditional covariance removed is
diag(q) - q q^T, whose trace is 1 - sum_c q_c^2 (law of total variance; Menon et al. 2021 Lemma 1 for the risk).
Reported at the 128^3 label grid and at 16^3 blocks (the coarse level of the earlier Student's density).
Run: nice -n 19 python e4d_rao_blackwell_gain.py > out/e4d_rao_blackwell_gain.json"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes"); G = 128
res = {"128": [], "16": [], "L": []}
for sc in ("000000", "000001", "000002", "000003"):
    d = ROOT / sc
    for x in json.loads((d / "labels" / "LABELS.json").read_text())["labels"]:
        if x.get("status") != "certified":
            continue
        with np.load(d / "labels" / "tasks" / f"w{x['window']:03d}_s{x['source']}.npz") as z:
            cell = z["cell"].astype(np.int64); lo = z["lo"]
        cells, inv = np.unique(cell, return_inverse=True); m = np.bincount(inv, weights=lo); m = m / m.sum()
        res["128"].append(1 - float((m ** 2).sum()))
        cx, cy, cz = cells // (G * G), (cells // G) % G, cells % G
        b = (cx // 8 * 16 + cy // 8) * 16 + cz // 8
        mb = np.bincount(b, weights=m); res["16"].append(1 - float((mb ** 2).sum())); res["L"].append(int(x["L"]))
out = {}
for lev in ("128", "16"):
    v = np.array(res[lev]); L = np.array(res["L"])
    out[f"grid_{lev}"] = dict(mean_trace_removed=float(v.mean()), quantiles={str(q): float(np.quantile(v, q)) for q in (.1, .25, .5, .75, .9)},
                             by_side={str(s): float(v[L == s].mean()) for s in (64, 128, 256)},
                             note="one-hot gradient covariance trace = posterior-label trace + this quantity (per example, given V)")
json.dump(out, sys.stdout, indent=1)
