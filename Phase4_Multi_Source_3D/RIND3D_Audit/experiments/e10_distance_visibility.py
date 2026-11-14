"""E10: is darkness distance-based after marginalising hidden geometry? Two measurements on release_1k_v1 (read-only).

(a) World-level: per scene, random pairs (s, x) with both points FREE (float rows), P(segment blocked | length) by
    family, with a one-parameter exponential fit P(unblocked) = exp(-lambda * length) and its worst-bin error.
(b) Dataset-level: for every (accepted window, channel), the distance from source j to the window centre, split by the
    channel's response type (all-lit / all-dark / informative). This includes the selection law of the release.
Run: nice -n 19 python e10_distance_visibility.py > out/e10_distance_visibility.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
NP = 3000
rng = np.random.default_rng(10)
EDGES = np.array([0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.75])


def solid(P, K, X):
    s = np.zeros(len(X), bool)
    for c in range(len(K)):
        R = P[c, :K[c]]; s |= np.all(X @ R[:, :3].T + R[:, 3] <= 0, 1)
    return s


def blocked(P, K, A, B):
    hit = np.zeros(len(A), bool)
    for c in range(len(K)):
        R = P[c, :K[c]]
        f0 = A @ R[:, :3].T + R[:, 3]; f1 = B @ R[:, :3].T + R[:, 3]; df = f1 - f0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -f0 / df
        lo = np.where(df < 0, t, -np.inf).max(1); hi = np.where(df > 0, t, np.inf).min(1)
        par_bad = np.any((df == 0) & (f0 > 0), 1)
        hit |= (~par_bad) & (np.maximum(lo, 0) <= np.minimum(hi, 1))
    return hit


fam_acc = {}
dist = {"alllit": [], "alldark": [], "informative": []}
dist_by_side = {}
for d in sorted(ROOT.iterdir()):
    meta = json.loads((d / "scene.json").read_text()); fam = meta["family"]
    g = np.load(d / "privileged" / "geometry_rows.npz"); P, K = g["P"], g["K"]
    X = rng.random((3 * NP, 3)); X = X[~solid(P, K, X)]
    n = len(X) // 2; A, B = X[:n], X[n:2 * n]
    ln = np.linalg.norm(A - B, axis=1); bl = blocked(P, K, A, B)
    a = fam_acc.setdefault(fam, dict(n=np.zeros(len(EDGES) - 1), b=np.zeros(len(EDGES) - 1)))
    idx = np.clip(np.digitize(ln, EDGES) - 1, 0, len(EDGES) - 2)
    np.add.at(a["n"], idx, 1); np.add.at(a["b"], idx, bl)
    src = np.array([[float.fromhex(v) for v in s] for s in json.loads((d / "privileged" / "sources.json").read_text())["sources_hex"]])
    for rec in json.loads((d / "privileged" / "windows_full.json").read_text()):
        c = (np.array(rec["origin"]) + rec["L"] / 2) / meta["resolution"]
        for j, (lit, sh, so) in enumerate(rec["counts"]):
            t = "alllit" if sh == 0 and lit > 0 else ("alldark" if lit == 0 and sh > 0 else ("informative" if lit > 0 and sh > 0 else None))
            if t:
                dd = float(np.linalg.norm(src[j] - c)); dist[t].append(dd)
                dist_by_side.setdefault(f"L{rec['Lp']}", {}).setdefault(t, []).append(dd)
out = dict(config=dict(pairs_per_scene=NP, edges=EDGES.tolist()), world_level={}, dataset_level={})
for fam, a in sorted(fam_acc.items()):
    p = np.where(a["n"] > 0, a["b"] / np.maximum(a["n"], 1), np.nan)
    mid = 0.5 * (EDGES[1:] + EDGES[:-1]); ok = a["n"] > 200
    # least-squares fit of log P(unblocked) = -lambda * length through the origin
    y = np.log(np.clip(1 - p[ok], 1e-9, 1)); x = mid[ok]; lam = float(-(x @ y) / (x @ x))
    out["world_level"][fam] = dict(p_blocked=np.round(p, 4).tolist(), n=a["n"].astype(int).tolist(), exp_fit_lambda=lam,
                                   exp_fit_max_abs_err=float(np.max(np.abs((1 - np.exp(-lam * x)) - p[ok]))))
for t, v in dist.items():
    v = np.array(v); out["dataset_level"][t] = dict(n=len(v), mean=float(v.mean()), quantiles={str(q): float(np.quantile(v, q)) for q in (.1, .25, .5, .75, .9)})
out["dataset_level_by_side"] = {L: {t: dict(n=len(v), median=float(np.median(v))) for t, v in dd.items()} for L, dd in dist_by_side.items()}
json.dump(out, sys.stdout, indent=1)
