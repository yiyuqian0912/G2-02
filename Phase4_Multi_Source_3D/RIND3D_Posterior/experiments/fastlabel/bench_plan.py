"""Reproducible stratified benchmark plan (train split only; the held-out test split is never touched).
* DEV: 2 train scenes per geometry family (tuning and Gate A validation windows).
* HOLDOUT: 2 other train scenes per family, chosen with an independent seed, used only for the final assessment.
* GATEA: one informative window per (family, side) of the DEV scenes, random source channel.
Writes bench_plan.json."""
import os as _os; _P4 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../..'))   # Phase4 root
import json, sys
import numpy as np
sys.path.insert(0, _os.path.join(_P4, "RIND3D_Dataset1K"))
from rind3d_1k.loader import split_of
R = "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes"
used = {0, 1, 2, 3, 623, 905, 949, 954}          # release-label scenes and scenes already used in earlier runs (run4*)
for l in open("random120.txt"):
    used.add(int(l.split()[1]))
fams = {}
for i in range(1000):
    d = json.load(open(f"{R}/{i:06d}/scene.json"))
    if split_of(d["scene_id"]) == "train" and i not in used:
        fams.setdefault(d["family"], []).append((i, d["k"]))
rd = np.random.default_rng(20261112); rh = np.random.default_rng(777001)
dev, hold = [], []
for f in sorted(fams):
    c = fams[f]
    a = rd.choice(len(c), 2, replace=False); dev += [c[j][0] for j in a]
    rest = [x for j, x in enumerate(c) if j not in set(a)]
    # holdout: stratified on k (one scene with k = 4, one with k < 4 when available)
    k4 = [x for x in rest if x[1] == 4]; kl = [x for x in rest if x[1] < 4]
    hold += [k4[rh.integers(len(k4))][0] if k4 else rest[rh.integers(len(rest))][0],
             kl[rh.integers(len(kl))][0] if kl else rest[rh.integers(len(rest))][0]]
gate = []
rg = np.random.default_rng(5)
for i in dev[::2]:                                  # first DEV scene of each family
    W = np.load(f"{R}/{i:06d}/public/windows.npy")
    for L in (64, 128, 256):
        rows = [r for r in W if int(r["L"]) == L and int(r["mask"])]
        r = rows[rg.integers(len(rows))]; js = [j for j in range(int(r["k"])) if (int(r["mask"]) >> j) & 1]
        gate.append(f"{i:06d}:{int(r['window'])}:{js[rg.integers(len(js))]}")
json.dump(dict(dev=[f"{i:06d}" for i in dev], holdout=[f"{i:06d}" for i in hold], gatea=gate,
               note="train split only; seeds 20261112 (dev), 777001 (holdout), 5 (gate A windows)"), open("bench_plan.json", "w"), indent=1)
print(len(dev), len(hold), len(gate))
