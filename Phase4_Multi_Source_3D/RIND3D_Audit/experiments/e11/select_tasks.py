"""E11 task selection: uninformative (all-lit / all-dark) channels and solid-separated informative channels from
TRAIN-split scenes (never test), for a FastLabel certification-feasibility probe. Read-only.
Output lines: scene_dir window source L type."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K")
from rind3d_1k.loader import split_of
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
rng = np.random.default_rng(13)
per = {(Lp, t): [] for Lp in (64, 128, 256) for t in ("alllit", "alldark")}
for d in sorted(ROOT.iterdir())[10:400]:
    meta = json.loads((d / "scene.json").read_text())
    if split_of(meta["scene_id"]) != "train":
        continue
    full = json.loads((d / "privileged" / "windows_full.json").read_text())
    for w, rec in enumerate(full):
        cnt = np.asarray(rec["counts"])
        for j in range(len(cnt)):
            lit, sh = cnt[j, 0], cnt[j, 1]
            t = "alllit" if (sh == 0 and lit > 0) else ("alldark" if (lit == 0 and sh > 0) else None)
            if t:
                per[(int(rec["Lp"]), t)].append((str(d), w, j, int(rec["Lp"]), t))
for key, v in per.items():
    for i in rng.choice(len(v), min(6, len(v)), replace=False):
        print(*v[i])
# solid-separated informative channels found by E9 (first examples at L64 / L128 in train scenes)
for f in ("e9_L64.json", "e9_L128.json"):
    ex = json.load(open(Path(__file__).resolve().parent.parent / "out" / f))["examples"]
    k = 0
    for e in ex:
        d = ROOT / e["scene"]
        if split_of(json.loads((d / "scene.json").read_text())["scene_id"]) == "train" and k < 3:
            print(str(d), e["window"], e["source"], ex and json.load(open(Path(__file__).resolve().parent.parent / "out" / f))["Lp"], "separated"); k += 1
