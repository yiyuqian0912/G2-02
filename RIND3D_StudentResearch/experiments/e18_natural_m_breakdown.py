"""E18: natural effective-layer distribution of admissible windows, by side x physical source count K x geometry family.

Source: the 1K rate streams (1024 fixed natural proposals per scene and side, never thinned; scene.json sides[*].rate_m),
i.e. for every scene the natural per-proposal probabilities pi_m(G, S, L) of an admissible window with m = 1..4 mixed
channels. P(m >= 1) = sum_m pi_m is the natural acceptance rate (no source in box, FREE > 0, >= 1 mixed channel).
Reported per (L, K), per (L, family) for K = 4 (the only K where m = 4 exists), and per (L, family, K) in the JSON.
Shares are of WINDOWS among admissible m >= 1 windows, and of CHANNELS (examples: m channels per window).
Scenes are weighted equally (each scene one world). usage: e18_natural_m_breakdown.py > out/e18_natural_m.json"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
SIDES = (64, 128, 256)
rows = []
for d in sorted((ROOT / "scenes").iterdir()):
    if not (d / "SEAL.json").exists():
        continue
    m = json.loads((d / "scene.json").read_text())
    sides = {int(s["Lp"]): s for s in m["sides"]}
    rows.append(dict(k=int(m["k"]), family=m["family"],
                     pi={L: np.array(sides[L]["rate_m"], float) / sides[L]["rate_n"] for L in SIDES}))
fams = sorted({r["family"] for r in rows})


def summarise(sel, L):
    P = np.array([r["pi"][L] for r in sel])                   # scenes x 4
    win = P.mean(0); pa = float(win.sum())
    ch = win * np.arange(1, 5)
    # scene-level: share of scenes in which this m has natural probability >= 1% per proposal
    return dict(n_scenes=len(sel), P_m_ge_1=pa, window_share=(win / pa).round(4).tolist() if pa else None,
                channel_share=(ch / ch.sum()).round(4).tolist() if pa else None,
                per_proposal=win.round(5).tolist(),
                scenes_with_pi_m_ge_0p01=[float(np.mean(P[:, j] >= 0.01)) for j in range(4)])


out = dict(by_L_K={}, by_L_family_K4={}, by_L_family_K={}, by_L_family_all={})
for L in SIDES:
    for k in range(1, 5):
        out["by_L_K"][f"L{L}_K{k}"] = summarise([r for r in rows if r["k"] == k], L)
    for f in fams:
        out["by_L_family_K4"][f"L{L}_{f}"] = summarise([r for r in rows if r["k"] == 4 and r["family"] == f], L)
        out["by_L_family_all"][f"L{L}_{f}"] = summarise([r for r in rows if r["family"] == f], L)
        for k in range(1, 5):
            sel = [r for r in rows if r["k"] == k and r["family"] == f]
            if sel:
                out["by_L_family_K"][f"L{L}_{f}_K{k}"] = summarise(sel, L)
json.dump(out, sys.stdout, indent=1)
