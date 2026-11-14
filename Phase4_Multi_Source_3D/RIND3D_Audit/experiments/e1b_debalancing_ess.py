"""E1b: cost of removing the effective-layer balancing from the single-channel training population.

For a flattened informative example (scene, window W, channel j), the 1K joint contains the thinning factor
c(m(W), L, k) with m = 1 + m_{-j}; m_{-j} depends on the OTHER (hidden) sources. Weighting by omega / c(m) cancels it
exactly (target: natural window proposals, acceptance A only). m is public (windows.npy) and c is in scene.json.
This script reports the Kish ESS of omega and omega / c(m) over informative examples, per side and pooled,
and the spread of the selection factor c(m) that a single-channel Student cannot observe.
Run: nice -n 19 python e1b_debalancing_ess.py <release_root> > out/e1b_debalancing_ess.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
SIDES = (64, 128, 256)


def kish(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / np.sum(w ** 2))


ex = []   # (Lp, k, m, omega_rate, c(m))
for d in sorted((ROOT / "scenes").iterdir()):
    meta = json.loads((d / "scene.json").read_text()); W = np.load(d / "public" / "windows.npy")
    k = int(meta["k"])
    for w in W:
        Lp = int(w["Lp"]); m = int(w["m"]); c = float(meta["c_table"][str(Lp)][m - 1]); om = float(meta["Lambda_hat"][str(Lp)])
        ex += [(Lp, k, m, om, c)] * m          # m informative channels per window
ex = np.array(ex)
out = {}
for Lp in SIDES + (0,):
    e = ex if Lp == 0 else ex[ex[:, 0] == Lp]
    om, c = e[:, 3], e[:, 4]
    if Lp == 0:   # pooled: normalise omega by side mean (function of observed L only; optimum unchanged)
        om = om.copy()
        for L in SIDES:
            s = e[:, 0] == L; om[s] /= om[s].mean()
    nat = om / c
    if Lp == 0:
        for L in SIDES:
            s = e[:, 0] == L; nat[s] /= nat[s].mean()
    out["pooled" if Lp == 0 else f"L{Lp}"] = dict(
        n=int(len(e)), kish_omega=kish(om), kish_omega_over_c=kish(nat),
        c_of_examples=dict(share_c_le_0p02=float(np.mean(c <= 0.02)), quantiles={str(q): float(np.quantile(c, q)) for q in (0.05, .25, .5, .75)}),
        share_examples_by_m={str(m): float(np.mean(e[:, 2] == m)) for m in range(1, 5)},
        weight_share_by_m_under_omega_over_c={str(m): float(nat[e[:, 2] == m].sum() / nat.sum()) for m in range(1, 5)},
        top1pct_weight_share_omega_over_c=float(np.sort(nat)[::-1][:max(1, len(nat) // 100)].sum() / nat.sum()))
# within k = 4 scenes only (largest balancing effect)
for Lp in SIDES:
    e = ex[(ex[:, 0] == Lp) & (ex[:, 1] == 4)]
    out[f"L{Lp}_k4"] = dict(n=int(len(e)), kish_omega=kish(e[:, 3]), kish_omega_over_c=kish(e[:, 3] / e[:, 4]),
                            ratio_max_min_c=float(e[:, 4].max() / e[:, 4].min()))
json.dump(out, sys.stdout, indent=1)
