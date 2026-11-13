"""E15: window/source-count designs for RIND3D-100K (12 windows per scene, 4 per side), evaluated with the natural
per-proposal mixed-count distributions pi_m(G, S, L) of the 1000 1K scenes (rate streams, never thinned).

For each design and side:
* M-histogram of stored windows (the effective-layer balance a multi-channel study wants);
* Kish ESS fraction of the single-channel T* examples (weights that make the stored examples a T* sample);
* mixed examples (= label tasks) per scene and side;
* largest weight relative to the mean.
Designs:
  D0  m-balanced thinning c(m) (the 1K tables; T* needs omega / c)
  D1  natural windows (c = 1, quota 4; T* needs only omega = P(m >= 1), estimated per scene)
  D2  natural windows + source-count steering (kappa chosen to flatten the pooled M-histogram; k independent of G, S)
  D3  hybrid: per side 2 m-balanced + 2 natural windows, combined by the balance heuristic
  D4  D1 with a stored natural m = 0 reservoir (all-lit / all-dark only windows) for OOD (no labels needed for T*)
The scene weight of T* is per natural proposal, so ESS accounts for the spread of omega across scenes."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
SIDES = (64, 128, 256)
NW = 4                                     # windows per side in the 100K


def load():
    S = []
    for d in sorted((ROOT / "scenes").iterdir()):
        meta = json.loads((d / "scene.json").read_text())
        sides = {int(s["Lp"]): s for s in meta["sides"]}
        S.append(dict(k=int(meta["k"]), c={L: np.array(meta["c_table"][str(L)]) for L in SIDES},
                      pi={L: np.array(sides[L]["rate_m"], float) / sides[L]["rate_n"] for L in SIDES}))
    return S


def ess_frac(dens_list, target_list):
    """dens: expected stored windows per proposal unit by m (array over m=1..4) for each scene; target: natural pi_m.
    Each stored window of mixed count m carries m examples of weight target/dens. Returns Kish ESS / examples."""
    s1 = s2 = n = 0.0
    for dens, pi in zip(dens_list, target_list):
        ms = np.arange(1, 5)
        ok = dens > 0
        w = np.where(ok, pi / np.where(ok, dens, 1), 0)     # weight per stored window of mixed count m
        cnt = dens * ms                                      # expected examples by m
        s1 += np.sum(cnt * w); s2 += np.sum(cnt * w ** 2); n += np.sum(cnt)
    return (s1 ** 2 / s2) / n if s2 > 0 else 0.0


def main():
    S = load()
    out = {}
    ks = np.array([s["k"] for s in S])
    for L in SIDES:
        res = {}
        # D0: m-balanced (1K c tables), NW windows: stored density per proposal = NW * pi c / Lambda
        dens, tg, H = [], [], np.zeros(4)
        for s in S:
            pi, c = s["pi"][L], s["c"][L]; lam = np.sum(pi * c)
            if lam <= 0:
                continue
            d = NW * pi * c / lam; dens.append(d); tg.append(pi); H += d
        res["D0_mbalanced"] = dict(M_hist=(H / H.sum()).tolist(), ess_frac=ess_frac(dens, tg),
                                   examples_per_scene=float(np.mean([np.sum(d * np.arange(1, 5)) for d in dens])))
        # D1: natural, quota NW among m >= 1 windows: density NW * pi / P(m>=1); T* weight = P(m>=1) per scene
        dens, tg, H = [], [], np.zeros(4)
        for s in S:
            pi = s["pi"][L]; pa = pi.sum()
            if pa <= 0:
                continue
            d = NW * pi / pa; dens.append(d); tg.append(pi); H += d
        res["D1_natural"] = dict(M_hist=(H / H.sum()).tolist(), ess_frac=ess_frac(dens, tg),
                                 examples_per_scene=float(np.mean([np.sum(d * np.arange(1, 5)) for d in dens])))
        # D2: natural + kappa steering: choose kappa on {1..4} to flatten the pooled M-histogram (grid search)
        pik = {k: np.mean([s["pi"][L] / max(s["pi"][L].sum(), 1e-12) for s in S if s["k"] == k], 0) for k in range(1, 5)}
        best = None
        grid = [np.array(v) / 20 for v in np.ndindex(21, 21, 21) if sum(v) <= 20]
        for v in grid:
            kap = np.append(v, 1 - v.sum())
            h = sum(kap[k - 1] * pik[k] for k in range(1, 5))
            dev = float(np.sum((h - 0.25) ** 2))
            if best is None or dev < best[0]:
                best = (dev, kap, h)
        res["D2_natural_kappa_steered"] = dict(kappa=best[1].tolist(), M_hist=best[2].tolist(), ess_frac=res["D1_natural"]["ess_frac"],
                                               note="T* weights as D1; kappa is part of the declared population")
        # D3: hybrid 2 balanced + 2 natural per side
        dens, tg, H = [], [], np.zeros(4)
        for s in S:
            pi, c = s["pi"][L], s["c"][L]; lam = np.sum(pi * c); pa = pi.sum()
            if lam <= 0 or pa <= 0:
                continue
            d = 2 * pi * c / lam + 2 * pi / pa; dens.append(d); tg.append(pi); H += d
        res["D3_hybrid_2bal_2nat"] = dict(M_hist=(H / H.sum()).tolist(), ess_frac=ess_frac(dens, tg),
                                          examples_per_scene=float(np.mean([np.sum(d * np.arange(1, 5)) for d in dens])))
        # natural m = 0 share (all channels extreme) among admissible proposals: the D4 reservoir size
        res["natural_P_m_ge_1"] = float(np.mean([s["pi"][L].sum() for s in S]))
        res["natural_k4_P_m_ge_1"] = float(np.mean([s["pi"][L].sum() for s in S if s["k"] == 4]))
        out[f"L{L}"] = res
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
