"""E19: natural-target 100K window designs with explicit complexity-coverage safeguards (expected values; no sampling).

World model: the 1000 1K scenes' natural per-proposal probabilities pi_m(G, S, L) (rate streams), re-weighted from the
1K source-count histogram to a declared kappa (k is drawn independently of G and S, so this is a valid re-weighting).
100K scale: 100,000 scenes, split 80 / 10 / 10 (train / val / test). 4 windows per side and scene (12 total).

Two logically distinct streams per (scene, side), both declared before the scene:
* natural:  n_nat windows; accept iff admissible (no source in the box, FREE > 0, m >= 1); no thinning.
            Density of stored windows of mixed count m, per natural proposal: n_nat pi_m / P_A, P_A = sum_m pi_m.
* coverage: n_cov windows; accept an admissible proposal with probability c(m | k) (a declared table, c > 0 for every
            m <= k, c(k) = 1); density n_cov pi_m c(m) / Lambda_c, Lambda_c = sum_m pi_m c(m).
T* weight of any stored window (balance heuristic; per natural-proposal unit):
            w_m = 1 / (n_nat / P_A + n_cov c(m) / Lambda_c)   <= P_A / n_nat (bounded as long as n_nat >= 1).
P_A and Lambda_c are estimated per scene-side by the rate stream (as in the 1K); here the true values are used.
Allocation: coverage slots only for scenes with k >= k_min and sides in a declared set; elsewhere 4 natural windows.

Reported per design and kappa:
* per (L, m): expected windows, effective channels (m per window), independent scenes with >= 1 such window (test and
  train splits), within-stratum T* Kish ESS (train), and whether the minimum-coverage targets are met;
* per L: stored-window m-histogram, channel shares, overall T* ESS fraction, max weight / mean weight;
* cost: mixed-channel labels per scene and natural + coverage proposals per scene, relative to pure natural;
* per family: m = 4 independent test / train scenes at each side.
Targets (declared in the report, judgement based on the 1K pilot's scene-count scaling):
  test : >= 300 independent scenes and >= 1,000 windows per (L, m);
  train: >= 3,000 independent scenes, >= 20,000 effective channels, T* within-stratum ESS >= 5,000 per (L, m);
  family: m = 4 at every side: >= 30 independent test scenes and >= 300 train scenes per family.
usage: e19_coverage_designs.py > out/e19_coverage_designs.json"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
SIDES = (64, 128, 256)
N = 100_000; SPLIT = dict(train=0.8, test=0.1)
NW = 4
TARGET = dict(test_scenes=300, test_windows=1000, train_scenes=3000, train_channels=20000, train_ess=5000,
              fam_test_scenes=30, fam_train_scenes=300)
KAPPAS = {"1K_realised": [0.154, 0.135, 0.158, 0.553], "rising": [0.1, 0.2, 0.3, 0.4], "uniform": [0.25] * 4,
          "heavy4": [0.1, 0.1, 0.2, 0.6]}

S = []
for d in sorted((ROOT / "scenes").iterdir()):
    if not (d / "SEAL.json").exists():
        continue
    m = json.loads((d / "scene.json").read_text())
    sides = {int(s["Lp"]): s for s in m["sides"]}
    S.append(dict(k=int(m["k"]), family=m["family"],
                  pi={L: np.array(sides[L]["rate_m"], float) / sides[L]["rate_n"] for L in SIDES}))
K = np.array([s["k"] for s in S]); FAM = np.array([s["family"] for s in S]); fams = sorted(set(FAM))
nk = np.bincount(K, minlength=5)[1:]


def ctable(k, rho):
    """c(m | k) = rho^(k - m) for m <= k (the highest achievable level always accepted), 0 for m > k (impossible)."""
    return np.array([rho ** (k - m) if m <= k else 0.0 for m in range(1, 5)])


def design_eval(kappa, n_cov_by_side, rho, k_min):
    """n_cov_by_side: {L: number of the 4 windows drawn from the coverage stream} for scenes with k >= k_min."""
    om = np.array([kappa[s["k"] - 1] / nk[s["k"] - 1] for s in S])          # fraction of the 100K represented by scene s
    res = {}
    lab = prop = lab_nat = prop_nat = 0.0
    for L in SIDES:
        win = np.zeros(4); ch = np.zeros(4); sc_test = np.zeros(4); sc_train = np.zeros(4)
        s1 = s2 = 0.0; s1m = np.zeros(4); s2m = np.zeros(4); wmax = 0.0; cov_mass = tot_mass = 0.0
        fam_sc = {f: np.zeros(2) for f in fams}
        cov_props = []
        for i, s in enumerate(S):
            pi = s["pi"][L]; pa = pi.sum()
            if pa <= 0:
                continue
            nc = n_cov_by_side.get(L, 0) if s["k"] >= k_min else 0
            nn = NW - nc
            c = ctable(s["k"], rho); lam = float(np.sum(pi * c))
            d_nat = nn * pi / pa
            d_cov = nc * pi * c / lam if nc and lam > 0 else np.zeros(4)
            dens = d_nat + d_cov                                              # expected stored windows by m (per scene)
            ms = np.arange(1, 5)
            w = np.where(dens > 0, pi / np.where(dens > 0, dens, 1), 0)      # T* weight per stored window of value m
            win += om[i] * N * dens; ch += om[i] * N * dens * ms
            p_any = 1 - (1 - pi / pa) ** nn * (1 - (pi * c / lam if (nc and lam > 0) else 0)) ** nc
            sc_test += om[i] * N * SPLIT["test"] * p_any; sc_train += om[i] * N * SPLIT["train"] * p_any
            # Kish ESS over channels (train split); per natural proposal units
            a1 = ms * dens * w; a2 = ms * dens * w ** 2
            s1 += om[i] * np.sum(a1); s2 += om[i] * np.sum(a2); s1m += om[i] * a1; s2m += om[i] * a2
            if np.any(dens > 0):
                wmax = max(wmax, float(np.max(w[dens > 0]) / (pa / NW)))       # relative to the pure-natural weight pa/4
            if nc and lam > 0:
                cov_props.append((om[i], 1.0 / lam))                          # expected proposals per coverage window
            cov_mass += om[i] * np.sum(ms * d_cov * w); tot_mass += om[i] * np.sum(a1)
            fam_sc[s["family"]] += om[i] * N * np.array([SPLIT["test"], SPLIT["train"]]) * p_any[3]
            lab += om[i] * N * np.sum(ms * dens); prop += om[i] * N * (nn / pa + (nc / lam if nc and lam > 0 else 0))
            lab_nat += om[i] * N * np.sum(ms * NW * pi / pa); prop_nat += om[i] * N * NW / pa
        ntr = N * SPLIT["train"]
        ess = s1 ** 2 / s2 * ntr / 1.0                                        # (sum om = 1) -> scale to train scenes
        ess_m = np.where(s2m > 0, s1m ** 2 / np.where(s2m > 0, s2m, 1), 0) * ntr
        n_ch_train = ch * SPLIT["train"]
        meets = {f"m{m + 1}": dict(test_scenes=bool(sc_test[m] >= TARGET["test_scenes"]),
                                    test_windows=bool(win[m] * SPLIT["test"] >= TARGET["test_windows"]),
                                    train_scenes=bool(sc_train[m] >= TARGET["train_scenes"]),
                                    train_channels=bool(n_ch_train[m] >= TARGET["train_channels"]),
                                    train_ess=bool(ess_m[m] >= TARGET["train_ess"])) for m in range(4)}
        res[f"L{L}"] = dict(
            window_hist=(win / win.sum()).round(4).tolist(), channel_share=(ch / ch.sum()).round(4).tolist(),
            test_windows=(win * SPLIT["test"]).round(0).tolist(), test_scenes=sc_test.round(0).tolist(),
            train_scenes=sc_train.round(0).tolist(), train_channels=n_ch_train.round(0).tolist(),
            train_ess_by_m=ess_m.round(0).tolist(), train_ess_total=round(ess), ess_frac=ess / float(n_ch_train.sum()),
            max_weight_over_natural=wmax,
            coverage_proposals_per_window_p50_p99=(lambda a: [float(np.quantile(a[:, 1], q, method="inverted_cdf",
                                                    weights=a[:, 0])) for q in (0.5, 0.99)])(np.array(cov_props)) if cov_props else None, coverage_share_of_Tstar_mass=cov_mass / tot_mass if tot_mass else 0.0,
            all_targets_met=all(all(v.values()) for v in meets.values()), targets=meets,
            family_m4_test_train_scenes={f: v.round(0).tolist() for f, v in fam_sc.items()},
            family_m4_targets_met=all(v[0] >= TARGET["fam_test_scenes"] and v[1] >= TARGET["fam_train_scenes"]
                                      for v in fam_sc.values()))
    res["cost"] = dict(labels_per_scene=lab / N, labels_vs_natural=lab / lab_nat, proposals_per_scene=prop / N,
                       proposals_vs_natural=prop / prop_nat)
    return res


DESIGNS = {
    "D1_natural_4+0": (dict(), 0.5, 5),
    "C31_L64_K4_rho0.5": ({64: 1}, 0.5, 4),
    "C31_L64_K3+_rho0.5": ({64: 1}, 0.5, 3),
    "C31_L64_K3+_rho0.25": ({64: 1}, 0.25, 3),
    "C31_L64L128_K3+_rho0.25": ({64: 1, 128: 1}, 0.25, 3),
    "C31_L64L128_K3+_rho0.1": ({64: 1, 128: 1}, 0.1, 3),
    "C31_all_K2+_rho0.25": ({64: 1, 128: 1, 256: 1}, 0.25, 2),
    "C22_L64L128_K3+_rho0.25": ({64: 2, 128: 2}, 0.25, 3),
    "C22_all_K2+_rho0.25": ({64: 2, 128: 2, 256: 2}, 0.25, 2),
    "C22_L64L128_K3+_rho0.1": ({64: 2, 128: 2}, 0.1, 3),
}
out = dict(targets=TARGET, kappas=KAPPAS, results={})
for kn, kap in KAPPAS.items():
    for dn, (ncov, rho, kmin) in DESIGNS.items():
        out["results"][f"{kn}:{dn}"] = design_eval(kap, ncov, rho, kmin)
json.dump(out, sys.stdout, indent=1, default=float)
