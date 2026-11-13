"""E13: target population T* (natural multi-source worlds, single-channel mixed examples) vs the release population.

Read-only on release_1k_v1 (scene.json, public/windows.npy, privileged/selection_log.npz).

Outputs (JSON on stdout):
1. natural m-distribution, three independent estimates per side, with scene-bootstrap 95% intervals:
   (a) rate stream: 1024 fixed natural proposals per side and scene (scene.json sides[*].rate_m), never thinned;
   (b) the audit's estimate: release examples re-weighted by omega / c(m);
   (c) the first r_L proposals of the quota stream (selection_log; a fixed-size natural prefix, since every scene used
       >= r_L proposals), never thinned.
2. weight estimators of Lambda_L(G, S) (rate, quota, combined, and the Rao-Blackwellised rate rb = sum_m rate_m c(m) /
   1024): positivity, relative spread, correlation.
3. example weights for the two populations B1K (omega) and T* (omega / c), plus tempered (omega / c^a) and clipped
   variants: Kish ESS, CV, max/mean, top-1% share, per side and pooled (side-normalised).
4. hidden-variable shift between B1K and T* (k, family, m) as diagnostics of what the balancing changes.
5. design study (expected values computed from each scene's natural per-proposal m-distribution pi_m = rate_m / 1024):
   ESS of T* per labelled example, M-histograms, maximum weights, for release-only omega/c, a natural supplement with
   n_p fixed proposals per side, and the multiple-importance (balance-heuristic) union of both."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
SIDES = (64, 128, 256)
QUOTA = {64: 16, 128: 24, 256: 24}
ALPHA_QUOTA = {64: 0.20, 128: 0.09, 256: 0.04}
rng = np.random.default_rng(20261113)


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0


def wstats(w):
    w = np.asarray(w, float); mu = w.mean()
    return dict(n=int(len(w)), kish=kish(w), kish_frac=kish(w) / len(w), cv=float(w.std() / mu), max_over_mean=float(w.max() / mu),
                top1pct_share=float(np.sort(w)[::-1][:max(1, len(w) // 100)].sum() / w.sum()), zeros=int(np.sum(w == 0)))


scenes = []
for d in sorted((ROOT / "scenes").iterdir()):
    if not (d / "SEAL.json").exists():
        continue
    meta = json.loads((d / "scene.json").read_text())
    W = np.load(d / "public" / "windows.npy")
    log = np.load(d / "privileged" / "selection_log.npz")
    import hashlib
    v = int(hashlib.sha256(meta["scene_id"].encode()).hexdigest(), 16) % 100
    split = "train" if v < 80 else ("val" if v < 90 else "test")
    sides = {int(s["Lp"]): s for s in meta["sides"]}
    rec = dict(name=d.name, k=int(meta["k"]), family=meta["family"], split=split, c={L: np.array(meta["c_table"][str(L)]) for L in SIDES},
               W=W, sides=sides, log={L: log[f"L{L}"] for L in SIDES})
    est = {}
    for L in SIDES:
        s = sides[L]; c = rec["c"][L]; rm = np.array(s["rate_m"], float)
        rate = float(s["Lambda_hat"]); rb = float(np.sum(rm * c) / s["rate_n"])
        n = int(s["proposals"]) - int(s["reasons"].get("repeat", 0)); r = QUOTA[L]
        quota = (r - 1) / (n - 1)
        a = ALPHA_QUOTA[L]
        est[L] = dict(rate=rate, rb=rb, quota=quota, combined=(1 - a) * rate + a * quota, combined_rb=(1 - a) * rb + a * quota,
                      pi=rm / s["rate_n"], PA=s["rate_A"] / s["rate_n"])
    rec["est"] = est
    scenes.append(rec)

out = dict(n_scenes=len(scenes))

# ---------- 1. natural m-distribution ----------
def shares_by_m(per_scene):          # per_scene: array [n_scenes, 4] of example mass by m (already m-multiplied)
    tot = per_scene.sum(0); return tot / tot.sum()


def boot(per_scene, B=2000):
    n = len(per_scene); bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n); bs.append(shares_by_m(per_scene[idx]))
    bs = np.array(bs); return np.quantile(bs, 0.025, 0).tolist(), np.quantile(bs, 0.975, 0).tolist()


nat = {}
for L in SIDES:
    ms = np.arange(1, 5)
    # (a) rate stream: example mass per proposal = m * count(m); scenes weighted equally per proposal (T* weights
    # every scene by its natural admissible-window rate, which is what a fixed proposal count does)
    A = np.array([s["est"][L]["pi"] * ms for s in scenes])
    # (b) omega/c re-weighted release examples (uses the rate estimator, as the audit did)
    Bm = np.zeros((len(scenes), 4))
    # (c) first r_L proposals of the quota stream
    Cm = np.zeros((len(scenes), 4))
    for i, s in enumerate(scenes):
        Wd = s["W"][s["W"]["Lp"] == L]
        for w in Wd:
            m = int(w["m"]); Bm[i, m - 1] += m * s["est"][L]["rate"] / s["c"][L][m - 1]
        lg = s["log"][L][:QUOTA[L]]
        for row in lg:
            if row[3] == 1:
                Cm[i, int(row[4]) - 1] += row[4]
    raw = np.zeros(4)
    for s in scenes:
        for w in s["W"][s["W"]["Lp"] == L]:
            raw[int(w["m"]) - 1] += int(w["m"])
    nat[f"L{L}"] = dict(
        release_raw=(raw / raw.sum()).tolist(),
        rate_stream=dict(share=shares_by_m(A).tolist(), ci95=boot(A)),
        omega_over_c=dict(share=shares_by_m(Bm).tolist(), ci95=boot(Bm)),
        quota_prefix=dict(share=shares_by_m(Cm).tolist(), ci95=boot(Cm), n_examples=int(Cm.sum())),
        mixed_channels_per_natural_proposal=float(A.sum(1).mean()),
        admissible_rate_mean=float(np.mean([s["est"][L]["PA"] for s in scenes])))
# pooled with equal side shares (declared rho(L) = 1/3)
for key in ("rate_stream", "omega_over_c", "quota_prefix"):
    nat.setdefault("pooled_equal_sides", {})[key] = np.mean([nat[f"L{L}"][key]["share"] for L in SIDES], 0).tolist()
nat["pooled_equal_sides"]["release_raw"] = np.mean([nat[f"L{L}"]["release_raw"] for L in SIDES], 0).tolist()
out["natural_m_distribution"] = nat

# by k (rate stream): natural share of examples by m within k
byk = {}
for L in SIDES:
    for k in range(1, 5):
        A = np.array([s["est"][L]["pi"] * np.arange(1, 5) for s in scenes if s["k"] == k])
        byk[f"L{L}_k{k}"] = dict(n_scenes=len(A), share=shares_by_m(A).tolist(), mixed_per_proposal=float(A.sum(1).mean()))
out["natural_m_by_k"] = byk

# ---------- 2. Lambda estimators ----------
lam = {}
for L in SIDES:
    E = {key: np.array([s["est"][L][key] for s in scenes]) for key in ("rate", "rb", "quota", "combined", "combined_rb")}
    # within-scene Monte Carlo s.e. of rate (binomial on the thinned count) and of rb (sample s.d. of A c(M))
    se_rate, se_rb = [], []
    for s in scenes:
        pi = s["est"][L]["pi"]; c = s["c"][L]; n = 1024
        lamv = float(np.sum(pi * c)); ex2 = float(np.sum(pi * c ** 2))
        se_rate.append(np.sqrt(max(lamv * (1 - lamv), 0) / n) / max(lamv, 1e-12))
        se_rb.append(np.sqrt(max(ex2 - lamv ** 2, 0) / n) / max(lamv, 1e-12))
    lam[f"L{L}"] = dict(
        mean={k: float(v.mean()) for k, v in E.items()}, min={k: float(v.min()) for k, v in E.items()},
        zeros={k: int(np.sum(v == 0)) for k, v in E.items()},
        corr_rate_rb=float(np.corrcoef(E["rate"], E["rb"])[0, 1]), corr_rate_quota=float(np.corrcoef(E["rate"], E["quota"])[0, 1]),
        median_rel_se_rate=float(np.median(se_rate)), median_rel_se_rb=float(np.median(se_rb)))
out["lambda_estimators"] = lam

# ---------- 3. example weights ----------
ex = []   # per informative example: L, k, m, c, rate, rb, quota, combined, combined_rb, family idx, split, scene idx
fams = sorted({s["family"] for s in scenes})
for i, s in enumerate(scenes):
    for w in s["W"]:
        L = int(w["Lp"]); m = int(w["m"]); c = float(s["c"][L][m - 1]); e = s["est"][L]
        for j in range(int(w["k"])):
            if (int(w["mask"]) >> j) & 1:
                ex.append((L, s["k"], m, c, e["rate"], e["rb"], e["quota"], e["combined"], e["combined_rb"], fams.index(s["family"]),
                           ("train", "val", "test").index(s["split"]), i))
ex = np.array(ex)
out["n_examples"] = int(len(ex))
EST = dict(rate=4, rb=5, quota=6, combined=7, combined_rb=8)


def side_norm(w, Lcol):
    w = w.copy()
    for L in SIDES:
        sel = Lcol == L; w[sel] /= w[sel].mean()
    return w


wt = {}
for name, col in EST.items():
    om = ex[:, col]
    for pop, w in (("B1K_omega", om), ("Tstar_omega_over_c", om / ex[:, 3])):
        r = {f"L{L}": wstats(w[ex[:, 0] == L]) for L in SIDES}
        r["pooled_side_normalised"] = wstats(side_norm(w, ex[:, 0]))
        wt[f"{pop}:{name}"] = r
for a in (0.25, 0.5, 0.75):
    w = ex[:, 8] / ex[:, 3] ** a
    wt[f"tempered_a{a}:combined_rb"] = dict(pooled_side_normalised=wstats(side_norm(w, ex[:, 0])))
for q in (0.99, 0.95):
    w = side_norm(ex[:, 8] / ex[:, 3], ex[:, 0]); cap = np.quantile(w, q)
    wt[f"Tstar_clipped_q{q}:combined_rb"] = dict(pooled_side_normalised=wstats(np.minimum(w, cap)), cap=float(cap),
                                                  clipped_mass_share=float(np.sum(w - np.minimum(w, cap)) / w.sum()))
out["example_weights"] = wt

# ---------- 4. hidden-variable shift ----------
def wshare(w, col, vals):
    return {str(v): float(w[ex[:, col] == v].sum() / w.sum()) for v in vals}


wb = side_norm(ex[:, 8], ex[:, 0]); wn = side_norm(ex[:, 8] / ex[:, 3], ex[:, 0])
out["hidden_shift"] = dict(
    k=dict(B1K=wshare(wb, 1, range(1, 5)), Tstar=wshare(wn, 1, range(1, 5)), raw=wshare(np.ones(len(ex)), 1, range(1, 5))),
    m=dict(B1K=wshare(wb, 2, range(1, 5)), Tstar=wshare(wn, 2, range(1, 5))),
    family=dict(B1K={fams[int(f)]: v for f, v in wshare(wb, 9, range(len(fams))).items() for f in [f]},
                Tstar={fams[int(f)]: v for f, v in wshare(wn, 9, range(len(fams))).items() for f in [f]}))

# ---------- 5. design study (expected ESS from natural pi_m) ----------
des = {}
for L in SIDES:
    nL = QUOTA[L]
    for n_p in (0, 4, 8, 16, 24, 64):
        S1 = S2 = Nrel = Nsup = 0.0; wmax = 0.0
        S1r = S2r = 0.0
        for s in scenes:
            pi = s["est"][L]["pi"]; c = s["c"][L]; lamv = float(np.sum(pi * c))
            if lamv <= 0:
                continue
            ms = np.arange(1, 5)
            nat_mass = float(np.sum(ms * pi))             # target example mass per proposal unit (T*)
            dens_rel = nL * c / lamv                        # release windows per proposal unit, by m
            S1 += nat_mass
            S2 += float(np.sum(ms * pi / (dens_rel + n_p)))  # balance-heuristic weights 1/(dens_rel + n_p)
            Nrel += nL * float(np.sum(ms * pi * c)) / lamv; Nsup += n_p * nat_mass
            wmax = max(wmax, float(np.max(1 / (dens_rel + n_p)) / (1 / (nL + n_p))))
            # release-only omega/c weights (n_p = 0)
            S2r += float(np.sum(ms * pi * lamv / (nL * c)))
        des[f"L{L}_np{n_p}"] = dict(n_release_examples=Nrel, n_supplement_examples=Nsup, ess_mis=S1 ** 2 / S2,
                                    ess_release_only=S1 ** 2 / S2r if n_p == 0 else None,
                                    ess_supplement_only=Nsup if n_p else None,
                                    ess_mis_per_labelled_example=S1 ** 2 / S2 / (Nrel + Nsup))
out["design_mis_supplement"] = des
json.dump(out, sys.stdout, indent=1, default=float)
