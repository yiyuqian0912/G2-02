"""E1: selection statistics, training weights and effective sample sizes of release_1k_v1 (metadata only).

Reads scene.json, public/windows.npy and privileged/windows_full.json of every scene (read-only).
Run: nice -n 19 python e1_weights_selection.py <release_root> > out/e1_weights_selection.json
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
QUOTA = {64: 16, 128: 24, 256: 24}
ALPHA = {64: 0.20, 128: 0.09, 256: 0.04}
SIDES = (64, 128, 256)


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


rows = []          # one per scene
win = []           # one per window: (scene, Lp, m, k, family, lam_rate, lam_quota, lam_comb, n_inf, n_alllit, n_alldark)
for d in sorted((ROOT / "scenes").iterdir()):
    meta = json.loads((d / "scene.json").read_text())
    W = np.load(d / "public" / "windows.npy")
    full = json.loads((d / "privileged" / "windows_full.json").read_text())
    k = int(meta["k"]); fam = meta["family"]
    lam = {}
    for s in meta["sides"]:
        Lp = int(s["Lp"]); r = QUOTA[Lp]; n = int(s["proposals"]) - int(s["reasons"].get("repeat", 0))
        q = (r - 1) / (n - 1)
        lam[Lp] = (float(s["Lambda_hat"]), q, (1 - ALPHA[Lp]) * float(s["Lambda_hat"]) + ALPHA[Lp] * q,
                   float(s["P_A_hat"]), int(s["proposals"]), dict(s["reasons"]), list(s["rate_m"]))
    rows.append(dict(slot=int(meta["slot"]), k=k, family=fam, lam=lam, c=meta["c_table"]))
    for w, rec in zip(W, full):
        Lp = int(w["Lp"]); m = int(w["m"])
        cnt = np.asarray(rec["counts"])      # (k, 3): lit, shadow, solid
        lit, sh = cnt[:, 0], cnt[:, 1]
        inf = (lit > 0) & (sh > 0)
        assert int(inf.sum()) == m
        alllit = int(np.sum((sh == 0) & (lit > 0))); alldark = int(np.sum((lit == 0) & (sh > 0)))
        litfrac_inf = (lit[inf] / (lit[inf] + sh[inf])).tolist()
        win.append((int(meta["slot"]), Lp, m, k, fam, *lam[Lp][:3], int(inf.sum()), alllit, alldark,
                    float(cnt[0, 2]) / (cnt[0].sum()), litfrac_inf))

out = {}
# ---------------- per-side weight statistics
fams = sorted({r["family"] for r in rows})
for Lp in SIDES:
    lr = np.array([r["lam"][Lp][0] for r in rows]); lq = np.array([r["lam"][Lp][1] for r in rows])
    lc = np.array([r["lam"][Lp][2] for r in rows]); pa = np.array([r["lam"][Lp][3] for r in rows])
    ks = np.array([r["k"] for r in rows])
    # window-level weights: each scene contributes QUOTA windows of side Lp with the same omega
    # example-level: each window contributes m informative examples
    ex_w = {e: [] for e in ("rate", "quota", "combined")}
    for t in win:
        if t[1] != Lp:
            continue
        for e, v in zip(("rate", "quota", "combined"), t[5:8]):
            ex_w[e] += [v] * t[8]
    n_ex = len(ex_w["rate"])
    out[f"L{Lp}"] = dict(
        scenes=len(rows), windows=int(QUOTA[Lp] * len(rows)), informative_examples=n_ex,
        lam_rate=dict(mean=float(lr.mean()), median=float(np.median(lr)), q01=float(np.quantile(lr, .01)),
                      q99=float(np.quantile(lr, .99)), min=float(lr.min()), max=float(lr.max()),
                      n_zero=int((lr == 0).sum()), n_below_1pct=int((lr < 0.01).sum())),
        lam_quota=dict(mean=float(lq.mean()), median=float(np.median(lq)), min=float(lq.min())),
        corr_rate_quota=float(np.corrcoef(lr, lq)[0, 1]),
        mean_by_k={str(k): float(lr[ks == k].mean()) for k in range(1, 5)},
        P_A_mean_by_k={str(k): float(pa[ks == k].mean()) for k in range(1, 5)},
        kish_windows={e: kish(np.repeat(v, QUOTA[Lp])) for e, v in (("rate", lr), ("quota", lq), ("combined", lc))},
        kish_examples={e: kish(v) for e, v in ex_w.items()},
        kish_scenes={e: kish(v) for e, v in (("rate", lr), ("quota", lq), ("combined", lc))},
        # relative noise of the rate estimator at each scene's own Lambda (binomial, 1024 proposals)
        rate_rel_se_median=float(np.median(np.sqrt(np.maximum(lr * (1 - lr), 1e-12) / 1024) / np.maximum(lr, 1e-12))),
        max_over_mean_rate=float(lr.max() / lr.mean()),
        share_weight_top10pct_scenes=float(np.sort(lr)[::-1][:len(lr) // 10].sum() / lr.sum()),
    )
# all sides pooled, raw omega vs omega normalised by its side mean (a function of the observed L only)
allw = np.array([t[5] for t in win for _ in range(t[8])]); allL = np.array([t[1] for t in win for _ in range(t[8])])
norm = allw.copy()
for Lp in SIDES:
    norm[allL == Lp] /= allw[allL == Lp].mean()
out["pooled"] = dict(examples=int(len(allw)), kish_raw_rate=kish(allw), kish_side_normalised_rate=kish(norm),
                     weight_share_by_side_raw={str(Lp): float(allw[allL == Lp].sum() / allw.sum()) for Lp in SIDES},
                     example_share_by_side={str(Lp): float(np.mean(allL == Lp)) for Lp in SIDES})
# ---------------- k by family, examples by k
kf = {f: [0] * 4 for f in fams}
for r in rows:
    kf[r["family"]][r["k"] - 1] += 1
tab = np.array([kf[f] for f in fams], float)
exp_ = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
chi2 = float(((tab - exp_) ** 2 / exp_).sum())
from scipy.stats import chi2 as C2
out["k_by_family"] = dict(table=kf, chi2=chi2, dof=(len(fams) - 1) * 3, p_value=float(C2.sf(chi2, (len(fams) - 1) * 3)))
exk = {k: 0 for k in range(1, 5)}; wk = {k: 0.0 for k in range(1, 5)}
for t in win:
    exk[t[3]] += t[8]; wk[t[3]] += t[8] * t[5]
tot = sum(exk.values()); totw = sum(wk.values())
out["examples_by_k"] = {str(k): dict(scenes=int(sum(r["k"] == k for r in rows)), informative_examples=exk[k],
                                      share=exk[k] / tot, weighted_share=wk[k] / totw) for k in range(1, 5)}
out["total_informative_examples"] = tot
# ---------------- uninformative channels in accepted windows: all-lit vs all-dark
alllit = sum(t[9] for t in win); alldark = sum(t[10] for t in win)
out["uninformative_channels_in_accepted_windows"] = dict(all_lit=alllit, all_dark=alldark,
                                                         by_side={str(Lp): dict(all_lit=sum(t[9] for t in win if t[1] == Lp),
                                                                                all_dark=sum(t[10] for t in win if t[1] == Lp)) for Lp in SIDES})
# lit fraction of informative channels (how 'thin' are the shadows)
lf = np.array([x for t in win for x in t[12]])
out["informative_lit_fraction"] = dict(n=int(len(lf)), quantiles={str(q): float(np.quantile(lf, q)) for q in (0.01, .05, .1, .25, .5, .75, .9, .95, .99)},
                                       share_lit_ge_0p99=float(np.mean(lf >= 0.99)), share_lit_le_0p01=float(np.mean(lf <= 0.01)),
                                       share_dark_lt_1pct_or_lit_lt_1pct=float(np.mean((lf >= .99) | (lf <= .01))))
sf = np.array([t[11] for t in win])
out["solid_fraction_of_windows"] = dict(mean=float(sf.mean()), share_zero=float(np.mean(sf == 0)), q90=float(np.quantile(sf, .9)))
# ---------------- thinning table: how variable is c over the campaign (controller non-stationarity)
cstat = {}
for Lp in SIDES:
    for k in range(1, 5):
        cs = np.array([r["c"][str(Lp)][:k] for r in rows if r["k"] == k])
        if len(cs):
            cstat[f"L{Lp}_k{k}"] = dict(n=len(cs), min=cs.min(0).round(4).tolist(), max=cs.max(0).round(4).tolist(),
                                        median=np.median(cs, 0).round(4).tolist())
out["c_table_range"] = cstat
# ---------------- proposals and closeness to the cap (scene-rejection risk)
for Lp in SIDES:
    pr = np.array([r["lam"][Lp][4] for r in rows])
    out[f"L{Lp}"]["proposals"] = dict(mean=float(pr.mean()), max=int(pr.max()), n_over_3000=int((pr > 3000).sum()),
                                      repeats=int(sum(r["lam"][Lp][5].get("repeat", 0) for r in rows)),
                                      contains_source=int(sum(r["lam"][Lp][5].get("contains_source", 0) for r in rows)))
json.dump(out, sys.stdout, indent=1)
