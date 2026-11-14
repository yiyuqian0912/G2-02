"""Aggregate pilot evaluations: population-weighted means, scene-bootstrap 95% intervals, paired arm differences, strata.

usage: analyze.py <out.json> <arm>=<eval.npz> [...] [--pairs A:B,C:D]  (paired difference A minus B)
Weights per set: tstar / ood sets: equal per example within side, equal side shares (T*); b1k: omega_hat side-normalised.
Intervals: 2000 scene-bootstrap resamples (scenes are the independent units; channels of one scene share G).
Paired differences use the same examples for both arms."""
import json, sys
from pathlib import Path
import numpy as np

B = 2000
LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 2)


def load(p):
    z = np.load(p, allow_pickle=False)
    out = {}
    for k in z.files:
        s, f = k.split("/", 1)
        out.setdefault(s, {})[f] = z[k]
    return out


def weights(d, setname):
    Lp = d["Lp"]
    w = d["w_b1k"].astype(float) if setname == "b1k" else np.ones(len(Lp))
    out = np.zeros(len(Lp))
    for L in (64, 128, 256):
        s = Lp == L
        if s.any() and w[s].sum() > 0:
            out[s] = w[s] / w[s].sum() / 3.0
    return out / out.sum() if out.sum() > 0 else out


def wmean(x, w):
    ok = ~np.isnan(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def boot(fn, scenes, rng):
    us = np.unique(scenes)
    idx_by = {u: np.flatnonzero(scenes == u) for u in us}
    vals = []
    for _ in range(B):
        pick = rng.choice(us, len(us))
        ii = np.concatenate([idx_by[u] for u in pick])
        vals.append(fn(ii))
    vals = np.array(vals)
    return [float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))]


def summarise(d, setname, rng):
    w = weights(d, setname); sc = d["scene"]
    res = {}
    allidx = np.arange(len(w))
    metrics = ["nll", "nll_prior", "ce", "kl", "supp_mass", "brier16", "ent", "ig", "ig_ref", "hq"]
    for m in metrics:
        if m not in d:
            continue
        x = d[m].astype(float)
        if np.all(np.isnan(x)):
            continue
        res[m] = dict(mean=wmean(x, w), ci95=boot(lambda ii: wmean(x[ii], w[ii]), sc, rng), n=int(np.sum(~np.isnan(x))))
    gain = d["nll_prior"] - d["nll"]
    if "supp_mass" in d and "ig_ref" in d:
        # prior mass on the reference support: q_G is uniform on F at cell level up to boundary cells, so u(supp) ~ exp(-KL(q||u))
        ratio = d["supp_mass"] / np.exp(-d["ig_ref"])
        if not np.all(np.isnan(ratio)):
            res["supp_mass_over_prior"] = dict(mean=wmean(ratio, w), median=float(np.nanmedian(ratio)),
                                               ci95=boot(lambda ii: wmean(ratio[ii], w[ii]), sc, rng))
    res["info_gain_vs_prior"] = dict(mean=wmean(gain, w), ci95=boot(lambda ii: wmean(gain[ii], w[ii]), sc, rng))
    cov = {str(a): wmean((d["hpd_u"] < a).astype(float), w) for a in LEVELS}
    res["hpd_coverage"] = cov
    res["calib_mae"] = float(np.mean([abs(cov[str(a)] - a) for a in LEVELS]))
    res["calib_mae_ci95"] = boot(lambda ii: float(np.mean([abs(wmean((d["hpd_u"][ii] < a).astype(float), w[ii]) - a) for a in LEVELS])), sc, rng)
    res["by_side"] = {str(L): dict(n=int(np.sum(d["Lp"] == L)), nll=wmean(d["nll"][d["Lp"] == L], w[d["Lp"] == L]),
                                   gain=wmean(gain[d["Lp"] == L], w[d["Lp"] == L]),
                                   kl=wmean(d["kl"][d["Lp"] == L], w[d["Lp"] == L]) if "kl" in d else None)
                      for L in (64, 128, 256) if np.any(d["Lp"] == L)}
    res["by_k"] = {str(k): dict(n=int(np.sum(d["k"] == k)), gain=wmean(gain[d["k"] == k], w[d["k"] == k])) for k in range(1, 5) if np.any(d["k"] == k)}
    res["by_family"] = {str(f): dict(n=int(np.sum(d["family"] == f)), gain=wmean(gain[d["family"] == f], w[d["family"] == f]))
                        for f in np.unique(d["family"])}
    if "ig_ref" in d and np.sum(~np.isnan(d["ig_ref"])) > 10:
        ok = ~np.isnan(d["ig_ref"])
        from scipy.stats import spearmanr
        res["adaptivity_spearman_ig_vs_ig_ref"] = float(spearmanr(d["ig"][ok], d["ig_ref"][ok]).correlation)
        res["adaptivity_spearman_gain_vs_ig_ref"] = float(spearmanr(gain[ok], d["ig_ref"][ok]).correlation)
        oi = np.flatnonzero(ok)

        def rho(ii, a):
            ii = ii[ok[ii]]
            return float(spearmanr(a[ii], d["ig_ref"][ii]).correlation) if len(ii) > 5 else np.nan
        res["adaptivity_ig_ci95"] = boot(lambda ii: rho(ii, d["ig"]), sc, rng)
        res["adaptivity_gain_ci95"] = boot(lambda ii: rho(ii, gain), sc, rng)
    if "rel_p" in d:
        res["reliability"] = dict(pred=d["rel_p"].tolist(), hits=d["rel_h"].tolist(), n=d["rel_n"].tolist())
    res["n"] = int(len(w)); res["n_scenes"] = int(len(np.unique(sc)))
    return res


def main():
    out = sys.argv[1]
    pairs = None
    argv = sys.argv[2:]
    if "--pairs" in argv:
        i = argv.index("--pairs"); pairs = [tuple(x.split(":")) for x in argv[i + 1].split(",")]; argv = argv[:i] + argv[i + 2:]
    arms = dict(a.split("=", 1) for a in argv)
    data = {a: load(p) for a, p in arms.items()}
    rng = np.random.default_rng(7)
    R = dict(arms={})
    for a, D in data.items():
        R["arms"][a] = {s: summarise(d, s if s in ("b1k",) else "tstar", rng) for s, d in D.items() if len(d["Lp"])}
    # paired differences vs the first arm, per set, on the common examples
    names = list(arms)
    R["paired"] = {}
    todo = pairs if pairs is not None else [(a, b) for a in names[1:] for b in names if a != b]
    for a, b in todo:
        if a not in data or b not in data:
            continue
        if True:
            for s in data[a]:
                if s not in data[b] or not len(data[a][s]["Lp"]):
                    continue
                da, db = data[a][s], data[b][s]
                assert np.array_equal(da["idx"], db["idx"])
                w = weights(da, s if s == "b1k" else "tstar"); sc = da["scene"]
                for m in ("nll", "kl", "brier16"):
                    x = da[m].astype(float) - db[m].astype(float)
                    if np.all(np.isnan(x)):
                        continue
                    R["paired"].setdefault(f"{a}-{b}", {}).setdefault(s, {})[m] = dict(
                        mean=wmean(x, w), ci95=boot(lambda ii: wmean(x[ii], w[ii]), sc, rng))
    Path(out).write_text(json.dumps(R, indent=1))
    for a in names:
        for s, r in R["arms"][a].items():
            print(a, s, "n", r["n"], "nll %.3f" % r["nll"]["mean"], "gain %.3f" % r["info_gain_vs_prior"]["mean"],
                  "kl %.3f" % r.get("kl", {}).get("mean", float("nan")), "calib %.3f" % r["calib_mae"])


if __name__ == "__main__":
    main()
