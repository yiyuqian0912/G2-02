"""Comparison tables with scene-level uncertainty.

usage: python -m slab.analyze <name=eval.npz> [...] [--sets tstar,...] [--pairs a:b,...] [--by Lp|nf|family] [--json out.json]
Primary score: information gain over the uniform-legal prior, ig = nll_prior - nll (nats; higher is better), a strictly
proper score for p*(s | X) that needs no reference posterior. Means are side-balanced (mean over L of the within-L mean;
T* gives each side 1/3) unless --pooled. 95% intervals: scene bootstrap (2000 resamples of scenes; channels and windows
of a scene are never treated as independent). Paired differences use identical rows."""
import json, sys
import numpy as np

SIDES = (64, 128, 256)


def load(path):
    z = np.load(path); out = {}
    for k in z.files:
        s, m = k.split("/", 1); out.setdefault(s, {})[m] = z[k]
    for s, d in out.items():
        d["ig"] = d["nll_prior"] - d["nll"]
        d["kl_ok"] = d["kl"]
    return out


def _stat(v, Lp, balanced):
    if not balanced:
        return np.nanmean(v)
    ms = [np.nanmean(v[Lp == L]) for L in SIDES if np.any((Lp == L) & ~np.isnan(v))]
    return float(np.mean(ms)) if ms else np.nan


def boot(v, scene, Lp, balanced=True, n=2000, seed=0):
    ok = ~np.isnan(v); v, scene, Lp = v[ok], scene[ok], Lp[ok]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    us, inv = np.unique(scene, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(us))]
    rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        pick = np.concatenate([groups[g] for g in rng.integers(0, len(us), len(us))])
        est.append(_stat(v[pick], Lp[pick], balanced))
    return dict(mean=float(_stat(v, Lp, balanced)), lo=float(np.nanpercentile(est, 2.5)), hi=float(np.nanpercentile(est, 97.5)),
                n=int(len(v)), scenes=int(len(us)))


def hpd_mae(h):
    h = h[~np.isnan(h)]
    lv = np.linspace(0.1, 0.9, 9)
    return float(np.mean(np.abs([np.mean(h <= a) - a for a in lv]))) if len(h) else np.nan


def summarise(E, sets, balanced=True):
    out = {}
    for s in sets:
        if s not in E:
            continue
        d = E[s]
        out[s] = dict(ig=boot(d["ig"], d["scene"], d["Lp"], balanced), kl=boot(d["kl"], d["scene"], d["Lp"], balanced),
                      ig_ref=float(_stat(d["ig_ref"], d["Lp"], balanced)), hpd_mae=hpd_mae(d["hpd"]),
                      by_side={int(L): float(np.nanmean(d["ig"][d["Lp"] == L])) for L in SIDES if np.any(d["Lp"] == L)})
    return out


def paired(Ea, Eb, s, metric="ig", balanced=True):
    a, b = Ea[s], Eb[s]
    assert np.array_equal(a["idx"], b["idx"]), "paired comparison needs identical rows"
    return boot(a[metric] - b[metric], a["scene"], a["Lp"], balanced)


def strata(E, s, key, metric="ig"):
    d = E[s]; v = d[key]
    if key == "nf":
        v = np.minimum(v, 4)
    return {int(k): dict(n=int(np.sum(v == k)), mean=float(np.nanmean(d[metric][v == k]))) for k in np.unique(v)}


def fmt(b):
    return f"{b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    opt = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    runs = {a.split("=", 1)[0]: load(a.split("=", 1)[1]) for a in args}
    sets = opt("--sets", "tstar,b1k,sep,alllit,alldark").split(",")
    bal = "--pooled" not in sys.argv
    res = {n: summarise(E, sets, bal) for n, E in runs.items()}
    print("| run | " + " | ".join(f"{s} IG" for s in sets) + " | T* KL | T* HPD-MAE | T* IG L64 / L128 / L256 |")
    print("|---|" + "---|" * (len(sets) + 3))
    for n, R in res.items():
        cells = [fmt(R[s]["ig"]) if s in R else "—" for s in sets]
        t = R.get("tstar", {})
        print(f"| {n} | " + " | ".join(cells) + f" | {t.get('kl', {}).get('mean', np.nan):.2f} | {t.get('hpd_mae', np.nan):.3f} | "
              + " / ".join(f"{t.get('by_side', {}).get(L, np.nan):.2f}" for L in SIDES) + " |")
    pairs = opt("--pairs", "")
    pres = {}
    if pairs:
        print("\n| paired IG difference (a − b) | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets))
        for p in pairs.split(","):
            a, b = p.split(":")
            row = {s: paired(runs[a], runs[b], s) for s in sets if s in runs[a] and s in runs[b]}
            pres[p] = row
            print(f"| {a} − {b} | " + " | ".join(fmt(row[s]) if s in row else "—" for s in sets) + " |")
    by = opt("--by", "")
    if by:
        for n, E in runs.items():
            print(f"\n{n} by {by} (tstar):", json.dumps(strata(E, "tstar", by)))
    if "--json" in sys.argv:
        json.dump(dict(summary=res, pairs=pres), open(opt("--json", ""), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
