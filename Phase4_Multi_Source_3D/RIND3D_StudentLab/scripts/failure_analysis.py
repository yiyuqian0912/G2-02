"""Failure analysis of a projective Student on dev T* (diagnostic only: the privileged true source is used to *score*
facets, never as model input). usage: failure_analysis.py [run=eval.npz ...] [--json out.json]

For every row it measures the distance of the true source to each X-measured facet plane (>= 100 inliers) and calls the
row "law-violating" when some large facet misses the source by > 4 voxels (an occluder interface or a mis-fit). It then
tabulates each run's IG against the label's IG by facet count x violation, by family, and describes the lower tail."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slab.analyze import load, boot, _stat          # noqa: E402
from slab.paths import FEATS, facets_file           # noqa: E402

FAM = ("barriers", "convex_polyhedra", "corridors", "multiscale_convex", "occluder_clusters", "prisms", "rotated_boxes",
       "thin_slabs", "wedges")


def facet_miss(idx, big=100, tol=4.0):
    rows = np.concatenate([np.load(p) for p in sorted(FEATS.glob("shard*.npy"), key=lambda p: int(p.stem[5:]))])
    F = np.load(facets_file("v2"))["F"][idx]
    r = rows[idx]
    s = np.stack([r["sx"], r["sy"], r["sz"]], 1) * 1024          # facets are stored in world voxels
    d = np.abs(np.einsum("nkc,nkc->nk", s[:, None, :] - F[..., :3], F[..., 3:6]))
    m = F[..., 6] >= big
    dmax = np.where(m, d, 0).max(1)
    nbig = m.sum(1)
    return dmax, nbig, (m & (d > tol)).sum(1)


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    runs = {a.split("=", 1)[0]: load(a.split("=", 1)[1])["tstar"] for a in args}
    ref = next(iter(runs.values()))
    idx, Lp, fam, scene = ref["idx"], ref["Lp"], ref["family"], ref["scene"]
    for d in runs.values():
        assert np.array_equal(d["idx"], idx)
    dmax, nbig, nviol = facet_miss(idx)
    viol = nviol > 0
    lab = ref["ig_ref"]
    out = dict(n=int(len(idx)), violating_rows=float(viol.mean()))
    print(f"dev T* rows {len(idx)}; rows with a large facet missing the source by > 4 vox: {viol.mean():.1%}")
    nfb = np.minimum(nbig, 3)
    hdr = "| stratum | n | label IG | " + " | ".join(f"{k} IG" for k in runs) + " |"
    print("\n" + hdr + "\n|---|---|---|" + "---|" * len(runs))
    tab = {}
    for k in range(4):
        for v in (False, True):
            m = (nfb == k) & (viol == v)
            if m.sum() < 20:
                continue
            name = f"{k if k < 3 else '>=3'} large facets, {'violating' if v else 'law-obeying'}"
            vals = {n: float(_stat(d["ig"][m], Lp[m], True)) for n, d in runs.items()}
            tab[name] = dict(n=int(m.sum()), label=float(_stat(lab[m], Lp[m], True)), **vals)
            print(f"| {name} | {m.sum()} | {tab[name]['label']:.2f} | " + " | ".join(f"{vals[n]:.2f}" for n in runs) + " |")
    out["strata"] = tab
    last = list(runs)[-1]; gap = lab - runs[last]["ig"]
    print(f"\nshare of the pooled label - {last} gap ({np.nanmean(gap):.2f} nats/row) by stratum:")
    for k in range(4):
        for v in (False, True):
            m = (nfb == k) & (viol == v)
            if m.sum():
                print(f"  {k if k < 3 else '>=3'} large, {'violating' if v else 'obeying'}: rows {m.mean():.1%}, "
                      f"mean gap {np.nanmean(gap[m]):.2f}, contribution {np.nansum(gap[m]) / len(gap):.2f}")
                tab.setdefault(f"contrib_{k}_{int(v)}", float(np.nansum(gap[m]) / len(gap)))
    print("\n| family | n | violating | label IG | " + " | ".join(f"{k} IG" for k in runs) + " | gap to label (last run) |")
    print("|---|---|---|---|" + "---|" * (len(runs) + 1))
    fams = {}
    for f in range(len(FAM)):
        m = fam == f
        vals = {n: float(_stat(d["ig"][m], Lp[m], True)) for n, d in runs.items()}
        L = float(_stat(lab[m], Lp[m], True))
        fams[FAM[f]] = dict(n=int(m.sum()), violating=float(viol[m].mean()), label=L, **vals)
        print(f"| {FAM[f]} | {m.sum()} | {viol[m].mean():.0%} | {L:.2f} | " + " | ".join(f"{vals[n]:.2f}" for n in runs)
              + f" | {L - vals[last]:.2f} |")
    out["family"] = fams
    print("\nlower tail (pooled over sides):")
    tail = {}
    for n, d in runs.items():
        ig = d["ig"]; bad = ig < -2
        tail[n] = dict(frac_ig_below_m2=float(bad.mean()), frac_below_prior=float((ig < 0).mean()),
                       bad_violating=float(viol[bad].mean()) if bad.any() else np.nan,
                       bad_no_large_facet=float((nbig[bad] == 0).mean()) if bad.any() else np.nan,
                       p05=float(np.percentile(ig, 5)), worst_mean_1pct=float(np.sort(ig)[: max(1, len(ig) // 100)].mean()))
        print(f"  {n}: IG<0 {tail[n]['frac_below_prior']:.1%}; IG<-2 {tail[n]['frac_ig_below_m2']:.1%} "
              f"(of which violating {tail[n]['bad_violating']:.0%}, no large facet {tail[n]['bad_no_large_facet']:.0%}); "
              f"5th pct {tail[n]['p05']:.2f}; worst-1% mean {tail[n]['worst_mean_1pct']:.2f}")
    out["tail"] = tail
    if len(runs) >= 2:
        a, b = list(runs)[-1], list(runs)[0]
        dlt = runs[a]["ig"] - runs[b]["ig"]
        out["paired_violating"] = boot(dlt[viol], scene[viol], Lp[viol])
        out["paired_obeying"] = boot(dlt[~viol], scene[~viol], Lp[~viol])
        print(f"\n{a} - {b}: law-obeying rows {out['paired_obeying']['mean']:.2f} [{out['paired_obeying']['lo']:.2f}, "
              f"{out['paired_obeying']['hi']:.2f}]; violating rows {out['paired_violating']['mean']:.2f} "
              f"[{out['paired_violating']['lo']:.2f}, {out['paired_violating']['hi']:.2f}]")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
