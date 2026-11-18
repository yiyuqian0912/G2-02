"""Tables for docs/LOCALFIRST_COMPARISON.md from compare.py report rows.

usage: summarize.py DIR [DIR2 ...]   (each DIR holds report_rows.json from `compare.py report`)"""
import json, sys
from pathlib import Path

import numpy as np


def q(x):
    x = [v for v in x if v is not None]
    return "—" if not x else f"{np.median(x):.2f} / {np.percentile(x, 95):.2f} / {max(x):.2f}"


def main():
    for d in sys.argv[1:]:
        rows = json.loads((Path(d) / "report_rows.json").read_text())
        print(f"\n## {d}  ({len(rows)} tasks)\n")
        print("| method | certificate | certified (TV ≤ 0.05) | wall s p50 / p95 / max | peak RSS above baseline MB p50 / max | TV to tp p50 / max |")
        print("|---|---|---|---|---|---|")
        for m, cert in (("tp", "deterministic"), ("lf", "deterministic"), ("rand", "deterministic"), ("fl", "statistical (δ = 0.01 / 4.8M)")):
            R = [r[m] for r in rows if m in r]
            if not R:
                continue
            ok = sum(1 for r in R if r.get("status") in ("certified", "certified_stat"))
            tv = [r.get("tv_to_tp") for r in R if r.get("tv_to_tp") is not None]
            rss = [r.get("rss") for r in R if r.get("rss") is not None]
            print(f"| {m} | {cert} | {ok}/{len(R)} | {q([r.get('wall') for r in R])} | "
                  f"{(f'{np.median(rss):.0f} / {max(rss):.0f}') if rss else '—'} | {(f'{np.median(tv):.4f} / {max(tv):.4f}') if tv else '—'} |")
        print("\n| L | tasks | tp wall p50 | lf wall p50 | rand wall p50 | fl wall p50 | lf speed-up over tp (p50 of ratios) |")
        print("|---|---|---|---|---|---|---|")
        for L in sorted({r["L"] for r in rows}):
            g = [r for r in rows if r["L"] == L]
            med = lambda m: np.median([r[m]["wall"] for r in g if m in r and r[m].get("wall")]) if any(m in r and r[m].get("wall") for r in g) else float("nan")
            ratio = [r["tp"]["wall"] / r["lf"]["wall"] for r in g if "tp" in r and "lf" in r and r["tp"].get("wall") and r["lf"].get("wall")
                     and r["tp"].get("status") not in ("timeout", "error")]
            print(f"| {L} | {len(g)} | {med('tp'):.1f} | {med('lf'):.1f} | {med('rand'):.1f} | {med('fl'):.1f} | "
                  f"{(np.median(ratio) if ratio else float('nan')):.1f}× |")
        B = [(r["lf"].get("B_cells"), r["rand"].get("B_cells")) for r in rows if "lf" in r and "rand" in r and r["lf"].get("B_cells") and r["rand"].get("B_cells")]
        if B:
            rr = np.array([a / b for a, b in B])
            print(f"\nOuter bound B cells, local-first / random selection: median {np.median(rr):.2f}, "
                  f"local smaller in {int((rr < 1).sum())}/{len(rr)}, larger in {int((rr > 1).sum())}/{len(rr)}")
        loc = [r["local"] for r in rows if "local" in r]
        if loc:
            safe = sum(1 for x in loc if x["band_mass"] > 0.999 and x["source_in_band"])
            miss = sum(1 for x in loc if not x["source_in_band"])
            print(f"\nHeuristic local band (X only, never used for pruning): all label mass and the true source inside in "
                  f"{safe}/{len(loc)} tasks; true source outside in {miss}/{len(loc)}; median band/support cells "
                  f"{np.median([x['band_cells'] / max(x['support_cells'], 1) for x in loc]):.0f}×; median label mass inside "
                  f"{np.median([x['band_mass'] for x in loc]):.4f}, min {min(x['band_mass'] for x in loc):.4f}")
            ranks = [x["cond"]["rank"] for x in loc]
            print("planar rank of the local components (count of tasks by rank):", {k: ranks.count(k) for k in sorted(set(ranks))})


if __name__ == "__main__":
    main()
