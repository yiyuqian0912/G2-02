"""Tables for docs/LOCALFIRST_E123.md from e12.jsonl, e2_clip.jsonl, e2_profile.jsonl and the audit (oracle bounds).

usage: e12_report.py --e12 DIR --clip DIR --audit DIR [--traces DIR] > tables.md
The decision rules are the ones declared in LOCALFIRST_AUDIT.md §5.2 before any E1/E2 measurement:
* information: median over tasks of Vol(R_I n B)/Vol(B) <= 0.1, AND the independent rule below random at the same |I|;
* computation (per pathway): certified speed-up >= half of the pathway's oracle bound (exact phase: piece oracle)."""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def pct(x, qs=(10, 50, 90)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return "—"
    return " / ".join(f"{np.percentile(x, q):.3g}" for q in qs) + f" (max {x.max():.3g}, n={len(x)})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--e12", required=True); ap.add_argument("--clip", required=True)
    ap.add_argument("--audit", required=True); a = ap.parse_args()
    R = load(Path(a.e12) / "e12.jsonl"); C = load(Path(a.clip) / "e2_clip.jsonl"); A = {r["key"]: r for r in load(Path(a.audit) / "audit.jsonl") if "key" in r}
    ok = [r for r in R if "E1" in r]; bad = [r for r in R if "E1" not in r]
    print(f"## E1/E2 tasks: {len(ok)} measured, {len(bad)} failed/timeout: {[(r['key'], r.get('status')) for r in bad]}\n")
    alarms = [(r["key"], x["mode"], x["n"], x["ALARM"]) for r in ok for x in r["E1"] if "ALARM" in x]
    print(f"**Safety alarms:** {len(alarms)} {alarms[:5]}")
    fs = sum(x.get("F_samples", 0) for r in ok for x in r["E1"]); fo = sum(x.get("F_outside_R", 0) for r in ok for x in r["E1"])
    fc = sum(x.get("F_outside_cover", 0) for r in ok for x in r["E1"]); ts = sum(not x.get("true_source_in_R", True) for r in ok for x in r["E1"])
    print(f"F samples tested (all configs): {fs}; outside R: {fo}; outside the cover: {fc}; true source outside R: {ts}\n")
    # ---- E1 distributions per configuration
    cfgs = [("independent", n) for n in (8, 32, 128, 512)] + [("random", n) for n in (8, 32, 128, 512)] + [("all", 2048)]
    print("| selection | |I| | Vol(R∩B)/Vol(B) p10 / p50 / p90 | Vol(B)/Vol(R∩B) p50 (FastLabel acceptance gain) | cover10 Vol(R)/Vol(domain) p50 | Vol(R∩roots)/Vol(roots) p50 | cells of R (D7) p50 | B cells p50 | components p50 | candidates/edge p50 | box tests p50 | build s p50 / max | budget stops |")
    print("|" + "---|" * 13)
    med = {}
    for mode, n in cfgs:
        xs = [(r, x) for r in ok for x in r["E1"] if x["mode"] == mode and x["n"] == n and not x.get("empty")]
        share = [x.get("B_share_in_R") for r, x in xs]
        gain = [1.0 / max(x["B_share_in_R"], 1.0 / x.get("B_samples", 20000)) for r, x in xs if x.get("B_share_in_R") is not None]
        dom = [x.get("cover10_volume_est", np.nan) / r["vol_domain"] for r, x in xs]
        rr = [x.get("cover10_volume_est", np.nan) * 0 + (x.get("vol_R_roots_strat") or 0) / r["vol_roots"] if r["vol_roots"] > 0 else np.nan for r, x in xs]
        med[(mode, n)] = np.median([v for v in share if v is not None]) if share else np.nan
        print(f"| {mode} | {n} | {pct(share)} | {np.median(gain) if gain else float('nan'):.3g} | {np.median(dom):.3g} | {np.median(rr):.3g} | "
              f"{np.median([x['cells'] for r, x in xs]):.0f} | {np.median([r['B_cells'] for r, x in xs]):.0f} | {np.median([x['components'] for r, x in xs]):.0f} | "
              f"{np.median([x['cand_p50'] for r, x in xs]):.0f} | {np.median([x['octree']['tests'] for r, x in xs]):.3g} | "
              f"{np.median([x['t_build'] for r, x in xs]):.2f} / {np.max([x['t_build'] for r, x in xs]):.1f} | {sum(x['octree'].get('budget_stops', 0) > 0 for r, x in xs)} |")
    print("\n**Declared information criterion** (median Vol(R∩B)/Vol(B) ≤ 0.1 and independent < random at equal |I|):")
    for n in (8, 32, 128, 512):
        i, rnd = med[("independent", n)], med[("random", n)]
        print(f"* |I| = {n}: independent {i:.3g}, random {rnd:.3g} → volume clause {'met' if i <= 0.1 else 'not met'}, "
              f"independence clause {'met' if i < rnd else 'not met'}")
    print(f"* all eligible (cap 2048): {med[('all', 2048)]:.3g}")
    # paired comparison independent vs random
    for n in (32, 128, 512):
        pr = []
        for r in ok:
            d = {(x["mode"], x["n"]): x for x in r["E1"]}
            if ("independent", n) in d and ("random", n) in d and d[("independent", n)].get("B_share_in_R") is not None:
                pr.append((d[("independent", n)]["B_share_in_R"], d[("random", n)]["B_share_in_R"]))
        if pr:
            pr = np.array(pr)
            print(f"* paired |I| = {n}: independent smaller in {int((pr[:, 0] < pr[:, 1]).sum())}, random smaller in {int((pr[:, 0] > pr[:, 1]).sum())}, ties {int((pr[:, 0] == pr[:, 1]).sum())}")
    # by family and L at |I| = 512 (independent) and all
    print("\n| family | L | tasks | Vol(R∩B)/Vol(B) p50, independent 512 | all (2048) | unusable (non-free) edge share p50 |")
    print("|---|---|---|---|---|---|")
    g = defaultdict(list)
    for r in ok:
        d = {(x["mode"], x["n"]): x for x in r["E1"]}
        g[(r["family"], r["L"])].append((d.get(("independent", 512), {}).get("B_share_in_R"), d.get(("all", 2048), {}).get("B_share_in_R"),
                                         1 - r["free_edges"] / max(r["edges"], 1)))
    for (f, L), v in sorted(g.items()):
        v = np.array([[np.nan if y is None else y for y in t] for t in v], float)
        print(f"| {f} | {L} | {len(v)} | {np.nanmedian(v[:, 0]):.3g} | {np.nanmedian(v[:, 1]):.3g} | {np.nanmedian(v[:, 2]):.3g} |")
    # ---- E2
    print("\n### E2 (box-level restriction of the exact phase to B ∩ R, depth-10 cover)\n")
    print("| region | tasks | exact-phase speed-up p10 / p50 / p90 | including construction | refine s p50 / max | mass share (must be 1) | oracle (piece) speed-up p50, same tasks |")
    print("|---|---|---|---|---|---|---|")
    for name in ("independent_128", "all_2048"):
        xs = [(r, x) for r in ok for x in r.get("E2", []) if x["region"] == name and not x.get("empty")]
        orc = [A[r["key"]]["B_local"]["t_exact"] / A[r["key"]]["B_local"]["oracle_pieces"]["t_exact"] for r, x in xs
               if r["key"] in A and A[r["key"]]["B_local"].get("oracle_pieces")]
        print(f"| {name} | {len(xs)} | {pct([x['speedup_exact'] for r, x in xs])} | {pct([x['speedup_with_construction'] for r, x in xs])} | "
              f"{np.median([x['t_refine'] for r, x in xs]):.2f} / {np.max([x['t_refine'] for r, x in xs]):.1f} | "
              f"{np.min([x['mass_share'] for r, x in xs]):.6f}–{np.max([x['mass_share'] for r, x in xs]):.6f} | {np.median(orc) if orc else float('nan'):.3g} |")
    cc = [c for c in C if "baseline" in c]
    print(f"\n### E2b (certified root clipping, polytope level): {len(cc)} exact-phase tasks; alarms {[c['key'] for c in cc if 'ALARM' in c]}\n")
    print("| variant | speed-up vs baseline p10 / p50 / p90 | incl. clip construction | nodes ratio p50 | evals ratio p50 | expansions ratio p50 | roots proven empty p50 | unique-explanation share p50 | max mass rel. err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, key in (("d4 + clips", ("d4", "clipped")), ("c7 without clips", ("c7", "unclipped")), ("c7 + clips", ("c7", "clipped"))):
        xs = [c for c in cc if key[1] in c.get(key[0], {})]
        if not xs:
            continue
        b = lambda c: c["baseline"]; x = lambda c: c[key[0]][key[1]]
        print(f"| {name} | {pct([b(c)['seconds'] / x(c)['seconds'] for c in xs])} | "
              f"{pct([b(c)['seconds'] / (x(c)['seconds'] + c[key[0]]['t_clip_build'] + c['t_constraints']) for c in xs])} | "
              f"{np.median([x(c)['stats']['nodes'] / max(b(c)['stats']['nodes'], 1) for c in xs]):.3g} | "
              f"{np.median([x(c)['stats']['entry_evals'] / max(b(c)['stats']['entry_evals'], 1) for c in xs]):.3g} | "
              f"{np.median([x(c)['stats']['expansions'] / max(b(c)['stats']['expansions'], 1) for c in xs]):.3g} | "
              f"{np.median([c[key[0]]['roots_empty'] / max(c[key[0]]['roots'], 1) for c in xs]):.3g} | "
              f"{np.median([c[key[0]]['unique_share'] for c in xs]):.3g} | {max(x(c)['mass_rel_err'] for c in xs):.1e} |")
    tot_b = sum(c["baseline"]["seconds"] for c in cc)
    for name, key in (("c7 + clips", ("c7", "clipped")),):
        xs = [c for c in cc if key[1] in c.get(key[0], {})]
        if xs:
            tot_x = sum(c[key[0]][key[1]]["seconds"] + c[key[0]]["t_clip_build"] + c["t_constraints"] for c in xs)
            print(f"\nSummed exact-phase wall time over {len(xs)} tasks: baseline {tot_b:.1f} s, {name} incl. construction {tot_x:.1f} s.")
    # ---- per-task table (E1 at independent 512 / all, E2)
    print("\n### Per-task (E1 share of B kept; E2 exact-phase speed-up)\n")
    print("| task | family | L | edges / free | world edges | Vol(F) | Vol(B) | B cells | R∩B/B ind-8 / 32 / 128 / 512 | rand-8 / 32 / 128 / 512 | all | R cells (all) | E2 ind-128 / all | E2b c7+clips |")
    print("|" + "---|" * 14)
    cmap = {c["key"]: c for c in cc}
    for r in sorted(ok, key=lambda r: (r["L"], r["family"], r["key"])):
        d = {(x["mode"], x["n"]): x for x in r["E1"]}
        sh = lambda m, n: "—" if (m, n) not in d or d[(m, n)].get("B_share_in_R") is None else f"{d[(m, n)]['B_share_in_R']:.2g}"
        e2 = {x["region"]: x for x in r.get("E2", [])}
        e2s = " / ".join(f"{e2[k]['speedup_exact']:.2f}" if k in e2 and "speedup_exact" in e2[k] else "—" for k in ("independent_128", "all_2048"))
        c = cmap.get(r["key"]); cl = f"{c['baseline']['seconds'] / c['c7']['clipped']['seconds']:.2f}" if c and "clipped" in c.get("c7", {}) else "—"
        print(f"| {r['key']} | {r['family'][:12]} | {r['L']} | {r['edges']} / {r['free_edges']} | {r['world_edges']} | {r.get('vol_F', float('nan')):.2g} | {r['vol_B']:.2g} | {r['B_cells']} | "
              f"{' / '.join(sh('independent', n) for n in (8, 32, 128, 512))} | {' / '.join(sh('random', n) for n in (8, 32, 128, 512))} | {sh('all', 2048)} | "
              f"{d.get(('all', 2048), {}).get('cells', '—')} | {e2s} | {cl} |")


if __name__ == "__main__":
    main()
