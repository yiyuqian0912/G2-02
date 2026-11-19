"""Summary tables of the FastLabel v3.4 validation (FASTLABEL_V34_CERTIFICATION.md §7). Reads results/*.jsonl written by
compare_refs.py and run_fl.py, the cmp_v3 FastLabel run log, and the saved labels; writes results/SUMMARY.json.

Adds three checks that need the labels themselves:
* serialization: the loader's float q = b / sum(b) (rind3d_100k.loader.Scene.posterior) against the exact
  normalisation of the stored float64 volumes b: exact |sum q - 1| and exact TV(q_float, q_exact), against the a-priori
  bound gamma_(n+1) / 2 (n = number of cells, u = 2^-53);
* verifier/sampler consistency: on fresh production runs with a tight reference (t_ref < 1e-3), the validation stream's
  failure count (n = 20,000 points of U_B, exact verifier) against Binomial(n, 1 - p) for every p in the reference
  bracket of |F| / |B| (two-sided exact binomial p-value at the most favourable end);
* certificate rules: rind3d_100k.validate.check_stat_certificate on every fresh certificate.

usage: python summarize.py"""
import glob
import json
import os
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
WORK = "/home/yushu/Projects/RIND/RIND3D_100K_v3_work"


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def serialization(path):
    with np.load(path) as z:
        cell, b = z["cell"], np.asarray(z["lo"], np.float64)
    u, inv = np.unique(cell, return_inverse=True)
    b = np.bincount(inv, weights=b)                                     # per-cell, as labels.py stores them
    q = b / b.sum()                                                     # the loader's float q
    B = [Q(float(x)) for x in b]; S = sum(B, Q(0))
    Qf = [Q(float(x)) for x in q]
    tv = sum((abs(x - y / S) for x, y in zip(Qf, B)), Q(0)) / 2
    n = len(b); g = (n + 1) * 2.0 ** -53 / (1 - (n + 1) * 2.0 ** -53)         # gamma_(n+1), §6.3
    return dict(n=n, sum_minus_1=float(sum(Qf, Q(0)) - 1), tv_float_vs_exact=float(tv), bound=g / 2, ok=bool(tv <= Q(g / 2)))


def binom_two_sided(k, n, th):
    from scipy.stats import binom
    lo = binom.cdf(k, n, th); hi = binom.sf(k - 1, n, th)
    return float(min(1.0, 2 * min(lo, hi)))


def main():
    sys.path.insert(0, str(HERE.parent.parent.parent / "RIND3D_Dataset100K"))
    from rind3d_100k.validate import check_stat_certificate
    contract = json.loads((HERE.parent.parent.parent / "RIND3D_Dataset100K" / "contract" / "RIND3D_100K_CONTRACT_v3.json").read_text())
    fl_contract = contract["labels"]["fastlabel"]
    comp = load(RES / "compare_cmp_v3.jsonl") + load(RES / "compare_astra_fresh.jsonl")
    runs_cmp = {r["key"]: r for r in load(f"{WORK}/cmp_v3/fl.jsonl")}
    runs_new = {r["key"]: r for r in load(RES / "fresh_astra" / "fl_runs.jsonl")}
    out = dict(tasks={}, by_ref={}, violations=[])
    for r in comp:
        k = r["key"]
        run = runs_cmp.get(k) or runs_new.get(k)
        path = run["meta"]["path"] if "meta" in run else run["path"]
        t = out["tasks"].setdefault(k, dict(path=path, set="cmp_v3" if k in runs_cmp and k not in runs_new else "astra_fixture",
                                            refs={}))
        t["refs"][r["ref"]] = {x: r[x] for x in ("tv_upper", "tv_lower", "t_ref", "one_minus_p", "containment_excess_rel_total",
                                                   "containment_excess_rel_max_cell", "mass_outside_ref_support")}
        cls = r["ref"].split(":")[0]
        a = out["by_ref"].setdefault(cls, dict(n=0, n_stat=0, max_tv_upper=0.0, max_tv_upper_stat_key=None, max_excess_rel=0.0,
                                               n_over_eps=0, max_one_minus_p_upper=0.0))
        a["n"] += 1
        if path == "stat_certified":
            a["n_stat"] += 1
            if r["tv_upper"] > a["max_tv_upper"]:
                a["max_tv_upper"] = r["tv_upper"]; a["max_tv_upper_stat_key"] = k
            a["max_one_minus_p_upper"] = max(a["max_one_minus_p_upper"], r["one_minus_p"][1])
        a["max_excess_rel"] = max(a["max_excess_rel"], r["containment_excess_rel_total"])
        if r["tv_lower"] > 0.05:
            a["n_over_eps"] += 1; out["violations"].append(dict(key=k, ref=r["ref"], tv_lower=r["tv_lower"]))
    # tightest rigorous (independent) reference per task: tp_cert or astra
    best = {}
    for k, t in out["tasks"].items():
        ind = {n: v for n, v in t["refs"].items() if not n.startswith("tp_engine")}
        pool = ind or t["refs"]
        n = min(pool, key=lambda x: pool[x]["tv_upper"])
        best[k] = dict(ref=n, independent=bool(ind), path=t["path"], **pool[n])
    out["best_reference"] = best
    stat_best = [v for v in best.values() if v["path"] == "stat_certified"]
    out["stat_labels"] = dict(n=len(stat_best), n_independent=sum(v["independent"] for v in stat_best),
                              max_tv_upper=max(v["tv_upper"] for v in stat_best),
                              median_tv_upper=float(np.median([v["tv_upper"] for v in stat_best])),
                              max_tv_upper_independent=max(v["tv_upper"] for v in stat_best if v["independent"]),
                              max_one_minus_p_upper_tight=max(v["one_minus_p"][1] for v in stat_best if v["t_ref"] < 1e-3))
    # serialization
    ser = {}
    for k in out["tasks"]:
        p = (RES / "fresh_astra" / f"{k}_fl.npz") if k in runs_new else Path(f"{WORK}/cmp_v3/labels/{k}_fl.npz")
        ser[k] = serialization(p)
    out["serialization"] = dict(max_abs_sum_minus_1=max(abs(v["sum_minus_1"]) for v in ser.values()),
                                max_tv_float_vs_exact=max(v["tv_float_vs_exact"] for v in ser.values()),
                                all_within_bound=all(v["ok"] for v in ser.values()),
                                n_exactly_normalised=sum(v["sum_minus_1"] == 0 for v in ser.values()), n=len(ser))
    # verifier / sampler consistency on fresh runs
    cons = []
    for k, run in runs_new.items():
        v = run.get("validation")
        if not v:
            continue
        refs = out["tasks"][k]["refs"]
        tight = [x for x in refs.values() if x["t_ref"] < 1e-3]
        if not tight:
            continue
        x = min(tight, key=lambda y: y["t_ref"])
        lo, hi = max(x["one_minus_p"][0], 0.0), max(x["one_minus_p"][1], 0.0)
        pv = max(binom_two_sided(v["fail"], v["n"], th) for th in np.linspace(lo, hi, 21) if th > 0) if hi > 0 else \
            (1.0 if v["fail"] == 0 else 0.0)
        cons.append(dict(key=k, n=v["n"], fail=v["fail"], one_minus_p_exact=[lo, hi], p_value=pv))
    out["validation_consistency"] = cons
    # certificate rules on fresh production certificates
    chk = {}
    for k, run in runs_new.items():
        if run["status"] != "certified_stat":
            continue
        slot, w, s = int(k[:6]), int(k[8:10]), int(k[12:])
        with np.load(RES / "fresh_astra" / f"{k}_fl.npz") as z:
            lo, hi = z["lo"], z["hi"]
        m = dict(key=k, slot=slot, window=w, source=s, status=run["status"], tv_bound=run["tv_bound"],
                 engine_version=run["version"], certificate=run["certificate"])
        chk[k] = check_stat_certificate(m, lo, hi, fl_contract)
    out["certificate_rules"] = dict(n=len(chk), failures={k: v for k, v in chk.items() if v})
    # sample cost (rounds and points) of the statistical labels at the production budget
    rr = []
    for k, run in list(runs_cmp.items()) + list(runs_new.items()):
        rounds = run["meta"]["rounds"] if "meta" in run else run["rounds"]
        p = run["meta"]["path"] if "meta" in run else run["path"]
        if p == "stat_certified":
            rr.append(dict(key=k, rounds=len(rounds), tested=sum(int(x.get("tested", 0)) for x in rounds),
                           last_tested=int(rounds[-1].get("tested", 0))))
    out["cost"] = dict(n=len(rr), rounds_hist={str(n): sum(r["rounds"] == n for r in rr) for n in sorted({r["rounds"] for r in rr})},
                       mean_tested=float(np.mean([r["tested"] for r in rr])), max_tested=max(r["tested"] for r in rr),
                       stage1_fraction=float(np.mean([r["last_tested"] == 512 for r in rr])))
    json.dump(out, open(RES / "SUMMARY.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("tasks", "best_reference")}, indent=1, default=float))


if __name__ == "__main__":
    main()
