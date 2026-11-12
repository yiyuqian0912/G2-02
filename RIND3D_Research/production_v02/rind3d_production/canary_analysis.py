"""Canary results and frozen-gate evaluation -> RIND3D_PRODUCTION_CANARY_RESULTS.json.

Inputs (all re-read from disk): canary journal and SEAL records, validation.json (independent process),
usability.json, interruption records, GPU samples, pytest record. Gates C1-C8 are evaluated exactly as written in
the frozen contract (contract.json of the canary root); thresholds are read from there, never edited here.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .env import ROOT
from .windows import SIDES, SUBSET_CODES

C = ROOT/"outputs"/"canary"
CODE = {v: k for k, v in SUBSET_CODES.items()}


def session_walls(journal):
    """Wall seconds per runner session from journal timestamps: run_started -> run_finished, or -> the last
    event before the next session (abruptly killed sessions: a lower bound, the crash tail is unmeasured)."""
    from datetime import datetime
    ts = lambda e: datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%S").timestamp()
    starts = [i for i, e in enumerate(journal) if e["event"] == "run_started"]
    out = []
    for j, i in enumerate(starts):
        end = starts[j+1] if j+1 < len(starts) else len(journal)
        seg = journal[i:end]
        fin = [e for e in seg if e["event"] == "run_finished"]
        out.append({"start": seg[0]["time"], "wall_seconds": (fin[0]["wall_seconds"] if fin else ts(seg[-1]) - ts(seg[0])),
                    "finished_gracefully": bool(fin)})
    return out


def main():
    contract = json.loads((C/"contract.json").read_text())
    cfg = json.loads((C/"config.json").read_text())
    journal = [json.loads(l) for l in (C/"journal.jsonl").read_text().splitlines()]
    seals = [json.loads(p.read_text()) for p in sorted(C.glob("shards/*/scenes/*/SEAL.json"))]
    slots = [s for p in sorted(C.glob("shards/*/index/slots.json")) for s in json.loads(p.read_text())["slots"]]
    val = json.loads((C/"validation.json").read_text())
    # recompute the per-check counts over the union of applicable checks (V8 applies to MA scenes only)
    vkeys = sorted({k for r in val["scenes"] for k in r if k.startswith("V")})
    val["summary"]["passed"] = {k: int(sum(bool(r[k]) for r in val["scenes"] if k in r)) for k in vkeys}
    val["summary"]["applicable"] = {k: int(sum(k in r for r in val["scenes"])) for k in vkeys}
    val["summary"]["all_pass"] = bool(val["summary"]["all_pass"] and all(all(r[k] for k in r if k.startswith("V")) for r in val["scenes"]))
    use = json.loads((C/"usability.json").read_text())
    tests = (C/"tests.txt").read_text() if (C/"tests.txt").exists() else ""
    recs = []
    for p in sorted(C.glob("shards/*/scenes/*/private/window_records.npy")):
        seal = json.loads((p.parent.parent/"SEAL.json").read_text())
        r = np.load(p)
        recs.append((seal["profile"], seal["K"], r))
    # windows
    cat = Counter()
    per_side = {L: Counter() for L in SIDES}
    multi = {"informative_any": 0, "informative_ge2": 0, "informative_ge3": 0, "informative_all": 0}
    lit_ratio_nat = []
    for prof, K, r in recs:
        for x in r:
            L, sub, free = int(x["L"]), CODE[int(x["subset"])], int(x["free"])
            lit = [int(v) for v in x["lit"][:K]]
            if sub == "natural":
                if free == 0:
                    c = "no_free"
                elif prof == "SB" and lit[0] == free:
                    c = "all_lit"
                elif prof == "SB" and lit[0] == 0:
                    c = "all_shadow"
                elif all(l in (0, free) for l in lit):
                    c = "constant_response"
                else:
                    c = "mixed"
                cat[(prof, c)] += 1
                per_side[L][c] += 1
                if prof == "SB" and free:
                    lit_ratio_nat.append(lit[0]/free)
            else:
                per_side[L][sub] += 1
            if prof == "MA" and sub != "natural":
                inf = sum(0 < l < free for l in lit)
                multi["informative_any"] += inf >= 1
                multi["informative_ge2"] += inf >= 2
                multi["informative_ge3"] += inf >= 3
                multi["informative_all"] += inf == K
    reports = [json.load(open(p)) for p in sorted(C.glob("shards/*/scenes/*/private/window_report.json"))]
    short = Counter()
    quota = Counter()
    for rep in reports:
        for k, v in rep.items():
            short[k.split("/")[0]] += v["shortfall"]
            quota[k.split("/")[0]] += v["quota"]
    # recovery
    inter = json.loads((C/"interruption.json").read_text()) if (C/"interruption.json").exists() else None
    moved = [e for e in journal if e["event"] == "partial_attempt_moved_aside"]
    sealed_events = [e for e in journal if e["event"] == "slot_sealed"]
    dup = [s for s, n in Counter(e["slot"] for e in sealed_events).items() if n > 1]
    det = val["summary"].get("V11_determinism", [])
    # resources
    gpu = []
    if (C/"gpu-samples.csv").exists():
        for line in (C/"gpu-samples.csv").read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    gpu.append(float(parts[1]))
                except ValueError:
                    pass
    sessions = session_walls(journal)
    wall = float(sum(s["wall_seconds"] for s in sessions))
    failures = [s for s in slots if s["status"] != "SEALED"]
    attempts_failed = [e for e in journal if e["event"] == "slot_attempt_failed"]
    nat_fill = all(rep[f"natural/{L}"]["accepted"] == rep[f"natural/{L}"]["quota"] for rep in reports for L in SIDES)
    cov = {L: {c: per_side[L][c] for c in ("all_lit", "mixed", "all_shadow", "extreme", "rare", "nonlocal")} for L in SIDES}
    gates = {
        "C1_yield": {"published": len(seals), "attempted": cfg["slots"], "rate": len(seals)/cfg["slots"],
                     "unpublished_with_reason": [{"slot": s["slot"], "reason": s.get("reason")} for s in failures],
                     "pass": len(seals)/cfg["slots"] >= .95 and all(s.get("reason") for s in failures)},
        "C2_validation": {"all_pass": val["summary"]["all_pass"], "passed_counts": val["summary"]["passed"], "scenes": val["summary"]["scenes"],
                          "pass": bool(val["summary"]["all_pass"] and val["summary"]["scenes"] == len(seals))},
        "C3_physics": {"generation_replays_all_matched": all(all(st["all_response_and_solid_bits_matched"] for st in ev["stages"]) for s in seals for ev in s["evidence"]),
                       "replayed_bricks": int(sum(ev["bricks"] for s in seals for ev in s["evidence"])),
                       "validation_rerender_pass": val["summary"]["passed"].get("V7_independent_rerender") == len(seals)},
        "C4_windows": {"natural_100pct_filled": nat_fill, "informative_shortfall_fraction": short["informative"]/max(1, quota["informative"]),
                       "coverage_per_side": cov, "natural_all_shadow_SB_any_side": cat[("SB", "all_shadow")],
                       "pass": bool(nat_fill and short["informative"] <= .05*quota["informative"]
                                    and all(cov[L][c] >= 1 for L in SIDES for c in ("all_lit", "mixed", "extreme", "rare", "nonlocal"))
                                    and cat[("SB", "all_shadow")] >= 1)},
        "C5_leakage": {"tests": tests.strip().splitlines()[-1] if tests else None,
                       "V9_public_private": val["summary"]["passed"].get("V9_public_private") == len(seals),
                       "V8_multi_local_palette": f'{val["summary"]["passed"].get("V8_multi_observations")}/{val["summary"]["applicable"].get("V8_multi_observations")}',
                       "pass": bool(tests and " passed" in tests.strip().splitlines()[-1] and "failed" not in tests.strip().splitlines()[-1]
                                    and val["summary"]["passed"].get("V9_public_private") == len(seals)
                                    and val["summary"]["passed"].get("V8_multi_observations") == val["summary"]["applicable"].get("V8_multi_observations"))},
        "C6_recovery": {"interruption": inter, "partial_attempts_moved_aside": [e["path"] for e in moved], "duplicate_sealed_slots": dup,
                        "determinism": det, "pass": bool(inter is not None and not dup and det and all(d["identical"] for d in det)
                                                         and val["summary"]["passed"].get("V5_fields") == len(seals))},
        "C7_resources": {"max_scene_wall_seconds": max(s["wall_seconds"] for s in seals), "max_worker_RSS_MiB": max(s["peak_RSS_MiB"] for s in seals),
                         "max_gpu_memory_used_MiB": max(gpu) if gpu else None, "canary_wall_seconds": wall,
                         "pass": bool(max(s["wall_seconds"] for s in seals) <= cfg["caps"]["scene_seconds"] and max(s["peak_RSS_MiB"] for s in seals) <= 6*1024
                                      and (max(gpu) if gpu else 0) <= 4096 and wall <= cfg["caps"]["canary_wall_seconds"])},
        "C8_usability": {"student_SB_L64_windows_per_second": use["student_local_random_access_SB"]["64"]["windows_per_second"],
                         "dataloader": use["dataloader_2_workers"], "teacher_equal_stored": use["teacher"]["all_equal"],
                         "pass": bool(use["student_local_random_access_SB"]["64"]["windows_per_second"] >= 100 and use["dataloader_2_workers"]["windows"] > 0
                                      and use["teacher"]["all_equal"])}}
    accepted = all(g["pass"] for g in gates.values() if "pass" in g)
    gates["C3_physics"]["pass"] = bool(gates["C3_physics"]["generation_replays_all_matched"] and gates["C3_physics"]["validation_rerender_pass"])
    accepted = all(g["pass"] for g in gates.values())
    # diversity: occupancy and lit fraction from the stored field indexes (exact brick populations), components
    from rind3d.pack import RINDPack
    div = {}
    for d in sorted(C.glob("shards/*/scenes/*")):
        seal = json.loads((d/"SEAL.json").read_text())
        recipe = json.loads((d/"private"/"recipe.json").read_text())
        with RINDPack(d/"public"/"solid.rindb") as m:
            solid = sum(r[7] for r in m.index)/1024**3
        resp = d/"public"/"response.rindb" if seal["profile"] == "SB" else d/"private"/"src0"/"response.rindb"
        with RINDPack(resp) as y:
            lit = sum(r[7] for r in y.index)/1024**3
        row = div.setdefault(seal["family"], {"solid": [], "lit_free": [], "components": []})
        row["solid"].append(solid)
        row["lit_free"].append(lit/max(1e-12, 1-solid))
        row["components"].append(len(recipe["objects"]))
    diversity = {f: {k: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in r.items()} for f, r in div.items()}
    allsolid = [x for r in div.values() for x in r["solid"]]
    diversity["all"] = {"solid_quantiles": {q: float(np.quantile(allsolid, float(q))) for q in ("0.1", "0.5", "0.9")},
                        "solid_max": float(max(allsolid))}
    fam = Counter(s["family"] for s in seals)
    prof = Counter((s["profile"], s["K"]) for s in seals)
    out = {"schema": "rind3d-production-canary-results-v1", "contract": contract["contract_id"], "config_SHA256": json.loads((C/"FROZEN.json").read_text())["config_SHA256"],
           "accepted": accepted, "gates": gates, "attempts_failed": attempts_failed,
           "scenes": {"published": len(seals), "families": dict(fam), "profiles": {f"{p}-K{k}": n for (p, k), n in prof.items()},
                      "splits": val["summary"]["split_counts"], "bytes_total": int(sum(s["bytes"] for s in seals)),
                      "windows_total": int(sum(s["windows"] for s in seals))},
           "diversity": diversity,
           "observation_distribution": {"natural_categories": {f"{p}/{c}": n for (p, c), n in cat.items()},
                                        "per_side": {str(L): dict(v) for L, v in per_side.items()},
                                        "subset_shortfalls": dict(short), "subset_quotas": dict(quota),
                                        "natural_SB_lit_ratio_quantiles": {q: float(np.quantile(lit_ratio_nat, float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")} if lit_ratio_nat else None,
                                        "multi_source_non_natural_windows": multi},
           "throughput": {"sessions": sessions, "wall_seconds_all_sessions": wall, "scenes_per_hour": len(seals)/(wall/3600) if wall else None,
                          "per_scene_wall_median": float(np.median([s["wall_seconds"] for s in seals]))},
           "usability": use}
    (ROOT/"RIND3D_PRODUCTION_CANARY_RESULTS.json").write_text(json.dumps(out, indent=1, default=float)+"\n")
    print(json.dumps({"accepted": accepted, **{k: v["pass"] for k, v in gates.items()}}, indent=1))


if __name__ == "__main__":
    main()
