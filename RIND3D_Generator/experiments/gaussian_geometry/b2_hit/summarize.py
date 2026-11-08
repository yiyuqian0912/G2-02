"""Aggregate matched HIT timings and integrity-checked admission evidence."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_miss.bench import read
from ..study import atomic_json
from .freeze import OUT, PLAN, PRIMARY

POLICIES = ("P_J","P_JH1","P_JH4","P_JHA")


def tile_record(row, archive):
    w, a = row["work"], row["accounting"]
    if sha(archive) != row["archive_SHA256"]:
        raise ValueError("Measured trace checksum differs")
    with np.load(archive) as f:
        remaining = (f["evaluation_path"] == 6) & (f["propagated_seed"] < 0)
        original_miss = int(f["ray_nanoseconds"][remaining].sum())
    return {
        "id": row["id"], "case": row["case"], "origin": row["origin"], "split": row["split"],
        "valid_voxels": row["valid_voxels"], "wall_seconds": row["tile_call_seconds"],
        "CPU_seconds": row["tile_CPU_seconds"], "region_seconds": row["region_seconds"],
        "original_scalar_HIT": row["trace"]["HIT"],
        "actual_HIT_skipped": a["actual_HIT_skipped"], "actual_MISS_skipped": a["actual_MISS_skipped"],
        "original_HIT_seconds_avoided": a.get("original_HIT_ns_avoided",0)/1e9,
        "original_top_quartile_HIT_skipped": a.get("original_top_quartile_HIT_skipped",0),
        "original_top_quartile_HIT_seconds_avoided": a.get("original_top_quartile_HIT_ns_avoided",0)/1e9,
        "remaining_scalar_MISS_seconds": original_miss/1e9,
        "remaining_scalar_HIT_seconds": a["remaining_scalar_HIT_ns"]/1e9,
        "HIT_return_seconds": a["HIT_cache_path_ns"]/1e9,
        "HIT_marks": w.get("hit_cache_new_marks",0),
        "HIT_successful_tests": w.get("hit_short_success",0)+w.get("hit_long_success",0),
        "HIT_failed_tests": w.get("hit_short_failure",0)+w.get("hit_long_failure",0),
        "HIT_extension_seconds": w.get("hit_extension_ns",0)/1e9,
        "HIT_failed_extension_seconds": w.get("hit_failed_extension_ns",0)/1e9,
        "HIT_dispatch_seconds": w.get("hit_dispatch_ns",0)/1e9,
        "HIT_setup_seconds": w.get("hit_cache_setup_ns",0)/1e9,
        "HIT_index_seconds": w.get("hit_index_ns",0)/1e9,
        "cache_lookup_seconds": w.get("miss_cache_lookup_ns",0)/1e9,
        "original_path_exp_endpoints": w.get("exp_endpoints",0)+w.get("high_precision_exp_endpoints",0),
        "MISS_extension_exp_endpoints": w.get("primary_local_exp_endpoints",0),
        "HIT_extension_exp_endpoints": w.get("hit_extension_exp_endpoints",0),
        "physical_bits_SHA256": row["physical_bits_SHA256"],
    }


def trial(path, reference, ids):
    run = read(path/"summary.json")
    if [r["id"] for r in run["records"]] != ids:
        raise ValueError("Measured trial does not match the complete planned workload")
    expected = {r["id"]:r for r in reference["records"]}
    rows = []
    for r in run["records"]:
        if (not r["complete"] or r["physical_bits_SHA256"] != expected[r["id"]]["physical_bits_SHA256"]
                or r["geometry"] != expected[r["id"]]["geometry"]):
            raise ValueError("Incomplete field or F_J mismatch")
        rows.append(tile_record(r,path/(r["id"]+".npz")))
    sums = {k:sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k],(int,float))}
    return {"cold_wall_seconds":run["elapsed_wall_seconds"],"cold_CPU_seconds":run["CPU_seconds"],
            "primary_plus_independent_cold_seconds":run["elapsed_wall_seconds"]+reference["elapsed_wall_seconds"],
            "peak_RSS_MiB":run["peak_RSS_MiB"],"setup_seconds":sum(r["seconds"] for r in run["setups"]),
            "identity_SHA256":sha(path/"identity.json"),"totals":sums,"rows":rows}


def summarize(root):
    old = read(PRIMARY/"F_J/summary.json")
    new = read(root/"F_J-heldout/summary.json")
    ids = [r["id"] for r in read(PLAN)["tiles"]]
    heldout = [r["id"] for r in read(root/"heldout-plan.json")["tiles"]]
    hotspots = ["14-05","14-13","14-14"]
    info = read(root/"witness-audit.json")
    samples = [s for r in info["rows"] for s in r["samples"]]
    return {
        "plans": {"original":sha(PLAN),"heldout":sha(root/"heldout-plan.json"),
                  "audit":sha(root/"audit-plan.json")},
        "audit": {"rays":len(samples),"strict":sum(r["margin"]>0 for r in samples),
                  "kinds":dict(Counter(s["kind"] for s in samples)),
                  "paired_capture_minus_control_seconds":sum(s["capture_seconds"]-s["scalar_seconds"] for s in samples),
                  "original_trace_rows": [{k:v for k,v in r.items() if k!="samples"} for r in info["rows"]]},
        "original": {m:trial(root/m,old,ids) for m in POLICIES},
        "heldout": {m:trial(root/(m+"-heldout"),new,heldout) for m in POLICIES},
        "hotspot_controls": {m:trial(root/m,old,hotspots) for m in ("P_JHC","P_JHG4")},
        "independent_cold_wall_seconds": {"original":old["elapsed_wall_seconds"],
                                          "heldout":new["elapsed_wall_seconds"]},
        "Arb":dict(Counter(r["status"] for r in read(root/"arb-controls.json")["checks"])),
        "admission":read(root/"admission/admission.json"),
        "coverage":read(root/"admission/coverage.json"),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    atomic_json(a.output,summarize(a.run))
