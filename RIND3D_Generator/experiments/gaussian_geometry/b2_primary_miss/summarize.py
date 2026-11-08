"""Read the matched native traces and aggregate measured scalar work."""
import argparse
from pathlib import Path

from ..b2_angular.preservation import sha
from ..study import atomic_json
from .bench import MISS_OUT, OUT, PLAN, read


def tile_row(row, base, primary):
    w = row["work"]
    counts = row.get("accounting", {})
    saved = row.get("saved_scalar", {})
    skipped = counts.get("actual_MISS_skipped", saved.get("count", 0))
    original_ns = counts.get("original_MISS_ns_avoided", saved.get("original_MISS_nanoseconds", 0))
    expensive = counts.get("top_quartile_skipped", saved.get("original_top_quartile_count", 0))
    exp = (w.get("exp_endpoints", 0) + w.get("high_precision_exp_endpoints", 0)
           if primary else w.get("precision_low_exp_endpoints", 0) + w.get("precision_high_exp_endpoints", 0))
    if row["physical_bits_SHA256"] != base["physical_bits_SHA256"]:
        raise ValueError("Physical digest mismatch")
    return {
        "id": row["id"], "case": row["case"], "split": row["split"], "origin": row["origin"],
        "valid_voxels": row["valid_voxels"], "wall_seconds": row["tile_call_seconds"],
        "CPU_seconds": row["tile_CPU_seconds"], "region_seconds": row["region_seconds"],
        "original_MISS_rays": base["trace"]["MISS"], "scalar_MISS_rays": base["trace"]["MISS"] - skipped,
        "scalar_HIT_rays": row["trace"]["HIT"],
        "actual_MISS_skipped": skipped, "original_MISS_seconds_avoided": original_ns / 1e9,
        "top_quartile_skipped": expensive,
        "counted_exp_endpoints": exp,
        "extension_exp_endpoints": w.get("primary_local_exp_endpoints", w.get("miss_local_exp_endpoints", 0)),
        "long_success": w.get("axial_long_success", 0), "long_failure": w.get("axial_long_failure", 0),
        "short_success": w.get("axial_short_success", 0), "short_failure": w.get("axial_short_failure", 0),
        "long_dispatch_suppressed": w.get("axial_long_dispatch_suppressed", 0),
        "cache_setup_seconds": w.get("miss_cache_setup_ns", 0)/1e9,
        "cache_lookup_seconds": w.get("miss_cache_lookup_ns", 0)/1e9,
        "cache_index_seconds": w.get("miss_index_ns", 0)/1e9,
        "dispatch_seconds": w.get("miss_dispatch_ns", 0)/1e9,
        "extension_seconds": sum(w.get(f"axial_{d}_{s}_ns", 0)
                                 for d in ("long", "short") for s in ("success", "failure")) / 1e9
                             or w.get("miss_local_construction_ns", 0)/1e9,
        "physical_bits_SHA256": row["physical_bits_SHA256"],
    }


def summarize(out, plan):
    identifiers = [r["id"] for r in read(plan)["tiles"]]
    folders = {m: out / m for m in ("P", "P_H", "P_I", "P_J", "F_J")}
    folders.update({m: MISS_OUT / subdir for m, subdir in
                    (("F", "baseline"), ("F_H", "H"), ("F_I", "I"))})
    runs = {m: read(folder/"summary.json") for m, folder in folders.items()}
    primary = runs["P"]["records"]
    independent = runs["F"]["records"]
    if any(a["physical_bits_SHA256"] != b["physical_bits_SHA256"]
           for a, b in zip(primary, independent, strict=True)):
        raise ValueError("Baseline primary/independent disagreement")
    results = {}
    for method, trial in runs.items():
        if [r["id"] for r in trial["records"]] != identifiers:
            raise ValueError(f"Incomplete workload for {method}")
        rows = []
        for row, base in zip(trial["records"], primary if method.startswith("P") else independent, strict=True):
            if not row["complete"]:
                raise ValueError(f"Incomplete physical tile for {method}")
            item = tile_row(row, base, method.startswith("P"))
            if method in ("F_H", "F_I"):
                label = "short" if method == "F_H" else "long"
                item[label + "_success"] = row["work"].get("miss_local_certificates", 0)
                item[label + "_failure"] = row["work"].get("miss_local_failures", 0)
            rows.append(item)
        results[method] = {
            "cold_wall_seconds": trial["elapsed_wall_seconds"], "cold_CPU_seconds": trial["CPU_seconds"],
            "peak_RSS_MiB": trial["peak_RSS_MiB"],
            "setup_seconds": sum(r["seconds"] for r in trial["setups"]),
            "tile_wall_seconds": sum(r["wall_seconds"] for r in rows),
            "evaluation_tile_wall_seconds": sum(r["wall_seconds"] for r in rows if r["split"] == "evaluation"),
            "heldout_tile_wall_seconds": sum(r["wall_seconds"] for r in rows if r["case"] != "gaussian_sum-014"),
            "actual_MISS_skipped": sum(r["actual_MISS_skipped"] for r in rows),
            "original_MISS_seconds_avoided": sum(r["original_MISS_seconds_avoided"] for r in rows),
            "top_quartile_skipped": sum(r["top_quartile_skipped"] for r in rows),
            "rows": rows, "identity_SHA256": sha(folders[method]/"identity.json"),
        }
    return {"plan_SHA256": sha(plan), "matched": results,
            "original_probe": read(MISS_OUT/"admission/admission.json"),
            "new_probe": read(out/"admission/admission.json") if (out/"admission/admission.json").exists() else None}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=OUT)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    atomic_json(args.output, summarize(args.run, args.plan))
