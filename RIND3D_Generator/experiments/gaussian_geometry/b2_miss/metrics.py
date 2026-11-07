"""Aggregate frozen comparison records and audit the hot-ray directions."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ..b2_angular.preservation import sha
from ..study import atomic_json
from .audit import quantiles
from .bench import OUT, read


def hot_angles(record, archive):
    """Angles between adjacent expensive rays, for diagnosis only."""
    if sha(archive) != record["archive_SHA256"]:
        raise ValueError("Trace checksum mismatch")
    with np.load(archive) as data:
        mask = ((data["evaluation_path"] == 6) &
                (data["ray_nanoseconds"] >= record["clustering"]["threshold_nanoseconds"]))
    source = np.array([float.fromhex(x) for x in record["geometry"]["source_hex"]])
    origin = np.asarray(record["origin"])
    angles = []
    for axis in range(3):
        first, second = [slice(None)] * 3, [slice(None)] * 3
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        ijk = np.argwhere(mask[tuple(first)] & mask[tuple(second)])[:, ::-1]
        v = (ijk + origin + .5) / record["grid"] - source
        w = v.copy()
        w[:, 2-axis] += 1 / record["grid"]
        angles.extend(np.degrees(np.arctan2(np.linalg.norm(np.cross(v, w), axis=1),
                                            (v*w).sum(axis=1))))
    if len(angles) != record["clustering"]["hot_face_pairs"]:
        raise ValueError("Different adjacency definition")
    return quantiles(angles)


def group(records):
    return {
        "tiles": len(records),
        "valid_voxels": sum(r["valid_voxels"] for r in records),
        "tile_wall_seconds": sum(r["tile_call_seconds"] for r in records),
        "tile_CPU_seconds": sum(r["tile_CPU_seconds"] for r in records),
        "original_MISS_skipped": sum(r.get("saved_scalar", {}).get("count", 0) for r in records),
        "original_MISS_seconds_skipped": sum(r.get("saved_scalar", {}).get(
            "original_MISS_nanoseconds", 0) for r in records) / 1e9,
        "original_top_quartile_skipped": sum(r.get("saved_scalar", {}).get(
            "original_top_quartile_count", 0) for r in records),
        "region_seconds": sum(r["region_seconds"] for r in records),
        "local_construction_seconds": sum(r["work"].get("miss_local_construction_ns", 0)
                                          for r in records) / 1e9,
        "scalar_exp_endpoints": sum(r["work"].get("precision_low_exp_endpoints", 0) +
                                    r["work"].get("precision_high_exp_endpoints", 0)
                                    for r in records),
        "local_exp_endpoints": sum(r["work"].get("miss_local_exp_endpoints", 0) for r in records),
    }


def summarize(out):
    plan = read(out / "tile-plan.json")
    expected = [r["id"] for r in plan["tiles"]]
    methods = {m: read(out / path / "summary.json") for m, path in
               (("F", "baseline"), ("G", "G"), ("H", "H"), ("I", "I"))}
    for method in methods.values():
        if ([r["id"] for r in method["records"]] != expected or
                not all(r["complete"] for r in method["records"])):
            raise ValueError("All frozen tiles are required")
    baseline = {r["id"]: r for r in methods["F"]["records"]}
    hot = baseline["14-05"]
    matched = {}
    for name, method in methods.items():
        records = method["records"]
        if any(r["physical_bits_SHA256"] != baseline[r["id"]]["physical_bits_SHA256"]
               for r in records):
            raise ValueError("Different physical bits")
        matched[name] = {
            **group(records),
            "cold_end_to_end_wall_seconds": method["elapsed_wall_seconds"],
            "cold_end_to_end_CPU_seconds": method["CPU_seconds"],
            "cold_setup_seconds": sum(r["seconds"] for r in method["setups"]),
            "peak_RSS_MiB": method["peak_RSS_MiB"],
            "original16": group([r for r in records if r["case"] == "gaussian_sum-014"]),
            "heldout8": group([r for r in records if r["case"] != "gaussian_sum-014"]),
            "evaluation21": group([r for r in records if r["split"] == "evaluation"]),
        }
    local = read(out / "local-development.json")
    old_vs_local = [(r["old_volume"], r) for row in local["rows"] for probe in row["probes"]
                    for r in probe["directions"] if "old_volume" in r]
    global_audits = {}
    for split in ("development", "evaluation"):
        audit = read(out / f"global-{split}.json")
        counts = Counter()
        for row in audit["rows"]:
            counts.update(row["coverage_counts"])
        global_audits[split] = dict(counts)
    result = {
        "plan_SHA256": sha(out / "tile-plan.json"),
        "builds": {name: sha(out / folder / "identity.json") for name, folder in
                   (("F", "baseline"), ("G", "G"), ("H", "H"), ("I", "I"))},
        "matched": matched,
        "global_stencil_coverage_by_split": global_audits,
        "hot_neighbor_angle_degrees": hot_angles(hot, out / "baseline" / "14-05.npz"),
        "old_volume_control": {
            "attempts": len(old_vs_local),
            "old_FREE": sum(a["state"][0] == 1 for a, _ in old_vs_local),
            "local_FREE": sum(b["certified"] for _, b in old_vs_local),
            "both_FREE": sum(a["state"][0] == 1 and b["certified"] for a, b in old_vs_local),
            "old_seconds": quantiles([a["seconds"] for a, _ in old_vs_local]),
            "local_seconds": quantiles([b["seconds"] for _, b in old_vs_local]),
            "old_total_seconds": sum(a["seconds"] for a, _ in old_vs_local),
            "local_total_seconds": sum(b["seconds"] for _, b in old_vs_local),
        },
    }
    gate = out / "admission/admission.json"
    if gate.is_file():
        result["admission"] = read(gate)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=OUT)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    atomic_json(args.output, summarize(args.run))
