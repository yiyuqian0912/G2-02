"""Descriptive aggregates from completed pilots and exact private audits.

No measurements are manufactured for missing/partial experiments. Histograms
are exact; spatial coverage and nonlocal population estimates remain labelled
as estimates. This module does not alter any scientific artifact.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audit import OUT


def read(path):
    return json.loads(Path(path).read_text())


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(zip(("p50", "p95", "p99", "max"), map(float, [*np.percentile(a, [50, 95, 99]), a.max()])))


def pilot(name):
    root = OUT / name
    report = read(root / "report.json")
    if not report["complete"] or report["ready_scenes"] != report["requested_scenes"]:
        raise ValueError("Incomplete pilot cannot enter completed throughput tables")
    events = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    ready = sorted(e["event_unix_seconds"] for e in events if e["event"] == "READY_LITE" and "event_unix_seconds" in e)
    batches = []
    for width in (50, 100):
        for start in range(0, len(ready), width):
            end = min(start + width, len(ready))
            if end - start != width:
                continue
            # First batch includes cold start/preparation. Later batches use
            # the preceding READY timestamp, not the first item in the batch.
            beginning = next(e["event_unix_seconds"] for e in events if "event_unix_seconds" in e) if start == 0 else ready[start - 1]
            elapsed = ready[end - 1] - beginning
            batches.append({"start_ready": start + 1, "end_ready": end,
                            "seconds": elapsed, "ready_per_hour": width * 3600 / elapsed})
    gpu = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
    resources = [json.loads(line) for line in (root / "process-resources.jsonl").read_text().splitlines()]
    physical = defaultdict(list)
    work = defaultdict(Counter)
    for scene in report["scenes"]:
        for backend in scene["physical"]:
            physical[backend["stage"]].append(backend)
            for name, value in backend["work"].items():
                if name.startswith("max_"):
                    work[backend["stage"]][name] = max(work[backend["stage"]][name], value)
                else:
                    work[backend["stage"]][name] += value
    source_rejections = Counter()
    for event in events:
        if event["event"] == "source_accepted":
            for reject in event["rejections"]:
                source_rejections[str(reject.get("reason", reject.get("status", reject))) if isinstance(reject, dict) else str(reject)] += 1
    first = min(report["scenes"], key=lambda s: s["slot"])
    warm = [s for s in report["scenes"] if s["slot"] != first["slot"]]
    return {
        "complete": True, "ready": report["ready_scenes"],
        "unique_geometry_ids": len({s["geometry_id"] for s in report["scenes"]}),
        "unique_scene_ids": len({s["scene_id"] for s in report["scenes"]}),
        "source_geometry_pairs": len(report["scenes"]),
        "cold_cuda_setup_seconds": report["factory"]["cuda_cold_setup_seconds"],
        "cold_first_scene_source_setup_seconds": first["stages"]["source_sampling_and_renderer_setup"]["wall_seconds"],
        "cold_first_scene_field_seconds": first["stages"]["complete_field_and_index"]["wall_seconds"],
        "warm_source_setup_seconds": quantiles([s["stages"]["source_sampling_and_renderer_setup"]["wall_seconds"] for s in warm]),
        "warm_field_seconds": quantiles([s["stages"]["complete_field_and_index"]["wall_seconds"] for s in warm]),
        "kernel_seconds": quantiles([s["field"]["cuda_timings"]["kernel_seconds"] for s in report["scenes"]]),
        "transfer_seconds": quantiles([s["field"]["cuda_timings"]["H2D_seconds"] + s["field"]["cuda_timings"]["D2H_seconds"] for s in report["scenes"]]),
        "end_to_end_from_prepared_seconds": quantiles([s["latency_from_prepared_seconds"] for s in report["scenes"]]),
        "total_scene_latency_seconds": quantiles([s["prepare_wall_seconds"] + s["latency_from_prepared_seconds"] for s in report["scenes"]]),
        "verification": {k: {"wall_seconds": quantiles([r["wall_seconds"] for r in rows]),
                              "cpu_seconds": quantiles([r["cpu_seconds"] for r in rows]),
                              "unique_bricks_per_scene": quantiles([r["rendered_bricks"] for r in rows]),
                              "work_totals": dict(work[k])} for k, rows in physical.items()},
        "event_counts": dict(Counter(e["event"] for e in events)),
        "source_proposal_rejection_reasons": dict(source_rejections),
        "failures": report["failures"], "watchdogs": report.get("watchdogs", {}),
        "ready_batches": batches,
        "ready_timestamps_available": len(ready),
        "ready_interarrival_seconds": quantiles(np.diff(ready)),
        "gpu_utilization_percent": quantiles([r["utilization_percent"] for r in gpu if "utilization_percent" in r]),
        "gpu_utilization_percent_mean": float(np.mean([r["utilization_percent"] for r in gpu if "utilization_percent" in r])),
        "temperature_C": quantiles([r["temperature_C"] for r in gpu if "temperature_C" in r]),
        "cpu_cores_mean": float(np.mean([r["cpu_cores_used"] for r in resources])),
        "max_cuda_owned_bytes": max(s["field"]["renderer"]["device_allocated_peak_bytes"] for s in report["scenes"]),
        "candidate_histogram": dict(Counter({"1": 256 * report["ready_scenes"]})),
        "scientific_bytes": sum(s["scene_bytes"] for s in report["scenes"]),
        "root_total_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    }


def quality(path):
    path = Path(path)
    report = read(path / "distribution.json")
    scenes = report["scene_records"]
    by_family = {}
    for family in report["family_counts"]:
        rows = [s for s in scenes if s["family"] == family]
        by_family[family] = {
            "scenes": len(rows),
            "obstacles": quantiles([s["components"] for s in rows]),
            "actual_total_faces": quantiles([sum(s["actual_faces"]) for s in rows]),
            "solid_occupancy": quantiles([s["solid_occupancy"] for s in rows]),
            "shadow_free_occupancy": quantiles([s["shadow_occupancy"] for s in rows]),
        }
    pool_missing = Counter()
    fallback = Counter()
    rejections = Counter()
    proposals = accepted = controls = 0
    deficits = []
    for scene in scenes:
        deficits.append((scene["selection"]["dataset_balance"]["bins"], scene["selection"]["uniform_target_deficits"]))
        for side, r in scene["selection"]["by_side"].items():
            for b in r.get("unavailable_bins", []):
                pool_missing[(side, str(b))] += 1
            fallback.update(r.get("fallbacks", {}))
            d = r.get("discovery", {})
            proposals += d.get("proposed", 0)
            accepted += d.get("accepted", 0)
            controls += r.get("control", {}).get("accepted", 0)
            rejections.update(d.get("rejections", {}))
    return {"scenes": report["scenes"], "windows": report["windows"], "bins": report["bins"],
            "by_side": report["by_side"], "by_family": report["by_family"],
            "geometry_statistics": by_family,
            "uniform_pool_proposed": proposals, "uniform_pool_accepted": accepted,
            "uniform_pool_acceptance": accepted / proposals if proposals else None,
            "independent_control_windows": controls,
            "first_rejections": dict(rejections), "selection_fallbacks": dict(fallback),
            "unavailable_pool_cells": [{"side": s, "ratio_bin": b, "scene_count": n} for (s, b), n in sorted(pool_missing.items())],
            "final_cumulative_uniform_target_deficits": max(deficits, key=lambda v: sum(v[0]))[1] if deficits else None,
            "deficit_semantics": "Uniform-target cumulative shortage; mixed mode intentionally retains natural/extreme components. A missing pool bin is not proof no physical window exists.",
            "mean_pairwise_iou": float(np.mean([s["pairwise_iou_mean"] for s in scenes])),
            "mean_max_other_iou": float(np.mean([s["mean_max_other_iou"] for s in scenes]))}


def nonlocal_joint(counts, audit):
    import pyarrow.parquet as pq
    population = pq.read_table(counts).to_pylist()
    audit = read(audit)
    def key(row):
        return row["family"], row["side"], row["ratio_bin"]
    sizes = Counter(key(r) for r in population)
    sampled = Counter(key(r) for r in audit["records"])
    result = []
    for b in range(10):
        rows = [r for r in audit["records"] if r["ratio_bin"] == b]
        total = sum(n for (_, _, rb), n in sizes.items() if rb == b)
        values = {field: sum(sizes[key(r)] / sampled[key(r)] * r[field] for r in rows) / total
                  for field in ("weighted_external_first", "weighted_external_necessary")} if total else {}
        result.append({"ratio_bin": b, "population_windows": total, "sampled_windows": len(rows), **values})
    return {"by_ratio_bin": result, "population_estimates": audit.get("population_estimates"),
            "window_stratum_selection": audit.get("window_stratum_selection", "legacy"),
            "sampled_scenes": len({r["scene_id"] for r in audit["records"]}),
            "sampled_slot_range": [min(r["slot"] for r in audit["records"]),max(r["slot"] for r in audit["records"])],
            "scope": "GMP-certified point results; deterministically stratified descriptive population estimates, not exhaustive causal fractions or IID confidence claims."}


def inverse_joint():
    study = read(OUT / "inverse-v4-cold/summary.json")
    nonlocal_rows = read(OUT / "inverse-nonlocal/nonlocal-audit.json")["records"]
    lookup = {(r["scene_id"], r["window_id"]): r for r in nonlocal_rows}
    records = []
    for row in study["records"]:
        window = row["window"]
        sample = lookup[window["scene_id"], window["window_id"]]
        records.append({"scene_id": window["scene_id"], "window_id": window["window_id"],
            "family": window["family"], "side": window["side"], "lit_ratio": window["lit_ratio"],
            "external_necessary_estimate": sample["weighted_external_necessary"],
            "all_source_outer_volume_upper": row["ambiguity"]["all_source_outer"]["outer_cover_volume_upper_bound"],
            "separated_source_outer_volume_upper": row["ambiguity"]["separated_source_outer"]["outer_cover_volume_upper_bound"],
            "outer_evaluated_regions": row["ambiguity"]["all_source_outer"]["evaluated_regions"],
            "outer_termination": row["ambiguity"]["all_source_outer"]["termination"]})
    from scipy.stats import spearmanr
    correlations = {}
    for key in ("lit_ratio", "external_necessary_estimate"):
        correlations[key + "_vs_outer_upper_spearman"] = float(spearmanr(
            [r[key] for r in records], [r["all_source_outer_volume_upper"] for r in records]).statistic)
    return {"windows": len(records), "records": records, "descriptive_correlations": correlations,
            "caveat": "Purposively selected 27 frozen cases. Bounds use budget-limited sparse constraints, not exact feasible volumes; solver effort/family/size are confounders. No IID significance, causal trend or uniqueness claim."}


def run():
    result = {"pilots": {}, "quality": {}, "nonlocal": {}}
    for name in ("core-matched-100", "core-fast-mixed-100", "core-final-smoke-10", "core-final-100", "core-fast-mixed-300"):
        p = OUT / name / "report.json"
        if p.exists() and read(p)["complete"]:
            result["pilots"][name] = pilot(name)
    for name in ("fast-100-quality-audit", "final-100-quality-audit", "sustained-300-quality-audit"):
        path = OUT / name
        if (path / "distribution.json").exists():
            result["quality"][name] = quality(path)
            if (path / "nonlocal/nonlocal-audit.json").exists():
                result["nonlocal"][name] = nonlocal_joint(path / "window-counts.parquet", path / "nonlocal/nonlocal-audit.json")
    for name in ("stratified-nonlocal", "mixed-nonlocal", "natural-nonlocal-current", "sustained-300-nonlocal-hashed"):
        path = OUT / name / "nonlocal-audit.json"
        if path.exists():
            audit = read(path)
            result["nonlocal"][name] = nonlocal_joint(audit["count_records"], path)
    from rind3d.pack import atomic_json
    result["inverse_geometry_joint"] = inverse_joint()
    atomic_json(OUT / "extended-results.json", result)


if __name__ == "__main__":
    run()
