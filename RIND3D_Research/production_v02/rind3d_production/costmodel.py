"""Lifecycle cost model: T_total = T_generation + T_mesh_or_BVH + N_queries x T_query + T_validation_and_IO.

All constants are MEASURED on this machine (canary SEAL records, validation, benchmarks); every projection to
1K / 10K / 100K scenes is an EXTRAPOLATION (linear in scene count at the measured per-scene medians, with
p95 bands), labelled as such. Writes RIND3D_LIFECYCLE_COST_MODEL.json.
"""
import json
from pathlib import Path

import numpy as np

from .env import ROOT

CANARY = ROOT/"outputs"/"canary"


def q(xs):
    a = np.asarray([x for x in xs if x is not None], float)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.quantile(a, .95)), "max": float(a.max()), "n": int(len(a))}


def canary_records():
    out = []
    for p in CANARY.glob("shards/*/scenes/*/SEAL.json"):
        out.append(json.loads(p.read_text()))
    return out


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
    bench = json.loads((ROOT/"RIND3D_BACKEND_BENCHMARKS.json").read_text())
    recs = canary_records()
    val = json.loads((CANARY/"validation.json").read_text()) if (CANARY/"validation.json").exists() else None
    journal = [json.loads(l) for l in (CANARY/"journal.jsonl").read_text().splitlines()]
    runs = [e for e in journal if e["event"] == "run_finished"]
    by = {}
    for r in recs:
        key = "SB" if r["profile"] == "SB" else f"MA-K{r['K']}"
        by.setdefault(key, []).append(r)
    per = {}
    for k, rows in sorted(by.items()):
        per[k] = {"scenes": len(rows), "wall_seconds": q([r["wall_seconds"] for r in rows]), "cpu_seconds": q([r["cpu_seconds"] for r in rows]),
                  "bytes": q([r["bytes"] for r in rows]), "peak_RSS_MiB_worker": q([r["peak_RSS_MiB"] for r in rows]),
                  "stages_median_seconds": {s: float(np.median([r["stages"][s]["wall_seconds"] for r in rows])) for s in rows[0]["stages"]},
                  "windows": q([r["windows"] for r in rows])}
    sessions = session_walls(journal)
    wall = sum(s["wall_seconds"] for s in sessions)
    n = len(recs)
    mix = {k: v["scenes"]/n for k, v in per.items()}
    mean_wall = sum(mix[k]*per[k]["wall_seconds"]["mean"] for k in per)
    mean_bytes = sum(mix[k]*per[k]["bytes"]["mean"] for k in per)
    rate = n/(wall/3600) if wall else None
    val_s = (val["summary"]["validation_seconds"]/val["summary"]["scenes"]) if val else None
    windows_per_scene = float(np.mean([r["windows"] for r in recs]))
    proj = {}
    for N in (1_000, 10_000, 100_000):
        hours = N/rate if rate else None
        proj[str(N)] = {"status": "EXTRAPOLATED (linear at the measured canary rate and mix; one RTX 3060 laptop producer)",
                        "producer_hours": hours, "producer_days": hours/24 if hours else None,
                        "validation_cpu_hours_single_process": N*val_s/3600 if val_s else None,
                        "storage_GB": N*mean_bytes/1e9, "files": N*float(np.mean([len(r["files"]) for r in recs])),
                        "windows": N*windows_per_scene,
                        "with_4_gpu_workers_hours_estimate": hours/4 if hours else None}
    t = bench["materialization"]["windows"]
    teacher = bench["near_geometry_windows"]["A_certified_points"]
    cross = {}
    for L in ("64", "128", "256"):
        read = t[L]["A_read_warm_seconds"]["median"]
        render = t[L]["C_render_seconds"]["median"]
        cross[L] = {"A_read_warm_seconds": read, "A_read_cold_seconds": t[L]["A_read_cold_seconds"]["median"], "C_lazy_render_seconds": render,
                    "B_read_seconds": t[L]["B_read_seconds"]["median"], "B_bytes_per_window": t[L]["B_stored_bytes"]["median"],
                    "lazy_over_read_ratio": render/read}
    field_s = bench["full_1024_fields"]["cuda_certified_seconds"]["median"]
    field_bytes = bench["full_1024_fields"]["response_bytes"]["median"] + bench["full_1024_fields"]["solid_bytes"]["median"]
    model = {
        "schema": "rind3d-lifecycle-cost-model-v1", "equation": "T_total = T_scene_generation + T_mesh_or_BVH_build + N_queries x T_query + T_validation_and_IO",
        "measured": {
            "canary_scenes": n, "canary_sessions": sessions, "canary_wall_seconds_all_sessions": wall, "canary_scenes_per_hour": rate,
            "per_profile": per, "validation_seconds_per_scene": val_s, "windows_per_scene": windows_per_scene,
            "full_field_seconds_per_source": field_s, "full_field_bytes_per_source": field_bytes,
            "mesh_or_BVH_build": {"csg_direct_triangulation_seconds": bench["random_windows"]["mesh_direct_triangulation"]["seconds"]["median"],
                                  "csg_embree_build_seconds": bench["random_windows"]["E_embree_float32"]["bvh_build_seconds"]["median"],
                                  "csg_native_context_seconds": bench["random_windows"]["context_setup_seconds"]["median"],
                                  "gaussian_mesh_stage_seconds": bench["gaussian_mesh_reference"].get("mesh_stage_seconds_median")},
            "window_access": cross,
            "teacher_query": {"certified_points_pairs_per_second_16_threads": teacher["teacher_32x4096_pairs_per_second_16_threads"]["median"],
                              "certified_points_pairs_per_second_1_thread": teacher["rays_per_second_single_thread"]["median"],
                              "cuda_full_window_64_seconds_per_candidate": bench["materialization"]["teacher_full_window_64"]["cuda_seconds_per_candidate"]["median"],
                              "warp_screen_rays_per_second_not_certified": bench["near_geometry_windows"]["W_warp_gpu"]["rays_per_second"]["median"]}},
        "derived": {
            "student_epoch_read_hours_per_10M_windows_L64": 10e6*t["64"]["A_read_warm_seconds"]["median"]/3600,
            "student_epoch_read_hours_per_10M_windows_L256": 10e6*t["256"]["A_read_warm_seconds"]["median"]/3600,
            "materialization_crossover": ("For CSG the certified CUDA renderer produces a window about as fast as the Python RINDPack reader "
                                          "decodes it (ratios above), so compute never forces materialization; storing complete fields costs "
                                          f"{field_bytes/1e6:.2f} MB and {field_s:.2f} s per source once, and buys GPU-free, producer-free, "
                                          "verifiable reading. Full fields are therefore kept as the canonical lossless observation (A), "
                                          "windows are index records (no duplication), hypothetical-source Teacher queries are always lazy (C), "
                                          "and multi-source public observations are materialized per window (B) because their public form must "
                                          "not expose private channels."),
            "teacher_example": {"description": "1 epoch x 1M windows x 64 candidates x 4096 receivers (certified CPU, 16 threads)",
                                "hours": 1e6*64*4096/teacher["teacher_32x4096_pairs_per_second_16_threads"]["median"]/3600,
                                "cuda_full_window_alternative_hours": 1e6*64*bench["materialization"]["teacher_full_window_64"]["cuda_seconds_per_candidate"]["median"]/3600}},
        "projections": proj,
        "risks": {"yield": "measured canary yield applies to the certified CSG core only; Gaussian/mesh families have different yields (devcal: 26/28)",
                  "throughput_variance": "per-scene wall p95/median per profile above; multi-source cost grows ~linearly in K",
                  "storage": "dominated by multi-source public observation payloads and per-source private fields; duplicate per-source SOLID fields (~0.3 MB/source) are removable",
                  "single_producer": "one laptop GPU; a restart/crash costs at most the in-flight scene (measured recovery)"},
        "labels": {"measured": "everything under 'measured'", "extrapolated": "everything under 'projections' and 'derived'",
                   "hypothesis": "linear scaling to 100K (no large-campaign effects such as disk fragmentation, thermal throttling or catalogue growth were measured)"}}
    (ROOT/"RIND3D_LIFECYCLE_COST_MODEL.json").write_text(json.dumps(model, indent=1, default=float)+"\n")
    print(json.dumps({"rate": rate, "per": {k: (v["scenes"], round(v["wall_seconds"]["median"], 1), round(v["bytes"]["median"]/1e6, 2)) for k, v in per.items()},
                      "proj": {k: (round(v["producer_hours"], 1), round(v["storage_GB"], 1)) for k, v in proj.items()}}, indent=1))


if __name__ == "__main__":
    main()
