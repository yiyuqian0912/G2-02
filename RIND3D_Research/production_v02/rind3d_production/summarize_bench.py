"""Compile RIND3D_BACKEND_BENCHMARKS.json from the raw Stage B records (+ Gaussian-mesh reference costs)."""
import json
from pathlib import Path

import numpy as np

from .env import ROOT, sha256_file

B = ROOT/"outputs"/"benchmarks"
GAUSS = Path("/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Generator/outputs/gaussian-mesh-devcal-v1")


def q(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"median": float(np.median(a)), "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9)),
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def scenes(path):
    return json.loads(path.read_text())["scenes"]


def backend_block(rows):
    return {
        "context_setup_seconds": q([r["setup"]["context_seconds"] for r in rows]),
        "geometry_proof_seconds": q([r["setup"]["geometry_proof_seconds"] for r in rows]),
        "membership_262k_seconds": q([r["setup"]["membership_seconds_262k"] for r in rows]),
        "A_certified_points": {
            "source_bind_seconds": q([r["setup"]["source_bind_seconds"] for r in rows]),
            "dense_262k_first_seconds": q([r["A_dense"]["first_seconds"] for r in rows]),
            "dense_262k_repeat_seconds": q([r["A_dense"]["repeat_seconds"] for r in rows]),
            "rays_per_second_single_thread": q([r["A_dense"]["rays_per_second_single_thread"] for r in rows]),
            "s4096_seconds": q([r["A_s4096"]["repeat_seconds"] for r in rows]),
            "s256_seconds": q([r["A_s256"]["repeat_seconds"] for r in rows]),
            "teacher_32x4096_pairs_per_second_16_threads": q([r["A_teacher_batch"]["pairs_per_second_parallel"] for r in rows]),
            "teacher_pairs_per_second_single_thread_incl_bind": q([r["A_teacher_batch"]["pairs_per_second_sequential"] for r in rows])},
        "B_independent_projection": {"dense_3_sources_seconds": q([r["B_dense_3_sources"]["seconds"] for r in rows]),
                                     "mismatches_vs_A": int(sum(r["B_dense_3_sources"]["mismatches_vs_A"] for r in rows)),
                                     "unknown_A": int(sum(r["B_dense_3_sources"]["unknown_A"] for r in rows))},
        "R_certified_cpu_raster_64": {"seconds": q([r["R_window"]["seconds"] for r in rows]),
                                      "mismatch_vs_A": int(sum(r["R_window"]["mismatch_vs_A_FREE"] for r in rows)),
                                      "solid_mismatch": int(sum(r["R_window"]["solid_mismatch"] for r in rows))},
        "G_certified_cuda_raster_64": {"source_bind_seconds": q([r["G_window"].get("source_bind_seconds") for r in rows]),
                                       "first_seconds": q([r["G_window"].get("first_seconds") for r in rows]),
                                       "repeat_seconds": q([r["G_window"].get("repeat_seconds") for r in rows]),
                                       "batch8_seconds": q([r["G_window"].get("batch8_seconds") for r in rows]),
                                       "mismatch_vs_A": int(sum(r["G_window"].get("mismatch_vs_A_FREE", 0) for r in rows)),
                                       "errors": [r["G_window"]["error"] for r in rows if "error" in r["G_window"]]},
        "mesh_direct_triangulation": {"seconds": q([r["mesh"]["triangulation_seconds"] for r in rows]),
                                      "triangles": q([r["mesh"]["triangles"] for r in rows]),
                                      "max_vertex_plane_residual": q([r["mesh"]["max_vertex_plane_residual"] for r in rows])},
        "E_embree_float32": {"bvh_build_seconds": q([r["E"]["bvh_build_seconds"] for r in rows]),
                             "dense_first_seconds": q([r["E"]["first_seconds"] for r in rows]),
                             "dense_repeat_seconds": q([r["E"]["repeat_seconds"] for r in rows]),
                             "rays_per_second": q([r["E"]["rays_per_second"] for r in rows]),
                             "mismatch_vs_A": int(sum(r["E"]["mismatch_vs_A"] for r in rows)),
                             "teacher_32x4096_pairs_per_second": q([r["E_teacher_batch"]["pairs_per_second"] for r in rows]),
                             "teacher_mismatch_vs_A_first4": int(sum(r["E_teacher_batch"]["mismatch_vs_A_first4_sources"] for r in rows))},
        "W_warp_gpu": {"bvh_build_seconds": q([r["W"]["bvh_build_seconds"] for r in rows]),
                       "dense_repeat_seconds": q([r["W"]["repeat_seconds"] for r in rows]),
                       "rays_per_second": q([r["W"]["rays_per_second"] for r in rows]),
                       "mismatch_vs_A": int(sum(r["W"]["mismatch_vs_A"] for r in rows))},
        "H_hybrid_certified_arbitration": {"extra_seconds_after_E_W": q([r["H"]["seconds_after_E_W"] for r in rows]),
                                           "flagged": int(sum(r["H"]["flagged"] for r in rows)),
                                           "mismatch_vs_A_after_arbitration": int(sum(r["H"]["mismatch_vs_A_after_arbitration"] for r in rows)),
                                           "mismatch_vs_A_unflagged": int(sum(r["H"]["mismatch_vs_A_unflagged"] for r in rows))},
        "scenes": len(rows), "components": q([r["components"] for r in rows]), "planes": q([r["planes"] for r in rows]),
        "free_receivers_per_window": q([r["free_receivers"] for r in rows])}


def main():
    raw = json.loads((B/"backends-raw.json").read_text())
    near = scenes(B/"backends-near-raw.json")
    graz = json.loads((B/"grazing-csg-mesh.json").read_text())
    mat = scenes(B/"materialization-raw.json")
    fields = raw["full_fields"]
    mat_block = {}
    for L in (64, 128, 256):
        mat_block[str(L)] = {k: q([r[f"L{L}"][k] for r in mat]) for k in ("A_read_cold_seconds", "A_read_warm_seconds", "C_render_seconds",
                                                                         "B_stored_bytes", "B_read_seconds")}
        mat_block[str(L)]["render_equals_stored_field"] = all(r[f"L{L}"]["render_equals_stored_field"] for r in mat)
    gauss = {}
    if (GAUSS/"summary.json").exists():
        s1 = json.loads((GAUSS/"summary.json").read_text())["stage1"]["policy_set"]
        hs = json.loads((GAUSS/"heldout-summary.json").read_text())
        gauss = {"source": "outputs/gaussian-mesh-devcal-v1 (measured in the previous study; same machine)",
                 "mesh_stage_seconds_median": {"pilot_D": s1["A"]["D"]["median_mesh_seconds"], "additive_D": s1["B"]["D"]["median_mesh_seconds"]},
                 "faces_median": {"pilot_D": s1["A"]["D"]["median_faces"], "additive_D": s1["B"]["D"]["median_faces"]},
                 "max_peak_RSS_MiB": {"pilot": s1["A"]["D"]["max_peak_RSS_MiB"], "additive": s1["B"]["D"]["max_peak_RSS_MiB"]},
                 "heldout_pilot_256_full_pipeline_seconds_median": hs["cost"]["seconds"]["median"],
                 "heldout_256_labels_seconds_median": hs["cost"]["labels_seconds"]["median"],
                 "full_1024_field_seconds": [851, 949],
                 "label_semantics": "float32 mesh is the truth by definition of that family; Embree + Warp + translation probes + exact triangle oracle"}
    out = {"schema": "rind3d-backend-benchmarks-v1", "machine": "RTX 3060 Laptop (6 GiB), 16 logical CPUs, 31 GiB RAM",
           "benchmark_seed": raw["seed"], "status": "measured (single machine, bounded samples; medians with p10/p90)",
           "random_windows": backend_block(raw["scenes"]), "near_geometry_windows": backend_block(near),
           "full_1024_fields": {"cuda_certified_seconds": q([f["seconds"] for f in fields if f["backend"] == "cuda_certified"]),
                                "cpu_native_certified_seconds_4_workers": [f["seconds"] for f in fields if f["backend"] == "cpu_native_certified"],
                                "response_bytes": q([f["field_bytes"]["response"] for f in fields if f["backend"] == "cuda_certified"]),
                                "solid_bytes": q([f["field_bytes"]["solid"] for f in fields if f["backend"] == "cuda_certified"]),
                                "information_index_bytes": q([f["information_index_bytes"] for f in fields if f["backend"] == "cuda_certified"])},
           "grazing_csg_mesh": graz["totals"] | {"by_gap": graz["by_gap"]},
           "materialization": {"windows": mat_block,
                               "count_index_build_seconds": q([r["count_index_build_seconds"] for r in mat]),
                               "count_index_bytes": q([r["count_index_bytes"] for r in mat]),
                               "count_query_seconds": q([r["count_query_seconds"] for r in mat]),
                               "teacher_full_window_64": {"cuda_seconds_per_candidate": q([r["teacher_full_window_64"]["cuda_seconds_per_candidate"] for r in mat]),
                                                          "native_points_seconds_per_candidate_single_thread": q([r["teacher_full_window_64"]["native_points_seconds_per_candidate_single_thread"] for r in mat])}},
           "gaussian_mesh_reference": gauss,
           "raw_files": {p.name: sha256_file(p) for p in sorted(B.glob("*.json"))}}
    (ROOT/"RIND3D_BACKEND_BENCHMARKS.json").write_text(json.dumps(out, indent=1, default=float)+"\n")
    print(json.dumps({k: out[k] for k in ("full_1024_fields", "grazing_csg_mesh")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
