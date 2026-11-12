"""RIND3D_TRAINING_SCALE_COST.json: generation, indexing, export, storage and training-epoch costs for
1K / 10K / 100K scenes x 102 / 256 windows, for the three meanings of "100K scenes".

All inputs are measured on this laptop (RTX 3060 Laptop, 16 CPU threads) and are cited by source file:
  physics generation   canary SEAL.json wall seconds per profile (v0.2 runner, one GPU worker)
  validation           canary validation.json seconds per scene (one process)
  index v0.3           outputs/privileged/index/*/P256/report.json seconds per scene (12 workers share the CPU)
  export v0.3          outputs/privileged/export/*/export_summary.json
  storage              actual bytes of canary scene directories, index files and student shards
  training             outputs/bench/student_bench_*.json (B4 loader-fed end-to-end windows/s per L)
Projections [X] are linear in scenes; they assume the canary family/profile mix and throughput.
"""
import json
from pathlib import Path

import numpy as np

from .env import CANARY, OUT, ROOT

SCALES = (1_000, 10_000, 100_000)
WINDOWS = {"P102": {64: 34, 128: 34, 256: 34}, "P256": {64: 86, 128: 85, 256: 85}}


def canary_costs():
    per = {"SB": [], "MA": []}
    size = {"SB": [], "MA": []}
    for s in CANARY.glob("shards/*/scenes/*/SEAL.json"):
        r = json.loads(s.read_text())
        per[r["profile"]].append(r["wall_seconds"])
        d = s.parent
        size[r["profile"]].append(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()))
    val = json.loads((CANARY/"validation.json").read_text())
    vs = {"SB": [], "MA": []}
    for r in val["scenes"]:
        vs[r["profile"]].append(r["seconds"])
    return {k: {"generation_wall_s_mean": float(np.mean(per[k])), "generation_wall_s_median": float(np.median(per[k])),
                "scene_bytes_mean": float(np.mean(size[k])), "validation_s_mean": float(np.mean(vs[k]))} for k in per}


def index_costs(release):
    d = OUT/"privileged"/"index"/release/"P256"
    if not (d/"summary.json").exists():
        return None
    s = json.loads((d/"summary.json").read_text())
    rows = {"SB": [], "MA": []}
    by = {"SB": [], "MA": []}
    for sd in sorted(p for p in d.iterdir() if p.is_dir()):
        m = json.loads((sd/"report.json").read_text())
        rows[m["data_profile"]].append(m["seconds"])
        by[m["data_profile"]].append(sum(p.stat().st_size for p in sd.iterdir()))
    return {"wall_seconds": s["wall_seconds"], "workers": s["workers"], "scenes": s["scenes"],
            "scenes_per_hour_12_workers": s["scenes"]/s["wall_seconds"]*3600,
            **{f"{k}_scene_cpu_s_mean": float(np.mean(v)) for k, v in rows.items() if v},
            **{f"{k}_index_bytes_mean": float(np.mean(v)) for k, v in by.items() if v}}


def export_costs(name):
    p = OUT/"privileged"/"export"/name
    if not (p/"export_summary.json").exists():
        return None
    s = json.loads((p/"export_summary.json").read_text())
    join = np.load(p/"join.npy")
    sizes = {}
    for f in (OUT/"student"/name).rglob("*.idx.npy"):
        idx = np.load(f)
        for mod, nm in ((0, "SB"), (1, "MA")):
            for L in (64, 128, 256):
                m = (idx["modality"] == mod) & (idx["L"] == L)
                if m.any():
                    b = (idx["resp_bytes"][m].astype(np.int64) + idx["solid_bytes"][m]).sum() + m.sum()*idx.dtype.itemsize
                    k = f"{nm}/L{L}"
                    sizes.setdefault(k, [0, 0])
                    sizes[k][0] += int(b)
                    sizes[k][1] += int(m.sum())
    per_window = {k: v[0]/v[1] for k, v in sizes.items()}
    scenes = len(set(join["scene"].tolist()))
    return {"wall_seconds": s["wall_seconds"], "workers": s["workers"], "scenes": scenes,
            "scenes_per_hour": scenes/s["wall_seconds"]*3600, "student_bytes_per_window": per_window,
            "privileged_join_bytes_per_window": join.dtype.itemsize}


def training_rates(name):
    p = OUT/"bench"/f"student_bench_{name}.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text())["B4_training"]
    return {L: {"loader_fed_windows_per_s": next(v["windows_per_s"] for k, v in b.items() if k.startswith(f"SB/L{L}/loader_fed")),
                "gpu_resident_windows_per_s": b[f"SB/L{L}/gpu_resident"]["windows_per_s"],
                "batch": b[f"SB/L{L}/gpu_resident"]["batch"]} for L in (64, 128, 256)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="canary-v03")
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    cc = canary_costs()
    ix = index_costs("canary-v0.2")
    ix01 = index_costs("v0.1-1k")
    ex = export_costs(a.export)
    ex01 = export_costs("v01-sb-v03")
    tr = training_rates(a.bench or a.export)
    gpu_h = {k: v["generation_wall_s_mean"]/3600 for k, v in cc.items()}
    out = {"schema": "rind3d-training-scale-cost-v0.3", "inputs": {"canary": cc, "index_canary": ix, "index_v01": ix01,
                                                                   "export_canary": ex, "export_v01": ex01, "training": tr},
           "definitions": {
               "SB-only": "N single-source binary scenes (one release).",
               "combined": "N scenes of one release mixing SB and MA 50/50 (the canary schedule; MA K = 2/3/4 cycled).",
               "separate": "two independently versioned releases of N scenes each: SB-N and MA-N (2N scenes in total).",
               "availability_vs_consumption": "available unique windows = scenes x windows/scene in the fixed index; consumed windows = epochs x windows drawn per epoch. With the hybrid profile the dynamic part adds unique windows per epoch without storage; it does not add scenes (independent geometry/source draws)."},
           "scenarios": {}}
    ib = {"SB": (ix or {}).get("SB_index_bytes_mean", 0), "MA": (ix or {}).get("MA_index_bytes_mean", 0)}
    icpu = {"SB": (ix01 or ix or {}).get("SB_scene_cpu_s_mean", 0), "MA": (ix or {}).get("MA_scene_cpu_s_mean", 0)}
    for scen, mix in (("SB-only", {"SB": 1.0}), ("combined", {"SB": .5, "MA": .5}), ("separate", {"SB": 1.0, "MA": 1.0})):
        for N in SCALES:
            for prof in ("P102", "P256"):
                n = {k: N*v for k, v in mix.items()}
                wps = sum(WINDOWS[prof].values())
                st = ex["student_bytes_per_window"] if ex else {}
                student = sum(n[k]*sum(WINDOWS[prof][L]*st.get(f"{k}/L{L}", 0) for L in (64, 128, 256)) for k in n)
                fields = sum(n[k]*cc[k]["scene_bytes_mean"] for k in n)
                index = sum(n[k]*ib[k]*(1 if prof == "P256" else 102/256) for k in n)
                priv = sum(n.values())*wps*(ex["privileged_join_bytes_per_window"] if ex else 0)
                row = {"scenes": int(sum(n.values())), "windows_per_scene": wps, "windows_available": int(sum(n.values())*wps),
                       "physics_generation_gpu_hours_1_laptop": sum(n[k]*gpu_h[k] for k in n),
                       "validation_cpu_hours": sum(n[k]*cc[k]["validation_s_mean"] for k in n)/3600,
                       "index_cpu_hours": sum(n[k]*icpu[k] for k in n)/3600*(1 if prof == "P256" else 0.45),
                       "storage_GB": {"fields_and_scene_files": fields/1e9, "student_export": student/1e9,
                                      "privileged_join": priv/1e9, "index": index/1e9,
                                      "total": (fields+student+priv+index)/1e9}}
                if tr:
                    # SB training rate per L applied to every window (MA decode cost is measured separately in B1/B3)
                    ep = sum(sum(n.values())*WINDOWS[prof][L]/tr[L]["loader_fed_windows_per_s"] for L in (64, 128, 256))
                    steps = sum(sum(n.values())*WINDOWS[prof][L]/tr[L]["batch"] for L in (64, 128, 256))
                    row["epoch_hours_one_RTX3060_end_to_end"] = ep/3600
                    row["steps_per_epoch"] = int(steps)
                    row["mean_step_seconds"] = ep/steps
                    row["epoch_hours_gpu_compute_bound"] = sum(sum(n.values())*WINDOWS[prof][L]/tr[L]["gpu_resident_windows_per_s"] for L in (64, 128, 256))/3600
                out["scenarios"][f"{scen}/{N}/{prof}"] = row
    out["v01_reuse_1k_SB"] = {"physics_generation_gpu_hours": 0.0,
                              "note": "the 1000 v0.1 SB scenes already have certified complete fields; v0.3 adds only the index and the two-view export",
                              "index": ix01, "export": ex01}
    (ROOT/"RIND3D_TRAINING_SCALE_COST.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else vv.get("total")) for kk, vv in v.items()} for k, v in out["scenarios"].items() if "/100000/" in k}, indent=1, default=float))


if __name__ == "__main__":
    main()
