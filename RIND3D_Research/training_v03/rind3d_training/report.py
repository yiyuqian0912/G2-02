"""Assemble RIND3D_LOADER_BENCHMARKS.json and the label-free public summaries used by the review archive.

  run.sh rind3d_training.report
"""
import json
import shutil

from .env import OUT, ROOT


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    b = OUT/"bench"
    out = {"schema": "rind3d-loader-benchmarks-v0.3",
           "machine": "RTX 3060 Laptop (6 GB), 16 CPU threads, NVMe; one process unless stated; cold = page cache dropped with posix_fadvise(DONTNEED)",
           "field_side_producer_venv": load(b/"field_bench.json"),
           "student_canary": load(b/"student_bench_canary-v03.json"),
           "student_v01_1k": load(b/"student_bench_v01-sb-v03.json"),
           "blocksparse_canary": load(b/"blocksparse_canary-v03.json"),
           "teacher_call_costs": (load(OUT/"tests"/"interop_audit.json") or {}).get("A7_cost_per_call"),
           "explanation_of_v02_numbers": {
               "156_SB_L64_per_s": "v0.2 ProductionDataset, one process, random cross-scene SB L64 windows: ~4 ms per window of which ~3.7 ms is opening the scene (RINDPack headers + windows.npy; LRU 8 of 216 scenes always misses) and ~3.8 ms read+unpack. Re-measured: 134/s cold, 164/s warm, 281/s when consecutive windows share a scene (F1).",
               "10.64_s_for_256_windows": "the 2-worker DataLoader pass draws 256 NATURAL windows of all sides and BOTH profiles (composition in F2). One third are L256; an SB L256 window returns 33.5 MB of unpacked bool (response + SOLID) and an MA L256 window 151 MB (float64 response + SOLID). The pass returns 7.5 GB, which 2 workers must pickle through IPC: 9.8 s with 2 workers vs 7.7 s single-process (F2). It is dominated by L256 decode and IPC volume, not by the small-window rate: the two numbers measure different things and are consistent.",
               "fix": "v0.3 student shards keep payloads packed and compressed in sharded files (one pread per payload, no per-scene opens), ship packed bits through the DataLoader and unpack on the GPU"}}
    (ROOT/"RIND3D_LOADER_BENCHMARKS.json").write_text(json.dumps(out, indent=1, default=float))
    pub = OUT/"public_summaries"
    pub.mkdir(parents=True, exist_ok=True)
    for rel in ("canary-v0.2", "v0.1-1k"):
        s = OUT/"privileged"/"index"/rel/"P256"/"summary.json"
        if s.exists():
            shutil.copy(s, pub/f"index_summary_{rel}.json")                     # counts and timings only
    for name in ("canary-v03", "v01-sb-v03"):
        s = OUT/"privileged"/"export"/name/"export_summary.json"
        if s.exists():
            shutil.copy(s, pub/f"export_summary_{name}.json")
    print("written", ROOT/"RIND3D_LOADER_BENCHMARKS.json")


if __name__ == "__main__":
    main()
