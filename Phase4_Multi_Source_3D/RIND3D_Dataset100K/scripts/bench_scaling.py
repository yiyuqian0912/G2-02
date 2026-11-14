#!/usr/bin/env python3
"""Multi-GPU scaling benchmark: runs bench_labels.py (weak scaling: work proportional to the GPU count) and a physical
node canary for G in --gpu-counts, and summarises throughput and scaling efficiency.

  bench_scaling.py --out-dir /scratch/r3d_scaling --gpu-counts 1,2,4,8,16 --cpu-per-gpu 8 --gpu-workers-per-gpu 2 \
                   --label-repeat-per-gpu 1 --scenes-per-gpu 16

CPU workers per configuration = cpu-per-gpu x G (capped by the machine), so the CPU share follows the GPU share.
Efficiency(G) = throughput(G) / (G x throughput(1))."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--gpu-counts", default="1,2,4,8,16")
    ap.add_argument("--cpu-per-gpu", type=int, default=8); ap.add_argument("--gpu-workers-per-gpu", type=int, default=2)
    ap.add_argument("--label-repeat-per-gpu", type=int, default=1); ap.add_argument("--scenes-per-gpu", type=int, default=16)
    ap.add_argument("--set", default="all"); ap.add_argument("--skip-physical", action="store_true"); ap.add_argument("--skip-labels", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    import os
    ncpu = len(os.sched_getaffinity(0)); rows = []
    for G in [int(x) for x in a.gpu_counts.split(",")]:
        gl = f"0-{G - 1}" if G > 1 else "0"
        row = dict(gpus=G)
        if not a.skip_labels:
            cw = min(a.cpu_per_gpu * G, max(0, ncpu - a.gpu_workers_per_gpu * G - 1))
            f = out / f"labels_G{G}.json"
            subprocess.run([sys.executable, str(HERE / "scripts" / "bench_labels.py"), "--gpus", gl, "--cpu-workers", str(cw),
                            "--gpu-workers-per-gpu", str(a.gpu_workers_per_gpu), "--set", a.set,
                            "--repeat", str(a.label_repeat_per_gpu * G), "--out", str(f)], check=False)
            if f.exists():
                b = json.loads(f.read_text())
                row.update(labels_per_s=b["labels_per_s"], label_wall_s=b["wall_s"], digests_ok=b["all_digests_ok"],
                           cpu_util=b["cpu_utilisation"], gpu_util=b["gpu_util_mean"], cpu_workers=cw)
        if not a.skip_physical:
            root = out / f"phys_G{G}"
            t = time.perf_counter()
            subprocess.run([sys.executable, str(HERE / "scripts" / "run_node.py"), "--root", str(root), "--slots",
                            f"0:{a.scenes_per_gpu * G}", "--no-labels", "--gpus", gl, "--epoch-scenes", str(a.scenes_per_gpu * G),
                            "--prior-manifest", str(HERE / "handoff" / "prior" / "release_1k_rate_stats.jsonl")], check=False)
            el = time.perf_counter() - t
            n = sum(1 for l in (root / "manifest" / "physical.jsonl").read_text().splitlines() if '"accepted"' in l)
            row.update(scenes=n, phys_wall_s=el, scenes_per_hour=3600 * n / el)
        rows.append(row); print(json.dumps(row), flush=True)
    base = rows[0]
    for r in rows:
        for k in ("labels_per_s", "scenes_per_hour"):
            if k in r and k in base and base[k]:
                r[k + "_efficiency"] = r[k] / (r["gpus"] * base[k] / base["gpus"])
    (out / "SCALING.json").write_text(json.dumps(rows, indent=1)); print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
