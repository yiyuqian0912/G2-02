#!/usr/bin/env python3
"""Labeller throughput benchmark on the shipped 1024^3 fixtures, using the production worker pool.

  bench_labels.py --gpus 0 --cpu-workers 8 --gpu-workers-per-gpu 2 --set all --repeat 2 --out bench_1gpu.json
  bench_labels.py --gpus 0-15 --cpu-workers auto --gpu-workers-per-gpu 2 --set all --repeat 16

Every result is checked against the fixture's reference digest; a mismatch is reported as an error, never ignored.
Reports wall time, labels/s, per-fixture CPU and wall times by engine, CPU utilisation (/proc/stat), per-GPU utilisation
and memory (nvidia-smi samples), and the peak worker RSS. Sets: smoke (5 cheap fixtures), all (14 fixtures incl. heavy
tails), or a comma list of names."""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]


def cpu_times():
    f = open("/proc/stat").readline().split()[1:]
    v = list(map(int, f)); idle = v[3] + v[4]
    return sum(v), idle


def sampler(stop, out):
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            out.append([[int(x) for x in l.split(",")] for l in r.stdout.strip().splitlines()])
        stop.wait(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0"); ap.add_argument("--cpu-workers", default="auto")
    ap.add_argument("--gpu-workers-per-gpu", type=int, default=2); ap.add_argument("--set", default="smoke")
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--out"); ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--rss-gb", type=float, default=8.0); ap.add_argument("--gpu-cap", type=int, default=1 << 22)
    a = ap.parse_args()
    from run_node import parse_gpus, default_env  # noqa
    import tempfile
    default_env(os.environ.get("R3D_CACHE_ROOT", str(Path(tempfile.gettempdir()) / "r3d_bench_cache")))   # never inside the repository
    from rind3d_100k.fixtures import index, SMOKE
    from rind3d_100k.pool import WorkerPool, _rss
    gpus = parse_gpus(a.gpus) if a.gpus not in ("", "none") else []
    if gpus is None:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout
        gpus = [int(x) for x in out.split()]
    idx = {d["name"]: d for d in index()}
    names = SMOKE if a.set == "smoke" else (list(idx) if a.set == "all" else a.set.split(","))
    ncpu = len(os.sched_getaffinity(0))
    ng = a.gpu_workers_per_gpu * len(gpus)
    nc = max(0, ncpu - ng - 1) if a.cpu_workers == "auto" else int(a.cpu_workers)
    contract = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v1.json").read_text())
    kw = dict(tier1={k: v for k, v in contract["labels"]["tier1"].items() if k != "gpu_cap"}, gpu=dict(gpu_cap=a.gpu_cap))
    env = dict(OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1,
               **{k: os.environ[k] for k in ("NUMBA_CACHE_DIR", "RIND_NVRTC_LIBRARY", "CUPY_CACHE_DIR") if os.environ.get(k)})
    specs = [dict(name=f"g{i}", kind="label", gpu=True, fn="rind3d_100k.fixtures:fixture_task", engine="gpu", kw=kw,
                  env=dict(env, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])) for i in range(ng)]
    specs += [dict(name=f"c{i}", kind="label", gpu=False, fn="rind3d_100k.fixtures:fixture_task", engine="cpu", kw=kw,
                   env=dict(env, CUDA_VISIBLE_DEVICES="")) for i in range(nc)]
    # heavy first, so tails overlap with the rest
    tasks = sorted([n for n in names for _ in range(a.repeat)], key=lambda n: -idx[n]["ref_cpu_s"])
    heavy = deque(t for t in tasks if idx[t]["L"] >= 256); light = deque(t for t in tasks if idx[t]["L"] < 256)
    # warm the numba / cupy on-disk caches once per engine (one process each, sequentially) so that concurrent
    # first-compiles in fresh workers do not pollute the timing (each engine compiles in ~40-60 s)
    for eng in sorted({sp["engine"] for sp in specs}):
        sp0 = next(sp for sp in specs if sp["engine"] == eng)
        wp = WorkerPool(mp.get_context("spawn"), [dict(sp0, name=f"warm-{eng}")], sys_path=sys.path[:3])
        wp.submit(wp.slots[0], SMOKE[0], 3600)
        while not wp.poll(1.0):
            pass
        wp.shutdown(grace=5)
    pool = WorkerPool(mp.get_context("spawn"), specs, sys_path=sys.path[:3], rss_limit_bytes=a.rss_gb * 2 ** 30)
    stop = threading.Event(); samples = []
    th = threading.Thread(target=sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter(); c0 = cpu_times(); res = []; peak = 0
    while heavy or light or pool.busy():
        for sl in pool.idle("label"):
            q1, q2 = (heavy, light) if sl.get("gpu") else (light, heavy)
            q = q1 if q1 else q2
            if not q:
                break
            pool.submit(sl, q.popleft(), a.timeout)
        for (sl, t, kind, payload) in pool.poll(1.0):
            if kind == "result":
                payload["ok"] = payload.get("digest") == idx[t]["digest"]; payload["worker"] = sl["name"]
                res.append(payload)
            else:
                res.append(dict(name=t, engine=sl["engine"], worker=sl["name"], error=kind, ok=False, **payload))
        peak = max(peak, sum(_rss(s["proc"].pid) for s in pool.slots))
    wall = time.perf_counter() - t0; c1 = cpu_times(); stop.set(); pool.shutdown(grace=5)
    util = 1 - (c1[1] - c0[1]) / max(c1[0] - c0[0], 1)
    by = {}
    for r in res:
        by.setdefault((r["name"], r["engine"]), []).append(r)
    rep = dict(gpus=gpus, gpu_workers=ng, cpu_workers=nc, tasks=len(tasks), wall_s=wall, labels_per_s=len(tasks) / wall,
               all_digests_ok=all(r.get("ok") for r in res), errors=[r for r in res if not r.get("ok")][:10],
               cpu_utilisation=util, peak_worker_rss_gb=peak / 2 ** 30,
               gpu_util_mean=[sum(s[i][1] for s in samples) / len(samples) for i in range(len(samples[0]))] if samples else None,
               gpu_mem_max_mb=[max(s[i][2] for s in samples) for i in range(len(samples[0]))] if samples else None,
               per_fixture={f"{n}|{e}": dict(n=len(v), cpu_mean=sum(x.get("cpu", 0) for x in v) / len(v),
                                             wall_mean=sum(x.get("wall", 0) for x in v) / len(v), ref_cpu_s=idx[n]["ref_cpu_s"])
                            for (n, e), v in by.items()})
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
