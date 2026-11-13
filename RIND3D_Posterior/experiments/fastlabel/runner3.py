"""Production-like throughput benchmark for FastLabel v3 (adaptive3 + gpuverify): persistent workers (one CUDA context, NVRTC module and numba load per
process; per-scene caches), every informative channel of K random train scenes, tasks in scene order.

  runner.py <n_scenes> <workers> <seed> <out.jsonl> [--kw json]
"""
import json, os, sys, time, subprocess, threading
import multiprocessing as mp
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")


def _init(kw, lock):
    sys.path[:0] = [str(HERE), str(HERE.parent / "twophase")]
    global KW, A
    KW = kw
    import adaptive3 as A_
    A = A_
    A.GPU_LOCK = lock


def _work(t):
    sc, w, s, L = t
    t0 = time.perf_counter()
    try:
        r = A.run(sc, w, s, **KW)
        keep = {k: v for k, v in r.items()}
        keep["L"] = L; keep["task_wall"] = time.perf_counter() - t0; keep["pid"] = os.getpid()
        return keep
    except Exception as e:
        import traceback
        return dict(sc=sc, w=w, s=s, L=L, error=repr(e)[:300], trace=traceback.format_exc()[-1500:], task_wall=time.perf_counter() - t0)


def main():
    import numpy as np
    sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K")
    from rind3d_1k.loader import split_of
    k, nw, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    kw = json.loads(sys.argv[sys.argv.index("--kw") + 1]) if "--kw" in sys.argv else {}
    rng = np.random.default_rng(seed)
    if "--scenes" in sys.argv:
        scenes = [int(x) for x in sys.argv[sys.argv.index("--scenes") + 1].split(",")]
    else:
        cand = [i for i in range(4, 1000) if split_of(json.load(open(R / f"{i:06d}" / "scene.json"))["scene_id"]) == "train"]
        scenes = sorted(rng.choice(cand, k, replace=False))
    tasks = []
    for i in scenes:
        W = np.load(R / f"{i:06d}" / "public" / "windows.npy")
        for r in W:
            for j in range(int(r["k"])):
                if (int(r["mask"]) >> j) & 1:
                    tasks.append((f"{i:06d}", int(r["window"]), j, int(r["L"])))
    print(json.dumps(dict(scenes=[f"{i:06d}" for i in scenes], tasks=len(tasks), workers=nw)), flush=True)
    util = []; stop = threading.Event()
    def samp():
        while not stop.is_set():
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            if q: util.append(int(q))
            stop.wait(5)
    th = threading.Thread(target=samp, daemon=True); th.start()
    t0 = time.perf_counter(); n = 0
    ctx = mp.get_context("spawn")
    gpu_permits = int(sys.argv[sys.argv.index("--gpu-permits") + 1]) if "--gpu-permits" in sys.argv else 0
    lock = ctx.Semaphore(gpu_permits) if gpu_permits > 0 else None
    with ctx.Pool(nw, initializer=_init, initargs=(kw, lock)) as pool, open(out, "w") as f:
        for r in pool.imap_unordered(_work, tasks, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n"); f.flush(); n += 1
    wall = time.perf_counter() - t0; stop.set()
    print(json.dumps(dict(tasks=n, wall=wall, labels_per_s=n / wall, gpu_util_mean=sum(util) / max(len(util), 1),
                          proj_1k_hours=159802 / (n / wall) / 3600)), flush=True)


if __name__ == "__main__":
    main()
