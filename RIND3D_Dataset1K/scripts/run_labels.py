"""Resumable label stage: run_labels.py <root> <workers> <gpu_workers> [--uninformative] [--scenes a:b]
Drains all unfinished (scene, window, source) tasks of accepted scenes, heaviest windows first, with a mixed pool of
GPU-assisted (exact GPU pass) and CPU workers. Every finished task is appended to <root>/labels_log.jsonl. Task
files are atomic and skipped on restart; LABELS.json is rebuilt for every scene at the end."""
import sys, os, json, time
import multiprocessing as mp
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent


def _init(counter, kg, kw):
    sys.path.insert(0, str(HERE))
    global ENG
    KW.update(kw)
    with counter.get_lock():
        i = counter.value; counter.value += 1
    ENG = "gpu" if i < kg else "cpu"


KW = {}


def _work(t):
    from rind3d_1k.labels import run_task
    try:
        kw = dict(KW.get("all", {}))
        m = run_task(t, ENG, gpu_min=1024, gpu_prefix=64, **kw, **KW.get("gpu", {})) if ENG == "gpu" else run_task(t, ENG, **kw)
        m["scene"] = Path(t[0]).name
        return m
    except Exception as e:
        import traceback
        return dict(scene=Path(t[0]).name, window=t[1], source=t[2], error=repr(e)[:400], trace=traceback.format_exc()[-1200:])


def main():
    sys.path.insert(0, str(HERE))
    from rind3d_1k.labels import tasks_for_scene, task_path, scene_label_summary
    root = Path(sys.argv[1]); k = int(sys.argv[2]); kg = int(sys.argv[3])
    uninf = "--uninformative" in sys.argv
    scenes = sorted(p for p in (root / "scenes").iterdir() if (p / "scene.json").exists())
    if "--scenes" in sys.argv:
        a, b = (int(x) for x in sys.argv[sys.argv.index("--scenes") + 1].split(":"))
        scenes = [p for p in scenes if a <= int(p.name) < b]
    tasks = [t for s in scenes for t in tasks_for_scene(s, uninf) if not task_path(t[0], t[1], t[2]).exists()]
    order = {str(s): i for i, s in enumerate(scenes)}
    if "--scene-order" in sys.argv:                      # complete scenes accumulate: batches of 4 scenes, heavy-first inside
        tasks.sort(key=lambda t: (order[t[0]] // 4, -t[3], not t[4]))
    else:
        tasks.sort(key=lambda t: (-t[3], not t[4]))      # big windows first, informative first
    if "--sample" in sys.argv:                           # stratified random sample per side: --sample n64,n128,n256,seed
        import numpy as np
        n64, n128, n256, sd = (int(x) for x in sys.argv[sys.argv.index("--sample") + 1].split(","))
        rng = np.random.default_rng(sd); pick = []
        for L, n in ((256, n256), (128, n128), (64, n64)):
            pool_L = [t for t in tasks if t[3] == L and t[4]]
            pick += [pool_L[i] for i in sorted(rng.choice(len(pool_L), size=min(n, len(pool_L)), replace=False))]
        tasks = pick
    kw = json.loads(sys.argv[sys.argv.index("--kw") + 1]) if "--kw" in sys.argv else {}
    print(json.dumps(dict(scenes=len(scenes), pending=len(tasks))), flush=True)
    ctx = mp.get_context("spawn"); counter = ctx.Value("i", 0)
    t0 = time.perf_counter(); n = 0
    import signal
    with ctx.Pool(k, initializer=_init, initargs=(counter, kg, kw)) as pool, open(root / "labels_log.jsonl", "a") as log:
        def _stop(*_):
            pool.terminate(); sys.exit(143)
        signal.signal(signal.SIGTERM, _stop)
        for m in pool.imap_unordered(_work, tasks, chunksize=1):
            if m.get("skip"):
                continue
            n += 1
            log.write(json.dumps(m) + "\n"); log.flush()
            if n % 50 == 0:
                print(json.dumps(dict(done=n, of=len(tasks), wall=time.perf_counter() - t0)), flush=True)
    cov = {}
    for s in scenes:
        c, _ = scene_label_summary(s, uninf)
        for kk, v in c.items():
            cov[kk] = cov.get(kk, 0) + v
    print(json.dumps(dict(finished=n, wall=time.perf_counter() - t0, coverage=cov)), flush=True)


if __name__ == "__main__":
    main()
