"""Resumable label stage: run_labels.py <root> <workers> <gpu_workers> [--uninformative] [--scenes a:b]
                                    [--scene-order] [--sample n64,n128,n256,seed] [--kw json]
                                    [--timeout t64,t128,t256] [--retry-failed] [--nice n] [--engine fastlabel]

--engine fastlabel: FastLabel v3.4 (rind3d_1k.fastlabel; needs CUDA). Statistical labels get status "certified_stat"
with the certificate in the task meta; <gpu_workers> is ignored. Recommended with --scene-order and 15 workers.

Drains all unfinished (scene, window, source) tasks of accepted scenes with a supervised pool of GPU-assisted (exact
GPU pass) and CPU workers. Every finished task is appended to <root>/labels_log.jsonl. Task files are atomic and
skipped on restart; LABELS.json is rebuilt for every touched scene at the end.

Supervision (added after the first release pass, where native worker crashes made multiprocessing.Pool wait forever):
* one task per worker at a time, over a private pipe;
* a worker that dies (signal, abort, segfault) is recorded in <root>/labels_failures.jsonl as kind="crash" and
  replaced; the pass continues;
* a task that exceeds its wall-clock limit (per side, default 1800/3600/7200 s) is killed, recorded as kind="timeout",
  and its worker is replaced;
* tasks with a recorded failure are quarantined: later passes skip them unless --retry-failed is given. No label is
  written for a failed task; its slot reports status "engine_failure" or "timeout" (never a fabricated label)."""
import sys, os, json, time, signal
import multiprocessing as mp
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent


def main():
    sys.path.insert(0, str(HERE))
    from rind3d_1k.labels import tasks_for_scene, task_path, scene_label_summary, load_failures
    from rind3d_1k.supervise import Supervisor
    root = Path(sys.argv[1]); k = int(sys.argv[2]); kg = int(sys.argv[3])
    arg = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    uninf = "--uninformative" in sys.argv
    if not (root / "scenes").exists():
        print(json.dumps(dict(scenes=0, pending=0, note="no published scenes yet")), flush=True); return
    scenes = sorted(p for p in (root / "scenes").iterdir() if (p / "scene.json").exists())
    if arg("--scenes"):
        a, b = (int(x) for x in arg("--scenes").split(":"))
        scenes = [p for p in scenes if a <= int(p.name) < b]
    fails = load_failures(root)
    retry = "--retry-failed" in sys.argv
    tasks = [t for s in scenes for t in tasks_for_scene(s, uninf) if not task_path(t[0], t[1], t[2]).exists()]
    n_quar = sum(1 for t in tasks if (Path(t[0]).name, t[1], t[2]) in fails)
    if not retry:
        tasks = [t for t in tasks if (Path(t[0]).name, t[1], t[2]) not in fails]
    order = {str(s): i for i, s in enumerate(scenes)}
    if "--scene-order" in sys.argv:                      # complete scenes accumulate: batches of 4 scenes, heavy-first inside
        tasks.sort(key=lambda t: (order[t[0]] // 4, -t[3], not t[4]))
    else:
        tasks.sort(key=lambda t: (-t[3], not t[4]))      # big windows first, informative first
    if arg("--sample"):                                  # stratified random sample per side: --sample n64,n128,n256,seed
        import numpy as np
        n64, n128, n256, sd = (int(x) for x in arg("--sample").split(","))
        rng = np.random.default_rng(sd); pick = []
        for L, n in ((256, n256), (128, n128), (64, n64)):
            pool_L = [t for t in tasks if t[3] == L and t[4]]
            pick += [pool_L[i] for i in sorted(rng.choice(len(pool_L), size=min(n, len(pool_L)), replace=False))]
        tasks = pick
    kw = json.loads(arg("--kw")) if arg("--kw") else {}
    lim = dict(zip((64, 128, 256), (float(x) for x in (arg("--timeout") or "1800,3600,7200").split(","))))
    nice = int(arg("--nice") or 0)
    print(json.dumps(dict(scenes=len(scenes), pending=len(tasks), quarantined=0 if retry else n_quar,
                          timeout=lim, start=time.strftime("%Y-%m-%dT%H:%M:%S"))), flush=True)
    if not tasks:
        return
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter(); n = 0; nf = 0; touched = set()
    init = None
    if arg("--engine") == "fastlabel":
        # FastLabel v3.4: every worker labels with the certified GPU verifier; one semaphore serialises GPU submissions
        engines = ["fastlabel"] * k
        init = ("rind3d_1k.fastlabel:set_gpu_lock", (ctx.Semaphore(1),))
    else:
        engines = ["gpu" if i < kg else "cpu" for i in range(k)]
    sup = Supervisor(ctx, engines, "rind3d_1k.labels:supervised_task", kw, nice,
                     sys_path=[str(HERE)] + [p for p in sys.path if "RIND3D_Posterior" in p], init=init)

    def _stop(*_):
        sup.terminate(); sys.exit(143)
    signal.signal(signal.SIGTERM, _stop)
    with open(root / "labels_log.jsonl", "a") as log, open(root / "labels_failures.jsonl", "a") as flog:
        def on_result(m):
            nonlocal n
            if m.get("skip"):
                return
            n += 1; touched.add(m["scene"])
            log.write(json.dumps(m) + "\n"); log.flush()
            if n % 50 == 0:
                print(json.dumps(dict(done=n, of=len(tasks), failed=nf, wall=time.perf_counter() - t0)), flush=True)

        def on_failure(t, rec):
            nonlocal nf
            nf += 1; touched.add(Path(t[0]).name)
            rec = dict(scene=Path(t[0]).name, window=t[1], source=t[2], L=t[3], informative=t[4], kw=kw,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"), **rec)
            flog.write(json.dumps(rec) + "\n"); flog.flush()
            print(json.dumps(dict(failure=rec)), flush=True)

        sup.run(tasks, lambda t: lim.get(t[3], max(lim.values())), on_result, on_failure)
    cov = {}
    for s in scenes:
        if s.name in touched or not (s / "labels" / "LABELS.json").exists():
            c, _ = scene_label_summary(s, uninf, failures=load_failures(root))
            for kk, v in c.items():
                cov[kk] = cov.get(kk, 0) + v
    print(json.dumps(dict(finished=n, failed=nf, wall=time.perf_counter() - t0, coverage_touched=cov)), flush=True)


if __name__ == "__main__":
    main()
