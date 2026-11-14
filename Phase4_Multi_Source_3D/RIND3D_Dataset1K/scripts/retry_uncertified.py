"""Tier-2 retry: every label slot with TV bound > 0.05 is recomputed with queue capacity 2^26 (CPU reference, at most
`workers` processes because of memory). The better certificate is kept, and both attempts are recorded in the slot
metadata. Usage: retry_uncertified.py <root> <workers> [--timeout s] [--nice n]
Runs under the crash/hang supervisor (rind3d_1k.supervise); failures are recorded with tier=2 and not retried."""
import sys, os, json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent


def main():
    sys.path.insert(0, str(HERE))
    from rind3d_1k.labels import scene_label_summary, load_failures
    from rind3d_1k.supervise import Supervisor
    root = Path(sys.argv[1]); k = int(sys.argv[2])
    limit = float(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 4 * 3600.0
    nice = int(sys.argv[sys.argv.index("--nice") + 1]) if "--nice" in sys.argv else 0
    failed2 = {(sc, w, s) for (sc, w, s), recs in load_failures(root).items() if any(r.get("tier") == 2 for r in recs)}
    todo = []
    for p in sorted(root.glob("scenes/*/labels/tasks/w*_s*.npz")):
        with np.load(p) as z:
            m = json.loads(str(z["meta"]))
            key = (p.parent.parent.parent.name, int(m["window"]), int(m["source"]))
            if float(z["tv_bound"]) > 0.05 and m.get("tier", 1) == 1 and "tier2_attempt" not in m and key not in failed2:
                todo.append(str(p))
    print(json.dumps(dict(retry=len(todo), timeout=limit)), flush=True)
    if not todo:
        return
    sup = Supervisor(mp.get_context("spawn"), ["cpu"] * k, "rind3d_1k.labels:retry_task", {}, nice,
                     sys_path=[str(HERE)] + [q for q in sys.path if "RIND3D_Posterior" in q])
    with open(root / "labels_log.jsonl", "a") as log, open(root / "labels_failures.jsonl", "a") as flog:
        def on_result(r):
            log.write(json.dumps(dict(r, retry=True)) + "\n"); log.flush()
            print(json.dumps(r), flush=True)

        def on_failure(p, rec):
            p = Path(p)
            with np.load(p) as z:
                m = json.loads(str(z["meta"]))
            rec = dict(scene=p.parent.parent.parent.name, window=m["window"], source=m["source"], L=m["L"],
                       informative=m["informative"], tier=2, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **rec)
            flog.write(json.dumps(rec) + "\n"); flog.flush()
            print(json.dumps(dict(failure=rec)), flush=True)
        sup.run(todo, lambda p: limit, on_result, on_failure)
    for d in sorted({Path(p).parent.parent.parent for p in todo}):
        scene_label_summary(d, failures=load_failures(root))


if __name__ == "__main__":
    main()
