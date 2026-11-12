"""Tier-2 retry: every label slot with TV bound > 0.05 is recomputed with queue capacity 2^26 (CPU reference, at most
`workers` processes because of memory). The better certificate is kept, and both attempts are recorded in the slot
metadata. Usage: retry_uncertified.py <root> <workers>"""
import sys, os, json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent


def work(p):
    sys.path.insert(0, str(HERE))
    from rind3d_1k.labels import SceneObs, STATUS_CERT, STATUS_LOOSE, STATUS_UNCERT
    from rind3d_posterior import label_window, ENGINE_VERSION
    p = Path(p)
    with np.load(p) as z:
        old = json.loads(str(z["meta"])); old_tv = float(z["tv_bound"])
    scene_dir = p.parent.parent.parent
    S = SceneObs(scene_dir)
    w, s = old["window"], old["source"]
    t0 = time.perf_counter(); c0 = time.process_time()
    r = label_window(S.P, S.K, S.cls(w, s), S.origin(w), int(S.scene["resolution"]), engine="cpu",
                     nq_cap=1 << 26, ne_cap=1 << 27)
    tv = float(r["tv_bound"])
    rec = dict(tier=2, tv_bound=tv, wall=time.perf_counter() - t0, cpu=time.process_time() - c0,
               overflow=int(r["stats"]["overflow"]), undet=int(r["stats"]["undet"]))
    if tv < old_tv:
        meta = dict(old, tv_bound=tv, status=STATUS_CERT if tv <= 0.05 else (STATUS_LOOSE if tv <= 0.10 else STATUS_UNCERT),
                    tier=2, tier1=dict(tv_bound=old_tv, status=old["status"]), engine="cpu", engine_version=ENGINE_VERSION,
                    params=dict(nq_cap=1 << 26, ne_cap=1 << 27), n_records=int(len(r["cell"])))
        tmp = p.with_name(p.stem + f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, cell=np.asarray(r["cell"], np.int32), lo=r["lo"], hi=r["hi"],
                            kind=np.asarray(r["kind"], np.int8), tv_bound=tv, meta=json.dumps(meta))
        os.replace(tmp, p)
        rec["replaced"] = True
    else:
        meta = dict(old, tier2_attempt=rec)
        tmp = p.with_name(p.stem + f".tmp{os.getpid()}.npz")
        with np.load(p) as z:
            np.savez_compressed(tmp, cell=z["cell"], lo=z["lo"], hi=z["hi"], kind=z["kind"], tv_bound=z["tv_bound"],
                                meta=json.dumps(meta))
        os.replace(tmp, p)
        rec["replaced"] = False
    rec.update(scene=scene_dir.name, window=w, source=s, old_tv=old_tv)
    return rec


def main():
    root = Path(sys.argv[1]); k = int(sys.argv[2])
    todo = []
    for p in sorted(root.glob("scenes/*/labels/tasks/w*_s*.npz")):
        with np.load(p) as z:
            m = json.loads(str(z["meta"]))
            if float(z["tv_bound"]) > 0.05 and m.get("tier", 1) == 1 and "tier2_attempt" not in m:
                todo.append(str(p))
    print(json.dumps(dict(retry=len(todo))), flush=True)
    with mp.get_context("spawn").Pool(k) as pool, open(root / "labels_log.jsonl", "a") as log:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            log.write(json.dumps(dict(r, retry=True)) + "\n"); log.flush()
            print(json.dumps(r), flush=True)
    sys.path.insert(0, str(HERE))
    from rind3d_1k.labels import scene_label_summary
    for d in sorted({Path(p).parent.parent.parent for p in todo}):
        scene_label_summary(d)


if __name__ == "__main__":
    main()
