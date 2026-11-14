"""OOD reference labels for natural extreme channels of NAT-1K pilot-eval scenes (research references, not production).

usage: label_ood.py plan <out_tasks.txt>                 declare the random sample (seed 20261113), before labelling
       label_ood.py run <tasks.txt> <workers> [timeout_s] label it, supervised; results in <scene>/labels/ood/

Sample: for each side L and type in {all_lit, all_dark}: up to N_PER = 60 channels, uniformly at random among all
natural channels of that (side, type) in pilot-eval scenes (the 1K val split). The sample is a random subset of a
natural T* stratum, so its references are representative of that stratum; coverage failures are recorded per task.
Engines: all-lit -> production FastLabel v3.4 (label_task); all-dark -> the S6 prototype (s6/fastlabel_s6.py), which
excludes solid interiors from the outer bound. Per-label delta = 0.01 / n_tasks of this OOD family (separate family).
Unresolved tasks (timeout, crash, no certificate) are written as status records; no reference is ever invented."""
import json, os, sys, time
from pathlib import Path
import numpy as np

NAT = (Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "natural_supplement_v1")
HERE = Path(__file__).resolve().parent
N_PER = 60
SEED_BASE = 20261115          # label stream base seed of this family (differs from the release and NAT-1K mixed labels)


def plan(out):
    rng = np.random.default_rng(20261113)
    pool = {}
    for d in sorted((NAT / "scenes").iterdir()):
        nj = d / "natural.json"
        if not nj.exists():
            continue
        nat = json.loads(nj.read_text())
        if nat["role"] != "pilot-eval":
            continue
        for wm in nat["windows"]:
            for j, t in enumerate(wm["types"]):
                if t in ("all_lit", "all_dark"):
                    pool.setdefault((int(wm["Lp"]), t), []).append((str(d), int(wm["window"]), j, int(wm["Lp"]), t))
    tasks = []
    summary = {}
    for key in sorted(pool):
        p = pool[key]
        pick = sorted(rng.choice(len(p), min(N_PER, len(p)), replace=False).tolist())
        tasks += [p[i] for i in pick]
        summary[f"L{key[0]}:{key[1]}"] = dict(pool=len(p), sampled=len(pick))
    Path(out).write_text("".join(" ".join(map(str, t)) + "\n" for t in tasks))
    (Path(out).with_suffix(".plan.json")).write_text(json.dumps(dict(seed=20261113, n_per=N_PER, n_tasks=len(tasks),
                                                                    strata=summary), indent=1))
    print(json.dumps(summary))


def worker(t, engine, kw):
    """Supervisor entry: label one extreme channel; writes <scene>/labels/ood/w{w:03d}_s{s}.npz."""
    sys.path.insert(0, str(HERE.parent / "s6"))
    scene_dir, w, s, L, typ = t
    out = Path(scene_dir) / "labels" / "ood" / f"w{w:03d}_s{s}.npz"
    if out.exists():
        return dict(skip=True)
    t0 = time.perf_counter()
    cfg = dict(base_seed=SEED_BASE, n_tasks=int(kw["n_tasks"]), family_delta=0.01)
    try:
        if typ == "all_dark":
            from fastlabel_s6 import label_task_s6
            lab = label_task_s6(scene_dir, w, s, **cfg); eng = "fastlabel-s6-research"
        else:
            from rind3d_1k.fastlabel import label_task
            lab = label_task(scene_dir, w, s, **cfg); eng = "fastlabel"
    except Exception as e:
        return dict(scene=Path(scene_dir).name, window=w, source=s, L=L, type=typ, error=repr(e)[:300])
    m = lab["meta"]
    meta = dict(window=w, source=s, L=L, type=typ, informative=False, status=lab["status"], tv_bound=float(lab["tv_bound"]),
                engine=eng, engine_version=m.get("version"), path=m.get("path"), certificate=m.get("certificate"),
                wall=time.perf_counter() - t0, n_cells=m.get("n_cells"), solid_removal=m.get("solid_removal"),
                rejected_inside_solid=m.get("rejected_inside_solid"), purpose="OOD reference (research)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, cell=np.asarray(lab["cell"], np.int32), lo=np.asarray(lab["lo"], float), hi=np.asarray(lab["hi"], float),
                        kind=np.asarray(lab["kind"], np.int8), tv_bound=float(lab["tv_bound"]), meta=json.dumps(meta, default=float))
    os.replace(tmp, out)
    return dict(meta, scene=Path(scene_dir).name)


def run(tasks_p, k, timeout):
    import multiprocessing as mp
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RIND3D_Dataset1K"))
    from rind3d_1k.supervise import Supervisor
    tasks = []
    for line in open(tasks_p):
        sd, w, s, L, typ = line.split()
        tasks.append((sd, int(w), int(s), int(L), typ))
    n_tasks = len(tasks)
    todo = [t for t in tasks if not (Path(t[0]) / "labels" / "ood" / f"w{t[1]:03d}_s{t[2]}.npz").exists()]
    ctx = mp.get_context("spawn")
    sup = Supervisor(ctx, ["ood"] * k, "label_ood:worker", dict(n_tasks=n_tasks), 5,
                     sys_path=[str(HERE), str(Path(__file__).resolve().parents[2] / "RIND3D_Dataset1K"),
                               str(Path(__file__).resolve().parents[2] / "RIND3D_Posterior/src")],
                     init=("rind3d_1k.fastlabel:set_gpu_lock", (ctx.Semaphore(1),)))
    log = open(Path(tasks_p).with_suffix(".log.jsonl"), "a")

    def ok(m):
        if not m.get("skip"):
            log.write(json.dumps(m, default=float) + "\n"); log.flush()

    def fail(t, rec):
        log.write(json.dumps(dict(scene=Path(t[0]).name, window=t[1], source=t[2], L=t[3], type=t[4], status="unresolved", **rec),
                             default=float) + "\n"); log.flush()

    sup.run(todo, lambda t: timeout, ok, fail)
    print(json.dumps(dict(done=len(todo))))


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(sys.argv[2])
    else:
        run(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 900.0)
