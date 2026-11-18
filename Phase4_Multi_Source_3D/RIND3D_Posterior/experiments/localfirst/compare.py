"""Tau-pencil vs FastLabel vs local-first: same worlds, windows, grid and target (docs/LOCALFIRST_COMPARISON.md).

usage:
  compare.py tasks  --root ROOT --out tasks.json [--per 1] [--seed 0]
  compare.py run    --tasks tasks.json --methods tp,lf,rand --out DIR [--workers 8] [--timeout 1800]
  compare.py run    --tasks tasks.json --methods fl --out DIR --workers 1            (GPU, statistical)
  compare.py report --tasks tasks.json --out DIR
Methods (one fresh process per (task, method), so wall time and peak RSS (VmHWM) belong to that label alone):
  tp    tau-pencil reference (rind3d_posterior.label_window, CPU engine): deterministic certificate;
  lf    local-first: receivers from the local planar components, outer bound B, exact tau-pencil phase 2 on B;
  rand  as lf with FastLabel v3.4's receiver choice (random interface + interior receivers): isolates the selection;
  fl    FastLabel v3.4 (label_window_fast): statistical certificate (GPU verifier);
  noop  imports only (baseline RSS).
Each label is saved as DIR/labels/<key>_<method>.npz (cell, lo, hi); DIR/<method>.jsonl holds the measurements."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
D100K = HERE.parents[3] / "RIND3D_Dataset100K"


def vmhwm_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1]) / 1024
    return None


def cmd_tasks(a):
    sys.path.insert(0, str(D100K))
    from rind3d_100k.storage import SceneFile
    rng = np.random.default_rng(a.seed)
    groups = {}
    for p in sorted(Path(a.root).glob("scenes/*/*.npz")):
        S = SceneFile(p)
        for w, r in enumerate(S.W):
            for s in range(S.k):
                if (int(r["mask"]) >> s) & 1:
                    groups.setdefault((S.meta["family"], int(r["Lp"])), []).append(dict(container=str(p), slot=int(S.meta["slot"]), w=w, s=s,
                                                                                         family=S.meta["family"], L=int(r["L"]), k=S.k, m=int(r["m"])))
    tasks = []
    for key in sorted(groups):
        g = groups[key]
        for i in rng.choice(len(g), min(a.per, len(g)), replace=False):
            t = g[int(i)]; t["key"] = f"{t['slot']:06d}_w{t['w']:02d}_s{t['s']}"; tasks.append(t)
    Path(a.out).write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} tasks from {len(groups)} (family, L) groups")


def cmd_one(a):
    t = json.loads(a.task); method = a.method; out = Path(a.out)
    from rind3d_posterior.fastlabel import window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes
    task = window_from_100k(t["container"], t["w"], t["s"])
    base = vmhwm_mb()
    t0 = time.perf_counter(); c0 = time.process_time(); meta = {}
    if method == "noop":
        lab = dict(cell=np.zeros(0, np.int32), lo=np.zeros(0), hi=np.zeros(0), tv_bound=None, status="noop")
    elif method == "tp":
        from rind3d_posterior import label_window
        r = label_window(task.P, task.K, window_classes(task), task.origin, task.N)
        tv = float(r["tv_bound"])
        lab = dict(cell=np.asarray(r["cell"], np.int32), lo=r["lo"], hi=r["hi"], tv_bound=tv,
                   status="certified" if tv <= 0.05 else ("certified_loose" if tv <= 0.10 else "uncertified"))
        meta = dict(stats={k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in (r.get("stats") or {}).items()
                           if isinstance(v, (int, float, np.integer, np.floating))})
    elif method in ("lf", "rand"):
        from rind3d_posterior.localfirst.pipeline import label_window_localfirst
        lab = label_window_localfirst(task, select="local" if method == "lf" else "random")
        meta = lab.pop("meta")
    elif method == "fl":
        from rind3d_posterior.fastlabel import label_window_fast
        lab = label_window_fast(task, n_tasks=4800000, base_seed=20261118)
        m = lab.pop("meta")
        meta = dict(path=m.get("path"), rounds=[{k: v for k, v in r.items() if k in ("round", "items", "p1_s", "records", "undet", "tested", "K")}
                                                for r in m.get("rounds", [])], gpu_verifier=m.get("verifier"))
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    (out / "labels").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "labels" / f"{t['key']}_{method}.npz", cell=np.asarray(lab["cell"], np.int64),
                        lo=np.asarray(lab["lo"], float), hi=np.asarray(lab["hi"], float))
    rec = dict(key=t["key"], method=method, family=t["family"], L=t["L"], wall=wall, cpu=cpu, rss_mb=vmhwm_mb(), rss_base_mb=base,
               status=lab["status"], tv_bound=lab["tv_bound"], n_records=int(len(lab["cell"])), meta=meta)
    print(json.dumps(rec, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",")
    for method in methods:
        fp = out / f"{method}.jsonl"
        done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
        todo = [t for t in tasks if t["key"] not in done]
        # largest windows first, so long labels do not end up alone at the end
        todo.sort(key=lambda t: -t["L"])

        def one(t):
            t0 = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, str(HERE), "one", "--task", json.dumps(t), "--method", method, "--out", str(out)],
                                   capture_output=True, text=True, timeout=a.timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                rec = json.loads(line[-1]) if line else dict(key=t["key"], method=method, family=t["family"], L=t["L"], status="error",
                                                              error=p.stderr[-2000:])
            except subprocess.TimeoutExpired:
                rec = dict(key=t["key"], method=method, family=t["family"], L=t["L"], status="timeout", wall=a.timeout)
            rec["outer_wall"] = time.perf_counter() - t0
            with open(fp, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(method, rec["key"], rec["status"], round(rec.get("wall") or 0, 1), flush=True)

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))


def _q(path):
    z = np.load(path); c, lo, hi = z["cell"], z["lo"], z["hi"]
    u, inv = np.unique(c, return_inverse=True)
    L = np.bincount(inv, weights=lo, minlength=len(u)); U = np.bincount(inv, weights=hi, minlength=len(u))
    return u, L, U


def _tv(a, b):
    ua, qa = a; ub, qb = b
    allc = np.union1d(ua, ub)
    va = np.zeros(len(allc)); vb = np.zeros(len(allc))
    va[np.searchsorted(allc, ua)] = qa; vb[np.searchsorted(allc, ub)] = qb
    return 0.5 * float(np.abs(va - vb).sum())


def cmd_report(a):
    sys.path.insert(0, str(D100K))
    tasks = {t["key"]: t for t in json.loads(Path(a.tasks).read_text())}
    out = Path(a.out)
    recs = {}
    for fp in out.glob("*.jsonl"):
        for l in fp.read_text().splitlines():
            if l.strip():
                r = json.loads(l); recs[(r["key"], r["method"])] = r
    from rind3d_posterior.fastlabel import window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, conditioning, local_band_cells
    rows = []
    for key, t in tasks.items():
        row = dict(key=key, family=t["family"], L=t["L"])
        ref = None
        p = out / "labels" / f"{key}_tp.npz"
        if p.exists() and recs.get((key, "tp"), {}).get("status") not in (None, "error", "timeout"):
            u, Lv, Uv = _q(p); ref = (u, Lv / Lv.sum() if Lv.sum() > 0 else Uv / Uv.sum())
        for m in ("tp", "lf", "rand", "fl"):
            r = recs.get((key, m))
            if r is None:
                continue
            row[m] = dict(status=r.get("status"), wall=r.get("wall"), cpu=r.get("cpu"), rss=(r.get("rss_mb") or 0) - (r.get("rss_base_mb") or 0),
                          tv_bound=r.get("tv_bound"))
            if m in ("lf", "rand"):
                mm = r.get("meta") or {}
                row[m].update(B_cells=(mm.get("B") or {}).get("cells"), B_volume=(mm.get("B") or {}).get("volume"),
                              path=mm.get("path"), times=mm.get("times"), cond=((mm.get("local") or {}).get("conditioning")))
            if m == "fl":
                row[m].update(path=(r.get("meta") or {}).get("path"))
            lp = out / "labels" / f"{key}_{m}.npz"
            if ref is not None and lp.exists() and r.get("status") not in ("error", "timeout"):
                u, Lv, Uv = _q(lp)
                q = Lv / Lv.sum() if Lv.sum() > 0 else (Uv / Uv.sum() if Uv.sum() > 0 else Uv)
                row[m]["tv_to_tp"] = _tv((u, q), ref)
        # X-only local diagnostics against the reference label
        if ref is not None:
            task = window_from_100k(t["container"], t["w"], t["s"])
            cls = window_classes(task)
            li, da = interface_edges(cls); comps = plane_components(li, da)
            cond = conditioning(comps)
            band = local_band_cells(comps, task.origin, task.N)
            u, q = ref
            from rind3d_100k.storage import SceneFile
            s_ = SceneFile(t["container"]).sources()[t["s"]]; i_ = np.minimum((s_ * 128).astype(np.int64), 127)
            sc = int((i_[0] * 128 + i_[1]) * 128 + i_[2])
            row["local"] = dict(cond=cond, n_planes=len(comps), band_cells=int(band.sum()), band_mass=float(q[band[u]].sum()),
                                support_cells=int(len(u)), source_in_band=bool(band[sc]))
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    (out / "report_rows.json").write_text(json.dumps(rows, default=float, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("one"); p.add_argument("--task", required=True); p.add_argument("--method", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("--tasks", required=True); p.add_argument("--methods", required=True); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--timeout", type=float, default=1800)
    p = sub.add_parser("report"); p.add_argument("--tasks", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(tasks=cmd_tasks, one=cmd_one, run=cmd_run, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
