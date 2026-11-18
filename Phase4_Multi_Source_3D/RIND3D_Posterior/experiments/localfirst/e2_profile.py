"""E2 operation-level profile of the exact phase (tau-pencil phase 2 on B, outer-bound filter) and the coarse-root
experiment (docs/LOCALFIRST_E123.md §5).

usage:
  e2_profile.py run --tasks tasks.json --out OUT [--workers 3] [--timeout 1800]
  e2_profile.py one --task JSON

Per task (fresh process): the lf pathway's stage 1 and B, then the exact phase on B with the engine's per-root and
per-depth counters exposed (cell_nodes / cell_evals / dep_evals / dep_nodes, otherwise discarded by the wrapper):
* how many stage-1 roots are actually evaluated, and how the item evaluations split over roots and BSP depths;
* depth-0 evaluations (the full item list scanned at the root node of every root cell) versus deeper ones;
* the coarse-root variant: the same exact phase with roots merged to depth 3 and to depth 2 (every root box replaced
  by its ancestor; sound: the engine accepts any octree box as a root and the filter keeps B), timed back to back,
  with the label mass and tv_bound compared to the depth-4 run."""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()


def stage2_profiled(geom, win, codes, kls, roots, bf, max_records=1 << 22, ne_cap=1 << 26, nq_cap=1 << 25, max_vertices=1 << 23,
                    max_nodes_cell=2_000_000, max_nodes_total=5_000_000):
    """fastlabel.engine.stage2 with bf, returning the per-root / per-depth counters as well."""
    from rind3d_posterior.tpl import engine_p2 as E, engine_p2x as EX
    from rind3d_posterior.tpl.run_p2 import window_aspects
    P, K, clo, chi, adj, aoff = geom.tup()
    wmode, wfront = window_aspects(geom, win)
    rec_cell = np.empty(max_records, np.int64); rec_lo = np.empty(max_records); rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8); rec_voff = np.empty(max_records, np.int64); rec_vn = np.empty(max_records, np.int64)
    vpool = np.empty((max_vertices, 3)); vtop = np.zeros(1, np.int64); stats = np.zeros(len(E.STAT_NAMES), np.int64)
    n = len(roots)
    cell_nodes = np.zeros(max(n, 1), np.int64); cell_evals = np.zeros(max(n, 1), np.int64)
    dep_evals = np.zeros(64, np.int64); dep_nodes = np.zeros(64, np.int64)
    fbc = np.zeros(geom.C, np.int64); fs = np.zeros(geom.C, np.int64)
    t0 = time.perf_counter()
    nrec = EX.stage2(P, K, clo, chi, adj, aoff, win.origin, float(win.N), win.L, codes, kls, np.ascontiguousarray(roots, np.int64), n,
                     max_nodes_cell, max_nodes_total, 400, 24, True, True, 2.0, wmode, wfront, 2.0 ** -26,
                     rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
                     fbc, fs, dep_evals, dep_nodes, ne_cap, nq_cap, *bf, True)
    sec = time.perf_counter() - t0
    st = dict(zip(E.STAT_NAMES, stats.tolist()))
    mass = float(np.sum(0.5 * (rec_lo[:nrec] + rec_hi[:nrec])))
    from rind3d_posterior.tpl.run_p2 import summarize
    summ = summarize(dict(cell=rec_cell[:nrec], lo=rec_lo[:nrec], hi=rec_hi[:nrec], kind=rec_kind[:nrec]))
    return dict(seconds=sec, stats=st, mass=mass, tv=float(summ["tv_bound"]), cell_nodes=cell_nodes[:n], cell_evals=cell_evals[:n],
                dep_evals=dep_evals, dep_nodes=dep_nodes, nrec=int(nrec))


def merge_roots(roots, depth):
    sh = roots[:, 0] - depth
    anc = np.stack([np.full(len(roots), depth), roots[:, 1] >> sh, roots[:, 2] >> sh, roots[:, 3] >> sh], 1)
    return np.unique(anc, axis=0).astype(np.int64)


def cmd_one(a):
    from rind3d_posterior.fastlabel import engine as T, window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes, LazyWindow2
    from rind3d_posterior.fastlabel.filters import build_filter2, items_from_vox2
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, select_receivers
    from rind3d_posterior.localfirst.pipeline import DEFAULTS as CFG
    t = json.loads(a.task); out = dict(key=t["key"], family=t["family"], L=t["L"])
    task = window_from_100k(t["container"], t["w"], t["s"])
    geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
    cls = window_classes(task); win = LazyWindow2(cls, np.asarray(task.origin, np.int64), int(task.N))
    lit, dark = interface_edges(cls); comps = plane_components(lit, dark, seed=CFG["seed"])
    roots, _ = T.stage1(geom, win, d1=CFG["s1_d1"], n_boundary=CFG["s1_nb"], n_class=CFG["s1_nc"], n_learn=CFG["s1_learn"], max_tests=CFG["s1_tests"])
    vox, _ = select_receivers(cls, CFG["nbd"], CFG["nin"], seed=CFG["seed"], comps=comps); xc, xk = items_from_vox2(win, vox)
    B = T.stage2(geom, win, xc, xk, roots)
    cells = np.unique(B["cell"])
    if len(cells) == 1 and B["stats"]["undet"] == 0 and len(B["cell"]) > 0:
        out["path"] = "single_cell_support"; print(json.dumps(out)); return
    codes, kls = win.top_items(); bf = build_filter2(B)
    out["items"] = int(len(codes)); out["roots"] = int(len(roots))
    stage2_profiled(geom, win, codes, kls, roots[:1], bf)                         # warm-up (JIT / cache load)
    runs = {}
    for name, R in (("d4", roots), ("d3", merge_roots(roots, 3)), ("d2", merge_roots(roots, 2))):
        r = stage2_profiled(geom, win, codes, kls, R, bf)
        ev = r["cell_evals"]; nd = r["cell_nodes"]
        runs[name] = dict(roots=int(len(R)), seconds=r["seconds"], mass=r["mass"], tv=r["tv"], nrec=r["nrec"],
                          stats={k: r["stats"][k] for k in ("nodes", "out", "in", "undet", "entry_evals", "comp_evals", "expansions", "passes",
                                                           "face_splits", "tau_splits", "geo_splits", "budget_undet", "overflow")},
                          roots_evaluated=int((ev > 0).sum()), roots_with_nodes_gt1=int((nd > 1).sum()),
                          evals_top_root_share=float(np.sort(ev)[::-1][:1].sum() / max(ev.sum(), 1)),
                          evals_depth0=int(r["dep_evals"][0]), evals_total=int(r["dep_evals"].sum()),
                          evals_by_depth=[int(x) for x in r["dep_evals"][:24]], nodes_by_depth=[int(x) for x in r["dep_nodes"][:24]])
    out["runs"] = runs
    m4 = runs["d4"]["mass"]
    for k in ("d3", "d2"):
        runs[k]["mass_rel"] = runs[k]["mass"] / max(m4, 1e-300); runs[k]["speedup"] = runs["d4"]["seconds"] / max(runs[k]["seconds"], 1e-9)
    print(json.dumps(out, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / "e2_profile.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = sorted([t for t in tasks if t["key"] not in done], key=lambda t: -t["L"])

    def one(t):
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(HERE), "one", "--task", json.dumps(t)], capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else dict(key=t["key"], status="error", error=p.stderr[-3000:])
        except subprocess.TimeoutExpired:
            rec = dict(key=t["key"], status="timeout")
        rec["outer_wall"] = time.perf_counter() - t0
        with open(fp, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(rec["key"], rec.get("status", "ok"), round(rec["outer_wall"], 1), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--tasks"); ap.add_argument("--task"); ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    dict(run=cmd_run, one=cmd_one)[a.cmd](a)


if __name__ == "__main__":
    main()
