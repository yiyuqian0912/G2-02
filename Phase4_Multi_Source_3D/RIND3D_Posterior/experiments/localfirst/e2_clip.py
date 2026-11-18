"""E2b: certified root clipping of the exact phase (docs/LOCALFIRST_E123.md §5): the smallest sound change that lets the
tau-pencil engine use Proposition A' constraints at the polytope (not box) level.

usage:
  e2_clip.py run --tasks tasks.json --out OUT [--workers 3] [--timeout 2400] [--n-edges 256]
  e2_clip.py one --task JSON [--n-edges 256]

Per exact-phase task (fresh process, fixed seeds; the true source is never read):
* baseline: stage 1 roots (depth 4), B (lf receiver choice), exact phase on B with the research engine copy
  `tpl/engine_p2xc` and no clip rows (identical to engine_p2x);
* interface edges I: the 'independent' selection of e12.py (|I| = n_edges), H1-free only;
* for root sets {d4 = stage-1 roots, c7 = B's 128^3 cells as depth-7 roots, only if B has <= C7_MAX_CELLS cells}:
  explanations of every interface edge on every root box (`explanations_for_roots`, one shared octree descent; equal
  to the brute-force `box_explanations`, see test_certregion and e2_clip_cost.py), certified wedge rows for unique
  explanations (`root_clips`), roots with an interface edge without explanation dropped (no feasible source);
* runs: d4+clips, c7 without clips (control), c7+clips; each with the B filter. Labels must have the baseline mass
  (relative difference <= 1e-6, else ALARM) and tv_bound <= the baseline's + 1e-12."""
import argparse, json, subprocess, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
C7_MAX_CELLS = 300          # the cell-root variant only when B has at most this many 128^3 cells


def run_engine(geom, win, codes, kls, roots, bf, clip_off, clip_rows, max_records=1 << 22, ne_cap=1 << 26, nq_cap=1 << 25,
               max_vertices=1 << 23, max_nodes_cell=2_000_000, max_nodes_total=5_000_000):
    from rind3d_posterior.tpl import engine_p2 as E, engine_p2xc as EC
    from rind3d_posterior.tpl.run_p2 import window_aspects, summarize
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
    nrec = EC.stage2(P, K, clo, chi, adj, aoff, win.origin, float(win.N), win.L, codes, kls, np.ascontiguousarray(roots, np.int64), n,
                     max_nodes_cell, max_nodes_total, 400, 24, True, True, 2.0, wmode, wfront, 2.0 ** -26,
                     rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
                     fbc, fs, dep_evals, dep_nodes, ne_cap, nq_cap, *bf, True, np.asarray(clip_off, np.int64),
                     np.ascontiguousarray(clip_rows, float).reshape(-1, 4) if len(clip_rows) else np.zeros((1, 4)))
    sec = time.perf_counter() - t0
    st = dict(zip(E.STAT_NAMES, stats.tolist()))
    summ = summarize(dict(cell=rec_cell[:nrec], lo=rec_lo[:nrec], hi=rec_hi[:nrec], kind=rec_kind[:nrec]))
    return dict(seconds=sec, mass=float(np.sum(0.5 * (rec_lo[:nrec] + rec_hi[:nrec]))), tv=float(summ["tv_bound"]),
                stats={k: st[k] for k in ("nodes", "out", "in", "undet", "entry_evals", "comp_evals", "expansions", "passes", "clips", "overflow",
                                          "face_splits", "tau_splits", "geo_splits", "budget_undet")})


def cmd_one(a):
    sys.path.insert(0, str(HERE.parents[3] / "RIND3D_Dataset100K"))
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.fastlabel import engine as T, window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes, LazyWindow2
    from rind3d_posterior.fastlabel.filters import build_filter2, items_from_vox2
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, select_receivers, edge_free_mask
    from rind3d_posterior.localfirst.pipeline import DEFAULTS as CFG
    from rind3d_posterior.localfirst import certregion as CR
    from e12 import select_edges
    t = json.loads(a.task); out = dict(key=t["key"], family=t["family"], L=t["L"], n_edges=a.n_edges)
    task = window_from_100k(t["container"], t["w"], t["s"])
    S = SceneFile(t["container"])
    ctx = BackendFactory(cuda_allowed=False).context(S.geometry(), POLICY); rows = [c.rows for c in ctx.components]
    geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
    cls = window_classes(task); o = np.asarray(task.origin, np.int64); N = int(task.N)
    win = LazyWindow2(cls, o, N)
    lit, dark = interface_edges(cls); comps = plane_components(lit, dark, seed=CFG["seed"])
    roots, _ = T.stage1(geom, win, d1=CFG["s1_d1"], n_boundary=CFG["s1_nb"], n_class=CFG["s1_nc"], n_learn=CFG["s1_learn"], max_tests=CFG["s1_tests"])
    vox, _ = select_receivers(cls, CFG["nbd"], CFG["nin"], seed=CFG["seed"], comps=comps); xc, xk = items_from_vox2(win, vox)
    B = T.stage2(geom, win, xc, xk, roots); Bcells = np.unique(B["cell"])
    if len(Bcells) == 1 and B["stats"]["undet"] == 0 and len(B["cell"]) > 0:
        out["path"] = "single_cell_support"; print(json.dumps(out)); return
    codes, kls = win.top_items(); bf = build_filter2(B)
    none = lambda n: (np.zeros(n + 1, np.int64), np.zeros((0, 4)))
    run_engine(geom, win, codes, kls, roots[:1], bf, *none(1))                     # warm-up
    base = run_engine(geom, win, codes, kls, roots, bf, *none(len(roots)))
    out["baseline"] = base
    # interface constraints
    tt = time.perf_counter()
    free = edge_free_mask(rows, o, N, lit, dark)
    rng = np.random.default_rng([20261118, zlib.crc32(t["key"].encode()), zlib.crc32(b"independent"), a.n_edges])
    I = select_edges(lit, dark, free, comps, a.n_edges, "independent", rng)
    Lp = (o + lit[I] + .5) / N; Kp = (o + dark[I] + .5) / N
    EA, EB, Fr, _ = CR.world_edges(np.asarray(task.P, float), np.asarray(task.K))
    off, idx = CR.admissible_edges(Fr, Lp, Kp)
    out["t_constraints"] = time.perf_counter() - tt; out["used_edges"] = int(len(I))
    c7 = np.stack([np.full(len(Bcells), 7), Bcells // (128 * 128), (Bcells // 128) % 128, Bcells % 128], 1).astype(np.int64)
    CR.explanations_for_roots(EA, EB, Lp[:1], Kp[:1], off[:2], idx, roots[:1], o / N, (o + int(task.L)) / N)   # JIT warm-up
    variants = [("d4", roots)] + ([("c7", c7)] if len(Bcells) <= C7_MAX_CELLS else [])
    if len(Bcells) > C7_MAX_CELLS:
        out["c7"] = dict(skipped=f"B has {len(Bcells)} cells > {C7_MAX_CELLS}: one root per cell repeats the root work (measured 0.1x)")
    for name, R in variants:
        tt = time.perf_counter()
        expl, _ = CR.explanations_for_roots(EA, EB, Lp, Kp, off, idx, R, o / N, (o + int(task.L)) / N)
        coff, crow, empty = CR.root_clips(EA, EB, Lp, Kp, expl)
        keep = ~empty
        # drop provably empty roots; rebuild the CSR for the kept roots
        cnt = np.diff(coff)[keep]; koff = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        krow = np.concatenate([crow[coff[q]:coff[q + 1]] for q in np.flatnonzero(keep)]) if keep.any() and len(crow) else np.zeros((0, 4))
        tcl = time.perf_counter() - tt
        rec = dict(roots=int(len(R)), roots_empty=int(empty.sum()), clip_rows=int(len(krow)), t_clip_build=tcl,
                   unique_share=float(np.mean(expl >= 0)), multi_share=float(np.mean(expl == -2)))
        if keep.any():
            r = run_engine(geom, win, codes, kls, R[keep], bf, koff, krow)
            rec.update(clipped=r)
            if name == "c7":
                rec["unclipped"] = run_engine(geom, win, codes, kls, R, bf, *none(len(R)))
        out[name] = rec
        for k in ("clipped", "unclipped"):
            if k in rec:
                rel = abs(rec[k]["mass"] - base["mass"]) / max(base["mass"], 1e-300)
                rec[k]["mass_rel_err"] = rel
                if rel > 1e-6:
                    out["ALARM"] = f"{name}/{k}: label mass differs from the unrestricted exact phase ({rel:.2e})"
    print(json.dumps(out, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / "e2_clip.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = sorted([t for t in tasks if t["key"] not in done], key=lambda t: -t["L"])
    if a.only:
        todo = [t for t in todo if t["key"] in a.only.split(",")]

    def one(t):
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(HERE), "one", "--task", json.dumps(t), "--n-edges", str(a.n_edges)],
                               capture_output=True, text=True, timeout=a.timeout)
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
    ap.add_argument("--timeout", type=float, default=2400); ap.add_argument("--n-edges", type=int, default=256); ap.add_argument("--only")
    a = ap.parse_args()
    dict(run=cmd_run, one=cmd_one)[a.cmd](a)


if __name__ == "__main__":
    main()
