"""Audit of the local-first pathway on the 54 cmp_v3 tasks (docs/LOCALFIRST_AUDIT.md). Diagnostics only: nothing here
produces or replaces a label, and the restricted exact runs (oracle / band) are never used as labels.

usage:
  audit.py run    --tasks tasks.json --labels DIR/labels --out OUT [--workers 4]
  audit.py one    --task JSON --labels DIR/labels                        (one task, prints one JSON line)
  audit.py report --out OUT

Per task (one fresh process, deterministic: same seeds as label_window_localfirst and FastLabel v3.4):
1. Receiver selection. X'_local (select_receivers on the planar components) and X'_random (FastLabel's
   point_items with the label's keyed seed): sizes, overlap (Jaccard), class mix, interface share, coverage of the
   planar components.
2. Global stages with engine counters. stage 1 (sentinel coarse search: root boxes), stage 2 on X' (outer bound B)
   for both selections, then the exact phase 2 on B (all receivers, outer-bound filter).
3. Local constraints, measured but not applied.
   - Per planar component (>= 24 edges): H1 (exact free-edge test against the scene's bodies) and H2 at the TRUE
     source (LP: a plane through the true source crosses every edge of the component).
   - The heuristic band (local_band_cells): its volume, the exact label mass outside it (tau-pencil reference label),
     and the share of B's volume inside it.
4. Upper bounds on any region reduction (diagnostic exact runs, restricted to sub-regions of B):
   - ORACLE: B restricted to the 128^3 cells of the reference label's support (the best a perfect local region
     could do at cell granularity): time and engine counters;
   - BAND: B restricted to the band cells: time, engine counters and label mass lost (what uncertified pruning by
     the heuristic band would do)."""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
D100K = HERE.parents[3] / "RIND3D_Dataset100K"
KEYS = ("nodes", "out", "in", "undet", "budget_undet", "overflow", "passes", "entry_evals", "comp_evals", "ambig_evals",
        "expansions", "cells", "cells_out", "cells_in_any", "face_splits", "tau_splits", "window_splits", "geo_splits")


def _stab_through(li, da, s_vox):
    """LP: is there a plane through s_vox with every lit endpoint on its active side and every dark endpoint on the
    other side (closed)? Window voxel coordinates, receiver v at v + 1/2."""
    from scipy.optimize import linprog
    l = li + .5; k = da + .5; g = (l - k).mean(0); gn = np.linalg.norm(g)
    if gn == 0:
        return None
    g /= gn
    A_ub = np.r_[np.c_[-l, np.ones(len(l))], np.c_[k, -np.ones(len(k))]]
    A_eq = np.array([np.r_[s_vox, -1.0], np.r_[g, 0.0]])
    r = linprog(np.zeros(4), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq, b_eq=[0.0, 1.0], bounds=[(None, None)] * 4, method="highs")
    return bool(r.status == 0)


def _subset(res, keep):
    keep = np.asarray(keep, bool)
    return dict(cell=res["cell"][keep], lo=res["lo"][keep], hi=res["hi"][keep], kind=res["kind"][keep], voff=res["voff"][keep],
                vn=res["vn"][keep], vpool=res["vpool"])


def _st(res):
    return {k: int(res["stats"].get(k, 0)) for k in KEYS}


def _vol(res):
    return float(np.sum(res["hi"])), int(len(np.unique(res["cell"])))


def _cellmass(path):
    z = np.load(path); c, lo, hi = z["cell"], z["lo"], z["hi"]
    u, inv = np.unique(c, return_inverse=True)
    return u, np.bincount(inv, weights=0.5 * (lo + hi), minlength=len(u))


def cmd_one(a):
    sys.path.insert(0, str(D100K))
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.fastlabel import engine as T, window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes, LazyWindow2
    from rind3d_posterior.fastlabel.filters import build_filter2, items_from_vox2
    from rind3d_posterior.fastlabel.pipeline import label_rng
    from rind3d_posterior.localfirst.local import (interface_edges, plane_components, select_receivers, local_band_cells,
                                                   edge_free_mask, conditioning)
    from rind3d_posterior.localfirst.pipeline import DEFAULTS as CFG
    t = json.loads(a.task)
    task = window_from_100k(t["container"], t["w"], t["s"])
    S = SceneFile(t["container"]); s_true = S.sources()[t["s"]]
    rows = [c.rows for c in BackendFactory(cuda_allowed=False).context(S.geometry(), POLICY).components]
    geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
    cls = window_classes(task); o = np.asarray(task.origin, np.int64); N = int(task.N)
    win = LazyWindow2(cls, o, N)
    out = dict(key=t["key"], family=t["family"], L=t["L"])

    # ---- 1. local description and the two receiver subsets
    tt = time.perf_counter()
    lit, dark = interface_edges(cls)
    comps = plane_components(lit, dark, seed=CFG["seed"])
    out["t_local"] = time.perf_counter() - tt
    big = [c for c in comps if c["n"] >= 24]
    out["edges"] = int(len(lit)); out["components"] = len(big); out["conditioning"] = conditioning(comps)
    vox_l, _ = select_receivers(cls, CFG["nbd"], CFG["nin"], seed=CFG["seed"], comps=comps)
    xc_l, xk_l = items_from_vox2(win, vox_l)
    sc, w, s = (int(v) for v in task.key)
    xc_r, xk_r = T.point_items(win, CFG["nbd"], CFG["nin"], label_rng(20261112, sc, w, s)[0])
    dec = lambda c: {(int(v >> 32) & 0xFFFF, int(v >> 16) & 0xFFFF, int(v) & 0xFFFF) for v in c}
    Sl, Sr = dec(xc_l), dec(xc_r)
    endp = {tuple(v) for v in lit.tolist()} | {tuple(v) for v in dark.tolist()}
    comp_vox = set()
    for c in big:
        comp_vox |= {tuple(v) for v in lit[c["idx"]].tolist()} | {tuple(v) for v in dark[c["idx"]].tolist()}
    def mix(Sx, xk):
        return dict(n=len(Sx), active=int((xk == 1).sum()), inactive=int((xk == 0).sum()), interface=len(Sx & endp),
                    on_components=len(Sx & comp_vox),
                    components_touched=sum(1 for c in big if len(Sx & ({tuple(v) for v in lit[c["idx"]].tolist()} |
                                                                    {tuple(v) for v in dark[c["idx"]].tolist()}))))
    out["sel"] = dict(local=mix(Sl, xk_l), random=mix(Sr, xk_r), intersection=len(Sl & Sr), union=len(Sl | Sr),
                      jaccard=len(Sl & Sr) / max(len(Sl | Sr), 1),
                      interface_jaccard=len(Sl & Sr & endp) / max(len((Sl | Sr) & endp), 1))

    # ---- 2. global stages with engine counters
    tt = time.perf_counter()
    roots, _ = T.stage1(geom, win, d1=CFG["s1_d1"], n_boundary=CFG["s1_nb"], n_class=CFG["s1_nc"], n_learn=CFG["s1_learn"],
                        max_tests=CFG["s1_tests"])
    out["stage1"] = dict(t=time.perf_counter() - tt, roots=int(len(roots)), root_volume=float(len(roots)) / 16 ** 3)
    codes, kls = win.top_items()
    out["all_items"] = int(len(codes))
    B = {}
    for name, (xc, xk) in dict(local=(xc_l, xk_l), random=(xc_r, xk_r)).items():
        tt = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots); t2 = time.perf_counter() - tt
        vol, ncell = _vol(res)
        rec = dict(t_outer=t2, B_volume=vol, B_cells=ncell, B_records=int(len(res["cell"])), outer=_st(res), items=int(len(xc)))
        single = ncell == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0
        rec["path"] = "single_cell_support" if single else "exact_on_B"
        if not single:
            tt = time.perf_counter(); fin = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res)); rec["t_exact"] = time.perf_counter() - tt
            rec["exact"] = _st(fin); rec["F_volume_hi"] = float(np.sum(fin["hi"])); rec["F_volume_lo"] = float(np.sum(fin["lo"]))
            rec["tv_bound"] = float(fin["tv_bound"])
            if name == "local":
                # piece-level oracle: the exact phase restricted to the pieces of its own answer F (outer-bound filter
                # built from the final records: full cells for coarse pieces, boxes for fine ones). No region
                # reduction of any kind can make the exact phase cheaper than this.
                inr = fin["hi"] > 0
                if inr.any():
                    tt = time.perf_counter(); f2 = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(_subset(fin, inr)))
                    rec["oracle_pieces"] = dict(t_exact=time.perf_counter() - tt, exact=_st(f2), filter_volume=float(np.sum(fin["hi"][inr])),
                                                mass_share=float(np.sum(0.5 * (f2["lo"] + f2["hi"])) / max(np.sum(0.5 * (fin["lo"] + fin["hi"])), 1e-300)))
        B[name] = (res, rec); out[f"B_{name}"] = rec

    # ---- 3. local constraints, measured, never applied
    s_vox = s_true * N - o
    H = []
    for c in big:
        li, da = lit[c["idx"]], dark[c["idx"]]
        fr = edge_free_mask(rows, o, N, li, da)
        H.append(dict(n=int(c["n"]), h1_free_share=float(fr.mean()), h1=bool(fr.all()), h2_true_source=_stab_through(li, da, s_vox),
                      h2_true_source_free_edges=_stab_through(li[fr], da[fr], s_vox) if fr.sum() >= 3 else None))
    out["comp_hyp"] = H
    ref = Path(a.labels) / f"{t['key']}_tp.npz"
    if ref.exists():
        fc, fm = _cellmass(ref); fm = fm / max(fm.sum(), 1e-300)
        out["F_cells"] = int(len(fc))
        band = local_band_cells(comps, o, N).reshape(-1)
        out["band"] = dict(volume_share=float(band.mean()), cells=int(band.sum()), mass_outside=float(fm[~band[fc]].sum()),
                           true_source_inside=bool(band[(int(s_true[0] * 128) * 128 + int(s_true[1] * 128)) * 128 + int(s_true[2] * 128)]))
        res, rec = B["local"]
        if rec["path"] == "exact_on_B":
            inb = band[res["cell"]]
            out["band"]["B_volume_inside"] = float(np.sum(res["hi"][inb]) / max(np.sum(res["hi"]), 1e-300))
            out["band"]["B_cells_inside"] = int(len(np.unique(res["cell"][inb])))
            # ---- 4. upper bounds on region reduction (diagnostic exact runs on sub-regions of B)
            for name, keep in (("restrict_oracle", np.isin(res["cell"], fc)), ("restrict_band", inb)):
                if not keep.any():
                    out[name] = dict(empty=True); continue
                sub = _subset(res, keep)
                tt = time.perf_counter(); fin = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(sub)); tr = time.perf_counter() - tt
                u, inv = np.unique(fin["cell"], return_inverse=True)
                m = np.bincount(inv, weights=0.5 * (fin["lo"] + fin["hi"]), minlength=len(u)) if len(u) else np.zeros(0)
                ref_mass = 0.5 * (rec["F_volume_hi"] + rec["F_volume_lo"])
                mass_vs_ref = float(m.sum() / max(ref_mass, 1e-300))
                out[name] = dict(t_exact=tr, B_volume=_vol(sub)[0], B_cells=_vol(sub)[1], exact=_st(fin), mass_share_of_exact_on_B=mass_vs_ref,
                                 tv_bound=float(fin["tv_bound"]))
    print(json.dumps(out, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / "audit.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = sorted([t for t in tasks if t["key"] not in done], key=lambda t: -t["L"])

    def one(t):
        try:
            p = subprocess.run([sys.executable, str(HERE), "one", "--task", json.dumps(t), "--labels", a.labels],
                               capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else dict(key=t["key"], status="error", error=p.stderr[-3000:])
        except subprocess.TimeoutExpired:
            rec = dict(key=t["key"], status="timeout")
        with open(fp, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(rec["key"], rec.get("status", "ok"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def cmd_table(a):
    """Per-task markdown table; stage times from the clean cmp_v3 traces (--traces DIR), counters from the audit."""
    R = {r["key"]: r for r in (json.loads(l) for l in (Path(a.out) / "audit.jsonl").read_text().splitlines() if l.strip()) if "sel" in r}
    tr = {m: {json.loads(l)["key"]: json.loads(l) for l in (Path(a.traces) / f"{m}.jsonl").read_text().splitlines() if l.strip()} for m in ("lf", "rand", "tp")}
    hdr = ["task", "fam", "L", "edges", "comp", "H1 / H2(s*)", "Jacc", "B vol l/r", "B/F vol", "B/F cells", "lf: local | s1 | outer | exact s",
           "rand: s1 | outer | exact s", "comp_evals outer / exact (M)", "oracle-pieces speed-up", "band: B share | F mass out"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for k in sorted(R, key=lambda k: (R[k]["L"], R[k]["family"], k)):
        r = R[k]; bl, br = r["B_local"], r["B_random"]; tl = tr["lf"][k]["meta"]["times"]; trn = tr["rand"][k]["meta"]["times"]
        Fv = 0.5 * (bl["F_volume_hi"] + bl["F_volume_lo"]) if "F_volume_hi" in bl else None
        h = r["comp_hyp"]; h1 = sum(x["h1"] for x in h); h2 = sum(bool(x["h2_true_source"]) for x in h)
        op = bl.get("oracle_pieces"); bd = r.get("band", {})
        print(f"| {k} | {r['family'][:10]} | {r['L']} | {r['edges']} | {r['components']} | {h1}/{len(h)} , {h2}/{len(h)} | {r['sel']['jaccard']:.3f} | "
              f"{bl['B_volume'] / max(br['B_volume'], 1e-300):.2f} | {bl['B_volume'] / Fv if Fv else float('nan'):.1f} | "
              f"{bl['B_cells'] / max(r.get('F_cells', 1), 1):.2f} | {tl.get('local', 0):.2f} / {tl.get('stage1', 0):.2f} / {tl.get('outer', 0):.2f} / {tl.get('exact_on_B', 0):.2f} | "
              f"{trn.get('stage1', 0):.2f} / {trn.get('outer', 0):.2f} / {trn.get('exact_on_B', 0):.2f} | "
              f"{bl['outer']['comp_evals'] / 1e6:.1f} / {bl.get('exact', {}).get('comp_evals', 0) / 1e6:.1f} | "
              f"{(bl['t_exact'] / op['t_exact']) if op else float('nan'):.2f} | {bd.get('B_volume_inside', float('nan')):.2f} , {bd.get('mass_outside', float('nan')):.3f} |")


def cmd_report(a):
    R = [json.loads(l) for l in (Path(a.out) / "audit.jsonl").read_text().splitlines() if l.strip()]
    R = [r for r in R if "sel" in r]
    q = lambda x: (np.percentile(x, 50), np.percentile(x, 90), np.max(x)) if len(x) else (np.nan,) * 3
    print(f"tasks audited: {len(R)}")
    J = [r["sel"]["jaccard"] for r in R]; JI = [r["sel"]["interface_jaccard"] for r in R]
    print("receiver overlap Jaccard p50/p90/max", q(J), "min", min(J), "| interface-receiver Jaccard p50", np.median(JI))
    for k in ("n", "active", "inactive", "interface", "on_components", "components_touched"):
        print(f"  {k:20s} local p50 {np.median([r['sel']['local'][k] for r in R]):8.1f}   random p50 {np.median([r['sel']['random'][k] for r in R]):8.1f}")
    BV = [(r["B_local"]["B_volume"], r["B_random"]["B_volume"]) for r in R]
    rat = [x / y for x, y in BV if y > 0]
    print("B volume ratio local/random p50", np.median(rat), "range", min(rat), max(rat), "| identical B", sum(abs(x - y) <= 1e-15 * max(y, 1e-300) for x, y in BV))
    ex = [r for r in R if r["B_local"]["path"] == "exact_on_B"]
    print("paths: exact_on_B", len(ex), "single_cell_support", len(R) - len(ex))
    for name in ("restrict_oracle", "restrict_band"):
        g = [r for r in ex if name in r and not r[name].get("empty")]
        sp = [r["B_local"]["t_exact"] / max(r[name]["t_exact"], 1e-9) for r in g]
        nd = [r["B_local"]["exact"]["nodes"] / max(r[name]["exact"]["nodes"], 1) for r in g]
        print(f"{name}: n={len(g)} exact-time speed-up p50 {np.median(sp) if sp else np.nan:.2f} max {max(sp) if sp else np.nan:.2f}; node ratio p50 {np.median(nd) if nd else np.nan:.2f}")
    _pieces_report(ex)


def _pieces_report(R):
    g = [r for r in R if r["B_local"].get("oracle_pieces")]
    sp = [r["B_local"]["t_exact"] / max(r["B_local"]["oracle_pieces"]["t_exact"], 1e-9) for r in g]
    print(f"piece-level oracle: n={len(g)} exact-time speed-up p50 {np.median(sp):.2f} p90 {np.percentile(sp, 90):.2f} max {max(sp):.2f}")


def cmd_summary(a):
    """Aggregate statistics for docs/LOCALFIRST_AUDIT.md (JSON to stdout)."""
    R = [r for r in (json.loads(l) for l in (Path(a.out) / "audit.jsonl").read_text().splitlines() if l.strip()) if "sel" in r]
    tr = {m: {json.loads(l)["key"]: json.loads(l) for l in (Path(a.traces) / f"{m}.jsonl").read_text().splitlines() if l.strip()} for m in ("lf", "rand", "tp")}
    P = lambda x: dict(n=len(x), p10=float(np.percentile(x, 10)), p50=float(np.median(x)), p90=float(np.percentile(x, 90)),
                       min=float(np.min(x)), max=float(np.max(x))) if len(x) else None
    out = {"tasks": len(R)}
    out["jaccard"] = P([r["sel"]["jaccard"] for r in R]); out["interface_jaccard"] = P([r["sel"]["interface_jaccard"] for r in R])
    out["intersection"] = P([r["sel"]["intersection"] for r in R])
    for side in ("local", "random"):
        out[f"mix_{side}"] = {k: P([r["sel"][side][k] for r in R]) for k in ("n", "active", "inactive", "interface", "on_components", "components_touched")}
    out["components"] = P([r["components"] for r in R])
    vr = [r["B_local"]["B_volume"] / r["B_random"]["B_volume"] for r in R if r["B_random"]["B_volume"] > 0]
    cr = [r["B_local"]["B_cells"] / r["B_random"]["B_cells"] for r in R]
    out["B_volume_ratio_local_over_random"] = P(vr); out["B_cells_ratio_local_over_random"] = P(cr)
    out["B_volume_local_smaller_2x"] = sum(x < 0.5 for x in vr); out["B_volume_local_larger_2x"] = sum(x > 2 for x in vr)
    for side in ("local", "random"):
        g = [r[f"B_{side}"] for r in R]
        out[f"path_{side}"] = {p_: sum(x["path"] == p_ for x in g) for p_ in ("single_cell_support", "exact_on_B")}
        ex = [x for x in g if x["path"] == "exact_on_B"]
        out[f"counters_{side}"] = {f"outer_{k}": P([x["outer"][k] for x in g]) for k in ("nodes", "comp_evals", "entry_evals", "ambig_evals", "undet", "budget_undet", "overflow")}
        out[f"counters_{side}"].update({f"exact_{k}": P([x["exact"][k] for x in ex]) for k in ("nodes", "comp_evals", "entry_evals", "ambig_evals", "undet", "budget_undet", "overflow")})
        out[f"tv_{side}"] = P([x["tv_bound"] for x in ex])
    both = [r for r in R if r["B_local"]["path"] == "exact_on_B"]
    out["B_over_F_volume"] = P([r["B_local"]["B_volume"] / (0.5 * (r["B_local"]["F_volume_hi"] + r["B_local"]["F_volume_lo"])) for r in both
                                if r["B_local"]["F_volume_hi"] > 0])
    out["B_over_F_cells"] = P([r["B_local"]["B_cells"] / r["F_cells"] for r in R if r.get("F_cells")])
    out["items_X_prime"] = P([r["B_local"]["items"] for r in R]); out["items_all"] = P([r["all_items"] for r in R])
    out["stage1_roots"] = P([r["stage1"]["roots"] for r in R]); out["stage1_root_volume"] = P([r["stage1"]["root_volume"] for r in R])
    # stage times from the clean comparison traces
    for m in ("lf", "rand"):
        T_ = [tr[m][r["key"]]["meta"]["times"] for r in R]; tot = [tr[m][r["key"]]["wall"] for r in R]
        out[f"times_{m}"] = {k: P([t.get(k, 0.0) for t in T_]) for k in ("local", "stage1", "outer", "exact_on_B")}
        out[f"times_{m}"]["wall"] = P(tot)
        out[f"share_{m}"] = {k: float(np.sum([t.get(k, 0.0) for t in T_]) / np.sum(tot)) for k in ("local", "stage1", "outer", "exact_on_B")}
    out["tp_wall"] = P([tr["tp"][r["key"]]["wall"] for r in R])
    # oracles (diagnostic, audit-process timings: ratios within one process)
    for name in ("restrict_oracle", "restrict_band"):
        g = [r for r in both if name in r and not r[name].get("empty")]
        out[name] = dict(speedup=P([r["B_local"]["t_exact"] / r[name]["t_exact"] for r in g]),
                         node_ratio=P([r["B_local"]["exact"]["nodes"] / max(r[name]["exact"]["nodes"], 1) for r in g]),
                         comp_evals_ratio=P([r["B_local"]["exact"]["comp_evals"] / max(r[name]["exact"]["comp_evals"], 1) for r in g]),
                         mass_share=P([r[name]["mass_share_of_exact_on_B"] for r in g]))
    g = [r for r in both if r["B_local"].get("oracle_pieces")]
    out["oracle_pieces"] = dict(speedup=P([r["B_local"]["t_exact"] / r["B_local"]["oracle_pieces"]["t_exact"] for r in g]),
                                node_ratio=P([r["B_local"]["exact"]["nodes"] / max(r["B_local"]["oracle_pieces"]["exact"]["nodes"], 1) for r in g]),
                                comp_evals_ratio=P([r["B_local"]["exact"]["comp_evals"] / max(r["B_local"]["oracle_pieces"]["exact"]["comp_evals"], 1) for r in g]),
                                mass_share=P([r["B_local"]["oracle_pieces"]["mass_share"] for r in g]))
    # the whole-pipeline bound: perfect region reduction removes stage 1 + outer and runs the piece oracle
    pb = []
    for r in g:
        t = tr["lf"][r["key"]]["meta"]["times"]; tot = tr["lf"][r["key"]]["wall"]
        frac = r["B_local"]["oracle_pieces"]["t_exact"] / max(r["B_local"]["t_exact"], 1e-9)
        pb.append(tot / max(tot - t.get("stage1", 0) - t.get("outer", 0) - t.get("local", 0) - t.get("exact_on_B", 0) * (1 - frac), 1e-9))
    out["pipeline_bound_speedup"] = P(pb)
    H = [h for r in R for h in r["comp_hyp"]]
    out["components_total"] = len(H)
    out["comp_h1"] = sum(h["h1"] for h in H); out["comp_h2_true"] = sum(bool(h["h2_true_source"]) for h in H)
    out["comp_h1_and_h2"] = sum(h["h1"] and bool(h["h2_true_source"]) for h in H)
    out["comp_h2_free_edges"] = sum(bool(h["h2_true_source_free_edges"]) for h in H)
    out["tasks_all_components_h2"] = sum(all(bool(h["h2_true_source"]) for h in r["comp_hyp"]) for r in R if r["comp_hyp"])
    out["tasks_with_components"] = sum(bool(r["comp_hyp"]) for r in R)
    out["band"] = dict(volume_share=P([r["band"]["volume_share"] for r in R if "band" in r]),
                       mass_outside=P([r["band"]["mass_outside"] for r in R if "band" in r]),
                       true_source_inside=sum(r["band"]["true_source_inside"] for r in R if "band" in r),
                       B_volume_inside=P([r["band"]["B_volume_inside"] for r in R if "B_volume_inside" in r.get("band", {})]))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "one", "report", "table", "summary"]); ap.add_argument("--traces")
    ap.add_argument("--tasks"); ap.add_argument("--task"); ap.add_argument("--labels"); ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--timeout", type=float, default=2400)
    a = ap.parse_args()
    dict(run=cmd_run, one=cmd_one, report=cmd_report, table=cmd_table, summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    main()
