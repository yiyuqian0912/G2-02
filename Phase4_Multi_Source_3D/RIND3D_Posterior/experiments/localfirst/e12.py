"""E1 (information of certified interface regions) and E2 (computational value) on the 54 cmp_v3 tasks
(docs/LOCALFIRST_E123.md). Regions are built with Proposition A' only; the true source and the reference labels are used
ONLY as after-the-fact checks.

usage:
  e12.py run    --tasks tasks.json --labels DIR/labels --out OUT [--workers 4] [--timeout 3600]
  e12.py one    --task JSON --labels DIR/labels
  e12.py report --out OUT

Per task, one fresh process, fixed seeds:
* baseline: stage 1 roots, outer bound B (local-first receiver choice), exact phase on B (gives F's pieces);
* E1, for selection in {independent, random} x |I| in {8, 32, 128, 512} and |I| = all eligible edges (capped at
  ALL_CAP by a uniform subsample; the cap and the eligible count are recorded):
  certified cover at depth 7 over the whole cube (cells, 26-connected components, box tests, time), Monte Carlo:
  Vol(R) (uniform in the cover), Vol(R n roots), Vol(R n B)/Vol(B) (uniform in B), F inclusion (uniform in F's exact
  pieces: every sample must lie in R, else ALARM), true source in R;
* E2, for (independent, 128) and (all): cover refined to depth 10 inside B's cells; exact phase restricted to
  B n cover (outer-bound filter) - time, engine counters, mass relative to the unrestricted exact phase, tv_bound; plus
  the FastLabel acceptance ratios Vol(F)/Vol(B) and Vol(F)/Vol(B n R)."""
import argparse, json, subprocess, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
D100K = HERE.parents[3] / "RIND3D_Dataset100K"
SIZES = (8, 32, 128, 512)
ALL_CAP = 2048
TEST_BUDGET = 400_000_000   # box tests per region build (beyond it boxes are kept: sound, recorded as budget_stops)
E2_BUDGET = 100_000_000     # box tests for the E2 depth-10 refinement inside B (same rule)
MC_R, MC_B, MC_F = 20000, 20000, 4000
VOL_CELLS = 400            # depth-7 cover cells refined to depth 10 for the stratified volume estimate


def vmhwm_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1]) / 1024
    return None


def _fps(P, m, rng):
    if len(P) <= m:
        return np.arange(len(P))
    first = int(rng.integers(len(P))); ch = [first]; d = np.linalg.norm(P - P[first], axis=1)
    for _ in range(m - 1):
        j = int(np.argmax(d)); ch.append(j); d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return np.array(ch)


def select_edges(lit, dark, free, comps, n, mode, rng):
    """Indices (into the interface edge arrays) of n H1-free edges. 'independent': components ordered greedily to
    maximise the smallest singular value of their stacked normals, edges taken round-robin over the components in
    farthest-point order (spread), remaining budget filled at random; 'random': uniform among the free edges."""
    elig = np.flatnonzero(free)
    if n >= len(elig):
        return elig
    if mode == "random":
        return np.sort(rng.choice(elig, n, replace=False))
    big = [c for c in comps if c["n"] >= 24]
    order, rem = [], list(range(len(big)))
    while rem:
        def score(j):
            Nm = np.array([big[i]["normal"] for i in order + [j]])
            s = np.linalg.svd(Nm, compute_uv=False)
            return (s[min(len(Nm), 3) - 1], big[j]["n"])
        b = max(rem, key=score); order.append(b); rem.remove(b)
    mid = 0.5 * (lit + dark).astype(float)
    pools = []
    for j in order:
        ids = np.asarray(big[j]["idx"]); ids = ids[free[ids]]
        if len(ids):
            pools.append(ids[_fps(mid[ids], min(len(ids), n), rng)])
    chosen, seen, r = [], set(), 0
    while len(chosen) < n and pools and any(r < len(p) for p in pools):
        for p in pools:
            if r < len(p) and p[r] not in seen and len(chosen) < n:
                chosen.append(int(p[r])); seen.add(int(p[r]))
        r += 1
    if len(chosen) < n:
        rest = np.setdiff1d(elig, np.array(chosen, np.int64))
        chosen += rng.choice(rest, min(n - len(chosen), len(rest)), replace=False).tolist()
    return np.sort(np.array(chosen, np.int64))


def _st(res):
    keys = ("nodes", "out", "in", "undet", "budget_undet", "overflow", "passes", "entry_evals", "comp_evals", "ambig_evals", "expansions")
    return {k: int(res["stats"].get(k, 0)) for k in keys}


def _subset(res, keep):
    keep = np.asarray(keep, bool)
    return dict(cell=res["cell"][keep], lo=res["lo"][keep], hi=res["hi"][keep], kind=res["kind"][keep], voff=res["voff"][keep],
                vn=res["vn"][keep], vpool=res["vpool"])


def _filter_from_leaves(lo, dep, keep_cells=None, G=128):
    """Outer-bound filter (rind3d_posterior.fastlabel.filters format) from cover leaves.
    * leaves at depth >= 7 (inside one 128^3 cell): kind-1 cells listing the leaf boxes; a depth-7 leaf makes its cell
      full (kind 2);
    * leaves coarser than a cell (depth < 7, e.g. from a test-budget stop): EVERY cell they cover becomes full.
    keep_cells (e.g. B's cells) restricts the filter; sound whenever F is inside those cells."""
    lo = np.asarray(lo, float); dep = np.asarray(dep, np.int64); h = 1.0 / (1 << dep)
    coarse = dep < 7
    full_cells = []
    for q in np.flatnonzero(coarse):
        n = 1 << (7 - dep[q]); g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        ijk = np.floor(lo[q] * G + 1e-9).astype(np.int64) + g
        full_cells.append((ijk[:, 0] * G + ijk[:, 1]) * G + ijk[:, 2])
    full_cells = np.unique(np.concatenate(full_cells)) if full_cells else np.zeros(0, np.int64)
    lo, h, dep = lo[~coarse], h[~coarse], dep[~coarse]
    ijk = np.floor(lo * G + 1e-9).astype(np.int64); cell = (ijk[:, 0] * G + ijk[:, 1]) * G + ijk[:, 2]
    full_cells = np.union1d(full_cells, cell[dep == 7])
    if keep_cells is not None:
        m = np.isin(cell, keep_cells); lo, h, cell, dep = lo[m], h[m], cell[m], dep[m]
        full_cells = np.intersect1d(full_cells, keep_cells)
    fine = ~np.isin(cell, full_cells)
    lo, h, cell = lo[fine], h[fine], cell[fine]
    order = np.argsort(cell, kind="stable"); lo, h, cell = lo[order], h[order], cell[order]
    fc, fstart, fln = np.unique(cell, return_index=True, return_counts=True)
    cells = np.union1d(fc, full_cells)
    kind = np.where(np.isin(cells, full_cells), 2, 1).astype(np.int8)
    start = np.zeros(len(cells), np.int64); ln = np.zeros(len(cells), np.int64)
    pos = np.searchsorted(cells, fc); start[pos] = fstart; ln[pos] = fln
    boxes = np.concatenate([lo - 1e-13, lo + h[:, None] + 1e-13], axis=1) if len(lo) else np.zeros((1, 6))
    return (cells.astype(np.int64), kind, start, ln, np.ascontiguousarray(boxes, float))


def cmd_one(a):
    sys.path.insert(0, str(D100K))
    from scipy import ndimage
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.fastlabel import engine as T, window_from_100k
    from rind3d_posterior.fastlabel.window import window_classes, LazyWindow2
    from rind3d_posterior.fastlabel.filters import build_filter2, items_from_vox2
    from rind3d_posterior.fastlabel.sampler import BSampler
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, select_receivers, edge_free_mask
    from rind3d_posterior.localfirst.pipeline import DEFAULTS as CFG
    from rind3d_posterior.localfirst import certregion as CR
    t = json.loads(a.task); out = dict(key=t["key"], family=t["family"], L=t["L"])
    T0 = time.perf_counter()
    task = window_from_100k(t["container"], t["w"], t["s"])
    S = SceneFile(t["container"]); s_true = S.sources()[t["s"]]                      # diagnostics only
    ctx = BackendFactory(cuda_allowed=False).context(S.geometry(), POLICY); rows = [c.rows for c in ctx.components]
    geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
    cls = window_classes(task); o = np.asarray(task.origin, np.int64); N = int(task.N); L = int(task.L)
    win = LazyWindow2(cls, o, N); wlo, whi = o / N, (o + L) / N
    rng0 = np.random.default_rng([20261118, zlib.crc32(t["key"].encode())])
    # ---- baseline (the lf pathway)
    tt = time.perf_counter()
    lit, dark = interface_edges(cls); comps = plane_components(lit, dark, seed=CFG["seed"])
    free = edge_free_mask(rows, o, N, lit, dark)
    out["t_h1"] = time.perf_counter() - tt
    out["edges"] = int(len(lit)); out["free_edges"] = int(free.sum())
    tt = time.perf_counter(); roots, _ = T.stage1(geom, win, d1=CFG["s1_d1"], n_boundary=CFG["s1_nb"], n_class=CFG["s1_nc"],
                                                   n_learn=CFG["s1_learn"], max_tests=CFG["s1_tests"]); out["t_stage1"] = time.perf_counter() - tt
    vox, _ = select_receivers(cls, CFG["nbd"], CFG["nin"], seed=CFG["seed"], comps=comps); xc, xk = items_from_vox2(win, vox)
    tt = time.perf_counter(); Bres = T.stage2(geom, win, xc, xk, roots); out["t_outer"] = time.perf_counter() - tt
    codes, kls = win.top_items()
    Bcells = np.unique(Bres["cell"]); vol_B = float(np.sum(Bres["hi"]))
    out.update(roots=int(len(roots)), vol_roots=len(roots) / 4096.0, vol_B=vol_B, B_cells=int(len(Bcells)))
    single = len(Bcells) == 1 and Bres["stats"]["undet"] == 0 and len(Bres["cell"]) > 0
    out["path"] = "single_cell_support" if single else "exact_on_B"
    fin = None
    if not single:
        # warm-up of the filtered engine on one cell (JIT/cache load), then the timed baseline
        T.stage2(geom, win, codes, kls, roots, bf=build_filter2(_subset(Bres, Bres["cell"] == Bcells[0])))
        tt = time.perf_counter(); fin = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(Bres)); out["t_exact_B"] = time.perf_counter() - tt
        out["exact_B"] = _st(fin); out["tv_B"] = float(fin["tv_bound"])
        vol_F = float(np.sum(0.5 * (fin["lo"] + fin["hi"])))
        out["vol_F"] = vol_F
    Fcells = None
    ref = Path(a.labels) / f"{t['key']}_tp.npz"
    if ref.exists():
        Fcells = np.unique(np.load(ref)["cell"])
    # samplers (exact pieces): B uniform, F uniform (IN pieces of the exact answer)
    Bpts = BSampler(Bres, np.random.default_rng(1), {}).draw(MC_B) if vol_B > 0 else np.zeros((0, 3))
    Fpts = np.zeros((0, 3))
    if fin is not None:
        inr = (fin["lo"] > 0) & (fin["kind"] == 0)
        if inr.any():
            smp = BSampler(_subset(fin, inr), np.random.default_rng(2), {})
            if smp.ok == "ok":
                Fpts = smp.draw(MC_F)
    # solid volume and the legal domain (Monte Carlo, exact containment)
    U = np.random.default_rng(3).uniform(0, 1, (100000, 3))
    vol_solid = float(np.mean(ctx.contains(np.ascontiguousarray(U)))); vol_W = float(np.prod(whi - wlo))
    out["vol_domain"] = 1.0 - vol_W - vol_solid
    # ---- occluder edges and candidate explanations
    tt = time.perf_counter(); P_raw = np.asarray(task.P, float); EA, EB, Fr, _ = CR.world_edges(P_raw, np.asarray(task.K))
    out["t_world_edges"] = time.perf_counter() - tt; out["world_edges"] = int(len(EA))
    Lp_all = (o + lit + .5) / N; Kp_all = (o + dark + .5) / N
    rootcell = np.zeros((16, 16, 16), bool); rootcell[roots[:, 1], roots[:, 2], roots[:, 3]] = True
    E1 = []
    configs = [(m, n) for m in ("independent", "random") for n in SIZES] + [("all", ALL_CAP)]
    keep_regions = {}
    for mode, n in configs:
        rng = np.random.default_rng([20261118, zlib.crc32(t["key"].encode()), zlib.crc32(mode.encode()), n])
        if mode == "all":
            elig = np.flatnonzero(free)
            I = elig if len(elig) <= ALL_CAP else np.sort(rng.choice(elig, ALL_CAP, replace=False))
        else:
            I = select_edges(lit, dark, free, comps, n, mode, rng)
        rec = dict(mode=mode, n=int(n), used=int(len(I)), eligible=int(free.sum()))
        if not len(I):
            rec["empty"] = True; E1.append(rec); continue
        Lp, Kp = Lp_all[I], Kp_all[I]
        tt = time.perf_counter(); off, idx = CR.admissible_edges(Fr, Lp, Kp); rec["t_admissible"] = time.perf_counter() - tt
        cnt = np.diff(off); rec["cand_p50"] = float(np.median(cnt)); rec["cand_max"] = int(cnt.max()); rec["cand_zero"] = int((cnt == 0).sum())
        if rec["cand_zero"]:
            rec["ALARM"] = "a free interface edge with no admissible explanation (claims F empty)"
        tt = time.perf_counter(); reg = CR.build_region(EA, EB, Lp, Kp, off, idx, wlo, whi, D=7, max_tests=TEST_BUDGET); rec["t_build"] = time.perf_counter() - tt
        rec["octree"] = reg["stats"]
        cells = CR.leaf_cells(reg["lo"], reg["dep"]); rec["cells"] = int(len(cells))
        g = np.zeros(128 ** 3, bool); g[cells] = True
        rec["components"] = int(ndimage.label(g.reshape(128, 128, 128), structure=np.ones((3, 3, 3)))[1]) if len(cells) else 0
        rec["cover_volume"] = len(cells) / 128.0 ** 3
        # stratified volume: refine a random subset of the cover cells to depth 10, Monte Carlo inside the refined
        # boxes; Vol(R) ~ (#cells / #refined) * sum over refined cells of Vol(R n cell). The depth-10 cover volume
        # of the refined cells, scaled the same way, is a certified upper bound estimate.
        if len(cells):
            rs = np.random.default_rng(6)
            sub = cells if len(cells) <= VOL_CELLS else rs.choice(cells, VOL_CELLS, replace=False)
            tt = time.perf_counter()
            r10 = CR.build_region(EA, EB, Lp, Kp, off, idx, wlo, whi, D=10, target_cells=sub, max_tests=TEST_BUDGET)
            rec["refine10"] = r10["stats"]
            h10 = 1.0 / (1 << r10["dep"]); v10 = float(np.sum(h10 ** 3))
            scale = len(cells) / len(sub)
            rec["cover10_volume_est"] = v10 * scale; rec["refined_cells"] = int(len(sub)); rec["leaves10"] = int(len(r10["lo"]))
            if len(r10["lo"]):
                w = h10 ** 3 / v10; pick = rs.choice(len(r10["lo"]), MC_R, p=w)
                pts = r10["lo"][pick] + rs.uniform(0, 1, (MC_R, 3)) * h10[pick, None]
                inW = np.all((pts >= wlo) & (pts <= whi), 1)
                hit = CR.in_region_batch(pts, EA, EB, Lp, Kp, off, idx, CR.MARGIN) & ~inW
                rec["vol_R_strat"] = float(hit.mean() * v10 * scale)
                ijk7 = np.minimum((pts * 128).astype(np.int64), 127)
                rt = rootcell[ijk7[:, 0] // 8, ijk7[:, 1] // 8, ijk7[:, 2] // 8]
                rec["vol_R_roots_strat"] = float((hit & rt).mean() * v10 * scale)
            else:
                rec["vol_R_strat"] = rec["vol_R_roots_strat"] = 0.0
            rec["t_volume"] = time.perf_counter() - tt
        # Monte Carlo inside the depth-7 cover (uniform over its cells)
        if len(cells):
            ci = cells[np.random.default_rng(4).integers(len(cells), size=MC_R)]
            ijk = np.stack([ci // (128 * 128), (ci // 128) % 128, ci % 128], 1)
            pts = (ijk + np.random.default_rng(5).uniform(0, 1, (MC_R, 3))) / 128.0
            inW = np.all((pts >= wlo) & (pts <= whi), 1)
            tt = time.perf_counter(); hit = CR.in_region_batch(pts, EA, EB, Lp, Kp, off, idx, CR.MARGIN) & ~inW; rec["t_mc"] = time.perf_counter() - tt
            rec["vol_R"] = float(hit.mean() * rec["cover_volume"])
            rt = rootcell[ijk[:, 0] // 8, ijk[:, 1] // 8, ijk[:, 2] // 8]
            rec["vol_R_roots"] = float((hit & rt).mean() * rec["cover_volume"])
        else:
            rec["vol_R"] = rec["vol_R_roots"] = 0.0
        if len(Bpts):
            hb = CR.in_region_batch(Bpts, EA, EB, Lp, Kp, off, idx, CR.MARGIN)
            rec["B_share_in_R"] = float(hb.mean()); rec["B_samples"] = int(len(Bpts))
            rec["B_share_in_R_ub95"] = float((hb.sum() + 3.0) / len(Bpts)) if hb.sum() == 0 else None
        if len(Fpts):
            fin_R = CR.in_region_batch(Fpts, EA, EB, Lp, Kp, off, idx, CR.MARGIN)
            rec["F_samples"] = int(len(Fpts)); rec["F_outside_R"] = int((~fin_R).sum())
            Fij = np.minimum((Fpts * 128).astype(np.int64), 127); Fc = (Fij[:, 0] * 128 + Fij[:, 1]) * 128 + Fij[:, 2]
            rec["F_outside_cover"] = int((~np.isin(Fc, cells)).sum())
            if rec["F_outside_R"] or rec["F_outside_cover"]:
                rec["ALARM"] = "feasible source outside the certified region"
        rec["true_source_in_R"] = bool(CR.in_region_batch(s_true[None], EA, EB, Lp, Kp, off, idx, CR.MARGIN)[0])
        if Fcells is not None:
            rec["F_cells_outside_cover"] = int((~np.isin(Fcells, cells)).sum())
        E1.append(rec)
        if (mode, n) in (("independent", 128), ("all", ALL_CAP)):
            keep_regions[f"{mode}_{n}"] = (Lp, Kp, off, idx, cells)
    out["E1"] = E1
    # ---- E2: exact phase restricted to B n R (cover refined to depth 10 inside B's cells)
    E2 = []
    if fin is not None:
        for name, (Lp, Kp, off, idx, cells) in keep_regions.items():
            rc = np.intersect1d(Bcells, cells)
            rec = dict(region=name, B_cells_in_cover=int(len(rc)))
            if not len(rc):
                rec["empty"] = True; E2.append(rec); continue
            tt = time.perf_counter()
            reg = CR.build_region(EA, EB, Lp, Kp, off, idx, wlo, whi, D=10, target_cells=rc, max_tests=E2_BUDGET)
            rec["t_refine"] = time.perf_counter() - tt; rec["leaves"] = int(len(reg["lo"])); rec["refine"] = reg["stats"]
            rec["cover10_volume"] = float(np.sum((1.0 / (1 << reg["dep"])) ** 3))
            bf = _filter_from_leaves(reg["lo"], reg["dep"], keep_cells=Bcells)
            tt = time.perf_counter(); f2 = T.stage2(geom, win, codes, kls, roots, bf=bf); rec["t_exact"] = time.perf_counter() - tt
            rec["exact"] = _st(f2); rec["tv"] = float(f2["tv_bound"])
            rec["mass_share"] = float(np.sum(0.5 * (f2["lo"] + f2["hi"])) / max(out["vol_F"], 1e-300))
            rec["speedup_exact"] = out["t_exact_B"] / max(rec["t_exact"], 1e-9)
            rec["speedup_with_construction"] = out["t_exact_B"] / max(rec["t_exact"] + rec["t_refine"], 1e-9)
            E2.append(rec)
    out["E2"] = E2
    out["t_total"] = time.perf_counter() - T0; out["rss_mb"] = vmhwm_mb()
    print(json.dumps(out, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / "e12.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = sorted([t for t in tasks if t["key"] not in done], key=lambda t: -t["L"])
    if a.only:
        todo = [t for t in todo if t["key"] in a.only.split(",")]

    def one(t):
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(HERE), "one", "--task", json.dumps(t), "--labels", a.labels],
                               capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else dict(key=t["key"], family=t["family"], L=t["L"], status="error", error=p.stderr[-3000:])
        except subprocess.TimeoutExpired:
            rec = dict(key=t["key"], family=t["family"], L=t["L"], status="timeout", timeout=a.timeout)
        rec["outer_wall"] = time.perf_counter() - t0
        with open(fp, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(rec["key"], rec.get("status", "ok"), round(rec["outer_wall"], 1), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--tasks"); ap.add_argument("--task"); ap.add_argument("--labels"); ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--timeout", type=float, default=3600); ap.add_argument("--only")
    a = ap.parse_args()
    dict(run=cmd_run, one=cmd_one)[a.cmd](a)


if __name__ == "__main__":
    main()
