"""FastLabel-S6 (research prototype; production rind3d_1k.fastlabel is NOT modified).

Audit issue S6: for an all-dark observation, a source inside a closed solid sees no receiver and is therefore consistent
with "every FREE receiver dark". Phase 1's outer bound B ⊇ F then contains solid interiors. The verifier rejects
samples there as illegal, with no violated receivers. So the sequential test cannot reach p = vol(F)/vol(B) >= 1 - eps,
and the label falls back to the exact engine, whose bound is loose for the same reason (audit E8/E11).

Fix (statistical path only; everything else is the v3.4 pipeline, imported unchanged):
* B' = B minus the union of the closed solids. F ⊆ B', since F ⊆ D(G) excludes the solids.
* Uniform samples on B' are uniform samples on B, with samples inside a closed solid discarded. The test is the exact
  closed-body predicate context.contains, the one used by legal_source. The accepted sequence is i.i.d. Uniform(B'),
  so the v3.4 two-stage design applies unchanged to p' = vol(F) / vol(B').
* Label q_B': the per-cell volumes of B', computed per piece P as vol(P) − vol(P ∩ ∪C).
  * Pieces fully inside a body: removed.
  * Pieces disjoint from every body (a separating face row): kept.
  * Partial pieces: float64 half-space intersection volumes, with inclusion–exclusion for up to 3 bodies.
  * The float error enters the TV bound as a deterministic term eta (recorded; ~1e-12).
* Certificate: TV(q_B', q_F) <= eps + eta, except with probability delta (same per-label delta policy).

API: label_task_s6(scene_dir, w, s, **cfg), returning the same dict as rind3d_1k.fastlabel.label_task."""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K")
from rind3d_1k.fastlabel import pipeline as PL          # noqa: E402
from rind3d_1k.fastlabel.pipeline import (DEFAULTS, VERSION, STAT_KIND, scene_cache, label_rng, design2, kstar,  # noqa: E402
                                          cp_lower, process_factory)
from rind3d_1k.fastlabel import engine as T              # noqa: E402
from rind3d_1k.fastlabel.filters import build_filter2, items_from_vox2   # noqa: E402
from rind3d_1k.fastlabel.sampler import BSampler          # noqa: E402
from rind3d_1k.fastlabel.window import fast_cls3, LazyWindow2   # noqa: E402

VERSION_S6 = VERSION + "+s6-research"


def _cheb(A, b):
    """Chebyshev centre of {x : A x <= b}; None if empty or degenerate."""
    from scipy.optimize import linprog
    nrm = np.linalg.norm(A, axis=1)
    r = linprog(np.r_[np.zeros(3), -1.0], A_ub=np.c_[A, nrm], b_ub=b, bounds=[(None, None)] * 3 + [(0, None)],
                method="highs")
    if r.status != 0 or r.x[3] <= 1e-14:
        return None
    return r.x[:3]


def _hvol(A, b):
    """Volume of the polytope {A x <= b} (float64)."""
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    c = _cheb(A, b)
    if c is None:
        return 0.0
    hs = HalfspaceIntersection(np.c_[A, -b], c)
    try:
        return float(ConvexHull(hs.intersections).volume)
    except Exception:
        return 0.0


def _piece_hs(V):
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    return h.equations[:, :3], -h.equations[:, 3]


def solid_free_volumes(res, P, K):
    """Remaining volume of every piece after removing the closed solids. Returns (vol, stats)."""
    vol = res["hi"].astype(float).copy()
    voff, vn, vpool = res["voff"], res["vn"], res["vpool"]
    n = len(vol)
    # per-piece vertex bounding boxes, vectorised: one row index per vertex of every piece
    vn64 = vn.astype(np.int64); start = np.repeat(voff.astype(np.int64) - np.cumsum(np.r_[0, vn64[:-1]]), vn64)
    rows = np.arange(int(vn64.sum()), dtype=np.int64) + start
    pid = np.repeat(np.arange(n), vn64)
    VV = vpool[rows]
    lo = np.full((n, 3), np.inf); hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, pid, VV); np.maximum.at(hi, pid, VV)
    st = dict(pieces=n, inside=0, partial=0, multi=0, removed_volume=0.0, unresolved=0, unresolved_volume=0.0)
    touch = [[] for _ in range(n)]
    for bi in range(len(K)):
        rows = P[bi, :K[bi]]; a, d = rows[:, :3], rows[:, 3]
        # body bounding box from its half-space representation is not stored: use the vertex test directly on pieces whose
        # bounding box is not separated by any face row evaluated at the box (conservative prefilter)
        # face row value at the box corner that minimises it: a.x + d minimal over the box
        sep = np.zeros(n, bool)
        for f in range(len(d)):                            # box minimum of each face row, one face at a time
            sep |= (d[f] + np.where(a[f] > 0, lo * a[f], hi * a[f]).sum(1)) >= 0
        cand = np.flatnonzero(~sep)
        for i in cand:
            V = vpool[voff[i]:voff[i] + vn[i]]
            val = V @ a.T + d
            if np.any(np.all(val >= 0, axis=0)):
                continue                                   # separated by a face plane
            touch[i].append((bi, bool(np.all(val <= 0))))
    for i in range(n):
        if not touch[i]:
            continue
        if any(full for _, full in touch[i]):
            st["inside"] += 1; st["removed_volume"] += vol[i]; vol[i] = 0.0; continue
        V = vpool[voff[i]:voff[i] + vn[i]]
        Ap, bp = _piece_hs(V)
        bodies = [bi for bi, _ in touch[i]]
        if len(bodies) > 3:
            raise RuntimeError("piece meets more than 3 bodies (inclusion-exclusion not implemented)")
        st["partial"] += 1; st["multi"] += len(bodies) > 1
        rem = 0.0
        from itertools import combinations
        try:
            for r in range(1, len(bodies) + 1):
                for comb in combinations(bodies, r):
                    A = np.vstack([Ap] + [P[b, :K[b], :3] for b in comb]); bb = np.concatenate([bp] + [-P[b, :K[b], 3] for b in comb])
                    rem += (-1) ** (r + 1) * _hvol(A, bb)
        except Exception:
            # degenerate sliver (qhull cannot place an interior point): keep the whole piece. The label then covers a
            # superset of B' (still ⊇ F); its extra mass is at most vol(piece), added to the deterministic term eta.
            st["unresolved"] += 1; st["unresolved_volume"] += vol[i]; continue
        rem = min(max(rem, 0.0), vol[i])
        st["removed_volume"] += rem; vol[i] -= rem
    return vol, st


def label_task_s6(scene_dir, w, s, **kw):
    from rind3d_1k.fastlabel import gpuverify as GV
    cfg = dict(DEFAULTS); cfg.update(kw)
    eps = float(cfg["eps"])
    delta = float(cfg["delta"]) if cfg["delta"] is not None else float(cfg["family_delta"]) / int(cfg["n_tasks"])
    scene_dir = Path(scene_dir); sc = scene_dir.name
    out = dict(version=VERSION_S6, eps=eps, delta=delta)
    tt = {}
    t0 = time.perf_counter(); c0 = time.process_time()
    scn = scene_cache(scene_dir); S = scn["S"]; geom = scn["geom"]; ctx = scn["ctx"]
    cls = fast_cls3(S, w, s); win = LazyWindow2(cls, np.asarray(S.origin(w), np.int64), int(S.scene["resolution"]))
    seed_items, rng_s, rng_c = label_rng(cfg["base_seed"], int(sc), w, s)
    roots, _ = T.stage1(geom, win, d1=cfg["s1_d1"], n_boundary=cfg["s1_nb"], n_class=cfg["s1_nc"], n_learn=cfg["s1_learn"],
                        max_tests=cfg["s1_tests"])
    xc, xk = T.point_items(win, cfg["nbd"], cfg["nin"], seed_items)
    V = None; bf = None; rounds = []; vpool_n = 1 << 23; res = None; tri_cache = {}
    n_rej = 0; n_drawn = 0
    try:
        for r in range(1, cfg["R"] + 1):
            t = time.perf_counter()
            res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
            rr = dict(round=r, items=len(xc), p1_s=time.perf_counter() - t, records=len(res["cell"]), undet=res["stats"]["undet"])
            rounds.append(rr)
            cells = np.unique(res["cell"])
            if len(cells) == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0:
                out["path"] = "single_cell_support"; out["cell"] = int(cells[0]); break
            if V is None:
                V = GV.WindowVerifier(process_factory(), ctx, S, w, s, cls, lock=PL.GPU_LOCK)
            n_r = int(cfg["n_cert"]); d_r = delta * 0.5 ** r; n1 = int(cfg["n_stage1"] or n_r)
            a1, k_r, alpha_r = design2(n1, n_r, eps, d_r) if n1 < n_r else (-1, kstar(n_r, eps, d_r), None)
            rr.update(n=n_r, n1=n1, delta_r=d_r, a1=a1, kstar=k_r)
            smp = BSampler(res, rng_s, tri_cache)
            while smp.ok.startswith("piece without vertices") and vpool_n < cfg["max_vpool"]:
                vpool_n = int(min(4 * vpool_n, cfg["max_vpool"]))
                res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
                smp = BSampler(res, rng_s, tri_cache)
            if smp.ok != "ok":
                rr["sample"] = smp.ok; out["path"] = "fallback_exact"; break
            K = 0; tested = 0; viol = []; rej_r = 0
            bounds = [int(b) for b in np.cumsum(cfg["chunks"]) if b < n_r] + [n_r]
            stop = False
            for hi in bounds:
                while tested < hi and not stop:
                    need = hi - tested
                    pts = smp.draw(max(need, 32)); n_drawn += len(pts)
                    for p, (ok, vox, why) in zip(pts, V.test_batch(pts)):
                        if tested >= hi:
                            break                          # surplus samples are discarded unseen (order preserved)
                        if not ok and why == "illegal" and bool(ctx.contains(np.asarray(p, "<f8")[None])[0]):
                            rej_r += 1; continue           # inside a closed solid: outside B', not part of the test
                        tested += 1
                        if not ok:
                            K += 1
                            if vox is not None and len(vox):
                                viol.append(vox if len(vox) <= cfg["M"] else vox[rng_c.choice(len(vox), cfg["M"], replace=False)])
                    if rej_r > 200 * n_r:
                        stop = True                        # B' is a tiny part of B: give up (recorded)
                if stop or K > k_r or (tested == n1 and n1 < n_r and K <= a1):
                    break
            n_rej += rej_r
            rr.update(tested=tested, K=K, rejected_inside_solid=rej_r)
            stage = 1 if (n1 < n_r and tested == n1 and K <= a1) else (2 if (tested == n_r and K <= k_r) else 0)
            if stage and not stop:
                out.update(path="stat_certified_s6", round=r, n=tested, K=K, stage=stage, a1=a1, a2=k_r, n1=n1, n_max=n_r,
                           delta_r=d_r, alpha_r=alpha_r, p_lower=cp_lower(tested - K, tested, d_r))
                break
            if stop:
                out["path"] = "fallback_exact"; rr["note"] = "B' negligible within B"; break
            if not viol:
                out["path"] = "fallback_exact"; rr["note"] = "failures without violated receivers"; break
            v = np.unique(np.concatenate(viol), axis=0)
            if len(v) > cfg["M_round"]:
                v = v[rng_c.choice(len(v), cfg["M_round"], replace=False)]
            nc, nk = items_from_vox2(win, v)
            xc = np.concatenate([xc, nc]); xk = np.concatenate([xk, nk])
            bf = build_filter2(res)
        else:
            out["path"] = "fallback_exact"
    finally:
        if V is not None:
            out["verifier"] = {k: (round(x, 5) if isinstance(x, float) else x) for k, x in V.stats.items()}
            V.close()
    out["rounds"] = rounds; out["rejected_inside_solid"] = n_rej; out["drawn"] = n_drawn
    if out["path"] == "stat_certified_s6":
        t = time.perf_counter()
        vol, st = solid_free_volumes(res, np.asarray(S.P, float), np.asarray(S.K))
        out["solid_removal"] = dict(st, seconds=time.perf_counter() - t)
        keep = vol > 0
        cm = T.cell_masses(res["cell"][keep], vol[keep], vol[keep])
        cell = np.array(list(cm), np.int64); v = np.array(list(cm.values()))
        # deterministic extra TV: float64 volume error of partial pieces (generous 1e-12 relative) + kept degenerate slivers
        eta = (1e-12 * st["partial"] * float(np.max(res["hi"])) + st["unresolved_volume"]) / max(float(v.sum() - st["unresolved_volume"]), 1e-300)
        lab = dict(cell=cell.astype(np.int32), lo=v, hi=v.copy(), kind=np.full(len(cell), STAT_KIND, np.int8),
                   tv_bound=eps + eta, status="certified_stat")
        out["certificate"] = dict(kind="statistical", target="Uniform(F_W^j) cell masses", label="q_B' (B' = B minus closed solids ⊇ F)",
                                  eps=eps, eta_float_volume=eta, delta_label=delta, round=out["round"], stage=out["stage"],
                                  n=out["n"], K=out["K"], a1=out["a1"], a2=out["a2"], alpha_round=out["alpha_r"],
                                  p_lower=out["p_lower"], seed=[int(cfg["base_seed"]), int(sc), int(w), int(s)],
                                  n_tasks=int(cfg["n_tasks"]), note="research prototype (S6); not a production label")
    elif out["path"] == "single_cell_support":
        lab = dict(cell=np.array([out["cell"]], np.int32), lo=np.array([1.0]), hi=np.array([1.0]),
                   kind=np.zeros(1, np.int8), tv_bound=0.0, status="certified")
        out["certificate"] = dict(kind="support")
    else:
        lab = dict(cell=np.zeros(0, np.int32), lo=np.zeros(0), hi=np.zeros(0), kind=np.zeros(0, np.int8), tv_bound=1.0,
                   status="unresolved")
        out["certificate"] = dict(kind="none", note="S6 prototype has no exact fallback; recorded as unresolved")
    out["n_cells"] = int(len(np.unique(lab["cell"])))
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    lab["meta"] = out
    return lab
