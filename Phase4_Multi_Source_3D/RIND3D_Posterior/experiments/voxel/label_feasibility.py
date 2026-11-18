"""Can Model B (voxel physics) posteriors be certified at an acceptable cost? Measurements on small windows.

usage: label_feasibility.py --ab DIR --out OUT [--max-tasks 8] [--samples 600]
Uses the Model A/B windows of voxel_ab.py (polytope scenes, sources legal in Model B). Per task (window, channel):
1. Model B observation X_B (GPU exact rendering).
2. Sound outer bound with the EXISTING tau-pencil engine and relaxed geometry (VOXEL_PHYSICS_DESIGN.md Lemma 6):
   inactive receivers need s in the occlusion set of G_V, which lies inside that of G_i (+) [-h, h]^3, itself inside
   the face-offset polytopes a.x + d - h|a|_1 <= 0 (outer); active receivers need s outside the occlusion set of G_V,
   which contains that of the original bodies G_i (inner). B_dark = F(dark receivers; outer), B_lit = F(lit; inner),
   both by stage 2 on a receiver subset (interface + interior receivers), so F_B is inside B_dark and inside B_lit.
3. Monte Carlo (uniform in B_dark): share in B_lit (exact Model A predicate with the inner bodies), and share of
   B_dark ∩ B_lit that is truly feasible in Model B (exact Model B rendering of the window, compared with X_B).
   A statistical certificate in the style of FastLabel needs the latter >= 1 - eps.
4. Certified LOCAL occlusion pruning (Theorem 9): uniform points of the cube minus the window, pruned iff some
   active receiver's segment meets an observed SOLID cell of the window; the true source must never be pruned.
5. Timings: stage 2 runs, one exact Model B verification (= one window rendering)."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); P4 = HERE.parents[3]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P4 / "RIND3D_Dataset100K")); sys.path.insert(0, str(P4 / "RIND3D_Dataset100K" / "tests"))
N = 1024


def offset_rows(P, K, h):
    Q = np.array(P, float).copy()
    for c in range(len(K)):
        a = Q[c, :K[c], :3]; Q[c, :K[c], 3] -= h * np.abs(a).sum(1)
    return Q


def seg_hits(q, X, P, K, tol=0.0):
    """Float Cyrus-Beck test: does the closed segment [q, x] meet some polytope {a.p + d <= tol}? (n,) bool.
    Used only to COUNT ambiguous receivers (step 6), never to decide a label."""
    hit = np.zeros(len(X), bool); D = X - q[None]
    for c in range(len(K)):
        A = P[c, :K[c], :3]; d = P[c, :K[c], 3] - tol
        num = A @ q + d                                              # (k,)  g(t) = num + t den <= 0
        den = D @ A.T                                                # (n, k)
        lo = np.zeros(len(X)); hi = np.ones(len(X)); ok = np.ones(len(X), bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -num[None] / den
        pos = den > 0; neg = den < 0; zer = ~pos & ~neg
        hi = np.minimum(hi, np.where(pos, t, np.inf).min(1)); lo = np.maximum(lo, np.where(neg, t, -np.inf).max(1))
        ok &= ~np.any(zer & (num[None] > 0), 1)
        hit |= ok & (lo <= hi)
    return hit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ab", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tasks", type=int, default=8); ap.add_argument("--samples", type=int, default=600)
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import geometry_rows
    from rind3d_voxel.world import bodies_from_context, build_world, VoxelWorld
    from rind3d_voxel.cuda_dda import GPURenderer
    from rind3d_posterior.fastlabel import engine as T
    from rind3d_posterior.fastlabel.window import LazyWindow2
    from rind3d_posterior.fastlabel.sampler import BSampler
    from voxel_ab import scene_geometry
    fac = BackendFactory(cuda_allowed=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    files = sorted(Path(a.ab).glob("A_v3_*.npz")) + sorted(Path(a.ab).glob("A_knife_*.npz"))
    done = 0
    for p in files:
        if done >= a.max_tasks:
            break
        z = np.load(p); kind, fam, slot = str(z["kind"]), str(z["fam"]), int(z["slot"])
        b, balls, ctx, knife = scene_geometry(kind, fam, slot, fac)
        Wb = build_world(bodies_from_context(ctx, balls), N, "supercover"); G = GPURenderer(Wb)
        P, K = geometry_rows(ctx); Pout = offset_rows(P, K, 1.0 / N)
        g_in = T.Geom(np.asarray(P, float), np.asarray(K)); g_out = T.Geom(Pout, np.asarray(K))
        for wi, (o, L) in enumerate(zip(z["origins"], z["sides"])):
            L = int(L)
            if L != 64 or done >= a.max_tasks:
                continue
            for j, s in enumerate(z["sources"]):
                if not Wb.legal_source(s):
                    continue
                lit, sol = G.window(s, o, L)
                from rind3d_100k.interface import interface_axes
                if sum(interface_axes(lit, sol)) == 0:
                    continue
                cls = np.where(sol, 2, np.where(lit, 1, 0)).transpose(2, 1, 0).astype(np.uint8)       # [x, y, z]
                win = LazyWindow2(cls, np.asarray(o, np.int64), N)
                xc, xk = T.point_items(win, 256, 128, 0)
                rec = dict(scene=f"{kind}:{fam}:{slot}", window=wi, source=j, L=L)
                t = time.perf_counter()
                roots_o, _ = T.stage1(g_out, win); roots_i, _ = T.stage1(g_in, win)
                dk = xk == 0; lk = xk == 1
                Bd = T.stage2(g_out, win, xc[dk], xk[dk], roots_o); Bl = T.stage2(g_in, win, xc[lk], xk[lk], roots_i)
                rec["stage_s"] = time.perf_counter() - t
                rec["B_dark_cells"] = int(len(np.unique(Bd["cell"]))); rec["B_lit_cells"] = int(len(np.unique(Bl["cell"])))
                rec["B_dark_vol"] = float(np.sum(Bd["hi"])); rec["B_lit_vol"] = float(np.sum(Bl["hi"]))
                smp = BSampler(Bd, np.random.default_rng(1), {})
                if smp.ok != "ok":
                    rec["note"] = smp.ok; rows.append(rec); continue
                pts = smp.draw(a.samples)
                cl = xc[lk]                                                 # level-0 item codes: (x << 32) | (y << 16) | z
                Ylit = (np.asarray(o)[None] + np.stack([(cl >> 32) & 0xFFFF, (cl >> 16) & 0xFFFF, cl & 0xFFFF], 1) + .5) / N
                in_lit = np.zeros(len(pts), bool); in_F = np.zeros(len(pts), bool)
                t_ver = 0.0; amb = []
                zz, yy, xx = np.meshgrid(*[np.arange(L)] * 3, indexing="ij")
                Xall = (np.asarray(o)[None] + np.stack([xx, yy, zz], -1).reshape(-1, 3) + .5) / N
                fr = ~sol.ravel(); litr = lit.ravel()
                for i, q in enumerate(pts):
                    if not Wb.legal_source(q):
                        continue
                    y, _, _ = ctx.bind(q).native.points(np.ascontiguousarray(Ylit))
                    in_lit[i] = np.all(y == 1)
                    if in_lit[i]:
                        t1 = time.perf_counter(); l2, _ = G.window(q, o, L); t_ver += time.perf_counter() - t1
                        in_F[i] = np.array_equal(l2, lit)
                        if len(amb) < 40:
                            # 6. receivers the relaxation cannot decide for this candidate: lit x whose segment meets
                            # the OUTER bodies, dark y whose segment misses the INNER bodies (everything else is
                            # decided by Model A predicates on inner / outer geometry; only these need the voxel walk)
                            ho = seg_hits(q, Xall[fr & litr], Pout, K); hi_ = seg_hits(q, Xall[fr & ~litr], np.asarray(P, float), K)
                            amb.append(int(ho.sum() + (~hi_).sum()))
                nl = int(in_lit.sum())
                rec.update(samples=len(pts), share_in_B_lit=nl / len(pts), share_feasible_in_B=(int(in_F.sum()) / nl) if nl else None,
                           verify_s_per_point=(t_ver / nl) if nl else None,
                           ambiguous_receivers_p50=float(np.median(amb)) if amb else None,
                           ambiguous_receivers_max=int(np.max(amb)) if amb else None, free_receivers=int(fr.sum()))
                # certified local occlusion pruning: a world holding only the observed SOLID cells of the window
                Wl = VoxelWorld(N); Wl.add_cells(np.asarray(o)[None] + np.argwhere(sol)[:, ::-1]); Gl = GPURenderer(Wl)
                rng = np.random.default_rng(2); pruned = 0; tested = 0
                for _ in range(300):
                    q = rng.uniform(0, 1, 3)
                    if np.all((q >= np.asarray(o) / N) & (q <= (np.asarray(o) + L) / N)) or not Wl.legal_source(q):
                        continue
                    l3, _ = Gl.window(q, o, L); tested += 1
                    pruned += int(np.any(lit & ~l3))                     # an active receiver blocked by observed SOLID
                l_true, _ = Gl.window(s, o, L)
                rec.update(local_pruned_share=pruned / max(tested, 1), local_tested=tested, true_source_pruned=bool(np.any(lit & ~l_true)),
                           window_solid_cells=int(sol.sum()))
                rows.append(rec); done += 1
                print(json.dumps(rec, default=float), flush=True)
                with open(out / "label_feasibility.jsonl", "a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                if done >= a.max_tasks:
                    break


if __name__ == "__main__":
    main()
