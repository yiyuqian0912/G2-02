"""E3: the exact X-only local source region of Model B (VOXEL_PHYSICS_DESIGN.md Theorem 10) on real windows.

usage: e3_local.py --ab DIR --out OUT [--windows 16] [--uniform 400] [--dirs 12]
Uses the Model A/B scenes of voxel_ab.py (Model B worlds rebuilt from the same generative geometry). Per selected
window and driver source (legal in Model B, window with a direct interface):
* the observation X_B = (observed SOLID cells, binary responses) is rendered once (GPU, exact);
* L_B(X_B) is evaluated with `rind3d_voxel.local_region.LocalRegionGPU`, which reads ONLY X_B, the window origin and
  side (and the declared physics). The hidden world and the true source are used only to render X_B and, afterwards,
  as checks;
* uniform candidates in the cube: shares outside the domain (window box / illegal w.r.t. observed SOLID), pruned by
  (a) (an active segment touches observed SOLID), pruned by (b) (an unexplained inactive receiver has no admissible
  blocker), kept;
* safety: the true source and feasible sources of the true world (perturbations of the source that reproduce X_B
  exactly) must be kept;
* local structure: along random directions through the true source, the extent of L_B and of F_B (first exit by
  bisection, scales 1e-6 .. 0.5), i.e. how much thicker the X-only region is than the fixed-world feasible set."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); P4 = HERE.parents[3]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(P4 / "RIND3D_Dataset100K")); sys.path.insert(0, str(P4 / "RIND3D_Dataset100K" / "tests"))
N = 1024


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ab", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=16); ap.add_argument("--uniform", type=int, default=400); ap.add_argument("--dirs", type=int, default=12)
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_voxel.world import bodies_from_context, build_world
    from rind3d_voxel.cuda_dda import GPURenderer
    from rind3d_voxel.local_region import LocalRegionGPU
    from rind3d_100k.interface import interface_axes
    from voxel_ab import scene_geometry
    fac = BackendFactory(cuda_allowed=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fp = out / "e3.jsonl"
    done = {(json.loads(l)["scene"], json.loads(l)["window"], json.loads(l)["source"]) for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    files = sorted(Path(a.ab).glob("A_*.npz"))
    per_scene = max(1, a.windows // len(files) + 1)
    n_done = 0
    for p in files:
        z = np.load(p); kind, fam, slot = str(z["kind"]), str(z["fam"]), int(z["slot"])
        b, balls, ctx, knife = scene_geometry(kind, fam, slot, fac)
        Wb = build_world(bodies_from_context(ctx, balls), N, "supercover"); G = GPURenderer(Wb)
        taken = 0
        for wi, (o, L) in enumerate(zip(z["origins"], z["sides"])):
            L = int(L)
            if L != 64 or taken >= per_scene:
                continue
            for j, s in enumerate(z["sources"]):
                key = (f"{kind}:{fam}:{slot}", wi, j)
                if key in done or not Wb.legal_source(s) or taken >= per_scene:
                    continue
                lit, sol = G.window(s, o, L)
                if sum(interface_axes(lit, sol)) == 0:
                    continue
                t0 = time.perf_counter()
                R = LocalRegionGPU(N, o, L, sol, lit)                  # reads the observation only
                t_setup = time.perf_counter() - t0
                rng = np.random.default_rng([slot, wi, j, 33])
                U = rng.uniform(0, 1, (a.uniform, 3)); cnt = dict(domain=0, a=0, b=0, kept=0); ts = []; deg = 0
                for q in U:
                    r = R.test(q); ts.append(r["seconds"]); deg += r.get("degenerate", 0)
                    cnt[r["by"] if r["pruned"] else "kept"] += 1
                rec = dict(scene=key[0], window=wi, source=j, L=L, setup_s=t_setup, test_s_p50=float(np.median(ts)), test_s_max=float(np.max(ts)),
                           uniform=a.uniform, degenerate_rays=int(deg), solid_cells=int(sol.sum()), lit=int((lit & ~sol).sum()),
                           **{f"share_{k}": v / a.uniform for k, v in cnt.items()})
                # safety: the true source, and fixed-world feasible sources near it
                rec["true_source_kept"] = not R.test(s)["pruned"]
                feas_kept = feas = 0
                for sc in (1e-6, 1e-5, 1e-4, 1e-3):
                    for q in s + rng.normal(0, sc, (6, 3)):
                        if not Wb.legal_source(q) or np.all((q >= o / N) & (q <= (o + L) / N)):
                            continue
                        l2, _ = G.window(q, o, L)
                        if np.array_equal(l2, lit):
                            feas += 1; feas_kept += int(not R.test(q)["pruned"])
                rec["feasible_tested"] = feas; rec["feasible_kept"] = feas_kept
                if feas_kept < feas or not rec["true_source_kept"]:
                    rec["ALARM"] = "a fixed-world feasible source was pruned by the X-only region"
                # local extent along random directions (first exit, bisection over log-scales)
                ext_L, ext_F = [], []
                dirs = rng.normal(0, 1, (a.dirs, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
                for d in dirs:
                    for kind_, test in (("L", lambda q: not R.test(q)["pruned"]),
                                        ("F", lambda q: Wb.legal_source(q) and not np.all((q >= o / N) & (q <= (o + L) / N)) and np.array_equal(G.window(q, o, L)[0], lit))):
                        lo_, hi_ = 0.0, None
                        for t in np.geomspace(1e-6, 0.5, 20):
                            q = s + t * d
                            if not np.all((q > 0) & (q < 1)) or not test(q):
                                hi_ = t; break
                            lo_ = t
                        if hi_ is None:
                            e = 0.5
                        else:
                            for _ in range(10):
                                m = 0.5 * (lo_ + hi_); q = s + m * d
                                if np.all((q > 0) & (q < 1)) and test(q):
                                    lo_ = m
                                else:
                                    hi_ = m
                            e = lo_
                        (ext_L if kind_ == "L" else ext_F).append(e)
                rec["extent_L_p50"] = float(np.median(ext_L)); rec["extent_F_p50"] = float(np.median(ext_F))
                rec["extent_L"] = [float(x) for x in ext_L]; rec["extent_F"] = [float(x) for x in ext_F]
                rec["seconds"] = time.perf_counter() - t0
                print(json.dumps({k: v for k, v in rec.items() if not k.startswith("extent_") or k.endswith("p50")}, default=float), flush=True)
                with open(fp, "a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                taken += 1; n_done += 1


if __name__ == "__main__":
    main()
