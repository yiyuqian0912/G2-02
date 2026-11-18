"""Model A (continuous physics) versus Model B (authoritative voxel physics) on identical generative scenes.

usage:
  voxel_ab.py a --out DIR      # process 1: geometry, sources, windows, Model A crops (CPU certified predicates; no CUDA)
  voxel_ab.py b --out DIR      # process 2: Model B worlds and crops (rind3d_voxel; CuPy GPU), metrics
  voxel_ab.py report --out DIR
The two models run in separate processes (the repository rule for incompatible CUDA runtimes; Model A here is CPU).

Scenes (fixed seeds): v3 polytope families (thin slabs >= 2 voxels), v0.1 sub-voxel thin slabs, the explicit
knife-edge wedge of LOCAL_INVERSE_GEOMETRY.md Theorem 3', and the curved research families (balls). Per scene: two
driver sources legal in Model A, four windows (three L64, one L128) chosen where Model A shows a direct interface for
some channel; Model B uses the same sources (when also legal in B), windows and receivers.
Metrics: occupancy change by voxelization (supercover vs centre sampling = Model A's SOLID samples), SOLID and response
differences in the windows, channel classes, direct-interface edges, non-free interface edges (A; zero in B by
VOXEL_PHYSICS_DESIGN.md Proposition 2), whether a plane through the true source crosses the largest planar component
(Proposition 4's hypothesis H2), source-legality disagreement, and timings."""
import argparse, json, sys, time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
P4 = HERE.parents[3]
sys.path.insert(0, str(P4 / "RIND3D_Dataset100K")); sys.path.insert(0, str(P4 / "RIND3D_Dataset100K" / "tests"))
N = 1024

SCENES = ([("v3", fam, slot) for fam in ("rotated_boxes", "thin_slabs", "wedges", "convex_polyhedra", "multiscale_convex", "occluder_clusters")
           for slot in (0, 1)] + [("v01thin", "thin_slabs", slot) for slot in (0, 1)] + [("knife", "wedges", 121)]
          + [("curved", fam, slot) for fam in ("spheres", "spheres_polyhedra") for slot in (0, 1)])


def scene_geometry(kind, fam, slot, fac):
    """(polytope bundle or None, balls (n, 4), context or None) — deterministic from the declared parameters."""
    from rind3d_1k.physical import POLICY
    balls = np.zeros((0, 4))
    if kind in ("v3", "v01thin"):
        from rind3d_100k.geometry_v3 import generate_v3
        cfgf = ["rotated_boxes", "thin_slabs", "prisms", "wedges", "convex_polyhedra", "multiscale_convex", "corridors", "barriers", "occluder_clusters"]
        s = cfgf.index(fam) + 9 * slot
        b, rec = generate_v3(20261118, s, fam, 0, thin_min_vox=2 if kind == "v3" else 0)
    elif kind == "knife":
        from test_local_geometry_v3 import knife_scene
        k = knife_scene(fac); return None, balls, k["ctx"], k
    else:
        from rind3d_curved.families import generate_curved
        b, balls, rec = generate_curved(20261119, slot * 2 + ("spheres", "spheres_polyhedra").index(fam), fam, 0)
    ctx = fac.context(b, POLICY) if b is not None else None
    return b, balls, ctx, None


def model_a_crop(ctx, balls, s, o, L):
    """Model A: exact SOLID and responses of an L^3 window (certified CPU predicates; balls exact)."""
    from rind3d_curved.spheres import BallSet
    z, y, x = np.meshgrid(*[np.arange(L)] * 3, indexing="ij")
    X = (np.asarray(o)[None] + np.stack([x, y, z], -1).reshape(-1, 3) + .5) / N
    if ctx is not None:
        sol = ctx.contains(np.ascontiguousarray(X)); lit, _, _ = ctx.bind(s).native.points(np.ascontiguousarray(X)); lit = lit == 1
    else:
        sol = np.zeros(len(X), bool); lit = np.ones(len(X), bool)
    if len(balls):
        B = BallSet(balls[:, :3], balls[:, 3]); sb = B.contains(X); sol |= sb
        lit &= ~B.blocked(s, X)
    lit &= ~sol
    return lit.reshape(L, L, L), sol.reshape(L, L, L)


def legal_a(ctx, balls, s):
    from rind3d_1k.physical import legal_source
    from rind3d_curved.spheres import BallSet
    ok = bool(np.all((s > 0) & (s < 1)))
    if ok and ctx is not None:
        ok = legal_source(ctx, s) is not None
    if ok and len(balls):
        ok = BallSet(balls[:, :3], balls[:, 3]).legal_source(s)
    return ok


def cmd_a(a):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k.interface import interface_axes
    fac = BackendFactory(cuda_allowed=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for kind, fam, slot in SCENES:
        p = out / f"A_{kind}_{fam}_{slot}.npz"
        if p.exists():
            continue
        t0 = time.perf_counter()
        b, balls, ctx, knife = scene_geometry(kind, fam, slot, fac)
        rng = np.random.default_rng([slot, len(fam), 7])
        if knife is not None:
            srcs = [knife["s"]]; wins = [(knife["o"], 64)]
        else:
            srcs = []
            while len(srcs) < 2:
                s = rng.uniform(.03, .97, 3)
                if legal_a(ctx, balls, s):
                    srcs.append(s)
            wins = []; tries = 0
            while len(wins) < 4 and tries < 200:
                tries += 1
                L = 64 if len(wins) < 3 else 128
                o = rng.integers(0, N - L + 1, 3)
                if any(np.all((s >= o / N) & (s <= (o + L) / N)) for s in srcs):
                    continue
                lit, sol = model_a_crop(ctx, balls, srcs[len(wins) % len(srcs)], o, L)
                if sum(interface_axes(lit, sol)) > 0:
                    wins.append((o, L))
        crops = {}; ta = 0.0
        for wi, (o, L) in enumerate(wins):
            for j, s in enumerate(srcs):
                t = time.perf_counter(); lit, sol = model_a_crop(ctx, balls, s, o, L); ta += time.perf_counter() - t
                crops[f"lit_{wi}_{j}"] = np.packbits(lit.ravel()); crops[f"sol_{wi}"] = np.packbits(sol.ravel())
        # source legality disagreement sample (uniform points; Model A legality here, Model B in process 2)
        P = rng.uniform(0, 1, (300, 3)); la = np.array([legal_a(ctx, balls, q) for q in P])
        np.savez_compressed(p, kind=kind, fam=fam, slot=slot, sources=np.array(srcs), origins=np.array([w[0] for w in wins]),
                            sides=np.array([w[1] for w in wins]), balls=balls, legal_pts=P, legal_a=la,
                            t_a_render=ta, t_total=time.perf_counter() - t0, **crops)
        print("A", kind, fam, slot, len(wins), "windows", round(time.perf_counter() - t0, 1), "s", flush=True)


def _stab_through(li, da, s_vox):
    from scipy.optimize import linprog
    l = li + .5; k = da + .5; g = (l - k).mean(0); gn = np.linalg.norm(g)
    if gn == 0:
        return None
    g /= gn
    A_ub = np.r_[np.c_[-l, np.ones(len(l))], np.c_[k, -np.ones(len(k))]]
    A_eq = np.array([np.r_[s_vox, -1.0], np.r_[g, 0.0]])
    r = linprog(np.zeros(4), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq, b_eq=[0.0, 1.0], bounds=[(None, None)] * 4, method="highs")
    return r.status == 0


def channel_metrics(lit, sol, s, o, rows=None):
    from rind3d_100k.interface import interface_axes, channel_class
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, edge_free_mask
    cls = np.where(sol, 2, np.where(lit, 1, 0)).transpose(2, 1, 0)                # [x, y, z]
    li, da = interface_edges(cls)
    m = dict(cls=channel_class(lit, sol), edges=int(len(li)))
    if rows is not None and len(li):
        m["nonfree"] = int((~edge_free_mask(rows, o, N, li, da)).sum())
    if len(li) >= 24:
        comps = plane_components(li, da, max_planes=4, seed=0)
        if comps:
            c = comps[0]; idx = c["idx"]
            m["largest_component"] = int(c["n"]); m["stab_through_source"] = _stab_through(li[idx], da[idx], s * N - np.asarray(o))
            m["normal_axis_alignment"] = float(np.max(np.abs(c["normal"])))             # 1 = axis-aligned plane
            m["normal_min_abs"] = float(np.min(np.abs(c["normal"])))                    # 0 = plane contains an axis direction
            m["plane_rms"] = float(c["rms"])
    return m


def cmd_b(a):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_voxel.world import bodies_from_context, build_world
    from rind3d_voxel.cuda_dda import GPURenderer
    fac = BackendFactory(cuda_allowed=False)                   # CPU contexts only: geometry decoding, no CUDA runtime
    out = Path(a.out); rows_out = []
    for p in sorted(out.glob("A_*.npz")):
        z = np.load(p, allow_pickle=False); kind, fam, slot = str(z["kind"]), str(z["fam"]), int(z["slot"])
        b, balls, ctx, knife = scene_geometry(kind, fam, slot, fac)
        bodies = bodies_from_context(ctx, balls)
        t = time.perf_counter(); Wb = build_world(bodies, N, "supercover"); tv = time.perf_counter() - t
        t = time.perf_counter(); Wc = build_world(bodies, N, "center"); tc = time.perf_counter() - t
        t = time.perf_counter(); G = GPURenderer(Wb); tg0 = time.perf_counter() - t
        rows = [c.rows for c in ctx.components] if ctx is not None else None
        srcs = z["sources"]; legal_b = [bool(Wb.legal_source(s)) for s in srcs]
        lb = np.array([Wb.legal_source(q) for q in z["legal_pts"]])
        sc = dict(kind=kind, family=fam, slot=slot, occ_B=Wb.n_occupied(), occ_A_samples=Wc.n_occupied(),
                  voxelize_s=tv, center_s=tc, gpu_setup_s=tg0, sparse_MB=Wb.nbytes_sparse() / 2 ** 20, bricks=len(Wb.bricks),
                  sources_illegal_in_B=int(len(srcs) - sum(legal_b)),
                  legality_disagree=float(np.mean(lb != z["legal_a"])), legal_A_only=int(np.sum(z["legal_a"] & ~lb)), legal_B_only=int(np.sum(~z["legal_a"] & lb)),
                  t_a_render=float(z["t_a_render"]), windows=[])
        tb = 0.0
        for wi, (o, L) in enumerate(zip(z["origins"], z["sides"])):
            L = int(L); solA = np.unpackbits(z[f"sol_{wi}"], count=L ** 3).astype(bool).reshape(L, L, L)
            for j, s in enumerate(srcs):
                litA = np.unpackbits(z[f"lit_{wi}_{j}"], count=L ** 3).astype(bool).reshape(L, L, L)
                t = time.perf_counter(); litB, solB = G.window(s, o, L); tb += time.perf_counter() - t
                both_free = ~solA & ~solB
                w = dict(L=L, j=j, legal_B=legal_b[j], solid_A=int(solA.sum()), solid_B=int(solB.sum()),
                         solid_A_not_B=int((solA & ~solB).sum()), solid_B_not_A=int((solB & ~solA).sum()),
                         lit_B_not_A=int((litB & ~litA).sum()), lit_A_not_B=int((litA & ~litB & both_free).sum()),
                         response_diff=float(np.mean(litA[both_free] != litB[both_free])) if both_free.any() else None,
                         A=channel_metrics(litA, solA, s, o, rows), B=channel_metrics(litB, solB, s, o) if legal_b[j] else None)
                sc["windows"].append(w)
        sc["t_b_render"] = tb; sc["gpu_flagged"] = G.stats["flagged"]
        rows_out.append(sc)
        print("B", kind, fam, slot, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items() if k != "windows"}), flush=True)
        with open(out / "ab_rows.jsonl", "a") as f:
            f.write(json.dumps(sc, default=float) + "\n")


def cmd_report(a):
    out = Path(a.out); R = [json.loads(l) for l in (out / "ab_rows.jsonl").read_text().splitlines() if l.strip()]
    print("| scene | occupied B / A-samples | dilation | voxelize s | sparse MB | legality disagree | SOLID B-not-A in windows | response diff (both FREE) | class changes | A edges non-free | H2 fails A / B | B normal axis-aligned |")
    print("|" + "---|" * 12)
    for r in R:
        W = r["windows"]; WB = [w for w in W if w["B"]]
        cc = sum(w["A"]["cls"] != w["B"]["cls"] for w in WB)
        nf = sum(w["A"].get("nonfree", 0) for w in W); ed = sum(w["A"]["edges"] for w in W)
        h2a = [w["A"].get("stab_through_source") for w in W if w["A"].get("stab_through_source") is not None]
        h2b = [w["B"].get("stab_through_source") for w in WB if w["B"].get("stab_through_source") is not None]
        ax = [w["B"].get("normal_axis_alignment") for w in WB if w["B"].get("normal_axis_alignment") is not None]
        rd = [w["response_diff"] for w in W if w["response_diff"] is not None]
        print(f"| {r['kind']}:{r['family']}:{r['slot']} | {r['occ_B']:,} / {r['occ_A_samples']:,} | {r['occ_B'] / max(r['occ_A_samples'], 1) - 1:+.1%} | "
              f"{r['voxelize_s']:.1f} | {r['sparse_MB']:.1f} | {r['legality_disagree']:.1%} | {sum(w['solid_B_not_A'] for w in W):,} | "
              f"{np.mean(rd):.2%} (max {np.max(rd):.2%}) | {cc}/{len(WB)} | {nf}/{ed} | {sum(x is False for x in h2a)}/{len(h2a)} , {sum(x is False for x in h2b)}/{len(h2b)} | "
              f"{np.median(ax) if ax else float('nan'):.3f} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["a", "b", "report"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(a=cmd_a, b=cmd_b, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
