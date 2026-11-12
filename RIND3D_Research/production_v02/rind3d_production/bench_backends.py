"""Stage B1: matched geometry/physics backend benchmark on fresh certified CSG scenes.

Per scene (2 per family, benchmark seed 20261201, never used for data): one legal true source, 32 legal
hypothetical (Teacher-style) sources, one random 64^3 window at 1024^3 and its FREE receivers, and random
FREE subsets of 4096 and 256 receivers. Backends on exactly the same queries:
  A  certified native point clipping (cpu_native_certified, parametric intervals + GMP fallback)
  B  independent certified projected-shadow predicate (rational Fourier-Motzkin shadow, interval + GMP)
  R  certified CPU raster of the whole window (hierarchical box certificates)
  G  certified CUDA raster of the whole window (directed binary64 intervals, exact CPU fallback)
  E  directly triangulated components (float32) + Embree BVH (any-hit), float32 source
  W  the same mesh + Warp BVH on the GPU
  H  hybrid: E, W, +-2^-20 translation probes, certified A arbitration of every disagreement/unstable ray
Reference: A, cross-checked by B on every queried ray of the true source and two hypothetical sources.
Recorded: setup (context, per-source bind, CUDA context/geometry, triangulation, BVH builds), first-query and
repeated-query seconds, rays/s, mismatches of every backend against A, mesh vertex plane residuals.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rind3d.rendering.backends import BackendFactory

from . import scenes as S
from .env import POLICY, ROOT, mesh_engines
from .triangulate import residuals, triangulate

SEED = 20261201
OUT = ROOT/"outputs"/"benchmarks"
NEAR = False


def window_points(origin, side, n=1024):
    a = np.arange(side)+.5
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    o = np.asarray(origin, np.float64)
    return np.column_stack([(o[0]+x.ravel())/n, (o[1]+y.ravel())/n, (o[2]+z.ravel())/n])


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter()-t


def certified_points(context, source, pts, independent=False):
    m = context.bind(source, independent=independent)
    y, _, work = m.native.points(np.ascontiguousarray(pts), independent)
    return y.astype(np.int8)


def bench_scene(factory, engines, oracle_mod, family, slot, rng, record):
    bundle, recipe = S.geometry(SEED, slot, family)
    context, t_ctx = timed(factory.context, bundle, POLICY)
    _, t_proof = timed(context.geometry_proof)
    truth, _, srep = S.sample_sources(context, SEED, slot, 1)
    truth = truth[0]
    hyp, tried = S.hypothetical_sources(context, rng, 32)
    if NEAR:
        # informative variant: a 64^3 window centred on a random point of a random component's surface region
        comp = context.components[int(rng.integers(len(context.components)))]
        anchor = np.array([float(v) for v in comp.anchor])
        origin = np.clip(np.floor(anchor*1024).astype(np.int64) - 32 + rng.integers(-24, 25, 3), 0, 1024-64)
    else:
        origin = rng.integers(0, 1024-64, 3)
    W = window_points(origin, 64)
    solid, t_solid = timed(context.contains, W)
    free = np.flatnonzero(~solid)
    P = W[free]
    s4k = P[rng.choice(len(P), min(4096, len(P)), replace=False)]
    s256 = P[rng.choice(len(P), min(256, len(P)), replace=False)]
    row = {"family": family, "slot": slot, "geometry_id": context.geometry_id, "components": len(context.components),
           "planes": int(sum(len(c.rows) for c in context.components)), "window_origin": origin.tolist(),
           "free_receivers": int(len(P)), "solid_fraction_window": float(solid.mean()),
           "setup": {"context_seconds": t_ctx, "geometry_proof_seconds": t_proof, "membership_seconds_262k": t_solid},
           "hypothetical_proposals": tried}
    # A: certified points, first and repeated query per source, three receiver set sizes.
    t = time.perf_counter()
    m = context.bind(truth)
    row["setup"]["source_bind_seconds"] = time.perf_counter()-t
    A = {}
    for name, pts in (("dense", P), ("s4096", s4k), ("s256", s256)):
        y, t1 = timed(lambda: m.native.points(np.ascontiguousarray(pts), False)[0])
        _, t2 = timed(lambda: m.native.points(np.ascontiguousarray(pts), False)[0])
        A[name] = y.astype(np.int8)
        row[f"A_{name}"] = {"rays": int(len(pts)), "first_seconds": t1, "repeat_seconds": t2,
                            "rays_per_second_single_thread": len(pts)/max(t2, 1e-12)}
    # Teacher-style: 32 hypothetical sources x 4096 receivers, bind + query each, single thread and 16 threads.
    def one(src):
        mm = context.bind(src)
        return mm.native.points(np.ascontiguousarray(s4k), False)[0]
    _, t_seq = timed(lambda: [one(s) for s in hyp[:8]])
    context._cache.clear()
    with ThreadPoolExecutor(16) as ex:
        _, t_par = timed(lambda: list(ex.map(one, hyp)))
    row["A_teacher_batch"] = {"sources_sequential": 8, "seconds_sequential": t_seq,
                              "pairs_per_second_sequential": 8*len(s4k)/t_seq,
                              "sources_parallel": int(len(hyp)), "threads": 16, "seconds_parallel": t_par,
                              "pairs_per_second_parallel": len(hyp)*len(s4k)/t_par}
    # B: independent certified projection on the dense set (true source + 2 hypothetical sources).
    mism_AB, t_B = 0, 0.
    for src in [truth, *hyp[:2]]:
        a = certified_points(context, src, P)
        b, dt = timed(certified_points, context, src, P, True)
        t_B += dt
        mism_AB += int((a != b).sum())
    row["B_dense_3_sources"] = {"seconds": t_B, "mismatches_vs_A": mism_AB, "unknown_A": int((A["dense"] < 0).sum())}
    # R: certified CPU raster of the window.
    (yR, mR, _), t_R = timed(m.render, origin.tolist(), 64, 1024)
    row["R_window"] = {"seconds": t_R, "mismatch_vs_A_FREE": int((yR.ravel()[free].astype(np.int8) != A["dense"]).sum()),
                       "solid_mismatch": int((mR.ravel() != solid).sum())}
    # G: certified CUDA raster (cold source bind + first window, then repeated).
    try:
        cs, t_bind = timed(factory.source, bundle, truth, "cuda_certified", POLICY)
        packed, t_g1 = timed(cs.packed_batch, [origin.tolist()], 64, 1024)
        packed, t_g2 = timed(cs.packed_batch, [origin.tolist()], 64, 1024)
        yG = np.unpackbits(np.frombuffer(packed[0][0], np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
        _, t_g8 = timed(cs.packed_batch, [(rng.integers(0, 960, 3)).tolist() for _ in range(8)], 64, 1024)
        row["G_window"] = {"source_bind_seconds": t_bind, "first_seconds": t_g1, "repeat_seconds": t_g2,
                           "batch8_seconds": t_g8, "mismatch_vs_A_FREE": int((yG.ravel()[free].astype(np.int8) != A["dense"]).sum())}
        cs.close()
    except Exception as e:                                                # noqa: BLE001 - recorded
        row["G_window"] = {"error": f"{type(e).__name__}: {e}"}
    # E/W/H: direct triangulation + BVH.
    mesh, t_tri = timed(triangulate, bundle)
    row["mesh"] = {"triangulation_seconds": t_tri, "triangles": int(len(mesh["faces"])), "vertices": int(len(mesh["vertices32"])),
                   "max_vertex_plane_residual": residuals(context, mesh)}
    V32, F = mesh["vertices32"], mesh["faces"]
    em, t_eb = timed(engines.Embree, V32, F)
    wa, t_wb = timed(engines.Warp, V32, F)
    s32 = truth.astype(np.float32)
    P32 = P.astype(np.float32)
    oe, t_e1 = timed(em.occluded, s32, P32)
    oe, t_e2 = timed(em.occluded, s32, P32)
    ow, t_w1 = timed(wa.occluded, s32, P32)
    ow, t_w2 = timed(wa.occluded, s32, P32)
    litA = A["dense"] == 1
    row["E"] = {"bvh_build_seconds": t_eb, "first_seconds": t_e1, "repeat_seconds": t_e2, "rays_per_second": len(P)/t_e2,
                "mismatch_vs_A": int(((~oe) != litA).sum())}
    row["W"] = {"bvh_build_seconds": t_wb, "first_seconds": t_w1, "repeat_seconds": t_w2, "rays_per_second": len(P)/t_w2,
                "mismatch_vs_A": int(((~ow) != litA).sum())}
    # Teacher batch on Embree: 32 sources x 4096 receivers.
    s4k32 = s4k.astype(np.float32)
    _, t_et = timed(lambda: [em.occluded(h.astype(np.float32), s4k32) for h in hyp])
    eA = 0
    for h in hyp[:4]:
        eA += int((em.occluded(h.astype(np.float32), s4k32) != (certified_points(context, h, s4k) == 0)).sum())
    row["E_teacher_batch"] = {"sources": int(len(hyp)), "seconds": t_et, "pairs_per_second": len(hyp)*len(s4k)/t_et,
                              "mismatch_vs_A_first4_sources": eA}
    # H: hybrid with certified arbitration.
    t = time.perf_counter()
    from gaussian_mesh_dataset.labels import perpendiculars
    t1v, t2v = perpendiculars(P32 - s32)
    unstable = np.zeros(len(P32), bool)
    delta = np.float32(2.0**-20)
    for v in (t1v, -t1v, t2v, -t2v):
        unstable |= em.segments(s32 + delta*v, P32 + delta*v) != oe
    flag = np.flatnonzero(unstable | (oe != ow))
    lit_h = ~oe
    if len(flag):
        lit_h[flag] = certified_points(context, truth, P[flag]) == 1
    t_h = time.perf_counter()-t
    row["H"] = {"seconds_after_E_W": t_h, "flagged": int(len(flag)), "mismatch_vs_A_after_arbitration": int((lit_h != litA).sum()),
                "mismatch_vs_A_unflagged": int(((~oe) != litA)[np.setdiff1d(np.arange(len(P)), flag)].sum())}
    record.append(row)
    print(family, slot, "comp", row["components"], "A s/262k %.2f" % row["A_dense"]["repeat_seconds"],
          "R %.2f" % t_R, "G", row["G_window"].get("repeat_seconds"), "E %.4f" % t_e2, "W %.4f" % t_w2,
          "E mism", row["E"]["mismatch_vs_A"], "H mism", row["H"]["mismatch_vs_A_after_arbitration"], "AB", mism_AB, flush=True)


def full_fields(factory, families, out):
    """A: complete 1024^3 response+solid fields through the unchanged certified CUDA stream writer."""
    from rind3d.rendering.stream import render_stream
    rows = []
    cfg = {"resolution": 1024, "workers": 4, "queue_size": 6, "checkpoint_bricks": 32, "generation_seconds": 600,
           "generation_points": 10_000_000_000, "build_information_index": True}
    for i, family in enumerate(families):
        bundle, _ = S.geometry(SEED, 100+i, family)
        context = factory.context(bundle, POLICY)
        truth, _, _ = S.sample_sources(context, SEED, 100+i, 1)
        for backend in ("cuda_certified", "cpu_native_certified") if i == 0 else ("cuda_certified",):
            r = factory.source(bundle, truth[0], backend, POLICY)
            root = out/"fields"/f"{family}-{backend}"
            res, t = timed(render_stream, r, root, cfg)
            rows.append({"family": family, "backend": backend, "seconds": t, "field_bytes": res["field_bytes"],
                         "information_index_bytes": (res.get("information_index") or {}).get("persistent_bytes"),
                         "first_brick_seconds": res["first_brick_seconds"], "cuda_timings": res.get("cuda_timings")})
            print("field", family, backend, round(t, 2), res["field_bytes"], flush=True)
            if hasattr(r, "close"):
                r.close()
    return rows


def main():
    global NEAR
    NEAR = len(sys.argv) > 2 and sys.argv[2] == "near"
    OUT.mkdir(parents=True, exist_ok=True)
    engines, oracle_mod = mesh_engines()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1]))
    record = []
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with BackendFactory(geometry_cache=2, cuda_batch=32) as factory:
        slot = 0
        for family in S.FAMILIES:
            for _ in range(n_per):
                bench_scene(factory, engines, oracle_mod, family, slot, rng, record)
                slot += 1
        cuda_report = factory.report()
        fields = [] if NEAR else full_fields(factory, list(S.FAMILIES), OUT)
    (OUT/("backends-near-raw.json" if NEAR else "backends-raw.json")).write_text(json.dumps({"seed": SEED, "scenes": record, "full_fields": fields,
                                                     "cuda": {k: v for k, v in cuda_report.items() if k != "cuda_runtime"}},
                                                    indent=1, default=float))


if __name__ == "__main__":
    main()
