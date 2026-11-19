"""Physical stage for curved-occluder research scenes (spheres; contract rind3d-100k-curved-research-v0).

Same law as contract v3 (rind3d_100k.physical: kappa source count, natural and coverage streams, direct-interface
eligibility, exact-m rate stream), with the geometry of `rind3d_curved.families` and the composite exact scene
`curved_scene.CurvedLocalScene`. Differences:
* Driver sources are legal iff they are legal for the polytopes (when present) and outside every closed ball with
  clearance > 2^-26 (exact).
* Every ball must lie strictly inside the open world and have a nonempty 6-connected voxel-centre sampling.
* There is no all-active hull certificate (cost only) and no posterior label: no certified labeller supports balls yet.
  These scenes are not part of T*_v3.
* The container stores the balls (`balls`, float64 [n, 4]) next to the polytope geometry (when present).
"""
import hashlib
import io
import json
import time

import numpy as np
import zstandard

from rind3d.geometry import keyed_rng, pack_arrays
from rind3d_1k.physical import POLICY, STREAM, legal_source as poly_legal, geometry_rows, DUP_SEP

from . import sampling as SM
from .physical import select_side_v2, _packz, SIDES


def _sources(context, balls, seed, slot, k, attempt, limit=4096):
    from rind3d_curved.spheres import BallSet
    B = BallSet(balls[:, :3], balls[:, 3]) if len(balls) else None
    rng = keyed_rng(seed, slot, STREAM["sources"] + 97 * attempt)
    src, rej = [], {"illegal": 0, "duplicate": 0}; n = 0
    while len(src) < k:
        if n >= limit:
            raise RuntimeError("source proposal budget exhausted")
        n += 1
        p = rng.random(3)
        ok = bool(np.all((p > 0) & (p < 1)))
        if ok and context is not None:
            ok = poly_legal(context, p) is not None
        if ok and B is not None:
            ok = B.legal_source(p)
        if not ok:
            rej["illegal"] += 1; continue
        if any(np.max(np.abs(p - q)) < DUP_SEP for q in src):
            rej["duplicate"] += 1; continue
        src.append(p)
    return np.array(src, "<f8"), {"proposals": n, "rejections": rej}


def run_scene_curved(cfg, slot, k, factory, attempt=0):
    from rind3d_curved.families import generate_curved, ball_sampled_components
    from .curved_scene import CurvedLocalScene
    seed, N = cfg["seed"], cfg["resolution"]
    T = {}; t0 = time.perf_counter()
    fams = cfg["families"]; family = fams[slot % len(fams)]
    gcfg = cfg.get("geometry") or {}
    bundle, balls, recipe = generate_curved(seed, slot, family, attempt, thin_min_vox=float(gcfg.get("thin_min_thickness_vox", 2.0)))
    T["generate"] = time.perf_counter() - t0
    if recipe["proposal_failures"]:
        return dict(slot=slot, attempt=attempt, status="rejected", reason="geometry_proposal_failure", k=k, family=family), None
    t = time.perf_counter()
    context = factory.context(bundle, POLICY) if bundle is not None else None
    T["context"] = time.perf_counter() - t
    t = time.perf_counter()
    ball_rep = []
    for b in balls:
        nv, nc = ball_sampled_components(b[:3], b[3], N)
        inside = bool(np.all(b[:3] - b[3] > 0) and np.all(b[:3] + b[3] < 1))
        ball_rep.append(dict(n_vox=nv, n_components=nc, inside=inside, radius_vox=float(b[3] * N)))
    T["ball_check"] = time.perf_counter() - t
    if not all(r["n_vox"] > 0 and r["n_components"] == 1 and r["inside"] for r in ball_rep):
        return dict(slot=slot, attempt=attempt, status="rejected", reason="ball_invalid", k=k, family=family), None
    t = time.perf_counter()
    acceptance = None
    if context is not None:
        from rind3d_1k.families import acceptance as geometry_acceptance
        acceptance = geometry_acceptance(context, "convex_polyhedra", seed, slot, attempt)
    T["acceptance"] = time.perf_counter() - t
    T["geometry"] = time.perf_counter() - t0
    t = time.perf_counter()
    sources, srep = _sources(context, balls, seed, slot, k, attempt)
    T["sources"] = time.perf_counter() - t
    gid = hashlib.sha256((context.geometry_id if context is not None else "").encode() + balls.tobytes()).hexdigest()
    sid = hashlib.sha256(cfg["contract"].encode() + bytes.fromhex(gid) + np.asarray(sources, "<f8").tobytes()).hexdigest()
    t = time.perf_counter()
    scene = CurvedLocalScene(factory, bundle, balls, sources, POLICY, N, cfg.get("backend", "cuda_certified"))
    if cfg.get("hull_certificate", True):
        scene.enable_hull(context)
    T["bind"] = time.perf_counter() - t
    probe_rng = np.random.default_rng([seed, slot, attempt, 77001])
    try:
        t = time.perf_counter()
        windows, sides, logs = [], [], {}
        for Lp in SIDES:
            acc, rep, log = select_side_v2(scene, sources, cfg, slot, attempt, Lp, k, probe_rng)
            windows += acc; sides.append(rep); logs[f"L{Lp}"] = log
        T["selection"] = time.perf_counter() - t
        work = dict(scene.stats)
        if scene.btiles is not None:
            work.update(ball_tiles=scene.btiles.stats["tiles"], ball_unknown=scene.btiles.stats["unknown"])
        if scene.balls is not None:
            work.update(ball_exact=scene.balls.stats["exact"])
    finally:
        scene.close()
    base = dict(slot=slot, attempt=attempt, k=k, family=family, scene_id=sid, sides=sides, timing=T, work=work)
    if not all(s["filled"] for s in sides):
        return dict(base, status="rejected", reason="quota_unfilled"), None
    t = time.perf_counter()
    cctx = zstandard.ZstdCompressor(level=3); blobs = io.BytesIO(); index = []; wstats = []
    for w, rec in enumerate(windows):
        b = _packz(rec["solid"], cctx); index.append((w, -1, blobs.tell(), len(b))); blobs.write(b)
        free = ~rec["solid"]
        for j in range(k):
            b = _packz(rec["lits"][j], cctx); index.append((w, j, blobs.tell(), len(b))); blobs.write(b)
        wstats.append(dict(free=int(free.sum()), active_frac=[float(rec["lits"][j][free].mean()) for j in range(k)]))
    fields = [("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"), ("Lp", "<i4"), ("m", "<i1"),
              ("mask", "<u1"), ("k", "<i1"), ("stream", "<i1"), ("mixed_mask", "<u1")]
    W = np.zeros(len(windows), dtype=fields)
    for w, r in enumerate(windows):
        W[w] = (w, *r["origin"], r["L"], r["Lp"], r["m"], r["mask"], k, SM.STREAM_ID[r["stream"]], int(r["mixed_mask"]))
    gbuf = io.BytesIO()
    if bundle is not None:
        np.savez_compressed(gbuf, **pack_arrays(bundle))
        P, K = geometry_rows(context)
    else:
        P, K = np.zeros((0, 1, 4)), np.zeros(0, np.int32)
    lam = {str(s["Lp"]): s["Lambda_hat"] for s in sides}
    meta = dict(contract=cfg["contract"], scene_id=sid, slot=slot, attempt=attempt, seed=seed, resolution=N, family=family,
                n_bodies=(len(context.bodies) if context is not None else 0), n_balls=len(balls), geometry_id=gid, k=k,
                sides=sides, Lambda_hat=lam, quota=cfg["quota"], n_windows=len(windows), timing=T, work=work,
                windows_full=[dict({kk: v for kk, v in r.items() if kk not in ("lits", "solid")}, **wstats[w]) for w, r in enumerate(windows)],
                sources_report=srep, balls_report=ball_rep, curved_version=recipe["version"],
                sampling=dict(law="declared", kappa=[float(x) for x in SM.kappa(cfg)], k=k, selection=cfg["selection"],
                              eligibility=cfg["eligibility"], rate_exact="all"))
    out = io.BytesIO()
    np.savez(out, meta=np.frombuffer(json.dumps(meta, default=str).encode(), np.uint8), windows=W,
             obs=np.frombuffer(blobs.getvalue(), np.uint8), obs_index=np.array(index, np.int64),
             geometry=np.frombuffer(gbuf.getvalue(), np.uint8), P=P, K=K, balls=balls,
             sources_hex=np.array([[float(v).hex() for v in s] for s in sources]), **{f"log_{kk}": v for kk, v in logs.items()})
    T["serialize"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    rec = dict(base, status="accepted", m_hist=np.bincount([r["m"] for r in windows], minlength=5)[1:].tolist(),
               Lambda_hat=lam, bytes=len(out.getbuffer()), windows=[[int(r["L"]), int(r["mask"]), int(r["m"])] for r in windows])
    return rec, out.getvalue()
