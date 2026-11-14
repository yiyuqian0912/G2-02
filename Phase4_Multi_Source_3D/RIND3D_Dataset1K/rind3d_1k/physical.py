"""RIND3D 1K, physical stage: one scene, i.e. geometry, 1-4 sources, per-source exact fields, the 16/24/24 window
quotas with effective-layer thinning, a rate stream, and source-separated window observations.

Semantics (canonical, unchanged): rind3d closed-solid / closed-segment visibility (POLICY of the v0.2 producer),
receivers at voxel centres (i + 1/2)/N, SOLID receivers unlit.  Every number used for selection comes from exact
count indices built from the certified fields.

Sampling law (see docs/SELECTION_LAW_1K.md):
1. geometry: family = FAMILIES[slot % 9], generator and acceptance of the v0.2 producer;
2. k (source count): chosen by the dataset-level controller before any source of this scene is drawn;
3. sources: k independent uniform proposals in (0,1)^3, rejected unless legal (certified FREE plus clearance > 2^-26);
   the only interaction is an anti-duplicate rule |s_i - s_j|_inf >= 2^-20 (TV effect <= 1e-14, see the doc);
4. windows, per side L: i.i.d. uniform origins on {0..N-L}^3 from a stream keyed by (seed, slot, L) only;
   A(W) = [every source outside the closed box] and [FREE > 0] and [m >= 1], with m = #channels having both lit and
   dark FREE receivers; an A-window is accepted iff U < c(m, L), with U drawn from its own keyed stream; this stops
   at the quota (16/24/24) or at the cap, and repeated origins are skipped;
5. rate stream: R further proposals per side from another keyed stream give unbiased estimates of
   Lambda_L = P(accept | K, S) (training weights) and of P(m | L) (controller statistics);
6. scene acceptance: all three quotas filled within their caps, otherwise the scene is rejected and the reason
   recorded."""
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import zstandard

from rind3d.counts import CountIndex
from rind3d.geometry import keyed_rng, save_geometry
from rind3d.lite_acceptance import DRIVER_RADIUS, verify_clearance
from rind3d.lite_generators import FAMILIES

from .families import generate, acceptance as geometry_acceptance
from rind3d.pack import RINDPack
from rind3d.rendering.stream import render_stream

CONTRACT = "rind3d-1k-contract-v1"
SIDES_PHYS = (64, 128, 256)
QUOTA = {64: 16, 128: 24, 256: 24}
POLICY = {"max_precision_bits": 256, "max_subdivisions_per_item": 4096,
          "max_arrangement_events_per_item": 1048576, "qfnra_rlimit_per_item": 4194304}
DUP_SEP = 2.0 ** -20
STREAM = {"sources": 51001, "quota": 52000, "thin": 53000, "rate": 54000, "rate_thin": 55000}


def _rng(seed, slot, stream, attempt):
    return keyed_rng(seed, slot, stream + 97 * attempt)


def legal_source(context, p):
    p = np.asarray(p, dtype="<f8").reshape(3)
    if not np.all((p > 0) & (p < 1)):
        return None
    if context.contains(p[None])[0]:
        return None
    proof = context.clearance(p, DRIVER_RADIUS)
    if proof["status"] != "GT":
        return None
    verify_clearance(context, p, proof)
    return proof


def sample_sources(context, seed, slot, k, attempt=0, limit=4096):
    rng = _rng(seed, slot, STREAM["sources"], attempt)
    src, rej = [], {"illegal": 0, "duplicate": 0}
    n = 0
    while len(src) < k:
        if n >= limit:
            raise RuntimeError("source proposal budget exhausted")
        n += 1
        p = rng.random(3)
        if legal_source(context, p) is None:
            rej["illegal"] += 1
            continue
        if any(np.max(np.abs(p - q)) < DUP_SEP for q in src):
            rej["duplicate"] += 1
            continue
        src.append(p)
    return np.array(src, dtype="<f8"), {"proposals": n, "rejections": rej}


def contains_any(sources, o, L, N):
    lo = np.asarray(o, dtype=np.float64) / N
    hi = (np.asarray(o, dtype=np.float64) + L) / N
    return bool(np.any(np.all((sources >= lo) & (sources <= hi), axis=1)))


def geometry_rows(context):
    """Halfspace rows of the canonical convex bodies (rows a.x + d <= 0) as float64 arrays for the labeller.
    Exact rows are rationals; float() rounds them to binary64 (the Phase-1/2 convention, covered by the engine's
    decision margins)."""
    B = context.bodies
    K = np.array([len(b) for b in B], np.int32)
    P = np.zeros((len(B), int(K.max()), 4))
    P[..., 3] = -1.0
    for i, b in enumerate(B):
        P[i, :len(b)] = [[float(v) for v in r] for r in b]
    return P, K


class Fields:
    def __init__(self, dirs, N):
        self.N = N
        self.resp = [RINDPack(d / "response.rindb", cache_bricks=64) for d in dirs]
        self.solid = RINDPack(dirs[0] / "solid.rindb", cache_bricks=64)
        self.counts = [CountIndex.build(r, s) for r, s in
                       zip(self.resp, [RINDPack(d / "solid.rindb", cache_bricks=16) for d in dirs])]

    def channel_counts(self, o, L):
        out = np.zeros((len(self.counts), 3), np.int64)
        for j, ci in enumerate(self.counts):
            c = ci.query(tuple(int(v) for v in o), int(L))
            out[j] = (c.lit, c.shadow, c.solid)
        return out


def evaluate(F, sources, o, L, N):
    """(A, m, informative mask, counts) for one proposal."""
    if contains_any(sources, o, L, N):
        return 0, 0, 0, None, "contains_source"
    cnt = F.channel_counts(o, L)
    free = L ** 3 - cnt[0, 2]
    if free <= 0:
        return 0, 0, 0, cnt, "no_free"
    inf = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
    m = int(inf.sum())
    mask = int(sum(1 << j for j in range(len(inf)) if inf[j]))
    if m == 0:
        return 0, 0, 0, cnt, "uninformative"
    return 1, m, mask, cnt, "A"


def select_side(F, sources, seed, slot, attempt, Lp, N, c_table, cap, rate_n, quota=None):
    L = Lp * N // 1024
    n_quota = (quota or QUOTA)[Lp]
    rq = _rng(seed, slot, STREAM["quota"] + Lp, attempt)
    ru = _rng(seed, slot, STREAM["thin"] + Lp, attempt)
    acc, seen, log = [], set(), []
    reasons = {"contains_source": 0, "no_free": 0, "uninformative": 0, "A": 0, "thinned": 0, "repeat": 0}
    nprop = 0
    while len(acc) < n_quota and nprop < cap:
        o = tuple(int(v) for v in rq.integers(0, N - L + 1, 3))
        U = float(ru.random())
        nprop += 1
        if o in seen:
            reasons["repeat"] += 1
            continue
        seen.add(o)
        A, m, mask, cnt, why = evaluate(F, sources, o, L, N)
        reasons[why] += 1
        ok = bool(A and U < c_table[str(Lp)][m - 1])
        if A and not ok:
            reasons["thinned"] += 1
        log.append((o[0], o[1], o[2], A, m, mask, U, int(ok)))
        if ok:
            acc.append(dict(origin=o, L=L, Lp=Lp, m=m, mask=mask, counts=cnt.tolist(), U=U, proposal=nprop))
    # rate stream (fixed size, same rule): Lambda_L and the natural m distribution among A-windows
    rr = _rng(seed, slot, STREAM["rate"] + Lp, attempt)
    rt = _rng(seed, slot, STREAM["rate_thin"] + Lp, attempt)
    lam = 0.0; mh = np.zeros(5, np.int64); nA = 0
    for _ in range(rate_n):
        o = tuple(int(v) for v in rr.integers(0, N - L + 1, 3))
        U = float(rt.random())
        A, m, mask, cnt, why = evaluate(F, sources, o, L, N)
        if A:
            nA += 1; mh[m] += 1
            if U < c_table[str(Lp)][m - 1]:
                lam += 1
    return acc, dict(Lp=Lp, L=L, quota=n_quota, proposals=nprop, reasons=reasons, filled=len(acc) >= n_quota,
                     rate_n=rate_n, rate_A=nA, rate_m=mh[1:].tolist(), Lambda_hat=lam / rate_n,
                     P_A_hat=nA / rate_n), np.array(log, dtype=np.float64).reshape(-1, 8)


def _packz(a, cctx):
    return cctx.compress(np.packbits(np.ascontiguousarray(a, dtype=bool).ravel()).tobytes())


def materialize(F, windows, k, stage, cctx):
    """Public observation store: per window the local SOLID crop once and every source's response crop (packed bits,
    zstd), with an index; consistency of each crop's counts with the count index is asserted."""
    blobs = io.BytesIO()
    index = []          # (window, source(-1 = solid), offset, nbytes)
    stats = []
    for w, rec in enumerate(windows):
        o, L = rec["origin"], rec["L"]
        sol = F.solid.read_window(*o, L)                    # bool [Z,Y,X]
        b = _packz(sol, cctx); index.append((w, -1, blobs.tell(), len(b))); blobs.write(b)
        free = ~sol
        lits = []
        for j in range(k):
            lit = F.resp[j].read_window(*o, L)
            assert not np.any(lit & sol)
            nl = int(lit.sum()); ns = int((free & ~lit).sum())
            assert [nl, ns, int(sol.sum())] == rec["counts"][j], (w, j)
            b = _packz(lit, cctx); index.append((w, j, blobs.tell(), len(b))); blobs.write(b)
            lits.append(lit)
        S = np.sum(np.stack(lits), axis=0)[free]
        ov = np.bincount(S, minlength=k + 1)[:k + 1]
        inf = [j for j in range(k) if (rec["mask"] >> j) & 1]
        ham = [float(np.mean(lits[a][free] != lits[b][free])) for ai, a in enumerate(inf) for b in inf[ai + 1:]]
        stats.append(dict(overlap_hist=ov.tolist(), free=int(free.sum()),
                          lit_frac=[float(lits[j][free].mean()) for j in range(k)],
                          min_pair_hamming=min(ham) if ham else None))
    (stage / "public" / "observations.bin").write_bytes(blobs.getvalue())
    np.save(stage / "public" / "observations_index.npy", np.array(index, dtype=np.int64))
    return stats


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def run_scene(cfg, slot, k, c_table, root, factory, attempt=0):
    """Physical stage of one scene; publishes root/scenes/<slot>/ atomically or returns a rejection record."""
    seed, N = cfg["seed"], cfg["resolution"]
    T = {}
    t0 = time.perf_counter()
    fams = cfg.get("families", FAMILIES)
    family = fams[slot % len(fams)]
    quota = {int(a): int(b) for a, b in cfg["quota"].items()} if "quota" in cfg else QUOTA
    bundle, recipe = generate(seed, slot, family, attempt)
    if bundle is None or recipe["proposal_failures"]:
        return dict(slot=slot, attempt=attempt, status="rejected", reason="geometry_proposal_failure")
    context = factory.context(bundle, POLICY)
    acceptance = geometry_acceptance(context, family, seed, slot, attempt)
    T["geometry"] = time.perf_counter() - t0
    t = time.perf_counter()
    sources, srep = sample_sources(context, seed, slot, k, attempt)
    T["sources"] = time.perf_counter() - t
    sid = hashlib.sha256(CONTRACT.encode() + bytes.fromhex(context.geometry_id) +
                         np.asarray(sources, "<f8").tobytes()).hexdigest()
    stage = Path(root) / "stage" / f"{slot:06d}-{attempt}"
    shutil.rmtree(stage, ignore_errors=True)
    for d in ("public", "privileged/fields", "labels"):
        (stage / d).mkdir(parents=True)
    save_geometry(stage / "privileged" / "geometry.npz", bundle)
    P, K = geometry_rows(context)
    np.savez(stage / "privileged" / "geometry_rows.npz", P=P, K=K)
    t = time.perf_counter()
    dirs = []
    for j in range(k):
        r = factory.source(bundle, sources[j], cfg.get("backend", "auto"), POLICY)
        try:
            d = stage / "privileged" / "fields" / f"src{j}"
            render_stream(r, d, {"resolution": N, "workers": 4, "queue_size": 6, "checkpoint_bricks": 32,
                                 "generation_seconds": cfg.get("field_seconds", 1800),
                                 "generation_points": 10_000_000_000, "build_information_index": False})
            dirs.append(d)
        finally:
            r.close()
    T["fields"] = time.perf_counter() - t
    ref = None
    for d in dirs:                       # SOLID identical across sources (brick CRC and population)
        with RINDPack(d / "solid.rindb") as m:
            idx = [(rec[5], rec[7]) for rec in m.index]
        if ref is None:
            ref = idx
        elif idx != ref:
            raise AssertionError("SOLID differs between sources")
    t = time.perf_counter()
    F = Fields(dirs, N)
    T["count_index"] = time.perf_counter() - t
    t = time.perf_counter()
    windows, sides, logs = [], [], {}
    for Lp in SIDES_PHYS:
        acc, rep, log = select_side(F, sources, seed, slot, attempt, Lp, N, c_table, cfg["cap"][str(Lp)],
                                    cfg["rate_n"], quota)
        windows += acc; sides.append(rep); logs[f"L{Lp}"] = log
    T["selection"] = time.perf_counter() - t
    np.savez_compressed(stage / "privileged" / "selection_log.npz", **logs)
    if not all(s["filled"] for s in sides):
        rec = dict(slot=slot, attempt=attempt, status="rejected", reason="quota_unfilled", k=k, family=family,
                   sides=sides, timing=T, scene_id=sid)
        shutil.rmtree(stage, ignore_errors=True)
        return rec
    t = time.perf_counter()
    cctx = zstandard.ZstdCompressor(level=3)
    wstats = materialize(F, windows, k, stage, cctx)
    T["materialize"] = time.perf_counter() - t
    W = np.zeros(len(windows), dtype=[("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"),
                                      ("Lp", "<i4"), ("m", "<i1"), ("mask", "<u1"), ("k", "<i1")])
    for w, r in enumerate(windows):
        W[w] = (w, *r["origin"], r["L"], r["Lp"], r["m"], r["mask"], k)
    np.save(stage / "public" / "windows.npy", W)
    lam = {str(s["Lp"]): s["Lambda_hat"] for s in sides}
    (stage / "privileged" / "sources.json").write_text(json.dumps(
        dict(sources_hex=[[float(v).hex() for v in s] for s in sources], k=k, report=srep), indent=1))
    (stage / "privileged" / "windows_full.json").write_text(json.dumps(
        [dict(r, **wstats[w]) for w, r in enumerate(windows)], indent=0))
    scene = dict(contract=CONTRACT, scene_id=sid, slot=slot, attempt=attempt, seed=seed, resolution=N,
                 family=family, n_bodies=len(context.bodies), faces=[len(b) for b in context.bodies],
                 geometry_id=context.geometry_id, k=k, c_table=c_table, sides=sides, Lambda_hat=lam,
                 quota=quota, n_windows=len(windows), timing=T, recipe_keys=sorted(recipe.keys()))
    (stage / "scene.json").write_text(json.dumps(scene, indent=1))
    (stage / "privileged" / "geometry_acceptance.json").write_text(json.dumps(acceptance, default=str)[:200000])
    seal = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage / "SEAL.json").write_text(json.dumps(seal, indent=0))
    final = Path(root) / "scenes" / f"{slot:06d}"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(stage, final)
    T["total"] = time.perf_counter() - t0
    return dict(slot=slot, attempt=attempt, status="accepted", scene_id=sid, k=k, family=family,
                m_hist=np.bincount([r["m"] for r in windows], minlength=5)[1:].tolist(),
                m_by_side={str(Lp): np.bincount([r["m"] for r in windows if r["Lp"] == Lp], minlength=5)[1:].tolist()
                           for Lp in SIDES_PHYS},
                sides=[dict((kk, s[kk]) for kk in ("Lp", "proposals", "rate_m", "rate_A", "Lambda_hat", "reasons"))
                       for s in sides], timing=T)
