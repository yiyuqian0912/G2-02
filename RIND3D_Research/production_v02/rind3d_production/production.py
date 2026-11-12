"""Production scene job (contract rind3d-production-contract-v0.2): one slot -> one sealed scene directory.

Slot s of a campaign is deterministic: family = FAMILIES[s mod 9]; profile = single-source binary (SB) or
multi-source additive (MA) by the campaign's profile schedule; K = 1 (SB) or K in {2,3,4} (MA).
Steps (all certified physics comes unchanged from the producer checkout):
 1 geometry   lite_generators.generate(seed, slot, family, attempt); bounded world-fitting attempts
 2 accept     geometry_acceptance (convex-interior/world-dual proof, canonical roundtrip, probes)
 3 sources    K legal sources (certified FREE + clearance GT), pairwise separation, binary64 strengths
 4 fields     complete 1024^3 response + solid per source via the certified CUDA stream writer (RINDPack,
              CRC/SHA, information index); the SOLID bits of every source must be identical
 5 counts     exact CountIndex per source
 6 windows    obs-policy-v0.2 records (windows.select)
 7 verify     two independent physical replays (A parametric clipping/GMP, B rational shadow projection)
              of every brick covering a frozen verification subset of windows, per source
 8 public     single-source: response is the public observation (binary); multi-source: per-source fields
              are private, and the public observation of each selected window is the combined response
              stored as a LOCAL palette (values present in that window only) + uint8 codes
 9 seal       per-file SHA256, scene record, atomic rename staging -> published
A failed slot is recorded with its reason; it is never replaced by an easier geometry.
"""
import json
import os
import resource
import shutil
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import zstandard as zstd

from rind3d.counts import CountIndex
from rind3d.geometry import save_geometry
from rind3d.pack import RINDPack, atomic_json
from rind3d.physical_evidence import verify_physical_bricks
from rind3d.rendering.backends import BackendFactory
from rind3d.rendering.polyhedra import geometry_split
from rind3d.rendering.stream import render_stream

from . import scenes as S
from . import windows as WS
from .env import POLICY, sha256_file

Win = namedtuple("Win", "origin size")
CONTRACT = "rind3d-production-contract-v0.2"
PROFILES = {"SB": "rind3d-csg-single-binary-v0.2", "MA": "rind3d-csg-multi-additive-v0.2"}


def slot_plan(cfg, slot):
    family = S.FAMILIES[slot % len(S.FAMILIES)]
    sched = cfg["profile_schedule"]                      # e.g. ["SB", "MA"] cycled per family round
    prof = sched[(slot // len(S.FAMILIES)) % len(sched)]
    K = 1 if prof == "SB" else cfg["multi_K"][(slot // (len(S.FAMILIES)*len(sched))) % len(cfg["multi_K"])]
    return family, prof, int(K)


def field_cfg(cfg):
    return {"resolution": 1024, "workers": 4, "queue_size": 6, "checkpoint_bricks": 32,
            "generation_seconds": cfg["caps"]["field_seconds"], "generation_points": 10_000_000_000,
            "build_information_index": True}


def verification_windows(seed, slot, recs):
    """Frozen subset: per side one natural + one informative window (when present), seeded by the slot."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, slot, 4401]))
    out = []
    for L in WS.SIDES:
        for code in (0, 1):
            idx = np.flatnonzero((recs["L"] == L) & (recs["subset"] == code))
            if len(idx):
                r = recs[int(rng.choice(idx))]
                out.append(Win((int(r["x"]), int(r["y"]), int(r["z"])), int(L)))
    return out


def pattern_values(strengths):
    """Combined value of every lit pattern c: sequential binary64 sum of the lit strengths in channel order
    (exactly what sum_k a_k lit_k evaluates to, since adding +0.0 is exact)."""
    K = len(strengths)
    out = np.zeros(1 << K, np.float64)
    for c in range(1 << K):
        v = 0.0
        for k in range(K):
            v = v + (float(strengths[k]) if c >> k & 1 else 0.0)
        out[c] = v
    return out


def combined_window(responses, strengths, solid_pack, origin, L):
    """Combined additive response of one window: LOCAL palette (ascending values present in the window,
    SOLID receivers contribute 0) + uint8 codes. No value absent from the window is stored."""
    x, y, z = origin
    pattern = np.zeros((L, L, L), np.uint8)
    for k, pack in enumerate(responses):
        pattern |= pack.read_window(x, y, z, L).astype(np.uint8) << k
    solid = solid_pack.read_window(x, y, z, L)
    if np.any(pattern[solid]):
        raise AssertionError("lit SOLID receiver")
    values = pattern_values(strengths)
    present = np.flatnonzero(np.bincount(pattern.ravel(), minlength=len(values)))
    palette, inverse = np.unique(values[present], return_inverse=True)
    lut = np.zeros(len(values), np.uint8)
    lut[present] = inverse.astype(np.uint8)
    return palette, lut[pattern]


def run_scene(cfg, slot, stage_root, factory, attempt=0):
    seed = cfg["seed"]
    family, prof, K = slot_plan(cfg, slot)
    t_all, cpu0 = time.perf_counter(), time.process_time()
    stages = {}

    def tick(name, t, c):
        stages[name] = {"wall_seconds": time.perf_counter()-t, "cpu_seconds": time.process_time()-c}

    t, c = time.perf_counter(), time.process_time()
    bundle, recipe = S.geometry(seed, slot, family, attempt)
    context = factory.context(bundle, POLICY)
    acceptance = S.geometry_acceptance(context, family, seed, slot, attempt)
    gid = context.geometry_id
    tick("geometry_and_acceptance", t, c)
    t, c = time.perf_counter(), time.process_time()
    sources, strengths, srep = S.sample_sources(context, seed, slot, K, attempt)
    sid = S.scene_id(PROFILES[prof], gid, sources, strengths)
    tick("sources", t, c)
    stage = Path(stage_root)/f"{slot:06d}-{attempt}"
    if stage.exists():
        shutil.rmtree(stage)                                   # partial work of a dead attempt: never published
    (stage/"public").mkdir(parents=True)
    (stage/"private").mkdir()
    save_geometry(stage/"public"/"geometry.npz", bundle)
    t, c = time.perf_counter(), time.process_time()
    fields = []
    for k in range(K):
        r = factory.source(bundle, sources[k], cfg["renderer_backend"], POLICY)
        try:
            fields.append(render_stream(r, stage/"private"/f"src{k}", field_cfg(cfg)))
        finally:
            r.close()
    tick("fields", t, c)
    # SOLID identity across sources (decoded CRC + population of every brick).
    solid_index = []
    for k in range(K):
        with RINDPack(stage/"private"/f"src{k}"/"solid.rindb") as m:
            solid_index.append([(rec[5], rec[7]) for rec in m.index])
    if any(si != solid_index[0] for si in solid_index[1:]):
        raise AssertionError("SOLID fields differ between sources")
    t, c = time.perf_counter(), time.process_time()
    packs = [RINDPack(stage/"private"/f"src{k}"/"response.rindb", cache_bricks=64) for k in range(K)]
    solids = [RINDPack(stage/"private"/f"src{k}"/"solid.rindb", cache_bricks=16) for k in range(K)]
    solid = solids[0]
    counts = [CountIndex.build(packs[k], solids[k]) for k in range(K)]
    for k, ci in enumerate(counts):
        ci.save(stage/"private"/f"src{k}"/"counts.rindc")
    tick("count_index", t, c)

    def lit_window_fn(o, L):
        return packs[0].read_window(*o, L), solid.read_window(*o, L)
    t, c = time.perf_counter(), time.process_time()
    recs, wreport = WS.select(seed, slot, sources, counts, context, lit_window_fn)
    tick("windows", t, c)
    t, c = time.perf_counter(), time.process_time()
    vwins = verification_windows(seed, slot, recs)
    evidence = []
    for k in range(K):
        _, ev = verify_physical_bricks(stage/"private"/f"src{k}", vwins, context, sources[k],
                                       seconds=cfg["caps"]["verification_seconds"])
        evidence.append({"source": k, "bricks": len(ev["bricks"]), "stages": ev["stages"]})
    tick("independent_verification", t, c)
    t, c = time.perf_counter(), time.process_time()
    public_windows = None
    if prof == "MA":
        comp = zstd.ZstdCompressor(level=3)
        blobs, index = [], []
        offset = 0
        for i, r in enumerate(recs):
            o = (int(r["x"]), int(r["y"]), int(r["z"]))
            pal, codes = combined_window(packs, strengths, solid, o, int(r["L"]))
            blob = comp.compress(codes.tobytes())
            blobs.append(pal.astype("<f8").tobytes() + blob)
            index.append((offset, len(pal), len(blob)))
            offset += 8*len(pal) + len(blob)
        with open(stage/"public"/"observations.bin", "wb") as f:
            for b in blobs:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
        np.save(stage/"public"/"observations_index.npy", np.array(index, dtype=np.int64))
        public_windows = {"windows": int(len(recs)), "bytes": offset}
    for p in packs + solids:
        p.close()
    # Files are moved, not copied: a single-source response is the public observation; the SOLID mask is
    # public for both profiles; per-source multi-source channels and all indexes/evidence stay private.
    moves = ["solid.rindb", "solid.rindb.sha256"] + (["response.rindb", "response.rindb.sha256"] if prof == "SB" else [])
    for name in moves:
        os.replace(stage/"private"/"src0"/name, stage/"public"/name)
    np.save(stage/"public"/"windows.npy", recs[["window", "x", "y", "z", "L"]])
    np.save(stage/"private"/"window_records.npy", recs)
    atomic_json(stage/"private"/"sources.json", {
        "profile": PROFILES[prof], "K": K, "sources_hex": [[float(v).hex() for v in s] for s in sources],
        "strengths_hex": [float(a).hex() for a in strengths], "channel_order": "generation order k = 0..K-1",
        "clearance_proofs": srep["clearance"], "proposals": srep["proposals"], "rejections": srep["rejections"]})
    atomic_json(stage/"private"/"recipe.json", recipe)
    atomic_json(stage/"private"/"geometry_acceptance.json", acceptance)
    atomic_json(stage/"private"/"window_report.json", wreport)
    tick("publication_files", t, c)
    files = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "SHA256": sha256_file(p)}
             for p in sorted(stage.rglob("*")) if p.is_file()}
    public = {"contract": CONTRACT, "profile": PROFILES[prof], "scene_id": sid, "geometry_id": gid,
              "split": geometry_split(gid, 0), "family": family, "resolution": 1024,
              "receiver_centres": "(index + 1/2)/1024, ZYX arrays, packed little bit order along X",
              "observation": ("binary lit FREE receivers (public/response.rindb), SOLID mask public/solid.rindb"
                              if prof == "SB" else
                              "combined additive response per declared window: local palette + uint8 codes"),
              "windows": int(len(recs)), "public_files": {k: v for k, v in files.items() if k.startswith("public/")}}
    atomic_json(stage/"public"/"scene.json", public)
    record = {"slot": slot, "attempt": attempt, "scene_id": sid, "geometry_id": gid, "family": family, "profile": prof,
              "K": K, "split": public["split"], "windows": int(len(recs)), "window_report": wreport,
              "fields": [{"wall_seconds": f["wall_seconds"], "bytes": f["field_bytes"], "backend": f["backend"],
                          "cuda_memory": f.get("cuda_memory")} for f in fields],
              "evidence": evidence, "stages": stages, "public_windows": public_windows,
              "wall_seconds": time.perf_counter()-t_all, "cpu_seconds": time.process_time()-cpu0,
              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "files": files, "bytes": int(sum(v["bytes"] for v in files.values()))}
    atomic_json(stage/"SEAL.json", record)
    return stage, record
