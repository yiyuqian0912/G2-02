"""Safe two-view export of an obs-index-v0.3 index (trusted process; reads stored certified fields only).

  run.sh rind3d_training.export canary|v01 --name NAME [--profile P256] [--workers 12]

Student view   outputs/student/NAME/<split>/
  STUDENT_MANIFEST.json                 schema, feature keys, shard list (SHA256), counts; no seed, no id map
  shards/<mod>-L<LLL>-<nnnnn>.bin       concatenated payloads (zstd)
  shards/<mod>-L<LLL>-<nnnnn>.idx.npy   one row per sample (STUDENT_IDX)
Payloads: SB response and SOLID = packed bits (ZYX, little bit order along X), zstd. MA response = local
palette (float64, ascending, only values present in the window) + ceil(log2(npal)) packed bit planes of the
palette code, zstd. Samples of several scenes are interleaved by a private permutation inside a shard.
sample_id: opaque 64-bit HMAC of a private export key (join key only, never a feature); group: opaque 32-bit
per-scene token used only by the scene-balanced sampler (never a feature, never returned in a feature batch).

Privileged view outputs/privileged/export/NAME/ (directory mode 0700)
  join.npy        sample_id -> scene ordinal, index record, split, task flags, weights, targets (PRIV_JOIN)
  scenes.json     scene ordinal -> scene_id, geometry_id, release, family, profile, K, field directory
  selection_law.json   per scene and side: P(A_s), P(contains source), proposal reports
"""
import json
import os
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zstandard as zstd

from .env import OUT, PRIVILEGED_SCHEMA, STUDENT_SCHEMA, keyed_int, private_secret, sha256_file
from .index import CODES, PROFILES, SIDES
from rind3d_production import windows as W2

MOD = {"SB": 0, "MA": 1}
STUDENT_IDX = np.dtype([("sample_id", "<u8"), ("group", "<u4"), ("modality", "u1"), ("L", "<u2"),
                        ("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                        ("resp_offset", "<u8"), ("resp_bytes", "<u4"), ("npal", "u1"), ("nplanes", "u1"),
                        ("solid_offset", "<u8"), ("solid_bytes", "<u4"), ("resp_crc", "<u4"), ("solid_crc", "<u4")])
PRIV_JOIN = np.dtype([("sample_id", "<u8"), ("scene", "<u4"), ("record", "<u4"), ("split", "u1"), ("modality", "u1"),
                      ("L", "<u2"), ("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("subset", "u1"), ("rank", "<u4"),
                      ("in_P102", "?"), ("in_P256", "?"), ("contains_source", "?"), ("no_free", "?"), ("category", "u1"),
                      ("free", "<u4"), ("A_informative", "?"), ("A_extreme", "?"), ("A_rare", "?"), ("nonlocal_flag", "?"),
                      ("w_global_mixture", "<f8"), ("w_srcexcl_mixture", "<f8"),
                      ("K", "u1"), ("sources", "<f8", (4, 3)), ("strengths", "<f8", (4,))])
SPLITS = {"train": 0, "val": 1, "test": 2}
STUDENT_FEATURES = ("observed_response", "observed_solid", "window_origin", "window_size", "modality", "local_coordinates")


def in_profile(rec, profile):
    """Membership of index records in a (smaller) prefix profile: rank < quota and proposal <= that profile's cap."""
    P = PROFILES[profile]
    ok = np.zeros(len(rec), bool)
    for s, code in CODES.items():
        for L in SIDES:
            q = P["natural"][L] if s == "natural" else P[s]
            cap = q if s == "natural" else int(W2.CAPS[s]*(1 if profile == "P102" else 2.5))
            m = (rec["subset"] == code) & (rec["L"] == L) & (rec["rank"] < q)
            if s != "natural":
                m &= rec["proposal_index"] <= cap
            ok |= m
    return ok


def planes(codes, npal):
    nb = max(1, int(np.ceil(np.log2(npal)))) if npal > 1 else 0
    return nb, b"".join(np.packbits(((codes >> b) & 1).astype(np.uint8).ravel(), bitorder="little").tobytes() for b in range(nb))


def _shard_job(args):
    from .scenes import SceneRef
    from rind3d.pack import RINDPack
    from rind3d_production.production import combined_window
    job, out = args
    comp = zstd.ZstdCompressor(level=3)
    rows = {}
    t0 = time.perf_counter()
    opened = {}
    try:
        for so, sc_args, items in job["scenes"]:
            sc = SceneRef(**sc_args)
            opened[so] = ([RINDPack(p, cache_bricks=64) for p in sc.responses], RINDPack(sc.solid[0], cache_bricks=64),
                          sc.privileged_labels()[1])
        files = {}
        for so, w, L, o, sid, grp in job["order"]:
            packs, sol, strengths = opened[so]
            key = (int(L))
            if key not in files:
                name = f"{job['mod']}-L{L:03d}-{job['shard']:05d}"
                files[key] = (open(Path(out)/f"{name}.bin", "wb"), name)
                rows[key] = []
            f, name = files[key]
            r = np.zeros((), STUDENT_IDX)
            r["sample_id"], r["group"], r["modality"], r["L"] = sid, grp, MOD[job["mod"]], L
            r["x"], r["y"], r["z"] = o
            if job["mod"] == "SB":
                resp = packs[0].read_window(*o, L, packed=True).data.tobytes()
                r["npal"], r["nplanes"] = 2, 1
                payload = resp
            else:
                pal, codes = combined_window(packs, strengths, sol, tuple(o), L)
                nb, pl = planes(codes, len(pal))
                r["npal"], r["nplanes"] = len(pal), nb
                payload = pal.astype("<f8").tobytes() + pl
            solid = sol.read_window(*o, L, packed=True).data.tobytes()
            r["resp_crc"], r["solid_crc"] = zlib.crc32(payload), zlib.crc32(solid)
            b1, b2 = comp.compress(payload), comp.compress(solid)
            r["resp_offset"] = f.tell()
            r["resp_bytes"] = len(b1)
            f.write(b1)
            r["solid_offset"] = f.tell()
            r["solid_bytes"] = len(b2)
            f.write(b2)
            rows[key].append(r)
        out_rows = {}
        for key, (f, name) in files.items():
            f.flush()
            os.fsync(f.fileno())
            f.close()
            idx = np.array(rows[key], STUDENT_IDX)
            np.save(Path(out)/f"{name}.idx.npy", idx)
            out_rows[name] = len(idx)
    finally:
        for packs, sol, _ in opened.values():
            for p in packs:
                p.close()
            sol.close()
    return {"shard": job["shard"], "mod": job["mod"], "files": out_rows, "seconds": time.perf_counter()-t0}


def canonical_set(sources, strengths):
    """Unordered source set stored in lexicographic (x, y, z) order; the order carries no meaning."""
    o = np.lexsort(sources.T[::-1])
    return sources[o], strengths[o]


def export(release, name, profile="P256", scenes_per_shard=16, workers=12, limit_scenes=None):
    from dataclasses import asdict
    from .scenes import canary_scenes, v01_scenes
    from .selection import load_index, weights
    t0 = time.perf_counter()
    rel = {"canary": "canary-v0.2", "v01": "v0.1-1k"}[release]
    scenes_meta, rec, reports, _ = load_index(rel, profile)
    refs = canary_scenes() if release == "canary" else v01_scenes()
    assert [r.scene_id for r in refs] == [s["scene_id"] for s in scenes_meta]
    key = private_secret("export_key", create=True)
    w_g, used_g, _ = weights(rec, reports, "global")
    w_x, used_x, _ = weights(rec, reports, "src-excluded")
    p102, p256 = in_profile(rec, "P102"), in_profile(rec, profile)
    sid = np.array([keyed_int(key, name, "sample", scenes_meta[s]["scene_id"], int(w)) for s, w in zip(rec["scene"], rec["window"])], np.uint64)
    if len(np.unique(sid)) != len(sid):
        raise AssertionError("opaque sample id collision")
    grp = {s["ordinal"]: keyed_int(key, name, "group", s["scene_id"], bits=32) for s in scenes_meta}
    priv = OUT/"privileged"/"export"/name
    priv.mkdir(parents=True, exist_ok=True)
    (OUT/"privileged").chmod(0o700)
    join = np.zeros(len(rec), PRIV_JOIN)
    labels = {}
    for so, sc in enumerate(refs):
        s, a = sc.privileged_labels()
        if sc.K > 1:
            s, a = canonical_set(s, a)
        labels[so] = (s, a)
    for f in ("L", "x", "y", "z", "subset", "rank", "contains_source", "no_free", "category", "free", "A_informative",
              "A_extreme", "A_rare", "nonlocal_flag"):
        join[f] = rec[f]
    join["sample_id"], join["scene"], join["record"] = sid, rec["scene"], rec["window"]
    join["in_P102"], join["in_P256"] = p102, p256
    join["w_global_mixture"] = np.where(used_g, w_g, 0.0)
    join["w_srcexcl_mixture"] = np.where(used_x, w_x, 0.0)
    for i, so in enumerate(rec["scene"]):
        sc = refs[so]
        s, a = labels[int(so)]
        join["split"][i], join["modality"][i], join["K"][i] = SPLITS[sc.split], MOD[sc.profile], sc.K
        src = np.full((4, 3), np.nan)
        st = np.full(4, np.nan)
        src[:sc.K], st[:sc.K] = s, a
        join["sources"][i], join["strengths"][i] = src, st
    np.save(priv/"join.npy", join)
    (priv/"scenes.json").write_text(json.dumps([{**m, "release": rel} for m in scenes_meta], indent=0))
    # jobs: per split and modality, scenes grouped by a private permutation, windows interleaved privately
    rng = np.random.default_rng(keyed_int(key, name, "shuffle"))
    jobs = []
    order_all = rng.permutation(len(rec))
    by_scene = {}
    for i in order_all:
        by_scene.setdefault(int(rec["scene"][i]), []).append(int(i))
    shard = 0
    for split in SPLITS:
        for mod in MOD:
            ss = [k for k, sc in enumerate(refs) if sc.split == split and sc.profile == mod]
            if limit_scenes:
                ss = ss[:limit_scenes]
            ss = [ss[i] for i in rng.permutation(len(ss))]
            out = OUT/"student"/name/split/"shards"
            out.mkdir(parents=True, exist_ok=True)
            for g in range(0, len(ss), scenes_per_shard):
                group = ss[g:g+scenes_per_shard]
                items = [i for so in group for i in by_scene[so]]
                items = [items[j] for j in rng.permutation(len(items))]
                sc_args = {so: {k: (v if not isinstance(v, Path) else v) for k, v in asdict(refs[so]).items()} for so in group}
                jobs.append(({"shard": shard, "mod": mod, "split": split,
                              "scenes": [(so, sc_args[so], None) for so in group],
                              "order": [(int(rec["scene"][i]), int(rec["window"][i]), int(rec["L"][i]),
                                         (int(rec["x"][i]), int(rec["y"][i]), int(rec["z"][i])), int(sid[i]), int(grp[int(rec["scene"][i])]))
                                        for i in items]}, out))
                shard += 1
    res = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_shard_job, jobs):
            res.append(r)
    # student manifests
    for split in SPLITS:
        d = OUT/"student"/name/split
        if not d.exists():
            continue
        files = {}
        counts = {}
        for p in sorted((d/"shards").iterdir()):
            files[f"shards/{p.name}"] = {"bytes": p.stat().st_size, "SHA256": sha256_file(p)}
            if p.name.endswith(".idx.npy"):
                idx = np.load(p)
                k = p.name.split("-")[0] + "-" + p.name.split("-")[1]
                counts[k] = counts.get(k, 0) + len(idx)
        man = {"schema": STUDENT_SCHEMA, "export": name, "split": split,
               "features": list(STUDENT_FEATURES),
               "non_features": {"sample_id": "opaque 64-bit join key (HMAC of a private key); never a model input",
                                "group": "opaque 32-bit per-scene token for scene-balanced sampling only; never a model input"},
               "encoding": {"SB": "response and SOLID: packed bits ZYX, little bit order along X, zstd",
                            "MA": "response: float64 local palette (ascending, values present in the window only) + packed bit planes of the palette code, zstd; SOLID as SB"},
               "coordinates": "receiver (i,j,k) of a window at origin o, side L has world centre (o + (i,j,k) + 1/2)/1024 (XYZ); arrays ZYX",
               "modality": {"0": "SB single-source binary", "1": "MA multi-source additive"},
               "samples": counts, "files": files}
        (d/"STUDENT_MANIFEST.json").write_text(json.dumps(man, indent=1))
    summary = {"schema": PRIVILEGED_SCHEMA, "export": name, "release": rel, "index_profile": profile, "samples": int(len(rec)),
               "in_P102": int(p102.sum()), "jobs": len(jobs), "wall_seconds": time.perf_counter()-t0, "workers": workers,
               "student_bytes": int(sum(p.stat().st_size for p in (OUT/"student"/name).rglob("*") if p.is_file())),
               "shard_seconds_total": float(sum(r["seconds"] for r in res))}
    (priv/"export_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("release", choices=["canary", "v01"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--profile", default="P256")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes-per-shard", type=int, default=16)
    a = ap.parse_args()
    export(a.release, a.name, a.profile, a.scenes_per_shard, a.workers)


if __name__ == "__main__":
    main()
