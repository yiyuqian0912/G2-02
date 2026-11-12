"""Independent validation of a production root (separate process; re-derives everything from stored files).

  run.sh rind3d_production.validate ROOT [--determinism SLOT SLOT]
Checks (contract canary_gates C2/C3/C5 and verification.validation):
 V1  shard MANIFEST: every file present with its size and SHA256; no unlisted file
 V2  shard index consistent with per-scene public windows and scene records; slot records complete
 V3  geometry: canonical roundtrip + convex-interior/world-dual proof re-verified; geometry_id
 V4  sources: certified legality re-proved (FREE, clearance > 2^-26), separation >= 1/32, strengths range, K/profile
 V5  fields: every RINDPack brick CRC (verify_all), identities bound to (geometry, source, policy), SOLID equal
 V6  windows: every record recounted exactly (CountIndex rebuilt from the stored fields); subset rules
 V7  physics: independent projection re-render of 4 random 64^3 bricks per source == stored response + SOLID
 V8  MA public observations == sum_k a_k lit_k from private channels, local palettes only
 V9  public/private separation (declared public files only; no private keys in public JSON)
 V10 splits: geometry-group split, unique scene and geometry ids
 V11 determinism (optional): re-run slots in a temporary root; scientific files byte-identical
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from rind3d.counts import CountIndex
from rind3d.geometry import load_geometry
from rind3d.lite_acceptance import verify_geometry_proof
from rind3d.pack import RINDPack
from rind3d.rendering.context import GeometryContext
from rind3d.rendering.polyhedra import geometry_split
from rind3d.stream import scene_identity

from . import scenes as S
from . import windows as WS
from .env import POLICY, sha256_file
from .production import PROFILES, combined_window

PUBLIC_SB = {"geometry.npz", "solid.rindb", "solid.rindb.sha256", "response.rindb", "response.rindb.sha256", "windows.npy", "scene.json"}
PUBLIC_MA = {"geometry.npz", "solid.rindb", "solid.rindb.sha256", "observations.bin", "observations_index.npy", "windows.npy", "scene.json"}
PRIVATE_WORDS = ("source", "strength", "clearance", "recipe", "seed", "subset", "nonlocal", "candidate")


def check_scene(d, rng):
    ok = {}
    pub = json.loads((d/"public"/"scene.json").read_text())
    src = json.loads((d/"private"/"sources.json").read_text())
    bundle = load_geometry(d/"public"/"geometry.npz")
    ctx = GeometryContext(bundle, POLICY)
    verify_geometry_proof(bundle, ctx.geometry_proof())
    ok["V3_geometry"] = ctx.geometry_id == pub["geometry_id"]
    K = src["K"]
    sources = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    strengths = np.array([float.fromhex(v) for v in src["strengths_hex"]])
    prof = "SB" if pub["profile"] == PROFILES["SB"] else "MA"
    legal = all(S.legal_source(ctx, s) is not None for s in sources)
    sep = all(np.linalg.norm(sources[i]-sources[j]) >= S.MIN_SEPARATION for i in range(K) for j in range(i))
    rng_ok = bool(np.all(strengths == 1.0)) if prof == "SB" else bool(np.all((strengths >= .1) & (strengths <= 1)))
    ok["V4_sources"] = legal and sep and rng_ok and (K == 1) == (prof == "SB") and pub["scene_id"] == S.scene_id(pub["profile"], ctx.geometry_id, sources, strengths)
    paths = [d/"public"/"response.rindb"] if prof == "SB" else [d/"private"/f"src{k}"/"response.rindb" for k in range(K)]
    solids = [d/"public"/"solid.rindb"] + [d/"private"/f"src{k}"/"solid.rindb" for k in range(1, K)]
    v5 = True
    crc = None
    for k in range(K):
        with RINDPack(paths[k], verify_file=True) as y, RINDPack(solids[k], verify_file=True) as m:
            y.verify_all()
            m.verify_all()
            v5 &= y.identity == scene_identity(ctx, sources[k]) == m.identity and y.policy == ctx.policy_hash
            c = [(r[5], r[7]) for r in m.index]
            v5 &= crc is None or c == crc
            crc = c
    ok["V5_fields"] = bool(v5)
    recs = np.load(d/"private"/"window_records.npy")
    packs = [RINDPack(p, cache_bricks=64) for p in paths]
    sol = [RINDPack(p, cache_bricks=64) for p in solids]
    counts = [CountIndex.build(packs[k], sol[k]) for k in range(K)]
    v6 = True
    for r in recs:
        o, L = (int(r["x"]), int(r["y"]), int(r["z"])), int(r["L"])
        free, solid, lit = WS.window_counts(counts, o, L)
        v6 &= free == r["free"] and solid == r["solid"] and list(lit) == list(r["lit"][:K])
        mf, mc = WS.minority(free, lit)
        cs = WS.contains_source(o, L, sources)
        v6 &= bool(cs) == bool(r["contains_source"])
        sub = int(r["subset"])
        if sub >= 1:
            v6 &= mf > 0 and not cs
        if sub == 2:
            v6 &= mf <= WS.EXTREME
        if sub == 3:
            v6 &= mc <= WS.RARE
    rep = json.loads((d/"private"/"window_report.json").read_text())
    for L in WS.SIDES:
        n_nat = int(((recs["subset"] == 0) & (recs["L"] == L)).sum())
        v6 &= n_nat == WS.QUOTA["natural"] and rep[f"natural/{L}"]["proposals"] == n_nat
    ok["V6_windows"] = bool(v6)
    v7 = True
    for k in range(K):
        model = ctx.bind(sources[k], independent=True)
        for _ in range(4):
            b = tuple(int(v) for v in rng.integers(0, 16, 3))
            y, m, _ = model.render([64*b[0], 64*b[1], 64*b[2]], 64, 1024)
            v7 &= bool(np.array_equal(y, packs[k].read_global_brick(b)) and np.array_equal(m, sol[k].read_global_brick(b)))
    ok["V7_independent_rerender"] = bool(v7)
    if prof == "MA":
        idx = np.load(d/"public"/"observations_index.npy")
        import zstandard as zstd
        v8 = True
        with open(d/"public"/"observations.bin", "rb") as f:
            for r, (off, npal, nb) in zip(recs, idx):
                o, L = (int(r["x"]), int(r["y"]), int(r["z"])), int(r["L"])
                f.seek(int(off))
                pal = np.frombuffer(f.read(8*int(npal)), "<f8")
                codes = np.frombuffer(zstd.ZstdDecompressor().decompress(f.read(int(nb))), np.uint8).reshape(L, L, L)
                pal2, codes2 = combined_window(packs, strengths, sol[0], o, L)
                v8 &= bool(np.array_equal(pal, pal2) and np.array_equal(codes, codes2) and len(np.unique(codes)) == len(pal))
        ok["V8_multi_observations"] = bool(v8)
    for p in packs + sol:
        p.close()
    names = {p.name for p in (d/"public").iterdir()}
    leak = any(w in json.dumps(pub).lower() for w in PRIVATE_WORDS)
    ok["V9_public_private"] = names == (PUBLIC_SB if prof == "SB" else PUBLIC_MA) and not leak and pub["split"] == geometry_split(ctx.geometry_id, 0)
    return ok, {"scene_id": pub["scene_id"], "geometry_id": ctx.geometry_id, "split": pub["split"], "profile": prof, "K": K,
                "family": pub["family"]}


def validate(root, rng_seed=20261203):
    root = Path(root)
    rng = np.random.default_rng(rng_seed)
    results, scenes, shard_checks = [], [], []
    for shard in sorted((root/"shards").glob("shard-*")):
        man = json.loads((shard/"MANIFEST.json").read_text())
        listed = set(man["files"])
        present = {str(p.relative_to(shard)) for p in shard.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
        v1 = listed == present and all((shard/k).stat().st_size == v["bytes"] and sha256_file(shard/k) == v["SHA256"] for k, v in man["files"].items())
        rows = [json.loads(l) for l in (shard/"index"/"scenes.jsonl").read_text().splitlines()]
        win = np.load(shard/"index"/"windows.npy")
        v2 = True
        for r in rows:
            pw = np.load(shard/"scenes"/r["scene_id"]/"public"/"windows.npy")
            sel = win[win["scene_ordinal"] == r["ordinal"]]
            v2 &= len(sel) == len(pw) and all(np.array_equal(sel[k], pw[k]) for k in ("x", "y", "z", "L"))
        slots = json.loads((shard/"index"/"slots.json").read_text())["slots"]
        v2 &= len(rows) == sum(s["status"] == "SEALED" for s in slots)
        shard_checks.append({"shard": shard.name, "V1_manifest": bool(v1), "V2_index": bool(v2), "scenes": len(rows),
                             "failed_slots": [s["slot"] for s in slots if s["status"] != "SEALED"]})
        for r in rows:
            t = time.perf_counter()
            ok, meta = check_scene(shard/"scenes"/r["scene_id"], rng)
            results.append({**meta, **ok, "seconds": time.perf_counter()-t})
            scenes.append(meta)
            if not all(ok.values()):
                print("FAIL", r["scene_id"][:12], {k: v for k, v in ok.items() if not v}, flush=True)
    gids = [s["geometry_id"] for s in scenes]
    split_of = {}
    for s in scenes:
        split_of.setdefault(s["geometry_id"], set()).add(s["split"])
    v10 = len(set(s["scene_id"] for s in scenes)) == len(scenes) and len(set(gids)) == len(gids) and all(len(v) == 1 for v in split_of.values())
    keys = sorted({k for r in results for k in r if k.startswith("V")})          # union: V8 exists for MA scenes only
    summary = {"scenes": len(results), "shards": shard_checks, "V10_splits": bool(v10),
               "passed": {k: int(sum(bool(r[k]) for r in results if k in r)) for k in keys},
               "applicable": {k: int(sum(k in r for r in results)) for k in keys},
               "all_pass": bool(all(all(r[k] for k in r if k.startswith("V")) for r in results) and v10
                                and all(s["V1_manifest"] and s["V2_index"] for s in shard_checks)),
               "split_counts": {s: sum(1 for x in scenes if x["split"] == s) for s in ("train", "val", "test")},
               "validation_seconds": float(sum(r["seconds"] for r in results))}
    return summary, results


def determinism(root, slots):
    """Re-run slots in a fresh temporary directory; compare every scientific file of the sealed scene."""
    from rind3d.rendering.backends import BackendFactory
    from .production import run_scene
    cfg = json.loads((Path(root)/"config.json").read_text())
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="rind3d-determinism-"))
    try:
        with BackendFactory(geometry_cache=1) as f:
            for slot in slots:
                stage, rec = run_scene(cfg, slot, tmp, f)
                orig = next(Path(root).glob(f"shards/*/scenes/{rec['scene_id']}"))
                seal = json.loads((orig/"SEAL.json").read_text())
                sci = [k for k in rec["files"] if k.endswith((".rindb", ".npz", ".npy", ".bin", ".rindc", ".rindi"))]
                same = all(rec["files"][k]["SHA256"] == seal["files"][k]["SHA256"] for k in sci)
                out.append({"slot": slot, "scene_id": rec["scene_id"], "scientific_files": len(sci), "identical": bool(same)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--determinism", type=int, nargs="*", default=[])
    a = ap.parse_args()
    summary, results = validate(a.root)
    if a.determinism:
        summary["V11_determinism"] = determinism(a.root, a.determinism)
    Path(a.root, "validation.json").write_text(json.dumps({"summary": summary, "scenes": results}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
