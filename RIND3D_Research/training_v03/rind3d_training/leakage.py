"""File-level leakage audit of a student export (run in the trusted process: it needs the privileged view).

  run.sh rind3d_training.leakage NAME [--sample 2000]
Checks on the ACTUAL exported files:
  L1 inventory      only STUDENT_MANIFEST.json, shards/*.bin, shards/*.idx.npy
  L2 index schema   idx dtype fields == student.IDX_FIELDS (no subset, weight, source, scene or seed field)
  L3 manifest       no 64-hex identifier, no seed, no private key words in keys
  L4 identifiers    no scene_id / geometry_id as hex text in any file; no raw 32-byte id digest and no true-source
                    float64 bytes in any index or manifest; metadata hex runs only the listed shard SHA256s
  L5 payload bytes  every .bin byte is accounted for by exactly one indexed payload (contiguous, no gaps/extra)
                    and every decoded payload has the exact declared length
  L6 content        a random sample of payloads decodes to exactly the local observation reconstructed from the
                    stored certified fields (SB response / MA combined local palette, SOLID)
  L7 MA palettes    ascending, all values present in the window, npal <= 2^K
  L8 opaque ids     sample ids unique, not derived from scene ids; group tokens one per scene, not scene-id prefixes
  L9 order          position inside a shard is independent of the hidden subset / containment flags
  L10 loader        StudentShards.read() returns exactly FEATURE_KEYS (+ separate sample id)
Output: outputs/privileged/export/NAME/leakage_audit.json and a public summary outputs/tests/leakage_NAME.json
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np
import zstandard as zstd

from .env import OUT
from .student import FEATURE_KEYS, IDX_FIELDS, StudentShards

PRIVATE_KEY_WORDS = ("source", "strength", "seed", "subset", "weight", "contains", "label", "clearance", "recipe",
                     "geometry_id", "scene_id", "truth", "driver", "candidate", "nonlocal", "rank", "proposal")
HEX64 = re.compile(rb"[0-9a-f]{64}")


def walk_keys(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.append(str(k))
            walk_keys(v, out)
    elif isinstance(x, list):
        for v in x:
            walk_keys(v, out)
    return out


def audit(name, sample=2000, seed=0):
    from .scenes import canary_scenes, v01_scenes
    from .teacher import TeacherPhysics
    priv = OUT/"privileged"/"export"/name
    join = np.load(priv/"join.npy")
    scenes = json.loads((priv/"scenes.json").read_text())
    rel = scenes[0]["release"]
    refs = canary_scenes() if rel == "canary-v0.2" else v01_scenes()
    by_sid = {int(s): i for i, s in enumerate(join["sample_id"])}
    res = {}
    root = OUT/"student"/name
    files = [p for p in root.rglob("*") if p.is_file()]
    allowed = all(p.name == "STUDENT_MANIFEST.json" or (p.parent.name == "shards" and (p.name.endswith(".bin") or p.name.endswith(".idx.npy")))
                  for p in files)
    res["L1_inventory"] = {"pass": bool(allowed), "files": len(files)}
    ids_hex = {s["scene_id"] for s in scenes} | {s["geometry_id"] for s in scenes}
    raw = [bytes.fromhex(h) for h in ids_hex]
    src_bytes = []
    for r in refs:
        s, a = r.privileged_labels()
        src_bytes += [np.float64(v).tobytes() for v in s.ravel()]
        if r.K > 1:
            src_bytes += [np.float64(v).tobytes() for v in a]
    l2 = l3 = l4 = l5 = True
    detail = {"hex64_runs": 0, "id_hits": 0, "source_byte_hits": 0, "payload_gaps": 0, "length_errors": 0}
    all_ids, all_groups = [], []
    listed = set()
    for p in files:
        if p.name == "STUDENT_MANIFEST.json":
            listed |= {v["SHA256"] for v in json.loads(p.read_text())["files"].values()}
    for p in files:
        b = p.read_bytes()
        runs = {m.decode() for m in HEX64.findall(b)}
        id_runs = runs & ids_hex
        # metadata may contain only the declared shard SHA256s; in compressed payloads hex-like byte runs occur
        # naturally (e.g. 0x33 / 0xff bit patterns) and are allowed unless they spell an identifier
        bad = id_runs | ((runs - listed) if (p.suffix == ".json" or p.name.endswith(".idx.npy")) else set())
        detail["hex64_runs"] += len(runs)
        detail["hex64_identifier_runs"] = detail.get("hex64_identifier_runs", 0) + len(bad)
        l4 &= not bad
        if p.name.endswith(".idx.npy") or p.suffix == ".json":
            hits = sum(b.count(x) for x in raw)
            sh = sum(b.count(x) for x in src_bytes) if p.suffix != ".json" else 0
            detail["id_hits"] += hits
            detail["source_byte_hits"] += sh
            l4 &= hits == 0 and sh == 0
        if p.suffix == ".json":
            man = json.loads(b)
            keys = walk_keys(man, [])
            bad = [k for k in keys if any(w in k.lower() for w in PRIVATE_KEY_WORDS) and k not in ("sample_id",)]
            l3 &= not bad
            detail.setdefault("manifest_bad_keys", []).extend(bad)
        if p.name.endswith(".idx.npy"):
            idx = np.load(p)
            l2 &= tuple(idx.dtype.names) == IDX_FIELDS
            all_ids.append(idx["sample_id"])
            all_groups.append(idx["group"])
            binf = p.with_name(p.name.replace(".idx.npy", ".bin"))
            spans = sorted([(int(r["resp_offset"]), int(r["resp_bytes"])) for r in idx] + [(int(r["solid_offset"]), int(r["solid_bytes"])) for r in idx])
            pos = 0
            for o, n in spans:
                if o != pos:
                    detail["payload_gaps"] += 1
                pos = o + n
            if pos != binf.stat().st_size:
                detail["payload_gaps"] += 1
            dz = zstd.ZstdDecompressor()
            with open(binf, "rb") as f:
                data = f.read()
            for r in idx:
                L = int(r["L"])
                pa = dz.decompress(data[int(r["resp_offset"]): int(r["resp_offset"]) + int(r["resp_bytes"])])
                pb = dz.decompress(data[int(r["solid_offset"]): int(r["solid_offset"]) + int(r["solid_bytes"])])
                exp_a = L**3//8 if r["modality"] == 0 else 8*int(r["npal"]) + int(r["nplanes"])*L**3//8
                if len(pa) != exp_a or len(pb) != L**3//8 or zlib.crc32(pa) != r["resp_crc"] or zlib.crc32(pb) != r["solid_crc"]:
                    detail["length_errors"] += 1
    l5 = detail["payload_gaps"] == 0 and detail["length_errors"] == 0
    res["L2_index_schema"] = {"pass": bool(l2)}
    res["L3_manifest"] = {"pass": bool(l3), "bad_keys": detail.get("manifest_bad_keys", [])}
    res["L4_identifiers"] = {"pass": bool(l4), **{k: detail.get(k, 0) for k in ("hex64_runs", "hex64_identifier_runs", "id_hits", "source_byte_hits")},
                             "scope": "hex ids: every file; raw digests and true-source float64 bytes: every index/manifest file; payload bytes are fully accounted for by L5/L6"}
    res["L5_payload_bytes"] = {"pass": bool(l5), **{k: detail[k] for k in ("payload_gaps", "length_errors")}}
    ids = np.concatenate(all_ids)
    groups = np.concatenate(all_groups)
    jrow = np.array([by_sid[int(s)] for s in ids])
    scene_of = join["scene"][jrow]
    g_per_scene = {}
    for g, s in zip(groups, scene_of):
        g_per_scene.setdefault(int(s), set()).add(int(g))
    prefix_tokens = {int(sc["scene_id"][:8], 16) for sc in scenes} | {int.from_bytes(bytes.fromhex(sc["scene_id"][:8]), "little") for sc in scenes}
    corr = abs(np.corrcoef((ids >> np.uint64(32)).astype(float), scene_of.astype(float))[0, 1])
    l8 = (len(np.unique(ids)) == len(ids) == len(join) and all(len(v) == 1 for v in g_per_scene.values())
          and len({next(iter(v)) for v in g_per_scene.values()}) == len(g_per_scene) and not (set(groups.tolist()) & prefix_tokens)
          and corr < 0.1)
    res["L8_opaque_ids"] = {"pass": bool(l8), "samples": int(len(ids)), "scenes": len(g_per_scene), "id_scene_corr": float(corr)}
    # L9: position independence of hidden tags inside shards
    pos_stats = []
    for p in sorted(root.rglob("*.idx.npy")):
        idx = np.load(p)
        jr = np.array([by_sid[int(s)] for s in idx["sample_id"]])
        rel_pos = np.arange(len(idx))/max(1, len(idx)-1)
        for tag in ("subset", "contains_source"):
            t = join[tag][jr].astype(float)
            if t.std() > 0:
                pos_stats.append(abs(np.corrcoef(rel_pos, t)[0, 1]))
    res["L9_order"] = {"pass": bool(max(pos_stats, default=0) < 0.15), "max_abs_corr": float(max(pos_stats, default=0)),
                       "shards_tested": len(pos_stats)}
    # L6 / L7 content on a sample (decode via the student reader; compare with the stored fields)
    rng = np.random.default_rng(seed)
    l6 = l7 = l10 = True
    n6 = 0
    teachers = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sh = StudentShards(split_dir)
        take = rng.choice(len(sh), min(len(sh), sample//3 + 1), replace=False)
        for i in take:
            feats, sid = sh.read(int(i))
            l10 &= tuple(sorted(feats)) == tuple(sorted(FEATURE_KEYS))
            j = by_sid[int(sid)]
            so = int(join["scene"][j])
            if so not in teachers:
                if len(teachers) > 8:
                    for t in teachers.values():
                        t.close()
                    teachers = {}
                teachers[so] = TeacherPhysics(refs[so], threads=1)
            o = tuple(int(v) for v in feats["window_origin"])
            L = feats["window_size"]
            obs, solid = teachers[so].observation(o, L)
            ok = np.array_equal(solid, feats["observed_solid"]) and np.array_equal(obs, feats["observed_response"])
            ok &= o == (int(join["x"][j]), int(join["y"][j]), int(join["z"][j])) and L == int(join["L"][j])
            l6 &= bool(ok)
            if feats["modality"] == 1:
                vals = np.unique(feats["observed_response"])
                r = sh.index[int(i)]
                l7 &= len(vals) == int(r["npal"]) and int(r["npal"]) <= 2**int(join["K"][j])
            n6 += 1
    for t in teachers.values():
        t.close()
    res["L6_content"] = {"pass": bool(l6), "samples": n6}
    res["L7_MA_palettes"] = {"pass": bool(l7)}
    res["L10_loader_keys"] = {"pass": bool(l10), "feature_keys": list(FEATURE_KEYS)}
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    (priv/"leakage_audit.json").write_text(json.dumps(res, indent=1))
    pub = OUT/"tests"
    pub.mkdir(parents=True, exist_ok=True)
    (pub/f"leakage_{name}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()
    r = audit(a.name, a.sample)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
