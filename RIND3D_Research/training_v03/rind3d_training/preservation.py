"""Preservation evidence: SHA256 trees of every read-only input, before and after the v0.3 work.

  run.sh rind3d_training.preservation snapshot before|after
  run.sh rind3d_training.preservation compare
Covered: the v0.2 production tree (code, deliverables, frozen contract, accepted canary), the restricted v0.1
campaign scene files, and the public v0.1 snapshot. Output: outputs/preservation/{before,after}.json and
outputs/preservation/compare.json. Only digests are written (no labels).
"""
import json
import sys
import time

from .env import CANARY, OUT, PRODUCTION, V01, V01_PUBLIC, sha256_file

TREES = {"v02_production_code": PRODUCTION/"rind3d_production", "v02_tests": PRODUCTION/"tests",
         "v02_configs": PRODUCTION/"configs", "v02_canary": CANARY, "v01_restricted_scenes": V01/"scenes",
         "v01_public_snapshot": V01_PUBLIC/"scenes"}
TOP = ["RIND3D_PRODUCTION_DECISION.md", "RIND3D_PRODUCTION_CONTRACT.json", "RIND3D_BACKEND_BENCHMARKS.json",
       "RIND3D_LIFECYCLE_COST_MODEL.json", "RIND3D_PRODUCTION_CANARY_RESULTS.json", "DATASET_TEACHER_INTERFACE_PROPOSAL.md",
       "EVIDENCE_MANIFEST.json", "README.md", "run.sh"]


def tree(root):
    rows = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rows[str(p.relative_to(root))] = sha256_file(p)
    return rows


def snapshot(tag):
    t = time.time()
    out = {"tag": tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "trees": {}}
    for k, root in TREES.items():
        rows = tree(root)
        out["trees"][k] = {"root": str(root), "files": len(rows), "digest": __import__("hashlib").sha256(
            json.dumps(rows, sort_keys=True).encode()).hexdigest(), "rows": rows}
    out["v02_top"] = {p: sha256_file(PRODUCTION/p) for p in TOP if (PRODUCTION/p).exists()}
    out["seconds"] = time.time() - t
    d = OUT/"preservation"
    d.mkdir(parents=True, exist_ok=True)
    (d/f"{tag}.json").write_text(json.dumps(out, indent=0))
    print(tag, {k: (v["files"], v["digest"][:16]) for k, v in out["trees"].items()}, f"{out['seconds']:.0f}s")


def compare():
    d = OUT/"preservation"
    a, b = (json.loads((d/f"{t}.json").read_text()) for t in ("before", "after"))
    res = {}
    for k in TREES:
        ra, rb = a["trees"][k]["rows"], b["trees"][k]["rows"]
        res[k] = {"files_before": len(ra), "files_after": len(rb), "digest_before": a["trees"][k]["digest"],
                  "digest_after": b["trees"][k]["digest"], "unchanged": ra == rb,
                  "added": sorted(set(rb) - set(ra))[:20], "removed": sorted(set(ra) - set(rb))[:20],
                  "modified": sorted(x for x in set(ra) & set(rb) if ra[x] != rb[x])[:20]}
    res["v02_top"] = {"unchanged": a["v02_top"] == b["v02_top"], "files": len(a["v02_top"])}
    res["all_unchanged"] = all(v["unchanged"] for v in res.values() if isinstance(v, dict))
    res["before_time"], res["after_time"] = a["time"], b["time"]
    (d/"compare.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["unchanged"] if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    else:
        compare()
