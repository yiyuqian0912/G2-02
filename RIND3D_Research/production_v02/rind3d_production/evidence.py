"""EVIDENCE_MANIFEST.json (SHA256 of every deliverable, code file, config and evidence record) and the compact
review archive (deliverables, code, configs, tests, JSON/text evidence; no fields, meshes or binaries).

  run.sh rind3d_production.evidence [--archive PATH]
"""
import argparse
import json
import zipfile
from pathlib import Path

from .env import ROOT, sha256_file

EXCLUDE_SUFFIX = (".rindb", ".rindc", ".rindi", ".npz", ".npy", ".bin", ".pyc", ".sha256", ".tmp")
INCLUDE_TOP = ["README.md", "RIND3D_PRODUCTION_DECISION.md", "RIND3D_PRODUCTION_CONTRACT.json", "RIND3D_BACKEND_BENCHMARKS.json",
               "RIND3D_LIFECYCLE_COST_MODEL.json", "RIND3D_PRODUCTION_CANARY_RESULTS.json", "DATASET_TEACHER_INTERFACE_PROPOSAL.md", "run.sh"]


def files():
    out = [ROOT/p for p in INCLUDE_TOP if (ROOT/p).exists()]
    for sub in ("rind3d_production", "configs", "tests"):
        out += sorted(p for p in (ROOT/sub).rglob("*") if p.is_file() and "__pycache__" not in str(p))
    ev = ROOT/"outputs"
    for p in sorted(ev.rglob("*")):
        if not p.is_file() or p.suffix in EXCLUDE_SUFFIX or "__pycache__" in str(p) or p.name.endswith(".sha256"):
            continue
        rel = p.relative_to(ev)
        parts = rel.parts
        if parts[0] == "benchmarks" and "fields" in parts:
            continue
        if parts[0] == "canary" and ("scenes" in parts or "failed-attempts" in parts):
            if p.name not in ("SEAL.json", "scene.json", "window_report.json", "slots.json"):
                continue                                   # private source labels never enter the review archive
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    a = ap.parse_args()
    fl = files()
    rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "SHA256": sha256_file(p)} for p in fl]
    canary = ROOT/"outputs"/"canary"
    data = {}
    if canary.exists():
        for p in sorted(canary.glob("shards/*/MANIFEST.json")):
            m = json.loads(p.read_text())
            data[p.parent.name] = {"MANIFEST_SHA256": sha256_file(p), "files": len(m["files"]), "bytes": m["bytes"], "scenes": m["scenes"]}
    man = {"schema": "rind3d-evidence-manifest-v1", "files": rows, "canary_shards": data,
           "note": "canary scientific data (fields, geometry, windows) are covered by the per-shard MANIFEST.json files listed here"}
    (ROOT/"EVIDENCE_MANIFEST.json").write_text(json.dumps(man, indent=1)+"\n")
    print(len(rows), "files;", len(data), "shards")
    if a.archive:
        with zipfile.ZipFile(a.archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in fl + [ROOT/"EVIDENCE_MANIFEST.json"]:
                z.write(p, f"RIND3D_PRODUCTION_REVIEW/{p.relative_to(ROOT)}")
        print(a.archive, Path(a.archive).stat().st_size, sha256_file(a.archive))


if __name__ == "__main__":
    main()
