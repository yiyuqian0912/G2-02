"""EVIDENCE_MANIFEST.json (SHA256 of deliverables, code, configs, tests, docs and JSON evidence) and the compact
review archive. Never included: ./private (secrets), outputs/privileged (labels, weights, selection law),
student payload shards (.bin/.npy), per-file preservation rows (they list restricted v0.1 label files), .venv-gpu.

  run.sh rind3d_training.evidence [--archive PATH]
"""
import argparse
import json
import zipfile
from pathlib import Path

from .env import OUT, ROOT, sha256_file

TOP = ["README.md", "RIND3D_TRAINING_READY_DECISION.md", "RIND3D_TRAINING_CONTRACT.json", "RIND3D_STUDENT_TARGET_SCHEMA.md",
       "RIND3D_DATASET_TEACHER_INTEROP.md", "RIND3D_LOADER_BENCHMARKS.json", "RIND3D_102_VS_256_WINDOW_STUDY.md",
       "RIND3D_TRAINING_SCALE_COST.json", "run.sh", "run_gpu.sh"]
SUBDIRS = ("rind3d_training", "tests", "configs", "docs")
EVIDENCE_DIRS = ("bench", "study", "tests", "public_summaries")


def files():
    out = [ROOT/p for p in TOP if (ROOT/p).exists()]
    for sub in SUBDIRS:
        out += sorted(p for p in (ROOT/sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for sub in EVIDENCE_DIRS:
        d = OUT/sub
        if d.exists():
            out += sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in (".json", ".txt", ".md", ".csv"))
    if (OUT/"preservation"/"compare.json").exists():
        out.append(OUT/"preservation"/"compare.json")
    out += sorted((OUT/"student").glob("*/*/STUDENT_MANIFEST.json"))
    out += sorted((OUT/"logs").glob("*.log"))
    for p in out:
        rel = p.relative_to(ROOT).parts
        assert "privileged" not in rel and "private" not in rel and p.suffix not in (".bin", ".npy", ".secret"), p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    a = ap.parse_args()
    fl = files()
    rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "SHA256": sha256_file(p)} for p in fl]
    data = {}
    for man in sorted((OUT/"student").glob("*/*/STUDENT_MANIFEST.json")):
        m = json.loads(man.read_text())
        data[str(man.parent.relative_to(OUT))] = {"MANIFEST_SHA256": sha256_file(man), "files": len(m["files"]),
                                                  "bytes": sum(v["bytes"] for v in m["files"].values()), "samples": m["samples"]}
    out = {"schema": "rind3d-evidence-manifest-v1", "candidate": "rind3d-training-contract-v0.3", "files": rows,
           "student_exports": data,
           "note": "student payload shards are covered by their STUDENT_MANIFEST.json SHA256 lists; privileged views and private keys are deliberately excluded"}
    (ROOT/"EVIDENCE_MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(rows), "files;", len(data), "student exports")
    if a.archive:
        with zipfile.ZipFile(a.archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in fl + [ROOT/"EVIDENCE_MANIFEST.json"]:
                z.write(p, f"RIND3D_TRAINING_READY_REVIEW/{p.relative_to(ROOT)}")
        print(a.archive, Path(a.archive).stat().st_size, sha256_file(a.archive))


if __name__ == "__main__":
    main()
