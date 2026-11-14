"""Integrity snapshot of release_1k_v1 around the FastLabel backfill (read-only).
usage: integrity_snapshot.py <release_root> <out.json>
* verifies every scene's SEAL.json (sha256 of every sealed physical file);
* records sha256 of every existing label task file (so preserved labels can be checked afterwards);
* counts temporary files left by interrupted writers."""
import hashlib, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scene(d):
    seal = json.loads((d / "SEAL.json").read_text())
    bad = [k for k, v in seal.items() if not (d / k).exists() or sha(d / k) != v]
    labs = {p.name: sha(p) for p in sorted((d / "labels" / "tasks").glob("w*_s*.npz"))} if (d / "labels" / "tasks").exists() else {}
    tmp = [p.name for p in (d / "labels").rglob("*.tmp*")] if (d / "labels").exists() else []
    return d.name, dict(sealed=len(seal), seal_mismatch=bad, labels=labs, tmp=tmp)


root = Path(sys.argv[1])
dirs = sorted(p for p in (root / "scenes").iterdir() if (p / "SEAL.json").exists())
with ThreadPoolExecutor(4) as ex:
    res = dict(ex.map(scene, dirs))
summ = dict(scenes=len(res), sealed_files=sum(r["sealed"] for r in res.values()),
            seal_mismatches=sum(len(r["seal_mismatch"]) for r in res.values()),
            label_files=sum(len(r["labels"]) for r in res.values()), tmp_files=sum(len(r["tmp"]) for r in res.values()))
Path(sys.argv[2]).write_text(json.dumps(dict(summary=summ, scenes=res)))
print(json.dumps(summ))
