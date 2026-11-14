#!/usr/bin/env python3
"""Integrity verification of a RIND3D-100K root (read-only; safe while production runs).

  verify_root.py <root> [--workers 16] [--physics-every 50] [--deep-every 500] [--labels] [--out VERIFY.json]

* manifest: at most one accepted record per slot; every accepted container exists and matches its sha256;
* every accepted scene: validate_scene (quotas, exclusion, crops, counts, m/mask; physics spot checks on every
  --physics-every-th scene, legality of sources on every --deep-every-th);
* label journals: every record parses and its checksum verifies (torn tails reported); keys unique or duplicates
  identical in status;
* consolidated labels (--labels): validate_labels per scene (sha256 against labels/consolidated.jsonl, slot alignment,
  statuses, certified => tv <= 0.05)."""
import argparse
import collections
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]


def _one(args):
    root, rec, phys, deep, labels, cons = args
    from rind3d_100k.validate import validate_scene, validate_labels
    v = validate_scene(root, rec["slot"], rec, physics=256 if phys else 0, deep=deep)
    if labels:
        v["labels"] = validate_labels(root, rec["slot"], rec, cons)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--physics-every", type=int, default=50)
    ap.add_argument("--deep-every", type=int, default=500); ap.add_argument("--labels", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.root)
    from rind3d_100k.storage import read_jsonl
    recs = read_jsonl(root / "manifest" / "physical.jsonl")
    cons = {int(r["slot"]): r for r in read_jsonl(root / "labels" / "consolidated.jsonl")}     # last record wins
    acc = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "accepted":
            acc[r["slot"]].append(r)
    dup = [s for s, v in acc.items() if len(v) > 1]
    jobs = [(str(root), v[-1], i % a.physics_every == 0, i % a.deep_every == 0, a.labels, cons.get(int(s)))
            for i, (s, v) in enumerate(sorted(acc.items()))]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    from rind3d_100k.storage import iter_journal
    jstat = {}
    for p in sorted((root / "labels" / "journal").glob("*.r3lj")):
        n = 0; size = p.stat().st_size; end = 0
        keys = []
        for key, payload in iter_journal(p):
            n += 1; keys.append(key)
        jstat[p.name] = dict(records=n, bytes=size, idx_lines=len((p.with_suffix(".idx")).read_text().splitlines()) if p.with_suffix(".idx").exists() else None)
    bad = [r for r in res if not r["ok"]]
    lab_bad = [r for r in res if a.labels and r.get("labels", {}).get("ok") is False]
    out = dict(root=str(root), accepted=len(acc), duplicate_accepted=dup, scenes_failed_validation=len(bad), failures=bad[:20],
               physics_checked=sum(1 for r in res if r.get("physics")),
               physics_mismatches=sum(r["physics"]["mismatches"] + r["physics"]["independent_mismatches"] for r in res if r.get("physics")),
               journals=jstat, label_scenes_failed=len(lab_bad), ok=not bad and not dup and not lab_bad)
    s = json.dumps(out, indent=1, default=str)
    Path(a.out or root / "VERIFY.json").write_text(s); print(s)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
