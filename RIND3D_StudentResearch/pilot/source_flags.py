"""Privileged analysis flags per (scene, source): is SOLID within 8 / 32 voxels (inf-norm) of the true source?
Read from the stored 1024^3 SOLID field (read-only). Used only to stratify evaluation (F_OOD_BENCHMARK.md stratum E).
usage: source_flags.py <out.json>   (labeller venv: rind3d.pack)"""
import os
import json, sys
from pathlib import Path
import numpy as np
from rind3d.pack import RINDPack

REL = (Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "release_1k_v1")
out = {}
for d in sorted((REL / "scenes").iterdir()):
    if not (d / "SEAL.json").exists():
        continue
    src = json.loads((d / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    with RINDPack(d / "privileged" / "fields" / "src0" / "solid.rindb") as m:
        for j, s in enumerate(S):
            v = np.floor(s * 1024).astype(int)
            res = {}
            for r in (8, 32):
                lo = np.clip(v - r, 0, 1024 - 2 * r - 1)
                blk = m.read_window(int(lo[0]), int(lo[1]), int(lo[2]), 2 * r + 1)      # [Z, Y, X]
                a = np.maximum(v - r, 0) - lo; b = np.minimum(v + r, 1023) - lo + 1     # true neighbourhood in block coords
                res[f"solid_within_{r}"] = bool(blk[a[2]:b[2], a[1]:b[1], a[0]:b[0]].any())
            out[f"{d.name}:{j}"] = res
Path(sys.argv[1]).write_text(json.dumps(out))
print(len(out), sum(v["solid_within_8"] for v in out.values()), sum(v["solid_within_32"] for v in out.values()))
