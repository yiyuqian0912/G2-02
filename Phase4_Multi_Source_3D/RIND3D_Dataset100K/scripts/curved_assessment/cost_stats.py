"""Cost, storage and channel statistics for docs/CURVED_ASSESSMENT.md §2-3 (CPU only, reads existing run outputs).

* bench: per-group cost of the laptop per-family benchmark (scripts/bench_families.py output): mean / median wall,
  least-squares fit wall = a + b k, wall at the contract's E[k] = 3.1, per-tile render and per-probe cost.
* storage and channels: container sizes and the response-active fraction of informative channels, for the v3 polytope
  containers of the sampling-impact run and the curved containers of the curved-statistics run.
* 100K projection: T_phys(f) = N_s [(1 - f) t_P + f t_C] for curved fractions f.

Usage: python cost_stats.py [WORK_DIR]  (default $RIND3D_V3_WORK or /home/yushu/Projects/RIND/RIND3D_100K_v3_work)"""
import glob
import json
import os
import sys

import numpy as np

WORK = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RIND3D_V3_WORK", "/home/yushu/Projects/RIND/RIND3D_100K_v3_work")
KAPPA = [0.15, 0.15, 0.15, 0.55]
EK = sum((i + 1) * p for i, p in enumerate(KAPPA))


def bench():
    G = {}
    for f in sorted(glob.glob(f"{WORK}/bench/*.jsonl")):
        for line in open(f):
            r = json.loads(line); G.setdefault("polytope (9 v3 families)" if os.path.basename(f).startswith("v3_") else r["family"], []).append(r)
    out = {}
    for g, R in G.items():
        w = np.array([r["wall"] for r in R]); k = np.array([r["k"] for r in R]); W = [r["work"] for r in R]
        a, b = np.linalg.lstsq(np.c_[np.ones(len(k)), k], w, rcond=None)[0]
        tiles = sum(x["tiles"] for x in W)
        out[g] = dict(n=len(R), mean=w.mean(), median=np.median(w), mean_k=k.mean(), fit=(a, b), at_Ek=a + b * EK,
                      ratio_at_Ek=w.sum() / k.sum() * EK, render_ms_per_tile=1e3 * sum(x["render_s"] for x in W) / tiles,
                      probe_us=1e6 * sum(x["probe_s"] for x in W) / sum(x["probes"] for x in W),
                      geometry_s=np.mean([r["timing"]["geometry"] for r in R]), rate_share=sum(x["rate_s"] for x in W) / w.sum())
        print(f"{g:26s} n={len(R):2d} mean {w.mean():6.2f} s  median {np.median(w):6.2f}  mean k {k.mean():.2f}  "
              f"fit {a:.2f} + {b:.2f} k -> {a + b * EK:.2f} s at E[k] = {EK:.2f} (ratio of means: {out[g]['ratio_at_Ek']:.2f})  "
              f"render {out[g]['render_ms_per_tile']:.2f} ms/tile  probe {out[g]['probe_us']:.1f} us  geometry {out[g]['geometry_s']:.2f} s  "
              f"rate-stream share {out[g]['rate_share']:.2f}")
    return out


def containers(pattern, label):
    sizes, lf, m, byk = [], [], [], {}
    for p in sorted(glob.glob(pattern)):
        z = np.load(p, allow_pickle=False); meta = json.loads(bytes(z["meta"]).decode())
        kb = os.path.getsize(p) / 1e3; sizes.append(kb); byk.setdefault(meta["k"], []).append(kb)
        for w in meta["windows_full"]:
            m.append(w["m"])
            lf += [f for j, f in enumerate(w["lit_frac"]) if int(w["mask"]) >> j & 1]
    sizes = np.array(sizes)
    print(f"{label}: {len(sizes)} containers, mean {sizes.mean():.0f} KB (min {sizes.min():.0f}, max {sizes.max():.0f}); "
          f"by k: " + ", ".join(f"k={k}: {np.mean(v):.0f} KB (n={len(v)})" for k, v in sorted(byk.items())) +
          f"; informative channels {len(lf)}, active fraction p10/p50/p90 {np.round(np.percentile(lf, [10, 50, 90]), 3).tolist()}, "
          f"m per window {np.mean(m):.2f}")


if __name__ == "__main__":
    out = bench()
    containers(f"{WORK}/sampling_impact/v3/scenes/*/*.npz", "v3 polytope")
    containers(f"{WORK}/curved_stats/scenes/*/*.npz", "curved")
    tP = out["polytope (9 v3 families)"]["at_Ek"]
    tC = [np.mean([out["spheres"][key], out["spheres_polyhedra"][key]]) for key in ("at_Ek", "ratio_at_Ek")]
    for f in (0.0, 0.1, 0.2):
        h = [1e5 * ((1 - f) * tP + f * t) / 3600 for t in tC]
        print(f"100K physical, curved fraction {f:.0%}: {min(h):.0f}-{max(h):.0f} laptop-GPU-worker-hours (t_P = {tP:.2f} s, t_C = {min(tC):.1f}-{max(tC):.1f} s)")
