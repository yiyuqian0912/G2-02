"""102 vs 256 vs dynamic vs hybrid window study on the stored canary fields (no physics regenerated).

  run.sh rind3d_training.study canary [--release canary-v0.2]
Inputs: the obs-index-v0.3 P256 index (P102 = its exact v0.2 prefix), the uniform rate stream (dynamic natural
windows), and the exact 16^3 count leaves of the stored fields. Measures per profile and side:
  * category distribution (no-FREE / all-lit / all-shadow / constant-lit (MA) / mixed / rare / extreme /
    nonlocal / source-containing), natural and selected;
  * spatial coverage of the world and of the scene's shadow-boundary (mixed 16^3 leaves) and SOLID-boundary
    leaves, window multiplicity, pairwise IoU, marginal new volume of the extra windows;
  * intra-scene correlation (one-way ANOVA ICC) of window statistics and design effects / effective windows
    per scene; population-weight effective sample size of the selected mixtures.
Output: outputs/study/window_study_<release>.json
"""
import json
import time
from pathlib import Path

import numpy as np

from .env import OUT
from .export import in_profile
from .index import CODES, SIDES
from .selection import load_index, weights

CELL = 8                       # coverage lattice: 128^3 cell centres (8 i + 4)


def icc(values, groups):
    """One-way random-effects ICC(1) with unequal group sizes; returns (icc, k groups, N)."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    ug, inv = np.unique(g, return_inverse=True)
    k, N = len(ug), len(v)
    if k < 2 or N <= k:
        return float("nan"), k, N
    n = np.bincount(inv)
    mean = v.mean()
    gm = np.bincount(inv, v)/n
    ssb = float((n*(gm-mean)**2).sum())
    ssw = float(((v-gm[inv])**2).sum())
    msb, msw = ssb/(k-1), ssw/(N-k)
    n0 = (N - (n**2).sum()/N)/(k-1)
    den = msb + (n0-1)*msw
    return (float((msb-msw)/den) if den > 0 else float("nan")), int(k), int(N)


def cover_counts(boxes, grid, cell):
    """Multiplicity of each lattice cell centre (cell*i + cell/2) inside the half-open boxes [o, o+L)."""
    c = np.zeros((grid,)*3, np.uint16)
    h = cell/2
    for x, y, z, L in boxes:
        lo = [int(np.ceil((v - h)/cell)) for v in (x, y, z)]
        hi = [int(np.floor((v + L - 1 - h)/cell)) + 1 for v in (x, y, z)]
        lo = [max(0, a) for a in lo]
        c[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] += 1
    return c


def pair_iou(b):
    """Mean IoU and fraction of overlapping pairs among same-side windows (exact, analytic)."""
    if len(b) < 2:
        return float("nan"), float("nan")
    o = b[:, :3].astype(float)
    L = float(b[0, 3])
    d = np.abs(o[:, None, :] - o[None, :, :])
    inter = np.prod(np.clip(L - d, 0, None), axis=2)
    iou = inter/(2*L**3 - inter)
    iu = np.triu_indices(len(b), 1)
    return float(iou[iu].mean()), float((inter[iu] > 0).mean())


def leaves_of(scene_dir_counts):
    out = []
    for p in scene_dir_counts:
        with np.load(p) as z:
            out.append(z["leaves"].astype(np.int32))
    return out


def scene_geometry_stats(leaves):
    lit = np.stack([l[..., 0] for l in leaves])
    solid = leaves[0][..., 1]
    shadow = 4096 - lit - solid[None]
    mixed = np.any((lit > 0) & (shadow > 0), axis=0)
    solid_boundary = (solid > 0) & (solid < 4096)
    return mixed, solid_boundary


def study(release="canary-v0.2", profile="P256", max_scenes=None):
    from .scenes import canary_scenes, v01_scenes
    t0 = time.perf_counter()
    scenes, rec, reports, rates = load_index(release, profile)
    refs = canary_scenes() if release == "canary-v0.2" else v01_scenes()
    p102 = in_profile(rec, "P102")
    prof_masks = {"P102": p102, "P256": np.ones(len(rec), bool)}
    out = {"release": release, "scenes": len(scenes), "records": len(rec), "P102_records": int(p102.sum())}
    # ---------------------------------------------------------------- 1. categories
    cats = {}
    for mod in ("SB", "MA"):
        sel_scene = np.array([s["profile"] == mod for s in scenes])
        ms = sel_scene[rec["scene"]]
        for pname, pm in prof_masks.items():
            for L in SIDES:
                m = ms & pm & (rec["L"] == L)
                r = rec[m]
                nat = r[r["subset"] == 0]
                row = {"windows_per_scene": float(m.sum()/max(1, sel_scene.sum())),
                       "all": category_table(r), "natural": category_table(nat),
                       "by_subset": {s: int((r["subset"] == c).sum()) for s, c in CODES.items()}}
                cats[f"{mod}/{pname}/L{L}"] = row
        for L in SIDES:
            rr = np.concatenate([rates[k][rates[k]["L"] == L] for k in range(len(scenes)) if sel_scene[k]])
            cats[f"{mod}/dynamic-natural/L{L}"] = {"population_rate_stream": category_table(rr, rate=True),
                                                   "proposals": int(len(rr))}
    out["categories"] = cats
    # ---------------------------------------------------------------- 2. shortfalls (difficult strata availability)
    sf = {}
    for pname in ("P102", "P256"):
        for s in ("informative", "extreme", "rare", "nonlocal"):
            for L in SIDES:
                q = {"P102": {"informative": 8, "extreme": 4, "rare": 2, "nonlocal": 4},
                     "P256": {"informative": 20, "extreme": 11, "rare": 6, "nonlocal": 8}}[pname][s]
                got = np.array([int(((rec["scene"] == k) & (rec["subset"] == CODES[s]) & (rec["L"] == L) & prof_masks[pname]).sum())
                                for k in range(len(scenes))])
                sf[f"{pname}/{s}/L{L}"] = {"quota": q, "mean_accepted": float(got.mean()), "scenes_with_shortfall": int((got < q).sum()),
                                           "scenes_with_none": int((got == 0).sum())}
    out["availability"] = sf
    # ---------------------------------------------------------------- 3. coverage / redundancy / boundary information
    grid = 1024//CELL
    cov_rows = []
    ks = range(len(scenes)) if not max_scenes else range(min(max_scenes, len(scenes)))
    for k in ks:
        r = rec[rec["scene"] == k]
        pm = p102[rec["scene"] == k]
        leaves = leaves_of(refs[k].counts)
        mixed, sb = scene_geometry_stats(leaves)
        rr = rates[k]
        variants = {"P102": r[pm], "P256": r,
                    "P102_natural": r[pm & (r["subset"] == 0)], "P256_natural": r[r["subset"] == 0],
                    "dynamic_256": np.concatenate([rr[rr["L"] == L][:n] for L, n in zip(SIDES, (86, 85, 85))]),
                    "dynamic_1024": np.concatenate([rr[rr["L"] == L][:341] for L in SIDES]),
                    "dynamic_3072": rr}
        row = {"scene": k, "profile": scenes[k]["profile"], "mixed_leaves": int(mixed.sum()), "solid_boundary_leaves": int(sb.sum())}
        for vname, v in variants.items():
            boxes = np.column_stack([v["x"], v["y"], v["z"], v["L"]]).astype(int)
            c = cover_counts(boxes, grid, CELL)
            lc = cover_counts(boxes, 64, 16)
            d = {"windows": int(len(v)), "world_coverage": float((c > 0).mean()), "mean_multiplicity_covered": float(c[c > 0].mean()) if (c > 0).any() else 0.0,
                 "voxel_visits_per_covered_voxel": float(sum(int(L)**3 for L in v["L"])/max(1, (c > 0).sum()*CELL**3)),
                 "shadow_boundary_leaf_coverage": float((lc[mixed] > 0).mean()) if mixed.any() else float("nan"),
                 "solid_boundary_leaf_coverage": float((lc[sb] > 0).mean()) if sb.any() else float("nan")}
            for L in SIDES:
                b = boxes[boxes[:, 3] == L]
                d[f"L{L}_pair_iou"], d[f"L{L}_pair_overlap_frac"] = pair_iou(b)
                d[f"L{L}_coverage"] = float((cover_counts(b, grid, CELL) > 0).mean())
            row[vname] = d
        # marginal new volume of the P256-only windows over P102
        c102 = cover_counts(np.column_stack([r[pm]["x"], r[pm]["y"], r[pm]["z"], r[pm]["L"]]).astype(int), grid, CELL)
        new = r[~pm]
        newvol, totvol = 0, 0
        for x, y, z, L in np.column_stack([new["x"], new["y"], new["z"], new["L"]]).astype(int):
            one = cover_counts([(x, y, z, L)], grid, CELL) > 0
            newvol += int((one & (c102 == 0)).sum())
            totvol += int(one.sum())
        row["P256_extra_new_volume_fraction"] = newvol/max(1, totvol)
        cov_rows.append(row)
    agg = {}
    for vname in ("P102", "P256", "P102_natural", "P256_natural", "dynamic_256", "dynamic_1024", "dynamic_3072"):
        for prof in ("SB", "MA"):
            rows = [r[vname] for r in cov_rows if r["profile"] == prof]
            if not rows:
                continue
            agg[f"{prof}/{vname}"] = {k: float(np.nanmedian([x[k] for x in rows])) for k in rows[0]}
    agg["P256_extra_new_volume_fraction_median"] = float(np.median([r["P256_extra_new_volume_fraction"] for r in cov_rows]))
    out["coverage_median"] = agg
    out["coverage_scenes"] = len(cov_rows)
    # ---------------------------------------------------------------- 4. intra-scene correlation
    out["icc"] = icc_table(scenes, refs, rec, p102)
    # ---------------------------------------------------------------- 5. population ESS of selected mixtures
    ess = {}
    for task in ("global", "src-excluded"):
        for pname, pm in prof_masks.items():
            r = rec[pm]
            reps = reports
            w, used, diag = weights(r, reps, task)
            for L in SIDES:
                dd = [d for d in diag if d["L"] == L]
                ess[f"{task}/{pname}/L{L}"] = {"records_per_scene": float(np.mean([d["n"] for d in dd])),
                                              "population_ess_per_scene": float(np.mean([d["ess"] for d in dd])),
                                              "max_weight_median": float(np.median([d["max_weight"] for d in dd]))}
    out["population_ess"] = ess
    out["seconds"] = time.perf_counter()-t0
    d = OUT/"study"
    d.mkdir(parents=True, exist_ok=True)
    (d/f"window_study_{release}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def category_table(r, rate=False):
    n = max(1, len(r))
    t = {"n": int(len(r)),
         "no_free": float((r["category"] == 0).sum()/n), "all_lit": float((r["category"] == 1).sum()/n),
         "all_shadow": float((r["category"] == 2).sum()/n), "constant_lit": float((r["category"] == 4).sum()/n),
         "mixed": float((r["category"] == 3).sum()/n),
         "rare_mixed_minority_le_64": float(((r["category"] == 3) & (r["minority_count"] <= 64)).sum()/n),
         "extreme_mixed_minority_le_1pct": float(((r["category"] == 3) & (r["minority_fraction"] <= 0.01)).sum()/n),
         "contains_source": float(r["contains_source"].sum()/n)}
    if not rate:
        t["nonlocal"] = float(r["nonlocal_flag"].sum()/n)
        t["nonlocal_among_informative"] = float(r["nonlocal_flag"].sum()/max(1, r["A_informative"].sum()))
    return t


def icc_table(scenes, refs, rec, p102):
    out = {}
    src = {k: refs[k].privileged_labels()[0] for k in range(len(scenes))}
    L = rec["L"].astype(float)
    centre = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(float)/1024 + (L/2048)[:, None]
    dist = np.array([np.min(np.linalg.norm(src[int(s)] - c, axis=1)) for s, c in zip(rec["scene"], centre)])/(L/1024)
    free = rec["free"].astype(float)
    stats = {"lit_fraction": np.where(free > 0, rec["lit"][:, 0]/np.maximum(free, 1), np.nan),
             "solid_fraction": rec["solid"]/L**3,
             "mixed": (rec["category"] == 3).astype(float),
             "log10_minority": np.where(rec["category"] == 3, np.log10(np.maximum(rec["minority_fraction"], 1e-9)), np.nan),
             "source_distance_over_L": dist}
    for name, v in stats.items():
        for set_name, mask in (("natural", rec["subset"] == 0), ("P256_all", np.ones(len(rec), bool)), ("P102_all", p102)):
            for side in SIDES:
                m = mask & (rec["L"] == side)
                val, k, N = icc(v[m], rec["scene"][m])
                mbar = N/max(k, 1)
                out[f"{name}/{set_name}/L{side}"] = {"icc": val, "scenes": k, "windows": N, "m_per_scene": mbar,
                                                     "design_effect": 1 + (mbar-1)*val if np.isfinite(val) else None,
                                                     "effective_windows_per_scene": mbar/(1+(mbar-1)*val) if np.isfinite(val) else None}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="canary-v0.2")
    ap.add_argument("--max-scenes", type=int, default=None)
    a = ap.parse_args()
    out = study(a.release, max_scenes=a.max_scenes)
    print(json.dumps({k: out[k] for k in ("release", "scenes", "records", "P102_records", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
