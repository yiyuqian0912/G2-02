"""Freeze new scene-disjoint tiles and cost-stratified scalar HIT probes."""
import argparse
from pathlib import Path

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import INPUT, read_input
from ..b2_miss.bench import OUT as OLD_MISS, read
from ..b2_primary_miss.build import OUT as PRIMARY
from ..study import atomic_json

OUT = Path("outputs/gaussian-b2-hit-v1")
PLAN = OLD_MISS / "tile-plan.json"
PRECISION = Path("outputs/gaussian-b2-precision-v1")
NATIVE = PRIMARY / "propagation-native"
AUDIT_TILES = ("14-05", "14-13", "14-14", "7-hot", "60-hot", "63-hot")


def free_boundary(response, solid):
    boundary = np.zeros(solid.shape, dtype=bool)
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        transition = (response[a] != response[b]) & ~(solid[a] | solid[b])
        boundary[a] |= transition
        boundary[b] |= transition
    return boundary


def holdout(plan):
    rng = np.random.default_rng(202611072226)
    used = {r["geometry"]["geometry_id"] for r in plan["tiles"]}
    specs = [(0, INPUT, "frozen64"), (21, INPUT, "frozen64"),
             (62, Path("outputs/gaussian-b2-independent-v1/heldout"), "holdout64"),
             (64, Path("outputs/gaussian-b2-independent-v1/heldout"), "holdout64"),
             (75, PRECISION / "fresh-inputs", "fresh64")]
    tiles = []
    for number, root, folder in specs:
        case = f"gaussian_sum-{number:03d}"
        scene, source, identity = read_input(root / case)
        if identity["geometry_id"] in used:
            raise ValueError("Held-out geometry overlaps the prior tile plan")
        used.add(identity["geometry_id"])
        archive = PRECISION / folder / f"{case}-64-F.npz"
        meta = read(archive.with_suffix(".json"))
        if (not meta["complete"] or meta["archive_SHA256"] != sha(archive)
                or any(meta[k] != identity[k] for k in ("geometry_id", "source_SHA256"))):
            raise ValueError("Frozen coarse field identity differs")
        with np.load(archive) as bits:
            y, solid = bits["response"].astype(bool), bits["solid"].astype(bool)
            boundary = np.argwhere(free_boundary(y, solid))[:, ::-1]
        uniform = rng.integers(0, 225, size=3)
        for index in rng.permutation(len(boundary)):
            center = boundary[index]
            origin = np.clip(4*center + 2 - [13, 17, 11], 0, 224)
            if np.any(np.abs(origin-uniform) >= 32):
                break
        else:
            raise ValueError("No disjoint boundary tile")
        scales = [np.linalg.svd(k.matrix, compute_uv=False) for k in scene.kernels]
        for tag, xyz in (("uniform", uniform), ("boundary", origin)):
            tiles.append({"id": f"new-{number}-{tag}", "case": case, "input": str(root),
                          "geometry": identity, "grid": 256, "side": 32, "origin": xyz.tolist(),
                          "split": "heldout_new", "selection": tag,
                          "selection_archive": str(archive), "selection_archive_SHA256": sha(archive),
                          "source_center_distance": float(np.linalg.norm(source - (xyz+16)/256)),
                          "scene_solid_fraction_grid64": float(solid.mean()),
                          "scene_free_shadow_fraction_grid64": float((~y & ~solid).mean()),
                          "max_kernel_axis_ratio": float(max(s[0]/s[-1] for s in scales)),
                          "boundary_center_grid64": center.tolist() if tag == "boundary" else None})
    return tiles


def seeds(plan):
    rng = np.random.default_rng(202611072227)
    rows = []
    for tile in plan["tiles"]:
        if tile["id"] not in AUDIT_TILES:
            continue
        path = PRIMARY / "P_J" / (tile["id"] + ".npz")
        info = read(path.with_suffix(".json"))
        if not info["complete"] or sha(path) != info["archive_SHA256"]:
            raise ValueError("Frozen P_J trace differs")
        with np.load(path) as f:
            elapsed = f["ray_nanoseconds"].ravel()
            hits = np.flatnonzero(f["evaluation_path"].ravel() == 7)
            boundary = free_boundary(f["response"], f["solid"]).ravel()
        ranked = hits[np.argsort(elapsed[hits], kind="stable")]
        slices = np.split(ranked, [len(ranked)//4, 3*len(ranked)//4])
        for group, indices in zip(("cheap", "moderate", "expensive", "boundary"),
                                  [*slices, ranked[boundary[ranked]]], strict=True):
            limit = 128 if group != "boundary" else 64
            sample = np.sort(rng.choice(indices, min(limit, len(indices)), replace=False))
            rows.append({"tile": tile, "stratum": group, "population": len(indices),
                         "archive": str(path), "archive_SHA256": info["archive_SHA256"],
                         "indices": sample.tolist(), "original_nanoseconds": elapsed[sample].tolist()})
    return rows


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / f for f in ("heldout-plan.json", "audit-plan.json")]
    if any(p.exists() for p in paths):
        raise ValueError("The study plans are frozen")
    plan = read(args.plan)
    atomic_json(paths[0], {"prior_plan_SHA256": sha(args.plan), "tiles": holdout(plan)})
    atomic_json(paths[1], {"prior_plan_SHA256": sha(args.plan), "strata": seeds(plan)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUT)
    freeze(parser.parse_args())
