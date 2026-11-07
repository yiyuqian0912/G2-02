"""Cost and causal coverage of bounded axial MISS extensions."""
import argparse
from pathlib import Path
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import config
from ..b2_precision.bench import settings
from ..b2_precision.build import extension
from ..study import atomic_json
from .audit import quantiles, sample_rows
from .bench import OUT, load_inputs, make_session, read

DIRECTIONS = (-1, 1, -2, 2, -3, 3)


def traversal_key(xyz, side):
    """Order of scalar visits before pruning, for a tile of power-of-two side."""
    child = sum(((int(xyz[k]) >> b) & 1) << (3 * (b - 1) + k)
                for b in range(1, side.bit_length() - 1) for k in range(3))
    return 8 * ((side // 2)**3 - 1 - child) + sum((xyz[k] & 1) << k for k in range(3))


def run(args):
    module = extension(args.native)
    rows = []
    for tile, stratum, population, archive, indices, costs in sample_rows(
            read(args.plan), args.baseline, "development"):
        side, grid = tile["side"], tile["grid"]
        session, setup = make_session(module, tile, 0)
        scene, source = load_inputs(tile)
        region = module.Engine([k.center.tolist() for k in scene.kernels],
                               [k.matrix.ravel().tolist() for k in scene.kernels],
                               [k.weight for k in scene.kernels], scene.threshold.value,
                               scene.threshold.kind == "exp", True)
        opts, cfg = settings(module, "F", config())
        cfg.mode = 2
        probes = []
        with np.load(archive) as data:
            path, ns = (data[k].ravel() for k in ("evaluation_path", "ray_nanoseconds"))
            response, solid = (data[k].ravel() for k in ("response", "solid"))
        for ordinal, (index, cost) in enumerate(zip(indices[:args.samples], costs[:args.samples])):
            xyz = np.array(np.unravel_index(index, (side,)*3))[::-1]
            receiver = (xyz + tile["origin"] + .5) / grid
            direction_rows = []
            for step in (1, 4):
                eligible = {}
                for direction in DIRECTIONS:
                    axis = abs(direction) - 1
                    delta = 1 if direction > 0 else -1
                    points = []
                    for offset in range(1, step+1):
                        dest = xyz.copy()
                        dest[axis] += offset * delta
                        if (dest < 0).any() or (dest >= side).any():
                            break
                        i = int(np.ravel_multi_index(dest[::-1], (side,)*3))
                        points.append((i, traversal_key(dest, side) > traversal_key(xyz, side)))
                    if len(points) == step:
                        eligible[direction] = points
                probe = session.miss_probe(receiver, list(eligible), step)
                if probe["status"] != 0:
                    raise AssertionError((tile["id"], int(index), probe))
                for direction, points in eligible.items():
                    key = f"local_{direction}"
                    bounds = probe["bounds"]
                    if key + "_FREE" not in bounds:
                        continue
                    certified = bool(bounds[key + "_FREE"])
                    if certified and not all(response[i] and not solid[i] for i, _ in points):
                        raise AssertionError("Local certificate contradicts frozen F")
                    row = {"step": step, "direction": direction, "certified": certified,
                           "seconds": bounds[key + "_seconds"],
                           "exp_endpoints": bounds[key + "_exp_endpoints"],
                           "causal_MISS": [[i, int(ns[i])] for i, future in points
                                           if future and path[i] == 6],
                           "causal_voxels": sum(bool(future) for _, future in points)}
                    # A fixed sparse comparison against the old from-scratch test.
                    if ordinal % 8 == 0 and direction == next(iter(eligible)):
                        dest = xyz.copy()
                        dest[abs(direction)-1] += step * (1 if direction > 0 else -1)
                        dest = (dest + tile["origin"] + .5) / grid
                        start = time.perf_counter()
                        state, work = region.region(source, np.minimum(receiver, dest),
                                                    np.maximum(receiver, dest), opts, cfg)
                        row["old_volume"] = {"seconds": time.perf_counter()-start,
                                             "state": state, "work": work}
                        if state[0] == 0 and certified:
                            raise AssertionError("Certified physical decision disagrees")
                    direction_rows.append(row)
            probes.append({"index": int(index), "baseline_ray_ns": int(cost),
                           "directions": direction_rows})
        summary = []
        for step in (1, 4):
            items = [r for p in probes for r in p["directions"] if r["step"] == step]
            forward = [r for r in items if r["causal_voxels"]]
            saves = {i: t for r in forward if r["certified"] for i, t in r["causal_MISS"]}
            summary.append({"step": step, "attempts": len(items),
                            "certificates": sum(r["certified"] for r in items),
                            "time": quantiles([r["seconds"] for r in items]),
                            "forward_attempts": len(forward),
                            "forward_certificates": sum(r["certified"] for r in forward),
                            "forward_construction_seconds": sum(r["seconds"] for r in forward),
                            "forward_unique_possible_saves": len(saves),
                            "forward_original_possible_seconds": sum(saves.values()) / 1e9})
        rows.append({"tile": tile["id"], "stratum": stratum, "population": population,
                     "setup_seconds": setup, "summary": summary, "probes": probes})
        print(tile["id"], stratum, len(probes), summary, flush=True)
    atomic_json(args.output, {"build": module.study_identity,
                              "plan_SHA256": sha(args.plan), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, default=OUT / "local-native")
    p.add_argument("--plan", type=Path, default=OUT / "tile-plan.json")
    p.add_argument("--baseline", type=Path, default=OUT / "baseline")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=96)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Keep the frozen local audit record")
    run(args)
