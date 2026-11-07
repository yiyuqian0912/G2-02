"""Measure F's certified margin and gradient radius on frozen cost strata."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..b2_angular.preservation import sha
from ..b2_precision.build import extension
from ..study import atomic_json
from .bench import OUT, make_session, read


def quantiles(values):
    return {f"p{q}": float(np.percentile(values, q)) for q in (0, 10, 50, 90, 95, 99, 100)}


def sample_rows(plan, baseline, split):
    rng = np.random.default_rng(202611071601)
    for tile in plan["tiles"]:
        if split != "all" and tile["split"] != split:
            continue
        path = baseline / (tile["id"] + ".npz")
        meta = read(path.with_suffix(".json"))
        if sha(path) != meta["archive_SHA256"] or not meta["complete"]:
            raise ValueError("Audit requires frozen valid trace")
        with np.load(path) as f:
            t = f["ray_nanoseconds"].ravel()
            miss = np.flatnonzero(f["evaluation_path"].ravel() == 6)
        if not len(miss):
            continue
        ranked = miss[np.argsort(t[miss], kind="stable")]
        cap = 512 if tile["id"] == "14-05" else 128
        for name, population in (("remaining", ranked[:3*len(miss)//4]),
                                 ("expensive", ranked[3*len(miss)//4:])):
            if len(population):
                indices = np.sort(rng.choice(population, min(len(population), cap), replace=False))
                yield tile, name, len(population), path, indices, t[indices]


def run(args):
    module = extension(args.native)
    plan = read(args.plan)
    rows, setups = [], []
    session = None
    last = None
    for tile, name, population, archive, indices, old_cost in sample_rows(plan, args.baseline, args.split):
        if tile["case"] != last:
            session, seconds = make_session(module, tile, 0)
            setups.append({"case": tile["case"], "seconds": seconds})
            last = tile["case"]
        items = []
        for index, cost in zip(indices, old_cost):
            local = np.array(np.unravel_index(index, (tile["side"],)*3))[::-1]
            receiver = ((local + tile["origin"] + .5) / tile["grid"]).tolist()
            probe = session.miss_probe(receiver)
            if probe["status"] != 0:
                raise AssertionError((tile["id"], int(index), probe))
            items.append({"index": int(index), "baseline_ray_ns": int(cost), **probe})
        available = [r for r in items if r["bounds"]["complete_proof"]]
        counts = Counter()
        work = Counter()
        for r in items:
            work.update(r["work"])
        for r in available:
            n = r["bounds"]["certified_receivers_in_radius4_stencil"]
            counts["1" if n == 1 else "2–8" if n <= 8 else "9–31" if n <= 31 else "32–99" if n <= 99 else "100+"] += 1
        metrics = {k: quantiles([r["bounds"][k] for r in available])
                   for k in ("cube_voxels", "axis_0", "axis_1", "axis_2",
                             "margin_lower", "true_margin_upper_from_F_witnesses",
                             "limiting_hi", "leaves")} if available else {}
        if len(available) > 2 and len({r["bounds"]["cube_voxels"] for r in available}) > 1:
            rho = spearmanr([r["baseline_ray_ns"] for r in available],
                            [r["bounds"]["cube_voxels"] for r in available]).statistic
            metrics["radius_cost_spearman"] = float(rho) if np.isfinite(rho) else None
        rows.append({"tile": tile["id"], "stratum": name, "population": population,
                     "archive_SHA256": sha(archive), "metrics": metrics,
                     "coverage_counts": dict(counts), "work": dict(work), "samples": items})
        print(tile["id"], name, len(items), dict(counts),
              metrics.get("cube_voxels", {}).get("p50"), flush=True)
    atomic_json(args.output, {"build": module.study_identity, "plan_SHA256": sha(args.plan),
                              "split": args.split, "rows": rows, "setups": setups})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, default=OUT / "audit-native")
    p.add_argument("--plan", type=Path, default=OUT / "tile-plan.json")
    p.add_argument("--baseline", type=Path, default=OUT / "baseline")
    p.add_argument("--split", choices=["development", "evaluation", "all"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Keep the frozen audit record")
    run(args)
