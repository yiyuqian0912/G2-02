"""Matched traced runs on the previous B2 MISS tile plan."""
import argparse
from pathlib import Path
import resource
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import config, trace_summary
from ..b2_miss.bench import OUT as MISS_OUT, clustering, compare, load_inputs, read
from ..b2_precision.bench import settings
from ..b2_precision.build import extension
from ..b2_precision.tiles import digest
from ..study import atomic_json
from .build import OUT

PLAN = MISS_OUT / "tile-plan.json"
# independent, distance mode, primary audit mode
VARIANTS = {
    "P_profile": (False, 0, 1),
    "P_capture": (False, 0, 2),
    "P_H": (False, 2, 0),
    "P_I": (False, 3, 0),
    "P_J": (False, 4, 0),
    "F_J": (True, 4, 0),
}


def make_session(module, row, variant, traced):
    independent, mode, audit = VARIANTS[variant]
    scene, source = load_inputs(row)
    start, cpu = time.perf_counter(), time.process_time()
    engine = module.Engine([k.center.tolist() for k in scene.kernels],
                           [k.matrix.ravel().tolist() for k in scene.kernels],
                           [k.weight for k in scene.kernels], scene.threshold.value,
                           scene.threshold.kind == "exp", independent)
    opts, cfg = settings(module, "F" if independent else "A", config())
    cfg.trace = traced
    cfg.study.miss_mode = mode
    cfg.study.primary_audit = audit
    session = engine.session(source, row["grid"], opts, cfg, True)
    return session, time.perf_counter() - start, time.process_time() - cpu


def accounting(bits, path):
    """Count only scalar MISS queries that actually returned via the cache."""
    with np.load(path) as baseline:
        old = baseline["evaluation_path"]
        old_time = baseline["ray_nanoseconds"]
        new = bits["evaluation_path"]
        if not np.array_equal(new, old):
            raise AssertionError("Scalar or region traversal changed")
        hit = bits.get("propagated_seed", np.full(old.shape, -1)) >= 0
        if not np.all(old[hit] == 6):
            raise AssertionError("Cache bypassed a non-MISS scalar path")
        count = int(hit.sum())
        total = int(old_time[old == 6].sum())
        saved = int(old_time[hit].sum())
        quartile = float(np.quantile(old_time[old == 6], .75)) if total else float("inf")
        return {"actual_MISS_skipped": count, "original_MISS_ns_avoided": saved,
                "original_MISS_ns": total, "top_quartile_skipped": int((old_time[hit] >= quartile).sum()),
                "top_quartile_ns_avoided": int(old_time[hit & (old_time >= quartile)].sum()),
                "remaining_scalar_MISS_ns": int(bits["ray_nanoseconds"][(new == 6) & ~hit].sum()),
                "cache_hit_ns": int(bits["ray_nanoseconds"][hit].sum())}


def run(args):
    plan = read(args.plan)["tiles"]
    if args.ids:
        selected = set(args.ids)
        plan = [r for r in plan if r["id"] in selected]
        if {r["id"] for r in plan} != selected:
            raise ValueError("Unknown frozen tile")
    if args.case14:
        plan = [r for r in plan if r["case"] == "gaussian_sum-014"]
    if not plan:
        raise ValueError("Empty workload")
    root = args.output
    root.mkdir(parents=True)
    module = extension(args.native)
    atomic_json(root / "identity.json",
                {"build": module.study_identity, "plan_SHA256": sha(args.plan),
                 "variant": args.variant, "ids": [r["id"] for r in plan],
                 "reference": str(args.reference), "driver_SHA256": sha(Path(__file__)),
                 "options": config()})
    start, cpu = time.perf_counter(), time.process_time()
    setups, rows = [], []
    case = None
    for tile in plan:
        if tile["case"] != case:
            session, secs, cpu_secs = make_session(module, tile, args.variant, True)
            case = tile["case"]
            setups.append({"case": case, "seconds": secs, "CPU_seconds": cpu_secs,
                           "info": session.info(), "work": session.preparation_work()})
        t, c = time.perf_counter(), time.process_time()
        result = session.tile(tile["origin"], tile["side"], config()["field_seconds"])
        wall, cpu_s = time.perf_counter()-t, time.process_time()-c
        bits = {k: np.asarray(result.pop(k), dtype=bool) for k in ("response", "solid", "valid")}
        for k in ("evaluation_path", "ray_nanoseconds", "point_nanoseconds", "propagated_seed"):
            if k in result:
                bits[k] = np.asarray(result.pop(k))
        archive = root / (tile["id"] + ".npz")
        reference = args.reference / archive.name
        complete = bool(bits["valid"].all())
        if complete != result["complete"]:
            raise AssertionError("Tile completeness disagrees with validity")
        compare(bits, reference)
        if tile.get("prior_archive"):
            compare(bits, Path(tile["prior_archive"]), tile["prior_archive_SHA256"])
        np.savez_compressed(archive, **bits)
        row = {**tile, **result, "tile_call_seconds": wall, "tile_CPU_seconds": cpu_s,
               "valid_voxels": int(bits["valid"].sum()), "archive_SHA256": sha(archive),
               "physical_bits_SHA256": digest(bits) if complete else None,
               "trace": trace_summary(bits),
               "clustering": clustering(bits["evaluation_path"], bits["ray_nanoseconds"]),
               "accounting": accounting(bits, reference) if complete else None}
        atomic_json(archive.with_suffix(".json"), row)
        rows.append(row)
        atomic_json(root / "summary.json",
                    {"records": rows, "setups": setups, "elapsed_wall_seconds": time.perf_counter()-start,
                     "CPU_seconds": time.process_time()-cpu,
                     "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        counts = row["accounting"]
        print(tile["id"], round(wall, 3), row["valid_voxels"],
              counts["actual_MISS_skipped"] if counts else "incomplete", flush=True)
        if not complete:
            raise RuntimeError((tile["id"], result["reason"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=VARIANTS)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--reference", type=Path, default=OUT / "P")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--case14", action="store_true")
    run(p.parse_args())
