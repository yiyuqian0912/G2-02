"""Replay the historical 16-tile admission probe without per-voxel tracing."""
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_precision.build import extension
from ..b2_precision.tiles import CAPS, RSS_CAP_MIB, arrays, digest, plan_256
from ..study import atomic_json
from .bench import F_BUILD, OUT, compare, make_session, read

LIMITS = {"P": CAPS["P"], "I": CAPS["F"]}


def trial(args, plan):
    backend = args.backend
    native = args.native if backend == "I" else F_BUILD
    folder = args.output / backend
    folder.mkdir()
    module = extension(native)
    wall, cpu = time.perf_counter(), time.process_time()
    session, setup = make_session(module, plan[0], 3 if backend == "I" else 0,
                                  independent=backend == "I", trace=False)
    rows = []
    for tile in plan:
        remaining = LIMITS[backend] - (time.perf_counter() - wall)
        if remaining <= 0:
            break
        start = time.perf_counter()
        result = session.tile(tile["origin"], tile["side"], min(180., remaining))
        bits = arrays(result)
        archive = folder / (tile["id"] + ".npz")
        np.savez_compressed(archive, **bits)
        elapsed = time.perf_counter() - start
        complete = bool(bits["valid"].all())
        if complete != result["complete"]:
            raise AssertionError("Tile validity mismatch")
        compare(bits, Path(tile["prior_archive"]), tile["prior_archive_SHA256"])
        rows.append({"id": tile["id"], "origin": tile["origin"], "complete": complete,
                     "valid_voxels": int(bits["valid"].sum()), "tile_seconds": elapsed,
                     "native_seconds": result["wall_seconds"], "reason": result["reason"],
                     "physical_bits_SHA256": digest(bits) if complete else None,
                     "archive_SHA256": sha(archive), "work": result["work"]})
        atomic_json(archive.with_suffix(".json"), rows[-1])
        print(backend, tile["id"], round(elapsed, 3), rows[-1]["valid_voxels"], flush=True)
        if not complete or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > RSS_CAP_MIB:
            break
    atomic_json(folder / "probe.json", {
        "backend": backend, "rows": rows, "build": module.study_identity,
        "plan_SHA256": sha(args.plan), "setup_seconds": setup, "session": session.info(),
        "elapsed_wall_seconds": time.perf_counter() - wall, "CPU_seconds": time.process_time() - cpu,
        "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })


def admit(record):
    times = [r["tile_seconds"] for r in record["rows"]]
    complete = len(times) == 16 and all(r["complete"] for r in record["rows"])
    setup = 2 * record["setup_seconds"]
    # Identical paired-octant point estimate and guard to the historical gate.
    projected = setup + 4 * sum(times)
    guarded = setup + 1.2 * sum(8 * max(times[i:i+2]) for i in range(0, len(times), 2))
    cap = LIMITS[record["backend"]]
    return {"accepted": complete and guarded <= cap and record["peak_RSS_MiB"] <= RSS_CAP_MIB,
            "probe_complete": complete, "seconds_cap": cap, "memory_cap_MiB": RSS_CAP_MIB,
            "projected_wall_seconds": projected if complete else None,
            "guarded_planning_seconds": guarded if complete else None,
            "peak_RSS_MiB": record["peak_RSS_MiB"],
            "probe_seconds": record["elapsed_wall_seconds"]}


def run(args):
    plan = read(args.plan)["tiles"][:16]
    if [tuple(r["origin"]) for r in plan] != plan_256()[0]:
        raise ValueError("Admission requires the historical probe order")
    if args.backend:
        return trial(args, plan)
    args.output.mkdir()
    results = {}
    for backend, cap in LIMITS.items():
        cmd = [sys.executable, "-m", "experiments.gaussian_geometry.b2_miss.gate",
               "--output", str(args.output), "--plan", str(args.plan), "--native", str(args.native),
               "--backend", backend]
        try:
            subprocess.run(cmd, check=True, timeout=cap + 30)
        except subprocess.TimeoutExpired:
            results[backend] = {"accepted": False, "reason": "process watchdog"}
            continue
        results[backend] = admit(read(args.output / backend / "probe.json"))
    paired = 0
    for tile in plan:
        p = args.output / "P" / (tile["id"] + ".npz")
        i = args.output / "I" / p.name
        if not p.exists() or not i.exists():
            continue
        with np.load(p) as a, np.load(i) as b:
            common = a["valid"] & b["valid"]
            if any(not np.array_equal(a[k][common], b[k][common]) for k in ("response", "solid")):
                raise AssertionError("Independent admission probes disagree")
            paired += int(common.sum())
    atomic_json(args.output / "admission.json", {"estimates": results,
                "admitted": all(r["accepted"] for r in results.values()),
                "matched_voxels": paired, "coverage_fraction": paired / 256**3,
                "complete_dual_field": False, "plan_SHA256": sha(args.plan)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=OUT / "tile-plan.json")
    p.add_argument("--native", type=Path, default=OUT / "propagation-native")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=LIMITS, help=argparse.SUPPRESS)
    run(p.parse_args())
