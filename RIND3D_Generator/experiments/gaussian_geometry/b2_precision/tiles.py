"""Within-backend tile reuse and a gated, bounded 256-cubed control."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import INPUT, config, identity_file, read_input
from ..study import atomic_json
from .bench import settings
from .build import OUT, extension

CAPS = {"P": 360., "F": 900.}
RSS_CAP_MIB = 512
SIDE = 64


def make_session(module, scene, source, grid, backend, cover):
    start = time.perf_counter()
    engine = module.Engine(
        [k.center.tolist() for k in scene.kernels],
        [k.matrix.ravel().tolist() for k in scene.kernels],
        [k.weight for k in scene.kernels], scene.threshold.value,
        scene.threshold.kind == "exp", backend == "F",
    )
    opts, cfg = settings(module, "F" if backend == "F" else "A", config())
    cfg.trace = False
    session = engine.session(source, grid, opts, cfg, cover)
    return session, time.perf_counter() - start


def arrays(result):
    return {k: np.asarray(result[k], dtype=bool) for k in ("response", "solid", "valid")}


def digest(bits):
    return hashlib.sha256(b"".join(np.packbits(bits[k]).tobytes()
                                    for k in ("response", "solid"))).hexdigest()


def summarize_tiles(work):
    return {key: work.get(key, 0) for key in
            ("cover_box_queries", "cover_nodes", "reference_projection_builds",
             "region_filled_voxels", "reference_ray_queries", "primary_ray_queries",
             "precision_source_offset_preparations", "tile_reused_cover")}


def compare_tiles(args, module, scene, source, identity):
    # Three adjacent pairs with unaligned origins. Two pairs lie near
    # predetermined kernel centers, and one is at a fixed world location.
    side = 32
    bases = [[37, 213, 701]]
    for k in (scene.kernels[0], scene.kernels[len(scene.kernels)//2]):
        bases.append(np.clip(np.floor(k.center * 1024).astype(int) - [23, 11, 9],
                             1, 1024 - 2 * side - 1).tolist())
    origins = [np.add(base, [i * side, 0, 0]).tolist() for base in bases for i in (0, 1)]
    rows, expected = [], {}
    for backend in ("P", "F"):
        for mode in ("cold", "geometry", "cover"):
            start = time.perf_counter()
            cpu = time.process_time()
            session, setup = None, 0.
            if mode != "cold":
                session, setup = make_session(module, scene, source, 1024, backend, mode == "cover")
            work = Counter()
            measurements = []
            for i, origin in enumerate(origins):
                if mode == "cold":
                    session, elapsed = make_session(module, scene, source, 1024, backend, False)
                    setup += elapsed
                row = session.tile(origin, side, 120)
                bits = arrays(row)
                if not row["complete"] or not bits["valid"].all():
                    raise RuntimeError((backend, mode, origin, row["reason"]))
                physical = digest(bits)
                if expected.setdefault(i, physical) != physical:
                    raise AssertionError("Tile reuse changed physical bits")
                work.update(row["work"])
                measurements.append({
                    "origin_xyz": origin, "physical_bits_SHA256": physical,
                    "native_seconds": row["wall_seconds"], "cover_seconds": row["cover_seconds"],
                })
            rows.append({"backend": backend, "policy": mode, "setup_seconds": setup,
                         "wall_seconds": time.perf_counter() - start,
                         "CPU_seconds": time.process_time() - cpu, "work": summarize_tiles(work),
                         "session_after": session.info(), "session_preparation_work": session.preparation_work(),
                         "process_cumulative_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         "tiles": measurements})
            atomic_json(args.output / "reuse.json", {
                "build": module.study_identity, "geometry": identity, "world_grid": 1024,
                "side": side, "rows": rows,
            })
            print(backend, mode, round(rows[-1]["wall_seconds"], 3), flush=True)

    # The grid-64 reference is a complete matched field. Crops at arbitrary
    # origins must use its exact global centers, not an 8-cubed local grid.
    baseline = args.monolithic
    ref = json.loads(baseline.with_suffix(".json").read_text())
    if (ref["resolution"] != 64 or ref["archive_SHA256"] != sha(baseline)
            or any(ref[k] != identity[k] for k in ("geometry_id", "source_SHA256"))):
        raise ValueError("Monolithic identity or archive differs")
    with np.load(baseline) as field:
        if not field["valid"].all():
            raise ValueError("Complete monolithic control required")
        checks = []
        for backend in ("P", "F"):
            session, _ = make_session(module, scene, source, 64, backend, True)
            for origin in ((3, 11, 17), (29, 34, 40), (55, 53, 49)):
                x, y, z = origin
                bits = arrays(session.tile(origin, 8, 60))
                roi = np.s_[z:z+8, y:y+8, x:x+8]
                if not all(np.array_equal(bits[k], field[k][roi]) for k in bits):
                    raise AssertionError("Arbitrary-origin tile differs from the monolithic slice")
                checks.append({"backend": backend, "origin_xyz": origin, "valid_voxels": 512})
    atomic_json(args.output / "monolithic.json", {"grid": 64, "side": 8,
                 "reference_SHA256": sha(baseline), "checks": checks})


def plan_256():
    rng = np.random.default_rng(20261107256)
    probe, others = [], []
    for octant in product(range(2), repeat=3):
        tiles = [tuple(SIDE * (2*a + b) for a, b in zip(octant, offset))
                 for offset in product(range(2), repeat=3)]
        order = rng.permutation(8)
        probe.extend(tiles[int(i)] for i in order[:2])
        others.extend(tiles[int(i)] for i in order[2:])
    return probe, others


def stem_for(root, backend, origin):
    return root / backend / "-".join(map(str, origin))


def scale_worker(args, module, scene, source, identity):
    backend, phase = args.worker, args.phase
    start, cpu = time.perf_counter(), time.process_time()
    probe, rest = plan_256()
    root = args.output
    plan = json.loads((root / "scale-identity.json").read_text())
    if (plan["geometry"] != identity or plan["build"] != module.study_identity
            or plan["driver_SHA256"] != sha(Path(__file__))):
        raise ValueError("Scale worker input or build differs from the admitted plan")
    remaining = CAPS[backend]
    if phase == "remaining":
        initial = json.loads((root / backend / "probe.json").read_text())
        remaining -= initial["elapsed_wall_seconds"]
    rows = []
    if remaining > 0:
        session, setup = make_session(module, scene, source, 256, backend, True)
        for origin in probe if phase == "probe" else rest:
            budget = remaining - (time.perf_counter() - start)
            if budget <= 0:
                break
            stem = stem_for(root, backend, origin)
            if stem.with_suffix(".json").exists():
                previous = json.loads(stem.with_suffix(".json").read_text())
                if sha(stem.with_suffix(".npz")) != previous["archive_SHA256"]:
                    raise ValueError("Corrupted scale checkpoint")
                rows.append(previous)
                if not previous["complete"]:
                    break
                continue
            tile_start = time.perf_counter()
            result = session.tile(origin, SIDE, min(180., budget))
            bits = arrays(result)
            if result["complete"] != bool(bits["valid"].all()):
                raise AssertionError("Tile completeness and validity disagree")
            np.savez_compressed(stem.with_suffix(".npz"), **bits)
            elapsed = time.perf_counter() - tile_start
            record = {
                "origin_xyz": origin, "world_grid": 256, "side": SIDE, "backend": backend,
                "phase": phase, "complete": bool(bits["valid"].all()), "reason": result["reason"],
                "valid_voxels": int(bits["valid"].sum()), "tile_seconds": elapsed,
                "native_seconds": result["wall_seconds"], "archive_SHA256": sha(stem.with_suffix(".npz")),
                "physical_bits_SHA256": digest(bits) if bits["valid"].all() else None,
                "work": result["work"],
            }
            atomic_json(stem.with_suffix(".json"), record)
            rows.append(record)
            print(backend, phase, origin, record["valid_voxels"], round(elapsed, 3), flush=True)
            if not record["complete"] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > RSS_CAP_MIB:
                break
        session_state = session.info()
    else:
        setup, session_state = 0., None
    atomic_json(root / backend / (phase + ".json"), {
        "backend": backend, "phase": phase, "input": identity,
        "elapsed_wall_seconds": time.perf_counter() - start,
        "CPU_seconds": time.process_time() - cpu, "setup_seconds": setup,
        "session": session_state, "planned_tiles": len(probe if phase == "probe" else rest),
        "rows": rows, "process_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })


def scaling_summary(root):
    expected = [*plan_256()[0], *plan_256()[1]]
    compared = Counter()
    rows = {b: [] for b in CAPS}
    for origin in expected:
        bits = {}
        for backend in CAPS:
            stem = stem_for(root, backend, origin)
            if stem.with_suffix(".json").exists():
                r = json.loads(stem.with_suffix(".json").read_text())
                assert tuple(r["origin_xyz"]) == origin and r["backend"] == backend
                if sha(stem.with_suffix(".npz")) != r["archive_SHA256"]:
                    raise ValueError("Stored tile failed its integrity check")
                rows[backend].append(r)
                with np.load(stem.with_suffix(".npz")) as data:
                    bits[backend] = arrays(data)
        if len(bits) == 2:
            common = bits["P"]["valid"] & bits["F"]["valid"]
            compared["valid_voxels"] += int(common.sum())
            for k in ("response", "solid"):
                compared[k + "_mismatches"] += int(np.count_nonzero(common & (bits["P"][k] != bits["F"][k])))
    if any(v for k, v in compared.items() if k.endswith("mismatches")):
        raise AssertionError("Independent 256-grid tile disagreement")
    result = {b: {"tiles": len(data), "valid_voxels": sum(r["valid_voxels"] for r in data),
                   "complete": len(data) == 64 and all(r["complete"] for r in data),
                   "tile_seconds_sum": sum(r["tile_seconds"] for r in data)}
              for b, data in rows.items()}
    return {"grid": 256, "tile_side": SIDE, "by_backend": result,
            "compared": dict(compared), "complete_dual_field":
                compared["valid_voxels"] == 256**3 and all(r["complete"] for r in result.values())}


def scale(args, module, scene, source, identity):
    if args.worker:
        scale_worker(args, module, scene, source, identity)
        return
    root = args.output
    identity_file(root / "scale-identity.json", {
        "build": module.study_identity, "geometry": identity, "budget_seconds": CAPS,
        "memory_admission_limit_MiB": RSS_CAP_MIB, "planned_origins": [[list(x) for x in phase] for phase in plan_256()],
        "driver_SHA256": sha(Path(__file__)), "case": args.case,
    })

    def run(backend, phase):
        (root / backend).mkdir(exist_ok=True)
        if any((root / backend / (phase + suffix)).exists() for suffix in (".json", "-watchdog.json")):
            return
        cmd = [sys.executable, "-m", "experiments.gaussian_geometry.b2_precision.tiles", "scale256",
               "--worker", backend, "--phase", phase, "--case", str(args.case),
               "--output", str(root), "--native", str(args.native), "--input", str(args.input)]
        try:
            subprocess.run(cmd, check=True, timeout=CAPS[backend] + 30)
        except subprocess.TimeoutExpired:
            atomic_json(root / backend / (phase + "-watchdog.json"),
                        {"status": "WATCHDOG", "phase": phase, "elapsed_cap": CAPS[backend] + 30})

    estimates = {}
    for backend in CAPS:
        run(backend, "probe")
        meta = root / backend / "probe.json"
        if not meta.exists():
            estimates[backend] = {"accepted": False, "reason": "probe watchdog"}
            continue
        trial = json.loads(meta.read_text())
        times = [r["tile_seconds"] for r in trial["rows"]]
        complete = len(times) == 16 and all(r["complete"] for r in trial["rows"])
        projected = 2 * trial["setup_seconds"] + 4 * sum(times)
        # Each pair represents an equal-sized octant. Use its slower
        # observation plus a planning margin, not a probabilistic bound.
        guarded = (2 * trial["setup_seconds"] +
                   1.2 * sum(8 * max(times[i:i+2]) for i in range(0, len(times), 2)))
        estimates[backend] = {
            "valid_probe_tiles": sum(r["complete"] for r in trial["rows"]),
            "projected_wall_seconds": projected, "guarded_planning_seconds": guarded,
            "peak_probe_RSS_MiB": trial["process_peak_RSS_MiB"],
            "accepted": complete and guarded <= CAPS[backend]
                        and trial["process_peak_RSS_MiB"] <= RSS_CAP_MIB,
        }
    atomic_json(root / "budget-gate.json", {"estimates": estimates, "probe_comparison": scaling_summary(root)})
    if all(e["accepted"] for e in estimates.values()):
        for backend in CAPS:
            run(backend, "remaining")
    atomic_json(root / "coverage.json", scaling_summary(root))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["compare", "scale256"])
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native", type=Path, default=OUT / "tile-native")
    p.add_argument("--case", type=int, default=14)
    p.add_argument("--monolithic", type=Path, default=OUT / "frozen64/gaussian_sum-014-64-F.npz")
    p.add_argument("--worker", choices=CAPS, help=argparse.SUPPRESS)
    p.add_argument("--phase", choices=["probe", "remaining"], help=argparse.SUPPRESS)
    a = p.parse_args()
    if bool(a.worker) != bool(a.phase) or (a.worker and a.task != "scale256"):
        p.error("Scale worker requires both a backend and a phase")
    a.output.mkdir(parents=True, exist_ok=True)
    module = extension(a.native)
    scene, source, identity = read_input(a.input / f"gaussian_sum-{a.case:03d}")
    if a.task == "compare":
        compare_tiles(a, module, scene, source, identity)
    else:
        scale(a, module, scene, source, identity)
