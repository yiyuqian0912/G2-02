"""Matched B2 ray and small-field trials with exact-input fingerprints."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_native.api import NativeOracle
from ..b2_regions.api import RegionOracle
from ..models import load, record
from ..queries import FieldOracle, Limits
from ..render import small_field
from ..study import atomic_json
from .build import DEFAULT, extension
from .freeze import CONFIG

INPUT = Path("outputs/gaussian-geometry-v1/distribution-v2")
METHODS = {
    "A": (0, 64, False),
    "B": (1, 64, False),
    "C": (2, 64, False),
    "D": (3, 64, False),
    "B128": (1, 128, False),
    "D128": (3, 128, False),
    "C_Newton": (2, 64, True),
    "D_Newton": (3, 64, True),
}
LABELS = {-2: "INVALID", -1: "UNKNOWN", 0: "MISS", 1: "HIT"}
TRACE_NAMES = (
    "region_lit", "region_shadow", "region_solid", "unresolved_membership",
    "scalar_solid", "MISS", "HIT", "unresolved_visibility", "cached_lit",
    "cached_shadow",
)


def config():
    return json.loads(CONFIG.read_text())


def identity_file(path, identity):
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"Use a new output directory after changing inputs: {path}")
    else:
        atomic_json(path, identity)


def read_input(folder):
    scene = load(folder / "geometry.json")
    row = json.loads((folder / "result.json").read_text())
    source = np.asarray(row["source_xyz_synthetic"], dtype="<f8")
    geometry_id = record(scene)["geometry_id"]
    if row["geometry_id"] != geometry_id:
        raise ValueError(f"Geometry identity mismatch: {folder}")
    return scene, source, {
        "geometry_id": geometry_id,
        "geometry_file_SHA256": sha(folder / "geometry.json"),
        "source_file_SHA256": sha(folder / "result.json"),
        "source_SHA256": hashlib.sha256(source.tobytes()).hexdigest(),
        "source_hex": [float(x).hex() for x in source],
        "kernels": len(scene.kernels),
    }


def native_engine(module, scene):
    return module.Engine(
        [k.center.tolist() for k in scene.kernels],
        [k.matrix.ravel().tolist() for k in scene.kernels],
        [k.weight for k in scene.kernels],
        scene.threshold.value, scene.threshold.kind == "exp", True,
    )


def native_settings(module, method, cfg):
    opts, settings = module.Options(), module.Settings()
    opts.max_nodes = cfg["max_ray_nodes"]
    opts.max_depth = cfg["max_ray_depth"]
    opts.query_seconds = cfg["query_seconds"]
    settings.mode = 5
    settings.cover_depth = cfg["cover_depth"]
    settings.cover_nodes = cfg["cover_nodes"]
    settings.shadow_boxes = cfg["shadow_boxes"]
    settings.trace = True
    settings.study.method, settings.study.filter_bits, settings.study.stationary = METHODS[method]
    return opts, settings


def exact_hex(row, key):
    return row.get(f"{key}_hex", [float(x).hex() for x in row[f"{key}_xyz"]])


def check_status(actual, expected):
    if actual != expected:
        raise AssertionError(f"Frozen certified query expected {expected}, got {actual}")


def ray_trials(args):
    module = extension(args.native)
    fixture = json.loads(args.fixture.read_text())
    selected = [r for r in fixture["rays"] if args.split == "all" or r["split"] == args.split]
    opts_settings = {m: native_settings(module, m, config()) for m in args.methods}
    cases = sorted({r["case"] for r in selected})
    identities = {case: read_input(args.input / case)[2] for case in cases}
    identity_file(args.output / "identity.json", {
        "build": module.study_identity, "fixture_SHA256": sha(args.fixture),
        "cases": identities, "driver_SHA256": sha(Path(__file__)),
        "config_SHA256": sha(CONFIG), "methods": args.methods,
        "split": args.split, "repeats": args.repeats, "Arb_per_case": args.arb_per_case,
    })
    for case in cases:
        path = args.output / f"{case}.json"
        if path.exists():
            continue
        began = time.perf_counter()
        scene, source, identity = read_input(args.input / case)
        saved = next(r for r in fixture["scenes"] if r["case"] == case)
        if saved["geometry_id"] != identity["geometry_id"]:
            raise ValueError("Fixture geometry mismatch")
        start = time.perf_counter()
        engine = native_engine(module, scene)
        setup = time.perf_counter() - start
        clearance = NativeOracle(scene, independent=True).source_clearance(source)
        if clearance.status != "FREE":
            raise ValueError("Original independent source clearance required")
        rows = [r for r in selected if r["case"] == case]
        trials = []
        for i, row in enumerate(rows):
            if exact_hex(row, "source") != identity["source_hex"]:
                raise ValueError("Frozen source coordinate mismatch")
            receiver = [float.fromhex(x) for x in exact_hex(row, "receiver")]
            for rep in range(args.repeats):
                offset = (i + rep) % len(args.methods)
                for method in args.methods[offset:] + args.methods[:offset]:
                    start = time.perf_counter()
                    cpu = time.process_time()
                    trial = engine.ray(source, receiver, *opts_settings[method])
                    trial["cpu_seconds"] = time.process_time() - cpu
                    trial["wall_seconds"] = time.perf_counter() - start
                    trial["status"] = LABELS[trial["status"]]
                    check_status(trial["status"], row["expected"])
                    trials.append({
                        "id": row["ray_id"], "split": row["split"],
                        "strata": row.get("strata", []), "method": method,
                        "repeat": rep, **trial,
                    })
        # Original Arb remains a separately searching third oracle. IDs chosen
        # by hash, before observing this run's timings.
        arb_rows = []
        oracle = FieldOracle(scene, Limits(max_nodes=65536, max_depth=48, seconds=10))
        for row in sorted(rows, key=lambda r: r["ray_id"])[:args.arb_per_case]:
            start = time.perf_counter()
            decision = oracle.segment(source, [float.fromhex(x) for x in exact_hex(row, "receiver")])
            check_status(decision.status, row["expected"])
            arb_rows.append({"id": row["ray_id"], "status": decision.status,
                             "seconds": time.perf_counter() - start, "nodes": decision.nodes})
        atomic_json(path, {
            "case": case, "input": identity, "setup_seconds": setup,
            "wall_seconds": time.perf_counter() - began, "Arb": arb_rows, "trials": trials,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        })
        print(f"{case}: {len(rows)} rays, {len(trials)} trials", flush=True)


def latency(values_ns):
    if not len(values_ns):
        return {"count": 0, "seconds": 0, "P50_ms": None, "P95_ms": None,
                "P99_ms": None, "max_ms": None}
    p50, p95, p99 = np.percentile(values_ns, [50, 95, 99]) / 1e6
    return {"count": int(len(values_ns)), "seconds": float(values_ns.sum() / 1e9),
            "P50_ms": float(p50), "P95_ms": float(p95), "P99_ms": float(p99),
            "max_ms": float(values_ns.max() / 1e6)}


def trace_summary(arrays):
    paths = arrays["evaluation_path"]
    rays = arrays["ray_nanoseconds"]
    result = {name: int(np.count_nonzero(paths == i)) for i, name in enumerate(TRACE_NAMES, 1)}
    result["not_attempted"] = int(np.count_nonzero(paths == 0))
    for tag, label in ((6, "MISS"), (7, "HIT"), (8, "UNKNOWN")):
        result[f"{label}_latency"] = latency(rays[paths == tag])
    miss = np.sort(rays[paths == 6])
    result["MISS_lower_half_seconds"] = float(miss[:len(miss)//2].sum() / 1e9)
    result["MISS_upper_half_seconds"] = float(miss[len(miss)//2:].sum() / 1e9)
    result["point_seconds"] = float(arrays["point_nanoseconds"].sum() / 1e9)
    return result


def compare(arrays, baseline, input_record):
    row = json.loads(baseline.with_suffix(".json").read_text())
    if any(row[k] != input_record[k] for k in ("geometry_id", "source_SHA256")):
        raise ValueError(f"Compared field inputs differ: {baseline}")
    if sha(baseline) != row["archive_SHA256"]:
        raise ValueError(f"Compared field archive corrupt: {baseline}")
    with np.load(baseline) as base:
        if not base["valid"].all():
            raise ValueError(f"Compared field baseline is incomplete: {baseline}")
        common = arrays["valid"] & base["valid"].astype(bool)
        mismatch = {k: int(np.count_nonzero((arrays[k] != base[k]) & common)) for k in ("response", "solid")}
        same_paths = (np.array_equal(arrays["evaluation_path"], base["evaluation_path"])
                      if "evaluation_path" in arrays and "evaluation_path" in base else None)
    if any(mismatch.values()):
        raise AssertionError((baseline, mismatch))
    return {"baseline_archive_SHA256": row["archive_SHA256"],
            "matched_voxels": int(common.sum()), "bit_mismatches": mismatch,
            "same_evaluation_paths": same_paths}


def field_one(args):
    method, ordinal, n = args.methods[0], args.ordinals[0], args.resolutions[0]
    case = f"gaussian_sum-{ordinal:03d}"
    scene, source, info = read_input(args.input / case)
    t = time.perf_counter()
    if NativeOracle(scene, independent=True).source_clearance(source).status != "FREE":
        raise ValueError("Original independent source clearance required")
    clearance = time.perf_counter() - t
    cfg = config()
    t = time.perf_counter()
    cpu = time.process_time()
    if method in ("P", "A"):
        oracle = RegionOracle(scene, independent=method == "A", mode=5, trace=True,
                              cover_depth=cfg["cover_depth"], cover_nodes=cfg["cover_nodes"],
                              shadow_boxes=cfg["shadow_boxes"], max_nodes=cfg["max_ray_nodes"],
                              max_depth=cfg["max_ray_depth"], query_seconds=cfg["query_seconds"])
        setup = time.perf_counter() - t
        result = oracle.field(source, n, args.seconds)
    elif method == "Arb":
        oracle = FieldOracle(scene, Limits(max_nodes=65536, max_depth=48, seconds=10))
        setup = time.perf_counter() - t
        result = small_field(oracle, source, n, args.seconds)
    else:
        module = extension(args.native)
        engine = native_engine(module, scene)
        opts, settings = native_settings(module, "A" if method == "A_profile" else method, cfg)
        if module.study_identity["profile"] != (method == "A_profile"):
            raise ValueError("Use a timing-only build only for A_profile")
        setup = time.perf_counter() - t
        result = engine.field(source, n, args.seconds, opts, settings)
    cpu_seconds = time.process_time() - cpu
    cold_compute = time.perf_counter() - t
    arrays = {k: np.asarray(result.pop(k), dtype=bool) for k in ("response", "solid", "valid")}
    for k in ("evaluation_path", "ray_nanoseconds", "point_nanoseconds"):
        if k in result:
            arrays[k] = np.asarray(result.pop(k))
    if result["complete"] != bool(arrays["valid"].all()):
        raise AssertionError("Inconsistent completeness flag")
    comparison = None
    if args.baseline or method not in ("A", "P"):
        b = args.baseline or args.output
        baseline = b / f"{case}-{n}-A.npz"
        if not baseline.exists():
            baseline = b / f"{case}-{n}-A_independent.npz"
        comparison = compare(arrays, baseline, info)
    physical_sha = hashlib.sha256(np.packbits(arrays["response"]).tobytes() +
                                  np.packbits(arrays["solid"]).tobytes()).hexdigest()
    path = args.output / f"{case}-{n}-{method}"
    t = time.perf_counter()
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write = time.perf_counter() - t
    row = {
        **result, **info, "case": case, "resolution": n, "method": method,
        "CPU_seconds": cpu_seconds, "engine_setup_seconds": setup,
        "cold_compute_seconds": cold_compute, "source_clearance_seconds": clearance,
        "output_write_seconds": write,
        "cold_total_seconds": cold_compute + clearance + write,
        "process_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "valid_voxels": int(arrays["valid"].sum()), "solid_voxels": int(arrays["solid"].sum()),
        "archive_SHA256": sha(path.with_suffix(".npz")), "comparison": comparison,
        "physical_bits_SHA256": physical_sha if result["complete"] else None,
        "trace": trace_summary(arrays) if "evaluation_path" in arrays else None,
    }
    atomic_json(path.with_suffix(".json"), row)
    print(f"{case} {n} {method}: {row['valid_voxels']} valid, {row['cold_total_seconds']:.3f} s", flush=True)


def field_trials(args):
    rows = []
    native_identity = (json.loads((args.native / "identity.json").read_text())
                       if any(m not in ("A", "P", "Arb") for m in args.methods) else None)
    cases = [f"gaussian_sum-{i:03d}" for i in args.ordinals]
    baselines = {}
    if args.baseline:
        for case in cases:
            for n in args.resolutions:
                path = args.baseline / f"{case}-{n}-A.json"
                if not path.exists():
                    path = args.baseline / f"{case}-{n}-A_independent.json"
                baselines[path.name] = sha(path)
    identity_file(args.output / "identity.json", {
        "native": native_identity, "methods": args.methods, "resolutions": args.resolutions,
        "seconds": args.seconds, "config_SHA256": sha(CONFIG),
        "driver_SHA256": sha(Path(__file__)), "baseline_records_SHA256": baselines,
        "input": {c: read_input(args.input / c)[2] for c in cases},
    })
    for ordinal in args.ordinals:
        for n in args.resolutions:
            for method in args.methods:
                path = args.output / f"gaussian_sum-{ordinal:03d}-{n}-{method}.json"
                if not path.exists():
                    cmd = [sys.executable, "-m", __name__, "fields", "--one"]
                    # __name__ is __main__ when this driver is a module CLI.
                    cmd[2] = "experiments.gaussian_geometry.b2_independent.benchmark"
                    cmd += ["--input", str(args.input), "--output", str(args.output),
                            "--native", str(args.native), "--ordinals", str(ordinal),
                            "--resolutions", str(n), "--methods", method, "--seconds", str(args.seconds)]
                    if args.baseline:
                        cmd += ["--baseline", str(args.baseline)]
                    start = time.perf_counter()
                    subprocess.run(cmd, timeout=args.seconds + 120, check=True)
                    row = json.loads(path.read_text())
                    row["fresh_process_end_to_end_seconds"] = time.perf_counter() - start
                    atomic_json(path, row)
                row = json.loads(path.read_text())
                if row["archive_SHA256"] != sha(path.with_suffix(".npz")):
                    raise ValueError("Corrupt field checkpoint")
                rows.append(row)
                atomic_json(args.output / "summary.json", {"results": rows})
                if not row["complete"]:
                    raise RuntimeError(f"Incomplete field recorded at {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    r, f = sub.add_parser("rays"), sub.add_parser("fields")
    for parser in (r, f):
        parser.add_argument("--native", type=Path, default=DEFAULT / "native")
        parser.add_argument("--input", type=Path, default=INPUT)
        parser.add_argument("--output", type=Path, required=True)
    r.add_argument("--fixture", type=Path, required=True)
    r.add_argument("--methods", nargs="+", choices=METHODS, default=["A", "B", "C", "D"])
    r.add_argument("--split", choices=["all", "development", "evaluation"], default="all")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--arb-per-case", type=int, default=0)
    f.add_argument("--baseline", type=Path)
    f.add_argument("--methods", nargs="+", choices=[*METHODS, "P", "Arb", "A_profile"], required=True)
    f.add_argument("--ordinals", nargs="+", type=int, default=config()["frozen_ordinals"])
    f.add_argument("--resolutions", nargs="+", type=int, choices=[8, 16, 32, 64, 128], default=config()["resolutions"])
    f.add_argument("--seconds", type=float, default=config()["field_seconds"])
    f.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.task == "rays" and (args.repeats < 1 or args.arb_per_case < 0):
        p.error("Invalid repetition/Arb count")
    if args.task == "fields" and args.one and any(len(x) != 1 for x in (args.methods, args.ordinals, args.resolutions)):
        p.error("One method, case and resolution required for a worker")
    if args.task == "fields" and (128 in args.resolutions and not 0 < args.seconds <= 480):
        p.error("128³ requires a bounded field budget of at most 480 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task == "rays":
        ray_trials(args)
    elif args.one:
        field_one(args)
    else:
        field_trials(args)


if __name__ == "__main__":
    main()
