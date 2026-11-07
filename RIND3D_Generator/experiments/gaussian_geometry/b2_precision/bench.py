"""Matched residual rays and fresh-process complete B2 fields."""
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
from ..b2_independent.adversarial import fixtures
from ..b2_independent.benchmark import (
    INPUT, LABELS, compare, config, exact_hex, identity_file, latency,
    native_engine, native_settings, read_input, trace_summary,
)
from ..b2_independent.build import extension as old_extension
from ..b2_native.api import NativeOracle
from ..b2_regions.api import RegionOracle
from ..queries import FieldOracle, Limits
from ..study import atomic_json
from .build import CONFIG, D_BUILD, OUT, extension


def settings(module, method, cfg):
    opts, cfg_native = native_settings(module, "A" if method == "A" else "D", cfg)
    if method in ("E", "F"):
        cfg_native.study.method = {"E": 4, "F": 5}[method]
    return opts, cfg_native


def run_rays(args):
    old, new = old_extension(args.baseline), extension(args.native)
    run_identity = {
        "builds": [old.study_identity, new.study_identity], "config_SHA256": sha(CONFIG),
        "driver_SHA256": sha(Path(__file__)), "repetitions": 3,
    }
    cases = {}
    for name, scene, source, receiver, expected in fixtures():
        trial = {}
        for method, module in (("A", old), ("D", old), ("E", new)):
            opts, native_cfg = settings(module, method, config())
            opts.max_nodes, opts.query_seconds = 2048, .4
            r = native_engine(module, scene).ray(source, receiver, opts, native_cfg)
            if LABELS[r["status"]] not in (expected, "UNKNOWN"):
                raise AssertionError((name, method, r))
            trial[method] = {**r, "status": LABELS[r["status"]]}
        oracle = FieldOracle(scene, Limits(max_nodes=8192, max_depth=40, seconds=.5))
        d = oracle.segment(source, receiver)
        if d.status in ("MISS", "HIT") and d.status != expected:
            raise AssertionError((name, "Arb", d))
        trial["Arb"] = {"status": d.status, "reason": d.reason, "nodes": d.nodes}
        cases[name] = {"expected": expected, "results": trial}
    atomic_json(args.output / "adversarial.json", {"identity": run_identity, "cases": cases})
    for population in ("residual", "followup", "ordinary"):
        fixture = args.fixtures / population / "frozen-rays.json"
        frozen = json.loads(fixture.read_text())
        groups = {}
        all_trials = []
        for case in sorted(frozen["scenes"], key=lambda r: r["case"]):
            scene, source, identity = read_input(args.input / case["case"])
            if identity["geometry_id"] != case["geometry_id"]:
                raise ValueError("Frozen ray geometry differs")
            engines = {m: (native_engine(mod, scene), settings(mod, m, config()))
                       for m, mod in (("A", old), ("D", old), ("E", new))}
            if NativeOracle(scene, independent=True).source_clearance(source).status != "FREE":
                raise ValueError("Frozen ray clearance differs")
            for j, row in enumerate(r for r in frozen["rays"] if r["case"] == case["case"]):
                if exact_hex(row, "source") != identity["source_hex"]:
                    raise ValueError("Frozen ray source differs")
                receiver = [float.fromhex(x) for x in exact_hex(row, "receiver")]
                for rep in range(3):
                    order = ("A", "D", "E")
                    offset = (rep + j) % 3
                    for method in order[offset:] + order[:offset]:
                        engine, opts = engines[method]
                        cpu = time.process_time()
                        start = time.perf_counter()
                        r = engine.ray(source, receiver, *opts)
                        wall = time.perf_counter() - start
                        cpu = time.process_time() - cpu
                        label = LABELS[r["status"]]
                        if label != row["expected"]:
                            raise AssertionError((population, row["ray_id"], method, label))
                        result = {**r, "status": label, "method": method, "id": row["ray_id"],
                                  "rep": rep, "wall_seconds": wall, "CPU_seconds": cpu}
                        all_trials.append(result)
                        strata = ("all", label) + (("expensive_MISS",) if "expensive_MISS" in row["strata"] else ())
                        for stratum in strata:
                            groups.setdefault((method, stratum), {}).setdefault(row["ray_id"], []).append(result)
        summary = []
        for (method, stratum), rays in sorted(groups.items()):
            rows = [x for values in rays.values() for x in values]
            work = Counter()
            for x in rows:
                work.update(x["work"])
            med = np.array([np.median([r["native_seconds"] for r in x]) for x in rays.values()])
            summary.append({"method": method, "stratum": stratum, "native": latency(med * 1e9),
                            "maximum_trial_seconds": max(r["native_seconds"] for r in rows),
                            "mean_CPU_total_seconds": sum(r["CPU_seconds"] for r in rows) / 3,
                            "work_all_trials": dict(work)})
        atomic_json(args.output / f"rays-{population}.json", {
            "identity": run_identity, "fixture_SHA256": sha(fixture), "summary": summary,
            "trials": all_trials,
        })
        print(population, len(all_trials), flush=True)


def prior_reference(previous, ordinal, n):
    case = f"gaussian_sum-{ordinal:03d}"
    group = f"fields128-case{ordinal}" if n == 128 else ("heldout-fields" if ordinal >= 60 else "frozen-fields")
    p = previous / group / f"{case}-{n}-D.npz"
    return p if p.exists() else None


def field_one(args):
    i, n, method = args.ordinals[0], args.resolutions[0], args.methods[0]
    case = f"gaussian_sum-{i:03d}"
    scene, source, identity = read_input(args.input / case)
    cfg = config()
    began = time.perf_counter()
    if NativeOracle(scene, independent=True).source_clearance(source).status != "FREE":
        raise ValueError("Independent source clearance required")
    clearance = time.perf_counter() - began
    began = time.perf_counter()
    cpu = time.process_time()
    if method in ("A", "P"):
        oracle = RegionOracle(scene, independent=method == "A", mode=5, trace=True,
                              cover_depth=cfg["cover_depth"], cover_nodes=cfg["cover_nodes"],
                              shadow_boxes=cfg["shadow_boxes"], max_nodes=cfg["max_ray_nodes"],
                              max_depth=cfg["max_ray_depth"], query_seconds=cfg["query_seconds"])
        setup = time.perf_counter() - began
        result = oracle.field(source, n, args.seconds)
    else:
        module = old_extension(args.baseline) if method == "D" else extension(args.native)
        oracle = native_engine(module, scene)
        opts, cfg_native = settings(module, method, cfg)
        setup = time.perf_counter() - began
        result = oracle.field(source, n, args.seconds, opts, cfg_native)
    compute = time.perf_counter() - began
    cpu = time.process_time() - cpu
    arrays = {k: np.asarray(result.pop(k), dtype=bool) for k in ("response", "solid", "valid")}
    for k in ("evaluation_path", "ray_nanoseconds", "point_nanoseconds"):
        arrays[k] = np.asarray(result.pop(k))
    valid = bool(arrays["valid"].all())
    if result["complete"] != valid:
        raise AssertionError("Native field completeness and validity disagree")
    baseline = prior_reference(args.previous, i, n)
    if baseline is None and method != "A":
        baseline = args.output / f"{case}-{n}-A.npz"
    reference = compare(arrays, baseline, identity) if baseline else None
    stem = args.output / f"{case}-{n}-{method}"
    began = time.perf_counter()
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    write = time.perf_counter() - began
    row = {
        **result, **identity, "case": case, "resolution": n, "method": method,
        "source_clearance_seconds": clearance, "engine_setup_seconds": setup,
        "cold_compute_seconds": compute, "output_write_seconds": write,
        "cold_total_seconds": clearance + compute + write, "CPU_seconds": cpu,
        "process_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "archive_SHA256": sha(stem.with_suffix(".npz")), "valid_voxels": int(arrays["valid"].sum()),
        "physical_bits_SHA256": hashlib.sha256(np.packbits(arrays["response"]).tobytes() +
                                              np.packbits(arrays["solid"]).tobytes()).hexdigest() if valid else None,
        "trace": trace_summary(arrays), "comparison": reference,
    }
    atomic_json(stem.with_suffix(".json"), row)
    print(case, n, method, row["valid_voxels"], round(row["cold_total_seconds"], 3), flush=True)


def run_fields(args):
    inputs = {i: read_input(args.input / f"gaussian_sum-{i:03d}")[2] for i in args.ordinals}
    refs = [prior_reference(args.previous, i, n) for i in args.ordinals for n in args.resolutions]
    identity_file(args.output / "identity.json", {
        "methods": args.methods, "ordinals": args.ordinals, "resolutions": args.resolutions,
        "seconds": args.seconds, "inputs": {str(k): v for k, v in inputs.items()},
        "reference_records": {str(p): sha(p.with_suffix(".json")) for p in refs if p},
        "local_build": json.loads((args.native / "identity.json").read_text()),
        "D_build": json.loads((args.baseline / "identity.json").read_text()),
        "driver_SHA256": sha(Path(__file__)), "config": config(),
    })
    rows = []
    for n in args.resolutions:
        for i in args.ordinals:
            for method in args.methods:
                path = args.output / f"gaussian_sum-{i:03d}-{n}-{method}.json"
                if not path.exists():
                    cmd = [sys.executable, "-m", "experiments.gaussian_geometry.b2_precision.bench", "fields", "--one",
                           "--native", str(args.native), "--baseline", str(args.baseline), "--input", str(args.input),
                           "--output", str(args.output), "--previous", str(args.previous), "--resolutions", str(n),
                           "--ordinals", str(i), "--methods", method, "--seconds", str(args.seconds)]
                    start = time.perf_counter()
                    subprocess.run(cmd, check=True, timeout=args.seconds + 120)
                    row = json.loads(path.read_text())
                    row["fresh_process_end_to_end_seconds"] = time.perf_counter() - start
                    atomic_json(path, row)
                row = json.loads(path.read_text())
                if sha(path.with_suffix(".npz")) != row["archive_SHA256"]:
                    raise ValueError("Field archive differs from checkpoint")
                rows.append(row)
                atomic_json(args.output / "summary.json", {"results": rows})
                if not row["complete"]:
                    raise RuntimeError(f"Incomplete field recorded at {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    rays, fields = sub.add_parser("rays"), sub.add_parser("fields")
    for cli in (rays, fields):
        cli.add_argument("--native", type=Path, default=OUT / "native-final")
        cli.add_argument("--baseline", type=Path, default=D_BUILD)
        cli.add_argument("--input", type=Path, default=INPUT)
        cli.add_argument("--output", type=Path, required=True)
    rays.add_argument("--fixtures", type=Path, default=Path("experiments/gaussian_geometry/fixtures/b2-max-cert-v1"))
    fields.add_argument("--previous", type=Path, default=Path("outputs/gaussian-b2-independent-v1"))
    fields.add_argument("--ordinals", nargs="+", type=int, default=[0, 7, 14, 21])
    fields.add_argument("--resolutions", nargs="+", type=int, choices=[32, 64, 128], required=True)
    fields.add_argument("--methods", nargs="+", choices=["A", "P", "D", "E", "F"], required=True)
    fields.add_argument("--seconds", type=float, default=180)
    fields.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "rays":
        run_rays(args)
    elif args.one:
        field_one(args)
    else:
        run_fields(args)


if __name__ == "__main__":
    main()
