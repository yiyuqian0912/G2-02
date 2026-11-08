"""Original 256-grid admission and conditional, bounded dual-field run."""
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_miss.bench import compare, make_session as previous_session
from ..b2_precision.build import extension
from ..b2_precision.tiles import CAPS, RSS_CAP_MIB, arrays, digest, plan_256
from ..study import atomic_json
from .bench import MISS_OUT, OUT, PLAN, make_session, read

LIMITS = {"primary": CAPS["P"], "independent": CAPS["F"]}
GRID, SIDE = 256, 64


def origins(plan):
    probe, rest = plan_256()
    first = [row for row in read(plan)["tiles"] if row["case"] == "gaussian_sum-014"]
    if [tuple(row["origin"]) for row in first] != probe:
        raise ValueError("Original paired-octant probe order required")
    second = [{k: first[0][k] for k in ("case", "input", "geometry", "grid", "side")}
              | {"id": f"extra-{i:02}", "origin": list(x)}
              for i, x in enumerate(rest)]
    return first, second


def trial(args, probe, rest):
    role, phase = args.worker, args.phase
    variant = args.primary if role == "primary" else args.independent
    identity = read(args.output / "identity.json")
    if identity[role] != variant or identity["plan_SHA256"] != sha(args.plan):
        raise ValueError("Worker does not match the admitted plan")
    if phase == "remaining" and not read(args.output / "admission.json")["admitted"]:
        raise ValueError("Both probe estimates must pass before rendering remaining tiles")
    folder = args.output / role
    folder.mkdir(exist_ok=True)
    if variant == "F_I":
        module = extension(MISS_OUT / "propagation-native")
        factory = lambda: previous_session(module, probe[0], 3, independent=True, trace=False)
    else:
        module = extension(args.native)
        factory = lambda: make_session(module, probe[0], variant, False)[:2]
    wall, cpu = time.perf_counter(), time.process_time()
    spent = read(folder / "probe.json")["wall_seconds"] if phase == "remaining" else 0.
    cap = LIMITS[role] - spent
    setup, info, rows = 0., None, []
    if cap > 0:
        session, setup = factory()
        info = session.info()
        for tile in probe if phase == "probe" else rest:
            remaining = cap - (time.perf_counter() - wall)
            if remaining <= 0 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > RSS_CAP_MIB:
                break
            t = time.perf_counter()
            data = session.tile(tile["origin"], SIDE, min(180., remaining))
            bits = arrays(data)
            archive = folder / (tile["id"] + ".npz")
            np.savez_compressed(archive, **bits)
            secs = time.perf_counter() - t
            complete = bool(bits["valid"].all())
            if complete != data["complete"]:
                raise AssertionError("Validity and completeness disagree")
            if phase == "probe":
                compare(bits, Path(tile["prior_archive"]), tile["prior_archive_SHA256"])
            row = {"id": tile["id"], "origin": tile["origin"], "complete": complete,
                   "reason": data["reason"], "tile_seconds": secs, "native_seconds": data["wall_seconds"],
                   "valid_voxels": int(bits["valid"].sum()), "archive_SHA256": sha(archive),
                   "physical_bits_SHA256": digest(bits) if complete else None, "work": data["work"]}
            atomic_json(archive.with_suffix(".json"), row)
            rows.append(row)
            print(role, phase, tile["id"], round(secs, 3), row["valid_voxels"], flush=True)
            if not complete:
                break
    atomic_json(folder / (phase + ".json"),
                {"role": role, "variant": variant, "build": module.study_identity,
                 "rows": rows, "session": info, "setup_seconds": setup,
                 "wall_seconds": time.perf_counter()-wall,
                 "CPU_seconds": time.process_time()-cpu,
                 "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def admission(record, cap):
    rows = record["rows"]
    complete = len(rows) == 16 and all(r["complete"] for r in rows)
    paired = [max(rows[i:i+2], key=lambda r: r["tile_seconds"])
              for i in range(0, len(rows), 2)]
    setup = 2 * record["setup_seconds"]
    projected = setup + 4 * sum(r["tile_seconds"] for r in rows)
    guarded = setup + 1.2 * sum(8 * r["tile_seconds"] for r in paired)
    return {"accepted": complete and guarded <= cap and record["peak_RSS_MiB"] <= RSS_CAP_MIB
                        and record["wall_seconds"] <= cap,
            "variant": record["variant"], "probe_complete": complete,
            "seconds_cap": cap, "memory_cap_MiB": RSS_CAP_MIB,
            "projected_wall_seconds": projected if complete else None,
            "guarded_planning_seconds": guarded if complete else None,
            "peak_RSS_MiB": record["peak_RSS_MiB"], "probe_seconds": record["wall_seconds"],
            "probe_CPU_seconds": record["CPU_seconds"],
            "octant_slowest": [{k: r[k] for k in ("id", "origin", "tile_seconds")} for r in paired]}


def validate_archives(root, tiles):
    valid = {role: 0 for role in LIMITS}
    matched, all_paired = 0, True
    for tile in tiles:
        bits = {}
        for role in LIMITS:
            archive = root / role / (tile["id"] + ".npz")
            meta = archive.with_suffix(".json")
            if not meta.is_file():
                continue
            row = read(meta)
            if row["origin"] != tile["origin"] or sha(archive) != row["archive_SHA256"]:
                raise ValueError("Tile identity or checksum mismatch")
            with np.load(archive) as data:
                field = arrays(data)
            if any(a.shape != (SIDE,)*3 for a in field.values()):
                raise ValueError("Tile array shape differs from the admission partition")
            if int(field["valid"].sum()) != row["valid_voxels"]:
                raise ValueError("Incorrect valid-voxel count")
            if row["complete"] != bool(field["valid"].all()):
                raise ValueError("Incorrect completeness flag")
            if row["complete"] and digest(field) != row["physical_bits_SHA256"]:
                raise ValueError("Incorrect physical digest")
            valid[role] += row["valid_voxels"]
            bits[role] = field
        if len(bits) == 2:
            a, b = (bits[role] for role in LIMITS)
            common = a["valid"] & b["valid"]
            if any(not np.array_equal(a[k][common], b[k][common]) for k in ("response", "solid")):
                raise AssertionError("Independent field disagreement")
            matched += int(common.sum())
            all_paired &= bool(common.all())
        else:
            all_paired = False
    return {"valid_by_backend": valid, "matched_voxels": matched,
            "coverage_fraction": matched / GRID**3,
            "complete_dual_field": len(tiles) == 64 and all_paired and matched == GRID**3}


def run(args):
    probe, rest = origins(args.plan)
    if args.worker:
        return trial(args, probe, rest)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "identity.json",
                {"plan_SHA256": sha(args.plan), "driver_SHA256": sha(Path(__file__)),
                 "primary": args.primary, "independent": args.independent,
                 "grid": GRID, "side": SIDE, "limits_seconds": LIMITS, "RSS_MiB": RSS_CAP_MIB})

    def worker(role, phase):
        cmd = [sys.executable, "-m", "experiments.gaussian_geometry.b2_primary_miss.gate",
               "--output", str(args.output), "--plan", str(args.plan), "--native", str(args.native),
               "--primary", args.primary, "--independent", args.independent,
               "--worker", role, "--phase", phase]
        try:
            subprocess.run(cmd, check=True, timeout=LIMITS[role] + 30)
        except subprocess.TimeoutExpired:
            atomic_json(args.output / f"{role}-{phase}-watchdog.json",
                        {"reason": "process watchdog", "phase": phase, "role": role})

    estimates = {}
    for role, cap in LIMITS.items():
        worker(role, "probe")
        path = args.output / role / "probe.json"
        estimates[role] = admission(read(path), cap) if path.is_file() else {
            "accepted": False, "reason": "process watchdog"}
    measured = validate_archives(args.output, probe)
    admitted = all(r["accepted"] for r in estimates.values()) and measured["matched_voxels"] == 16 * SIDE**3
    atomic_json(args.output / "admission.json", {"estimates": estimates, "admitted": admitted,
                                                "probe_coverage": measured})
    if admitted:
        for role in LIMITS:
            worker(role, "remaining")
    coverage = validate_archives(args.output, probe + rest if admitted else probe)
    stages = {}
    for role in LIMITS:
        phases = {p: read(args.output / role / f"{p}.json") for p in ("probe", "remaining")
                  if (args.output / role / f"{p}.json").exists()}
        stages[role] = {p: {k: row[k] for k in ("wall_seconds", "CPU_seconds", "peak_RSS_MiB")}
                        for p, row in phases.items()}
    within = all("probe" in stages[role] and (not admitted or "remaining" in stages[role])
                 and sum(r["wall_seconds"] for r in stages[role].values()) <= cap
                 and all(r["peak_RSS_MiB"] <= RSS_CAP_MIB for r in stages[role].values())
                 for role, cap in LIMITS.items())
    atomic_json(args.output / "coverage.json", {**coverage, "admitted": admitted,
                                                "within_resource_caps": within, "stages": stages})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--native", type=Path, default=OUT / "propagation-native")
    p.add_argument("--primary", choices=["P_H", "P_I", "P_J"], required=True)
    p.add_argument("--independent", choices=["F_I", "F_J"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=LIMITS, help=argparse.SUPPRESS)
    p.add_argument("--phase", choices=["probe", "remaining"], help=argparse.SUPPRESS)
    args = p.parse_args()
    if bool(args.worker) != bool(args.phase):
        p.error("Worker and phase must be specified together")
    run(args)
