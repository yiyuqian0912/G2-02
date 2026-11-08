"""Original paired-octant admission and at most one bounded dual field."""
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_miss.bench import compare, read
from ..b2_precision.build import extension
from ..b2_precision.tiles import arrays, digest
from ..b2_primary_miss.gate import (
    GRID, LIMITS, RSS_CAP_MIB, SIDE, admission, origins, validate_archives,
)
from ..study import atomic_json
from .bench import make_session
from .freeze import NATIVE, OUT, PLAN


def worker(args, probe, rest):
    role, phase = args.worker, args.phase
    plan = read(args.output/"identity.json")
    if plan["plan_SHA256"] != sha(args.plan) or plan["primary"] != args.primary:
        raise ValueError("Worker input differs from frozen trial")
    if phase == "remaining" and not read(args.output/"admission.json")["admitted"]:
        raise ValueError("Both backend estimates must pass before remaining tiles")
    folder = args.output/role
    folder.mkdir(exist_ok=True)
    result_path = folder/(phase + ".json")
    if result_path.exists():
        raise ValueError("Preserve the existing measured phase")
    variant = args.primary if role == "primary" else "F_J"
    module = extension(args.native if role == "primary" else NATIVE)
    if module.study_identity != plan["builds"][role]:
        raise ValueError("Worker native build differs from the trial identity")
    wall, cpu = time.perf_counter(), time.process_time()
    spent = read(folder/"probe.json")["wall_seconds"] if phase == "remaining" else 0.
    cap = LIMITS[role]-spent
    setup, info, rows = 0., None, []
    if cap > 0:
        session, setup, _ = make_session(module, probe[0], variant, False)
        info = session.info()
        for tile in probe if phase == "probe" else rest:
            remaining = cap-(time.perf_counter()-wall)
            if remaining <= 0 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > RSS_CAP_MIB:
                break
            archive = folder/(tile["id"] + ".npz")
            if archive.exists():
                raise ValueError("Preserve the existing tile")
            start = time.perf_counter()
            data = session.tile(tile["origin"], SIDE, min(180.,remaining))
            bits = arrays(data)
            np.savez_compressed(archive, **bits)
            seconds = time.perf_counter()-start
            complete = bool(bits["valid"].all())
            if complete != data["complete"]:
                raise AssertionError("Tile validity differs from completeness")
            if phase == "probe":
                compare(bits, Path(tile["prior_archive"]), tile["prior_archive_SHA256"])
            row = {"id": tile["id"], "origin": tile["origin"], "complete": complete,
                   "reason": data["reason"], "tile_seconds": seconds, "native_seconds": data["wall_seconds"],
                   "valid_voxels": int(bits["valid"].sum()), "archive_SHA256": sha(archive),
                   "physical_bits_SHA256": digest(bits) if complete else None, "work": data["work"]}
            atomic_json(archive.with_suffix(".json"), row)
            rows.append(row)
            print(role, phase, tile["id"], round(seconds,3), row["valid_voxels"], flush=True)
            if not complete:
                break
    atomic_json(result_path, {"role": role, "variant": variant, "build": module.study_identity,
                 "rows": rows, "session": info, "setup_seconds": setup,
                 "wall_seconds": time.perf_counter()-wall, "CPU_seconds": time.process_time()-cpu,
                 "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def run(args):
    probe, rest = origins(args.plan)
    if args.worker:
        return worker(args, probe, rest)
    args.output.mkdir(parents=True)
    atomic_json(args.output/"identity.json", {"plan_SHA256": sha(args.plan),
                "driver_SHA256": sha(Path(__file__)), "geometry": probe[0]["geometry"],
                "primary": args.primary, "independent": "F_J",
                "builds": {"primary": read(args.native/"identity.json"),
                           "independent": read(NATIVE/"identity.json")},
                "grid": GRID, "side": SIDE, "limits_seconds": LIMITS, "RSS_MiB": RSS_CAP_MIB})

    def launch(role, phase, cap):
        cmd = [sys.executable, "-m", "experiments.gaussian_geometry.b2_hit.gate",
               "--output", str(args.output), "--plan", str(args.plan), "--native", str(args.native),
               "--primary", args.primary, "--worker", role, "--phase", phase]
        try:
            subprocess.run(cmd, check=True, timeout=max(0.,cap)+30)
        except subprocess.TimeoutExpired:
            atomic_json(args.output/f"{role}-{phase}-watchdog.json",
                        {"reason": "process watchdog", "phase": phase, "role": role})

    estimates = {}
    for role, cap in LIMITS.items():
        launch(role,"probe",cap)
        path = args.output/role/"probe.json"
        estimates[role] = admission(read(path),cap) if path.is_file() else {
            "accepted": False, "reason": "process watchdog"}
    measured = validate_archives(args.output, probe)
    admitted = all(r["accepted"] for r in estimates.values()) and measured["matched_voxels"] == 16*SIDE**3
    atomic_json(args.output/"admission.json", {"estimates": estimates, "admitted": admitted,
                                              "probe_coverage": measured})
    if admitted:
        for role, cap in LIMITS.items():
            spent = read(args.output/role/"probe.json")["wall_seconds"]
            launch(role,"remaining",cap-spent)
    coverage = validate_archives(args.output, probe + rest if admitted else probe)
    stages = {}
    for role in LIMITS:
        stages[role] = {}
        for phase in ("probe","remaining"):
            path = args.output/role/(phase + ".json")
            if path.is_file():
                data = read(path)
                stages[role][phase] = {k: data[k] for k in ("wall_seconds","CPU_seconds","peak_RSS_MiB")}
    within = all("probe" in stages[role] and (not admitted or "remaining" in stages[role])
                 and sum(r["wall_seconds"] for r in stages[role].values()) <= cap
                 and all(r["peak_RSS_MiB"] <= RSS_CAP_MIB for r in stages[role].values())
                 for role, cap in LIMITS.items())
    atomic_json(args.output/"coverage.json", {**coverage, "admitted": admitted,
                "within_resource_caps": within, "stages": stages})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--native", type=Path, default=OUT/"live-native")
    p.add_argument("--primary", choices=["P_JH1","P_JH4","P_JHA"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=LIMITS, help=argparse.SUPPRESS)
    p.add_argument("--phase", choices=["probe","remaining"], help=argparse.SUPPRESS)
    args = p.parse_args()
    if bool(args.worker) != bool(args.phase):
        p.error("Worker and phase must be specified together")
    run(args)
