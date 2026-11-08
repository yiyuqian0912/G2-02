"""Compare exact primary tiles against unchanged independent F_J evidence."""
import argparse
from pathlib import Path
import resource
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import config, trace_summary
from ..b2_miss.bench import compare, load_inputs, read
from ..b2_precision.bench import settings
from ..b2_precision.build import extension
from ..b2_precision.tiles import digest
from ..b2_primary_miss.bench import make_session as previous_session
from ..study import atomic_json
from .freeze import NATIVE, OUT, PLAN, PRIMARY

MODES = {"P_J": 0, "P_JHC": 1, "P_JH1": 2, "P_JH4": 3, "P_JHA": 4, "P_JHG4": 5}


def make_session(module, row, variant, trace):
    if variant in ("P_J", "F_J"):
        return previous_session(module, row, variant, trace)
    scene, source = load_inputs(row)
    start, cpu = time.perf_counter(), time.process_time()
    engine = module.Engine([k.center.tolist() for k in scene.kernels],
                           [k.matrix.ravel().tolist() for k in scene.kernels],
                           [k.weight for k in scene.kernels], scene.threshold.value,
                           scene.threshold.kind == "exp", False)
    opts, cfg = settings(module, "A", config())
    cfg.trace = trace
    cfg.study.miss_mode = 4
    cfg.study.hit_mode = MODES[variant]
    session = engine.session(source, row["grid"], opts, cfg, True)
    return session, time.perf_counter()-start, time.process_time()-cpu


def accounting(bits, baseline, work):
    path = bits["evaluation_path"]
    used_miss = bits["propagated_seed"] >= 0
    used_hit = (bits["hit_propagated_seed"] >= 0 if "hit_propagated_seed" in bits
                else np.zeros(path.shape, dtype=bool))
    if (not np.all(path[used_hit] == 7) or not np.all(path[used_miss] == 6)
            or used_hit.sum() != work.get("hit_cache_hits",0)
            or used_miss.sum() != work.get("miss_cache_hits",0)):
        raise ValueError("Trace does not identify an actually skipped scalar query")
    hit_time = bits["ray_nanoseconds"][(path == 7) & ~used_hit].sum()
    row = {"actual_HIT_skipped": int(used_hit.sum()), "actual_MISS_skipped": int(used_miss.sum()),
           "remaining_scalar_HIT_ns": int(hit_time),
           "HIT_cache_path_ns": int(bits["ray_nanoseconds"][used_hit].sum())}
    if baseline is not None:
        if sha(baseline) != read(baseline.with_suffix(".json"))["archive_SHA256"]:
            raise ValueError("Original P_J trace checksum differs")
        with np.load(baseline) as a:
            if not (np.array_equal(path, a["evaluation_path"])
                    and np.array_equal(bits["propagated_seed"], a["propagated_seed"])):
                raise AssertionError("Original J MISS propagation or region path changed")
            original = a["ray_nanoseconds"]
            hit = original[path == 7]
            q = float(np.quantile(hit,.75)) if len(hit) else float("inf")
            row.update(original_HIT_ns=int(hit.sum()),
                       original_HIT_ns_avoided=int(original[used_hit].sum()),
                       original_top_quartile_HIT_skipped=int(np.count_nonzero(used_hit & (original >= q))),
                       original_top_quartile_HIT_ns_avoided=int(original[used_hit & (original >= q)].sum()))
    return row


def run(args):
    frozen = read(args.plan)["tiles"]
    plan = [r for r in frozen if not args.ids or r["id"] in args.ids]
    if not plan or (args.ids and {r["id"] for r in plan} != set(args.ids)):
        raise ValueError("Expected a nonempty subset of the frozen plan")
    root = args.output
    root.mkdir(parents=True)
    module = extension(args.native)
    atomic_json(root / "identity.json",
                {"build": module.study_identity, "plan_SHA256": sha(args.plan),
                 "variant": args.variant, "ids": [r["id"] for r in plan],
                 "independent_reference": str(args.reference), "scalar_reference": str(args.baseline),
                 "driver_SHA256": sha(Path(__file__)), "options": config()})
    wall, cpu = time.perf_counter(), time.process_time()
    rows, setups, case = [], [], None
    for tile in plan:
        if tile["case"] != case:
            session, secs, cpu_secs = make_session(module, tile, args.variant, True)
            case = tile["case"]
            setups.append({"case": case, "seconds": secs, "CPU_seconds": cpu_secs,
                           "info": session.info(), "work": session.preparation_work()})
        t, c = time.perf_counter(), time.process_time()
        data = session.tile(tile["origin"], tile["side"], config()["field_seconds"])
        secs, cpu_s = time.perf_counter()-t, time.process_time()-c
        bits = {k: np.asarray(data.pop(k),dtype=bool) for k in ("response", "solid", "valid")}
        for k in ("evaluation_path","ray_nanoseconds","point_nanoseconds","propagated_seed","hit_propagated_seed"):
            if k in data:
                bits[k] = np.asarray(data.pop(k))
        archive = root/(tile["id"] + ".npz")
        ref = args.reference/archive.name
        original = args.baseline/archive.name if args.baseline else None
        complete = bool(bits["valid"].all())
        if complete != data["complete"]:
            raise AssertionError("Validity disagrees with completeness")
        compare(bits, ref)
        np.savez_compressed(archive, **bits)
        row = {**tile, **data, "tile_call_seconds": secs, "tile_CPU_seconds": cpu_s,
               "valid_voxels": int(bits["valid"].sum()), "archive_SHA256": sha(archive),
               "physical_bits_SHA256": digest(bits) if complete else None,
               "trace": trace_summary(bits), "accounting": accounting(bits, original, data["work"]) if complete else None}
        if complete and row["physical_bits_SHA256"] != read(ref.with_suffix(".json"))["physical_bits_SHA256"]:
            raise AssertionError("Physical digest differs from F_J")
        atomic_json(archive.with_suffix(".json"), row)
        rows.append(row)
        atomic_json(root / "summary.json", {"records": rows, "setups": setups,
                    "elapsed_wall_seconds": time.perf_counter()-wall, "CPU_seconds": time.process_time()-cpu,
                    "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        print(tile["id"], round(secs,3), row["valid_voxels"],
              row["accounting"]["actual_HIT_skipped"] if complete else "incomplete", flush=True)
        if not complete:
            raise RuntimeError((tile["id"], data["reason"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=MODES)
    p.add_argument("--native", type=Path)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--reference", type=Path, default=PRIMARY/"F_J")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+")
    args = p.parse_args()
    if args.native is None:
        args.native = NATIVE if args.variant == "P_J" else OUT/"live-native"
    run(args)
