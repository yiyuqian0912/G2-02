"""Measure D preparation on a seeded sample of exact full-field residual rays."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import INPUT, config, native_engine, native_settings, read_input
from ..b2_independent.build import extension as old_extension
from ..study import atomic_json
from .build import CONFIG, D_BUILD, OUT, extension

PREVIOUS = Path("outputs/gaussian-b2-independent-v1")


def audit(output, native, baseline, previous, inputs, probe):
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Pin the diagnostic to a single CPU with taskset")
    if output.exists():
        raise ValueError("Choose a new audit record")
    if probe:
        os.environ["RIND_B2_AUDIT_PROBE64"] = "1"
    else:
        os.environ.pop("RIND_B2_AUDIT_PROBE64", None)
    profiled = extension(native)
    old = old_extension(baseline)
    plan = json.loads(CONFIG.read_text())
    rng = np.random.default_rng(plan["audit_seed"])
    rows = []
    for n, cases in ((64, plan["historical_cases"]), (128, [14, 7])):
        for i in cases:
            case = f"gaussian_sum-{i:03d}"
            archive = previous / ("frozen-fields" if n == 64 else f"fields128-case{i}") / f"{case}-{n}-D.npz"
            old_row = json.loads(archive.with_suffix(".json").read_text())
            scene, source, identity = read_input(inputs / case)
            if old_row["archive_SHA256"] != sha(archive) or any(old_row[k] != identity[k] for k in ("geometry_id", "source_SHA256")):
                raise ValueError(f"Frozen trace identity mismatch for {archive}")
            with np.load(archive) as f:
                if not old_row["complete"] or not f["valid"].all():
                    raise ValueError("Audit requires a complete D trace")
                path = f["evaluation_path"].ravel()
            engines = [(native_engine(m, scene), native_settings(m, "D", config())) for m in (old, profiled)]
            groups = []
            for tag in (6, 7):
                population = np.flatnonzero(path == tag)
                count = min(len(population), plan["audit_samples_per_label"] if not probe else 128)
                indices = np.sort(rng.choice(population, count, replace=False))
                work = Counter()
                seconds = [0., 0.]
                for j, idx in enumerate(indices):
                    z, y, x = np.unravel_index(idx, (n, n, n))
                    receiver = [(c + .5) / n for c in (x, y, z)]
                    # Counterfactual samples report only counts.
                    order = (1,) if probe else ((0, 1) if j % 2 == 0 else (1, 0))
                    for k in order:
                        engine, args = engines[k]
                        ans = engine.ray(source, receiver, *args)
                        if ans["status"] != tag - 6:
                            raise AssertionError((case, n, int(idx), k, ans))
                        seconds[k] += ans["native_seconds"]
                        if k == 1:
                            work.update(ans["work"])
                groups.append({
                    "label": "MISS" if tag == 6 else "HIT", "population": len(population),
                    "indices": indices.tolist(), "sample_size": count,
                    "native_seconds_D_and_instrumented": None if probe else seconds,
                    "sample_work": dict(work),
                })
            rows.append({
                "case": case, "grid": n, "input": identity, "trace": str(archive),
                "trace_SHA256": old_row["archive_SHA256"],
                "physical_bits_SHA256": old_row["physical_bits_SHA256"],
                "full_D_cold_seconds": old_row["cold_total_seconds"],
                "full_D_scalar_seconds": old_row["trace"]["MISS_latency"]["seconds"] + old_row["trace"]["HIT_latency"]["seconds"],
                "full_D_work": old_row["work"], "groups": groups,
            })
            print(case, n, [(g["label"], g["sample_size"]) for g in groups], flush=True)
    atomic_json(output, {
        "probe_only": probe, "affinity": sorted(os.sched_getaffinity(0)),
        "config_SHA256": sha(CONFIG), "instrumented_build": profiled.study_identity,
        "D_build": old.study_identity, "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "rows": rows,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native", type=Path, default=OUT / "audit-native")
    p.add_argument("--baseline", type=Path, default=D_BUILD)
    p.add_argument("--previous", type=Path, default=PREVIOUS)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--probe", action="store_true")
    a = p.parse_args()
    audit(a.output, a.native, a.baseline, a.previous, a.input, a.probe)
