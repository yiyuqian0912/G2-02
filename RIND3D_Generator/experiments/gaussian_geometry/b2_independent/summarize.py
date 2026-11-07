"""Aggregate measured workloads without treating a partial field as complete."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from ..b2_angular.preservation import sha
from ..study import atomic_json
from .benchmark import latency, trace_summary


def load_json(path):
    return json.loads(path.read_text())


def ray_summary(root):
    by_group = defaultdict(list)
    manifests = {}
    for path in sorted(root.glob("rays-*/identity.json")):
        parent = path.parent
        manifests[parent.name] = {"SHA256": sha(path), **load_json(path)}
        for case in sorted(parent.glob("gaussian_sum-*.json")):
            data = load_json(case)
            manifests[parent.name].setdefault("trial_files_SHA256", {})[case.name] = sha(case)
            manifests[parent.name].setdefault("geometry_setup_and_Arb", {})[data["case"]] = {
                k: data[k] for k in ("setup_seconds", "wall_seconds", "peak_RSS_MiB", "Arb")
            }
            for r in data["trials"]:
                group = (parent.name, r["method"])
                by_group[(*group, "all")].append(r)
                by_group[(*group, r["status"])].append(r)
                if "expensive_MISS" in r["strata"]:
                    by_group[(*group, "expensive_MISS")].append(r)
                    if r["split"] == "evaluation":
                        by_group[(*group, "evaluation_expensive_MISS")].append(r)
    results = []
    for (corpus, method, population), trials in sorted(by_group.items()):
        by_ray = defaultdict(list)
        for r in trials:
            by_ray[r["id"]].append(r)
        medians = np.array([np.median([r["native_seconds"] for r in rows]) for rows in by_ray.values()])
        repeat_count = len(trials) / len(by_ray)
        work = Counter()
        for r in trials:
            for key, value in r["work"].items():
                if key == "reference_max_depth":
                    work[key] = max(work[key], value)
                else:
                    work[key] += value
        results.append({
            "corpus": corpus, "method": method, "population": population,
            "native_latency": latency(medians * 1e9),
            "max_single_native_ms": 1000 * max(r["native_seconds"] for r in trials),
            "CPU_seconds_mean_total": sum(r["cpu_seconds"] for r in trials) / repeat_count,
            "wall_seconds_mean_total": sum(r["wall_seconds"] for r in trials) / repeat_count,
            "trials": len(trials), "work_all_trials": dict(work),
        })
    return {"workloads": manifests, "groups": results}


def field_summary(root, exclude=()):
    records = {}
    manifests = {}
    for path in sorted(root.glob("*/summary.json")):
        if path.parent.name in exclude:
            continue
        data = load_json(path)
        if not data.get("results") or "resolution" not in data["results"][0]:
            continue
        identity = path.parent / "identity.json"
        if not identity.exists():
            identity = path.parent / "manifest.json"
        manifests[path.parent.name] = {"summary_SHA256": sha(path), "identity_SHA256": sha(identity)}
        for r in data["results"]:
            name = f"{r['case']}-{r['resolution']}-{r['method']}"
            archive = path.parent / (name + ".npz")
            if sha(archive) != r["archive_SHA256"]:
                raise ValueError(f"Field trace identity differs: {archive}")
            row = {**r, "run": path.parent.name}
            if row.get("trace") is None:
                with np.load(archive) as a:
                    if "evaluation_path" in a:
                        row["trace"] = trace_summary(a)
            records[(row["run"], name)] = row
    # Count unique complete physical workloads, with experimental variants
    # attached to the same geometry, source and sampling resolution.
    by_support = defaultdict(list)
    for r in records.values():
        by_support[(r["geometry_id"], r["source_SHA256"], r["resolution"])].append(r)
    for same in by_support.values():
        if len({r["physical_bits_SHA256"] for r in same if r["complete"]}) > 1:
            raise AssertionError("Complete-field physical bits disagree")
    return {"workloads": manifests, "physical_supports": len(by_support),
            "records": list(records.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-run", action="append", default=[])
    args = parser.parse_args()
    atomic_json(args.output, {
        "schema": "b2-independent-verifier-measured-v1",
        "rays": ray_summary(args.root), "fields": field_summary(args.root, args.exclude_run),
        "excluded_field_runs": args.exclude_run,
    })
