"""Final five-method, cold-cache comparison on the fixed nine small windows.

Run separately from production timing. Reference coordinates never become
proposals. Each method pays its own observation/constraint construction and
full independent verification; common geometry preparation is also reported.
"""
import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rind3d.enrichment_research import search
from rind3d.geometry import load_geometry
from rind3d.pack import RINDPack, atomic_json
from rind3d.rendering.context import GeometryContext
from rind3d.windows import Window
from .audit import BASE, OUT

METHODS = ("random", "local", "silhouette", "inverse", "hybrid")


def run(label="inverse-five-method-cold"):
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        raise FileExistsError("Preserve this matched comparison; choose a new version")
    frozen = json.loads((OUT / "inverse-v4-cold/summary.json").read_text())
    selected = [r["window"] for r in frozen["records"] if r["window"]["side"] == 64]
    records = []
    start, cpu = time.perf_counter(), time.process_time()
    for row in selected:
        target = folder / f"{row['scene_id']}-{row['window_id']}.json"
        if target.exists():
            records.append(json.loads(target.read_text()))
            continue
        path = BASE / "scenes" / row["scene_id"]
        t, c = time.perf_counter(), time.process_time()
        meta = json.loads((path / "verification.json").read_text())
        ctx = GeometryContext(load_geometry(path / "geometry.npz"), meta["numerical_policy"])
        ctx.geometry_proof()
        common_geometry_seconds = time.perf_counter() - t
        common_geometry_cpu_seconds = time.process_time() - c
        truth = np.fromfile(path / "driver.f64", dtype="<f8")
        window = Window(tuple(row["origin"]), row["side"])
        tests = []
        with RINDPack(path / "response.rindb", verify_file=True) as y, RINDPack(
            path / "solid.rindb", verify_file=True
        ) as m:
            # A seeded rotation avoids always giving the same method the first
            # CPU/cache position. Decoded bricks and source models are reset.
            order = np.random.default_rng(row["slot"] + 1701).permutation(METHODS)
            for method in order:
                ctx._cache.clear()
                y.cache.clear()
                m.cache.clear()
                points, proofs, report = search(
                    ctx, truth, window, y, m, seconds=2.0, method=str(method),
                    seed=31000 + row["slot"] * 100 + row["window_id"],
                    probe_strategy="transition",
                )
                tests.append({"budget": 2.0, "method": str(method),
                              "positions": points.tolist(), "proofs": proofs, "report": report})
        record = {"window": row, "tests": tests,
                  "common_required_geometry_preparation_seconds": common_geometry_seconds,
                  "common_required_geometry_preparation_cpu_seconds": common_geometry_cpu_seconds,
                  "method_order": list(map(str, order)), "cold_method_caches": True,
                  "reference_coordinates_never_used_as_proposals": True}
        atomic_json(target, record)
        records.append(record)
        atomic_json(folder / "progress.json", {"complete": False, "windows": len(records)})
    summaries = {}
    common = sum(r["common_required_geometry_preparation_seconds"] for r in records)
    common_cpu = sum(r["common_required_geometry_preparation_cpu_seconds"] for r in records)
    for method in METHODS:
        tests = [t for r in records for t in r["tests"] if t["method"] == method]
        refs = [(r["window"]["scene_id"], np.asarray(p, dtype="<f8").tobytes())
                for r in records for t in r["tests"] if t["method"] == method
                for p in t["positions"][1:]]
        stats = Counter()
        terminations = Counter()
        for t in tests:
            stats.update(t["report"]["stats"])
            terminations[t["report"]["termination"]] += 1
        wall = sum(t["report"]["elapsed_seconds"] for t in tests)
        summaries[method] = {"windows": len(tests), "additional_window_references": len(refs),
            "unique_scene_nontruth_coordinates": len(set(refs)),
            "unique_nontruth_coordinates": len({coordinates for _, coordinates in refs}),
            "successful_windows": sum(len(t["positions"]) > 1 for t in tests),
            "wall_seconds": wall, "cpu_seconds": sum(t["report"]["cpu_seconds"] for t in tests),
            "common_geometry_preparation_seconds": common,
            "common_geometry_preparation_cpu_seconds": common_cpu,
            "wall_seconds_including_full_common_geometry_cost": wall + common,
            "cpu_seconds_including_full_common_geometry_cost": common_cpu + sum(t["report"]["cpu_seconds"] for t in tests),
            "references_per_total_wall_second": len(refs) / (wall + common),
            "stats": dict(stats), "terminations": dict(terminations),
            "minimum_separation": 32 / 1024, "verification": "complete original A and B"}
    atomic_json(folder / "summary.json", {"complete": True, "windows": len(records),
        "methods": summaries, "records": records, "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "common_cost_policy": "One required geometry preparation per window; each standalone method total charges the full common cost. No cross-method source-model or decoded-field cache.",
        "scope": "Nine matched frozen 64^3 windows; descriptive small-sample study, not a production benefit or uniqueness claim."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="inverse-five-method-cold")
    run(parser.parse_args().label)
