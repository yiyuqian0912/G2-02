"""Diagnostic-only storage and geometry measurements for a pinned snapshot."""

from __future__ import annotations

import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .generation import stream
from .geometry import GeometryView
from .publication import _canonical_json, validate_snapshot
from .storage import pack_arrays, read_bundle


def benchmark_snapshot(root: str | Path, *, slot: int = 0,
                       output: str | Path | None = None) -> dict:
    root = Path(root)
    summary = validate_snapshot(root)
    snapshot = root / "snapshots" / summary["snapshot_id"]
    dimension = summary["dimension"]
    core_path = snapshot / f"d{dimension}" / "core.zarr"
    start = time.perf_counter()
    bundle = read_bundle(core_path, dimension)
    cold_load_seconds = time.perf_counter() - start
    arrays = pack_arrays(bundle)
    files = [path for path in core_path.rglob("*") if path.is_file()]
    raw_bytes = sum(array.nbytes for array in arrays.values())
    compressed_bytes = sum(path.stat().st_size for path in files)
    shard_files = [path for path in files if path.name != "zarr.json"]
    ledger = pq.read_table(snapshot / "catalogs" / "candidate_ledger.parquet").to_pylist()
    row = next((entry for entry in ledger if entry["slot_index"] == slot
                and entry["candidate_state"] == "READY"), None)
    if row is None:
        raise ValueError("requested benchmark slot is not READY")
    view = GeometryView(bytes.fromhex(row["geometry_id"]), bundle)
    if dimension == 2:
        rng = stream(2, slot, row["attempt_index"], 0, "diagnostics")
        sizes, warm_size, latency_count = (2**10, 2**14, 2**18), 1024, 256
    else:
        from .generation3d import stream3
        rng = stream3(slot, row["attempt_index"], 0, "diagnostics")
        sizes, warm_size, latency_count = (2**8, 2**10, 2**12), 128, 64
    view.classify_points(rng.random((warm_size, dimension)))  # warm backend path
    batches = []
    for size in sizes:
        points = rng.random((size, dimension))
        begun = time.perf_counter()
        result = view.classify_points(points)
        seconds = time.perf_counter() - begun
        batches.append({
            "points": size, "seconds": seconds,
            "points_per_second": size / seconds,
            "unresolved": int(np.count_nonzero(~result.valid)),
        })
    points = rng.random((latency_count, dimension))
    latencies = []
    for point in points:
        begun = time.perf_counter_ns()
        view.classify_points(point)
        latencies.append((time.perf_counter_ns() - begun) / 1e6)
    ordered = sorted(latencies)
    percentile = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    report = {
        "snapshot_id": summary["snapshot_id"], "dimension": dimension,
        "slot_index": slot,
        "scene_id": row["scene_id"], "geometry_id": row["geometry_id"],
        "raw_canonical_array_bytes": raw_bytes,
        "compressed_core_file_bytes": compressed_bytes,
        "compression_ratio_raw_over_disk": raw_bytes / compressed_bytes,
        "cold_bundle_reload_seconds": cold_load_seconds,
        "core_file_count": len(files), "shard_or_chunk_object_count": len(shard_files),
        "batch_classify": batches,
        "single_item_latency_ms": {"p50": statistics.median(ordered),
                                   "p95": percentile(.95), "p99": percentile(.99)},
        "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "unresolved_rate_in_largest_batch": batches[-1]["unresolved"] / batches[-1]["points"],
        "diagnostic_only": True,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(_canonical_json(report) + b"\n")
    return report
