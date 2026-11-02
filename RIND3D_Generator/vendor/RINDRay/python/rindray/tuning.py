"""Bounded, reproducible calibration with disposable local JSON policies."""

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np

from . import _core
from .drivers import MODELS
from .scene import integer


@lru_cache(maxsize=8)
def hardware(available):
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    sources = b"".join((Path(__file__).parent / name).read_bytes() for name in ("scene.py", "drivers.py", "grids.py", "tuning.py", "sampling.py", "streaming.py"))
    fingerprint = hashlib.sha256(_core.build_id().encode() + sources).hexdigest()
    return {"cpu": cpu, "architecture": platform.machine(), "usable_cpus": available,
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "embree": _core.embree_version(), "core_version": "0.1.0", "build_fingerprint": fingerprint}


def policy_key(scene, n, operation, sources=0, model=None):
    lookup = (n, operation, sources, model, scene.engine.backend, scene.engine.num_threads,
              scene.engine.memory_budget_bytes, scene.engine.expected_scene_reuse,
              scene.engine._measurements is not None)
    if lookup in scene._policy_keys:
        return scene._policy_keys[lookup]
    descriptor = {"schema": 1, "hardware": hardware(scene.engine.available_threads),
                  "geometry_counts": list(scene._native.geometry_counts), "points": n,
                  "operation": operation, "sources": sources, "model": model,
                  "backend_request": scene.engine.backend,
                  "metrics_enabled": scene.engine._measurements is not None,
                  "expected_scene_reuse": scene.engine.expected_scene_reuse,
                  "thread_budget": scene.engine.num_threads, "memory_budget_bytes": scene.engine.memory_budget_bytes}
    if scene.engine.backend == "vulkan":
        descriptor["gpu"] = scene.engine._gpu_info
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:24]
    if len(scene._policy_keys) >= 64:
        scene._policy_keys.clear()
    scene._policy_keys[lookup] = digest, descriptor
    return digest, descriptor


def _remember(engine, key, record):
    if len(engine._policies) >= 128 and key not in engine._policies:
        engine._policies.pop(next(iter(engine._policies)))
    engine._policies[key] = record
    engine._policy_epoch += 1


def _valid_record(record, descriptor):
    try:
        chosen = record["selected"]
        allowed = ["reference", "embree"] if _core.embree_available() else ["reference"]
        if descriptor["backend_request"] != "auto":
            allowed = [descriptor["backend_request"]]
        return (record["workload"] == descriptor and chosen["backend"] in allowed
                and 1 <= chosen["num_threads"] <= descriptor["thread_budget"]
                and type(chosen["num_threads"]) is int
                and type(chosen["batch_size"]) is int and 0 < chosen["batch_size"] <= (
                    1_000_000 if chosen["backend"] == "vulkan" else 65536))
    except (KeyError, TypeError, ValueError):
        return False


def apply_policy(scene, n, operation, sources=0, model=None):
    default = {"backend": scene.engine.backend if scene.engine.backend != "auto" else (
        "embree" if _core.embree_available() else "reference"),
        "num_threads": scene.engine.num_threads, "batch_size": 1_000_000 if scene.engine.backend == "vulkan" else 65536}
    if not scene.engine._policies and not scene.engine._has_disk_cache:
        _select(scene, default)
        return
    key, descriptor = policy_key(scene, n, operation, sources, model)
    policies = scene.engine._policies
    if key not in policies:
        record = None
        if scene.engine.tuning_cache is not None:
            try:
                candidate = json.loads((scene.engine.tuning_cache / f"{key}.json").read_text())
                if _valid_record(candidate, descriptor):
                    record = candidate
            except (OSError, ValueError, TypeError):
                pass
        _remember(scene.engine, key, record)
    record = policies[key]
    if record is None:
        _select(scene, default)
        return
    _select(scene, record["selected"])


def _select(scene, policy):
    state = policy["backend"], policy["num_threads"], policy["batch_size"]
    if state != scene._configured:
        start = time.perf_counter()
        scene._native.configure(*state[:2])
        scene.build_seconds += time.perf_counter() - start
        scene._batch_size = state[2]
        scene._configured = state


def _mismatch_mask(expected, result):
    if isinstance(result, dict):
        mask = expected["hit"] != result["hit"]
        mask |= expected["geometry_id"] != result["geometry_id"]
        mask |= ~np.isclose(expected["t"], result["t"], rtol=2e-5, atol=2e-5)
        return mask
    if result.dtype == np.float32:
        return ~np.isclose(expected, result, rtol=2e-7, atol=1e-6)
    return expected != result


def _boundary_ambiguities(scene, operation, points, directions, drivers, expected, candidate, mismatch_mask, reserved=0):
    """Diagnose reference instability inside a float32-scale coordinate envelope.

    This never changes a label or hides the raw mismatch count. A disagreement
    is classified only if a bounded coordinate perturbation reproduces the
    candidate output. Reference instability alone cannot excuse unrelated errors.
    Return classified indices; cap diagnostic work for widespread errors.
    """
    indices = np.flatnonzero(mismatch_mask)[:4096]
    if not len(indices):
        return indices
    origins = points[indices]
    target = {k: v[indices] for k, v in expected.items()} if isinstance(expected, dict) else expected[indices]
    actual = {k: v[indices] for k, v in candidate.items()} if isinstance(candidate, dict) else candidate[indices]
    scale = np.maximum(np.max(np.abs(origins), axis=1), np.max(np.abs(scene.bounds)))
    if operation not in ("trace", "occluded"):
        for driver in drivers:
            if hasattr(driver, "position"):
                scale = np.maximum(scale, np.linalg.norm(origins.astype("f8")-driver.position, axis=1))
                scale = np.maximum(scale, np.max(np.abs(driver.position)))
    epsilon = 4*np.finfo("f4").eps*scale
    ambiguous = np.zeros(len(indices), dtype=bool)
    for axis in range(3):
        for sign in (-1, 1):
            perturbed = origins.copy()
            perturbed[:, axis] = (origins[:, axis].astype("f8") + sign*epsilon).astype("f4")
            if operation not in ("trace", "occluded") and any(hasattr(d, "travel_direction") for d in drivers):
                np.clip(perturbed, scene.bounds[0], scene.bounds[1], out=perturbed)
            if operation in ("trace", "occluded"):
                result = getattr(scene, operation)(perturbed, directions[indices], _reserved_bytes=reserved)
            else:
                result = scene.query(perturbed, drivers, model=operation, _reserved_bytes=reserved)
            ambiguous |= _mismatch_mask(target, result) & ~_mismatch_mask(actual, result)
    return indices[ambiguous]


def calibrate(scene, points, directions=None, *, operation="occluded", drivers=(), model="binary_any",
              expected_reuse=None, repeats=3, recalibrate=False):
    """Measure backend/threads and response chunk size on immutable inputs.

    Setup is amortized over expected_reuse; candidate outputs must match the
    reference. A 5% hysteresis retains the starting policy when effectively tied.
    Calibration is explicit, bounded to one million points and 1..7 repeats.
    """
    if operation not in ("trace", "occluded", "query"):
        raise ValueError("operation must be trace, occluded, or query")
    n = len(points)
    if not 0 < n <= 1_000_000:
        raise ValueError("calibration requires 1 to 1,000,000 representative points")
    repeats = integer(repeats, "repeats")
    if repeats > 7:
        raise ValueError("at most seven calibration repeats")
    expected_reuse = integer(scene.engine.expected_scene_reuse if expected_reuse is None else expected_reuse, "expected_reuse")
    scene.engine.expected_scene_reuse = expected_reuse
    drivers = tuple(drivers)
    if operation == "query" and model not in MODELS:
        raise ValueError("invalid response model")
    if operation != "query" and directions is None:
        raise ValueError("ray calibration requires directions")
    sources = len(drivers) if operation == "query" else 0
    key, descriptor = policy_key(scene, n, operation, sources, model if operation == "query" else None)
    if not recalibrate:
        apply_policy(scene, n, operation, sources, model if operation == "query" else None)
        cached = scene.engine._policies.get(key)
        if cached is not None and cached.get("expected_reuse") == expected_reuse:
            return cached
    scene.engine.check_memory(n * (24 if operation == "trace" else 16))
    scene._native.configure(scene.backend, scene._native.num_threads)
    original = scene.backend, scene._native.num_threads, scene._batch_size
    original_build = scene.build_seconds
    selected = None
    records = []
    scene._calibrating = True
    try:
        def run(reserved=0):
            if operation == "query":
                return scene.query(points, drivers, model=model, _reserved_bytes=reserved)
            return getattr(scene, operation)(points, directions, _reserved_bytes=reserved)
        scene._native.configure("reference", 1)
        baseline = run()
        baseline_bytes = sum(v.nbytes for v in baseline.values()) if isinstance(baseline, dict) else baseline.nbytes
        backends = ["reference", "embree"] if scene.engine.backend == "auto" and _core.embree_available() else [
            original[0] if scene.engine.backend != "auto" else "reference"]
        setup = {original[0]: original_build, "reference": 0.0}
        batch_candidates = (65536, 262144, 1_000_000) if scene.engine.backend == "vulkan" else (4096, 16384, 65536)
        batches = sorted({min(n, size) for size in batch_candidates}) if operation == "query" else [original[2]]
        plans = [original]
        for backend in backends:
            for batch in batches:
                maximum = min(scene.engine.num_threads, max(1, math.ceil((min(n, batch) if operation == "query" else n)/4096)))
                threads = sorted({1, *(2**i for i in range(11) if 2**i <= maximum), maximum})
                for count in threads:
                    plans.append((backend, count, batch))
        for backend, count, batch in dict.fromkeys(plans):
            start = time.perf_counter()
            scene._native.configure(backend, count)
            setup.setdefault(backend, time.perf_counter()-start)
            scene._batch_size = batch
            candidate = run(baseline_bytes)
            mismatch_mask = _mismatch_mask(baseline, candidate)
            mismatches = int(np.count_nonzero(mismatch_mask))
            ambiguous = 0
            if mismatches:
                scene._native.configure("reference", 1)
                try:
                    diagnostic_bytes = 2*baseline_bytes + mismatch_mask.nbytes + min(mismatches, 4096)*256 + 20*sources
                    scene._reserve(diagnostic_bytes)
                    ambiguous = len(_boundary_ambiguities(scene, model if operation == "query" else operation,
                                     points, directions, drivers, baseline, candidate, mismatch_mask,
                                     diagnostic_bytes-min(mismatches, 4096)*64))
                except MemoryError:
                    # An uncertifiable candidate is rejected, never excused by
                    # a broad tolerance or a reduced diagnostic distribution.
                    pass
                finally:
                    scene._native.configure(backend, count)
            valid = mismatches == ambiguous
            del candidate
            times = []
            if valid:
                for _ in range(repeats):
                    start = time.perf_counter(); run(baseline_bytes); times.append(time.perf_counter()-start)
            query_seconds = statistics.median(times) if valid else None
            record = {"backend": backend, "num_threads": count, "batch_size": batch,
                      "valid": valid, "query_ms": None if not valid else query_seconds*1000,
                      "raw_mismatches": mismatches, "boundary_ambiguities": ambiguous,
                      "setup_ms": setup[backend]*1000,
                      "score_ms": None if not valid else (query_seconds+setup[backend]/expected_reuse)*1000}
            records.append(record)
        valid = [r for r in records if r["valid"]]
        if not valid:
            raise RuntimeError("no calibration candidate matched the reference")
        best = min(valid, key=lambda r: r["score_ms"])
        starting = [r for r in valid if (r["backend"], r["num_threads"], r["batch_size"]) == original]
        selected = starting[0] if starting and starting[0]["score_ms"] <= best["score_ms"]*1.05 else best
        result = {"workload": descriptor, "expected_reuse": expected_reuse, "repeats": repeats,
                  "selected": selected, "candidates": records, "best_measured_score_ms": best["score_ms"],
                  "selected_to_best_ratio": selected["score_ms"]/best["score_ms"],
                  "default_measured_score_ms": starting[0]["score_ms"] if starting else None,
                  "selected_to_default_ratio": selected["score_ms"]/starting[0]["score_ms"] if starting else None,
                  "limitations": "Standalone generation measurements, including GPU transfers for an explicit Vulkan backend; no real training interference measured. CPU auto policies do not select a GPU."}
        _remember(scene.engine, key, result)
        if scene.engine.tuning_cache is not None:
            try:
                directory = scene.engine.tuning_cache
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as file:
                    json.dump(result, file, indent=2, allow_nan=False)
                    temporary = Path(file.name)
                temporary.replace(directory / f"{key}.json")
            except OSError:
                result["cache_warning"] = "Could not save disposable calibration cache; using in-memory policy."
        return result
    finally:
        policy = original if selected is None else (selected["backend"], selected["num_threads"], selected["batch_size"])
        try:
            scene._native.configure(*policy[:2])
            scene._batch_size = policy[2]
            scene._configured = policy
        finally:
            scene._calibrating = False
