"""Fixed-workload benchmark; report setup, query, correctness and limitations."""

import math
import hashlib
import os
import platform
import statistics
import time

import numpy as np

from . import _core
from .drivers import sample_drivers
from .scene import Engine, integer, usable_threads
from .tuning import hardware, _mismatch_mask, _boundary_ambiguities


def make_scene(backend, threads, complexity, *, engine=None, include_spheres=True):
    engine = engine or Engine(backend, num_threads=threads, tuning_cache=None)
    scene = engine.scene(bounds=((-10, -10, -10), (10, 10, 10)))
    rng = np.random.Generator(np.random.PCG64(412))
    for i, center in enumerate(rng.uniform(-5, 5, (4 if complexity == "simple" else 64, 3))):
        scene.add_box(center, (.6, .9, 1.2), i)
    if complexity == "cluttered" and include_spheres:
        for i, center in enumerate(rng.uniform(-5, 5, (16, 3))):
            scene.add_sphere(center, .4, i+64)
    scene.add_mesh(np.array([[-8, -8, -6], [8, -8, -6], [0, 8, -6]], "f4"), np.array([[0, 1, 2]], "u4"), 1000)
    return scene.commit()


def _peak_rss_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2 if platform.system() == "Darwin" else 1024)
    except ImportError:
        return None


def run_benchmark(*, sizes=(256, 4096, 65536), thread_counts=None, repeats=3,
                  complexities=("simple", "cluttered"), coherences=("sparse", "coherent"),
                  consumer_sleep_ms=0, include_vulkan=False):
    sizes = tuple(integer(n, "ray count") for n in sizes)
    if not sizes or max(sizes) > 1_000_000:
        raise ValueError("benchmark sizes must contain 1..1,000,000 points")
    repeats = integer(repeats, "repeats")
    if repeats > 20:
        raise ValueError("benchmark repeats must not exceed 20")
    if not math.isfinite(consumer_sleep_ms) or not 0 <= consumer_sleep_ms <= 1000:
        raise ValueError("consumer_sleep_ms must be between 0 and 1000")
    if any(c not in ("simple", "cluttered") for c in complexities):
        raise ValueError("complexity must be simple or cluttered")
    if any(c not in ("sparse", "coherent") for c in coherences):
        raise ValueError("coherence must be sparse or coherent")
    if thread_counts is None:
        thread_counts = sorted({1, min(4, usable_threads()), min(8, usable_threads())})
    thread_counts = tuple(integer(t, "thread count") for t in thread_counts)
    available = ["reference"] + (["embree"] if _core.embree_available() else [])
    device = None
    if include_vulkan:
        device = _core.vulkan_capabilities()
        if not device["available"]:
            raise RuntimeError(f"Vulkan benchmark unavailable: {device['error']}")
        available.append("vulkan")
    report = {
        "format": "rindray.benchmark", "schema_version": 1,
        "machine": {**hardware(usable_threads()), "gpu": device}, "repeats": repeats,
        "scene_family": "boxes_mesh" if include_vulkan else "mixed_boxes_spheres_mesh",
        "limitations": [
            "Standalone generation measurements; no real training interference measured.",
            "Cold total includes fresh scene construction and first query, excluding process startup/import. An Engine/device is reused; device_init_ms identifies its first initialization.",
            "Peak RSS is the process high-water mark, not a per-candidate allocation delta.",
            "Raw mismatch counts are preserved. Boundary ambiguities require reproducing the actual candidate output with a four-float32-epsilon reference-coordinate perturbation; at most 4096 disagreements per observation are classified.",
            ("Vulkan feature status is probed; all candidates use the same sphere-free geometry. Workspace bytes include staging and device buffers; full process GPU peak is not measured."
             if include_vulkan else "GPU/VRAM not probed; CPU transfer_ms is zero; no response cache is used."),
        ], "records": [],
    }
    try:
        report["machine"] = {**report["machine"], "ram_bytes": os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")}
    except (AttributeError, ValueError, OSError):
        report["machine"] = dict(report["machine"])
    operations = ("trace", "occluded", "binary_any", "sum_weighted_visibility")
    engines = {}
    for complexity in complexities:
        oracle = make_scene("reference", 1, complexity, include_spheres=not include_vulkan)
        for coherence in coherences:
            rng = np.random.Generator(np.random.PCG64(1024))
            origins = rng.uniform(-8, 8, (max(sizes), 3)).astype("f4")
            directions = rng.normal(size=origins.shape).astype("f4")
            if coherence == "coherent":
                origins[:, 0] = -8
                origins[:, 1:] *= .1
                directions[:, 0] = 1
                directions[:, 1:] *= .02
            drivers = sample_drivers(scene_id="benchmark_fixed", master_seed=817, count=4,
                                      bounds=((-8, -8, -8), (8, 8, 8)),
                                      strength={"distribution": "uniform", "min": .2, "max": 2})
            for n in sizes:
                o, d = origins[:n], directions[:n]
                def execute(scene, operation):
                    if operation in ("trace", "occluded"):
                        return getattr(scene, operation)(o, d)
                    return scene.query(o, drivers, model=operation)
                expected = {op: execute(oracle, op) for op in operations}
                for backend in available:
                    for threads in thread_counts:
                        for operation in operations:
                            key = backend, threads
                            if key not in engines:
                                engines[key] = Engine(backend, num_threads=threads, tuning_cache=None)
                            engine = engines[key]
                            initialized = engine._gpu_device is not None
                            start = time.perf_counter()
                            scene = make_scene(backend, threads, complexity, engine=engine, include_spheres=not include_vulkan)
                            setup_ms = (time.perf_counter()-start)*1000
                            start = time.perf_counter()
                            first = execute(scene, operation)
                            cold_ms = (time.perf_counter()-start)*1000
                            mismatch_mask = _mismatch_mask(expected[operation], first)
                            unexpected = mismatch_mask.copy()
                            unexpected[_boundary_ambiguities(oracle, operation, o, d, drivers, expected[operation], first, mismatch_mask)] = False
                            del first
                            times = []
                            gpu_times = []
                            for _ in range(repeats):
                                start = time.perf_counter(); value = execute(scene, operation)
                                times.append((time.perf_counter()-start)*1000)
                                metrics = scene.gpu_metrics()
                                if metrics is not None:
                                    gpu_times.append(metrics)
                                bad = _mismatch_mask(expected[operation], value)
                                unknown = bad.copy()
                                unknown[_boundary_ambiguities(oracle, operation, o, d, drivers, expected[operation], value, bad)] = False
                                mismatch_mask |= bad
                                unexpected |= unknown
                            ray_ms = statistics.median(times)
                            mismatches = int(np.count_nonzero(mismatch_mask))
                            ambiguous = mismatches-int(np.count_nonzero(unexpected))
                            start = time.perf_counter(); execute(scene, operation)
                            if consumer_sleep_ms:
                                time.sleep(consumer_sleep_ms/1000)
                            consumer_ms = (time.perf_counter()-start)*1000
                            sources = 0 if operation in ("trace", "occluded") else len(drivers)
                            rays = n*(sources or 1)
                            gpu = {name: statistics.median(r[name] for r in gpu_times) if all(r[name] is not None for r in gpu_times) else None
                                   for name in gpu_times[0]} if gpu_times else {}
                            report["records"].append({
                                "backend": backend, "num_threads": threads, "complexity": complexity,
                                "coherence": coherence, "operation": operation, "points": n,
                                "driver_count": sources, "ray_count": rays,
                                "geometry_counts": list(scene._native.geometry_counts),
                                "scene_build_ms": scene.build_seconds*1000, "setup_ms": setup_ms,
                                "cold_query_ms": cold_ms, "ray_ms": ray_ms, "transfer_ms": gpu.get("transfer_ms", 0.0),
                                "device_init_ms": 0 if initialized else engine.gpu_initialization_seconds*1000,
                                "gpu_metrics": gpu or None,
                                "total_ms": setup_ms+cold_ms, "rays_per_second": rays/(ray_ms/1000),
                                "peak_rss_mb": _peak_rss_mb(), "peak_gpu_mb": None,
                                "mismatches": mismatches, "boundary_ambiguities": ambiguous,
                                "unexpected_mismatches": mismatches-ambiguous, "consumer_total_ms": consumer_ms,
                                "consumer_sleep_ms": consumer_sleep_ms,
                            })
                            del scene
    return report


def table(report):
    lines = ["backend threads scene rays operation points setup_ms ray_ms Mrays/s mismatches ambiguous unexpected"]
    for r in report["records"]:
        lines.append(f"{r['backend']:9} {r['num_threads']:2} {r['complexity']:9} {r['coherence']:8} "
                     f"{r['operation']:24} {r['points']:7} {r['setup_ms']:8.3f} {r['ray_ms']:8.3f} "
                     f"{r['rays_per_second']/1e6:8.3f} {r['mismatches']} {r['boundary_ambiguities']} {r['unexpected_mismatches']}")
    return "\n".join(lines)


def benchmark_stream(dataset, scene_id, *, count=None, chunk_size=4096, depths=(0, 1, 2),
                     repeats=3, consumer=None, consumer_sleep_ms=0, engine=None,
                     model="sum_weighted_visibility", prefetch_budget_mb=64):
    """Measure whole-stream time with an identical indexed workload and consumer.

    The returned depth is advisory. No background adaptation, callback registry,
    or persisted policy for an unidentified training workload is introduced.
    """
    depths = tuple(integer(d, "prefetch depth", 0) for d in depths)
    if not depths or len(depths) > 4 or len(set(depths)) != len(depths) or 0 not in depths or max(depths) > 8:
        raise ValueError("depths must contain zero and at most four distinct depths in 0..8")
    repeats = integer(repeats, "repeats")
    if repeats > 7:
        raise ValueError("stream profiling allows at most seven repeats")
    if not math.isfinite(consumer_sleep_ms) or not 0 <= consumer_sleep_ms <= 1000:
        raise ValueError("consumer_sleep_ms must be between 0 and 1000")
    count = integer(dataset.get(scene_id)["queries"]["count"] if count is None else count, "count")
    if count > 1_000_000:
        raise ValueError("stream profiling is bounded to one million points")
    chunk_size = integer(chunk_size, "chunk_size")
    engine = Engine(collect_metrics=True, tuning_cache=None) if engine is None else engine
    records, expected_digest = [], None
    for depth in depths:
        times, statistics_values = [], []
        for _ in range(repeats):
            digest = hashlib.blake2b(digest_size=16)
            start = time.perf_counter()
            with dataset.stream(scene_id, count=count, chunk_size=chunk_size, prefetch=depth, engine=engine,
                                model=model, prefetch_budget_mb=prefetch_budget_mb) as stream:
                for sample in stream:
                    # Hash before a consumer may modify its own returned batch.
                    digest.update(sample.points)
                    digest.update(sample.labels)
                    step_start = time.perf_counter()
                    if consumer is not None:
                        consumer(sample)
                    if consumer_sleep_ms:
                        time.sleep(consumer_sleep_ms/1000)
                    stream.record_training_step((time.perf_counter()-step_start)*1000)
            times.append((time.perf_counter()-start)*1000)
            statistics_values.append(stream.statistics())
            actual = digest.hexdigest()
            if expected_digest is None:
                expected_digest = actual
            if actual != expected_digest:
                raise RuntimeError("prefetch candidates changed query coordinates or labels")
        records.append({"prefetch": depth, "total_ms": statistics.median(times),
                        "first_ms": times[0], "trials_ms": times, "statistics": statistics_values[-1]})
    best = min(records, key=lambda r: r["total_ms"])
    default = next(r for r in records if r["prefetch"] == 0)
    selected = default if default["total_ms"] <= best["total_ms"]*1.05 else best
    return {"format": "rindray.stream_benchmark", "schema_version": 1,
            "machine": hardware(engine.available_threads), "scene_id": scene_id, "points": count,
            "chunk_size": chunk_size, "model": model, "consumer_sleep_ms": consumer_sleep_ms,
            "consumer": "caller callback" if consumer is not None else "synthetic sleep",
            "payload_digest": expected_digest, "records": records,
            "selected_prefetch": selected["prefetch"], "selected_to_best_ratio": selected["total_ms"]/best["total_ms"],
            "selected_to_synchronous_ratio": selected["total_ms"]/default["total_ms"],
            "limitations": ["Single producer; query working memory and retained prefetch payloads have separate budgets.",
                            "Process RSS is a high-water mark; consumer-retained memory is outside the stream budget.",
                            "Caller GPU training callbacks must synchronize their own work for accurate timing.",
                            "No real JEPA training is implied by the synthetic consumer."]}
