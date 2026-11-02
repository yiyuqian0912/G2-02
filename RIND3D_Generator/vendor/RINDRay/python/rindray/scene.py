"""Small ownership/validation layer over the synchronous native batch core."""

import math
from dataclasses import dataclass
import os
from pathlib import Path
import time
import threading

import numpy as np

from . import _core

_EMPTY_BOOL = np.empty(0, dtype=np.bool_)
_EMPTY_FLOAT = np.empty(0, dtype=np.float32)
_EMPTY_INT = np.empty(0, dtype=np.int32)
_FIELDS = {"hit": np.bool_, "t": np.float32, "geometry_id": np.int32, "normal": np.float32}


def array(value, name, dtype, shape=None, *, writable=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a NumPy {np.dtype(dtype)} array; convert explicitly")
    if shape is not None and (value.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(value.shape, shape)
    )):
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be C-contiguous and aligned; copy explicitly")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def vector(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite XYZ triple")
    return result


def world_bounds(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2, 3) or not np.isfinite(result).all() or not (result[1] > result[0]).all():
        raise ValueError("bounds must be finite ((xmin,ymin,zmin),(xmax,ymax,zmax)) with increasing limits")
    return tuple(tuple(float(x) for x in row) for row in result)


def disjoint(outputs, inputs):
    outputs = tuple(outputs)
    for i, out in enumerate(outputs):
        for other in (*outputs[:i], *inputs):
            if np.shares_memory(out, other):
                raise ValueError("output buffers must not overlap inputs or other outputs")


def usable_threads():
    """Respect affinity and finite cgroup CPU quotas without changing the host."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1].lstrip("/")
        root = Path("/sys/fs/cgroup")
        group = root / relative
        while group.is_relative_to(root):
            quota = group / "cpu.max"
            if quota.exists():
                limit, period = quota.read_text().split()
                if limit != "max":
                    available = min(available, max(1, math.floor(int(limit) / int(period))))
            if group == root:
                break
            group = group.parent
    except (OSError, ValueError, IndexError):
        pass
    return max(1, available)


@dataclass(frozen=True)
class CustomRays:
    """Caller-defined batches; downstream operators run once for the full batch."""
    origins: np.ndarray
    directions: np.ndarray
    tmin: object = 0.0
    tmax: object = np.inf

    def __post_init__(self):
        array(self.origins, "origins", np.float32, (None, 3))
        array(self.directions, "directions", np.float32, self.origins.shape)

    def trace(self, scene, **kwargs):
        return scene.trace(self.origins, self.directions, self.tmin, self.tmax, **kwargs)

    def occluded(self, scene, out=None):
        return scene.occluded(self.origins, self.directions, self.tmin, self.tmax, out=out)

    def evaluate(self, scene, operator, **kwargs):
        return operator(self.trace(scene, **kwargs))


class Engine:
    def __init__(self, backend="auto", num_threads=None, memory_budget_mb=1024, tuning_cache=".cache/rindray-tuning", expected_scene_reuse=16, collect_metrics=False):
        if backend not in ("auto", "reference", "embree", "vulkan"):
            raise ValueError("backend must be auto, reference, embree, or vulkan")
        if backend == "vulkan" and not _core.vulkan_compiled():
            raise RuntimeError("Vulkan was not compiled; rebuild with RINDRAY_VULKAN=ON or use a CPU backend")
        if backend == "embree" and not _core.embree_available():
            raise RuntimeError("Embree is not compiled; install Embree 4 and rebuild, or use reference")
        self.backend = backend
        self.available_threads = usable_threads()
        if num_threads is None:
            from multiprocessing import parent_process
            self.num_threads = 1 if parent_process() is not None else min(8, max(1, self.available_threads // 2))
        else:
            self.num_threads = integer(num_threads, "num_threads")
        if self.num_threads > 1024:
            raise ValueError("num_threads must not exceed 1024")
        if isinstance(memory_budget_mb, bool) or not isinstance(memory_budget_mb, (float, int)) or not math.isfinite(memory_budget_mb) or memory_budget_mb <= 0:
            raise ValueError("memory_budget_mb must be finite and positive")
        self.memory_budget_bytes = max(1, int(memory_budget_mb * 1024**2))
        self.expected_scene_reuse = integer(expected_scene_reuse, "expected_scene_reuse")
        self.tuning_cache = None if tuning_cache is None else Path(tuning_cache)
        self._has_disk_cache = self.tuning_cache is not None and self.tuning_cache.is_dir()
        self._policies = {}
        self._policy_epoch = 0
        self._gpu_device = None
        self._gpu_info = None
        self._gpu_lock = threading.Lock()
        self.gpu_initialization_seconds = 0.0
        if not isinstance(collect_metrics, bool):
            raise TypeError("collect_metrics must be bool")
        from .streaming import Measurements
        self._measurements = Measurements() if collect_metrics else None

    def statistics(self):
        """Opt-in query EWMA/peak estimates; process RSS is a separate high water."""
        return None if self._measurements is None else self._measurements.snapshot()

    def record_training_step(self, milliseconds):
        if self._measurements is None:
            raise RuntimeError("create the Engine with collect_metrics=True to record training steps")
        from .streaming import training_step
        training_step(self._measurements, milliseconds)

    def _get_gpu_device(self):
        with self._gpu_lock:
            if self._gpu_device is None:
                start = time.perf_counter()
                self._gpu_device = _core.create_vulkan_device()
                self.gpu_initialization_seconds = time.perf_counter() - start
                self._gpu_info = _core.vulkan_capabilities()
            return self._gpu_device

    def check_memory(self, size):
        if size > self.memory_budget_bytes:
            raise MemoryError(f"query needs at least {size} bytes; budget is {self.memory_budget_bytes}; use smaller batches or tiles")

    def scene(self, *, bounds=None):
        return Scene(self, bounds=bounds)


class Scene:
    def __init__(self, engine, *, bounds=None):
        self.engine = engine
        backend = engine.backend
        if backend == "auto":
            # Commit the portable representation first. A measured policy can
            # avoid an unnecessary Embree build for a small one-shot query.
            backend = "reference"
        device = engine._get_gpu_device() if backend == "vulkan" else None
        self._native = _core.Scene(backend, engine.num_threads, engine.memory_budget_bytes, device)
        self._bounds = None if bounds is None else world_bounds(bounds)
        self._inferred_bounds = None
        self._next_id = 0
        self.build_seconds = 0.0
        self._batch_size = 1_000_000 if backend == "vulkan" else 65536
        self._calibrating = False
        self._configured = None
        self._policy_keys = {}
        self._last_policy = None
        self._policy_lock = threading.RLock()
        self._last_gpu_metrics = None

    def _reserve(self, output_bytes, reserved=0):
        self.engine.check_memory(output_bytes + reserved)
        if self.engine._measurements is not None:
            self.engine._measurements.maximum("planned_buffer_high_water_bytes", output_bytes+reserved)
        remaining = self.engine.memory_budget_bytes - output_bytes - reserved
        if self.engine._gpu_device is not None:
            self._native.trim_gpu_workspace(remaining)
        return remaining

    def _record_gpu(self, start, totals=None, native_points=0, serial_empty=False):
        if self.backend == "vulkan":
            values = self._native.gpu_metrics() if totals is None else totals
            names = ("pack_ms", "transfer_ms", "kernel_ms", "unpack_ms", "native_ms", "workspace_bytes")
            self._last_gpu_metrics = {name: float(value) if math.isfinite(value) else None for name, value in zip(names, values)}
            self._last_gpu_metrics["wall_ms"] = (time.perf_counter() - start)*1000
        else:
            self._last_gpu_metrics = None
        if self.engine._measurements is not None:
            if serial_empty and native_points and self.backend == "vulkan" and not any(self._native.geometry_counts):
                native_points = min(native_points, 4096)
            metrics = self.engine._measurements
            metrics.observe("generation_ms", (time.perf_counter()-start)*1000)
            metrics.update(backend=self.backend, configured_native_threads=self._native.num_threads,
                           effective_native_threads=min(self._native.num_threads, math.ceil(native_points/4096)))
            if self._last_gpu_metrics is not None:
                metrics.update(gpu_transfer_ms=self._last_gpu_metrics["transfer_ms"])
                metrics.maximum("gpu_workspace_high_water_bytes", self._last_gpu_metrics["workspace_bytes"] or 0)

    def gpu_metrics(self):
        """Last trace/occluded/visibility/query timing, not a full render's sum."""
        return None if self._last_gpu_metrics is None else dict(self._last_gpu_metrics)

    @property
    def backend(self):
        return self._native.backend

    @property
    def bounds(self):
        if self._bounds is not None:
            return self._bounds
        if self._inferred_bounds is None:
            values = self._native.bounds()
            self._inferred_bounds = tuple(values[:3]), tuple(values[3:])
        return self._inferred_bounds

    def _id(self, value):
        result = self._next_id if value is None else integer(value, "geometry_id", 0)
        if result > np.iinfo(np.int32).max:
            raise ValueError("geometry_id exceeds int32 range")
        return result

    def _prepare_mutation(self):
        if self.engine.backend == "auto" and self.backend != "reference":
            self._native.configure("reference", self.engine.num_threads)
            self._configured = None
            self._last_policy = None

    def _geometry_changed(self):
        self.build_seconds = 0.0
        self._configured = None
        self._inferred_bounds = None
        self._policy_keys.clear()
        self._last_policy = None

    def add_box(self, center, size, geometry_id=None):
        geometry_id = self._id(geometry_id)
        self._prepare_mutation()
        self._native.add_box(vector(center, "center"), vector(size, "size"), geometry_id)
        self._next_id = max(self._next_id, geometry_id + 1)
        self._geometry_changed()
        return geometry_id

    def add_sphere(self, center, radius, geometry_id=None):
        geometry_id = self._id(geometry_id)
        self._prepare_mutation()
        self._native.add_sphere(vector(center, "center"), float(radius), geometry_id)
        self._next_id = max(self._next_id, geometry_id + 1)
        self._geometry_changed()
        return geometry_id

    def add_mesh(self, vertices, indices, geometry_id=None):
        vertices = array(vertices, "vertices", np.float32, (None, 3))
        indices = array(indices, "indices", np.uint32, (None, 3))
        geometry_id = self._id(geometry_id)
        self._prepare_mutation()
        self._native.add_mesh(vertices, indices, geometry_id)
        self._next_id = max(self._next_id, geometry_id + 1)
        self._geometry_changed()
        return geometry_id

    def add_instance(self, source_geometry_id, transform, geometry_id=None):
        source_geometry_id = integer(source_geometry_id, "source_geometry_id", 0)
        geometry_id = self._id(geometry_id)
        transform = np.asarray(transform, dtype=np.float32)
        if transform.shape == (4, 4):
            if not np.array_equal(transform[3], [0, 0, 0, 1]):
                raise ValueError("transform must be affine, with final row (0,0,0,1)")
            transform = transform[:3]
        if transform.shape != (3, 4) or not np.isfinite(transform).all():
            raise ValueError("transform must be a finite row-major 3x4 or affine 4x4 matrix")
        self._prepare_mutation()
        self._native.add_instance(source_geometry_id, tuple(float(v) for v in transform.ravel()), geometry_id)
        self._next_id = max(self._next_id, geometry_id+1)
        self._geometry_changed()
        return geometry_id

    def commit(self):
        start = time.perf_counter()
        self._native.commit()
        self.build_seconds += time.perf_counter() - start
        return self

    def _apply_policy(self, n, operation, sources=0, model=None):
        if n and self.engine.backend == "auto" and not self._calibrating:
            # configure releases the GIL. Keep native policy changes and their
            # Python cache state together when readers request different sizes.
            with self._policy_lock:
                base = (n, operation, sources, model, self.engine.num_threads, self.engine.memory_budget_bytes,
                        self.engine.expected_scene_reuse, self.engine._measurements is not None)
                if self._last_policy == (base, self.engine._policy_epoch):
                    return
                from .tuning import apply_policy
                apply_policy(self, n, operation, sources, model)
                self._last_policy = base, self.engine._policy_epoch

    def calibrate(self, points, directions=None, **kwargs):
        from .tuning import calibrate
        return calibrate(self, points, directions, **kwargs)

    def _rays(self, origins, directions, tmin, tmax):
        origins = array(origins, "origins", np.float32, (None, 3))
        directions = array(directions, "directions", np.float32, origins.shape)
        n = len(origins)
        def bound(value, name):
            if isinstance(value, np.ndarray):
                value = array(value, name, np.float32)
                if value.shape not in ((), (1,), (n,)):
                    raise ValueError(f"{name} must be scalar or shape ({n},)")
                return value.reshape(-1)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
                raise TypeError(f"{name} must be a real scalar or a NumPy float32 array")
            value = float(value)
            if math.isfinite(value) and abs(value) > float(np.finfo(np.float32).max):
                raise ValueError(f"{name} scalar exceeds float32 range")
            return np.asarray([value], dtype=np.float32)
        return origins, directions, bound(tmin, "tmin"), bound(tmax, "tmax")

    def trace(self, origins, directions, tmin=0.0, tmax=np.inf,
              fields=("hit", "t", "geometry_id"), out=None, _reserved_bytes=0):
        start = time.perf_counter()
        rays = self._rays(origins, directions, tmin, tmax)
        n = len(origins)
        self._apply_policy(n, "trace")
        fields = tuple(fields)
        if not fields or len(set(fields)) != len(fields) or any(field not in _FIELDS for field in fields):
            raise ValueError(f"fields must be unique names from {tuple(_FIELDS)}")
        output_bytes = sum(n * (3 if name == "normal" else 1) * np.dtype(_FIELDS[name]).itemsize for name in fields)
        scratch = self._reserve(output_bytes, _reserved_bytes)
        if out is None:
            out = {name: np.empty((n, 3) if name == "normal" else (n,), dtype=_FIELDS[name]) for name in fields}
        if not isinstance(out, dict) or set(out) != set(fields):
            raise ValueError("out must be a dict containing exactly the selected fields")
        for name in fields:
            array(out[name], f"out[{name}]", _FIELDS[name], (n, 3) if name == "normal" else (n,), writable=True)
        disjoint(out.values(), rays)
        self._native.trace_into(*rays, out.get("hit", _EMPTY_BOOL), out.get("t", _EMPTY_FLOAT),
                                out.get("geometry_id", _EMPTY_INT), out.get("normal", _EMPTY_FLOAT), scratch)
        self._record_gpu(start, native_points=n)
        return out

    def occluded(self, origins, directions, tmin=0.0, tmax=np.inf, out=None, _reserved_bytes=0):
        start = time.perf_counter()
        rays = self._rays(origins, directions, tmin, tmax)
        n = len(origins)
        self._apply_policy(n, "occluded")
        scratch = self._reserve(n, _reserved_bytes)
        if out is None:
            out = np.empty(n, dtype=np.bool_)
        array(out, "out", np.bool_, (n,), writable=True)
        disjoint((out,), rays)
        self._native.occluded_into(*rays, out, scratch)
        self._record_gpu(start, native_points=n, serial_empty=True)
        return out

    def visibility(self, sources, points, *, output="per_driver", out=None, epsilon_multiplier=4, _reserved_bytes=0):
        start = time.perf_counter()
        from .drivers import driver_arrays
        points = array(points, "points", np.float32, (None, 3))
        if isinstance(sources, np.ndarray):
            vectors = array(sources, "sources", np.float32, (None, 3))
            kinds = np.zeros(len(vectors), dtype=np.int32)
        else:
            vectors, kinds, _ = driver_arrays(sources)
        if output != "per_driver":
            raise ValueError("visibility always returns raw per-driver bool[S,N]")
        shape = len(vectors), len(points)
        self._apply_policy(math.prod(shape), "occluded")
        scratch = self._reserve(math.prod(shape), _reserved_bytes + 20*len(vectors))
        if out is None:
            out = np.empty(shape, dtype=np.bool_)
        array(out, "out", np.bool_, shape, writable=True)
        disjoint((out,), (points, vectors, kinds))
        self._native.visibility_into(vectors, kinds, points, out, (*self.bounds[0], *self.bounds[1]), float(epsilon_multiplier), scratch)
        self._record_gpu(start, native_points=(len(points) if self.backend == "vulkan" else math.prod(shape)) if len(vectors) else 0, serial_empty=True)
        return out

    def contains_points(self, points, out=None, _reserved_bytes=0):
        """Solid box/sphere occupancy, deliberately separate from visibility."""
        start = time.perf_counter()
        points = array(points, "points", np.float32, (None, 3))
        self._reserve(len(points), _reserved_bytes)
        if out is None:
            out = np.empty(len(points), dtype=np.bool_)
        array(out, "out", np.bool_, (len(points),), writable=True)
        disjoint((out,), (points,))
        self._native.contains_into(points, out)
        if self.engine._measurements is not None:
            self.engine._measurements.observe("generation_ms", (time.perf_counter()-start)*1000)
            self.engine._measurements.update(backend="reference", configured_native_threads=self._native.num_threads,
                                             effective_native_threads=min(self._native.num_threads, math.ceil(len(points)/4096)))
        return out

    def query(self, points, drivers=(), *, model="binary_any", output="aggregate",
              return_visibility=False, visibility=None, epsilon_multiplier=4, _reserved_bytes=0):
        query_start = time.perf_counter()
        from .drivers import (MODELS, accumulate, driver_arrays, response_buffers, response_result)
        points = array(points, "points", np.float32, (None, 3))
        vectors, kinds, strengths = driver_arrays(drivers)
        s, n = len(vectors), len(points)
        if visibility is None and s and n and (return_visibility or np.any(strengths > 0)):
            self._apply_policy(n, "query", s, model)
        if s > np.iinfo(np.int32).max:
            raise ValueError("driver count exceeds int32 response range")
        if model not in MODELS or output not in ("aggregate", "per_driver", "both"):
            raise ValueError("invalid response model or output selection")
        if visibility is not None:
            array(visibility, "visibility", np.bool_, (s, n))
        dtype = np.dtype(MODELS[model])
        output_bytes = dtype.itemsize * ((n if output != "per_driver" else 0) + (s*n if output != "aggregate" else 0))
        raw_bytes = s*n if return_visibility and visibility is None else 0
        step = self._batch_size
        base_bytes = output_bytes + raw_bytes + 20*s + _reserved_bytes
        native_needed = visibility is None and s and n and (return_visibility or np.any(strengths > 0))
        gpu_pp = 128 if native_needed and self.backend == "vulkan" else 0
        mask_pp = 1 if native_needed and not return_visibility else 0
        if gpu_pp + mask_pp:
            step = min(step, (self.engine.memory_budget_bytes - base_bytes) // (gpu_pp + mask_pp))
            if n and step <= 0:
                raise MemoryError("response buffers leave no memory for visibility workspace; use smaller outputs or a larger budget")
        mask_bytes = min(n, step)*mask_pp
        self.engine.check_memory(base_bytes + min(n, step)*(gpu_pp + mask_pp))
        scratch = self._reserve(output_bytes + raw_bytes + 20*s + mask_bytes, _reserved_bytes)
        boundary = (*self.bounds[0], *self.bounds[1])
        # Validate all points and the commit state even with no active drivers,
        # without allocating a Python [N,3] finite-mask temporary.
        self._native.validate_visibility(vectors, kinds, points, boundary, float(epsilon_multiplier))
        aggregate, per_driver = response_buffers(model, output, s, n)
        raw = visibility if visibility is not None else (np.empty((s, n), dtype=np.bool_) if return_visibility else None)
        totals = np.zeros(6, dtype=np.float64)
        if s == 0 or n == 0 or (not native_needed and visibility is None and not return_visibility):
            if per_driver is not None:
                per_driver.fill(0)
            self._record_gpu(query_start, totals)
            return response_result(aggregate, per_driver, raw if return_visibility else None, return_visibility)
        scratch_mask = np.empty(min(n, step), dtype=np.bool_) if raw is None else None
        for start in range(0, n, step):
            stop = min(start + step, n)
            target = None if aggregate is None else aggregate[start:stop]
            for i, strength in enumerate(strengths):
                contribution = None if per_driver is None else per_driver[i, start:stop]
                if visibility is None and raw is None and strength == 0:
                    if contribution is not None:
                        contribution.fill(0)
                    continue
                mask = raw[i, start:stop] if raw is not None else scratch_mask[:stop-start]
                if visibility is None:
                    self._native.visibility_into(vectors[i:i+1], kinds[i:i+1], points[start:stop], mask,
                                                 boundary, float(epsilon_multiplier), scratch)
                    if self.backend == "vulkan":
                        metrics = self._native.gpu_metrics()
                        totals[:5] += metrics[:5]
                        totals[5] = max(totals[5], metrics[5])
                accumulate(mask, strength, model, target, contribution)
        self._record_gpu(query_start, totals, min(n, step) if native_needed else 0, serial_empty=True)
        return response_result(aggregate, per_driver, raw if return_visibility else None, return_visibility)

    def render(self, camera, size, *, channels=("hit", "depth", "normal", "geometry_id"),
               drivers=(), model="binary_any", tmin=0.0, tmax=np.inf):
        from .grids import render
        return render(self, camera, size, channels, tuple(drivers), model, tmin, tmax)

    def slice(self, origin, axis_u, axis_v, size, extent, *, drivers=(), channel="response", model="binary_any"):
        from .grids import planar_slice
        return planar_slice(self, origin, axis_u, axis_v, size, extent, tuple(drivers), channel, model)

    def volume(self, bounds, shape, *, drivers=(), channels=("response",),
               chunk_shape=(32, 64, 64), lazy=True, model="binary_any"):
        from .grids import Volume
        volume = Volume(self, bounds, shape, tuple(drivers), channels, chunk_shape, model)
        return volume if lazy else volume.materialize()
