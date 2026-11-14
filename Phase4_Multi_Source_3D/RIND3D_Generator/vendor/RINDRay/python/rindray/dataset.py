"""Versioned procedural descriptors: regenerate samples instead of dense fields."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .drivers import keyed_rng, sample_drivers
from .scene import Engine, integer, world_bounds
from .sampling import (parameters, scratch_bytes, driver_indices, free_points,
                       selected_points, intervention)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _count_range(value, name, minimum=0):
    if isinstance(value, (int, np.integer)):
        lo = hi = integer(value, name, minimum)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or inclusive (min,max)")
        lo, hi = (integer(v, name, minimum) for v in value)
    if hi < lo:
        raise ValueError(f"{name} max must be >= min")
    return lo, hi


def _safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("dataset paths must remain inside the dataset directory")
    return path


def scene_from_descriptor(descriptor, engine=None):
    engine = Engine() if engine is None else engine
    geometry = descriptor["geometry"]
    if geometry.get("generator") != "boxes_v1" or geometry.get("version") != 1:
        raise ValueError("unsupported geometry generator/version")
    bounds = np.asarray(world_bounds(descriptor["bounds"]), dtype=np.float64)
    params = geometry["params"]
    count = integer(params["count"], "box count", 0)
    size_range = np.asarray(params["size_range"], dtype=np.float64)
    if size_range.shape != (2,) or not np.isfinite(size_range).all() or not 0 < size_range[0] <= size_range[1] <= np.min(bounds[1]-bounds[0]):
        raise ValueError("size_range must be positive, ordered, and fit the scene bounds")
    rng = np.random.Generator(np.random.PCG64(integer(geometry["seed"], "geometry seed", 0)))
    sizes = rng.uniform(size_range[0], size_range[1], (count, 3))
    centers = rng.uniform(bounds[0]+sizes/2, bounds[1]-sizes/2)
    scene = engine.scene(bounds=bounds)
    for i, (center, size) in enumerate(zip(centers, sizes)):
        scene.add_box(center, size, geometry_id=i)
    return scene.commit()


@dataclass
class Sample:
    points: np.ndarray
    labels: object
    drivers: object
    metadata: dict


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != "rindray.dataset" or self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported dataset format/schema version")
        if self.manifest.get("coordinate_system") != "RH_Z_UP":
            raise ValueError("dataset coordinates must use RH_Z_UP; convert unsupported conventions explicitly")
        if not isinstance(self.manifest.get("units"), str) or not self.manifest["units"].strip():
            raise ValueError("dataset units must be a nonempty world-unit name; no implicit unit conversion is performed")
        generator = self.manifest.get("generator", {})
        if generator.get("name") != "boxes_v1" or generator.get("version") != 1:
            raise ValueError("unsupported dataset generator/version")
        if self.manifest.get("response", {}).get("version") != 2:
            raise ValueError("unsupported response semantics version")
        for relative in self.manifest["splits"].values():
            _safe_path(self.path, relative)

    @classmethod
    def create(cls, path, *, splits=None, master_seed=12345,
               bounds=((0, 0, 0), (20, 20, 20)), box_count=(16, 64), size_range=(.25, 2),
               driver_count=(1, 8), strength=None, query_count=4096, query_sampler="uniform_v1", query_params=None):
        splits = {"train": 8, "val": 2, "test": 2} if splits is None else dict(splits)
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("splits must specify train, val and test counts")
        splits = {name: integer(count, f"{name} count", 0) for name, count in splits.items()}
        master_seed = integer(master_seed, "master_seed", 0)
        bounds = world_bounds(bounds)
        box_range = _count_range(box_count, "box_count")
        driver_range = _count_range(driver_count, "driver_count")
        query_count = integer(query_count, "query_count", 0)
        query_params = parameters(query_sampler, query_params, bounds)
        if any(index >= driver_range[0] for index in driver_indices(query_sampler, query_params)):
            raise ValueError("query sampler driver_index must exist at the minimum driver_count")
        size_range = tuple(float(v) for v in size_range)
        if len(size_range) != 2 or not np.isfinite(size_range).all() or not 0 < size_range[0] <= size_range[1] <= min(np.subtract(bounds[1], bounds[0])):
            raise ValueError("size_range must be positive, ordered and fit within bounds")
        strength = {"distribution": "uniform", "min": .2, "max": 2} if strength is None else strength
        sample_drivers(scene_id="validation", count=0, strength=strength)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        (path / "scenes").mkdir()
        manifest = {
            "format": "rindray.dataset", "schema_version": 1,
            "coordinate_system": "RH_Z_UP", "units": "world",
            "generator": {"name": "boxes_v1", "version": 1, "master_seed": master_seed,
                          "rng": "blake2b-128+numpy.PCG64"},
            "response": {"supported_models": ["binary_any", "count_visible", "sum_weighted_visibility"],
                         "default_model": "binary_any", "version": 2},
            "splits": {split: f"scenes/{split}.jsonl" for split in ("train", "val", "test")},
        }
        for split in ("train", "val", "test"):
            with (path / manifest["splits"][split]).open("w") as file:
                for i in range(splits[split]):
                    scene_id = f"{split}_{i:08d}"
                    count = int(keyed_rng(master_seed, scene_id, stream="geometry_count").integers(box_range[0], box_range[1]+1))
                    seed = int(keyed_rng(master_seed, scene_id, stream="scene_seed").integers(0, 2**63))
                    descriptor = {
                        "id": scene_id, "bounds": bounds,
                        "geometry": {"generator": "boxes_v1", "version": 1, "seed": seed,
                                     "params": {"count": count, "size_range": size_range}},
                        "drivers": {"sampler": "point_v1", "version": 1, "master_seed": master_seed,
                                    "count_range": driver_range, "strength": strength},
                        "queries": {"sampler": query_sampler, "version": 1, "master_seed": master_seed, "count": query_count},
                    }
                    if query_params:
                        descriptor["queries"]["params"] = query_params
                    file.write(_json(descriptor) + "\n")
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return cls(path)

    def descriptors(self, split=None):
        names = (split,) if split is not None else tuple(self.manifest["splits"])
        for name in names:
            if name not in self.manifest["splits"]:
                raise ValueError("unknown dataset split")
            path = _safe_path(self.path, self.manifest["splits"][name])
            with path.open() as file:
                for number, line in enumerate(file, 1):
                    try:
                        descriptor = json.loads(line)
                        if not isinstance(descriptor["id"], str):
                            raise ValueError("scene id must be a string")
                        yield descriptor
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"invalid scene descriptor at {path.name}:{number}") from error

    def get(self, scene_id):
        for descriptor in self.descriptors():
            if descriptor["id"] == scene_id:
                return descriptor
        raise KeyError(scene_id)

    def scene(self, scene_id, engine=None):
        return scene_from_descriptor(self.get(scene_id), engine)

    def _descriptor(self, value):
        return value if isinstance(value, dict) else self.get(value)

    def drivers(self, scene, realization_id=0):
        descriptor = self._descriptor(scene)
        settings = descriptor["drivers"]
        count = self._driver_count(descriptor)
        return sample_drivers(scene_id=descriptor["id"], realization_id=realization_id,
                              master_seed=settings["master_seed"], count=count, bounds=descriptor["bounds"], strength=settings["strength"])

    def _driver_count(self, descriptor):
        settings = descriptor["drivers"]
        if settings.get("sampler") != "point_v1" or settings.get("version") != 1:
            raise ValueError("unsupported driver sampler/version")
        lo, hi = _count_range(settings["count_range"], "driver_count")
        return int(keyed_rng(settings["master_seed"], descriptor["id"], stream="driver_count").integers(lo, hi+1))

    def points(self, scene, query_id=0, *, count=None, start=0, stop=None, _scene=None, _drivers=None):
        descriptor = self._descriptor(scene)
        settings = descriptor["queries"]
        if settings.get("version") != 1:
            raise ValueError("unsupported query sampler/version")
        count = integer(settings["count"] if count is None else count, "count", 0)
        start = integer(start, "start", 0)
        stop = count if stop is None else integer(stop, "stop", 0)
        if not start <= stop <= count:
            raise ValueError("point range must lie inside the logical query count")
        bounds = np.asarray(world_bounds(descriptor["bounds"]), dtype=np.float64)
        params = parameters(settings["sampler"], settings.get("params"), bounds)
        if settings["sampler"] == "uniform_free_v1":
            actual_scene = scene_from_descriptor(descriptor) if _scene is None else _scene
            return free_points(actual_scene, settings, descriptor["id"], query_id, start, stop, bounds)
        if settings["sampler"] != "uniform_v1":
            actual_scene = self._sampling_scene(descriptor, _scene)
            drivers = self.drivers(descriptor) if _drivers is None else _drivers
            if any(index >= len(drivers) for index in driver_indices(settings["sampler"], params)):
                raise ValueError("query sampler driver_index does not exist in this scene")
            return selected_points(actual_scene, {**settings, "params": params}, descriptor["id"], query_id,
                                   start, stop, bounds, drivers)
        rng = keyed_rng(settings["master_seed"], descriptor["id"], query_id, stream="uniform_query")
        # Every float64 random coordinate consumes one PCG64 step. Advancing by
        # global point index makes chunk/worker scheduling irrelevant.
        rng.bit_generator.advance(3*start)
        return np.asarray(bounds[0] + rng.random((stop-start, 3))*(bounds[1]-bounds[0]), dtype=np.float32)

    def _sampling_scene(self, descriptor, scene):
        if descriptor["queries"]["sampler"] in ("uniform_v1", "uniform_free_v1"):
            return scene
        if scene is not None and scene.engine.backend == "reference":
            return scene
        engine = Engine("reference", tuning_cache=None) if scene is None else Engine(
            "reference", num_threads=scene.engine.num_threads,
            memory_budget_mb=scene.engine.memory_budget_bytes/1024**2, tuning_cache=None)
        return scene_from_descriptor(descriptor, engine)

    def blocks(self, scene_id, *, realization_id=0, query_id=0, count=None,
               chunk_size=65536, engine=None, model=None, output="aggregate"):
        descriptor = self.get(scene_id)
        engine = Engine() if engine is None else engine
        scene = scene_from_descriptor(descriptor, engine)
        drivers = self.drivers(descriptor, realization_id)
        settings = descriptor["queries"]
        count = integer(settings["count"] if count is None else count, "count", 0)
        chunk_size = integer(chunk_size, "chunk_size")
        model = self.manifest["response"]["default_model"] if model is None else model
        from .drivers import MODELS
        if model not in MODELS or output not in ("aggregate", "per_driver", "both"):
            raise ValueError("invalid response model or output selection")
        output_items = (1 if output != "per_driver" else 0) + (len(drivers) if output != "aggregate" else 0)
        # Includes float64 generation scratch and the float32 points/labels.
        point_scratch = scratch_bytes(settings["sampler"])
        engine.check_memory(min(chunk_size, count)*(point_scratch + np.dtype(MODELS[model]).itemsize*output_items))
        sampling_scene = self._sampling_scene(descriptor, scene)
        context = {}
        anchors = None
        if settings["sampler"] not in ("uniform_v1", "uniform_free_v1"):
            anchors = drivers if realization_id == 0 else self.drivers(descriptor)
            context = {"backend": "reference", "driver_realization": anchors.as_dict()}
            params = parameters(settings["sampler"], settings.get("params"), descriptor["bounds"])
            cf = params if settings["sampler"] == "counterfactual_v1" else (
                params.get("counterfactual_v1") if settings["sampler"] == "mixed_v1" and params["weights"][3] else None)
            if cf is not None:
                index = cf["driver_index"]
                context["intervention"] = {"driver_index": index,
                                           "position": list(intervention(anchors[index], np.asarray(descriptor["bounds"]), cf).position)}
        for start in range(0, count, chunk_size):
            stop = min(start+chunk_size, count)
            scene._reserve((stop-start)*(point_scratch + np.dtype(MODELS[model]).itemsize*output_items))
            points = self.points(descriptor, query_id, count=count, start=start, stop=stop, _scene=sampling_scene, _drivers=anchors)
            labels = scene.query(points, drivers, model=model, output=output, _reserved_bytes=points.nbytes)
            yield Sample(points, labels, drivers, {
                "scene_id": scene_id, "query_id": query_id, "query_sampler": settings["sampler"],
                "query_version": 1, "numpy_version": np.__version__, "scene_descriptor": descriptor,
                "start": start, "stop": stop, "count": count, "model": model,
                "response_version": 2, "driver_realization": drivers.as_dict(),
                "numerical_policy": "float32-eps4-v1", "label_backend": scene.backend,
                "coordinate_system": self.manifest["coordinate_system"], "units": self.manifest["units"],
                "query_settings": settings, "query_context": context,
            })
            del points, labels

    def sample(self, scene_id, *, realization_id=0, query_id=0, count=None,
               engine=None, model=None, output="aggregate"):
        descriptor = self.get(scene_id)
        count = integer(descriptor["queries"]["count"] if count is None else count, "count", 1)
        return next(self.blocks(scene_id, realization_id=realization_id, query_id=query_id,
                                count=count, chunk_size=count, engine=engine, model=model, output=output))

    def stream(self, scene_id, *, prefetch=0, prefetch_budget_mb=64, engine=None, **kwargs):
        """The same indexed blocks with optional one-thread bounded prefetch."""
        from .drivers import MODELS
        from .streaming import BatchStream
        engine = Engine(collect_metrics=True) if engine is None else engine
        descriptor = self.get(scene_id)
        requested_count = kwargs.get("count")
        count = integer(descriptor["queries"]["count"] if requested_count is None else requested_count, "count", 0)
        chunk = integer(kwargs.get("chunk_size", 65536), "chunk_size")
        sources = self._driver_count(descriptor)
        model = kwargs.get("model") or self.manifest["response"]["default_model"]
        output = kwargs.get("output", "aggregate")
        if model not in MODELS or output not in ("aggregate", "per_driver", "both"):
            raise ValueError("invalid response model or output selection")
        items = (1 if output != "per_driver" else 0)+(sources if output != "aggregate" else 0)
        max_bytes = min(count, chunk)*(12+np.dtype(MODELS[model]).itemsize*items)
        if isinstance(prefetch_budget_mb, bool) or not np.isfinite(prefetch_budget_mb) or prefetch_budget_mb <= 0:
            raise ValueError("prefetch_budget_mb must be finite and positive")
        return BatchStream(self.blocks(scene_id, engine=engine, **kwargs), integer(prefetch, "prefetch", 0), max_bytes,
                           int(prefetch_budget_mb*1024**2), engine.statistics)
