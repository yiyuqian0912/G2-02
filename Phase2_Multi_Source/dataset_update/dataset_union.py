"""Memory-mapped reader for old random-strength and new unit-strength RIND.

Virtual union keeps old scene IDs and files intact. Split assignment is made
per exact continuous obstacle geometry, so repeated geometries cannot leak
between train, validation, and test even when their intensities/views differ.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

COMMON_ARRAYS = (
    "drivers", "driver_counts", "driver_strengths", "obstacle_counts",
    "obstacle_types", "obstacle_params", "obstacle_group_ids", "obstacle_bits",
    "visibility_bits", "local_views", "view_counts",
)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (manifest.get("schema_version") != 3 or
            manifest.get("response_type") != "additive-driver-intensity" or
            manifest.get("cell_sampling") != "cell-center" or
            manifest.get("array_indexing") != "yx" or manifest.get("bit_order") != "little"):
        raise ValueError("Unsupported RIND physics/coordinate contract")
    mode = manifest.get("driver_strength_mode", "uniform-0-1")
    layout = manifest.get("candidate_layout", "padded-source-sets")
    derived_random = (manifest.get("dataset_layout") == "rind-strength-overlay-v1"
                      and mode == "uniform-0-1" and layout == "scene-source-sets"
                      and manifest.get("reference_policy") == "generating-configuration-only")
    if ((mode, layout) not in (("uniform-0-1", "padded-source-sets"),
                               ("fixed-one", "scene-source-sets")) and not derived_random):
        raise ValueError(f"Unsupported RIND strength/reference variant: {mode}/{layout}")
    for key in ("num_scenes", "global_size", "max_drivers", "view_capacity"):
        if type(manifest.get(key)) is not int or manifest[key] < 1:
            raise ValueError(f"Invalid manifest field: {key}")


class ComponentDataset:
    """Common semantic API; does not use references as teacher labels."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_manifest(self.manifest)
        self.num_scenes = int(self.manifest["num_scenes"])
        self.global_size = int(self.manifest["global_size"])
        self.scene_ids = np.arange(self.num_scenes, dtype=np.int64)
        self._base = self
        self.is_strength_overlay = self.manifest.get("dataset_layout") == "rind-strength-overlay-v1"
        self.base_array_root = self.root
        if self.is_strength_overlay:
            base_root = Path(self.manifest["base_root"])
            self.base_array_root = (base_root if base_root.is_absolute() else self.root / base_root).resolve()
            base_manifest = json.loads((self.base_array_root / "manifest.json").read_text())
            if (base_manifest.get("driver_strength_mode") != "fixed-one" or
                    int(base_manifest["num_scenes"]) != self.num_scenes or
                    hashlib.sha256((self.base_array_root / "manifest.json").read_bytes()).hexdigest()
                    != self.manifest.get("base_manifest_sha256")):
                raise ValueError("Derived strength overlay no longer matches its unit-source base")
        for name in COMMON_ARRAYS:
            array = np.load(self.array_root(name) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if len(array) != self.num_scenes:
                raise ValueError(f"Scene-axis mismatch: {name}")
            setattr(self, name, array)
        self._offsets = np.concatenate(([0], np.cumsum(self.view_counts, dtype=np.int64)))

    def array_root(self, name: str) -> Path:
        if self.is_strength_overlay and name != "driver_strengths":
            return self.base_array_root
        return self.root

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _check_scene(self, scene: int) -> int:
        if not isinstance(scene, (int, np.integer)) or not 0 <= scene < self.num_scenes:
            raise IndexError(f"Scene ID out of range: {scene}")
        return int(scene)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError("Observation index out of range")
        scene = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return self.get_observation(scene, int(index - self._offsets[scene]))

    def iter_scene_views(self, scene: int):
        scene = self._check_scene(scene)
        for view, row in enumerate(self.local_views[scene, :int(self.view_counts[scene])]):
            yield view, int(row[2])

    def iter_views(self, scene: int):
        return self.iter_scene_views(scene)

    def get_scene(self, scene: int) -> dict[str, Any]:
        scene = self._check_scene(scene)
        count, obstacles = int(self.driver_counts[scene]), int(self.obstacle_counts[scene])
        return {
            "scene_id": scene, "drivers": np.array(self.drivers[scene, :count], dtype=np.float64),
            "driver_strengths": np.array(self.driver_strengths[scene, :count], dtype=np.float64),
            "driver_count": count, "obstacle_count": obstacles,
            "obstacle_types": np.array(self.obstacle_types[scene, :obstacles], dtype=np.uint8),
            "obstacle_params": np.array(self.obstacle_params[scene, :obstacles], dtype=np.float64),
            "obstacle_group_ids": np.array(self.obstacle_group_ids[scene, :obstacles], dtype=np.uint16),
            "local_views": np.array(self.local_views[scene, :int(self.view_counts[scene])], dtype=np.uint16),
        }

    def get_source_params(self, scene: int) -> np.ndarray:
        value = self.get_scene(scene)
        return np.column_stack((value["drivers"], value["driver_strengths"]))

    @staticmethod
    def _crop_bits(bits: np.ndarray, world: int, x: int, y: int,
                   width: int, height: int) -> np.ndarray:
        flat = (np.arange(y, y + height)[:, None] * world
                + np.arange(x, x + width)[None, :]).reshape(-1)
        return ((bits[..., flat // 8] >> (flat % 8)) & 1).reshape(
            bits.shape[:-1] + (height, width)).astype(bool)

    def get_region(self, scene: int, x: int, y: int, width: int,
                   height: int | None = None) -> dict[str, np.ndarray]:
        scene = self._check_scene(scene)
        height = width if height is None else height
        if (not all(isinstance(item, (int, np.integer)) for item in (x, y, width, height)) or
                min(x, y) < 0 or min(width, height) < 1 or
                x + width > self.global_size or y + height > self.global_size):
            raise ValueError("Region must lie inside the world")
        count = int(self.driver_counts[scene])
        visible = self._crop_bits(self.visibility_bits[scene, :count], self.global_size,
                                  x, y, width, height)
        obstacle = self._crop_bits(self.obstacle_bits[scene], self.global_size,
                                   x, y, width, height)
        strengths = np.asarray(self.driver_strengths[scene, :count], dtype=np.float64)
        channels = visible.astype(np.float64) * strengths[:, None, None]
        return {"response": channels.sum(axis=0), "channels": channels, "obstacle": obstacle}

    def get_observation(self, scene: int, view: int) -> dict[str, Any]:
        scene = self._check_scene(scene)
        if not 0 <= view < int(self.view_counts[scene]):
            raise IndexError("View ID out of range")
        window = np.asarray(self.local_views[scene, view], dtype=np.int64)
        x, y, side = map(int, window)
        item = self.get_region(scene, x, y, side)
        return {"scene_id": scene, "view_id": int(view), "window": window,
                "response": item["response"].astype(np.float32), "obstacle": item["obstacle"]}

    def get_channels(self, scene: int, view: int) -> np.ndarray:
        observation = self.get_observation(scene, view)
        return self.get_region(scene, *map(int, observation["window"]))["channels"]

    def get_candidates(self, scene: int, view: int) -> list[np.ndarray]:
        self.get_observation(scene, view)
        if self.is_strength_overlay:
            # Unit-strength reference witnesses can cease to be equivalent
            # after reweighting. Expose only exact generating truth for legacy
            # evaluation callers; teacher proposals never use this interface.
            return [self.get_source_params(scene)]
        if self.manifest.get("candidate_layout") == "scene-source-sets":
            reference = np.load(self.root / "reference_sources.npy", mmap_mode="r")
            count = int(self.driver_counts[scene])
            return [np.column_stack((row[:count], np.ones(count))) for row in reference[scene]]
        values = np.load(self.root / "view_candidates.npy", mmap_mode="r")
        counts = np.load(self.root / "candidate_counts.npy", mmap_mode="r")
        numbers = np.load(self.root / "candidate_numbers.npy", mmap_mode="r")
        return [np.array(values[scene, view, item, :int(counts[scene, view, item])])
                for item in range(int(numbers[scene, view]))]

    def rerender(self, scene: int, window: np.ndarray, sources: np.ndarray, *,
                 backend="auto", return_channels=False):
        from rind_dataset.runtime import render_scene_region
        x, y, side = map(int, window)
        result = render_scene_region(self, scene, x, y, side, drivers=sources, backend=backend)
        return result if return_channels else result["response"]

    def geometry_group(self, scene: int) -> str:
        scene = self._check_scene(scene)
        count = int(self.obstacle_counts[scene])
        digest = hashlib.sha256(str(self.global_size).encode())
        pieces = []
        for primitive in range(count):
            params = np.asarray(self.obstacle_params[scene, primitive], dtype="<f8")
            params = np.where(params == 0.0, 0.0, params)
            pieces.append(bytes([int(self.obstacle_types[scene, primitive])]) + params.tobytes())
        for piece in sorted(pieces):
            digest.update(piece)
        return digest.hexdigest()

    def source_fingerprint(self, scene: int) -> str:
        """Content identity for cached teacher reuse, including views/strengths."""
        scene = self._check_scene(scene)
        count = int(self.driver_counts[scene])
        digest = hashlib.sha256(self.geometry_group(scene).encode())
        for value in (
            np.asarray(self.drivers[scene, :count], dtype="<f8"),
            np.asarray(self.driver_strengths[scene, :count], dtype="<f8"),
            np.asarray(self.local_views[scene, :int(self.view_counts[scene])], dtype="<u2"),
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def source_metadata(self, scene: int) -> dict[str, Any]:
        scene = self._check_scene(scene)
        return {
            "component_root": str(self.root), "local_scene_id": scene,
            "driver_strength_mode": self.manifest.get("driver_strength_mode", "uniform-0-1"),
            "view_sampling": self.manifest.get("view_sampling", "source-free-quadtree"),
            "strength_seed": self.manifest.get("strength_overlay_seed"),
            "reference_policy": self.manifest.get("reference_policy", "release-witnesses"),
        }


class _SceneArray:
    """Metadata row proxy for code using dataset._base.local_views/counts."""
    def __init__(self, owner: "UnionDataset", name: str):
        self.owner, self.name = owner, name

    def __getitem__(self, key):
        if isinstance(key, tuple):
            scene, suffix = key[0], key[1:]
        else:
            scene, suffix = key, ()
        if not isinstance(scene, (int, np.integer)):
            indices = np.arange(self.owner.num_scenes)[scene]
            return np.asarray([self[int(index)] for index in indices])
        component, local = self.owner.resolve(int(scene))
        row = getattr(component, self.name)[local]
        return row[suffix] if suffix else row


class UnionDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("dataset_layout") != "rind-virtual-union-v1":
            raise ValueError("Expected a RIND virtual union")
        self.components = []
        self.offsets = [0]
        for record in self.manifest["components"]:
            path = Path(record["root"])
            if not path.is_absolute():
                path = self.root / path
            component = ComponentDataset(path)
            if component.num_scenes != int(record["num_scenes"]):
                raise ValueError("Union component scene count changed")
            expected = record.get("manifest_sha256")
            actual = hashlib.sha256((component.root / "manifest.json").read_bytes()).hexdigest()
            if expected and expected != actual:
                raise ValueError("Union component manifest changed")
            self.components.append(component)
            self.offsets.append(self.offsets[-1] + component.num_scenes)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.num_scenes = int(self.offsets[-1])
        self.scene_ids = np.arange(self.num_scenes, dtype=np.int64)
        self.global_size = int(self.manifest["global_size"])
        self.scene_splits = json.loads((self.root / "scene_splits.json").read_text())["scene_splits"]
        self._base = self
        for name in COMMON_ARRAYS:
            setattr(self, name, _SceneArray(self, name))

    def resolve(self, scene: int) -> tuple[ComponentDataset, int]:
        if not 0 <= scene < self.num_scenes:
            raise IndexError("Union scene ID out of range")
        component = int(np.searchsorted(self.offsets, scene, side="right") - 1)
        return self.components[component], int(scene - self.offsets[component])

    def __len__(self) -> int:
        return sum(len(component) for component in self.components)

    def get_observation(self, scene: int, view: int) -> dict[str, Any]:
        component, local = self.resolve(scene)
        result = component.get_observation(local, view)
        result["scene_id"] = int(scene)
        return result

    def get_scene(self, scene: int) -> dict[str, Any]:
        component, local = self.resolve(scene)
        result = component.get_scene(local)
        result["scene_id"] = int(scene)
        return result

    def get_region(self, scene: int, *args, **kwargs):
        component, local = self.resolve(scene)
        return component.get_region(local, *args, **kwargs)

    def get_channels(self, scene: int, view: int):
        component, local = self.resolve(scene)
        return component.get_channels(local, view)

    def get_source_params(self, scene: int):
        component, local = self.resolve(scene)
        return component.get_source_params(local)

    def get_candidates(self, scene: int, view: int):
        component, local = self.resolve(scene)
        return component.get_candidates(local, view)

    def iter_scene_views(self, scene: int):
        component, local = self.resolve(scene)
        return component.iter_scene_views(local)

    iter_views = iter_scene_views

    def geometry_group(self, scene: int):
        component, local = self.resolve(scene)
        return component.geometry_group(local)

    def source_fingerprint(self, scene: int):
        component, local = self.resolve(scene)
        return component.source_fingerprint(local)

    def source_metadata(self, scene: int):
        component, local = self.resolve(scene)
        return component.source_metadata(local)

    def rerender(self, scene: int, *args, **kwargs):
        component, local = self.resolve(scene)
        return component.rerender(local, *args, **kwargs)


def open_dataset(root: str | Path) -> ComponentDataset | UnionDataset:
    path = Path(root).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("dataset_layout") == "rind-virtual-union-v1":
        return UnionDataset(path)
    return ComponentDataset(path)
