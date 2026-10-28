"""Low-file-count, resumable teacher storage for large experiments.

Each split/window-size group is one ``.rindshard`` directory containing a
small fixed set of memory-mapped NPY arrays.  Candidate coordinates are
deterministically reconstructed from the window and uniform-grid settings,
so they are not duplicated for every observation.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .search import CandidateDomain, uniform_grid


SCHEMA_VERSION = 1


def is_shard_reference(path):
    path = Path(path)
    return path.parent.name.endswith(".rindshard") and path.name.isdigit()


@lru_cache(maxsize=8192)
def _grid(window, world_size, spacing, quadtree_prior):
    grid = uniform_grid(
        CandidateDomain(tuple(window), int(world_size), bool(quadtree_prior)),
        float(spacing),
    )
    # Cached arrays are shared between records and must remain immutable.
    for value in (grid.candidate_xy, grid.grid_index, grid.x_axis, grid.y_axis):
        value.flags.writeable = False
    return grid


def _record_digest(record):
    digest = hashlib.sha256()
    for key in ("scene_id", "view_id", "window", "valid", "physical_cost",
                "teacher_prob", "temperature", "config_id"):
        value = np.asarray(record[key])
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class TeacherShard:
    """One fixed-capacity split/size store with crash-safe completion flags."""

    ARRAY_SPECS = {
        "scene_id": (np.int64, ()),
        "view_id": (np.int64, ()),
        "window": (np.int64, (3,)),
        "valid": (np.bool_, None),
        "physical_cost": (np.float32, None),
        "teacher_prob": (np.float32, None),
        "config_id": ("S64", ()),
        "run_id": ("S160", ()),
        "record_digest": ("S64", ()),
        "status": (np.uint8, ()),
    }

    def __init__(self, path, *, capacity=None, candidate_count=None, settings=None,
                 protocol_digest=None, mode="r+"):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text())
            expected = dict(schema_version=SCHEMA_VERSION,
                            capacity=int(capacity) if capacity is not None else self.manifest["capacity"],
                            candidate_count=(int(candidate_count) if candidate_count is not None
                                             else self.manifest["candidate_count"]),
                            protocol_digest=(protocol_digest if protocol_digest is not None
                                             else self.manifest["protocol_digest"]))
            for key, value in expected.items():
                if self.manifest.get(key) != value:
                    raise ValueError(f"teacher shard {key} mismatch: {self.path}")
        else:
            if capacity is None or candidate_count is None or settings is None or protocol_digest is None:
                raise FileNotFoundError(f"teacher shard does not exist: {self.path}")
            self.path.mkdir(parents=True, exist_ok=False)
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "capacity": int(capacity),
                "candidate_count": int(candidate_count),
                "protocol_digest": str(protocol_digest),
                "world_size": int(settings["world_size"]),
                "spacing": float(settings["spacing"]),
                "temperature": float(settings["temperature"]),
                "quadtree_prior": bool(settings.get("quadtree_prior", False)),
                "physics": settings.get("physics", {}),
                "storage": "npy_memmap_float32",
            }
            temp = manifest_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
            temp.replace(manifest_path)
        self.arrays = {}
        capacity = self.manifest["capacity"]
        candidates = self.manifest["candidate_count"]
        for name, (dtype, tail) in self.ARRAY_SPECS.items():
            shape = (capacity,) + ((candidates,) if tail is None else tail)
            array_path = self.path / f"{name}.npy"
            if array_path.is_file():
                array = np.load(array_path, mmap_mode=mode, allow_pickle=False)
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise ValueError(f"teacher shard array mismatch: {array_path}")
            else:
                if mode == "r":
                    raise FileNotFoundError(array_path)
                array = open_memmap(array_path, mode="w+", dtype=dtype, shape=shape)
                # Do not touch multi-GiB payload arrays during preallocation.
                # A zeroed status array is the sole authority for readable rows.
                if name == "status":
                    array[...] = 0
                    array.flush()
            self.arrays[name] = array
        self._pending = []

    def reference(self, slot):
        slot = int(slot)
        if not 0 <= slot < self.manifest["capacity"]:
            raise IndexError(slot)
        return self.path / f"{slot:08d}"

    def complete(self, slot):
        return int(self.arrays["status"][int(slot)]) == 1

    def digest(self, slot):
        return self.arrays["record_digest"][int(slot)].tobytes().rstrip(b"\0").decode()

    @property
    def pending_count(self):
        return len(self._pending)

    def write(self, slot, record):
        slot = int(slot)
        if self.complete(slot):
            raise ValueError(f"teacher shard slot already complete: {self.reference(slot)}")
        count = self.manifest["candidate_count"]
        if len(record["candidate_xy"]) != count:
            raise ValueError("candidate count changed within a teacher shard")
        if not np.isclose(float(record["temperature"]), self.manifest["temperature"], rtol=0, atol=1e-12):
            raise ValueError("teacher temperature changed within a shard")
        self.arrays["scene_id"][slot] = int(record["scene_id"])
        self.arrays["view_id"][slot] = int(record["view_id"])
        self.arrays["window"][slot] = np.asarray(record["window"], dtype=np.int64)
        self.arrays["valid"][slot] = np.asarray(record["valid"], dtype=bool)
        self.arrays["physical_cost"][slot] = np.asarray(record["physical_cost"], dtype=np.float32)
        self.arrays["teacher_prob"][slot] = np.asarray(record["teacher_prob"], dtype=np.float32)
        self.arrays["config_id"][slot] = str(record["config_id"]).encode()
        self.arrays["run_id"][slot] = str(record["run_id"]).encode()
        self.arrays["record_digest"][slot] = _record_digest(record).encode()
        self._pending.append(slot)

    def flush(self):
        if not self._pending:
            return
        # Persist payload first.  A slot becomes resumable only after all data
        # arrays have been flushed and its status byte is committed.
        for name, array in self.arrays.items():
            if name != "status":
                array.flush()
        self.arrays["status"][np.asarray(self._pending, dtype=np.int64)] = 1
        self.arrays["status"].flush()
        self._pending.clear()

    def load(self, slot):
        slot = int(slot)
        if not self.complete(slot):
            raise ValueError(f"teacher shard slot is incomplete: {self.reference(slot)}")
        window = tuple(int(x) for x in self.arrays["window"][slot])
        grid = _grid(window, self.manifest["world_size"], self.manifest["spacing"],
                     self.manifest["quadtree_prior"])
        if len(grid.candidate_xy) != self.manifest["candidate_count"]:
            raise ValueError("reconstructed candidate grid does not match shard")
        valid = np.asarray(self.arrays["valid"][slot]).copy()
        cost = np.asarray(self.arrays["physical_cost"][slot]).astype(np.float64)
        probability = np.asarray(self.arrays["teacher_prob"][slot]).astype(np.float64)
        config_id = self.arrays["config_id"][slot].tobytes().rstrip(b"\0").decode()
        run_id = self.arrays["run_id"][slot].tobytes().rstrip(b"\0").decode()
        metadata = {
            "record_schema_version": 1,
            "complete": True,
            "experiment_protocol": self.manifest["protocol_digest"],
            "storage": self.manifest["storage"],
            "settings": {
                "support": grid.metadata(),
                "physics": self.manifest.get("physics", {}),
                "temperature": self.manifest["temperature"],
                "search_mode": "uniform",
            },
        }
        return {
            "scene_id": int(self.arrays["scene_id"][slot]),
            "view_id": int(self.arrays["view_id"][slot]),
            "candidate_xy": grid.candidate_xy,
            "valid": valid,
            "evaluated": np.ones(len(valid), dtype=bool),
            "physical_cost": cost,
            "teacher_prob": probability,
            "temperature": float(self.manifest["temperature"]),
            "config_id": config_id,
            "run_id": run_id,
            "metadata": metadata,
            "window": np.asarray(window, dtype=np.int64),
            "grid_index": grid.grid_index,
            "x_axis": grid.x_axis,
            "y_axis": grid.y_axis,
            "search_level": np.zeros(len(valid), dtype=np.uint8),
        }


@lru_cache(maxsize=64)
def _readonly_shard(path):
    return TeacherShard(path, mode="r")


def load_shard_reference(path):
    path = Path(path)
    if not is_shard_reference(path):
        raise ValueError(f"not a teacher shard reference: {path}")
    return _readonly_shard(str(path.parent)).load(int(path.name))
