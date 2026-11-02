"""Driver realizations and source-order response composition, without ray callbacks."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import numpy as np

from .scene import array, integer, vector, world_bounds


def strength_value(value):
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > float(np.finfo(np.float32).max):
        raise ValueError("strength must be finite, nonnegative and representable as float32")
    with np.errstate(under="ignore"):
        rounded = np.float32(value)
    if value > 0 and rounded == 0:
        raise ValueError("positive strength underflows float32; increase it or use explicit zero")
    return float(rounded)


@dataclass(frozen=True)
class PointDriver:
    position: tuple
    strength: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(float(x) for x in vector(self.position, "position")))
        object.__setattr__(self, "strength", strength_value(self.strength))


@dataclass(frozen=True)
class DirectionalDriver:
    travel_direction: tuple
    strength: float = 1.0

    def __post_init__(self):
        direction = vector(self.travel_direction, "travel_direction")
        if not np.any(direction):
            raise ValueError("travel_direction must be nonzero")
        object.__setattr__(self, "travel_direction", tuple(float(x) for x in direction))
        object.__setattr__(self, "strength", strength_value(self.strength))


@dataclass(frozen=True)
class DriverRealization(Sequence):
    drivers: tuple
    master_seed: int
    scene_id: str | int
    realization_id: int
    geometry_realization_id: int = 0

    def __len__(self):
        return len(self.drivers)

    def __getitem__(self, index):
        return self.drivers[index]

    @property
    def strengths(self):
        return np.asarray([driver.strength for driver in self], dtype=np.float32)

    def as_dict(self):
        return {
            "sampler_version": 1, "rng": "blake2b-128+numpy.PCG64",
            "master_seed": self.master_seed, "scene_id": self.scene_id,
            "realization_id": self.realization_id,
            "geometry_realization_id": self.geometry_realization_id,
            "drivers": [
                {"kind": "point", "position": list(d.position), "strength": d.strength}
                if isinstance(d, PointDriver) else
                {"kind": "directional", "travel_direction": list(d.travel_direction), "strength": d.strength}
                for d in self
            ],
        }

    @classmethod
    def from_dict(cls, value):
        """Restore saved realized parameters verbatim, without random sampling."""
        if value.get("sampler_version") != 1 or value.get("rng") != "blake2b-128+numpy.PCG64":
            raise ValueError("unsupported driver realization version")
        master_seed = integer(value["master_seed"], "master_seed", 0)
        realization_id = integer(value["realization_id"], "realization_id", 0)
        geometry_id = integer(value.get("geometry_realization_id", 0), "geometry_realization_id", 0)
        scene_id = value["scene_id"]
        if isinstance(scene_id, bool) or not isinstance(scene_id, (str, int)):
            raise TypeError("scene_id must be a string or integer")
        drivers = []
        for driver in value["drivers"]:
            if driver["kind"] == "point":
                drivers.append(PointDriver(driver["position"], driver["strength"]))
            elif driver["kind"] == "directional":
                drivers.append(DirectionalDriver(driver["travel_direction"], driver["strength"]))
            else:
                raise ValueError("unsupported driver kind in realization")
        return cls(tuple(drivers), master_seed, scene_id, realization_id, geometry_id)


def keyed_rng(master_seed, scene_id, realization_id=0, index=0, stream="query"):
    """v1: canonical JSON identifiers -> BLAKE2b-128 little endian -> explicit PCG64."""
    integer(master_seed, "master_seed", 0)
    integer(realization_id, "realization_id", 0)
    integer(index, "index", 0)
    if isinstance(scene_id, bool) or not isinstance(scene_id, (str, int)):
        raise TypeError("scene_id must be a string or integer")
    payload = json.dumps([1, int(master_seed), scene_id, int(realization_id), int(index), stream],
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=16, person=b"rindray.rng.v1").digest(), "little")
    return np.random.Generator(np.random.PCG64(seed))


def sample_drivers(*, scene_id, realization_id=0, master_seed=0, kind="point", count=1,
                   bounds=((0, 0, 0), (1, 1, 1)), strength=1.0, geometry_realization_id=0):
    count = integer(count, "count", 0)
    realization_id = integer(realization_id, "realization_id", 0)
    geometry_realization_id = integer(geometry_realization_id, "geometry_realization_id", 0)
    integer(master_seed, "master_seed", 0)
    if kind not in ("point", "directional"):
        raise ValueError("kind must be point or directional")
    bounds = np.asarray(world_bounds(bounds), dtype=np.float64)
    if isinstance(strength, dict):
        distribution = strength.get("distribution")
        if distribution == "fixed":
            low = high = strength_value(strength["value"])
        elif distribution == "uniform":
            low, high = strength_value(strength["min"]), strength_value(strength["max"])
            if high < low:
                raise ValueError("uniform strength max must be >= min")
        else:
            raise ValueError("strength distribution must be fixed or uniform")
    else:
        low = high = strength_value(strength)
    # Validate identifier types even for an empty realization.
    keyed_rng(master_seed, scene_id, realization_id, stream="strength")
    drivers = []
    for i in range(count):
        geometry_rng = keyed_rng(master_seed, scene_id, geometry_realization_id, i, "driver_geometry")
        with np.errstate(under="ignore"):
            a = low if low == high else float(np.float32(keyed_rng(master_seed, scene_id, realization_id, i, "strength").uniform(low, high)))
        if kind == "point":
            drivers.append(PointDriver(tuple(geometry_rng.uniform(bounds[0], bounds[1])), a))
        else:
            direction = geometry_rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            drivers.append(DirectionalDriver(tuple(direction), a))
    return DriverRealization(tuple(drivers), int(master_seed), scene_id, realization_id, geometry_realization_id)


def driver_arrays(drivers):
    drivers = tuple(drivers)
    if any(not isinstance(d, (PointDriver, DirectionalDriver)) for d in drivers):
        raise TypeError("drivers must be PointDriver or DirectionalDriver instances")
    vectors = np.asarray([d.position if isinstance(d, PointDriver) else d.travel_direction for d in drivers],
                         dtype=np.float32).reshape(-1, 3)
    kinds = np.asarray([isinstance(d, DirectionalDriver) for d in drivers], dtype=np.int32)
    strengths = np.asarray([d.strength for d in drivers], dtype=np.float32)
    return vectors, kinds, strengths


class QueryResult(NamedTuple):
    aggregate: np.ndarray | None
    per_driver: np.ndarray | None
    visibility: np.ndarray | None = None


MODELS = {"binary_any": np.bool_, "count_visible": np.int32, "sum_weighted_visibility": np.float32}


def response_buffers(model, output, s, n):
    if model not in MODELS:
        raise ValueError(f"model must be one of {tuple(MODELS)}")
    if output not in ("aggregate", "per_driver", "both"):
        raise ValueError("output must be aggregate, per_driver, or both")
    dtype = MODELS[model]
    aggregate = np.zeros(n, dtype=dtype) if output != "per_driver" else None
    per_driver = np.empty((s, n), dtype=dtype) if output != "aggregate" else None
    return aggregate, per_driver


def accumulate(mask, strength, model, aggregate, contribution=None):
    """Source-order float32 addition; no normalization, saturation or attenuation."""
    active = strength > 0
    if contribution is not None:
        if not active:
            contribution.fill(0)
        elif model == "sum_weighted_visibility":
            np.multiply(mask, np.float32(strength), out=contribution)
        else:
            contribution[:] = mask
    if aggregate is not None and active:
        with np.errstate(over="raise", invalid="raise"):
            if model == "binary_any":
                np.logical_or(aggregate, mask, out=aggregate)
            else:
                np.add(aggregate, np.float32(strength) if model == "sum_weighted_visibility" else 1,
                       out=aggregate, where=mask)


def response_result(aggregate, per_driver, raw=None, return_visibility=False):
    if return_visibility or (aggregate is not None and per_driver is not None):
        return QueryResult(aggregate, per_driver, raw)
    return aggregate if aggregate is not None else per_driver


def compose(visibility, strengths, *, model="sum_weighted_visibility", output="aggregate"):
    """Reuse a raw visibility matrix for another strength realization, without tracing."""
    visibility = array(visibility, "visibility", np.bool_, (None, None))
    strengths = np.asarray(strengths, dtype=np.float64)
    if strengths.shape != (visibility.shape[0],):
        raise ValueError("strengths must have shape (S,)")
    strengths = np.asarray([strength_value(value) for value in strengths], dtype=np.float32)
    aggregate, contributions = response_buffers(model, output, *visibility.shape)
    for i, strength in enumerate(strengths):
        accumulate(visibility[i], strength, model, aggregate, None if contributions is None else contributions[i])
    return response_result(aggregate, contributions)
