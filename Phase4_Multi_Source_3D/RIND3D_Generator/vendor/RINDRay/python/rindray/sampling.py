"""Versioned, indexed receiver sampling; all label-based selection uses reference rays."""

import numpy as np

from .drivers import PointDriver, keyed_rng
from .scene import integer, vector

SAMPLERS = ("uniform_v1", "uniform_free_v1", "occlusion_boundary_v1",
            "long_range_v1", "counterfactual_v1", "mixed_v1")
_COMPONENTS = ("uniform_free_v1", "occlusion_boundary_v1", "long_range_v1", "counterfactual_v1")


def parameters(name, value, bounds):
    if name not in SAMPLERS:
        raise ValueError(f"query_sampler must be one of {SAMPLERS}")
    if value is not None and not isinstance(value, dict):
        raise TypeError("query_params must be an ordinary parameter dict")
    result = dict({} if value is None else value)
    allowed = {
        "uniform_v1": (), "uniform_free_v1": (),
        "occlusion_boundary_v1": ("driver_index", "max_attempts", "refinements"),
        "long_range_v1": ("driver_index", "max_attempts", "min_distance"),
        "counterfactual_v1": ("driver_index", "max_attempts", "intervention_position"),
        "mixed_v1": ("weights", "occlusion_boundary_v1", "long_range_v1", "counterfactual_v1"),
    }[name]
    if set(result)-set(allowed):
        raise ValueError(f"unknown parameters for {name}: {sorted(set(result)-set(allowed))}")
    if name in SAMPLERS[2:5]:
        result["driver_index"] = integer(result.get("driver_index", 0), "driver_index", 0)
        result["max_attempts"] = integer(result.get("max_attempts", 256), "max_attempts")
        if result["max_attempts"] > 4096:
            raise ValueError("max_attempts must not exceed 4096")
    if name == "occlusion_boundary_v1":
        result["refinements"] = integer(result.get("refinements", 10), "refinements")
        if result["refinements"] > 20:
            raise ValueError("refinements must not exceed 20")
    if name == "long_range_v1":
        distance = float(result.get("min_distance", .5*np.linalg.norm(np.subtract(bounds[1], bounds[0]))))
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("min_distance must be finite and nonnegative")
        result["min_distance"] = distance
    if name == "counterfactual_v1" and "intervention_position" in result:
        result["intervention_position"] = vector(result["intervention_position"], "intervention_position").tolist()
    if name == "mixed_v1":
        weights = np.asarray(result.get("weights", [.5, .2, .2, .1]), dtype=np.float64)
        if weights.shape != (4,) or not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
            raise ValueError("mixed weights must be four finite nonnegative values with at least one positive value")
        result["weights"] = weights.tolist()
        for component, weight in zip(_COMPONENTS[1:], weights[1:]):
            if weight:
                result[component] = parameters(component, result.get(component), bounds)
            elif component in result:
                result[component] = parameters(component, result[component], bounds)
    return result


def scratch_bytes(name):
    return 64 if name == "uniform_v1" else 256 if name == "uniform_free_v1" else 1024


def driver_indices(name, params):
    if name == "mixed_v1":
        return [params[c]["driver_index"] for c, w in zip(_COMPONENTS[1:], params["weights"][1:]) if w]
    return [params["driver_index"]] if name in SAMPLERS[2:5] else []


def _splitmix64(values):
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def free_points(scene, settings, scene_id, query_id, start, stop, bounds):
    """Original uniform_free_v1 stream; keep these counters unchanged."""
    if stop > np.iinfo(np.uint64).max // 3:
        raise ValueError("free-space sampler point indices exceed its uint64 counter range")
    scene._reserve(256*(stop-start))
    key = keyed_rng(settings["master_seed"], scene_id, query_id, stream="free_query_key").integers(0, 2**64, dtype=np.uint64)
    indices = np.arange(start, stop, dtype=np.uint64)
    result = np.empty((stop-start, 3), dtype=np.float32)
    pending = np.arange(stop-start, dtype=np.intp)
    attempts = np.zeros(stop-start, dtype=np.uint64)
    for _ in range(256):
        if not len(pending):
            return result
        counters = indices[pending, None]*np.uint64(3) + np.arange(3, dtype=np.uint64)
        counters ^= key
        counters ^= (attempts[pending, None]*np.uint64(0xD2B74407B1CE6E93))
        unit = (_splitmix64(counters) >> np.uint64(11)).astype(np.float64) * (2.0**-53)
        candidate = np.asarray(bounds[0] + unit*(bounds[1]-bounds[0]), dtype=np.float32)
        occupied = scene.contains_points(candidate, _reserved_bytes=192*(stop-start))
        result[pending[~occupied]] = candidate[~occupied]
        attempts[pending] += np.uint64(1)
        pending = pending[occupied]
    if len(pending):
        raise RuntimeError("free-space sampler exhausted 256 attempts for at least one point; reduce solid occupancy or choose other bounds")
    return result


def intervention(driver, bounds, params):
    position = params.get("intervention_position", np.sum(bounds, axis=0)-driver.position)
    return PointDriver(position, driver.strength)


def _indexed_points(scene, name, params, seed, scene_id, query_id, indices, bounds, drivers, reserved):
    result = np.empty((len(indices), 3), "f4")
    pending = np.arange(len(indices))
    pair_indices = indices // np.uint64(2) if name == "occlusion_boundary_v1" else indices
    key = keyed_rng(seed, scene_id, query_id, stream=f"query_{name}").integers(0, 2**64, dtype=np.uint64)
    anchor = None if name == "uniform_free_v1" else drivers[params["driver_index"]]
    other = intervention(anchor, bounds, params) if name == "counterfactual_v1" else None
    if other is not None and other.position == anchor.position:
        raise ValueError("counterfactual intervention must move the selected driver")
    for attempt in range(params.get("max_attempts", 256)):
        if not len(pending):
            return result
        counters = pair_indices[pending, None]*np.uint64(16) + np.arange(6, dtype=np.uint64)
        retry = np.uint64((attempt*0xD2B74407B1CE6E93) % 2**64)
        unit = (_splitmix64(counters ^ key ^ retry) >> np.uint64(11)).astype("f8") * (2.0**-53)
        a = np.asarray(bounds[0] + unit[:, :3]*(bounds[1]-bounds[0]), dtype="f4")
        keep = ~scene.contains_points(a, _reserved_bytes=reserved)
        if name == "long_range_v1":
            keep &= np.linalg.norm(a.astype("f8")-anchor.position, axis=1) >= params["min_distance"]
        if name == "counterfactual_v1":
            keep &= scene.visibility([anchor], a, _reserved_bytes=reserved)[0] != scene.visibility([other], a, _reserved_bytes=reserved)[0]
        if name == "occlusion_boundary_v1":
            b = np.asarray(bounds[0] + unit[:, 3:]*(bounds[1]-bounds[0]), dtype="f4")
            va = scene.visibility([anchor], a, _reserved_bytes=reserved)[0]
            vb = scene.visibility([anchor], b, _reserved_bytes=reserved)[0]
            keep &= ~scene.contains_points(b, _reserved_bytes=reserved) & (va != vb)
            chosen = np.flatnonzero(keep)
            if not len(chosen):
                continue
            left, right = a[chosen].copy(), b[chosen].copy()
            swap = ~va[chosen]
            left[swap], right[swap] = right[swap].copy(), left[swap].copy()
            for _ in range(params["refinements"]):
                middle = ((left.astype("f8")+right)*.5).astype("f4")
                visible = scene.visibility([anchor], middle, _reserved_bytes=reserved)[0]
                left[visible] = middle[visible]
                right[~visible] = middle[~visible]
            free = ~(scene.contains_points(left, _reserved_bytes=reserved) | scene.contains_points(right, _reserved_bytes=reserved))
            keep[chosen[~free]] = False
            odd = (indices[pending[chosen]] % np.uint64(2)).astype(bool)
            a[chosen] = np.where(odd[:, None], right, left)
        result[pending[keep]] = a[keep]
        pending = pending[~keep]
    if len(pending):
        raise RuntimeError(f"{name} exhausted its bounded attempts; the requested free-space/visibility condition may be impossible; change its parameters or scene")
    return result


def selected_points(scene, settings, scene_id, query_id, start, stop, bounds, drivers):
    """New v1 samplers use 16-counter indexed streams and the reference backend."""
    if stop > np.iinfo(np.uint64).max // 16:
        raise ValueError("query indices exceed the sampler's uint64 counter range")
    if scene.backend != "reference":
        raise ValueError("label-based sampling requires the reference backend to keep the distribution fixed")
    n = stop-start
    scene._reserve(1024*n)
    name, params = settings["sampler"], settings["params"]
    indices = np.arange(start, stop, dtype=np.uint64)
    if name != "mixed_v1":
        return _indexed_points(scene, name, params, settings["master_seed"], scene_id, query_id, indices, bounds, drivers, 768*n)
    key = keyed_rng(settings["master_seed"], scene_id, query_id, stream="query_mixture").integers(0, 2**64, dtype=np.uint64)
    unit = (_splitmix64(indices ^ key) >> np.uint64(11)).astype("f8") * (2.0**-53)
    weights = np.asarray(params["weights"])
    cumulative = np.cumsum(weights/weights.max())
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.0
    components = np.searchsorted(cumulative, unit, side="right")
    result = np.empty((n, 3), "f4")
    for i, component in enumerate(_COMPONENTS):
        selected = components == i
        if selected.any():
            result[selected] = _indexed_points(scene, component, params.get(component, {}), settings["master_seed"],
                                               scene_id, query_id, indices[selected], bounds, drivers, 768*n)
    return result
