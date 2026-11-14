"""Camera rays, arbitrary planes and lazy XYZ-to-ZYX tiled queries."""

from dataclasses import dataclass
import math

import numpy as np

from .drivers import MODELS
from .scene import integer, vector, world_bounds


def dimensions(value, name, count):
    value = tuple(value)
    if len(value) != count:
        raise ValueError(f"{name} must have {count} dimensions")
    return tuple(integer(v, name) for v in value)


def channel_names(channels, allowed):
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels) or any(c not in allowed for c in channels):
        raise ValueError(f"channels must be unique names from {tuple(allowed)}")
    return channels


def normalize(v, name):
    v = np.asarray(v, dtype=np.float64)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length == 0:
        raise ValueError(f"{name} must be finite and nonzero")
    return v / length


@dataclass(frozen=True)
class Camera:
    eye: tuple
    forward: tuple
    right: tuple
    up: tuple
    projection: str
    scale: float

    @staticmethod
    def _create(eye, look_at, up, projection, scale):
        eye = vector(eye, "eye").astype("f8")
        forward = normalize(vector(look_at, "look_at").astype("f8") - eye, "look direction")
        right = normalize(np.cross(forward, vector(up, "up")), "camera up cross look direction")
        up = np.cross(right, forward)
        return Camera(tuple(eye), tuple(forward), tuple(right), tuple(up), projection, float(scale))

    @classmethod
    def perspective(cls, eye, look_at, up=(0, 0, 1), fov_y_degrees=60):
        if not np.isfinite(fov_y_degrees) or not 0 < fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must lie strictly between 0 and 180")
        return cls._create(eye, look_at, up, "perspective", math.tan(math.radians(fov_y_degrees) / 2))

    @classmethod
    def orthographic(cls, eye, look_at, up=(0, 0, 1), vertical_size=2):
        if not np.isfinite(vertical_size) or vertical_size <= 0:
            raise ValueError("vertical_size must be finite and positive")
        return cls._create(eye, look_at, up, "orthographic", vertical_size / 2)

    def rays(self, size, *, start=0, stop=None):
        """Return contiguous float32 rays from global pixel centers, in row-major order."""
        width, height = dimensions(size, "size", 2)
        total = width * height
        start = integer(start, "start", 0)
        stop = total if stop is None else integer(stop, "stop", 0)
        if not start <= stop <= total:
            raise ValueError("pixel range lies outside image")
        indices = np.arange(start, stop, dtype=np.int64)
        x = ((indices % width + .5) / width * 2 - 1) * (width / height) * self.scale
        y = (1 - (indices // width + .5) / height * 2) * self.scale
        origins = np.empty((stop-start, 3), dtype=np.float32)
        directions = np.empty_like(origins)
        for k in range(3):
            offset = x * self.right[k] + y * self.up[k]
            origins[:, k] = self.eye[k] if self.projection == "perspective" else self.eye[k] + offset
            directions[:, k] = self.forward[k] + offset if self.projection == "perspective" else self.forward[k]
        return origins, directions


def render(scene, camera, size, channels, drivers, model, tmin, tmax):
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera")
    width, height = dimensions(size, "size", 2)
    channels = channel_names(channels, ("hit", "depth", "normal", "geometry_id", "response"))
    if model not in MODELS:
        raise ValueError("unknown response model")
    types = {"hit": np.bool_, "depth": np.float32, "normal": np.float32, "geometry_id": np.int32, "response": MODELS[model]}
    n = width * height
    output_bytes = sum(n * (3 if c == "normal" else 1) * np.dtype(types[c]).itemsize for c in channels)
    # Conservative bounded scratch allowance includes index arrays, generated
    # rays, hit buffers and first-hit response points. No full-frame ray array.
    scratch_per_pixel = 384 if scene.backend == "vulkan" else 256
    scene.engine.check_memory(output_bytes + 20*len(drivers) + scratch_per_pixel)
    step = min(65536, (scene.engine.memory_budget_bytes - output_bytes - 20*len(drivers)) // scratch_per_pixel)
    scene._reserve(output_bytes, 20*len(drivers) + scratch_per_pixel)
    result = {c: np.empty((n, 3) if c == "normal" else (n,), dtype=types[c]) for c in channels}
    fields = tuple(c for c in ("hit", "t", "normal", "geometry_id") if
                   c == "hit" or (c == "t" and ("depth" in channels or "response" in channels)) or
                   c in channels)
    for start in range(0, n, step):
        stop = min(start + step, n)
        scene._reserve(output_bytes, (stop-start)*192 + 20*len(drivers))
        origins, directions = camera.rays((width, height), start=start, stop=stop)
        hits = scene.trace(origins, directions, tmin=tmin, tmax=tmax, fields=fields,
                           _reserved_bytes=output_bytes + 64*(stop-start) + 20*len(drivers))
        for channel in channels:
            if channel != "response":
                result[channel][start:stop] = hits["t" if channel == "depth" else channel]
        if "response" in channels:
            target = result["response"][start:stop]
            target.fill(0)
            mask = hits["hit"]
            if np.any(mask):
                directions_hit = directions[mask].astype("f8")
                directions_hit /= np.linalg.norm(directions_hit, axis=1, keepdims=True)
                points = np.asarray(origins[mask].astype("f8") + directions_hit * hits["t"][mask, None], dtype="f4")
                target[mask] = scene.query(points, drivers, model=model,
                                           _reserved_bytes=output_bytes + 192*(stop-start))
    return {c: value.reshape((height, width, 3) if c == "normal" else (height, width)) for c, value in result.items()}


_POINT_CHANNELS = ("response", "per_driver", "visibility", "points")


def point_channels(scene, points, shape, drivers, channels, model, reserved_bytes=0):
    result = {}
    if "points" in channels:
        result["points"] = points.reshape((*shape, 3))
    if "response" in channels or "per_driver" in channels:
        output = "both" if "response" in channels and "per_driver" in channels else (
            "aggregate" if "response" in channels else "per_driver")
        query = scene.query(points, drivers, model=model, output=output,
                            return_visibility="visibility" in channels, _reserved_bytes=reserved_bytes)
        if output == "both" or "visibility" in channels:
            if "response" in channels:
                result["response"] = query.aggregate.reshape(shape)
            if "per_driver" in channels:
                result["per_driver"] = query.per_driver.reshape((len(drivers), *shape))
            if "visibility" in channels:
                result["visibility"] = query.visibility.reshape((len(drivers), *shape))
        elif output == "aggregate":
            result["response"] = query.reshape(shape)
        else:
            result["per_driver"] = query.reshape((len(drivers), *shape))
    elif "visibility" in channels:
        result["visibility"] = scene.visibility(drivers, points, _reserved_bytes=reserved_bytes).reshape((len(drivers), *shape))
    return result


def point_output_bytes(channels, model, sources):
    if model not in MODELS:
        raise ValueError("unknown response model")
    itemsize = np.dtype(MODELS[model]).itemsize
    return sum({"response": itemsize, "per_driver": itemsize*sources, "visibility": sources, "points": 12}[c]
               for c in channels)


def planar_slice(scene, origin, axis_u, axis_v, size, extent, drivers, channel, model):
    single = isinstance(channel, str)
    channels = channel_names((channel,) if single else channel, _POINT_CHANNELS)
    width, height = dimensions(size, "size", 2)
    origin = vector(origin, "origin").astype("f8")
    u = normalize(vector(axis_u, "axis_u"), "axis_u")
    v = vector(axis_v, "axis_v").astype("f8")
    v = normalize(v - np.dot(v, u)*u, "axis_v perpendicular component")
    extent = np.asarray(extent, dtype=np.float64)
    if extent.shape != (2,) or not np.isfinite(extent).all() or not (extent > 0).all():
        raise ValueError("extent must contain positive finite width and height")
    n = width*height
    scene._reserve(n * (32 + point_output_bytes(channels, model, len(drivers))) + 32*(width+height) + 20*len(drivers))
    x = ((np.arange(width, dtype="f8") + .5) / width - .5) * extent[0]
    y = ((np.arange(height, dtype="f8") + .5) / height - .5) * extent[1]
    points = np.empty((height, width, 3), dtype="f4")
    for k in range(3):
        points[:, :, k] = origin[k] + x[None, :]*u[k] + y[:, None]*v[k]
    result = point_channels(scene, points.reshape(-1, 3), (height, width), drivers, channels, model,
                            points.nbytes + x.nbytes + y.nbytes)
    return result[channels[0]] if single else result


class Volume:
    """Descriptor-only grid handle. Output spatial axes are always [z,y,x]."""

    def __init__(self, scene, bounds, shape, drivers, channels, chunk_shape, model):
        self.scene = scene
        self.bounds = world_bounds(bounds)
        self.shape = dimensions(shape, "shape", 3)
        self.chunk_shape = dimensions(chunk_shape, "chunk_shape", 3)
        self.channels = channel_names(channels, _POINT_CHANNELS)
        self.drivers = drivers
        self.model = model
        self.point_count = math.prod(self.shape)
        self._bytes_per_point = point_output_bytes(self.channels, model, len(drivers))

    def _tile_peak(self):
        shape = tuple(min(a, b) for a, b in zip(self.shape, self.chunk_shape))
        n = math.prod(shape)
        # Points are already included when returned as a channel. Axis generation
        # uses only short float64 vectors, never three full meshgrid arrays.
        return n*(self._bytes_per_point + (0 if "points" in self.channels else 12)) + min(n, 65536) + 32*sum(shape) + 20*len(self.drivers)

    def blocks(self, _reserved_bytes=0):
        self.scene._reserve(self._tile_peak(), _reserved_bytes)
        axes = [range(0, size, step) for size, step in zip(self.shape, self.chunk_shape)]
        # Product of range iterators must not be materialized: itertools.product
        # pools each input, which can be huge for a logical 64-bit volume.
        for z in axes[0]:
            for y in axes[1]:
                for x in axes[2]:
                    starts = (z, y, x)
                    stops = tuple(min(start+step, size) for start, step, size in zip(starts, self.chunk_shape, self.shape))
                    region = tuple(slice(a, b) for a, b in zip(starts, stops))
                    shape = tuple(b-a for a, b in zip(starts, stops))
                    self.scene._reserve(self._tile_peak(), _reserved_bytes)
                    points = np.empty((*shape, 3), dtype="f4")
                    for xyz in range(3):
                        axis = 2-xyz
                        lo, hi = self.bounds[0][xyz], self.bounds[1][xyz]
                        coords = lo + (np.arange(starts[axis], stops[axis], dtype="f8") + .5) * ((hi-lo)/self.shape[axis])
                        broadcast = [1, 1, 1]
                        broadcast[axis] = len(coords)
                        points[..., xyz] = coords.reshape(broadcast)
                    yield region, point_channels(self.scene, points.reshape(-1, 3), shape, self.drivers, self.channels, self.model,
                                                  _reserved_bytes + points.nbytes + 32*sum(shape))
                    del points

    def materialize(self):
        full_bytes = self.point_count*self._bytes_per_point
        self.scene._reserve(full_bytes + self._tile_peak())
        result = {}
        for channel in self.channels:
            dtype = np.bool_ if channel == "visibility" else (np.float32 if channel == "points" else MODELS[self.model])
            shape = (*self.shape, 3) if channel == "points" else (
                (len(self.drivers), *self.shape) if channel in ("per_driver", "visibility") else self.shape)
            result[channel] = np.empty(shape, dtype=dtype)
        for region, block in self.blocks(_reserved_bytes=full_bytes):
            for channel in self.channels:
                target = (slice(None), *region) if channel in ("per_driver", "visibility") else region
                result[channel][target] = block[channel]
            del block
        return result
