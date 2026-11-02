"""XYZ coordinates, ZYX arrays, exact power-of-two voxel-center conversion."""

from dataclasses import dataclass
import numpy as np

RESOLUTION = 1024
WINDOW_SIDES = (64, 128, 256)


def xyz64(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite float64 XYZ coordinates, shape (N,3)")
    return np.ascontiguousarray(a)


def voxel_centres(voxel_xyz, resolution=RESOLUTION):
    p = np.asarray(voxel_xyz)
    if p.dtype.kind not in "iu" or p.ndim < 1 or p.shape[-1] != 3:
        raise ValueError("Voxel indices must be integer XYZ triples")
    if np.any(p < 0) or np.any(p >= resolution):
        raise ValueError("Voxel outside world")
    return (p.astype(np.float64) + 0.5) / resolution


def world_to_voxel(world_xyz, resolution=RESOLUTION):
    p = xyz64(world_xyz)
    if np.any(p < 0) or np.any(p >= 1):
        raise ValueError("Voxel indexing uses the half-open world [0,1)")
    return np.floor(p * resolution).astype(np.int64)


@dataclass(frozen=True)
class Window:
    origin: tuple[int, int, int]
    side: int

    def __post_init__(self):
        if len(self.origin) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer))
            for v in (*self.origin, self.side)
        ):
            raise ValueError("Window requires integer XYZ origin and side")
        if self.side not in WINDOW_SIDES:
            raise ValueError("Stored observation side must be 64,128,256")
        if min(self.origin) < 0 or max(self.origin) + self.side > RESOLUTION:
            raise ValueError("Window outside world")
        object.__setattr__(self, "origin", tuple(map(int, self.origin)))
        object.__setattr__(self, "side", int(self.side))

    @property
    def world_origin(self):
        return np.asarray(self.origin, dtype=np.float64) / RESOLUTION

    @property
    def world_side(self):
        return self.side / RESOLUTION

    def contains(self, world_xyz):
        p = xyz64(world_xyz)
        lo = self.world_origin
        return np.all((p >= lo) & (p < lo + self.world_side), axis=1)

    def centres(self, local_xyz):
        p = np.asarray(local_xyz)
        if (
            p.dtype.kind not in "iu"
            or p.shape[-1] != 3
            or np.any(p < 0)
            or np.any(p >= self.side)
        ):
            raise ValueError("Local voxel outside window")
        return voxel_centres(p + np.asarray(self.origin))
