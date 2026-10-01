"""Read and rerender canonical schema-v3 data without generator dependencies."""

import numpy as np
from rind_dataset.geometry import is_driver_valid_mixed
from rind_dataset.render import render_region


def unpack_region_bits(bits, global_size, x, y, width, height):
    """Decode one packed raster (or a stack of rasters) over a rectangle."""
    offsets = (y + np.arange(height)[:, None]) * global_size + x + np.arange(width)[None, :]
    flat = offsets.reshape(-1)
    packed = np.asarray(bits)
    if packed.ndim == 1:
        return ((packed[flat // 8] >> (flat % 8)) & 1).reshape(height, width).astype(bool)
    return ((packed[:, flat // 8] >> (flat % 8)) & 1).reshape(len(packed), height, width).astype(bool)



def render_scene_region(dataset, scene_id, x, y, side, drivers=None, *, strengths=None,
                        height=None, backend="auto"):
    """Rerender continuous scene geometry for arbitrary candidate (x,y,strength) rows.

    `side` is the width; optional `height` permits rectangular regions.
    Returns {channels: [M,H,W], response: [H,W], obstacle: [H,W]}.
    The response is the sum of channels; obstacle cells have zero response.
    """
    scene = dataset.get_scene(scene_id)
    world = dataset.global_size
    rows = side if height is None else height
    if not all(isinstance(v, (int, np.integer)) for v in (x, y, side, rows)) or side <= 0 or rows <= 0 or x < 0 or y < 0 or x + side > world or y + rows > world:
        raise ValueError("region must be an integer rectangle inside the world")
    if drivers is None:
        positions = scene["drivers"]
        strengths = scene.get("driver_strengths", np.ones(len(positions)))
        candidate = np.column_stack((positions, strengths))
    else:
        candidate = np.asarray(drivers, dtype=np.float64)
        if candidate.ndim == 1 and candidate.shape[0] in (2, 3):
            candidate = candidate[None, :]
        if candidate.ndim != 2 or candidate.shape[1] not in (2, 3) or not len(candidate):
            raise ValueError("drivers must have shape [M,2] or [M,3]")
        if candidate.shape[1] == 2:
            if strengths is None:
                source_strengths = (scene.get("driver_strengths", np.ones(len(candidate)))
                                    if len(candidate) == len(scene["drivers"])
                                    else np.ones(len(candidate)))
            else:
                source_strengths = np.asarray(strengths, dtype=np.float64)
                if source_strengths.shape != (len(candidate),):
                    raise ValueError("strengths must have shape [M]")
            candidate = np.column_stack((candidate, source_strengths))
        elif strengths is not None:
            raise ValueError("strengths should be omitted when drivers include a strength column")
    if not np.isfinite(candidate).all() or np.any(candidate[:, :2] < 0) or np.any(candidate[:, :2] > world) or np.any(candidate[:, 2] < 0) or np.any(candidate[:, 2] > 1):
        raise ValueError("candidate coordinates or strengths out of range")
    types = scene["obstacle_types"]
    params = scene["obstacle_params"]
    for row in candidate:
        if not is_driver_valid_mixed(row[:2], types, params, len(types), domain_size=world):
            raise ValueError("candidate driver lies inside an obstacle")
    channels = np.empty((len(candidate), rows, side), dtype=np.float64)
    obstacle = None
    for i, row in enumerate(candidate):
        state = render_region(row[:2], params, obs_types=types, x=x, y=y,
                              size=side, height=rows, backend=backend)
        if obstacle is None:
            obstacle = state == 2
        channels[i] = (state == 1).astype(np.float64) * row[2]
    return {"channels": channels, "response": channels.sum(axis=0), "obstacle": obstacle}

