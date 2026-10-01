"""Rasterization and rendering routines for dataset-g1024-l128-binary.

Supports:
- Multi-driver visibility (OR semantics: visible if ANY driver has line of sight)
- Mixed primitives: Rotated Rectangle, Ellipse, Triangle, Simple Polygon
- Backwards-compatible single-driver and axis-aligned boxes
"""

import warnings
from typing import Sequence, Union, Optional, List
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    prange = range

from rind_dataset.constants import (
    GLOBAL_SIZE,
    FREE_NO_RESPONSE,
    FREE_RESPONSE,
    OBSTACLE,
    OBSTACLE_RECTANGLE,
    OBSTACLE_ELLIPSE,
    OBSTACLE_TRIANGLE,
    OBSTACLE_POLYGON,
    PARAM_WIDTH,
)
from rind_dataset.geometry import (
    point_in_obstacle,
    segment_intersects_obstacle,
    compute_obstacle_aabb,
)


def _convert_boxes_to_mixed(boxes: np.ndarray, target_width: int = PARAM_WIDTH):
    """Convert legacy (K, 4) axis-aligned boxes to (types, params)."""
    k = len(boxes)
    obs_types = np.zeros(k, dtype=np.uint8)
    obs_params = np.zeros((k, target_width), dtype=np.float64)
    for i in range(k):
        xmin, ymin, xmax, ymax = boxes[i]
        obs_params[i, 0] = (xmin + xmax) * 0.5
        obs_params[i, 1] = (ymin + ymax) * 0.5
        obs_params[i, 2] = (xmax - xmin) * 0.5
        obs_params[i, 3] = (ymax - ymin) * 0.5
        obs_params[i, 4] = 0.0
    return obs_types, obs_params


def _standardize_drivers(drivers: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Standardize driver input to (M, 2) float64 array."""
    arr = np.asarray(drivers, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return arr.reshape(1, 2)
        else:
            raise ValueError(f"1D driver array must have shape (2,), got {arr.shape}")
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"2D driver array must have shape (M, 2), got {arr.shape}")
        return arr
    else:
        raise ValueError(f"Driver array must be 1D or 2D, got {arr.ndim}D")


def render_global_reference(
    drivers: Union[Sequence[float], np.ndarray],
    obstacles: np.ndarray,
    obs_types: Optional[np.ndarray] = None,
    size: int = GLOBAL_SIZE,
    x0: int = 0,
    y0: int = 0,
    height: Optional[int] = None,
) -> np.ndarray:
    """Reference pure-Python / NumPy renderer supporting multi-driver and mixed primitives.

    Prioritizes obvious correctness over performance.
    """
    drivers_arr = _standardize_drivers(drivers)
    num_drivers = len(drivers_arr)

    if obs_types is None:
        if obstacles.ndim == 2 and obstacles.shape[1] == 4:
            obs_types_arr, obs_params_arr = _convert_boxes_to_mixed(obstacles, PARAM_WIDTH)
        else:
            obs_params_arr = obstacles
            obs_types_arr = np.zeros(len(obstacles), dtype=np.uint8)
    else:
        obs_types_arr = np.asarray(obs_types, dtype=np.uint8)
        obs_params_arr = np.asarray(obstacles, dtype=np.float64)

    num_obs = len(obs_types_arr)
    rows = size if height is None else height
    grid = np.empty((rows, size), dtype=np.uint8)

    for y in range(rows):
        qy = y0 + y + 0.5
        for x in range(size):
            qx = x0 + x + 0.5

            # Step 1: Check obstacle membership
            is_obs = False
            for k in range(num_obs):
                if point_in_obstacle(qx, qy, int(obs_types_arr[k]), obs_params_arr[k]):
                    is_obs = True
                    break

            if is_obs:
                grid[y, x] = OBSTACLE
                continue

            # Step 2: Multi-driver line-of-sight test (OR condition)
            any_visible = False
            for d in range(num_drivers):
                px, py = drivers_arr[d, 0], drivers_arr[d, 1]
                blocked = False
                for k in range(num_obs):
                    if segment_intersects_obstacle(px, py, qx, qy, int(obs_types_arr[k]), obs_params_arr[k]):
                        blocked = True
                        break

                if not blocked:
                    any_visible = True
                    break

            if any_visible:
                grid[y, x] = FREE_RESPONSE
            else:
                grid[y, x] = FREE_NO_RESPONSE

    return grid


render_reference = render_global_reference


@njit(parallel=True, fastmath=False)
def _render_numba_core(
    drivers: np.ndarray,       # (M, 2)
    obs_types: np.ndarray,     # (K,)
    obs_params: np.ndarray,    # (K, PARAM_WIDTH)
    obs_aabbs: np.ndarray,     # (K, 4)
    num_drivers: int,
    num_obs: int,
    size: int,
    x0: int = 0,
    y0: int = 0,
    height: int = -1,
) -> np.ndarray:
    """Numba-accelerated parallel renderer kernel supporting multi-driver and mixed shapes."""
    rows = size if height < 0 else height
    grid = np.empty((rows, size), dtype=np.uint8)

    for y in prange(rows):
        qy = y0 + y + 0.5
        for x in range(size):
            qx = x0 + x + 0.5

            # ---------------------------------------------------------------
            # 1. Obstacle membership check
            # ---------------------------------------------------------------
            is_obs = False
            for k in range(num_obs):
                # AABB early rejection
                if (qx < obs_aabbs[k, 0] or qx > obs_aabbs[k, 2]
                        or qy < obs_aabbs[k, 1] or qy > obs_aabbs[k, 3]):
                    continue

                otype = obs_types[k]
                if otype == 0:  # RECTANGLE
                    cx = obs_params[k, 0]
                    cy = obs_params[k, 1]
                    hw = obs_params[k, 2]
                    hh = obs_params[k, 3]
                    theta = obs_params[k, 4]
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    dx = qx - cx
                    dy = qy - cy
                    u = dx * cos_t + dy * sin_t
                    v = -dx * sin_t + dy * cos_t
                    if abs(u) <= hw and abs(v) <= hh:
                        is_obs = True
                        break

                elif otype == 1:  # ELLIPSE
                    cx = obs_params[k, 0]
                    cy = obs_params[k, 1]
                    rx = obs_params[k, 2]
                    ry = obs_params[k, 3]
                    theta = obs_params[k, 4]
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    dx = qx - cx
                    dy = qy - cy
                    u = (dx * cos_t + dy * sin_t) / rx
                    v = (-dx * sin_t + dy * cos_t) / ry
                    if (u * u + v * v) <= 1.0:
                        is_obs = True
                        break

                elif otype == 2:  # TRIANGLE
                    x1 = obs_params[k, 0]
                    y1 = obs_params[k, 1]
                    x2 = obs_params[k, 2]
                    y2 = obs_params[k, 3]
                    x3 = obs_params[k, 4]
                    y3 = obs_params[k, 5]

                    cp1 = (x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1)
                    cp2 = (x3 - x2) * (qy - y2) - (y3 - y2) * (qx - x2)
                    cp3 = (x1 - x3) * (qy - y3) - (y1 - y3) * (qx - x3)

                    has_neg = (cp1 < 0.0) or (cp2 < 0.0) or (cp3 < 0.0)
                    has_pos = (cp1 > 0.0) or (cp2 > 0.0) or (cp3 > 0.0)
                    if not (has_neg and has_pos):
                        is_obs = True
                        break

                elif otype == 3:  # POLYGON
                    n = int(np.round(obs_params[k, 0]))
                    # Point on edge check
                    on_edge = False
                    for i in range(n):
                        j = (i + 1) % n
                        vix = obs_params[k, 1 + 2 * i]
                        viy = obs_params[k, 2 + 2 * i]
                        vjx = obs_params[k, 1 + 2 * j]
                        vjy = obs_params[k, 2 + 2 * j]
                        cp = (vjx - vix) * (qy - viy) - (vjy - viy) * (qx - vix)
                        if cp == 0.0:
                            if min(vix, vjx) <= qx <= max(vix, vjx) and min(viy, vjy) <= qy <= max(viy, vjy):
                                on_edge = True
                                break
                    if on_edge:
                        is_obs = True
                        break

                    # Ray casting test
                    inside = False
                    for i in range(n):
                        j = (i + 1) % n
                        vix = obs_params[k, 1 + 2 * i]
                        viy = obs_params[k, 2 + 2 * i]
                        vjx = obs_params[k, 1 + 2 * j]
                        vjy = obs_params[k, 2 + 2 * j]
                        if (viy > qy) != (vjy > qy):
                            x_int = vix + (qy - viy) * (vjx - vix) / (vjy - viy)
                            if qx < x_int:
                                inside = not inside
                    if inside:
                        is_obs = True
                        break

            if is_obs:
                grid[y, x] = 2  # OBSTACLE
                continue

            # ---------------------------------------------------------------
            # 2. Multi-driver Line of sight: OR condition
            # ---------------------------------------------------------------
            any_visible = False

            for d in range(num_drivers):
                driver_x = drivers[d, 0]
                driver_y = drivers[d, 1]

                seg_min_x = min(driver_x, qx)
                seg_max_x = max(driver_x, qx)
                seg_min_y = min(driver_y, qy)
                seg_max_y = max(driver_y, qy)

                driver_blocked = False

                for k in range(num_obs):
                    # AABB overlap rejection
                    if (seg_max_x < obs_aabbs[k, 0] or seg_min_x > obs_aabbs[k, 2]
                            or seg_max_y < obs_aabbs[k, 1] or seg_min_y > obs_aabbs[k, 3]):
                        continue

                    otype = obs_types[k]

                    if otype == 0:  # RECTANGLE
                        cx = obs_params[k, 0]
                        cy = obs_params[k, 1]
                        hw = obs_params[k, 2]
                        hh = obs_params[k, 3]
                        theta = obs_params[k, 4]
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)

                        dpx = driver_x - cx
                        dpy = driver_y - cy
                        pu = dpx * cos_t + dpy * sin_t
                        pv = -dpx * sin_t + dpy * cos_t

                        dqx = qx - cx
                        dqy = qy - cy
                        qu = dqx * cos_t + dqy * sin_t
                        qv = -dqx * sin_t + dqy * cos_t

                        du = qu - pu
                        dv = qv - pv

                        # X slab [-hw, hw]
                        if du == 0.0:
                            if pu < -hw or pu > hw:
                                continue
                            t1x = -np.inf
                            t2x = np.inf
                        else:
                            inv_du = 1.0 / du
                            t_a = (-hw - pu) * inv_du
                            t_b = (hw - pu) * inv_du
                            if t_a <= t_b:
                                t1x = t_a
                                t2x = t_b
                            else:
                                t1x = t_b
                                t2x = t_a

                        # Y slab [-hh, hh]
                        if dv == 0.0:
                            if pv < -hh or pv > hh:
                                continue
                            t1y = -np.inf
                            t2y = np.inf
                        else:
                            inv_dv = 1.0 / dv
                            t_a = (-hh - pv) * inv_dv
                            t_b = (hh - pv) * inv_dv
                            if t_a <= t_b:
                                t1y = t_a
                                t2y = t_b
                            else:
                                t1y = t_b
                                t2y = t_a

                        t_enter = max(t1x, t1y)
                        t_exit = min(t2x, t2y)

                        if t_enter <= t_exit:
                            t_start = max(t_enter, 0.0)
                            t_end = min(t_exit, 1.0)
                            if t_start <= t_end:
                                driver_blocked = True
                                break

                    elif otype == 1:  # ELLIPSE
                        cx = obs_params[k, 0]
                        cy = obs_params[k, 1]
                        rx = obs_params[k, 2]
                        ry = obs_params[k, 3]
                        theta = obs_params[k, 4]
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)

                        dpx = driver_x - cx
                        dpy = driver_y - cy
                        pu = (dpx * cos_t + dpy * sin_t) / rx
                        pv = (-dpx * sin_t + dpy * cos_t) / ry

                        dqx = qx - cx
                        dqy = qy - cy
                        qu = (dqx * cos_t + dqy * sin_t) / rx
                        qv = (-dqx * sin_t + dqy * cos_t) / ry

                        du = qu - pu
                        dv = qv - pv

                        a = du * du + dv * dv
                        if a == 0.0:
                            if (pu * pu + pv * pv) <= 1.0:
                                driver_blocked = True
                                break
                        else:
                            b = 2.0 * (pu * du + pv * dv)
                            c = pu * pu + pv * pv - 1.0
                            disc = b * b - 4.0 * a * c
                            if disc >= 0.0:
                                sqrt_disc = np.sqrt(disc)
                                inv_2a = 1.0 / (2.0 * a)
                                t1 = (-b - sqrt_disc) * inv_2a
                                t2 = (-b + sqrt_disc) * inv_2a
                                t_start = max(t1, 0.0)
                                t_end = min(t2, 1.0)
                                if t_start <= t_end:
                                    driver_blocked = True
                                    break

                    elif otype == 2:  # TRIANGLE
                        x1 = obs_params[k, 0]
                        y1 = obs_params[k, 1]
                        x2 = obs_params[k, 2]
                        y2 = obs_params[k, 3]
                        x3 = obs_params[k, 4]
                        y3 = obs_params[k, 5]

                        # 1. Driver endpoint inside triangle
                        cp1 = (x2 - x1) * (driver_y - y1) - (y2 - y1) * (driver_x - x1)
                        cp2 = (x3 - x2) * (driver_y - y2) - (y3 - y2) * (driver_x - x2)
                        cp3 = (x1 - x3) * (driver_y - y3) - (y1 - y3) * (driver_x - x3)
                        has_neg = (cp1 < 0.0) or (cp2 < 0.0) or (cp3 < 0.0)
                        has_pos = (cp1 > 0.0) or (cp2 > 0.0) or (cp3 > 0.0)
                        if not (has_neg and has_pos):
                            driver_blocked = True
                            break

                        # 2. Target endpoint inside triangle
                        cp1 = (x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1)
                        cp2 = (x3 - x2) * (qy - y2) - (y3 - y2) * (qx - x2)
                        cp3 = (x1 - x3) * (qy - y3) - (y1 - y3) * (qx - x3)
                        has_neg = (cp1 < 0.0) or (cp2 < 0.0) or (cp3 < 0.0)
                        has_pos = (cp1 > 0.0) or (cp2 > 0.0) or (cp3 > 0.0)
                        if not (has_neg and has_pos):
                            driver_blocked = True
                            break

                        # 3. Intersect 3 edges
                        # Edge 1
                        d1 = (x2 - x1) * (driver_y - y1) - (y2 - y1) * (driver_x - x1)
                        d2 = (x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1)
                        d3 = (qx - driver_x) * (y1 - driver_y) - (qy - driver_y) * (x1 - driver_x)
                        d4 = (qx - driver_x) * (y2 - driver_y) - (qy - driver_y) * (x2 - driver_x)
                        edge_hit = False
                        if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0))
                                and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
                            edge_hit = True
                        elif d1 == 0.0 and (min(x1, x2) <= driver_x <= max(x1, x2) and min(y1, y2) <= driver_y <= max(y1, y2)):
                            edge_hit = True
                        elif d2 == 0.0 and (min(x1, x2) <= qx <= max(x1, x2) and min(y1, y2) <= qy <= max(y1, y2)):
                            edge_hit = True
                        elif d3 == 0.0 and (seg_min_x <= x1 <= seg_max_x and seg_min_y <= y1 <= seg_max_y):
                            edge_hit = True
                        elif d4 == 0.0 and (seg_min_x <= x2 <= seg_max_x and seg_min_y <= y2 <= seg_max_y):
                            edge_hit = True
                        if edge_hit:
                            driver_blocked = True
                            break

                        # Edge 2
                        d1 = (x3 - x2) * (driver_y - y2) - (y3 - y2) * (driver_x - x2)
                        d2 = (x3 - x2) * (qy - y2) - (y3 - y2) * (qx - x2)
                        d3 = (qx - driver_x) * (y2 - driver_y) - (qy - driver_y) * (x2 - driver_x)
                        d4 = (qx - driver_x) * (y3 - driver_y) - (qy - driver_y) * (x3 - driver_x)
                        if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0))
                                and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
                            edge_hit = True
                        elif d1 == 0.0 and (min(x2, x3) <= driver_x <= max(x2, x3) and min(y2, y3) <= driver_y <= max(y2, y3)):
                            edge_hit = True
                        elif d2 == 0.0 and (min(x2, x3) <= qx <= max(x2, x3) and min(y2, y3) <= qy <= max(y2, y3)):
                            edge_hit = True
                        elif d3 == 0.0 and (seg_min_x <= x2 <= seg_max_x and seg_min_y <= y2 <= seg_max_y):
                            edge_hit = True
                        elif d4 == 0.0 and (seg_min_x <= x3 <= seg_max_x and seg_min_y <= y3 <= seg_max_y):
                            edge_hit = True
                        if edge_hit:
                            driver_blocked = True
                            break

                        # Edge 3
                        d1 = (x1 - x3) * (driver_y - y3) - (y1 - y3) * (driver_x - x3)
                        d2 = (x1 - x3) * (qy - y3) - (y1 - y3) * (qx - x3)
                        d3 = (qx - driver_x) * (y3 - driver_y) - (qy - driver_y) * (x3 - driver_x)
                        d4 = (qx - driver_x) * (y1 - driver_y) - (qy - driver_y) * (x1 - driver_x)
                        if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0))
                                and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
                            edge_hit = True
                        elif d1 == 0.0 and (min(x3, x1) <= driver_x <= max(x3, x1) and min(y3, y1) <= driver_y <= max(y3, y1)):
                            edge_hit = True
                        elif d2 == 0.0 and (min(x3, x1) <= qx <= max(x3, x1) and min(y3, y1) <= qy <= max(y3, y1)):
                            edge_hit = True
                        elif d3 == 0.0 and (seg_min_x <= x3 <= seg_max_x and seg_min_y <= y3 <= seg_max_y):
                            edge_hit = True
                        elif d4 == 0.0 and (seg_min_x <= x1 <= seg_max_x and seg_min_y <= y1 <= seg_max_y):
                            edge_hit = True
                        if edge_hit:
                            driver_blocked = True
                            break

                    elif otype == 3:  # POLYGON
                        n = int(np.round(obs_params[k, 0]))

                        # 1. Driver endpoint inside polygon
                        # On-edge check
                        d_on_edge = False
                        for i in range(n):
                            j = (i + 1) % n
                            vix = obs_params[k, 1 + 2 * i]
                            viy = obs_params[k, 2 + 2 * i]
                            vjx = obs_params[k, 1 + 2 * j]
                            vjy = obs_params[k, 2 + 2 * j]
                            cp = (vjx - vix) * (driver_y - viy) - (vjy - viy) * (driver_x - vix)
                            if cp == 0.0 and min(vix, vjx) <= driver_x <= max(vix, vjx) and min(viy, vjy) <= driver_y <= max(viy, vjy):
                                d_on_edge = True
                                break
                        if d_on_edge:
                            driver_blocked = True
                            break

                        d_inside = False
                        for i in range(n):
                            j = (i + 1) % n
                            vix = obs_params[k, 1 + 2 * i]
                            viy = obs_params[k, 2 + 2 * i]
                            vjx = obs_params[k, 1 + 2 * j]
                            vjy = obs_params[k, 2 + 2 * j]
                            if (viy > driver_y) != (vjy > driver_y):
                                x_int = vix + (driver_y - viy) * (vjx - vix) / (vjy - viy)
                                if driver_x < x_int:
                                    d_inside = not d_inside
                        if d_inside:
                            driver_blocked = True
                            break

                        # 2. Target endpoint inside polygon
                        q_on_edge = False
                        for i in range(n):
                            j = (i + 1) % n
                            vix = obs_params[k, 1 + 2 * i]
                            viy = obs_params[k, 2 + 2 * i]
                            vjx = obs_params[k, 1 + 2 * j]
                            vjy = obs_params[k, 2 + 2 * j]
                            cp = (vjx - vix) * (qy - viy) - (vjy - viy) * (qx - vix)
                            if cp == 0.0 and min(vix, vjx) <= qx <= max(vix, vjx) and min(viy, vjy) <= qy <= max(viy, vjy):
                                q_on_edge = True
                                break
                        if q_on_edge:
                            driver_blocked = True
                            break

                        q_inside = False
                        for i in range(n):
                            j = (i + 1) % n
                            vix = obs_params[k, 1 + 2 * i]
                            viy = obs_params[k, 2 + 2 * i]
                            vjx = obs_params[k, 1 + 2 * j]
                            vjy = obs_params[k, 2 + 2 * j]
                            if (viy > qy) != (vjy > qy):
                                x_int = vix + (qy - viy) * (vjx - vix) / (vjy - viy)
                                if qx < x_int:
                                    q_inside = not q_inside
                        if q_inside:
                            driver_blocked = True
                            break

                        # 3. Intersect any of the n polygon edges
                        poly_edge_hit = False
                        for i in range(n):
                            j = (i + 1) % n
                            vix = obs_params[k, 1 + 2 * i]
                            viy = obs_params[k, 2 + 2 * i]
                            vjx = obs_params[k, 1 + 2 * j]
                            vjy = obs_params[k, 2 + 2 * j]

                            d1 = (vjx - vix) * (driver_y - viy) - (vjy - viy) * (driver_x - vix)
                            d2 = (vjx - vix) * (qy - viy) - (vjy - viy) * (qx - vix)
                            d3 = (qx - driver_x) * (viy - driver_y) - (qy - driver_y) * (vix - driver_x)
                            d4 = (qx - driver_x) * (vjy - driver_y) - (qy - driver_y) * (vjx - driver_x)

                            if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0))
                                    and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
                                poly_edge_hit = True
                                break
                            elif d1 == 0.0 and (min(vix, vjx) <= driver_x <= max(vix, vjx) and min(viy, vjy) <= driver_y <= max(viy, vjy)):
                                poly_edge_hit = True
                                break
                            elif d2 == 0.0 and (min(vix, vjx) <= qx <= max(vix, vjx) and min(viy, vjy) <= qy <= max(viy, vjy)):
                                poly_edge_hit = True
                                break
                            elif d3 == 0.0 and (seg_min_x <= vix <= seg_max_x and seg_min_y <= viy <= seg_max_y):
                                poly_edge_hit = True
                                break
                            elif d4 == 0.0 and (seg_min_x <= vjx <= seg_max_x and seg_min_y <= vjy <= seg_max_y):
                                poly_edge_hit = True
                                break

                        if poly_edge_hit:
                            driver_blocked = True
                            break

                # If this driver is not blocked, line of sight is clear!
                if not driver_blocked:
                    any_visible = True
                    break

            if any_visible:
                grid[y, x] = 1  # FREE_RESPONSE
            else:
                grid[y, x] = 0  # FREE_NO_RESPONSE

    return grid


def render_global(
    drivers: Union[Sequence[float], np.ndarray],
    obstacles: np.ndarray,
    obs_types: Optional[np.ndarray] = None,
    size: int = GLOBAL_SIZE,
    backend: str = "auto",
) -> np.ndarray:
    """Primary global renderer API supporting selectable backends ('auto', 'numba', 'reference').

    Supports multi-driver (OR visibility), mixed primitives (rect, ellipse, tri, poly),
    and legacy axis-aligned boxes.
    """
    backend = backend.lower()
    if backend not in ("auto", "numba", "reference"):
        raise ValueError(
            f"Invalid renderer backend '{backend}'. Supported options: 'auto', 'numba', 'reference'."
        )

    if backend == "reference":
        return render_global_reference(drivers, obstacles, obs_types=obs_types, size=size)

    if backend == "numba":
        if not HAS_NUMBA:
            raise RuntimeError(
                "Numba renderer requested (backend='numba'), but Numba is not installed or failed to import. "
                "Please install numba (e.g. `uv sync` or `pip install numba`) or use backend='auto' / 'reference'."
            )
    elif backend == "auto":
        if not HAS_NUMBA:
            warnings.warn(
                "Numba is not installed or unavailable; falling back to portable reference renderer. "
                "Rendering will be significantly slower.",
                UserWarning,
                stacklevel=2,
            )
            return render_global_reference(drivers, obstacles, obs_types=obs_types, size=size)

    drivers_arr = _standardize_drivers(drivers)
    num_drivers = len(drivers_arr)

    if obs_types is None:
        if obstacles.ndim == 2 and obstacles.shape[1] == 4:
            obs_types_arr, obs_params_arr = _convert_boxes_to_mixed(obstacles, PARAM_WIDTH)
        else:
            obs_params_arr = obstacles
            obs_types_arr = np.zeros(len(obstacles), dtype=np.uint8)
    else:
        obs_types_arr = np.ascontiguousarray(obs_types, dtype=np.uint8)
        obs_params_arr = np.ascontiguousarray(obstacles, dtype=np.float64)

    num_obs = len(obs_types_arr)

    # Pad params if necessary to PARAM_WIDTH
    if obs_params_arr.shape[1] < PARAM_WIDTH:
        padded = np.zeros((num_obs, PARAM_WIDTH), dtype=np.float64)
        padded[:, :obs_params_arr.shape[1]] = obs_params_arr
        obs_params_arr = padded

    # Precalculate AABBs for fast rejection
    obs_aabbs = np.empty((num_obs, 4), dtype=np.float64)
    for k in range(num_obs):
        obs_aabbs[k] = compute_obstacle_aabb(int(obs_types_arr[k]), obs_params_arr[k])

    return _render_numba_core(
        drivers_arr, obs_types_arr, obs_params_arr, obs_aabbs, num_drivers, num_obs, size
    )


def render_region(drivers, obstacles, *, obs_types=None, x: int, y: int,
                  size: int, height: Optional[int] = None,
                  backend: str = "auto") -> np.ndarray:
    """Render a world-coordinate region directly, without a global raster."""
    rows = size if height is None else height
    if not all(isinstance(v, (int, np.integer)) for v in (x, y, size, rows)) or size <= 0 or rows <= 0 or x < 0 or y < 0:
        raise ValueError("x, y, width (size) and height must be valid integers")
    if backend not in ("auto", "numba", "reference"):
        raise ValueError(f"Invalid renderer backend '{backend}'")
    if backend == "reference" or (backend == "auto" and not HAS_NUMBA):
        return render_global_reference(drivers, obstacles, obs_types=obs_types,
                                       size=size, x0=x, y0=y, height=rows)
    if not HAS_NUMBA:
        raise RuntimeError("Numba renderer requested but Numba is unavailable")
    drivers_arr = _standardize_drivers(drivers)
    obstacles_arr = np.asarray(obstacles)
    if obs_types is None and obstacles_arr.ndim == 2 and obstacles_arr.shape[1] == 4:
        types, params = _convert_boxes_to_mixed(obstacles_arr)
    else:
        types = np.zeros(len(obstacles_arr), dtype=np.uint8) if obs_types is None else np.ascontiguousarray(obs_types, dtype=np.uint8)
        params = np.ascontiguousarray(obstacles_arr, dtype=np.float64)
    if params.shape[1] < PARAM_WIDTH:
        padded = np.zeros((len(params), PARAM_WIDTH), dtype=np.float64)
        padded[:, :params.shape[1]] = params
        params = padded
    aabbs = np.empty((len(types), 4), dtype=np.float64)
    for k in range(len(types)):
        aabbs[k] = compute_obstacle_aabb(int(types[k]), params[k])
    return _render_numba_core(drivers_arr, types, params, aabbs,
                              len(drivers_arr), len(types), size, x, y, rows)
