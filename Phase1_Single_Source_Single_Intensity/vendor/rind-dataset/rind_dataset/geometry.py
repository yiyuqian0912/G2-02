"""Continuous 2D geometry functions for dataset-g1024-l128-binary.

Supports:
- Rotated Rectangle
- Ellipse
- Triangle
- Concave and Convex Simple Polygon
- Axis-aligned rectangle (backward compatibility)

No file I/O, no random generation, no visualization.
All coordinates and computations use float64.
"""

import math
from typing import Sequence, Union, List, Tuple
import numpy as np

from rind_dataset.constants import (
    OBSTACLE_RECTANGLE,
    OBSTACLE_ELLIPSE,
    OBSTACLE_TRIANGLE,
    OBSTACLE_POLYGON,
    MAX_POLYGON_VERTICES,
    PARAM_WIDTH,
)

# ---------------------------------------------------------------------------
# Backward Compatibility: Axis-Aligned Box
# ---------------------------------------------------------------------------

def point_in_box(
    x: float,
    y: float,
    box: Sequence[float],
) -> bool:
    """Check if continuous point (x, y) lies inside or on the closed box."""
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
    return bool(x_min <= x <= x_max and y_min <= y <= y_max)


def point_in_any_box(
    x: float,
    y: float,
    boxes: np.ndarray,
) -> bool:
    """Check if continuous point (x, y) lies inside or on any box in boxes."""
    for i in range(len(boxes)):
        if point_in_box(x, y, boxes[i]):
            return True
    return False


def is_driver_valid(
    driver: Sequence[float],
    boxes: np.ndarray,
    domain_size: float = 1024.0,
) -> bool:
    """Check if driver position is valid against axis-aligned boxes."""
    x, y = driver[0], driver[1]
    if not (0.0 <= x <= domain_size and 0.0 <= y <= domain_size):
        return False
    if point_in_any_box(x, y, boxes):
        return False
    return True


def segment_intersects_box(
    px: float,
    py: float,
    qx: float,
    qy: float,
    box: Sequence[float],
) -> bool:
    """Test if segment [p, q] intersects or touches the closed box."""
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

    dx = qx - px
    dy = qy - py

    if dx == 0.0:
        if px < x_min or px > x_max:
            return False
        t1x = -float("inf")
        t2x = float("inf")
    else:
        inv_dx = 1.0 / dx
        t_a = (x_min - px) * inv_dx
        t_b = (x_max - px) * inv_dx
        if t_a <= t_b:
            t1x = t_a
            t2x = t_b
        else:
            t1x = t_b
            t2x = t_a

    if dy == 0.0:
        if py < y_min or py > y_max:
            return False
        t1y = -float("inf")
        t2y = float("inf")
    else:
        inv_dy = 1.0 / dy
        t_a = (y_min - py) * inv_dy
        t_b = (y_max - py) * inv_dy
        if t_a <= t_b:
            t1y = t_a
            t2y = t_b
        else:
            t1y = t_b
            t2y = t_a

    t_enter = max(t1x, t1y)
    t_exit = min(t2x, t2y)

    if t_enter > t_exit:
        return False

    t_start = max(t_enter, 0.0)
    t_end = min(t_exit, 1.0)

    return t_start <= t_end


def segment_intersects_any_box(
    px: float,
    py: float,
    qx: float,
    qy: float,
    boxes: np.ndarray,
) -> bool:
    """Test if segment [p, q] intersects or touches any box in boxes."""
    for i in range(len(boxes)):
        if segment_intersects_box(px, py, qx, qy, boxes[i]):
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Rotated Rectangle
# ---------------------------------------------------------------------------

def point_in_rotated_rectangle(
    x: float,
    y: float,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    theta: float,
) -> bool:
    """Check if point (x, y) is inside or on rotated rectangle."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - cx
    dy = y - cy
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    return abs(u) <= hw and abs(v) <= hh


def segment_intersects_rotated_rectangle(
    px: float,
    py: float,
    qx: float,
    qy: float,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    theta: float,
) -> bool:
    """Test segment [p, q] vs rotated rectangle using local slab test."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    dpx = px - cx
    dpy = py - cy
    pu = dpx * cos_t + dpy * sin_t
    pv = -dpx * sin_t + dpy * cos_t

    dqx = qx - cx
    dqy = qy - cy
    qu = dqx * cos_t + dqy * sin_t
    qv = -dqx * sin_t + dqy * cos_t

    box = (-hw, -hh, hw, hh)
    return segment_intersects_box(pu, pv, qu, qv, box)


# ---------------------------------------------------------------------------
# 2. Ellipse
# ---------------------------------------------------------------------------

def point_in_ellipse(
    x: float,
    y: float,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    theta: float,
) -> bool:
    """Check if point (x, y) is inside or on rotated ellipse."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - cx
    dy = y - cy
    u = (dx * cos_t + dy * sin_t) / rx
    v = (-dx * sin_t + dy * cos_t) / ry
    return (u * u + v * v) <= 1.0


def segment_intersects_ellipse(
    px: float,
    py: float,
    qx: float,
    qy: float,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    theta: float,
) -> bool:
    """Test segment [p, q] vs rotated ellipse by transforming to unit circle."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    dpx = px - cx
    dpy = py - cy
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
        return (pu * pu + pv * pv) <= 1.0

    b = 2.0 * (pu * du + pv * dv)
    c = pu * pu + pv * pv - 1.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)
    inv_2a = 1.0 / (2.0 * a)
    t1 = (-b - sqrt_disc) * inv_2a
    t2 = (-b + sqrt_disc) * inv_2a

    t_start = max(t1, 0.0)
    t_end = min(t2, 1.0)
    return t_start <= t_end


# ---------------------------------------------------------------------------
# 3. Triangle & 2D Segment Primitives
# ---------------------------------------------------------------------------

def _cross_product_2d(
    ax: float, ay: float,
    bx: float, by: float,
    px: float, py: float,
) -> float:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def point_in_triangle(
    x: float,
    y: float,
    x1: float, y1: float,
    x2: float, y2: float,
    x3: float, y3: float,
) -> bool:
    """Check if point (x, y) is inside or on the closed triangle."""
    cp1 = _cross_product_2d(x1, y1, x2, y2, x, y)
    cp2 = _cross_product_2d(x2, y2, x3, y3, x, y)
    cp3 = _cross_product_2d(x3, y3, x1, y1, x, y)

    has_neg = (cp1 < 0.0) or (cp2 < 0.0) or (cp3 < 0.0)
    has_pos = (cp1 > 0.0) or (cp2 > 0.0) or (cp3 > 0.0)

    return not (has_neg and has_pos)


def _point_on_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> bool:
    """Test if point P lies on closed segment [A, B] given collinearity."""
    return (
        min(ax, bx) <= px <= max(ax, bx)
        and min(ay, by) <= py <= max(ay, by)
    )


def _segments_intersect_2d(
    p1x: float, p1y: float,
    p2x: float, p2y: float,
    p3x: float, p3y: float,
    p4x: float, p4y: float,
) -> bool:
    """Test if segment [P1, P2] intersects or touches segment [P3, P4]."""
    d1 = _cross_product_2d(p3x, p3y, p4x, p4y, p1x, p1y)
    d2 = _cross_product_2d(p3x, p3y, p4x, p4y, p2x, p2y)
    d3 = _cross_product_2d(p1x, p1y, p2x, p2y, p3x, p3y)
    d4 = _cross_product_2d(p1x, p1y, p2x, p2y, p4x, p4y)

    # Straddling test
    if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0))
        and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
        return True

    # Boundary and collinear contact
    if d1 == 0.0 and _point_on_segment(p1x, p1y, p3x, p3y, p4x, p4y):
        return True
    if d2 == 0.0 and _point_on_segment(p2x, p2y, p3x, p3y, p4x, p4y):
        return True
    if d3 == 0.0 and _point_on_segment(p3x, p3y, p1x, p1y, p2x, p2y):
        return True
    if d4 == 0.0 and _point_on_segment(p4x, p4y, p1x, p1y, p2x, p2y):
        return True

    return False


def segment_intersects_triangle(
    px: float,
    py: float,
    qx: float,
    qy: float,
    x1: float, y1: float,
    x2: float, y2: float,
    x3: float, y3: float,
) -> bool:
    """Test segment [p, q] vs closed triangle."""
    if point_in_triangle(px, py, x1, y1, x2, y2, x3, y3):
        return True
    if point_in_triangle(qx, qy, x1, y1, x2, y2, x3, y3):
        return True

    if _segments_intersect_2d(px, py, qx, qy, x1, y1, x2, y2):
        return True
    if _segments_intersect_2d(px, py, qx, qy, x2, y2, x3, y3):
        return True
    if _segments_intersect_2d(px, py, qx, qy, x3, y3, x1, y1):
        return True

    return False


# ---------------------------------------------------------------------------
# 4. Concave / Convex Simple Polygon
# ---------------------------------------------------------------------------

def point_in_polygon(
    x: float,
    y: float,
    vertices: Union[np.ndarray, Sequence[Sequence[float]]],
) -> bool:
    """Check if point (x, y) is inside or on the closed simple polygon.

    Uses ray-casting algorithm with explicit boundary/edge/vertex handling.
    """
    n = len(vertices)
    if n < 3:
        return False

    # 1. First check if point lies exactly on any edge or vertex
    for i in range(n):
        j = (i + 1) % n
        vix, viy = float(vertices[i][0]), float(vertices[i][1])
        vjx, vjy = float(vertices[j][0]), float(vertices[j][1])
        cp = _cross_product_2d(vix, viy, vjx, vjy, x, y)
        if cp == 0.0 and _point_on_segment(x, y, vix, viy, vjx, vjy):
            return True

    # 2. Ray casting in +x direction
    inside = False
    for i in range(n):
        j = (i + 1) % n
        vix, viy = float(vertices[i][0]), float(vertices[i][1])
        vjx, vjy = float(vertices[j][0]), float(vertices[j][1])

        # Test if horizontal ray from (x, y) crosses edge (V_i, V_j)
        if (viy > y) != (vjy > y):
            # Intersection x coordinate
            x_int = vix + (y - viy) * (vjx - vix) / (vjy - viy)
            if x < x_int:
                inside = not inside

    return inside


def segment_intersects_polygon(
    px: float,
    py: float,
    qx: float,
    qy: float,
    vertices: Union[np.ndarray, Sequence[Sequence[float]]],
) -> bool:
    """Test if segment [p, q] intersects or touches closed simple polygon.

    Blocked if:
    - either endpoint lies in or on the polygon, OR
    - segment intersects or touches any polygon edge.
    """
    n = len(vertices)
    if n < 3:
        return False

    # Check endpoints
    if point_in_polygon(px, py, vertices):
        return True
    if point_in_polygon(qx, qy, vertices):
        return True

    # Check edge intersections
    for i in range(n):
        j = (i + 1) % n
        vix, viy = float(vertices[i][0]), float(vertices[i][1])
        vjx, vjy = float(vertices[j][0]), float(vertices[j][1])
        if _segments_intersect_2d(px, py, qx, qy, vix, viy, vjx, vjy):
            return True

    return False


def is_polygon_simple(vertices: Union[np.ndarray, Sequence[Sequence[float]]]) -> bool:
    """Verify that polygon is simple (no non-adjacent edge intersections)."""
    n = len(vertices)
    if n < 3:
        return False

    for i in range(n):
        i_next = (i + 1) % n
        p1x, p1y = float(vertices[i][0]), float(vertices[i][1])
        p2x, p2y = float(vertices[i_next][0]), float(vertices[i_next][1])

        for j in range(i + 1, n):
            j_next = (j + 1) % n
            # Adjacent edges share a vertex, skip them
            if i == j or i_next == j or (i == 0 and j_next == 0):
                continue
            p3x, p3y = float(vertices[j][0]), float(vertices[j][1])
            p4x, p4y = float(vertices[j_next][0]), float(vertices[j_next][1])

            if _segments_intersect_2d(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return False
    return True


def is_polygon_concave(vertices: Union[np.ndarray, Sequence[Sequence[float]]]) -> bool:
    """Test whether polygon is concave (has at least one reflex angle)."""
    n = len(vertices)
    if n < 4:
        return False
    signs = []
    for i in range(n):
        prev = (i - 1 + n) % n
        nxt = (i + 1) % n
        ax, ay = float(vertices[prev][0]), float(vertices[prev][1])
        bx, by = float(vertices[i][0]), float(vertices[i][1])
        cx, cy = float(vertices[nxt][0]), float(vertices[nxt][1])
        cp = _cross_product_2d(ax, ay, bx, by, cx, cy)
        if cp > 1e-7:
            signs.append(1)
        elif cp < -1e-7:
            signs.append(-1)
    return (1 in signs) and (-1 in signs)


def extract_polygon_vertices(params: Sequence[float]) -> np.ndarray:
    """Extract (N, 2) vertex array from packed polygon params [N, x1, y1, ...]."""
    n = int(round(params[0]))
    verts = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        verts[i, 0] = params[1 + 2 * i]
        verts[i, 1] = params[2 + 2 * i]
    return verts


# ---------------------------------------------------------------------------
# 5. Generic Dispatch and Driver Validation
# ---------------------------------------------------------------------------

def point_in_obstacle(
    x: float,
    y: float,
    obs_type: int,
    params: Sequence[float],
) -> bool:
    """Check if point (x, y) is inside or on obstacle of given type."""
    if obs_type == OBSTACLE_RECTANGLE:
        return point_in_rotated_rectangle(
            x, y, params[0], params[1], params[2], params[3], params[4]
        )
    elif obs_type == OBSTACLE_ELLIPSE:
        return point_in_ellipse(
            x, y, params[0], params[1], params[2], params[3], params[4]
        )
    elif obs_type == OBSTACLE_TRIANGLE:
        return point_in_triangle(
            x, y, params[0], params[1], params[2], params[3], params[4], params[5]
        )
    elif obs_type == OBSTACLE_POLYGON:
        n = int(round(params[0]))
        verts = [(params[1 + 2 * i], params[2 + 2 * i]) for i in range(n)]
        return point_in_polygon(x, y, verts)
    return False


def segment_intersects_obstacle(
    px: float,
    py: float,
    qx: float,
    qy: float,
    obs_type: int,
    params: Sequence[float],
) -> bool:
    """Check if segment [p, q] intersects or touches obstacle of given type."""
    if obs_type == OBSTACLE_RECTANGLE:
        return segment_intersects_rotated_rectangle(
            px, py, qx, qy, params[0], params[1], params[2], params[3], params[4]
        )
    elif obs_type == OBSTACLE_ELLIPSE:
        return segment_intersects_ellipse(
            px, py, qx, qy, params[0], params[1], params[2], params[3], params[4]
        )
    elif obs_type == OBSTACLE_TRIANGLE:
        return segment_intersects_triangle(
            px, py, qx, qy, params[0], params[1], params[2], params[3], params[4], params[5]
        )
    elif obs_type == OBSTACLE_POLYGON:
        n = int(round(params[0]))
        verts = [(params[1 + 2 * i], params[2 + 2 * i]) for i in range(n)]
        return segment_intersects_polygon(px, py, qx, qy, verts)
    return False


def point_in_any_obstacle(
    x: float,
    y: float,
    obs_types: np.ndarray,
    obs_params: np.ndarray,
    count: int,
) -> bool:
    """Check if point (x, y) is inside or on any of the active obstacles."""
    for i in range(count):
        if point_in_obstacle(x, y, int(obs_types[i]), obs_params[i]):
            return True
    return False


def segment_intersects_any_obstacle(
    px: float,
    py: float,
    qx: float,
    qy: float,
    obs_types: np.ndarray,
    obs_params: np.ndarray,
    count: int,
) -> bool:
    """Check if segment [p, q] intersects or touches any active obstacle."""
    for i in range(count):
        if segment_intersects_obstacle(px, py, qx, qy, int(obs_types[i]), obs_params[i]):
            return True
    return False


def is_driver_valid_mixed(
    driver: Sequence[float],
    obs_types: np.ndarray,
    obs_params: np.ndarray,
    count: int,
    domain_size: float = 1024.0,
) -> bool:
    """Check if driver position is valid against mixed obstacles."""
    x, y = driver[0], driver[1]
    if not (0.0 <= x <= domain_size and 0.0 <= y <= domain_size):
        return False
    if point_in_any_obstacle(x, y, obs_types, obs_params, count):
        return False
    return True


def compute_obstacle_aabb(
    obs_type: int,
    params: Sequence[float],
) -> np.ndarray:
    """Compute axis-aligned bounding box [xmin, ymin, xmax, ymax] for obstacle."""
    if obs_type == OBSTACLE_RECTANGLE:
        cx, cy, hw, hh, theta = params[0], params[1], params[2], params[3], params[4]
        cos_t = abs(math.cos(theta))
        sin_t = abs(math.sin(theta))
        x_ext = hw * cos_t + hh * sin_t
        y_ext = hw * sin_t + hh * cos_t
        return np.array([cx - x_ext, cy - y_ext, cx + x_ext, cy + y_ext], dtype=np.float64)

    elif obs_type == OBSTACLE_ELLIPSE:
        cx, cy, rx, ry, theta = params[0], params[1], params[2], params[3], params[4]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_ext = math.sqrt((rx * cos_t) ** 2 + (ry * sin_t) ** 2)
        y_ext = math.sqrt((rx * sin_t) ** 2 + (ry * cos_t) ** 2)
        return np.array([cx - x_ext, cy - y_ext, cx + x_ext, cy + y_ext], dtype=np.float64)

    elif obs_type == OBSTACLE_TRIANGLE:
        x1, y1, x2, y2, x3, y3 = params[0], params[1], params[2], params[3], params[4], params[5]
        return np.array([
            min(x1, x2, x3),
            min(y1, y2, y3),
            max(x1, x2, x3),
            max(y1, y2, y3),
        ], dtype=np.float64)

    elif obs_type == OBSTACLE_POLYGON:
        n = int(round(params[0]))
        xs = [params[1 + 2 * i] for i in range(n)]
        ys = [params[2 + 2 * i] for i in range(n)]
        return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float64)

    return np.zeros(4, dtype=np.float64)
