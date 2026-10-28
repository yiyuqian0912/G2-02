"""Explicit JIT geometry for free-pixel adjacency, matching canonical RIND.

Use a closed, statically decorated call graph rather than registering Python
functions from another module at import time. Scalar float64 operations and
closed-boundary conventions match rind_dataset.geometry; no fastmath is used.
The packed polygon representation avoids list typing across Numba versions.
"""
import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function


@njit(cache=True)
def _cross(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@njit(cache=True)
def _on_segment(px, py, ax, ay, bx, by):
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


@njit(cache=True)
def _segments(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    d1 = _cross(p3x, p3y, p4x, p4y, p1x, p1y)
    d2 = _cross(p3x, p3y, p4x, p4y, p2x, p2y)
    d3 = _cross(p1x, p1y, p2x, p2y, p3x, p3y)
    d4 = _cross(p1x, p1y, p2x, p2y, p4x, p4y)
    if (((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and
            ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0))):
        return True
    if d1 == 0.0 and _on_segment(p1x, p1y, p3x, p3y, p4x, p4y):
        return True
    if d2 == 0.0 and _on_segment(p2x, p2y, p3x, p3y, p4x, p4y):
        return True
    if d3 == 0.0 and _on_segment(p3x, p3y, p1x, p1y, p2x, p2y):
        return True
    if d4 == 0.0 and _on_segment(p4x, p4y, p1x, p1y, p2x, p2y):
        return True
    return False


@njit(cache=True)
def _box(px, py, qx, qy, xmin, ymin, xmax, ymax):
    dx, dy = qx - px, qy - py
    if dx == 0.0:
        if px < xmin or px > xmax:
            return False
        tx1, tx2 = -np.inf, np.inf
    else:
        inv = 1.0 / dx
        a, b = (xmin - px) * inv, (xmax - px) * inv
        tx1, tx2 = min(a, b), max(a, b)
    if dy == 0.0:
        if py < ymin or py > ymax:
            return False
        ty1, ty2 = -np.inf, np.inf
    else:
        inv = 1.0 / dy
        a, b = (ymin - py) * inv, (ymax - py) * inv
        ty1, ty2 = min(a, b), max(a, b)
    enter, leave = max(tx1, ty1), min(tx2, ty2)
    return enter <= leave and max(enter, 0.0) <= min(leave, 1.0)


@njit(cache=True)
def _triangle_point(x, y, x1, y1, x2, y2, x3, y3):
    a = _cross(x1, y1, x2, y2, x, y)
    b = _cross(x2, y2, x3, y3, x, y)
    c = _cross(x3, y3, x1, y1, x, y)
    return not ((a < 0.0 or b < 0.0 or c < 0.0) and (a > 0.0 or b > 0.0 or c > 0.0))


@njit(cache=True)
def _polygon_point(x, y, params, n):
    if n < 3:
        return False
    for i in range(n):
        j = (i + 1) % n
        ax, ay = params[1 + 2 * i], params[2 + 2 * i]
        bx, by = params[1 + 2 * j], params[2 + 2 * j]
        if _cross(ax, ay, bx, by, x, y) == 0.0 and _on_segment(x, y, ax, ay, bx, by):
            return True
    inside = False
    for i in range(n):
        j = (i + 1) % n
        ax, ay = params[1 + 2 * i], params[2 + 2 * i]
        bx, by = params[1 + 2 * j], params[2 + 2 * j]
        if (ay > y) != (by > y):
            crossing = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < crossing:
                inside = not inside
    return inside


@njit(cache=True)
def segment_intersects_obstacle(px, py, qx, qy, kind, params):
    if kind == 0:  # Rotated rectangle: canonical local slab calculation.
        cx, cy, hw, hh, theta = params[0], params[1], params[2], params[3], params[4]
        co, si = math.cos(theta), math.sin(theta)
        dx, dy = px - cx, py - cy
        pu, pv = dx * co + dy * si, -dx * si + dy * co
        dx, dy = qx - cx, qy - cy
        qu, qv = dx * co + dy * si, -dx * si + dy * co
        return _box(pu, pv, qu, qv, -hw, -hh, hw, hh)
    if kind == 1:  # Ellipse: canonical quadratic intersection.
        cx, cy, rx, ry, theta = params[0], params[1], params[2], params[3], params[4]
        co, si = math.cos(theta), math.sin(theta)
        dx, dy = px - cx, py - cy
        pu, pv = (dx * co + dy * si) / rx, (-dx * si + dy * co) / ry
        dx, dy = qx - cx, qy - cy
        qu, qv = (dx * co + dy * si) / rx, (-dx * si + dy * co) / ry
        du, dv = qu - pu, qv - pv
        a = du * du + dv * dv
        if a == 0.0:
            return pu * pu + pv * pv <= 1.0
        b, c = 2.0 * (pu * du + pv * dv), pu * pu + pv * pv - 1.0
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return False
        root, inverse = math.sqrt(discriminant), 1.0 / (2.0 * a)
        t1, t2 = (-b - root) * inverse, (-b + root) * inverse
        return max(t1, 0.0) <= min(t2, 1.0)
    if kind == 2:  # Closed triangle.
        x1, y1, x2, y2, x3, y3 = params[0], params[1], params[2], params[3], params[4], params[5]
        if (_triangle_point(px, py, x1, y1, x2, y2, x3, y3) or
                _triangle_point(qx, qy, x1, y1, x2, y2, x3, y3)):
            return True
        return (_segments(px, py, qx, qy, x1, y1, x2, y2) or
                _segments(px, py, qx, qy, x2, y2, x3, y3) or
                _segments(px, py, qx, qy, x3, y3, x1, y1))
    if kind == 3:  # Packed closed polygon, including concave shapes.
        n = int(round(params[0]))
        if n < 3:
            return False
        if _polygon_point(px, py, params, n) or _polygon_point(qx, qy, params, n):
            return True
        for i in range(n):
            j = (i + 1) % n
            if _segments(px, py, qx, qy, params[1 + 2 * i], params[2 + 2 * i],
                         params[1 + 2 * j], params[2 + 2 * j]):
                return True
    return False
