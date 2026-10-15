"""Batched Torch visibility for the exact RIND closed-obstacle geometry."""
from __future__ import annotations

import math
import numpy as np

from rind_dataset.constants import (
    OBSTACLE_RECTANGLE,
    OBSTACLE_ELLIPSE,
    OBSTACLE_TRIANGLE,
    OBSTACLE_POLYGON,
)
from rind_dataset.geometry import compute_obstacle_aabb


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _slab_hit(pu, pv, qu, qv, half_u, half_v, torch):
    du, dv = qu - pu, qv - pv
    parallel_u, parallel_v = du == 0.0, dv == 0.0
    valid_u = (~parallel_u) | ((pu >= -half_u) & (pu <= half_u))
    valid_v = (~parallel_v) | ((pv >= -half_v) & (pv <= half_v))
    safe_du = torch.where(parallel_u, torch.ones_like(du), du)
    safe_dv = torch.where(parallel_v, torch.ones_like(dv), dv)
    ua, ub = (-half_u - pu) / safe_du, (half_u - pu) / safe_du
    va, vb = (-half_v - pv) / safe_dv, (half_v - pv) / safe_dv
    neg_inf = torch.full_like(du, -float("inf"))
    pos_inf = torch.full_like(du, float("inf"))
    u0 = torch.where(parallel_u, neg_inf, torch.minimum(ua, ub))
    u1 = torch.where(parallel_u, pos_inf, torch.maximum(ua, ub))
    v0 = torch.where(parallel_v, neg_inf, torch.minimum(va, vb))
    v1 = torch.where(parallel_v, pos_inf, torch.maximum(va, vb))
    enter = torch.maximum(torch.maximum(u0, v0), torch.zeros_like(du))
    leave = torch.minimum(torch.minimum(u1, v1), torch.ones_like(du))
    return valid_u & valid_v & (enter <= leave)


def _ellipse_hit(pu, pv, qu, qv, torch):
    du, dv = qu - pu, qv - pv
    a = du * du + dv * dv
    stationary = a == 0.0
    safe_a = torch.where(stationary, torch.ones_like(a), a)
    b = 2.0 * (pu * du + pv * dv)
    c = pu * pu + pv * pv - 1.0
    disc = b * b - 4.0 * safe_a * c
    real = disc >= 0.0
    root = torch.sqrt(torch.clamp_min(disc, 0.0))
    t1 = (-b - root) / (2.0 * safe_a)
    t2 = (-b + root) / (2.0 * safe_a)
    enter = torch.maximum(torch.minimum(t1, t2), torch.zeros_like(a))
    leave = torch.minimum(torch.maximum(t1, t2), torch.ones_like(a))
    return (stationary & ((pu * pu + pv * pv) <= 1.0)) | (
        (~stationary) & real & (enter <= leave)
    )


def _edge_hit(sx, sy, qx, qy, ax, ay, bx, by, torch):
    d1 = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)
    d2 = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
    d3 = (qx - sx) * (ay - sy) - (qy - sy) * (ax - sx)
    d4 = (qx - sx) * (by - sy) - (qy - sy) * (bx - sx)
    strict = (((d1 > 0.0) & (d2 < 0.0)) | ((d1 < 0.0) & (d2 > 0.0))) & (
        ((d3 > 0.0) & (d4 < 0.0)) | ((d3 < 0.0) & (d4 > 0.0))
    )
    source_on = (d1 == 0.0) & (sx >= min(ax, bx)) & (sx <= max(ax, bx)) & (
        sy >= min(ay, by)) & (sy <= max(ay, by)
    )
    target_on = (d2 == 0.0) & (qx >= min(ax, bx)) & (qx <= max(ax, bx)) & (
        qy >= min(ay, by)) & (qy <= max(ay, by)
    )
    seg_min_x, seg_max_x = torch.minimum(sx, qx), torch.maximum(sx, qx)
    seg_min_y, seg_max_y = torch.minimum(sy, qy), torch.maximum(sy, qy)
    a_on = (d3 == 0.0) & (ax >= seg_min_x) & (ax <= seg_max_x) & (
        ay >= seg_min_y) & (ay <= seg_max_y)
    b_on = (d4 == 0.0) & (bx >= seg_min_x) & (bx <= seg_max_x) & (
        by >= seg_min_y) & (by <= seg_max_y)
    return strict | source_on | target_on | a_on | b_on


class TorchVisibilityRenderer:
    """Render unit-source visibility in candidate batches on CUDA or CPU.

    Computation uses float64 and the same closed-boundary tests as the canonical
    reference/Numba renderer. Obstacle pixels are supplied by the observation
    and are always returned as invisible.
    """

    def __init__(self, window, obstacle, obstacle_types, obstacle_params, *,
                 device="cuda", pixel_chunk=4096):
        import torch
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA teacher renderer requested, but torch.cuda.is_available() is false")
        if not isinstance(pixel_chunk, int) or pixel_chunk < 1:
            raise ValueError("gpu_pixel_chunk must be a positive integer")
        self.torch, self.device, self.pixel_chunk = torch, requested_device, pixel_chunk
        self.window = tuple(map(int, window))
        obstacle = np.asarray(obstacle, dtype=bool)
        side = self.window[2]
        if obstacle.shape != (side, side):
            raise ValueError("obstacle mask must match the square observation window")
        self.shape = obstacle.shape
        flat = np.flatnonzero(~obstacle.ravel())
        yy, xx = np.divmod(flat, side)
        coords = np.column_stack((self.window[0] + xx + 0.5,
                                  self.window[1] + yy + 0.5)).astype(np.float64)
        self.free_flat = torch.as_tensor(flat, dtype=torch.long, device=self.device)
        self.pixels = torch.as_tensor(coords, dtype=torch.float64, device=self.device)
        types = np.asarray(obstacle_types, dtype=np.uint8)
        params = np.asarray(obstacle_params, dtype=np.float64)
        self.shapes = []
        for obstacle_type, row in zip(types, params):
            obstacle_type = int(obstacle_type)
            if obstacle_type in (OBSTACLE_RECTANGLE, OBSTACLE_ELLIPSE):
                cx, cy, ru, rv, theta = map(float, row[:5])
                self.shapes.append((obstacle_type, cx, cy, ru, rv,
                                    math.cos(theta), math.sin(theta)))
            elif obstacle_type == OBSTACLE_TRIANGLE:
                values = list(map(float, row[:6]))
                self.shapes.append((obstacle_type,
                                    [(values[0], values[1]), (values[2], values[3]),
                                     (values[4], values[5])]))
            elif obstacle_type == OBSTACLE_POLYGON:
                n = int(round(float(row[0])))
                self.shapes.append((obstacle_type,
                                    [(float(row[1 + 2 * i]), float(row[2 + 2 * i]))
                                     for i in range(n)]))
            else:
                raise ValueError(f"unknown obstacle type: {obstacle_type}")
        aabbs = np.array([compute_obstacle_aabb(int(t), p)
                          for t, p in zip(types, params)], dtype=np.float64).reshape(-1, 4)
        self.aabbs = torch.as_tensor(aabbs, dtype=torch.float64, device=self.device)

    def positions_valid_tensor(self, positions):
        """Return the closed-obstacle exclusion mask entirely on this device."""
        torch = self.torch
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("positions must be finite [N,2]")
        xy = torch.as_tensor(points, dtype=torch.float64, device=self.device)
        x, y = xy[:, 0], xy[:, 1]
        inside_any = torch.zeros(len(points), dtype=torch.bool, device=self.device)
        for shape in self.shapes:
            obstacle_type = shape[0]
            if obstacle_type in (OBSTACLE_RECTANGLE, OBSTACLE_ELLIPSE):
                _, cx, cy, ru, rv, cos_t, sin_t = shape
                dx, dy = x - cx, y - cy
                u = dx * cos_t + dy * sin_t
                v = -dx * sin_t + dy * cos_t
                if obstacle_type == OBSTACLE_RECTANGLE:
                    inside = (u.abs() <= ru) & (v.abs() <= rv)
                else:
                    inside = (u / ru).square() + (v / rv).square() <= 1.0
            elif obstacle_type == OBSTACLE_TRIANGLE:
                vertices = shape[1]
                crosses = []
                for i in range(3):
                    ax, ay = vertices[i];bx, by = vertices[(i + 1) % 3]
                    crosses.append((bx - ax) * (y - ay) - (by - ay) * (x - ax))
                has_negative = crosses[0] < 0
                has_positive = crosses[0] > 0
                for cross in crosses[1:]:
                    has_negative |= cross < 0;has_positive |= cross > 0
                inside = ~(has_negative & has_positive)
            elif obstacle_type == OBSTACLE_POLYGON:
                vertices = shape[1]
                inside = torch.zeros(len(points), dtype=torch.bool, device=self.device)
                boundary = torch.zeros_like(inside)
                for i in range(len(vertices)):
                    ax, ay = vertices[i];bx, by = vertices[(i + 1) % len(vertices)]
                    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
                    boundary |= ((cross == 0.0) & (x >= min(ax, bx)) &
                                 (x <= max(ax, bx)) & (y >= min(ay, by)) &
                                 (y <= max(ay, by)))
                    crosses_ray = ((ay > y) != (by > y))
                    safe_denominator = by - ay if by != ay else 1.0
                    x_intersection = ax + (y - ay) * (bx - ax) / safe_denominator
                    inside ^= crosses_ray & (x < x_intersection)
                inside |= boundary
            else:
                raise ValueError(f"unknown obstacle type: {obstacle_type}")
            inside_any |= inside
        return ~inside_any

    def _render_pixel_block(self, sources, pixels):
        torch = self.torch
        sx, sy = sources[:, 0:1], sources[:, 1:2]
        qx, qy = pixels[:, 0][None, :], pixels[:, 1][None, :]
        seg_min_x, seg_max_x = torch.minimum(sx, qx), torch.maximum(sx, qx)
        seg_min_y, seg_max_y = torch.minimum(sy, qy), torch.maximum(sy, qy)
        blocked = torch.zeros((len(sources), len(pixels)), dtype=torch.bool, device=self.device)
        for k, shape in enumerate(self.shapes):
            obstacle_type = shape[0]
            box = self.aabbs[k]
            overlap = ((seg_max_x >= box[0]) & (seg_min_x <= box[2]) &
                       (seg_max_y >= box[1]) & (seg_min_y <= box[3]))
            if obstacle_type in (OBSTACLE_RECTANGLE, OBSTACLE_ELLIPSE):
                _, cx, cy, ru, rv, cos_t, sin_t = shape
                dsx, dsy = sx - cx, sy - cy
                dqx, dqy = qx - cx, qy - cy
                pu = dsx * cos_t + dsy * sin_t
                pv = -dsx * sin_t + dsy * cos_t
                qu = dqx * cos_t + dqy * sin_t
                qv = -dqx * sin_t + dqy * cos_t
                if obstacle_type == OBSTACLE_RECTANGLE:
                    hit = _slab_hit(pu, pv, qu, qv, ru, rv, torch)
                else:
                    hit = _ellipse_hit(pu / ru, pv / rv, qu / ru, qv / rv, torch)
            elif obstacle_type == OBSTACLE_TRIANGLE:
                vertices = shape[1]
                hit = torch.zeros_like(blocked)
                for i in range(3):
                    hit |= _edge_hit(sx, sy, qx, qy, *vertices[i], *vertices[(i + 1) % 3], torch)
            elif obstacle_type == OBSTACLE_POLYGON:
                vertices = shape[1]
                hit = torch.zeros_like(blocked)
                for i in range(len(vertices)):
                    hit |= _edge_hit(sx, sy, qx, qy, *vertices[i], *vertices[(i + 1) % len(vertices)], torch)
            else:
                raise ValueError(f"unknown obstacle type: {obstacle_type}")
            blocked |= overlap & hit
        return ~blocked

    def render_tensor(self, positions):
        """Return a boolean ``[candidate,pixel_y,pixel_x]`` tensor on this device."""
        torch = self.torch
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("positions must be finite [N,2]")
        sources = torch.as_tensor(points, dtype=torch.float64, device=self.device)
        result = torch.zeros((len(points), self.shape[0] * self.shape[1]),
                             dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            for start in range(0, len(self.pixels), self.pixel_chunk):
                stop = min(start + self.pixel_chunk, len(self.pixels))
                result[:, self.free_flat[start:stop]] = self._render_pixel_block(
                    sources, self.pixels[start:stop])
        return result.reshape(len(points), *self.shape)

    def render(self, positions):
        return self.render_tensor(positions).cpu().numpy()
