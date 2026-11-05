"""Observation-only reference policies, not trained information-gain models."""
from __future__ import annotations

from collections import deque
import numpy as np

from .observation_env import STOP, geometric_actions


def _zero_cost_escape(state, limit=32768):
    """Reposition through cached windows if no immediate new acquisition fits.

    Breadth-first search uses coverage/coordinates only, with a finite cap. A
    return of STOP may reflect search exhaustion, not a proof of optimal stopping.
    """
    queue = deque([(state.window, None)])
    seen = {state.window}
    while queue and len(seen) <= limit:
        window, first = queue.popleft()
        for action in geometric_actions(window, state.world_size, state.sizes):
            cost = state.new_pixels(action.window)
            if 0 < cost <= state.remaining and first is not None:
                return first
            if cost == 0 and action.window not in seen:
                seen.add(action.window)
                queue.append((action.window, first or action))
    return STOP


def _continuation_points(state):
    """Extrapolate observed response-gradient tangents, without unseen pixels.

    This is an inexpensive boundary-shape proxy. It cannot distinguish a shadow
    boundary from an obstacle edge and is not an expected information gain.
    """
    w = state.window
    crop = state.response[w.slices]
    gy, gx = np.gradient(crop)
    magnitude = np.hypot(gx, gy)
    ys, xs = np.nonzero(magnitude > max(1e-10, float(crop.max()) * 1e-6))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    selected = np.linspace(0, len(xs) - 1, min(128, len(xs)), dtype=int)
    ys, xs = ys[selected], xs[selected]
    directions = np.column_stack((gy[ys, xs], -gx[ys, xs])) / magnitude[ys, xs, None]
    origin = np.column_stack((xs + w.x + .5, ys + w.y + .5))
    distances = np.r_[np.linspace(-2*w.size, -1, 32), np.linspace(1, 2*w.size, 32)]
    points = np.floor(origin[:, None, :] + directions[:, None, :] * distances[None, :, None])
    points = np.unique(points.astype(np.int64).reshape(-1, 2), axis=0)
    good = ((points >= 0) & (points < state.world_size)).all(axis=1)
    points = points[good]
    return points[~state.coverage[points[:, 1], points[:, 0]]]


def choose_action(state, actions, *, policy, rng):
    """Shared interface for future learned policies: state and legal actions only."""
    if policy not in ("random", "grow", "boundary"):
        raise ValueError("unknown policy; expected random, grow, or boundary")
    novel = [a for a in actions if a != STOP and state.new_pixels(a.window) > 0]
    if not novel:
        return _zero_cost_escape(state)
    if policy == "random":
        return novel[int(rng.integers(len(novel)))]
    if policy == "grow":
        growth = [a for a in novel if a.name.startswith("grow")]
        # Default centered growth when affordable, then anchored growth, then moves.
        return growth[0] if growth else novel[0]
    points = _continuation_points(state)
    def score(action):
        w = action.window
        hits = int(((points[:, 0] >= w.x) & (points[:, 0] < w.x + w.size)
                    & (points[:, 1] >= w.y) & (points[:, 1] < w.y + w.size)).sum())
        cost = state.new_pixels(w)
        # Cost-aware geometry proxy; .01 favors smaller queries on no-edge ties.
        return (hits + .01) / cost
    return max(novel, key=score)
