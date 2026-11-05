"""Response-only acquisition with exact union-of-pixels accounting.

The oracle is an acquisition device, not a policy input.  Public snapshots
contain observed values only; neither geometry nor source metadata is exposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, order=True)
class Window:
    x: int
    y: int
    size: int

    @property
    def area(self):
        return self.size * self.size

    @property
    def slices(self):
        return np.s_[self.y:self.y + self.size, self.x:self.x + self.size]


@dataclass(frozen=True)
class Action:
    name: str
    window: Window | None


STOP = Action("stop", None)


def geometric_actions(window: Window, world: int, sizes: tuple[int, ...]):
    """Half-side translations and doubled squares containing the old square.

    No clipping: invalid windows are absent. No obstacle/source-based vetoes.
    Same windows produced by different growth anchors are deduplicated.
    """
    x, y, side = window.x, window.y, window.size
    shift = side // 2
    proposals = [Action(name, Window(x + dx, y + dy, side)) for name, dx, dy in
                 (("left", -shift, 0), ("right", shift, 0),
                  ("up", 0, -shift), ("down", 0, shift))]
    if 2 * side in sizes:
        proposals.extend(Action(name, Window(gx, gy, 2 * side)) for name, gx, gy in (
            ("grow_center", x - side // 2, y - side // 2),
            ("grow_se", x, y), ("grow_sw", x - side, y),
            ("grow_ne", x, y - side), ("grow_nw", x - side, y - side)))
    seen = set()
    result = []
    for action in proposals:
        w = action.window
        if (w not in seen and w.x >= 0 and w.y >= 0
                and w.x + w.size <= world and w.y + w.size <= world):
            result.append(action)
            seen.add(w)
    return tuple(result)


@dataclass(frozen=True)
class StepRecord:
    step: int
    action: str
    window: Window | None
    new_pixels: int
    spent_pixels: int
    oracle_called: bool

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ObservationState:
    response: np.ndarray       # world [y,x], NaN outside coverage
    coverage: np.ndarray       # world [y,x], bool
    window: Window
    sizes: tuple[int, ...]
    budget: int
    spent: int
    steps: int                 # transitions after the initial observation
    max_steps: int
    terminated: bool
    termination_reason: str | None
    history: tuple[StepRecord, ...]

    @property
    def remaining(self):
        return self.budget - self.spent

    @property
    def world_size(self):
        return self.coverage.shape[0]

    def new_pixels(self, window: Window):
        return window.area - int(np.count_nonzero(self.coverage[window.slices]))


class BudgetedWindowEnv:
    """One static-scene episode; call reset once, then step until terminated.

    Budget includes the initial square, all pixels including obstacle interiors,
    and all-dark/all-positive windows. A repeated ROI uses the local cache.
    Public snapshots are copies and cannot expose later observations by aliasing.
    """
    def __init__(self, oracle: Callable[[Window], np.ndarray], *, world_size=1024,
                 budget=65536, sizes=(16, 32, 64, 128, 256, 512, 1024),
                 max_steps=128, overlap_atol=1e-10):
        for name, value in (("world_size", world_size), ("budget", budget),
                            ("max_steps", max_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if budget > world_size ** 2:
            raise ValueError("budget cannot exceed the total world area")
        if (not sizes or any(type(s) is not int or s < 2 or s > world_size
                             or s & (s - 1) for s in sizes)
                or tuple(sizes) != tuple(sorted(set(sizes)))):
            raise ValueError("sizes must be increasing distinct powers of two inside the world")
        if not np.isfinite(overlap_atol) or overlap_atol < 0:
            raise ValueError("overlap_atol must be finite and nonnegative")
        self._oracle = oracle
        self.world_size, self.budget, self.sizes = world_size, budget, tuple(sizes)
        self.max_steps, self.overlap_atol = max_steps, float(overlap_atol)
        self._response = np.full((world_size, world_size), np.nan, dtype=np.float64)
        self._coverage = np.zeros((world_size, world_size), dtype=bool)
        self._window = None
        self._history = []
        self._spent = self._steps = 0
        self._reason = None

    def _validate_window(self, w):
        if (not isinstance(w, Window) or any(type(v) is not int for v in (w.x, w.y, w.size))
                or w.size not in self.sizes or min(w.x, w.y) < 0
                or w.x + w.size > self.world_size or w.y + w.size > self.world_size):
            raise ValueError("window must be an allowed integer square inside the world")

    def _acquire(self, action, step):
        w = action.window
        self._validate_window(w)
        covered = self._coverage[w.slices]
        cost = w.area - int(np.count_nonzero(covered))
        if cost > self.budget - self._spent:
            raise ValueError("action exceeds the remaining unique-pixel budget")
        called = cost > 0
        if called:
            # Validate every return before committing any state/budget changes.
            response = np.array(self._oracle(w), dtype=np.float64, copy=True)
            if (response.shape != (w.size, w.size) or not np.isfinite(response).all()
                    or np.any(response < 0)):
                raise ValueError("oracle must return finite nonnegative response of the requested shape")
            if covered.any() and not np.allclose(response[covered],
                    self._response[w.slices][covered], rtol=0, atol=self.overlap_atol):
                raise ValueError("overlap changed: the episode requires a static response field")
            # Preserve previously observed values exactly, including tiny accepted roundoff.
            self._response[w.slices][~covered] = response[~covered]
            self._coverage[w.slices] = True
        self._spent += cost
        self._window = w
        self._steps = step
        self._history.append(StepRecord(step, action.name, w, cost, self._spent, called))
        if self._spent == self.budget:
            self._reason = "budget_exhausted"
        elif step >= self.max_steps:
            self._reason = "max_steps"

    def reset(self, window):
        if self._window is not None:
            raise RuntimeError("create a new environment for a new episode")
        self._acquire(Action("initial", window), 0)
        return self.snapshot()

    def legal_actions(self):
        if self._window is None:
            raise RuntimeError("reset first")
        if self._reason is not None:
            return ()
        actions = geometric_actions(self._window, self.world_size, self.sizes)
        remaining = self.budget - self._spent
        return (STOP,) + tuple(a for a in actions if
            a.window.area - int(np.count_nonzero(self._coverage[a.window.slices])) <= remaining)

    def step(self, action):
        if self._reason is not None:
            raise RuntimeError("episode already terminated")
        if action not in self.legal_actions():
            raise ValueError("action must be one of legal_actions()")
        if action == STOP:
            self._steps += 1
            self._reason = "policy_stop"
            self._history.append(StepRecord(self._steps, "stop", None, 0, self._spent, False))
        else:
            self._acquire(action, self._steps + 1)
        return self.snapshot()

    def snapshot(self):
        if self._window is None:
            raise RuntimeError("reset first")
        response, coverage = self._response.copy(), self._coverage.copy()
        response.flags.writeable = coverage.flags.writeable = False
        return ObservationState(response, coverage, self._window, self.sizes,
            self.budget, self._spent, self._steps, self.max_steps, self._reason is not None,
            self._reason, tuple(self._history))
