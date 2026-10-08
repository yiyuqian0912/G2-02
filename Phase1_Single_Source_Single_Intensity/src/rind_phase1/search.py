"""Explicit candidate domains and deterministic complete uniform search.

No adaptive pruning is implemented. The evaluator/cache and incomplete-result
contract allow later search traces to remain separate from final support.
"""

from dataclasses import dataclass
import math
import time

import numpy as np

from rind_phase1.physics import CandidateEvaluator


@dataclass(frozen=True)
class CandidateDomain:
    window: tuple
    world_size: int
    quadtree_prior: bool = False

    def __post_init__(self):
        values = np.asarray(self.window)
        if (values.shape != (3,) or values.dtype.kind not in "iu"
                or not isinstance(self.world_size, (int, np.integer)) or self.world_size <= 0
                or type(self.quadtree_prior) is not bool):
            raise ValueError("Domain requires an integer (x,y,L), positive world size, and boolean prior")
        x, y, side = map(int, values)
        if x < 0 or y < 0 or side <= 0 or x + side > self.world_size or y + side > self.world_size:
            raise ValueError("Observation window must be inside the world")
        if self.quadtree_prior and (side & (side - 1) or self.world_size & (self.world_size - 1)
                                    or self.world_size % (2 * side) or x % side or y % side):
            raise ValueError("Parent prior requires a valid aligned quadtree child window")
        object.__setattr__(self, "window", (x, y, side))
        object.__setattr__(self, "world_size", int(self.world_size))

    @property
    def bounds(self):
        x, y, side = self.window
        if not self.quadtree_prior:
            return (0, 0, self.world_size, self.world_size)
        parent_size = 2 * side
        px, py = (x // parent_size) * parent_size, (y // parent_size) * parent_size
        return (px, py, px + parent_size, py + parent_size)

    @property
    def name(self):
        return "quadtree_parent_minus_window" if self.quadtree_prior else "world_minus_window"

    def _inside(self, xy, bounds):
        x0, y0, x1, y1 = bounds
        right = xy[..., 0] <= x1 if x1 == self.world_size else xy[..., 0] < x1
        bottom = xy[..., 1] <= y1 if y1 == self.world_size else xy[..., 1] < y1
        return (xy[..., 0] >= x0) & (xy[..., 1] >= y0) & right & bottom

    def contains(self, candidate_xy):
        """Split edges belong right/down; the terminal world edge is included."""
        xy = np.asarray(candidate_xy, dtype=np.float64)
        if xy.shape[-1:] != (2,):
            raise ValueError("Candidate coordinates must end in a dimension of size 2")
        x, y, side = self.window
        return (np.isfinite(xy).all(axis=-1) & self._inside(xy, self.bounds)
                & ~self._inside(xy, (x, y, x + side, y + side)))

    def metadata(self):
        return {"candidate_domain": self.name, "quadtree_prior": self.quadtree_prior,
                "source_location_scope": "outside_observation_window",
                "world_size": self.world_size, "window": list(self.window),
                "bounds": list(self.bounds), "boundary_rule": "half-open; terminal world edges included"}


def deduplicate_candidates(candidate_xy):
    """Remove exact duplicate coordinates, preserving first-occurrence order."""
    xy = np.asarray(candidate_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("candidate_xy must be finite [N,2]")
    _, first = np.unique(xy, axis=0, return_index=True)
    return xy[np.sort(first)].copy()


@dataclass(frozen=True)
class UniformGrid:
    domain: CandidateDomain
    spacing: float
    candidate_xy: np.ndarray
    grid_index: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray

    @property
    def shape(self):
        return (len(self.y_axis), len(self.x_axis))

    def as_map(self, values, *, fill=np.nan):
        values = np.asarray(values)
        if values.shape != (len(self.candidate_xy),):
            raise ValueError("Map values must follow the candidate order")
        raster = np.full(self.shape, fill, dtype=np.float64)
        raster.flat[self.grid_index] = values
        return raster

    def metadata(self):
        return {**self.domain.metadata(), "candidate_spacing": self.spacing,
                "grid_origin": [0.0, 0.0], "grid_phase": "half_spacing_cell_centers",
                "ordering": "row-major: y ascending, then x ascending",
                "sampling_measure": "equal_mass_per_valid_uniform_grid_point",
                "target_semantics": ("quadtree_prior_conditioned_discrete_compatibility"
                                     if self.domain.quadtree_prior else "fixed_window_world_discrete_compatibility"),
                "reference_positions_in_support": False}


def uniform_grid(domain, spacing):
    """Globally anchored points ((k+.5)*spacing), filtered by the explicit domain.

    A common anchor makes prior-on and prior-off supports directly comparable.
    Spacing is in world units and need not equal the response sampling step.
    """
    if not np.isscalar(spacing) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("candidate spacing must be finite and positive")
    spacing = float(spacing)
    x0, y0, x1, y1 = domain.bounds

    def axis(low, high):
        start = max(0, math.ceil(low / spacing - 0.5))
        stop = max(start, math.ceil(high / spacing - 0.5))
        points = (np.arange(start, stop, dtype=np.float64) + 0.5) * spacing
        return points[(points >= low) & (points < high)]

    xs, ys = axis(x0, x1), axis(y0, y1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    indices = np.flatnonzero(domain.contains(xy))
    return UniformGrid(domain, spacing, xy[indices], indices, xs, ys)


@dataclass(frozen=True)
class SearchResult:
    grid: UniformGrid
    valid: np.ndarray
    evaluated: np.ndarray
    physical_cost: np.ndarray
    complete: bool
    num_physics_evaluations: int
    num_cache_hits: int
    elapsed_seconds: float
    max_evaluations: int | None = None
    search_mode: str = "uniform"


def uniform_search(domain, evaluator: CandidateEvaluator, *, spacing, max_evaluations=None):
    """Evaluate in stable grid order. A budget-limited result cannot become a teacher.

    The optional budget counts new renders, not invalid queries or cache hits.
    On reaching it, visits stop conservatively; remaining entries are unevaluated.
    """
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < 0):
        raise ValueError("max_evaluations must be a nonnegative integer or None")
    grid = uniform_grid(domain, spacing)
    count = len(grid.candidate_xy)
    valid, evaluated = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    costs = np.full(count, np.nan)
    renders, cache_hits = 0, 0
    started = time.perf_counter()
    for index, xy in enumerate(grid.candidate_xy):
        if max_evaluations is not None and renders >= max_evaluations:
            break
        result = evaluator.evaluate(xy)
        if result.num_physics_evaluations not in (0, 1):
            raise ValueError("A scalar evaluator must report zero or one new render")
        if result.valid and (not np.isfinite(result.physical_cost) or result.physical_cost < 0):
            raise ValueError("Valid physical costs must be finite and nonnegative")
        if not result.valid and result.num_physics_evaluations:
            raise ValueError("Invalid candidates must not be rendered")
        valid[index], evaluated[index] = result.valid, True
        costs[index] = result.physical_cost if result.valid else np.inf
        renders += result.num_physics_evaluations
        cache_hits += int(result.cache_hit)
    return SearchResult(grid, valid, evaluated, costs, bool(evaluated.all()), renders,
                        cache_hits, time.perf_counter() - started, max_evaluations)
