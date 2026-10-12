"""Declared world domain and reproducible sampling of complete source sets.

This baseline samples the joint space. It never claims to exhaust that space.
Reference witnesses and generating source labels are not proposal inputs.
"""

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np


class EvaluationBudgetExceeded(RuntimeError):
    """A physical evaluation budget was reached before support completion."""


class NoCandidatePositionsError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateDomain:
    window: tuple
    world_size: int

    def __post_init__(self):
        w = np.asarray(self.window)
        if (w.shape != (3,) or w.dtype.kind not in "iu" or
                not isinstance(self.world_size, (int, np.integer)) or self.world_size <= 0):
            raise ValueError("Domain requires integer (x,y,L) and positive integer world_size")
        x, y, side = map(int, w)
        if side <= 0 or min(x, y) < 0 or max(x + side, y + side) > self.world_size:
            raise ValueError("Observation window must be inside the world")
        object.__setattr__(self, "window", (x, y, side))
        object.__setattr__(self, "world_size", int(self.world_size))

    def contains(self, positions):
        """World minus W; partition edges belong to the right/bottom cell."""
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("positions must have shape [N,2]")
        x, y, side = self.window
        end = np.array([x + side, y + side])
        upper = (points < end) | ((end == self.world_size) & (points <= end))
        in_window = ((points >= [x, y]) & upper).all(axis=1)
        return (np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) &
                (points <= self.world_size).all(axis=1) & ~in_window)

    def metadata(self):
        return {"domain": "world_outside_observation_and_obstacles", "quadtree_prior": False,
                "source_location_scope": "outside_observation_window", "window": list(self.window),
                "world_size": self.world_size, "world_bounds": "closed [0,world_size]^2",
                "window_edges": "half-open; include the terminal world edge"}


@dataclass(frozen=True)
class CandidateEvaluation:
    valid: bool
    physical_cost: float
    num_source_renders: int = 0
    new_joint_evaluations: int = 1


class SourceSetEvaluator(Protocol):
    """Small mockable interface; no dataset labels or reference API is required."""

    def positions_valid(self, positions: np.ndarray) -> np.ndarray: ...

    def evaluate(self, sources: np.ndarray) -> CandidateEvaluation: ...


@dataclass
class SourceGrid:
    domain: CandidateDomain
    spacing: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    positions: np.ndarray
    grid_index: np.ndarray
    valid: np.ndarray

    def metadata(self):
        return {**self.domain.metadata(), "candidate_spacing": self.spacing,
                "grid_origin": [self.spacing / 2, self.spacing / 2],
                "grid_order": "y then x, both ascending", "num_positions": len(self.positions),
                "num_valid_positions": int(self.valid.sum()),
                "position_measure": "equal mass on geometrically valid uniform-grid points"}


def uniform_source_grid(domain, evaluator, *, spacing):
    """A global regular grid, followed by window and continuous-geometry masks."""
    if not np.isscalar(spacing) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    axis = np.arange(float(spacing) / 2, domain.world_size, float(spacing), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(len(axis)), np.arange(len(axis)), indexing="ij")
    indices = np.column_stack((yy.ravel(), xx.ravel()))
    points = np.column_stack((axis[xx.ravel()], axis[yy.ravel()]))
    inside = domain.contains(points)
    points, indices = points[inside], indices[inside]
    valid = np.asarray(evaluator.positions_valid(points))
    if valid.shape != (len(points),) or valid.dtype != np.dtype(bool):
        raise ValueError("positions_valid must return an aligned boolean [N] mask")
    return SourceGrid(domain, float(spacing), axis.copy(), axis.copy(), points, indices, valid)


def canonical_sources(sources):
    """Sort rows lexicographically; preserve co-located sources and their count."""
    rows = np.asarray(sources, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows):
        raise ValueError("sources must be a nonempty [K,3] parameter set")
    rows = rows.copy()
    rows[rows == 0] = 0.0  # Signed zero must not produce a different cache key.
    return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]


def source_set_key(sources):
    return canonical_sources(sources).tobytes()


@dataclass
class JointSupport:
    sources: np.ndarray             # [N,K_max,3]; inactive rows are zero
    source_counts: np.ndarray       # [N]; only [:source_counts[i]] is active
    base_mass: np.ndarray           # [N]; sampling/count-prior mass, not physical likelihood
    multiplicity: np.ndarray       # [N]; number of permutation-equivalent proposals
    metadata: dict

    def __len__(self):
        return len(self.source_counts)

    def source_set(self, index):
        return self.sources[index, :int(self.source_counts[index])]


def make_joint_support(source_sets, *, base_mass, metadata=None):
    """Deduplicate permutations in first-occurrence order, adding their mass.

    Explicit masses are required for caller-provided support. They prevent a
    hidden source-count prior when support mixes different K or proposal counts.
    Splitting one source into two remains a different count hypothesis.
    """
    rows = [canonical_sources(s) for s in source_sets]
    mass = np.asarray(base_mass, dtype=np.float64)
    if (not rows or mass.shape != (len(rows),) or not np.isfinite(mass).all() or
            np.any(mass <= 0) or not np.isfinite(mass.sum())):
        raise ValueError("Provide nonempty source sets and aligned positive finite base_mass")
    unique, weights, multiplicity, by_key = [], [], [], {}
    for sources, weight in zip(rows, mass):
        key = sources.tobytes()
        if key in by_key:
            i = by_key[key]
            weights[i] += float(weight)
            multiplicity[i] += 1
        else:
            by_key[key] = len(unique)
            unique.append(sources)
            weights.append(float(weight))
            multiplicity.append(1)
    counts = np.array([len(s) for s in unique], dtype=np.int64)
    padded = np.zeros((len(unique), int(counts.max()), 3), dtype=np.float64)
    for i, sources in enumerate(unique):
        padded[i, :len(sources)] = sources
    details = {**(metadata or {}), "source_order": "lexicographic x,y,strength",
               "deduplication": "permutation-equivalent rows; stable first occurrence; sum base mass",
               "num_proposals": len(rows), "num_unique_hypotheses": len(unique),
               "reference_solutions_used_as_targets": False}
    return JointSupport(padded, counts, np.array(weights), np.array(multiplicity, dtype=np.int64), details)


def sample_joint_support(grid, *, source_counts, samples_per_count, count_prior=None, seed):
    """Stratified Monte Carlo: iid uniform grid locations and U[0,1) strengths.

    Sources are drawn with replacement, then canonicalized. This samples the
    unordered measure induced by iid ordered draws, including multiplicities;
    it does not assign equal mass to enumerated unordered configurations.
    """
    counts = np.asarray(source_counts)
    if (counts.ndim != 1 or not len(counts) or counts.dtype.kind not in "iu" or
            np.any(counts <= 0) or len(np.unique(counts)) != len(counts)):
        raise ValueError("source_counts must be distinct positive integers in declared order")
    sizes = np.asarray(samples_per_count)
    if sizes.ndim == 0:
        sizes = np.repeat(sizes, len(counts))
    if sizes.shape != counts.shape or sizes.dtype.kind not in "iu" or np.any(sizes <= 0):
        raise ValueError("samples_per_count must be positive integer(s), one for each K")
    if count_prior is None:
        if len(counts) != 1:
            raise ValueError("Unknown source count requires an explicit count_prior")
        prior = np.ones(1)
    else:
        prior = np.asarray(count_prior, dtype=np.float64)
    if (prior.shape != counts.shape or not np.isfinite(prior).all() or np.any(prior <= 0) or
            not np.isclose(prior.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("count_prior must be positive, aligned with source_counts, and sum to one")
    prior = prior / prior.sum()
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    pool = grid.positions[grid.valid]
    if not len(pool):
        raise NoCandidatePositionsError("No valid position at this spacing; choose a different support explicitly")
    sets, weights = [], []
    for count, size, probability in zip(counts, sizes, prior):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(count)])))
        locations = rng.integers(len(pool), size=(int(size), int(count)))
        strengths = rng.random((int(size), int(count)))
        for xy, intensity in zip(pool[locations], strengths):
            sets.append(np.column_stack((xy, intensity)))
            weights.append(float(probability) / int(size))
    metadata = {"search_mode": "stratified_uniform_joint_monte_carlo", "grid": grid.metadata(),
                "source_counts": counts.tolist(), "samples_per_count": sizes.tolist(),
                "source_count_prior": prior.tolist(), "seed": int(seed), "rng": "PCG64; SeedSequence([seed,K])",
                "strength_sampling": "iid uniform [0,1)", "location_sampling": "iid uniform valid grid; with replacement",
                "sampling_measure": "count prior times iid position/strength prior; empirical Monte Carlo support",
                "base_mass_rule": "pi_K / N_K per draw; aggregate exact permutation duplicates",
                "completeness": "all declared joint proposals evaluated; NOT exhaustive joint-space coverage"}
    return make_joint_support(sets, base_mass=weights, metadata=metadata)


@dataclass
class JointSearchResult:
    support: JointSupport
    physical_cost: np.ndarray
    valid: np.ndarray
    evaluated: np.ndarray
    complete: bool
    num_joint_evaluations: int
    num_source_renders: int
    num_cache_hits: int
    elapsed_seconds: float
    failure_reason: str | None = None


def evaluate_joint_support(support, evaluator, *, max_evaluations=None):
    """Evaluate in support order. A stopped prefix can never become a target."""
    if max_evaluations is not None and (not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 0):
        raise ValueError("max_evaluations must be a nonnegative integer or None")
    costs = np.full(len(support), np.inf)
    valid = np.zeros(len(support), dtype=bool)
    evaluated = np.zeros(len(support), dtype=bool)
    started = time.perf_counter()
    joint_count = render_count = cache_hits = 0
    failure = None
    for i in range(len(support)):
        sources = support.source_set(i)
        cached = getattr(evaluator, "is_cached", lambda _: False)(sources)
        if max_evaluations is not None and joint_count >= max_evaluations and not cached:
            failure = "Unique joint-evaluation budget exhausted"
            break
        try:
            result = evaluator.evaluate(sources)
        except EvaluationBudgetExceeded as exc:
            failure = str(exc)
            break
        if result.valid and (not np.isfinite(result.physical_cost) or result.physical_cost < 0):
            raise ValueError("Evaluator returned an invalid physical cost for a valid source set")
        costs[i], valid[i], evaluated[i] = result.physical_cost, result.valid, True
        joint_count += result.new_joint_evaluations
        render_count += result.num_source_renders
        cache_hits += int(result.new_joint_evaluations == 0)
    return JointSearchResult(support, costs, valid, evaluated, bool(evaluated.all()), joint_count,
                             render_count, cache_hits, time.perf_counter() - started, failure)
