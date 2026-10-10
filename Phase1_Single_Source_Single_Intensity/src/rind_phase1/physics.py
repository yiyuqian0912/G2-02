"""Phase I physical evaluation shared by search/teacher and physics-cost diagnostics.

This module preserves the scalar CandidateEvaluator interface used by the
teacher/search pipeline while also supporting Alex's richer physical cost

    C = alpha * L_resp + beta * L_edge

where L_resp may use boundary-aware pixel weights and L_edge is a symmetric
nearest-boundary disagreement.  The default configuration
(alpha=1, beta=0, boundary_lambda=0) exactly reduces to the original uniform
mean-absolute-response baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rind_dataset.geometry import is_driver_valid_mixed
from rind_dataset.render import HAS_NUMBA

try:
    from numba import njit
except ImportError:  # Keep base installs usable when numba is unavailable.
    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator


# ---------------------------------------------------------------------------
# Basic validation and baseline response disagreement
# ---------------------------------------------------------------------------

def _response_array(response, name="response"):
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 2 or response.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(response).all():
        raise ValueError(f"{name} must contain only finite values")
    return response


def _window_array(window):
    window = np.asarray(window)
    if window.shape != (3,) or window.dtype.kind not in "iu":
        raise ValueError("window must contain integer (x, y, size)")
    if int(window[2]) <= 0:
        raise ValueError("window size must be positive")
    return window.astype(np.int64, copy=False)


def _candidate_array(candidate_xy):
    candidate = np.asarray(candidate_xy, dtype=np.float64)
    if candidate.shape != (2,) or not np.isfinite(candidate).all():
        raise ValueError("candidate_xy must contain two finite coordinates")
    return candidate


def response_disagreement(candidate_response, observed_response):
    """Unweighted mean absolute response error.

    For binary responses this is exactly the mismatching-pixel fraction.
    """
    candidate = _response_array(candidate_response, "candidate_response")
    observed = _response_array(observed_response, "observed_response")
    if candidate.shape != observed.shape:
        raise ValueError("Responses must have the same shape")
    return float(np.mean(np.abs(candidate - observed), dtype=np.float64))


# ---------------------------------------------------------------------------
# Exact squared Euclidean distance transform for boundary costs
# ---------------------------------------------------------------------------

@njit(cache=True)
def _edt_1d(f):
    n = f.shape[0]
    d = np.empty(n, dtype=np.float64)
    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        while True:
            p = v[k]
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * (q - p))
            if k == 0 or s > z[k]:
                break
            k -= 1
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        delta = q - v[k]
        d[q] = delta * delta + f[v[k]]
    return d


@njit(cache=True)
def _squared_distance_to_mask(mask):
    """Squared Euclidean distance from every pixel to the nearest True pixel."""
    h, w = mask.shape
    far = float(4 * (h * h + w * w) + 1)

    row_pass = np.empty((h, w), dtype=np.float64)
    row_input = np.empty(w, dtype=np.float64)
    for r in range(h):
        for c in range(w):
            row_input[c] = 0.0 if mask[r, c] else far
        row_pass[r, :] = _edt_1d(row_input)

    result = np.empty((h, w), dtype=np.float64)
    col_input = np.empty(h, dtype=np.float64)
    for c in range(w):
        for r in range(h):
            col_input[r] = row_pass[r, c]
        result[:, c] = _edt_1d(col_input)
    return result


# ---------------------------------------------------------------------------
# Boundary-aware response and edge costs
# ---------------------------------------------------------------------------

def response_boundary_mask(response):
    """Pixels touching an internal four-neighbor response boundary."""
    response = _response_array(response)
    boundary = np.zeros(response.shape, dtype=bool)

    horizontal = response[:, 1:] != response[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical = response[1:, :] != response[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def boundary_weight_map(response, *, boundary_lambda, boundary_sigma):
    """Observed-boundary weights for L_resp."""
    response = _response_array(response)
    boundary_lambda = float(boundary_lambda)
    boundary_sigma = float(boundary_sigma)

    if not np.isfinite(boundary_lambda) or boundary_lambda < 0:
        raise ValueError("boundary_lambda must be finite and nonnegative")
    if not np.isfinite(boundary_sigma) or boundary_sigma <= 0:
        raise ValueError("boundary_sigma must be finite and positive")

    boundary = response_boundary_mask(response)
    if boundary_lambda == 0 or not boundary.any():
        return np.ones(response.shape, dtype=np.float64)

    distance_squared = _squared_distance_to_mask(boundary)
    return 1.0 + boundary_lambda * np.exp(
        -distance_squared / (2.0 * boundary_sigma * boundary_sigma)
    )


def response_cost(observed_response, candidate_response, weights):
    """Boundary-weighted mean absolute response error L_resp."""
    observed = _response_array(observed_response, "observed_response")
    candidate = _response_array(candidate_response, "candidate_response")
    weights = np.asarray(weights, dtype=np.float64)

    if candidate.shape != observed.shape:
        raise ValueError("candidate_response and observed_response must have the same shape")
    if weights.shape != observed.shape:
        raise ValueError("weights and observed_response must have the same shape")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and strictly positive")

    weighted_error = np.sum(weights * np.abs(candidate - observed), dtype=np.float64)
    return float(weighted_error / np.sum(weights, dtype=np.float64))


def _directed_boundary_distance(source_boundary, target_boundary):
    source_boundary = np.asarray(source_boundary, dtype=bool)
    target_boundary = np.asarray(target_boundary, dtype=bool)

    if source_boundary.shape != target_boundary.shape:
        raise ValueError("boundary masks must have the same shape")
    if source_boundary.ndim != 2:
        raise ValueError("boundary masks must be 2D")
    if not source_boundary.any():
        return 0.0

    h, w = source_boundary.shape
    diagonal = float(np.hypot(max(h - 1, 0), max(w - 1, 0)))
    if diagonal == 0.0:
        return 0.0
    if not target_boundary.any():
        return 1.0

    distances = np.sqrt(_squared_distance_to_mask(target_boundary))
    return float(np.mean(distances[source_boundary]) / diagonal)


def edge_cost(observed_response, candidate_response):
    """Symmetric normalized nearest-boundary disagreement L_edge."""
    observed = _response_array(observed_response, "observed_response")
    candidate = _response_array(candidate_response, "candidate_response")
    if observed.shape != candidate.shape:
        raise ValueError("candidate_response and observed_response must have the same shape")

    observed_boundary = response_boundary_mask(observed)
    candidate_boundary = response_boundary_mask(candidate)
    if not observed_boundary.any() and not candidate_boundary.any():
        return 0.0

    return float(
        _directed_boundary_distance(observed_boundary, candidate_boundary)
        + _directed_boundary_distance(candidate_boundary, observed_boundary)
    )


# ---------------------------------------------------------------------------
# Candidate validity and physical-cost settings
# ---------------------------------------------------------------------------

def _point_in_window(candidate_xy, window, world_size):
    x, y, size = map(float, window)
    sx, sy = map(float, candidate_xy)
    right, bottom = x + size, y + size

    x_inside = x <= sx < right
    y_inside = y <= sy < bottom
    if right == world_size and sx == world_size:
        x_inside = True
    if bottom == world_size and sy == world_size:
        y_inside = True
    return x_inside and y_inside


def _scene_obstacle_count(scene):
    return int(scene.get("obstacle_count", len(scene["obstacle_types"])))


def _candidate_valid(candidate_xy, window, scene, world_size, exclude_observation_window):
    if np.any(candidate_xy < 0) or np.any(candidate_xy > world_size):
        return False
    if exclude_observation_window and _point_in_window(candidate_xy, window, world_size):
        return False
    return bool(
        is_driver_valid_mixed(
            candidate_xy,
            scene["obstacle_types"],
            scene["obstacle_params"],
            _scene_obstacle_count(scene),
            domain_size=world_size,
        )
    )


def _check_settings(alpha, beta, fixed_intensity):
    alpha = float(alpha)
    beta = float(beta)
    fixed_intensity = float(fixed_intensity)

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.isclose(fixed_intensity, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("the current Phase I dataset requires fixed_intensity=1.0")
    return alpha, beta


def _check_backend(backend):
    if backend not in ("auto", "numba", "reference"):
        raise ValueError("backend must be auto, numba, or reference")
    if backend == "numba" and not HAS_NUMBA:
        raise RuntimeError("Numba backend is unavailable")


# ---------------------------------------------------------------------------
# Batch/public Alex-style evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_candidate(
    dataset,
    scene_id,
    observed_response,
    window,
    candidate_xy,
    *,
    boundary_lambda,
    boundary_sigma,
    alpha=1.0,
    beta=0.0,
    fixed_intensity=1.0,
    exclude_observation_window=True,
    weights=None,
    backend="auto",
):
    """Rerender and score one candidate source."""
    _check_backend(backend)
    observed = _response_array(observed_response, "observed_response")
    window = _window_array(window)
    candidate = _candidate_array(candidate_xy)

    expected_shape = (int(window[2]), int(window[2]))
    if observed.shape != expected_shape:
        raise ValueError("observed_response shape must match window size")

    alpha, beta = _check_settings(alpha, beta, fixed_intensity)
    if weights is None:
        weights = boundary_weight_map(
            observed,
            boundary_lambda=boundary_lambda,
            boundary_sigma=boundary_sigma,
        )
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != observed.shape:
            raise ValueError("weights and observed_response must have the same shape")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("weights must be finite and strictly positive")

    world_size = float(dataset.manifest["global_size"])
    scene = dataset.get_scene(int(scene_id))
    valid = _candidate_valid(
        candidate,
        window,
        scene,
        world_size,
        exclude_observation_window,
    )

    if not valid:
        return {
            "candidate_xy": candidate.copy(),
            "candidate_response": None,
            "L_resp": float("inf"),
            "L_edge": float("inf") if beta > 0 else None,
            "physical_cost": float("inf"),
            "valid": False,
        }

    candidate_response = dataset.rerender(
        int(scene_id), window, candidate, backend=backend
    )
    l_resp = response_cost(observed, candidate_response, weights)
    l_edge = edge_cost(observed, candidate_response) if beta > 0 else None
    physical_cost = alpha * l_resp + (beta * l_edge if l_edge is not None else 0.0)

    return {
        "candidate_xy": candidate.copy(),
        "candidate_response": candidate_response,
        "L_resp": l_resp,
        "L_edge": l_edge,
        "physical_cost": float(physical_cost),
        "valid": True,
    }


def evaluate_candidates(
    dataset,
    scene_id,
    observed_response,
    window,
    candidate_xy,
    *,
    boundary_lambda,
    boundary_sigma,
    alpha=1.0,
    beta=0.0,
    fixed_intensity=1.0,
    exclude_observation_window=True,
    backend="auto",
):
    """Evaluate an ordered [N,2] candidate set, preserving input order."""
    _check_backend(backend)
    observed = _response_array(observed_response, "observed_response")
    window = _window_array(window)
    candidates = np.asarray(candidate_xy, dtype=np.float64)

    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidate_xy must have shape [N, 2]")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate_xy must contain only finite coordinates")
    if observed.shape != (int(window[2]), int(window[2])):
        raise ValueError("observed_response shape must match window size")

    alpha, beta = _check_settings(alpha, beta, fixed_intensity)
    weights = boundary_weight_map(
        observed,
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
    )

    world_size = float(dataset.manifest["global_size"])
    scene = dataset.get_scene(int(scene_id))

    valid = np.zeros(len(candidates), dtype=bool)
    l_resp = np.full(len(candidates), np.inf, dtype=np.float64)
    physical_cost = np.full(len(candidates), np.inf, dtype=np.float64)
    l_edge = (
        np.full(len(candidates), np.inf, dtype=np.float64)
        if beta > 0
        else None
    )

    for i, candidate in enumerate(candidates):
        if not _candidate_valid(
            candidate,
            window,
            scene,
            world_size,
            exclude_observation_window,
        ):
            continue

        candidate_response = dataset.rerender(
            int(scene_id), window, candidate, backend=backend
        )
        l_resp[i] = response_cost(observed, candidate_response, weights)

        if beta > 0:
            l_edge[i] = edge_cost(observed, candidate_response)
            physical_cost[i] = alpha * l_resp[i] + beta * l_edge[i]
        else:
            physical_cost[i] = alpha * l_resp[i]

        valid[i] = True

    return {
        "candidate_xy": candidates.copy(),
        "valid": valid,
        "L_resp": l_resp,
        "L_edge": l_edge,
        "physical_cost": physical_cost,
    }


# ---------------------------------------------------------------------------
# Yushu teacher/search scalar interface, now backed by the richer cost
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateEvaluation:
    valid: bool
    physical_cost: float
    num_physics_evaluations: int = 0
    cache_hit: bool = False


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate_xy) -> CandidateEvaluation: ...


class PhysicalEvaluator:
    """Bind one scene/window, cache scalar evaluations, and expose search API.

    The search CandidateDomain owns source-window exclusion.  This evaluator
    only checks world/obstacle validity, so it remains reusable with either the
    world-wide or quadtree-prior candidate domain.

    Defaults reproduce the original teacher baseline:
        alpha=1, beta=0, boundary_lambda=0.
    """

    def __init__(
        self,
        dataset,
        scene_id,
        view_id,
        *,
        alpha=1.0,
        beta=0.0,
        boundary_lambda=0.0,
        boundary_sigma=1.0,
        fixed_intensity=1.0,
        backend="auto",
    ):
        _check_backend(backend)
        self.alpha, self.beta = _check_settings(alpha, beta, fixed_intensity)
        self.boundary_lambda = float(boundary_lambda)
        self.boundary_sigma = float(boundary_sigma)
        self.fixed_intensity = float(fixed_intensity)

        self.dataset = dataset
        self.scene_id = int(scene_id)
        self.view_id = int(view_id)
        self.observation = dataset.get_observation(scene_id, view_id)
        self.scene = dataset.get_scene(scene_id)
        self.world_size = int(dataset.manifest["global_size"])
        self.backend = backend
        self.resolved_backend = (
            "numba"
            if backend == "auto" and HAS_NUMBA
            else ("reference" if backend == "auto" else backend)
        )

        if not np.isin(self.observation["response"], [0, 1]).all():
            raise ValueError("The Phase I baseline requires binary observed responses")

        self.weights = boundary_weight_map(
            self.observation["response"],
            boundary_lambda=self.boundary_lambda,
            boundary_sigma=self.boundary_sigma,
        )
        self._cache = {}
        self.num_physics_evaluations = 0

    @property
    def cost_config(self):
        baseline = (
            self.alpha == 1.0
            and self.beta == 0.0
            and self.boundary_lambda == 0.0
        )
        return {
            "name": (
                "mean_absolute_response_disagreement"
                if baseline
                else "boundary_weighted_response_plus_edge"
            ),
            "alpha": self.alpha,
            "boundary_lambda": self.boundary_lambda,
            "boundary_sigma": self.boundary_sigma,
            "beta": self.beta,
            "edge_computed": self.beta > 0,
            "pixel_weighting": (
                "uniform_all_window_pixels"
                if self.boundary_lambda == 0
                else "observed_boundary_weighted"
            ),
            "source_intensity": self.fixed_intensity,
            "requested_backend": self.backend,
            "resolved_backend": self.resolved_backend,
        }

    def evaluate(self, candidate_xy):
        xy = _candidate_array(candidate_xy)
        key = tuple(xy.tolist())

        if key in self._cache:
            valid, cost = self._cache[key]
            return CandidateEvaluation(valid, cost, cache_hit=True)

        valid = _candidate_valid(
            xy,
            self.observation["window"],
            self.scene,
            self.world_size,
            exclude_observation_window=False,
        )
        if not valid:
            self._cache[key] = (False, np.inf)
            return CandidateEvaluation(False, np.inf)

        response = self.dataset.rerender(
            self.scene_id,
            self.observation["window"],
            xy,
            backend=self.backend,
        )
        l_resp = response_cost(self.observation["response"], response, self.weights)
        l_edge = (
            edge_cost(self.observation["response"], response)
            if self.beta > 0
            else 0.0
        )
        cost = float(self.alpha * l_resp + self.beta * l_edge)

        self._cache[key] = (True, cost)
        self.num_physics_evaluations += 1
        return CandidateEvaluation(True, cost, num_physics_evaluations=1)


def check_reference_solutions(dataset, scene_id, view_id, *, backend="auto"):
    """Rerender all ten supplied witnesses independently of teacher support."""
    _check_backend(backend)
    observation = dataset.get_observation(scene_id, view_id)
    references = dataset.get_candidates(scene_id, view_id)

    if len(references) != 10:
        raise ValueError("Expected the ten supplied reference source positions")

    positions, costs = [], []
    for reference in references:
        if reference.shape != (1, 3) or reference[0, 2] != 1:
            raise ValueError("Reference solutions must be single unit-strength sources")

        xy = reference[0, :2]
        rendered = dataset.rerender(
            scene_id, observation["window"], xy, backend=backend
        )
        cost = response_disagreement(rendered, observation["response"])

        if cost != 0 or not np.array_equal(rendered, observation["response"]):
            raise AssertionError(
                f"Reference {len(positions)} does not exactly reproduce this observation"
            )

        positions.append(xy)
        costs.append(cost)

    return {
        "reference_xy": np.asarray(positions, dtype=np.float64),
        "reference_cost": np.asarray(costs, dtype=np.float64),
        "diagnostic_physics_evaluations": len(positions),
    }
