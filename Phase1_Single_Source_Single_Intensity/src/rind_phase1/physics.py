"""Unit-source forward evaluation and unweighted sampled-response disagreement.

Boundary weighting and edge costs are future extensions; this baseline uses
every observed pixel, including obstacle pixels, with equal weight.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rind_dataset.geometry import is_driver_valid_mixed
from rind_dataset.render import HAS_NUMBA


def response_disagreement(candidate_response, observed_response):
    """Mean absolute difference; for binary images, the mismatching pixel fraction."""
    candidate = np.asarray(candidate_response)
    observed = np.asarray(observed_response)
    if (candidate.shape != observed.shape or observed.ndim != 2 or not observed.size
            or not np.isfinite(candidate).all() or not np.isfinite(observed).all()):
        raise ValueError("Responses must be finite, nonempty images with the same shape")
    return float(np.mean(np.abs(candidate.astype(np.float64) - observed), dtype=np.float64))


@dataclass(frozen=True)
class CandidateEvaluation:
    valid: bool
    physical_cost: float
    num_physics_evaluations: int = 0
    cache_hit: bool = False


class CandidateEvaluator(Protocol):
    """Small scalar interface usable by search with real or synthetic physics.

    Invalid queries have infinite cost and zero renders. The count is actual
    new forward renders, so a cached evaluation reports zero.
    """

    def evaluate(self, candidate_xy) -> CandidateEvaluation: ...


class PhysicalEvaluator:
    """Bind one scene/window, validate continuous geometry, and cache scalar costs.

    Spatial priors and window exclusion belong to search.CandidateDomain.
    No source truth or reference solutions are used to construct these costs.
    """

    def __init__(self, dataset, scene_id, view_id, *, backend="auto"):
        if backend not in ("auto", "numba", "reference"):
            raise ValueError("backend must be auto, numba, or reference")
        if backend == "numba" and not HAS_NUMBA:
            raise RuntimeError("Numba backend is unavailable")
        self.dataset, self.scene_id, self.view_id = dataset, int(scene_id), int(view_id)
        self.observation = dataset.get_observation(scene_id, view_id)
        scene = dataset.get_scene(scene_id)
        self._types, self._params = scene["obstacle_types"], scene["obstacle_params"]
        self.world_size = int(dataset.manifest["global_size"])
        self.backend = backend
        self.resolved_backend = "numba" if backend == "auto" and HAS_NUMBA else (
            "reference" if backend == "auto" else backend)
        if not np.isin(self.observation["response"], [0, 1]).all():
            raise ValueError("The Phase I baseline requires binary observed responses")
        self._cache = {}
        self.num_physics_evaluations = 0

    @property
    def cost_config(self):
        return {"name": "mean_absolute_response_disagreement", "alpha": 1.0,
                "boundary_lambda": 0.0, "beta": 0.0, "edge_computed": False,
                "pixel_weighting": "uniform_all_window_pixels", "source_intensity": 1.0,
                "requested_backend": self.backend, "resolved_backend": self.resolved_backend}

    def evaluate(self, candidate_xy):
        xy = np.asarray(candidate_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("candidate_xy must contain two finite coordinates")
        key = tuple(xy.tolist())
        if key in self._cache:
            valid, cost = self._cache[key]
            return CandidateEvaluation(valid, cost, cache_hit=True)
        valid = is_driver_valid_mixed(xy, self._types, self._params,
                                      len(self._types), domain_size=self.world_size)
        if not valid:
            self._cache[key] = (False, np.inf)
            return CandidateEvaluation(False, np.inf)
        response = self.dataset.rerender(self.scene_id, self.observation["window"],
                                         xy, backend=self.backend)
        cost = response_disagreement(response, self.observation["response"])
        self._cache[key] = (True, cost)
        self.num_physics_evaluations += 1
        return CandidateEvaluation(True, cost, num_physics_evaluations=1)


def check_reference_solutions(dataset, scene_id, view_id, *, backend="auto"):
    """Independently rerender all ten witnesses; never add them to a teacher grid."""
    observation = dataset.get_observation(scene_id, view_id)
    references = dataset.get_candidates(scene_id, view_id)
    if len(references) != 10:
        raise ValueError("Expected the ten supplied reference source positions")
    positions, costs = [], []
    for reference in references:
        if reference.shape != (1, 3) or reference[0, 2] != 1:
            raise ValueError("Reference solutions must be single unit-strength sources")
        xy = reference[0, :2]
        rendered = dataset.rerender(scene_id, observation["window"], xy, backend=backend)
        cost = response_disagreement(rendered, observation["response"])
        if cost != 0 or not np.array_equal(rendered, observation["response"]):
            raise AssertionError(f"Reference {len(positions)} does not exactly reproduce this observation")
        positions.append(xy)
        costs.append(cost)
    return {"reference_xy": np.array(positions), "reference_cost": np.array(costs),
            "diagnostic_physics_evaluations": len(positions)}
