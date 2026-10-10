"""Physical comparison of complete multi-source parameter sets, teacher-only."""

import numpy as np
from rind_dataset.geometry import is_driver_valid_mixed
from rind_dataset.render import HAS_NUMBA

from rind_phase1_multi.search import (CandidateDomain, CandidateEvaluation,
                                     EvaluationBudgetExceeded, canonical_sources, source_set_key)

# Float64 addition may differ by a few ulps after permutation/source splitting.
# This is a renderer sanity tolerance, not a search or training hyperparameter.
REFERENCE_ATOL = 1e-12


def response_disagreement(candidate_response, observed_response):
    """Mean absolute additive-intensity error; responses are never binarized."""
    candidate = np.asarray(candidate_response, dtype=np.float64)
    observed = np.asarray(observed_response, dtype=np.float64)
    if (candidate.shape != observed.shape or observed.ndim != 2 or not observed.size
            or not np.isfinite(candidate).all() or not np.isfinite(observed).all()):
        raise ValueError("Responses must be finite nonempty images of equal shape")
    return float(np.mean(np.abs(candidate - observed)))


class JointResponseEvaluator:
    """Bind one observation; cache binary visibility by position, and costs by set.

    Linearity permits reusing unit-strength visibility for different strengths
    and source combinations. Cached visibility is bit-packed (one bit/pixel).
    Generating source parameters and reference solutions are never consulted.
    """

    def __init__(self, dataset, scene_id, view_id, *, backend="auto", max_source_renders=None):
        if backend not in ("auto", "numba", "reference"):
            raise ValueError("backend must be auto, numba or reference")
        if max_source_renders is not None and (
                not isinstance(max_source_renders, (int, np.integer)) or max_source_renders < 0):
            raise ValueError("max_source_renders must be a nonnegative integer or None")
        self.dataset, self.scene_id, self.view_id = dataset, int(scene_id), int(view_id)
        self.observation = dataset.get_observation(scene_id, view_id)
        self.world_size = int(dataset.manifest["global_size"])
        self.domain = CandidateDomain(tuple(self.observation["window"]), self.world_size)
        x, y, side = self.domain.window
        self.observed_response = np.asarray(dataset.get_region(scene_id, x, y, side)["response"],
                                            dtype=np.float64)
        if (self.observed_response.shape != (side, side) or
                not np.isfinite(self.observed_response).all()):
            raise ValueError("Observed response must be a finite float64 [L,L] image")
        scene = dataset.get_scene(scene_id)
        # Only obstacle geometry is extracted from the teacher-side scene API.
        self.obstacle_types = scene["obstacle_types"]
        self.obstacle_params = scene["obstacle_params"]
        self.obstacle_count = scene["obstacle_count"]
        self.backend, self.max_source_renders = backend, max_source_renders
        self.num_source_renders = self.num_joint_evaluations = self.num_cache_hits = 0
        self._visibility, self._valid_positions, self._costs = {}, {}, {}
        resolved = ("numba" if HAS_NUMBA else "reference") if backend == "auto" else backend
        self.cost_config = {"name": "response_mean_absolute_intensity_error", "response_dtype": "float64",
                            "pixel_normalization": "all L*L sampled pixels, including obstacle zeros",
                            "response_clipping": False, "strength_bounds": [0, 1],
                            "backend_requested": backend, "backend_resolved": resolved,
                            "visibility_cache": "unit-strength visibility, bit-packed by unique position"}

    @property
    def visibility_cache_bytes(self):
        return sum(a.nbytes for a in self._visibility.values())

    def positions_valid(self, positions):
        points = np.asarray(positions, dtype=np.float64)
        valid = self.domain.contains(points)
        for i in np.flatnonzero(valid):
            key = tuple(points[i])
            if key not in self._valid_positions:
                self._valid_positions[key] = bool(is_driver_valid_mixed(
                    points[i], self.obstacle_types, self.obstacle_params, self.obstacle_count,
                    domain_size=self.world_size))
            valid[i] = self._valid_positions[key]
        return valid

    def is_cached(self, sources):
        return source_set_key(sources) in self._costs

    def evaluate(self, sources):
        rows = canonical_sources(sources)
        key = rows.tobytes()
        if key in self._costs:
            self.num_cache_hits += 1
            valid, cost = self._costs[key]
            return CandidateEvaluation(valid, cost, 0, 0)
        valid = (np.isfinite(rows).all() and np.all((rows[:, 2] >= 0) & (rows[:, 2] <= 1))
                 and self.positions_valid(rows[:, :2]).all())
        if not valid:
            self.num_joint_evaluations += 1
            self._costs[key] = (False, float("inf"))
            return CandidateEvaluation(False, float("inf"))
        required = {tuple(row[:2]) for row in rows if row[2] != 0} - self._visibility.keys()
        if (self.max_source_renders is not None and
                self.num_source_renders + len(required) > self.max_source_renders):
            raise EvaluationBudgetExceeded("Unique unit-visibility render budget exhausted")
        renders_before = self.num_source_renders
        rendered = np.zeros_like(self.observed_response)
        for x, y, intensity in rows:
            if intensity == 0:
                continue
            point = (float(x), float(y))
            if point not in self._visibility:
                unit = self.dataset.rerender(self.scene_id, self.observation["window"],
                                             [[x, y, 1.0]], backend=self.backend)
                if unit.shape != rendered.shape or not np.all((unit == 0) | (unit == 1)):
                    raise ValueError("This cache requires binary unit-source visibility")
                self._visibility[point] = np.packbits(unit.reshape(-1).astype(bool), bitorder="little")
                self.num_source_renders += 1
            visible = np.unpackbits(self._visibility[point], count=rendered.size, bitorder="little")
            rendered += visible.reshape(rendered.shape) * intensity
        cost = response_disagreement(rendered, self.observed_response)
        self.num_joint_evaluations += 1
        self._costs[key] = (True, cost)
        return CandidateEvaluation(True, cost, self.num_source_renders - renders_before)


def evaluate_sources(dataset, scene_id, view_id, sources, *, backend="auto"):
    """One fresh evaluator; use JointResponseEvaluator to share caches across sets.

    Returns validity, raw MAE, and the actual number of unique position renders.
    Co-located rows share visibility. Source count can exceed the generation cap.
    """
    result = JointResponseEvaluator(dataset, scene_id, view_id, backend=backend).evaluate(sources)
    return {"valid": result.valid, "physical_cost": result.physical_cost,
            "num_source_renders": result.num_source_renders}
