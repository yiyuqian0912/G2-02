"""Physical comparison of complete multi-source parameter sets, teacher-only."""

import numpy as np
from rind_dataset.geometry import is_driver_valid_mixed
from rind_dataset.render import HAS_NUMBA

from rind_phase2.search import (CandidateDomain, CandidateEvaluation,
                                     EvaluationBudgetExceeded, canonical_sources, source_set_key)
from rind_phase2.gpu_visibility import TorchVisibilityRenderer, cuda_available

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

    def __init__(self, dataset, scene_id, view_id, *, backend="auto", max_source_renders=None,
                 gpu_batch_size=64, gpu_pixel_chunk=4096, gpu_verify_candidates=2,
                 gpu_device=None, gpu_min_batch_size=1024,
                 gpu_pair_budget=134217728):
        if backend not in ("auto", "cuda", "numba", "reference"):
            raise ValueError("backend must be auto, cuda, numba or reference")
        if max_source_renders is not None and (
                not isinstance(max_source_renders, (int, np.integer)) or max_source_renders < 0):
            raise ValueError("max_source_renders must be a nonnegative integer or None")
        for name, value in (("gpu_batch_size", gpu_batch_size),
                            ("gpu_pixel_chunk", gpu_pixel_chunk),
                            ("gpu_min_batch_size", gpu_min_batch_size),
                            ("gpu_pair_budget", gpu_pair_budget)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(gpu_verify_candidates, (int, np.integer)) or
                gpu_verify_candidates < 0):
            raise ValueError("gpu_verify_candidates must be a nonnegative integer")
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
        # Keep auto backward compatible; CUDA is selected explicitly by the
        # server training configuration so provenance records the acceleration.
        resolved = ("numba" if HAS_NUMBA else "reference") if backend == "auto" else backend
        if resolved == "cuda" and not cuda_available():
            raise RuntimeError("CUDA teacher backend requested, but torch.cuda.is_available() is false")
        self.backend, self.resolved_backend = backend, resolved
        self.max_source_renders = max_source_renders
        self.gpu_batch_size, self.gpu_pixel_chunk = int(gpu_batch_size), int(gpu_pixel_chunk)
        self.gpu_min_batch_size=min(int(gpu_min_batch_size),self.gpu_batch_size)
        self.gpu_pair_budget=int(gpu_pair_budget)
        self.gpu_device = (("cuda" if gpu_device is None else str(gpu_device))
                           if resolved == "cuda" else None)
        self.gpu_verify_remaining = int(gpu_verify_candidates)
        self.gpu_validation_cpu_renders = self.gpu_render_batches = 0
        self.gpu_scored_candidate_layers = 0
        self.gpu_oom_retries=0
        self.gpu_effective_batch_size=self.gpu_batch_size
        self.num_source_renders = self.num_joint_evaluations = self.num_cache_hits = 0
        self._visibility, self._valid_positions, self._costs = {}, {}, {}
        self._gpu_renderer = (TorchVisibilityRenderer(
            self.domain.window, self.observation["obstacle"], self.obstacle_types,
            self.obstacle_params, device=self.gpu_device, pixel_chunk=self.gpu_pixel_chunk)
            if resolved == "cuda" else None)
        self.cost_config = {"name": "response_mean_absolute_intensity_error", "response_dtype": "float64",
                            "pixel_normalization": "all L*L sampled pixels, including obstacle zeros",
                            "response_clipping": False, "strength_bounds": [0, 1],
                            "backend_requested": backend, "backend_resolved": resolved,
                            "visibility_cache": "unit-strength visibility, bit-packed by unique position",
                            "gpu_batch_size":self.gpu_batch_size,
                            "gpu_min_batch_size":self.gpu_min_batch_size,
                            "gpu_pair_budget":self.gpu_pair_budget,
                            "gpu_pixel_chunk":self.gpu_pixel_chunk,
                            "gpu_device":self.gpu_device,
                            "gpu_verify_candidates":int(gpu_verify_candidates)}

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

    def gpu_positions_valid(self, positions):
        """Evaluate obstacle membership as one GPU batch; grid already excludes the window."""
        if self.resolved_backend != "cuda":
            raise RuntimeError("gpu_positions_valid requires backend=cuda")
        return self._gpu_renderer.positions_valid_tensor(positions).cpu().numpy()

    def visibility_at(self, position):
        """Query cached unit-source visibility without evaluating a joint set."""
        xy = np.asarray(position, dtype=np.float64)
        if xy.shape != (2,) or not self.positions_valid(xy[None, :])[0]:
            raise ValueError("Unit-visibility position must be geometrically valid")
        key = tuple(xy)
        if key not in self._visibility:
            if self.max_source_renders is not None and self.num_source_renders >= self.max_source_renders:
                raise EvaluationBudgetExceeded("Unique unit-visibility render budget exhausted")
            if self.resolved_backend == "cuda":
                _, units = next(self.iter_visibility_batches(xy[None, :], batch_size=1))
                return units[0]
            unit = self.dataset.rerender(self.scene_id, self.observation["window"],
                                         [[*xy, 1.0]], backend=self.resolved_backend)
            if unit.shape != self.observed_response.shape or not np.all((unit == 0) | (unit == 1)):
                raise ValueError("This cache requires binary unit-source visibility")
            self._visibility[key] = np.packbits(unit.reshape(-1).astype(bool), bitorder="little")
            self.num_source_renders += 1
        return np.unpackbits(self._visibility[key], count=self.observed_response.size,
                             bitorder="little").reshape(self.observed_response.shape).astype(bool)

    def iter_visibility_batches(self, positions, *, batch_size=None):
        """Yield ``(start, [B,H,W] bool)`` with batched CUDA rendering when enabled."""
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("positions must be finite [N,2]")
        if not self.positions_valid(points).all():
            raise ValueError("visibility batches require geometrically valid positions")
        batch_size = self.gpu_batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for start in range(0, len(points), int(batch_size)):
            stop = min(start + int(batch_size), len(points))
            block = points[start:stop]
            units = np.empty((len(block), *self.observed_response.shape), dtype=bool)
            missing = []
            for i, xy in enumerate(block):
                packed = self._visibility.get(tuple(xy))
                if packed is None:
                    missing.append(i)
                else:
                    units[i] = np.unpackbits(
                        packed, count=self.observed_response.size,
                        bitorder="little").reshape(self.observed_response.shape).astype(bool)
            if missing:
                if (self.max_source_renders is not None and
                        self.num_source_renders + len(missing) > self.max_source_renders):
                    raise EvaluationBudgetExceeded("Unique unit-visibility render budget exhausted")
                if self.resolved_backend == "cuda":
                    rendered = self._gpu_renderer.render(block[missing])
                    self.gpu_render_batches += 1
                else:
                    rendered = np.stack([
                        self.dataset.rerender(
                            self.scene_id, self.observation["window"], [[*block[i], 1.0]],
                            backend=self.resolved_backend).astype(bool)
                        for i in missing])
                for local, unit in zip(missing, rendered):
                    if unit.shape != self.observed_response.shape:
                        raise ValueError("candidate visibility shape does not match observation")
                    if self.resolved_backend == "cuda" and self.gpu_verify_remaining:
                        expected = self.dataset.rerender(
                            self.scene_id, self.observation["window"], [[*block[local], 1.0]],
                            backend="numba" if HAS_NUMBA else "reference").astype(bool)
                        self.gpu_validation_cpu_renders += 1
                        self.gpu_verify_remaining -= 1
                        if not np.array_equal(unit, expected):
                            mismatch = int(np.count_nonzero(unit != expected))
                            raise RuntimeError(
                                f"CUDA/CPU visibility mismatch at {tuple(block[local])}: "
                                f"{mismatch} pixels")
                    units[local] = unit
                    self._visibility[tuple(block[local])] = np.packbits(
                        unit.reshape(-1), bitorder="little")
                self.num_source_renders += len(missing)
            yield start, units

    def gpu_layer_response_costs(self, positions, targets, weights, shadow, *,
                                 shadow_screening=True, alpha=1.0, temperature=0.05,
                                 boundary_lambda=0.0, boundary_sigma=1.0,
                                 positions_prevalidated=False):
        """Render, shadow-screen and score all layers without copying masks to CPU.

        Geometry and weighted response costs use float64.  Only the retained
        flags and the final ``layers x candidates`` cost matrix cross from the
        selected CUDA device to host memory.
        """
        if self.resolved_backend != "cuda":
            raise RuntimeError("gpu_layer_response_costs requires backend=cuda")
        points = np.asarray(positions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        shadow = np.asarray(shadow, dtype=bool)
        expected_shape = self.observed_response.shape
        if (points.ndim != 2 or points.shape[1] != 2 or
                targets.ndim != 3 or targets.shape[1:] != expected_shape or
                (weights is not None and weights.shape != targets.shape) or
                shadow.shape != expected_shape):
            raise ValueError("GPU layer scoring inputs are misaligned")
        if not np.isfinite(points).all() or not np.isfinite(targets).all():
            raise ValueError("GPU layer scoring inputs must be finite")
        if weights is not None and (not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError("GPU layer weights must be finite and positive")
        if type(positions_prevalidated) is not bool:
            raise ValueError("positions_prevalidated must be boolean")
        if (not positions_prevalidated and
                not self._gpu_renderer.positions_valid_tensor(points).all().item()):
            raise ValueError("GPU layer scoring requires geometrically valid positions")
        if (not np.isfinite(alpha) or alpha < 0 or
                not np.isfinite(temperature) or temperature <= 0 or
                not np.isfinite(boundary_lambda) or boundary_lambda < 0 or
                not np.isfinite(boundary_sigma) or boundary_sigma <= 0):
            raise ValueError("alpha must be nonnegative and temperature must be positive")

        torch = self._gpu_renderer.torch
        device = self._gpu_renderer.device
        target_t = torch.as_tensor(targets, dtype=torch.float64, device=device)
        if weights is None:
            generated=[];height,width=expected_shape
            yy,xx=torch.meshgrid(torch.arange(height,dtype=torch.float64,device=device),
                                 torch.arange(width,dtype=torch.float64,device=device),indexing='ij')
            pixels=torch.stack((yy.reshape(-1),xx.reshape(-1)),dim=1)
            for target in target_t:
                boundary=torch.zeros(expected_shape,dtype=torch.bool,device=device)
                horizontal=target[:,1:]!=target[:,:-1]
                boundary[:,1:]|=horizontal;boundary[:,:-1]|=horizontal
                vertical=target[1:,:]!=target[:-1,:]
                boundary[1:,:]|=vertical;boundary[:-1,:]|=vertical
                if boundary_lambda==0 or not boundary.any().item():
                    generated.append(torch.ones(expected_shape,dtype=torch.float64,device=device))
                    continue
                edge=torch.nonzero(boundary,as_tuple=False).to(dtype=torch.float64)
                distance_squared=torch.empty(len(pixels),dtype=torch.float64,device=device)
                for pixel_start in range(0,len(pixels),1024):
                    pixel_stop=min(pixel_start+1024,len(pixels))
                    delta=pixels[pixel_start:pixel_stop,None,:]-edge[None,:,:]
                    distance_squared[pixel_start:pixel_stop]=delta.square().sum(dim=2).min(dim=1).values
                generated.append(1.0+float(boundary_lambda)*torch.exp(
                    -distance_squared.reshape(expected_shape)/(2.0*float(boundary_sigma)**2)))
            weight_t=torch.stack(generated)
        else:
            weight_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
        denominator = weight_t.flatten(1).sum(dim=1)
        # For binary rendered visibility u and any target t in [0,1],
        # |u-t| = t + u(1-2t).  This turns the former per-layer pointwise loop
        # into one float64 GEMM, which is both mathematically equivalent and
        # substantially better suited to H20-class GPUs.
        target_flat=target_t.flatten(1)
        weight_flat=weight_t.flatten(1)
        linear_weight=weight_flat*(1.0-2.0*target_flat)
        target_base=(weight_flat*target_flat).sum(dim=1)
        shadow_t = torch.as_tensor(shadow, dtype=torch.bool, device=device)
        retained_all = torch.empty(len(points), dtype=torch.bool, device=device)
        costs_all = torch.empty((len(targets), len(points)), dtype=torch.float64, device=device)

        # Geometry memory scales primarily with candidate x free-pixel pairs.
        # Keep that product bounded, then round down to a power of two for
        # stable CUDA shapes. Small windows can still render the entire grid in
        # one launch while 128x128 windows start at a safer H20 batch.
        free_pixels=max(1,len(getattr(self._gpu_renderer,'pixels',())))
        if free_pixels==1 and not hasattr(self._gpu_renderer,'pixels'):
            free_pixels=int(np.prod(expected_shape))
        minimum_batch=min(getattr(self,'gpu_min_batch_size',self.gpu_batch_size),self.gpu_batch_size)
        pair_budget=getattr(self,'gpu_pair_budget',134217728)
        pair_limited=max(minimum_batch,pair_budget//free_pixels)
        initial=min(self.gpu_batch_size,len(points),pair_limited)
        active_batch=max(minimum_batch,1<<(max(1,int(initial)).bit_length()-1))
        active_batch=min(active_batch,len(points))
        self.gpu_effective_batch_size=active_batch
        start=0
        while start<len(points):
            stop = min(start + active_batch, len(points))
            block = points[start:stop]
            unit=value=batch_cost=None
            try:
                unit = self._gpu_renderer.render_tensor(block)

                verify = min(self.gpu_verify_remaining, len(block))
                if verify:
                    verification = unit[:verify].cpu().numpy()
                    for local in range(verify):
                        expected = self.dataset.rerender(
                            self.scene_id, self.observation["window"], [[*block[local], 1.0]],
                            backend="numba" if HAS_NUMBA else "reference").astype(bool)
                        self.gpu_validation_cpu_renders += 1
                        self.gpu_verify_remaining -= 1
                        if not np.array_equal(verification[local], expected):
                            mismatch = int(np.count_nonzero(verification[local] != expected))
                            raise RuntimeError(
                                f"CUDA/CPU visibility mismatch at {tuple(block[local])}: "
                                f"{mismatch} pixels")

                if shadow_screening:
                    retained_t = ~unit[:, shadow_t].any(dim=1)
                else:
                    retained_t = torch.ones(len(block), dtype=torch.bool, device=device)
                retained_all[start:stop] = retained_t

                value=unit.flatten(1).to(dtype=torch.float64)
                batch_cost=(value@linear_weight.transpose(0,1)+target_base).transpose(0,1)
                batch_cost.clamp_min_(0.0).div_(denominator[:,None])
                costs_all[:, start:stop] = float(alpha) * batch_cost
                self.gpu_render_batches += 1
                self.num_source_renders += len(block)
                self.gpu_scored_candidate_layers += len(block) * len(targets)
            except torch.cuda.OutOfMemoryError:
                if device.type!='cuda' or active_batch<=self.gpu_min_batch_size:
                    raise
                del unit,value,batch_cost
                torch.cuda.empty_cache()
                active_batch=max(self.gpu_min_batch_size,active_batch//2)
                self.gpu_effective_batch_size=active_batch
                self.gpu_oom_retries+=1
                continue
            start=stop
        costs_all[:, ~retained_all] = float("inf")
        probabilities = torch.zeros_like(costs_all)
        if retained_all.any().item():
            retained_costs = costs_all[:, retained_all]
            minimum = retained_costs.min(dim=1, keepdim=True).values
            unnormalized = torch.exp(-(retained_costs - minimum) / float(temperature))
            probabilities[:, retained_all] = unnormalized / unnormalized.sum(dim=1, keepdim=True)
        return (retained_all.cpu().numpy(), costs_all.cpu().numpy(),
                probabilities.cpu().numpy())

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
            rendered += self.visibility_at((x, y)) * intensity
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
