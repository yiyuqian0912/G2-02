"""Constant-k diagnostic octree. Sampled cells are never certified or pruned."""

from dataclasses import dataclass
import time
import numpy as np
from ..coordinates import RESOLUTION


@dataclass(frozen=True, order=True)
class Cell:
    origin: tuple[int, int, int]
    side: int

    @property
    def key(self):
        return f"{self.origin[0]},{self.origin[1]},{self.origin[2]}/{self.side}"

    @property
    def volume(self):
        return (self.side / RESOLUTION) ** 3

    def children(self):
        if self.side < 2 or self.side % 2:
            raise ValueError("Cell cannot be split into equal integer cubes")
        h = self.side // 2
        return tuple(
            Cell((self.origin[0] + x, self.origin[1] + y, self.origin[2] + z), h)
            for z in (0, h)
            for y in (0, h)
            for x in (0, h)
        )

    def sample(self, k, rng):
        return (
            np.asarray(self.origin, dtype=np.float64) + rng.random((k, 3)) * self.side
        ) / RESOLUTION


def initial_cells():
    return Cell((0, 0, 0), RESOLUTION).children()


class UniformRefinement:
    """Breadth-first, coordinate-stable baseline."""

    def choose(self, records, rng):
        return min(records, key=lambda r: (-r["cell"].side, r["cell"].origin))["cell"]


class VariationRefinement:
    """Heuristic variation + coverage, with explicit random exploration."""

    def __init__(self, exploration=0.25):
        if not 0 <= exploration <= 1:
            raise ValueError("Exploration fraction must be in [0,1]")
        self.exploration = exploration

    def choose(self, records, rng):
        if rng.random() < self.exploration:
            return records[int(rng.integers(len(records)))]["cell"]

        def score(r):
            variation = r["cost_range"] if r["cost_range"] is not None else 1.0
            return (
                variation + 1 / (r["valid_samples"] + 1),
                r["cell"].side,
                tuple(-v for v in r["cell"].origin),
            )

        return max(records, key=score)["cell"]


class OctreeTeacher:
    def __init__(
        self,
        evaluator,
        *,
        k=4,
        seed=0,
        max_candidates=64,
        max_pairs=4096,
        min_side=64,
        refinement=None,
    ):
        if (
            k < 1
            or max_candidates < 1
            or max_pairs < 1
            or min_side not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        ):
            raise ValueError("Invalid diagnostic budget or source-space minimum side")
        self.evaluator = evaluator
        self.k = k
        self.seed = seed
        self.max_candidates = max_candidates
        self.max_pairs = max_pairs
        self.min_side = min_side
        self.refinement = refinement or UniformRefinement()

    def run(self, max_refinements=1):
        if max_refinements < 0:
            raise ValueError("Nonnegative refinement count required")
        rng = np.random.default_rng(self.seed)
        records = {}
        trace = []
        pending = []
        candidates = 0
        pairs = 0
        valid_evaluations = 0
        start = time.perf_counter()
        cpu = time.process_time()
        nprobes = len(self.evaluator.probes.observed)

        def evaluate(cell):
            nonlocal candidates, pairs, valid_evaluations
            if (
                candidates + self.k > self.max_candidates
                or pairs + self.k * nprobes > self.max_pairs
            ):
                pending.append(cell)
                return False
            points = cell.sample(self.k, rng)
            cost, result = self.evaluator.evaluate(points)
            candidates += self.k
            pairs += result.evaluated_pairs
            valid_evaluations += int(result.validity.valid.sum())
            if pairs > self.max_pairs:
                raise AssertionError("Oracle exceeded global pair budget")
            finite = cost[np.isfinite(cost)]
            r = {
                "cell": cell,
                "sample_xyz": points.tolist(),
                "costs": [float(v) if np.isfinite(v) else None for v in cost],
                "source_status": list(result.validity.status),
                "source_reasons": list(result.validity.reasons),
                "valid_samples": len(finite),
                "invalid_samples": int((~result.validity.valid).sum()),
                "mean_cost": float(finite.mean()) if len(finite) else None,
                "cost_range": float(np.ptp(finite)) if len(finite) else None,
                "cost_std": float(finite.std()) if len(finite) else None,
                "sampling_uncertainty_proxy": float(
                    1 / np.sqrt(max(1, len(finite) * nprobes))
                ),
                "region_certification": "UNRESOLVED",
                "pruned": False,
            }
            records[cell.key] = r
            trace.append(r)
            return True

        for cell in initial_cells():
            evaluate(cell)
        refined = []
        for _ in range(max_refinements):
            choices = [
                r
                for r in records.values()
                if r["cell"].side > self.min_side and r["cell"].key not in refined
            ]
            if not choices or pending:
                break
            chosen = self.refinement.choose(choices, rng)
            refined.append(chosen.key)
            for child in chosen.children():
                evaluate(child)

        def export(r):
            c = r["cell"]
            return {
                **{k: v for k, v in r.items() if k != "cell"},
                "origin_xyz": list(c.origin),
                "side": c.side,
                "world_volume": c.volume,
            }

        return {
            "dataset_name": "RIND3D",
            "execution_complete": not pending,
            "teacher_support_complete": False,
            "source_space_coverage_certified": False,
            "termination": "declared_refinement_limit"
            if not pending
            else "query_budget_incomplete",
            "seed": self.seed,
            "k_per_visited_cell": self.k,
            "observation_probes": nprobes,
            "candidate_attempts": candidates,
            "valid_candidate_evaluations": valid_evaluations,
            "physical_source_receiver_pairs": pairs,
            "maximum_candidates": self.max_candidates,
            "maximum_pairs": self.max_pairs,
            "cells_visited": len(trace),
            "cells_refined": len(refined),
            "refined_cell_keys": refined,
            "pending_unresolved_cells": [
                {"origin_xyz": list(c.origin), "side": c.side} for c in pending
            ],
            "trace": [export(r) for r in trace],
            "wall_seconds": time.perf_counter() - start,
            "cpu_seconds": time.process_time() - cpu,
            "refinement_policy": type(self.refinement).__name__,
            "uncertainty_scope": "Descriptive proxy, not a certified confidence bound or safe-pruning rule.",
            "distribution": "No teacher probability target constructed from mixed-size trace cells.",
        }
