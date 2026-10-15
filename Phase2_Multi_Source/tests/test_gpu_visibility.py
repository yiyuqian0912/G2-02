"""The batched Torch renderer must reproduce canonical RIND visibility."""
import unittest

import numpy as np

from rind_dataset.constants import (
    OBSTACLE_ELLIPSE,
    OBSTACLE_POLYGON,
    OBSTACLE_RECTANGLE,
    OBSTACLE_TRIANGLE,
    PARAM_WIDTH,
)
from rind_dataset.geometry import point_in_any_obstacle
from rind_dataset.render import render_region
from rind_phase2.gpu_visibility import TorchVisibilityRenderer
from rind_phase2.physics import JointResponseEvaluator
from rind_phase1.physics import boundary_weight_map, response_cost
from rind_phase1.teacher import teacher_probabilities


class TorchVisibilityTests(unittest.TestCase):
    def test_on_device_layer_cost_and_probability_match_phase1(self):
        import torch

        masks = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]]], dtype=torch.bool)

        class Renderer:
            device = torch.device('cpu')
            def __init__(self): self.torch = torch
            def positions_valid_tensor(self, positions):
                return torch.ones(len(positions), dtype=torch.bool)
            def render_tensor(self, positions):
                return masks[:len(positions)]

        evaluator = JointResponseEvaluator.__new__(JointResponseEvaluator)
        evaluator.resolved_backend = 'cuda'
        evaluator._gpu_renderer = Renderer()
        evaluator.observed_response = np.zeros((2, 2), dtype=np.float64)
        evaluator.gpu_batch_size = 2
        evaluator.gpu_verify_remaining = 0
        evaluator.gpu_validation_cpu_renders = 0
        evaluator.gpu_render_batches = 0
        evaluator.gpu_scored_candidate_layers = 0
        evaluator.num_source_renders = 0
        targets = np.array([[[1., 0.], [0., 0.]], [[0., 1.], [0., 0.]]])
        retained, costs, probabilities = evaluator.gpu_layer_response_costs(
            np.array([[1., 1.], [2., 2.]]), targets, None,
            np.zeros((2, 2), dtype=bool), alpha=1.7, temperature=.2,
            boundary_lambda=2., boundary_sigma=1.)
        expected_costs = np.array([
            [1.7 * response_cost(target, mask.numpy(), boundary_weight_map(
                target, boundary_lambda=2., boundary_sigma=1.)) for mask in masks]
            for target in targets
        ])
        expected_probabilities = np.array([
            teacher_probabilities(row, np.ones(2, dtype=bool), temperature=.2)
            for row in expected_costs
        ])
        np.testing.assert_array_equal(retained, [True, True])
        np.testing.assert_allclose(costs, expected_costs, rtol=0, atol=1e-14)
        np.testing.assert_allclose(probabilities, expected_probabilities, rtol=0, atol=1e-14)

    def test_cpu_vectorization_matches_reference_for_all_obstacle_types(self):
        types = np.array([
            OBSTACLE_RECTANGLE,
            OBSTACLE_ELLIPSE,
            OBSTACLE_TRIANGLE,
            OBSTACLE_POLYGON,
        ], dtype=np.uint8)
        params = np.zeros((len(types), PARAM_WIDTH), dtype=np.float64)
        params[0, :5] = [16.0, 16.0, 5.0, 9.0, 0.37]
        params[1, :5] = [47.0, 16.0, 8.0, 5.0, -0.52]
        params[2, :6] = [8.0, 48.0, 23.0, 38.0, 25.0, 57.0]
        polygon = [(39.0, 39.0), (56.0, 37.0), (60.0, 49.0),
                   (50.0, 45.0), (42.0, 58.0)]
        params[3, 0] = len(polygon)
        for index, (x, y) in enumerate(polygon):
            params[3, 1 + 2 * index:3 + 2 * index] = [x, y]

        obstacle = np.zeros((64, 64), dtype=bool)
        for row in range(64):
            for column in range(64):
                obstacle[row, column] = point_in_any_obstacle(
                    column + 0.5, row + 0.5, types, params, len(types))

        candidates = np.array([
            [2.0, 2.0], [32.0, 2.0], [62.0, 2.0], [2.0, 32.0],
            [62.0, 32.0], [2.0, 62.0], [32.0, 62.0], [62.0, 62.0],
        ], dtype=np.float64)
        renderer = TorchVisibilityRenderer(
            (0, 0, 64), obstacle, types, params, device="cpu", pixel_chunk=257)
        actual = renderer.render(candidates)
        expected = np.stack([
            render_region([point], params, obs_types=types, x=0, y=0,
                          size=64, backend="reference") == 1
            for point in candidates
        ])
        np.testing.assert_array_equal(actual, expected)
        rng = np.random.default_rng(20261012)
        points = rng.uniform(0.0, 64.0, size=(1000, 2))
        actual_valid = renderer.positions_valid_tensor(points).cpu().numpy()
        expected_valid = np.array([
            not point_in_any_obstacle(x, y, types, params, len(types))
            for x, y in points
        ])
        np.testing.assert_array_equal(actual_valid, expected_valid)


if __name__ == "__main__":
    unittest.main()
