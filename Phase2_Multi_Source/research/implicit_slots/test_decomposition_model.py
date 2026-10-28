import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve()
for path in (HERE.parent, HERE.parents[3] / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decomposition_model import (ResponseDecompositionField, decomposition_training_loss,
                                 exact_batched_assignment, poisson_binomial_logits)


class DecompositionModelTests(unittest.TestCase):
    def test_gpu_compatible_hungarian_matches_bruteforce_with_dummy_rows(self):
        generator = torch.Generator().manual_seed(84)
        cost = torch.randn(6, 5, 5, generator=generator)
        cost[:, 3:, :] = torch.randn(6, 1, 5, generator=generator)
        assignment = exact_batched_assignment(cost)
        for batch in range(len(cost)):
            actual = cost[batch, torch.arange(5), assignment[batch]].sum().item()
            optimal = min(sum(float(cost[batch, row, column]) for row, column in enumerate(order))
                          for order in itertools.permutations(range(5)))
            self.assertAlmostEqual(actual, optimal, places=5)
            self.assertEqual(len(set(assignment[batch].tolist())), 5)

    def test_hungarian_handles_ties_and_32_slots(self):
        cost = torch.ones(2, 32, 32)
        cost[:, torch.arange(32), torch.arange(32)] = 0
        assignment = exact_batched_assignment(cost)
        self.assertTrue(torch.equal(assignment, torch.arange(32)[None].expand(2, -1)))
        tied = exact_batched_assignment(torch.zeros(1, 8, 8))
        self.assertEqual(len(set(tied[0].tolist())), 8)

    def test_count_distribution_agrees_with_independent_bernoulli(self):
        presence = torch.tensor([[0.2, 0.5, 0.8]], requires_grad=True)
        actual = poisson_binomial_logits(presence).exp()
        expected = torch.tensor([[0.08, 0.42, 0.42, 0.08]])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        (-poisson_binomial_logits(presence)[0, 2]).backward()
        self.assertTrue(torch.isfinite(presence.grad).all())

    def test_overlapping_teacher_layers_are_reconstructed_without_exclusivity(self):
        # Original three-region example: middle pixel belongs to both layers.
        target = torch.tensor([[[[1.0, 1.0, 0.0]], [[0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]]])
        intensity = torch.tensor([[0.3, 0.5, 0.0]])
        response = (target * intensity[..., None, None]).sum(1)
        # Permute the model's slots; exact matching must recover this order.
        permutation = torch.tensor([1, 2, 0])
        mask = target[:, permutation]
        a = torch.tensor([[0.5, 0.7, 0.3]])
        rho = torch.tensor([[0.99999, 0.00001, 0.99999]])
        output = {
            "mask_probability": mask,
            "mask_logits": torch.logit(mask.clamp(1e-5, 1 - 1e-5)),
            "intensity": a, "presence_probability": rho,
            "presence_logits": torch.logit(rho),
            "reconstruction": (mask * (a * rho)[..., None, None]).sum(1),
            "count_logits": poisson_binomial_logits(rho),
        }
        batch = {
            "teacher_mask": target, "teacher_intensity": intensity,
            "layer_mask": torch.tensor([[True, True, False]]),
            "localization_valid": torch.zeros(1, 3, dtype=torch.bool),
            "response": response, "obstacle": torch.zeros_like(response, dtype=torch.bool),
        }
        result = decomposition_training_loss(output, batch)
        self.assertEqual(result["assignment"][0].tolist(), [2, 0, 1])
        self.assertLess(float(result["reconstruction"]), 1e-5)
        self.assertLess(float(result["residual"]), 1e-5)
        self.assertAlmostEqual(float(result["mask_dice"]), 1.0, places=5)
        self.assertEqual(float(result["count_accuracy"]), 1.0)

    def test_frozen_phase1_keeps_mask_input_gradient(self):
        torch.manual_seed(7)
        model = ResponseDecompositionField(
            max_slots=2, decomposition_hidden=32, slot_decoder_layers=1,
            slot_attention_heads=4,
            backbone_config={
                "hidden": 64, "fourier_frequencies": 1, "coordinate_scale": 16.0,
                "use_obstacle": False, "pixel_coordinates": True,
                "normalization_support": "world", "architecture": "multiscale_boundary",
                "attention_heads": 8,
            },
        ).train()
        model.set_locator_trainable(False)
        response = torch.zeros(1, 16, 16)
        response[:, :, 4:12] = 0.8
        obstacle = torch.zeros_like(response, dtype=torch.bool)
        window = torch.tensor([[0.0, 0.0, 16.0]])
        candidate = torch.tensor([[[2.0, 2.0], [6.0, 6.0], [10.0, 10.0], [14.0, 14.0]]])
        for amp_enabled in (False, True):
            with self.subTest(bfloat16=amp_enabled):
                model.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp_enabled):
                    output = model(response, obstacle, window, candidate,
                                   torch.ones(1, 4, dtype=torch.bool), 2)
                loss = -output["slot_log_probability"][0, 0, 0]
                loss.backward()
                self.assertIsNotNone(model.mask_embedding.weight.grad)
                self.assertTrue(torch.isfinite(model.mask_embedding.weight.grad).all())
                self.assertGreater(float(model.mask_embedding.weight.grad.abs().sum()), 0)
                self.assertTrue(all(parameter.grad is None for parameter in model.locator.parameters()))
                self.assertTrue(torch.allclose(output["slot_probability"].sum(-1), torch.ones(1, 2)))
                self.assertTrue(torch.equal(output["mask_probability"][..., :4],
                                            torch.zeros_like(output["mask_probability"][..., :4])))


if __name__ == "__main__":
    unittest.main()
