import sys
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve()
PHASE2_ROOT = HERE.parents[2]
PACKAGE_ROOT = PHASE2_ROOT.parent
for path in (HERE.parent, PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slot_data import WindowSlotRecords, build_slot_index, highest_density_mask, slot_batch_count
from slot_model import (ImplicitContributionSlotField, masked_log_softmax,
                        sinkhorn_assignment, slot_training_loss)
from evaluate_slots import distribution_metrics, exact_assignment


class ImplicitSlotTests(unittest.TestCase):
    def test_index_builds_memmap_cache_without_changing_layer_targets(self):
        class Dataset:
            def get_observation(self, scene, view):
                return {
                    "response": np.array([[0.0, 0.3], [0.3, 0.3]], dtype=np.float64),
                    "obstacle": np.zeros((2, 2), dtype=bool),
                    "window": np.array([0, 0, 2]),
                }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            records = {name: [] for name in ("train", "validation", "test")}
            probability = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float64)
            for scene, split in enumerate(records, 1):
                relative = Path("teachers") / split / f"scene{scene}-view0"
                directory = teacher / relative
                directory.mkdir(parents=True)
                records[split].append({"scene_id": scene, "view_id": 0, "size": 2,
                                       "directory": str(relative)})
                (directory / "report.json").write_text(json.dumps({
                    "status": "complete", "partial_combined_distribution": False,
                    "layers_recovered": 1, "layers": [{"step": 1, "intensity": 0.3}],
                }))
                axis = np.array([0.5, 1.5])
                candidate = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
                np.savez(directory / "layer-01.npz", layer_response=np.array([[0, 1], [1, 1]], dtype=bool),
                         candidate_xy=candidate, grid_index=np.arange(4), x_axis=axis, y_axis=axis,
                         valid=np.ones(4, dtype=bool), probability=probability,
                         physical_cost=np.array([0.0, 0.1, 0.2, 0.3]))
            teacher.mkdir(exist_ok=True)
            (teacher / "records.json").write_text(json.dumps({"records": records}))
            config = {"layer_filter": {"maximum_training_physical_cost": 0.01},
                      "model": {"max_slots": 8}}
            output = root / "output"
            index = build_slot_index(config, Dataset(), teacher, output)
            split = WindowSlotRecords(index["records"]["train"], Dataset(),
                                      output / "training_cache", "train", 8, "world")
            record = split[0]
            self.assertTrue(np.allclose(record["teacher_probability"][0], probability))
            self.assertEqual(int(record["count_target"]), 1)
            self.assertTrue(bool(record["slot_supervised"]))
            self.assertTrue((output / "training_cache/train_probability.f32").is_file())

    def test_slot_probabilities_are_independently_normalized(self):
        energy = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])
        support = torch.tensor([[True, True, False]])
        log_p, p = masked_log_softmax(energy, support)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(1, 2)))
        self.assertTrue(torch.equal(p[..., 2], torch.zeros(1, 2)))
        self.assertTrue(torch.isfinite(log_p[..., :2]).all())

    def test_sinkhorn_is_doubly_stochastic(self):
        cost = torch.tensor([[[0.0, 3.0], [4.0, 0.0]]])
        assignment = sinkhorn_assignment(cost, temperature=0.1, iterations=50)
        self.assertTrue(torch.allclose(assignment.sum(1), torch.ones(1, 2), atol=1e-5))
        self.assertTrue(torch.allclose(assignment.sum(2), torch.ones(1, 2), atol=1e-5))
        self.assertGreater(float(assignment[0, 0, 0]), 0.99)

    def test_highest_density_mask_is_tie_inclusive(self):
        mask = highest_density_mask(np.array([0.5, 0.25, 0.25, 0.0]), 0.75)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_perfect_two_slot_prediction_has_low_matching_loss(self):
        q = torch.tensor([[[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]]])
        log_p = torch.log(q.clamp_min(1e-30))
        output = {
            "slot_log_probability": log_p,
            "slot_probability": q,
            "presence_logits": torch.full((1, 2), 8.0),
            "presence_probability": torch.sigmoid(torch.full((1, 2), 8.0)),
            "intensity": torch.tensor([[0.3, 0.5]]),
            "count_logits": torch.tensor([[-8.0, -8.0, 8.0]]),
            "occupancy": 1.0 - torch.prod(1.0 - q, dim=1),
        }
        result = slot_training_loss(
            output, q, q > 0, torch.tensor([[0.3, 0.5]]),
            torch.tensor([[True, True]]), torch.tensor([True]),
            output["occupancy"], torch.tensor([2]),
            core_weight=0.0, intensity_weight=0.0, presence_weight=0.1,
            aggregate_weight=1.0, count_weight=0.1, consistency_weight=0.1,
            sinkhorn_temperature=0.05, sinkhorn_iterations=30,
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertLess(float(result["aggregate"]), 1e-6)
        self.assertEqual(float(result["count_accuracy"]), 1.0)

    def test_candidate_activation_checkpoint_has_finite_backward(self):
        model = ImplicitContributionSlotField(
            max_slots=2, slot_decoder_layers=1, slot_attention_heads=8,
            activation_checkpoint_candidates=True,
            backbone_config={
                "hidden": 64, "fourier_frequencies": 1,
                "coordinate_scale": 16.0, "use_obstacle": False,
                "pixel_coordinates": True, "normalization_support": "world",
                "architecture": "multiscale_boundary", "attention_heads": 8,
            },
        ).train()
        response = torch.zeros(1, 16, 16)
        response[:, :, 8:] = 1.0
        obstacle = torch.zeros_like(response, dtype=torch.bool)
        window = torch.tensor([[0.0, 0.0, 16.0]])
        candidate = torch.tensor([[[2.0, 2.0], [6.0, 6.0], [10.0, 10.0], [14.0, 14.0]]])
        output = model(response, obstacle, window, candidate,
                       torch.ones(1, 4, dtype=torch.bool), candidate_chunk=2)
        loss = output["energy"].square().mean() + output["presence_logits"].square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_slot_batch_count_matches_homogeneous_ddp_groups(self):
        class Records:
            window_sizes = [64] * 17 + [128] * 9
            candidate_counts = [16_384] * 26
        # ceil(17 / (2*3)) + ceil(9 / (2*3)) = 3 + 2.
        self.assertEqual(slot_batch_count(Records(), batch_size=2, world_size=3), 5)

    def test_exact_rectangular_assignment_is_injective_and_optimal(self):
        assignment = exact_assignment(np.array([
            [0.0, 5.0, 4.0],
            [3.0, 0.0, 2.0],
        ]))
        self.assertEqual(assignment.tolist(), [0, 1])

    def test_perfect_distribution_metrics_have_perfect_shape(self):
        probability = np.array([0.7, 0.2, 0.1, 0.0])
        candidate = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        metrics = distribution_metrics(
            probability, probability, candidate, np.array([True, True, True, False]))
        self.assertAlmostEqual(metrics["forward_kl"], 0.0, places=12)
        self.assertAlmostEqual(metrics["jensen_shannon"], 0.0, places=12)
        self.assertAlmostEqual(metrics["hdr_dice_90"], 1.0, places=12)
        self.assertAlmostEqual(metrics["information_recovery"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
