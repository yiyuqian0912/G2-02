from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve()
for path in (HERE.parent, HERE.parents[3] / "src",
             HERE.parents[3].parent / "Phase2_Multi_Source/src",
             HERE.parents[3].parent / "Phase2_Multi_Source/vendor/rind-dataset"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_decomposition import evaluate, gpu_distribution_metrics, ratio_of_means
from evaluate_slots import distribution_metrics
from decomposition_data import build_decomposition_index
from train_decomposition import make_model
import test_decomposition_data


class DecompositionEvaluationTests(unittest.TestCase):
    def test_device_distribution_metrics_match_reference(self):
        q = torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.4, 0.4, 0.1, 0.1]])
        p = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.2, 0.4, 0.1, 0.3]])
        candidate = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        actual = gpu_distribution_metrics(q, p, candidate, valid)
        for row in range(len(q)):
            expected = distribution_metrics(q[row].numpy(), p[row].numpy(), candidate.numpy(), valid[row].numpy())
            for key, value in expected.items():
                with self.subTest(row=row, metric=key):
                    self.assertAlmostEqual(float(actual[key][row]), value, places=9)

    def test_empty_localization_rows_are_valid(self):
        metrics = gpu_distribution_metrics(
            torch.empty(0, 4), torch.empty(0, 4), torch.empty(0, 4, 2),
            torch.empty(0, 4, dtype=torch.bool))
        self.assertEqual(metrics["forward_kl"].shape, (0,))

    def test_information_recovery_uses_ratio_of_means(self):
        summary = {
            "uniform_cross_entropy": {"mean": 10.0},
            "cross_entropy": {"mean": 8.0},
            "teacher_entropy": {"mean": 6.0},
        }
        self.assertEqual(ratio_of_means(summary), 0.5)
        self.assertIsNone(ratio_of_means({}))

    def test_cache_only_evaluation_separates_oracle_and_predicted_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher, config, _ = test_decomposition_data.CompleteBookDataTests()._teacher(root)
            output = root / "output"
            config.update({"data_root": str(root / "raw_is_absent"),
                           "teacher_experiment_root": str(teacher), "output_root": str(output)})
            config["model"] = {
                "max_slots": 4, "hidden": 64, "fourier_frequencies": 1,
                "decomposition_hidden": 32, "slot_attention_heads": 4,
                "slot_decoder_layers": 1, "use_obstacle": False,
            }
            config["training"].update({"device": "cpu", "amp_dtype": "none",
                                       "candidate_chunk": 2, "batch_size": 1,
                                       "prefetch_batches": 0})
            config["loss"] = {}
            index = build_decomposition_index(config, None, teacher, output)
            model = make_model(config, index["global_size"], torch.device("cpu"), initialize=False)
            checkpoint = output / "models/seed_0/best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"schema": "phase2-response-decomposition-v2", "config": config,
                        "model": model.state_dict(), "epoch": 9}, checkpoint)
            report = evaluate(config, seed=0, bootstrap_repetitions=0)
            self.assertEqual(report["coverage"]["windows"], 1)
            self.assertEqual(report["coverage"]["full_layers"], 2)
            self.assertEqual(report["coverage"]["localizable_layers"], 1)
            self.assertIn("predicted_mask_localization", report)
            self.assertIn("oracle_mask_localization", report)
            self.assertEqual(report["coverage"]["all_layers_localizable_windows"], 0)
            self.assertTrue((output / "evaluation/decomposition/REPORT.zh-CN.md").is_file())


if __name__ == "__main__":
    unittest.main()
