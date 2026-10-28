import json
from pathlib import Path
import tempfile
import unittest

import torch

from rind_phase1.model import SourceEnergyField
from decomposition_data import build_decomposition_index
import test_decomposition_data
from train_decomposition import train


class DecompositionPipelineTests(unittest.TestCase):
    def test_two_cpu_epochs_cross_frozen_locator_schedule_and_evaluate_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher, config, _ = test_decomposition_data.CompleteBookDataTests()._teacher(root)
            output = root / "output"
            config["model"] = {
                "max_slots": 2, "hidden": 64, "fourier_frequencies": 1,
                "use_obstacle": False, "decomposition_hidden": 32,
                "slot_decoder_layers": 1, "slot_attention_heads": 8,
            }
            locator_config = {
                "hidden": 64, "fourier_frequencies": 1, "coordinate_scale": 1024.0,
                "use_obstacle": False, "pixel_coordinates": True,
                "normalization_support": "world", "architecture": "multiscale_boundary",
                "attention_heads": 8,
            }
            checkpoint = root / "phase1.pt"
            torch.save({"model": SourceEnergyField(**locator_config).state_dict(),
                        "model_config": locator_config}, checkpoint)
            config["data_root"] = str(root / "unused_raw_data")
            config["teacher_experiment_root"] = str(teacher)
            config["output_root"] = str(output)
            config["training"] = {
                "seeds": [0], "epochs": 2, "patience": 3,
                "batch_size": 1, "target_global_batch_size": 2,
                "candidate_chunk": 2, "amp_dtype": "none", "device": "cpu",
                "mask_warmup_epochs": 0, "locator_frozen_epochs": 1,
                "learning_rate": 1e-4, "locator_learning_rate": 1e-5,
                "weight_decay": 1e-4, "grad_clip": 1.0,
                "initial_phase1_checkpoint": str(checkpoint),
                "prefetch_batches": 0, "num_threads": 1,
                "activation_checkpoint_candidates": True,
                "activation_checkpoint_locator_encoder": True,
                "progress_every_batches": 1,
            }
            config["loss"] = {}
            index = build_decomposition_index(config, None, teacher, output)
            train(config, output, index)
            result_path = output / "models/seed_0/training.json"
            result = json.loads(result_path.read_text())
            self.assertEqual(len(result["history"]), 2)
            self.assertEqual(result["records"], {"train": 1, "validation": 1, "test": 1})
            self.assertTrue(torch.isfinite(torch.tensor(result["test"]["loss"])))
            self.assertTrue((output / "models/seed_0/best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
