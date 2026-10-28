import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rind_dataset.constants import PARAM_WIDTH
from mask_only_data import local_decomposition, prepare_mask_dataset, MaskRecords, build_mask_cache
from train_decomposition import train, make_model, load_config
from evaluate_masks import evaluate


def example(values=(.3, .8, .5)):
    response = np.zeros((8, 8), dtype=np.float64)
    response[:7, :2] = values[0]
    response[:7, 2:4] = values[1]
    response[:7, 4:] = values[2]
    return response


class LocalDataset:
    num_scenes = 3
    manifest = {"global_size": 1024, "num_scenes": 3}
    scene_splits = {"train": [0], "validation": [1], "test": [2]}
    def iter_scene_views(self, scene):
        return iter([(0, 8), (1, 8)])
    def get_scene(self, scene):
        # No source positions, intensities or source labels are exposed.
        return {"obstacle_types": np.empty(0, dtype=np.uint8),
                "obstacle_params": np.empty((0, PARAM_WIDTH)),
                "local_views": np.array([[0, 0, 8], [16, 0, 8]])}
    def get_region(self, scene, x, y, side):
        response = np.ones((side, side)) if x else example()
        return {"response": response, "obstacle": np.zeros_like(response, dtype=bool)}
    def get_channels(self, *args):
        raise AssertionError("per-source truth must never decide mask labels")
    get_source_params = get_channels


class MasksOnlyTests(unittest.TestCase):
    def _config(self, root):
        config = json.loads((Path(__file__).resolve().parents[2] / "configs/masks_only.json").read_text())
        config["data_root"] = str(root / "raw_not_installed")
        config["output_root"] = str(root / "output")
        config["label_preparation"].update(record_counts={"train": 1, "validation": 1, "test": 1},
                                          window_sizes=[8], max_views_per_scene=2)
        config["model"].update(max_slots=2, hidden=64, fourier_frequencies=1,
                               decomposition_hidden=32, slot_decoder_layers=1)
        config["training"].update(epochs=2, patience=3, batch_size=1, target_global_batch_size=1,
                                  amp_dtype="none", device="cpu", num_threads=1,
                                  prefetch_batches=0, progress_every_batches=1)
        return config

    def test_minimum_layer_examples_and_disconnected_equal_values(self):
        geometry = LocalDataset().get_scene(0)
        for values, expected in (((.3, .8, .5), (True, True, False)),
                                 ((.3, .8, .3), (True, True, True))):
            response = example(values)
            result = local_decomposition(response, np.zeros_like(response, bool), [0, 0, 8],
                                         geometry["obstacle_types"], geometry["obstacle_params"])
            self.assertEqual(result["status"], "complete")
            np.testing.assert_array_equal(result["masks"][0, 2, [0, 2, 5]], expected)
            self.assertEqual(result["candidate_renders"], 0)
            self.assertFalse(result["physical_single_source_checked"])
        response = np.zeros((8, 8))
        response[:3, :2] = response[5:, 6:] = .3
        result = local_decomposition(response, np.zeros_like(response, bool), [0, 0, 8],
                                     geometry["obstacle_types"], geometry["obstacle_params"])
        self.assertEqual(len(result["masks"]), 2)
        self.assertFalse(np.any(result["masks"][0] & result["masks"][1]))

    def test_uninformative_windows_are_skipped(self):
        geometry = LocalDataset().get_scene(0)
        for response, obstacle, reason in ((np.ones((8, 8)), np.zeros((8, 8), bool), "all_lit"),
                                           (np.zeros((8, 8)), np.zeros((8, 8), bool), "all_dark"),
                                           (np.zeros((8, 8)), np.ones((8, 8), bool), "no_free_pixels")):
            result = local_decomposition(response, obstacle, [0, 0, 8],
                                         geometry["obstacle_types"], geometry["obstacle_params"])
            self.assertEqual(result["reason"], reason)

    def test_label_cache_train_resume_evaluate_without_locator_or_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            output = root / "output"
            # Fail immediately if either candidate rendering or Phase1 is used.
            with patch("rind_phase1.model.SourceEnergyField.__init__", side_effect=AssertionError("Phase1 disabled")), \
                 patch("rind_phase2.physics.JointResponseEvaluator.__init__", side_effect=AssertionError("candidate rendering disabled")), \
                 patch("layered_residual.uniform_source_grid", side_effect=AssertionError("candidate pool disabled")):
                index = prepare_mask_dataset(config, LocalDataset())
                reused = prepare_mask_dataset(config, LocalDataset())
                self.assertEqual(index["corpus_sha256"], reused["corpus_sha256"])
                self.assertEqual(reused["counts"], {"train": 1, "validation": 1, "test": 1})
                self.assertEqual(index["candidate_renders"], 0)
                record = MaskRecords(index["records"]["train"], output / "mask_cache", "train", 2)[0]
                self.assertNotIn("teacher_probability", record)
                self.assertNotIn("candidate_xy", record)
                train(config, output, index)
                # Re-preparing changes reuse statistics, but must not invalidate resume.
                train(config, output, reused)
                metrics = evaluate(config, output, reused, repetitions=5)
                self.assertFalse(metrics["location_evaluated"])
                self.assertIn("mask_iou", metrics["window_weighted"])
                saved = torch.load(output / "models/seed_0/best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(saved["schema"], "phase2-local-mask-decomposition-v1")
                self.assertFalse(any(key.startswith("locator.") for key in saved["model"]))
                self.assertEqual(saved["config"]["task"], "decomposition_only")
                report = json.loads((output / "models/seed_0/training.json").read_text())
                self.assertEqual(len(report["history"]), 2)
            # Later full localization uses the exact same learned decomposer.
            full = copy.deepcopy(config)
            full.pop("task")
            from rind_phase1.model import SourceEnergyField
            locator = make_model(full, 1024, torch.device("cpu"), initialize=False)
            phase1 = root / "phase1.pt"
            torch.save({"model": locator.locator.state_dict(), "model_config": locator.backbone_config}, phase1)
            full["training"]["initial_phase1_checkpoint"] = str(phase1)
            full["training"]["initial_decomposition_checkpoint"] = str(output / "models/seed_0/best.pt")
            warm = make_model(full, 1024, torch.device("cpu"), initialize=True)
            for key, value in saved["model"].items():
                torch.testing.assert_close(warm.state_dict()[key], value)

    def test_cache_rejects_corrupt_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            source = root / "bad.npz"
            np.savez(source, response=np.ones((8, 8)), obstacle=np.zeros((8, 8), bool),
                     masks=np.zeros((1, 8, 8), bool), intensities=np.array([.3]), window=[0, 0, 8])
            labels = {"train": [{"file": str(source), "scene_id": 0, "view_id": 0, "size": 8}],
                      "validation": [], "test": []}
            with self.assertRaisesRegex(ValueError, "invalid complete local mask"):
                build_mask_cache(config, labels, root / "output")


if __name__ == "__main__":
    unittest.main()
