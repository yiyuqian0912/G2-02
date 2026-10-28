import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_masks_only as fixture
from mask_only_data import prepare_mask_dataset, MaskRecords
from expand_masks import prepare_expanded
from train_decomposition import make_model, train
from evaluate_masks import evaluate


class Dataset(fixture.LocalDataset):
    num_scenes = 5
    manifest = {"global_size": 1024, "num_scenes": 5}
    driver_counts = np.array([1, 2, 3, 2, 3])
    def __init__(self):
        self.scene_splits = {"train": [0, 3, 4], "validation": [1], "test": [2]}
        self.calls = []
    def iter_scene_views(self, scene):
        return iter([(view, 8) for view in range(4)])
    def get_scene(self, scene):
        row = super().get_scene(scene)
        row["local_views"] = np.array([[8 * view, 0, 8] for view in range(4)])
        return row
    def get_region(self, scene, x, y, side):
        self.calls.append((scene, x // 8))
        response = np.zeros((8, 8)) if x == 16 else fixture.example()
        return {"response": response, "obstacle": np.zeros((8, 8), bool)}
    def source_metadata(self, scene):
        return {"component_root": "old-release" if scene < 3 else "new-release"}


class ExpandMasksTests(unittest.TestCase):
    def _prepare(self, root):
        dataset = Dataset()
        config = fixture.MasksOnlyTests()._config(root)
        config["label_preparation"].update(record_counts={"train": 2, "validation": 1, "test": 1}, max_views_per_scene=4)
        base = prepare_mask_dataset(config, dataset)
        expanded = copy.deepcopy(config)
        expanded["output_root"] = str(root / "expanded")
        expanded["label_preparation"].update(reuse_from=config["output_root"],
                                               record_counts={"train": 8, "validation": 1, "test": 1},
                                               chunk_size=2)
        dataset.calls.clear()
        return dataset, config, expanded, base

    def test_append_resume_frozen_holdout_and_no_old_npz_or_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, old_config, config, old = self._prepare(root)
            base, output = Path(old_config["output_root"]), Path(config["output_root"])
            original = {p.name: p.read_bytes() for p in (base / "mask_cache").iterdir()}
            shutil.rmtree(base / "local_labels")
            with patch("rind_phase1.model.SourceEnergyField.__init__", side_effect=AssertionError("Phase1 disabled")), \
                 patch("rind_phase2.physics.JointResponseEvaluator.__init__", side_effect=AssertionError("renderer disabled")), \
                 patch("layered_residual.uniform_source_grid", side_effect=AssertionError("grid disabled")):
                with self.assertRaises(InterruptedError):
                    prepare_expanded(config, dataset, stop_after_chunks=1)
                self.assertFalse((output / "mask_records.json").exists())
                index = prepare_expanded(config, dataset)
                self.assertEqual(index["counts"], {"train": 8, "validation": 1, "test": 1})
                self.assertEqual(index["expansion"]["base_train_windows"], 2)
                self.assertEqual(index["expansion"]["new_train_windows"], 6)
                self.assertEqual(index["records"]["validation"], old["records"]["validation"])
                self.assertEqual(index["records"]["test"], old["records"]["test"])
                self.assertEqual(index["records"]["train"][:2], old["records"]["train"])
                self.assertEqual(len(dataset.calls), len(set(dataset.calls)))
                self.assertTrue(set(row["scene_id"] for row in index["records"]["train"]) <= {0, 3, 4})
                self.assertGreater(index["expansion"]["train_windows_by_component"].get("new-release", 0), 0)
                for name, content in original.items():
                    self.assertEqual((base / "mask_cache" / name).read_bytes(), content)
                    merged = (output / "mask_cache" / name).read_bytes()
                    self.assertEqual(merged[:len(content)], content)
                    if not name.startswith("train_"):
                        self.assertEqual(merged, content)
                records = MaskRecords(index["records"]["train"], output / "mask_cache", "train", 2)
                for i in range(len(records)):
                    row = records[i]
                    np.testing.assert_allclose((row["teacher_mask"] * row["teacher_intensity"][:, None, None]).sum(0), row["response"], atol=1e-7)
                dataset.calls.clear()
                again = prepare_expanded(config, dataset)
                self.assertEqual(again, index)
                self.assertEqual(dataset.calls, [])
                # A later growth stage can reuse this expanded cache as well.
                next_config = copy.deepcopy(config)
                next_config["output_root"] = str(root / "expanded_again")
                next_config["label_preparation"]["reuse_from"] = str(output)
                next_config["label_preparation"]["record_counts"]["train"] = 9
                next_index = prepare_expanded(next_config, dataset)
                self.assertEqual(next_index["counts"]["train"], 9)
                self.assertEqual(next_index["records"]["train"][:8], index["records"]["train"])
                previous_ids = {(r["scene_id"], r["view_id"]) for r in index["records"]["train"]}
                self.assertFalse(previous_ids & set(dataset.calls))

    def test_frozen_count_or_dataset_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, _, config, _ = self._prepare(Path(temporary))
            config["label_preparation"]["record_counts"]["validation"] = 2
            with self.assertRaisesRegex(ValueError, "must remain locked"):
                prepare_expanded(config, dataset)
            config["label_preparation"]["record_counts"]["validation"] = 1
            dataset.manifest = {**dataset.manifest, "release": "changed"}
            with self.assertRaisesRegex(ValueError, "dataset release differs"):
                prepare_expanded(config, dataset)

    def test_concurrent_preparation_is_rejected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temporary:
            dataset, _, config, _ = self._prepare(Path(temporary))
            output = Path(config["output_root"])
            output.mkdir()
            with (output / ".expansion.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    prepare_expanded(config, dataset)

    def test_corrupted_chunk_cannot_be_used_for_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, _, config, _ = self._prepare(Path(temporary))
            with self.assertRaises(InterruptedError):
                prepare_expanded(config, dataset, stop_after_chunks=1)
            output = Path(config["output_root"])
            path = output / "append_chunks/chunk-000000.npz"
            with np.load(path, allow_pickle=False) as z:
                payload = {key: np.array(z[key]) for key in z.files}
            payload["response"][0] += 1.0
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare_expanded(config, dataset)
            self.assertFalse((output / "mask_records.json").exists())

    def test_warm_start_baseline_is_retained_and_metrics_reuse_locked_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, old_config, config, _ = self._prepare(root)
            output = Path(config["output_root"])
            index = prepare_expanded(config, dataset)
            checkpoint = root / "old_best.pt"
            torch.manual_seed(0)
            model = make_model(old_config, 1024, torch.device("cpu"), initialize=False)
            torch.save({"schema": "phase2-local-mask-decomposition-v1", "config": old_config,
                        "model": model.state_dict()}, checkpoint)
            config["training"].update(initial_decomposition_checkpoint=str(checkpoint),
                                       evaluate_initial_checkpoint=True, epochs=1, learning_rate=1e-30)
            train(config, output, index)
            summary = json.loads((output / "training_summary.json").read_text())["0"]
            self.assertEqual(summary["best_epoch"], 0)
            self.assertIsNotNone(summary["initial_validation"])
            saved = torch.load(output / "models/seed_0/best.pt", map_location="cpu", weights_only=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved["model"][name])
            report = evaluate(config, output, index, repetitions=3)
            self.assertEqual(report["windows"], 1)
            self.assertEqual(report["best_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
