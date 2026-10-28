import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from decomposition_data import (DecompositionRecords, batch_count,
                                build_decomposition_index, decomposed_batches)


class CompleteBookDataTests(unittest.TestCase):
    def _teacher(self, root, corrupt_train=False):
        teacher = root / "teacher"
        rows = {split: [] for split in ("train", "validation", "test")}
        masks = [np.array([[1, 1], [0, 0]], dtype=bool),
                 np.array([[0, 1], [0, 1]], dtype=bool)]
        strengths = [0.3, 0.5]
        original = sum(a * mask for a, mask in zip(strengths, masks))
        candidate = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
        for scene, split in enumerate(rows):
            relative = Path("teachers") / split / f"scene{scene}"
            directory = teacher / relative
            directory.mkdir(parents=True)
            rows[split].append({"scene_id": scene, "view_id": 0, "size": 2,
                                "window": [0, 0, 2], "directory": str(relative)})
            (directory / "report.json").write_text(json.dumps({
                "status": "complete", "partial_combined_distribution": False,
                "layers_recovered": 2,
                "layers": [{"step": step + 1, "intensity": value}
                           for step, value in enumerate(strengths)],
            }))
            np.savez(directory / "combined.npz", original=original,
                     obstacle=np.zeros((2, 2), dtype=bool), residual=np.zeros((2, 2)))
            residual = original.copy()
            for index, (strength, mask) in enumerate(zip(strengths, masks)):
                after = residual - strength * mask
                np.savez(
                    directory / f"layer-{index + 1:02d}.npz", layer_response=mask,
                    residual_before=residual,
                    residual_after=after + (0.1 if corrupt_train and split == "train" and index == 0 else 0),
                    candidate_xy=candidate, grid_index=np.arange(4),
                    x_axis=np.array([0.5, 1.5]), y_axis=np.array([0.5, 1.5]),
                    valid=np.ones(4, dtype=bool), probability=np.array([0.6, 0.2, 0.1, 0.1]),
                    physical_cost=np.arange(4) * 0.1 + (0.2 if index else 0.0),
                )
                residual = after
        (teacher / "records.json").write_text(json.dumps({"records": rows}))
        config = {"model": {"max_slots": 4}, "decomposition": {"arithmetic_tolerance": 1e-10},
                  "layer_filter": {"maximum_training_physical_cost": 0.01},
                  "training": {"normalization_support": "world"}}
        return teacher, config, original

    def test_complete_masks_keep_low_confidence_layers_and_train_without_raw_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher, config, original = self._teacher(root)
            output = root / "output"
            index = build_decomposition_index(config, None, teacher, output)
            records = DecompositionRecords(index["records"]["train"], output / "decomposition_cache", "train", 4)
            sample = records[0]
            self.assertEqual(int(sample["count_target"]), 2)
            self.assertEqual(sample["layer_mask"].tolist(), [True, True, False, False])
            self.assertEqual(sample["localization_valid"].tolist(), [True, False, False, False])
            reconstruction = np.sum(sample["teacher_mask"] * sample["teacher_intensity"][:, None, None], axis=0)
            np.testing.assert_allclose(reconstruction, original, atol=1e-7)
            self.assertTrue(index["contract"]["all_arithmetic_layers_kept"])
            self.assertFalse(index["contract"]["raw_data_required_for_training"])
            self.assertEqual(len(index["corpus_sha256"]), 64)

    def test_corrupted_residual_is_rejected_as_whole_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher, config, _ = self._teacher(root, corrupt_train=True)
            index = build_decomposition_index(config, None, teacher, root / "output")
            self.assertEqual(index["counts"]["train"], 0)
            self.assertEqual(index["statistics"]["train"]["invalid_book"], 1)
            self.assertIn("violates subtraction", index["excluded_windows"][0]["detail"])

    def test_evaluation_has_no_padding_or_duplicate_window(self):
        class Records:
            window_sizes = [2] * 7
            candidate_counts = [4] * 7
            def __getitem__(self, index):
                return {"response": np.zeros((2, 2), dtype=np.float32),
                        "scene_id": np.int64(index), "view_id": np.int64(0)}
        seen = []
        for rank in range(3):
            count = batch_count(Records(), 2, world_size=3, rank=rank, training=False)
            batches = list(decomposed_batches(Records(), 2, rank=rank, world_size=3, training=False))
            self.assertEqual(count, len(batches))
            seen.extend(int(index) for batch in batches for index in batch["scene_id"])
        self.assertEqual(sorted(seen), list(range(7)))


if __name__ == "__main__":
    unittest.main()
