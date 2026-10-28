import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mask_only_data import build_mask_cache
from mask_validation import validate_mask_cache
import test_masks_only as mask_fixture
from train_decomposition import load_config


class MaskValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = mask_fixture.MasksOnlyTests()._config(self.root)
        self.output = self.root / "output"
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, :7, :4] = True
        masks[1, :7, 2:] = True
        labels = {}
        for scene, split in enumerate(("train", "validation", "test")):
            path = self.root / f"{split}.npz"
            np.savez(path, response=mask_fixture.example(), masks=masks, intensities=[.3, .5],
                     obstacle=np.zeros((8, 8), bool), window=[0, 0, 8])
            labels[split] = [{"file": str(path), "scene_id": scene, "view_id": 0,
                              "size": 8, "layer_count": 2}]
        self.index = build_mask_cache(self.config, labels, self.output)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_cache_is_accepted_without_raw_data(self):
        validate_mask_cache(self.index, self.output / "mask_cache", 2)

    def test_truncation_rejected_before_model_allocation(self):
        path = self.output / "mask_cache/train_response.f32"
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_mask_cache(self.index, self.output / "mask_cache", 2)

    def test_offsets_duplicates_scene_leakage_and_coordinates_rejected(self):
        cases = (("response_offset", 1, "non-contiguous"),
                 ("layer_count", 3, "layer count"),
                 ("window", [1020, 0, 8], "coordinates"))
        for key, value, message in cases:
            index = copy.deepcopy(self.index)
            index["records"]["train"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate_mask_cache(index, self.output / "mask_cache", 2)
        index = copy.deepcopy(self.index)
        index["records"]["validation"][0]["scene_id"] = 0
        with self.assertRaisesRegex(ValueError, "scene leakage"):
            validate_mask_cache(index, self.output / "mask_cache", 2)
        index = copy.deepcopy(self.index)
        index["records"]["train"].append(dict(index["records"]["train"][0]))
        index["counts"]["train"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate window"):
            validate_mask_cache(index, self.output / "mask_cache", 2)

    def test_invalid_task_or_localization_loss_cannot_enable_teacher_fallback(self):
        for mutate, error in ((lambda c: c.update(task="decompositon_only"), "task must"),
                              (lambda c: c["loss"].update(location_weight=1), "must be zero")):
            config = copy.deepcopy(self.config)
            mutate(config)
            path = self.root / "bad.json"
            path.write_text(json.dumps(config))
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
