import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rind_phase1.model import SourceEnergyField
from train_view_sets import (
    _ChunkedStudent, _make_units, _pad_tracks, _track_pairs, read_config)


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "layer_student_view_set.json"


class ViewSetTrainingTests(unittest.TestCase):
    def test_configuration_is_valid(self):
        config = read_config(CONFIG)
        self.assertEqual(config["training"]["normalization_support"], "world")
        self.assertGreater(config["training"]["set_loss_weight"], 0)

    def test_training_selects_one_track_per_scene(self):
        scenes = [
            {"scene_id": 3, "tracks": [
                {"entry_indices": [0, 1]}, {"entry_indices": [2, 3]}]},
            {"scene_id": 8, "tracks": [{"entry_indices": [4, 5, 6]}]},
        ]
        units = _make_units(
            scenes, training=True, rng=np.random.default_rng(7), max_views=2)
        self.assertEqual({unit["scene_id"] for unit in units}, {3, 8})
        self.assertEqual(len(units), 2)
        self.assertTrue(all(len(unit["indices"]) == 2 for unit in units))

    def test_evaluation_preserves_all_tracks(self):
        scenes = [{"scene_id": 3, "tracks": [
            {"entry_indices": [0, 1]}, {"entry_indices": [2, 3, 4]}]}]
        units = _make_units(
            scenes, training=False, rng=np.random.default_rng(0), max_views=4)
        self.assertEqual([unit["indices"] for unit in units], [[0, 1], [2, 3, 4]])

    def test_padding_keeps_track_boundaries(self):
        log_p = [torch.tensor([0.7, 0.3]).log(),
                 torch.tensor([0.6, 0.4]).log(),
                 torch.tensor([0.2, 0.8]).log()]
        teacher = [torch.tensor([0.8, 0.2]),
                   torch.tensor([0.5, 0.5]),
                   torch.tensor([0.1, 0.9])]
        units = [{"scene_id": 1, "indices": [4, 5]},
                 {"scene_id": 2, "indices": [8]}]
        student, target, mask = _pad_tracks(units, log_p, teacher)
        self.assertEqual(student.shape, (2, 2, 2))
        self.assertEqual(target.shape, (2, 2, 2))
        torch.testing.assert_close(mask, torch.tensor([[True, True], [True, False]]))

    def test_track_pairs_are_order_invariant(self):
        self.assertEqual(
            _track_pairs([[4, 2, 7], [9]]),
            {(2, 4), (4, 7), (2, 7)})

    def test_chunked_wrapper_matches_full_model(self):
        torch.manual_seed(4)
        model = SourceEnergyField(hidden=16, fourier_frequencies=2)
        wrapped = _ChunkedStudent.create(model, 3)
        response = torch.rand(2, 16, 16)
        obstacle = torch.zeros(2, 16, 16, dtype=torch.bool)
        window = torch.tensor([[0.0, 0.0, 16.0], [32.0, 48.0, 16.0]])
        candidate = torch.rand(2, 7, 2) * 1024
        model.eval()
        wrapped.eval()
        with torch.no_grad():
            expected = model(response, obstacle, window, candidate)
            actual = wrapped(response, obstacle, window, candidate)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
