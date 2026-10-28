import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/comprehensive_phase1_evaluation.py"
SPEC = importlib.util.spec_from_file_location("comprehensive_phase1_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComprehensiveEvaluationTests(unittest.TestCase):
    def test_identical_distribution_has_perfect_geometry(self):
        q = np.array([0.5, 0.3, 0.15, 0.05])
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        valid = np.ones(4, dtype=bool)
        result = MODULE._distribution_metrics(q, q, xy, valid, valid)
        self.assertAlmostEqual(result["forward_kl"], 0.0)
        self.assertAlmostEqual(result["jensen_shannon"], 0.0)
        self.assertAlmostEqual(result["overlap"], 1.0)
        for mass in (50, 75, 90, 95):
            self.assertAlmostEqual(result[f"support_dice_{mass}"], 1.0)
            self.assertAlmostEqual(result[f"support_iou_{mass}"], 1.0)
            self.assertAlmostEqual(result[f"polarized_core_overlap_{mass}"], 1.0)
            self.assertAlmostEqual(result[f"polarized_core_jensen_shannon_{mass}"], 0.0)

    def test_shape_precision_penalizes_extra_student_area(self):
        q = np.array([0.5, 0.5, 0.0, 0.0])
        p = np.full(4, 0.25)
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        valid = np.ones(4, dtype=bool)
        result = MODULE._distribution_metrics(q, p, xy, valid, valid)
        self.assertAlmostEqual(result["support_recall_90"], 1.0)
        self.assertAlmostEqual(result["support_precision_90"], 0.5)
        self.assertAlmostEqual(result["support_dice_90"], 2.0 / 3.0)

    def test_primary_core_polarization_removes_low_probability_tail(self):
        q = np.array([0.6, 0.2, 0.1, 0.1])
        p = np.array([0.6, 0.2, 0.19, 0.01])
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        valid = np.ones(4, dtype=bool)
        result = MODULE._distribution_metrics(q, p, xy, valid, valid)
        self.assertAlmostEqual(result["support_dice_75"], 1.0)
        self.assertAlmostEqual(result["polarized_core_overlap_75"], 1.0)
        self.assertGreater(result["total_variation"], 0.0)

    def test_invalid_mass_and_raw_vs_valid_peak_are_explicit(self):
        q = np.array([0.6, 0.4, 0.0])
        p = np.array([0.2, 0.3, 0.5])
        valid = np.array([True, True, False])
        cost = np.array([0.0, 0.2, np.nan])
        result = MODULE._physical_metrics(q, p, cost, valid, 0.01, 1.0)
        self.assertAlmostEqual(result["student_invalid_mass"], 0.5)
        self.assertEqual(result["student_raw_peak_is_valid"], 0.0)
        self.assertEqual(result["student_valid_peak_is_compatible"], 0.0)

    def test_coordinate_metrics_use_continuous_truth(self):
        xy = np.array([[8, 8], [24, 8], [8, 24]], dtype=float)
        q = np.array([0.8, 0.1, 0.1])
        p = np.array([0.2, 0.7, 0.1])
        valid = np.ones(3, dtype=bool)
        result = MODULE._coordinate_metrics(q, p, xy, valid, np.array([9, 9]), (8.0, 16.0, 32.0))
        self.assertAlmostEqual(result["teacher_peak_source_distance"], math.sqrt(2))
        self.assertEqual(result["teacher_recall_at_8"], 1.0)
        self.assertEqual(result["student_recall_at_8"], 0.0)
        self.assertEqual(result["student_recall_at_32"], 1.0)

    def test_cluster_bootstrap_uses_scene_clusters(self):
        rows = [
            {"scene_id": 1, "metrics": {"x": 0.0}},
            {"scene_id": 1, "metrics": {"x": 1.0}},
            {"scene_id": 2, "metrics": {"x": 2.0}},
        ]
        result = MODULE._cluster_bootstrap(rows, ["x"], 100, 0)["x"]
        self.assertEqual(result["independent_scene_clusters"], 2)
        self.assertLessEqual(result["lower"], result["estimate"])
        self.assertGreaterEqual(result["upper"], result["estimate"])


if __name__ == "__main__":
    unittest.main()
