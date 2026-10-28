import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_phase1_100k.py"
SPEC = importlib.util.spec_from_file_location("evaluate_phase1_100k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CurrentEvaluationTests(unittest.TestCase):
    def test_split_and_checkpoint_identities(self):
        rows = [
            {"scene_id": 1, "teacher_entropy": 1.0, "candidate_count": 16,
             "valid_candidates": 8},
            {"scene_id": 2, "teacher_entropy": 2.0, "candidate_count": 16,
             "valid_candidates": 8},
        ]
        summary = MODULE.split_statistics(rows, "world")
        self.assertAlmostEqual(summary["teacher_entropy_nats"]["mean"], 1.5)
        self.assertAlmostEqual(summary["uniform_cross_entropy_nats"]["mean"], math.log(16))
        training = {
            "history": [
                {"epoch": 1, "train_ce": 2.7, "validation_ce": 2.5},
                {"epoch": 2, "train_ce": 2.4, "validation_ce": 2.0},
            ],
            "best_validation_ce": 2.0,
        }
        result = MODULE.checkpoint_statistics(training, summary)
        self.assertEqual(result["best"]["epoch"], 2)
        self.assertAlmostEqual(result["best"]["validation_forward_kl_estimate_nats"], 0.5)
        expected = (math.log(16) - 2.0) / (math.log(16) - 1.5)
        self.assertAlmostEqual(result["best"]["validation_information_recovery"], expected)

    def test_end_to_end_cache_only_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = root / "response"
            seed = variant / "seed_0"
            seed.mkdir(parents=True)
            (root / "protocol.json").write_text(json.dumps({"config": {
                "training": {"normalization_support": "world"},
                "teacher": {"temperature": 0.05},
            }}))
            records = []
            for split, scene in (("train", 1), ("validation", 2), ("test", 3)):
                records.append({
                    "split": split, "scene_id": scene, "view_id": 0,
                    "teacher_entropy": 1.0, "candidate_count": 16,
                    "valid_candidates": 8,
                })
            (variant / "teachers.json").write_text(json.dumps({"records": records}))
            (seed / "training.json").write_text(json.dumps({
                "best_validation_ce": 2.0,
                "history": [{"epoch": 1, "train_ce": 2.1, "validation_ce": 2.0}],
            }))
            old = MODULE.parse_args
            MODULE.parse_args = lambda: type("Args", (), {
                "output_root": root, "variant": "response", "seed": 0,
                "run_label": None, "require_test": False,
            })()
            try:
                MODULE.main()
            finally:
                MODULE.parse_args = old
            report = json.loads((seed / "evaluation/current_evaluation.json").read_text())
            self.assertFalse(report["locked_test"]["available"])
            self.assertAlmostEqual(
                report["checkpoint"]["best"]["validation_forward_kl_estimate_nats"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
