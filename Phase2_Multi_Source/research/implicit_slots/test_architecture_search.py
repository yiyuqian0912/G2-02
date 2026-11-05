import copy
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import architecture_search as search
from search_cache import capacity_audit, stratified_subset, make_pilot, bind_cache
from mask_only_data import build_mask_cache, MaskRecords
from mask_validation import validate_mask_cache
from evaluate_masks import metric_rows, pad_missing_predictions
from decomposition_model import decomposition_training_loss, poisson_binomial_logits
from train_decomposition import train
import test_masks_only as fixture


def source_cache(root, *, overflow_test=False, train_views=2):
    template = fixture.MasksOnlyTests()._config(root)
    template["model"]["max_slots"] = 8
    template["training"].update(epochs=2, patience=3, target_global_batch_size=2)
    output = root / "source"
    labels = {}
    for scene, split in enumerate(("train", "validation", "test")):
        labels[split] = []
        for view in range(train_views if split == "train" else 1):
            masks = np.zeros((3 if overflow_test and split == "test" else 2, 8, 8), bool)
            masks[0, :7, :4], masks[1, :7, 2:] = True, True
            strengths = [.3, .5]
            if len(masks) == 3:
                masks[2, 7, 0] = True
                strengths.append(.2)
            response = (masks * np.array(strengths)[:, None, None]).sum(0)
            path = root / f"{split}_{view}.npz"
            np.savez(path, response=response, masks=masks, intensities=strengths,
                     obstacle=np.zeros((8, 8), bool), window=[0, 0, 8])
            labels[split].append({"file": str(path), "scene_id": scene, "view_id": view,
                                  "size": 8, "layer_count": len(masks)})
    return template, output, build_mask_cache(template, labels, output)


class ArchitectureSearchTests(unittest.TestCase):
    def test_capacity_uses_all_development_labels_and_never_test(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, index = source_cache(Path(t), overflow_test=True)
            report = capacity_audit(index, [1, 2, 6])
            self.assertEqual([r["eligible"] for r in report["candidates"]], [False, True, True])
            self.assertEqual(report["candidates"][1]["coverage"]["validation"], 1)
            self.assertNotIn("test", report["splits"])
            self.assertEqual(len(index["records"]["test"][0]["intensities"]), 3)

    def test_proportional_cohort_is_reproducible_and_has_no_duplicates(self):
        rows = [{"size": 16 if i < 80 else 32, "intensities": [.3], "id": i} for i in range(100)]
        result = stratified_subset(rows, 10, 42)
        self.assertEqual(result, stratified_subset(rows, 10, 42))
        self.assertEqual(len({r["id"] for r in result}), 10)
        self.assertEqual(sum(r["size"] == 32 for r in result), 2)
        rare = [{"size": 16 if i < 99 else 512, "intensities": [.3], "id": i} for i in range(100)]
        self.assertTrue(any(r["size"] == 512 for r in stratified_subset(rare, 5, 0)))
        with self.assertRaisesRegex(ValueError, "every layer-count"):
            stratified_subset(rare, 1, 0)

    def test_shared_pilot_copies_original_bytes_without_reading_test(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _, source, index = source_cache(root)
            for path in (source / "mask_cache").glob("test_*"):
                path.unlink()
            pilot_root = root / "pilot"
            pilot = make_pilot(source, index, pilot_root, {"train": 1, "validation": 1}, 4)
            self.assertEqual(pilot["counts"], {"train": 1, "validation": 1, "test": 0})
            validate_mask_cache(pilot, pilot_root / "mask_cache", 8, splits=("train", "validation"))
            view = bind_cache(root / "trial", pilot_root, pilot, 2)
            validate_mask_cache(view, root / "trial/mask_cache", 2, splits=("train", "validation"))
            a = MaskRecords(view["records"]["train"], root / "trial/mask_cache", "train", 2)[0]
            original = MaskRecords(index["records"]["train"], source / "mask_cache", "train", 8)[0]
            np.testing.assert_array_equal(a["response"], original["response"])
            np.testing.assert_array_equal(a["teacher_mask"], original["teacher_mask"][:2])
            self.assertTrue((root / "trial/mask_cache").is_symlink())
            self.assertEqual(make_pilot(source, index, pilot_root, {"train": 1, "validation": 1}, 4), pilot)

    def test_grid_includes_controlled_slot_comparison(self):
        cfg = search.load_search(search.PHASE2 / "configs/slot_search.json")
        cfg["max_trials"] = 6
        grid = search.trial_grid(cfg, [6, 8, 32])
        self.assertEqual([r["slots"] for r in grid[:3]], [6, 8, 32])
        self.assertEqual(len({(r["hidden"], r["depth"], r["learning_rate"]) for r in grid[:3]}), 1)
        self.assertEqual(grid, search.trial_grid(cfg, [6, 8, 32]))
        cfg["max_trials"] = 2
        with self.assertRaisesRegex(ValueError, "controlled sweep"):
            search.trial_grid(cfg, [6, 8, 32])

    def test_gpu_inventory_skips_busy_devices(self):
        args = type("Args", (), dict(device="cuda", gpu_ids=None, gpus="2", max_used_memory_mib=2048, max_utilization=10))()
        with patch("subprocess.check_output", side_effect=[
                "0, GPU-A, 0, 0\n1, GPU-B, 63000, 90\n2, GPU-C, 500, 0\n3, GPU-D, 4, 0\n",
                "GPU-C, 12345\n"]):
            self.assertEqual(search.select_gpus(args), ["0", "3"])

    def test_budget_resume_keeps_optimizer_and_never_requires_test(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            config, source, index = source_cache(root)
            config["output_root"] = str(source)
            config["training"].update(evaluate_test_after_training=False, batch_size=1)
            for path in (source / "mask_cache").glob("test_*"):
                path.unlink()
            train(config, source, index, epoch_limit=1)
            before = json.loads((source / "models/seed_0/training.json").read_text())["history"]
            self.assertEqual(len(before), 1)
            train(config, source, index, epoch_limit=2)
            after = json.loads((source / "models/seed_0/training.json").read_text())["history"]
            self.assertEqual(after[0], before[0])
            self.assertEqual(len(after), 2)
            summary = json.loads((source / "training_summary.json").read_text())["0"]
            self.assertIsNone(summary["test"])
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(summary["trained_through_epoch"], 2)

    def test_overflow_padding_preserves_reconstruction_and_counts_missing_layer(self):
        target = torch.tensor([[[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]]])
        strengths = torch.tensor([[.3, .5, .2]])
        presence = torch.ones(1, 2) * .99999
        output = {"mask_probability": target[:, :2],
                  "mask_logits": torch.logit(target[:, :2].clamp(1e-5, 1 - 1e-5)),
                  "presence_probability": presence, "presence_logits": torch.logit(presence),
                  "intensity": strengths[:, :2], "count_logits": poisson_binomial_logits(presence),
                  "reconstruction": (target[:, :2] * strengths[:, :2, None, None]).sum(1)}
        batch = {"teacher_mask": target, "teacher_intensity": strengths, "layer_mask": torch.ones(1, 3, dtype=torch.bool),
                 "response": (target * strengths[..., None, None]).sum(1), "obstacle": torch.zeros(1, 1, 3, dtype=torch.bool),
                 "scene_id": torch.tensor([0]), "view_id": torch.tensor([0]), "window": torch.tensor([[0, 0, 3]])}
        padded = pad_missing_predictions(output, 3)
        torch.testing.assert_close(padded["reconstruction"], output["reconstruction"])
        assignment = decomposition_training_loss(padded, batch)["assignment"]
        metrics = metric_rows(padded, batch, assignment)[0]["metrics"]
        self.assertAlmostEqual(metrics["mask_iou"], 2 / 3)
        self.assertEqual(metrics["count_accuracy"], 0)
        self.assertEqual(metrics["count_mae"], 1)
        self.assertEqual(metrics["strict_decomposition_success"], 0)

    def test_cli_search_then_automatic_fit_evaluates_complete_locked_test(self):
        with tempfile.TemporaryDirectory(prefix="rind search ") as t:
            root = Path(t)
            template, source, index = source_cache(root, overflow_test=True)
            template_file = root / "template.json"
            template_file.write_text(json.dumps(template))
            cfg = search.load_search(search.PHASE2 / "configs/slot_search.json")
            cfg.update(template_config=str(template_file), source_candidates=[str(source)],
                       output_root=str(root / "study"), slot_counts=[2, 3], hidden_widths=[32],
                       decoder_depths=[1], learning_rates=[.0001], max_trials=2,
                       pilot_records={"train": 2, "validation": 1}, epoch_budgets=[1, 2],
                       score_tolerance=1, final_training={"epochs": 1, "patience": 2, "seeds": [0]},
                       bootstrap_repetitions=2)
            cfg["autobatch"]["enabled"] = False
            path = root / "search.json"
            path.write_text(json.dumps(cfg))
            # Search must work even when locked-test bytes are unavailable.
            hidden = []
            for test in (source / "mask_cache").glob("test_*"):
                original = test.with_suffix(test.suffix + ".heldout")
                test.rename(original)
                hidden.append((test, original))
            command = [sys.executable, str(search.HERE), "search", "--config", str(path), "--device", "cpu"]
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            selected = json.loads((root / "study/selected.json").read_text())
            self.assertFalse(selected["test_used_for_selection"])
            # All survivors share full cohorts; no test inference was executed.
            self.assertFalse(list((root / "study/trials").glob("*/evaluation/decomposition_metrics.json")))
            for test, original in hidden:
                original.rename(test)
            command[2] = "all"
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            study = root / "study"
            self.assertTrue((study / "best_full/fit_complete.json").is_file())
            report = json.loads((study / "best_full/evaluation/decomposition_metrics.json").read_text())
            self.assertEqual(report["windows"], 1)
            self.assertEqual(report["split"], "test")
            # Verify completed search/fit can be invoked again without changing contracts.
            again = subprocess.run(command, text=True, capture_output=True, timeout=120)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertIn("already completed", again.stdout)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_fit_matches_serial_accumulation_and_evaluates_overflow(self):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except PermissionError:
            self.skipTest("sandbox disallows local rendezvous sockets")
        with tempfile.TemporaryDirectory(prefix="rind ddp search ") as t:
            root = Path(t)
            template, source, index = source_cache(root, overflow_test=True, train_views=4)
            template["model"]["max_slots"] = 2
            template["training"].update(batch_size=1, target_global_batch_size=4,
                                         epochs=1, evaluate_test_after_training=False)
            template["evaluation"] = {"capacity_overflow": "missing_predictions"}
            study = root / "study"
            output = study / "best_full"
            view = bind_cache(output, source, index, 2)
            config = copy.deepcopy(template)
            config["output_root"] = str(output)
            file = output / "config.json"
            file.write_text(json.dumps(config))
            (study / "selected.json").write_text(json.dumps({"trial": {"id": "controlled"}}))
            cfg = search.load_search(search.PHASE2 / "configs/slot_search.json")
            cfg["bootstrap_repetitions"] = 2
            path = root / "search.json"
            path.write_text(json.dumps(cfg))
            command = search.worker_command(file, path, final=True, gpus=["0", "1"])
            result = subprocess.run(command, env=search.child_env([], "cpu"), text=True,
                                    capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "evaluation/decomposition_metrics.json").read_text())
            self.assertEqual(report["windows"], 1)
            self.assertEqual(report["capacity_overflow_windows"], 1)
            self.assertEqual(report["window_weighted"]["count_accuracy"]["mean"], 0)
            serial = root / "serial"
            bind_cache(serial, source, index, 2)
            config["output_root"] = str(serial)
            train(config, serial, json.loads((serial / "mask_records.json").read_text()))
            distributed = torch.load(output / "models/seed_0/best.pt", map_location="cpu", weights_only=True)["model"]
            single = torch.load(serial / "models/seed_0/best.pt", map_location="cpu", weights_only=True)["model"]
            for key in single:
                torch.testing.assert_close(distributed[key], single[key], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
