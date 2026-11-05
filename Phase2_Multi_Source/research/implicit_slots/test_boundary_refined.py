import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from decomposition_model import decomposition_training_loss
from train_decomposition import make_model, load_config, train
from reuse_mask_cache import prepare_cached_labels
from evaluate_masks import evaluate, pad_missing_predictions, metric_rows
from test_architecture_search import source_cache
from mask_only_data import build_mask_cache


PACKAGE = Path(__file__).resolve().parents[3]


def config_for_test():
    config = load_config(PACKAGE / "Phase2_Multi_Source/configs/masks_boundary_refined.json")
    config["model"].update(max_slots=4, decomposition_hidden=32, slot_decoder_layers=1, mask_feature_dim=16)
    config["training"].update(device="cpu", amp_dtype="none", num_threads=1, epochs=2,
                              patience=5, batch_size=1, target_global_batch_size=2, prefetch_batches=0,
                              evaluation_bootstrap_repetitions=2)
    return config


def example(side=16, slots=4):
    masks = torch.zeros(1, slots, side, side)
    masks[0, 0, :side-1, :side//2+2] = 1
    masks[0, 1, :side-1, side//3:] = 1
    strengths = torch.zeros(1, slots)
    strengths[0, :2] = torch.tensor([.3, .5])
    response = (masks * strengths[..., None, None]).sum(1)
    return {"response": response, "obstacle": torch.zeros_like(response, dtype=torch.bool),
            "teacher_mask": masks, "teacher_intensity": strengths, "layer_mask": strengths > 0,
            "window": torch.tensor([[12., 48., side]])}


class BoundaryRefinedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_native_sizes_and_observation_scale_equivariance_without_geometry(self):
        config = config_for_test()
        model = make_model(config, 1024, torch.device("cpu"), False).eval()
        self.assertIsNone(model.locator)
        self.assertEqual(len(model.decoder), 1)
        self.assertEqual([role.heads for role in model.decoder[0].roles], [4, 2, 2])
        for side in (8, 16, 32, 64, 128):
            batch = example(side)
            with torch.inference_mode():
                output = model.decompose(batch["response"], batch["obstacle"], batch["window"])
                dim = model.decompose(batch["response"] * .01, ~batch["obstacle"], batch["window"])
            self.assertEqual(output["mask_probability"].shape, (1, 4, side, side))
            torch.testing.assert_close(output["mask_probability"], dim["mask_probability"], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(output["intensity"] * .01, dim["intensity"], atol=2e-6, rtol=2e-5)
            self.assertTrue(torch.all(output["intensity"] <= batch["response"].max()))
            self.assertTrue(torch.all(output["mask_probability"][:, :, -1] == 0))
            # Additive output masks need not sum to 1; overlapping layers are allowed.
            self.assertGreater(float(output["mask_probability"].sum(1).max()), 1)

    def test_public_forward_needs_no_candidates_or_teacher_layers(self):
        config, batch = config_for_test(), example()
        model = make_model(config, 1024, torch.device("cpu"), False).eval()
        with torch.inference_mode():
            output = model(batch["response"], batch["obstacle"], batch["window"])
        self.assertEqual(output["mask_probability"].shape, batch["teacher_mask"].shape)
        self.assertNotIn("slot_probability", output)
        self.assertNotIn("teacher_mask", output)
        with self.assertRaisesRegex(ValueError, "localization requires"):
            model(batch["response"], batch["obstacle"], batch["window"], compute_location=True)

    def test_all_branches_finite_gradients_and_only_one_assignment_for_all_rounds(self):
        config, batch = config_for_test(), example()
        for amp, checkpointed in ((False, False), (True, False), (False, True)):
            config["training"]["activation_checkpoint_decomposer"] = checkpointed
            model = make_model(config, 1024, torch.device("cpu"), False).train()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                output = model(batch["response"], batch["obstacle"], batch["window"], None, None, compute_location=False)
            from decomposition_model import exact_batched_assignment
            with patch("decomposition_model.exact_batched_assignment", wraps=exact_batched_assignment) as matching:
                losses = decomposition_training_loss(output, batch, **config["loss"])
            self.assertEqual(matching.call_count, 1)
            losses["loss"].backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for round_output in [*output["refinement_outputs"], output]:
                self.assertTrue(torch.isfinite(round_output["reconstruction"]).all())

    def test_competition_scores_use_fp32_under_bfloat16_autocast(self):
        from boundary_refined_model import RoleAttention
        attention = RoleAttention(32, 8, 4, True)
        original, dtypes = torch.matmul, []
        def capture(left, right):
            result = original(left, right)
            dtypes.append(result.dtype)
            return result
        with torch.autocast("cpu", dtype=torch.bfloat16), patch("torch.matmul", side_effect=capture):
            output = attention(torch.randn(1, 4, 32), torch.randn(1, 12, 32), torch.ones(1, 4, 12))
        self.assertEqual(dtypes, [torch.float32, torch.bfloat16])
        self.assertTrue(torch.isfinite(output).all())

    def test_overflow_pads_all_refinement_rounds_and_counts_missing_layer(self):
        config, batch = config_for_test(), example(slots=2)
        batch.update(scene_id=torch.tensor([0]), view_id=torch.tensor([0]))
        config["model"]["max_slots"] = 1
        model = make_model(config, 1024, torch.device("cpu"), False).eval()
        with torch.inference_mode():
            output = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            padded = pad_missing_predictions(output, 2)
            losses = decomposition_training_loss(padded, batch, **config["loss"])
            row = metric_rows(padded, batch, losses["assignment"])[0]
        self.assertEqual(output["mask_probability"].shape[1], 1)
        for current in [padded, *padded["refinement_outputs"]]:
            self.assertEqual(current["mask_probability"].shape[1], 2)
            self.assertTrue(torch.all(current["presence_probability"][:, 1] == 0))
        torch.testing.assert_close(padded["reconstruction"], output["reconstruction"])
        self.assertEqual(row["layer_count"], 2)
        self.assertEqual(row["metrics"]["count_accuracy"], 0)
        self.assertEqual(row["metrics"]["strict_decomposition_success"], 0)

    def test_slot_permutation_changes_only_slot_order(self):
        config, batch = config_for_test(), example()
        model = make_model(config, 1024, torch.device("cpu"), False).eval()
        order = torch.tensor([2, 0, 3, 1])
        with torch.inference_mode():
            before = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            model.slot_queries.copy_(model.slot_queries[order].clone())
            after = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            for key in ("mask_probability", "intensity", "presence_probability"):
                torch.testing.assert_close(after[key], before[key][:, order], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(after["reconstruction"], before["reconstruction"], atol=1e-5, rtol=1e-5)
            first = decomposition_training_loss(before, batch, **config["loss"])
            second = decomposition_training_loss(after, batch, **config["loss"])
            torch.testing.assert_close(first["loss"], second["loss"], atol=1e-5, rtol=1e-5)

    def test_final_loss_backpropagates_through_earlier_predicted_residuals(self):
        config, batch = config_for_test(), example()
        model = make_model(config, 1024, torch.device("cpu"), False).train()
        output = model.decompose(batch["response"], batch["obstacle"], batch["window"])
        first_mask = output["refinement_outputs"][0]["mask_logits"]
        first_mask.retain_grad()
        loss_config = {**config["loss"], "auxiliary_weight": 0.0}
        decomposition_training_loss(output, batch, **loss_config)["loss"].backward()
        self.assertIsNotNone(first_mask.grad)
        self.assertTrue(torch.isfinite(first_mask.grad).all())
        self.assertGreater(float(first_mask.grad.abs().sum()), 0)

    def test_scale_normalized_objective_is_invariant_to_intensity_scale(self):
        config, batch = config_for_test(), example()
        model = make_model(config, 1024, torch.device("cpu"), False).eval()
        losses = []
        with torch.inference_mode():
            for scale in (1., .01, 1e-7):
                current = {**batch, "response": batch["response"] * scale,
                           "teacher_intensity": batch["teacher_intensity"] * scale}
                output = model.decompose(current["response"], current["obstacle"], current["window"])
                losses.append(decomposition_training_loss(output, current, **config["loss"])["loss"])
        torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(losses[0], losses[2], atol=1e-5, rtol=1e-5)

    def test_supervised_optimization_reduces_loss_on_overlapping_layers(self):
        config, batch = config_for_test(), example()
        model = make_model(config, 1024, torch.device("cpu"), False).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        losses = []
        for _ in range(24):
            optimizer.zero_grad(set_to_none=True)
            output = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            loss = decomposition_training_loss(output, batch, **config["loss"])["loss"]
            losses.append(float(loss.detach()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        self.assertLess(sum(losses[-3:]) / 3, .8 * sum(losses[:3]) / 3)

    def test_cached_full_pipeline_resume_evaluate_without_raw_data_or_phase1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source, source_index = source_cache(root)
            config = config_for_test()
            config.update(cache_source_root=str(source), output_root=str(root / "new_model"))
            with patch("rind_phase1.model.SourceEnergyField.__init__", side_effect=AssertionError("Phase1 forbidden")):
                index = prepare_cached_labels(config)
                self.assertEqual(index["counts"], source_index["counts"])
                self.assertEqual(index["records"], source_index["records"])
                self.assertTrue((root / "new_model/mask_cache").is_symlink())
                output = root / "new_model"
                train(config, output, index, epoch_limit=1)
                before = torch.load(output / "models/seed_0/last.pt", weights_only=True)
                index = prepare_cached_labels(config)
                train(config, output, index, epoch_limit=2)
                after = torch.load(output / "models/seed_0/last.pt", weights_only=True)
                self.assertEqual(before["history"], after["history"][:1])
                uninterrupted = copy.deepcopy(config)
                uninterrupted["output_root"] = str(root / "uninterrupted")
                full_index = prepare_cached_labels(uninterrupted)
                train(uninterrupted, root / "uninterrupted", full_index)
                full = torch.load(root / "uninterrupted/models/seed_0/last.pt", weights_only=True)
                for name in full["model"]:
                    torch.testing.assert_close(full["model"][name], after["model"][name], atol=1e-7, rtol=1e-6)
                report = evaluate(config, output, index, repetitions=2)
                self.assertEqual(report["windows"], source_index["counts"]["test"])
                self.assertIn("strict_decomposition_success", report["scene_balanced"])
                self.assertFalse(report["location_evaluated"])
                self.assertEqual(report["candidate_renders"], 0)
                # Runtime evaluation edits cannot change the trained matching.
                edited = copy.deepcopy(config)
                edited["loss"]["intensity_weight"] = 100.0
                with patch("evaluate_masks.decomposition_training_loss", wraps=decomposition_training_loss) as loss_call:
                    reevaluated = evaluate(edited, output, index, repetitions=2)
                self.assertEqual(loss_call.call_args.kwargs["intensity_weight"], config["loss"]["intensity_weight"])
                self.assertEqual(report["window_weighted"], reevaluated["window_weighted"])
            # Incompatible legacy checkpoints must fail before partial loading.
            legacy = copy.deepcopy(config)
            legacy["model"].pop("decomposition_architecture")
            legacy["training"]["initial_decomposition_checkpoint"] = str(output / "models/seed_0/best.pt")
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                make_model(legacy, 1024, torch.device("cpu"), True)

    def test_cache_protocol_locks_source_and_rejects_capacity_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source, _ = source_cache(root)
            config = config_for_test()
            config.update(cache_source_root=str(source), output_root=str(root / "new_model"))
            prepare_cached_labels(config)
            config["model"]["max_slots"] = 1
            with self.assertRaisesRegex(ValueError, "no truncation"):
                prepare_cached_labels(config)

    def test_refined_launcher_prepares_from_cache_without_geometry_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shim = root / "python-shim"
            shim.write_text('#!/usr/bin/env bash\nif [[ "$1" == -c ]]; then exec "$REAL_PYTHON" "$@"; fi\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            shim.chmod(0o755)
            env = {**os.environ, "RIND_PYTHON": str(shim), "REAL_PYTHON": sys.executable, "CAPTURE": str(root / "capture")}
            for key in ("RIND_MASKS_CONFIG", "RIND_DECOMPOSITION_CONFIG"):
                env.pop(key, None)
            result = subprocess.run(["bash", str(PACKAGE / "run_refined_masks.sh"), "prepare", "--gpus", "auto"],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertNotIn("Teacher geometry preflight", result.stdout)
            capture = (root / "capture").read_text()
            self.assertIn("index", capture)
            self.assertIn("masks_boundary_refined.json", capture)

    def test_refined_tmux_selects_new_session_and_forwards_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, output = root / "config.json", root / "output"
            config.write_text(json.dumps({"output_root": str(output)}))
            tmux = root / "tmux"
            tmux.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n')
            tmux.chmod(0o755)
            environment = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                           "RIND_PYTHON": sys.executable, "RIND_MASKS_CONFIG": str(config)}
            environment.pop("RIND_TMUX_SESSION", None)
            result = subprocess.run(["bash", str(PACKAGE / "run_refined_masks_tmux.sh"), "start", "all", "--gpus", "8"],
                                    env=environment, text=True, capture_output=True, check=True)
            self.assertIn("rind_masks_refined", result.stdout)
            run = Path((output / "tmux_runs/latest_run.txt").read_text().strip())
            command = (run / "command.txt").read_text()
            self.assertIn("RIND_TMUX_TASK=all", command)
            self.assertIn("RIND_MASKS_CONFIG=" + str(config.resolve()), command)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_refined_ddp_accumulation_matches_serial_with_shared_round_weights(self):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except PermissionError:
            self.skipTest("sandbox disallows local rendezvous sockets")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source, source_index = source_cache(root, train_views=4)
            config = config_for_test()
            # Distinct training responses exercise real gradient averaging;
            # identical repeated fixtures could conceal sampler/reduction bugs.
            for view in range(4):
                path = root / f"train_{view}.npz"
                with np.load(path) as stored:
                    arrays = {key: stored[key] for key in stored.files}
                arrays["intensities"] = np.array([.2 + .04 * view, .7 - .03 * view])
                arrays["response"] = (arrays["masks"] * arrays["intensities"][:, None, None]).sum(0)
                np.savez(path, **arrays)
            labels = {split: [{**row, "file": str(root / f"{split}_{row['view_id']}.npz")} for row in rows]
                      for split, rows in source_index["records"].items()}
            build_mask_cache(config, labels, source)
            config.update(cache_source_root=str(source), output_root=str(root / "ddp"))
            config["training"].update(epochs=1, target_global_batch_size=4, evaluate_test_after_training=False)
            index = prepare_cached_labels(config)
            path = root / "config.json"
            path.write_text(json.dumps(config))
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                       str(PACKAGE / "Phase2_Multi_Source/research/implicit_slots/train_decomposition.py"),
                       "train", "--config", str(path)]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"):
                environment.pop(key, None)
            result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            distributed = torch.load(root / "ddp/models/seed_0/last.pt", weights_only=True)["model"]
            config["output_root"] = str(root / "serial")
            index = prepare_cached_labels(config)
            train(config, root / "serial", index)
            serial = torch.load(root / "serial/models/seed_0/last.pt", weights_only=True)["model"]
            for name in serial:
                torch.testing.assert_close(distributed[name], serial[name], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
