#!/usr/bin/env python3
"""Train a mask-supervised decomposition student followed by shared Phase1 localization."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
PHASE2_ROOT = HERE.parents[2]
for item in (HERE.parent, PHASE2_ROOT, PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset",
             PHASE2_ROOT.parent / "Phase1_Single_Source_Single_Intensity/src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from decomposition_data import (DecompositionRecords, build_decomposition_index,
                                decomposed_batches, batch_count)
from decomposition_model import ResponseDecompositionField, decomposition_training_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PHASE2_ROOT / path


def load_config(path):
    config = json.loads(Path(path).read_text())
    if "base_config" in config:
        base = load_config(resolve(config.pop("base_config")))
        base.update(config)
        config = base
    if config.get("task") not in (None, "full", "decomposition_only"):
        raise ValueError("task must be decomposition_only or full (or omitted for full training)")
    required = ["data_root", "output_root", "model", "training", "loss"]
    if config.get("task") != "decomposition_only":
        required.append("teacher_experiment_root")
    for key in required:
        if key not in config:
            raise ValueError(f"configuration is missing {key}")
    for key in ("epochs", "patience", "batch_size", "target_global_batch_size", "candidate_chunk"):
        if type(config["training"].get(key)) is not int or config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive integer")
    if config["training"].get("amp_dtype") not in ("none", "bfloat16"):
        raise ValueError("AMP supports none or bfloat16; float16 without a scaler is disallowed")
    if not 1 <= config["model"]["max_slots"] <= 32:
        raise ValueError("max_slots must lie in [1,32]")
    for key in ("learning_rate", "grad_clip"):
        value = config["training"].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if config.get("task") == "decomposition_only":
        for key in ("location_weight", "match_location_weight"):
            if config["loss"].get(key, 0) != 0:
                raise ValueError(f"loss.{key} must be zero for decomposition_only")
        if "label_preparation" not in config:
            raise ValueError("mask-only configuration requires label_preparation")
    return config


def semantic_config(config):
    value = copy.deepcopy(config)
    for key in ("batch_size", "candidate_chunk", "prefetch_batches", "num_threads",
                "progress_every_batches", "gpu_devices", "activation_checkpoint_candidates",
                "activation_checkpoint_locator_encoder", "channels_last", "device"):
        value["training"].pop(key, None)
    return value


def log(root, event, message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **fields}
    print(f"[{row['time']}] {message}", flush=True)
    with (Path(root) / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def make_model(config, world_size, device, initialize):
    model_config = config["model"]
    training = config["training"]
    masks_only = config.get("task") == "decomposition_only"
    backbone = {
        "hidden": model_config["hidden"], "fourier_frequencies": model_config["fourier_frequencies"],
        "coordinate_scale": float(world_size), "use_obstacle": model_config["use_obstacle"],
        "pixel_coordinates": True, "normalization_support": "world",
        "architecture": "multiscale_boundary", "attention_heads": 8,
    }
    model = ResponseDecompositionField(
        max_slots=model_config["max_slots"], backbone_config=backbone,
        decomposition_hidden=model_config["decomposition_hidden"],
        slot_decoder_layers=model_config.get("slot_decoder_layers", 2),
        slot_attention_heads=model_config.get("slot_attention_heads", 8),
        use_local_obstacle=model_config.get("use_local_obstacle", False),
        activation_checkpoint_candidates=training.get("activation_checkpoint_candidates", True),
        activation_checkpoint_locator_encoder=training.get("activation_checkpoint_locator_encoder", True),
        enable_localization=not masks_only,
    ).to(device)
    if initialize and not masks_only:
        import torch
        checkpoint_path = resolve(training["initial_phase1_checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Phase1 checkpoint missing: {checkpoint_path}")
        model.load_phase1_backbone(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    warm_start = training.get("initial_decomposition_checkpoint")
    if initialize and warm_start:
        import torch
        saved = torch.load(resolve(warm_start), map_location="cpu", weights_only=True)
        if saved.get("schema") != "phase2-local-mask-decomposition-v1":
            raise ValueError("initial_decomposition_checkpoint must be a mask-only checkpoint")
        for key in ("max_slots", "decomposition_hidden", "slot_decoder_layers", "slot_attention_heads", "use_local_obstacle"):
            if saved["config"]["model"].get(key) != model_config.get(key):
                raise ValueError(f"decomposition warm-start model.{key} differs")
        state = {key: value for key, value in saved["model"].items() if not key.startswith("locator.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("locator.") for key in missing):
            raise ValueError("decomposition warm-start weights do not match")
    return model


def run_epoch(model, records, config, device, epoch, rank, world_size, optimizer=None,
              rng=None, progress=None):
    import torch
    import torch.distributed as dist
    training = config["training"]
    learning = optimizer is not None
    model.train(learning)
    totals = {}
    records_seen = 0
    micro = training["batch_size"]
    batches = batch_count(records, micro, world_size=world_size, rank=rank, training=learning)
    accumulation = max(1, math.ceil(training["target_global_batch_size"] / (micro * world_size))) if learning else 1
    compute_location = (config.get("task") != "decomposition_only" and
                        epoch > int(training.get("mask_warmup_epochs", 5)))
    loss_config = dict(config["loss"])
    if not compute_location:
        loss_config["location_weight"] = 0.0
    if learning:
        optimizer.zero_grad(set_to_none=True)
    for number, batch in enumerate(decomposed_batches(
            records, micro, rng=rng, prefetch=training.get("prefetch_batches", 2),
            rank=rank, world_size=world_size, training=learning), 1):
        batch = {key: value.pin_memory().to(device, non_blocking=True)
                 if device.type == "cuda" else value.to(device) for key, value in batch.items()}
        start = ((number - 1) // accumulation) * accumulation + 1
        end = min(start + accumulation - 1, batches)
        divisor = end - start + 1
        with torch.set_grad_enabled(learning):
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and training["amp_dtype"] == "bfloat16"):
                result = model(batch["response"], batch["obstacle"], batch["window"],
                               batch.get("candidate_xy"), batch.get("support"),
                               candidate_chunk=training["candidate_chunk"],
                               compute_location=compute_location)
            losses = decomposition_training_loss(result, batch, **loss_config)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(f"nonfinite loss in batch {number}; scene IDs={batch['scene_id'].tolist()}")
            if learning:
                (losses["loss"] / divisor).backward()
                if number == end:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("nonfinite gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        size = len(batch["response"])
        records_seen += size
        for key, value in losses.items():
            if key == "assignment" or value.numel() != 1:
                continue
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
        if progress and (number == 1 or number % training.get("progress_every_batches", 20) == 0):
            progress(number, batches, {key: value / records_seen for key, value in totals.items()})
    # Empty evaluation ranks must still take part in the collective. A fixed
    # metric key contract is exchanged once; counts reflect real, unpadded eval rows.
    if world_size > 1:
        key_sets = [None] * world_size
        dist.all_gather_object(key_sets, sorted(totals))
        keys = sorted(set().union(*(set(value) for value in key_sets)))
        packed = torch.tensor([totals.get(key, 0.0) for key in keys] + [float(records_seen)],
                              dtype=torch.float64, device=device)
        dist.all_reduce(packed)
        records_seen = int(packed[-1].item())
        totals = {key: float(packed[index]) for index, key in enumerate(keys)}
    if not records_seen:
        raise ValueError("empty split")
    return {key: value / records_seen for key, value in totals.items()}


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp.pt")
    import torch
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, output, index):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    setting = config["training"]
    masks_only = config.get("task") == "decomposition_only"
    if masks_only:
        from mask_validation import validate_mask_cache
        validate_mask_cache(index, output / "mask_cache", config["model"]["max_slots"])
    cuda = setting.get("device", "cuda") == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set device=cpu for a smoke run")
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(max(1, setting.get("num_threads", 8) // world))
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if masks_only:
        from mask_only_data import MaskRecords
        record_type, cache = MaskRecords, output / "mask_cache"
    else:
        record_type, cache = DecompositionRecords, output / "decomposition_cache"
    splits = {name: record_type(index["records"][name], cache, name,
                                        config["model"]["max_slots"])
              for name in ("train", "validation", "test")}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if splits[first].scene_ids & splits[second].scene_ids:
            raise ValueError("scene leakage across splits")
    reports = {}
    for seed in setting["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        run = output / "models" / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        last = run / "last.pt"
        model = make_model(config, index.get("global_size", 1024), device, initialize=not last.is_file())
        # Keep parameters registered throughout the schedule. Locator weights
        # are frozen with its explicit method before wrapping below.
        model.set_locator_trainable(False)
        parameter_groups = [
            {"params": [p for name, p in model.named_parameters() if not name.startswith("locator.")],
             "lr": setting["learning_rate"]},
        ]
        if model.locator is not None:
            parameter_groups.append({"params": list(model.locator.parameters()),
                                     "lr": setting.get("locator_learning_rate", 1e-5)})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=setting["weight_decay"],
                                      fused=cuda and setting.get("fused_optimizer", True))
        signed_index = {key: value for key, value in index.items() if key != "statistics"} if masks_only else index
        signature = digest({"config": semantic_config(config), "index": signed_index})
        history, start, best, bad = [], 0, float("inf"), 0
        initial_validation = None
        if last.is_file():
            saved = torch.load(last, map_location="cpu", weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("checkpoint/config/cache mismatch; choose a new output_root")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history, start, best, bad = saved["history"], saved["epoch"], saved["best"], saved["bad"]
            best = float("inf") if best is None else best
            rng.bit_generator.state = saved["numpy_rng"]
            initial_validation = saved.get("initial_validation")
        elif masks_only and setting.get("initial_decomposition_checkpoint") and setting.get("evaluate_initial_checkpoint", False):
            # Keep the old best model as epoch 0 on the exact same validation
            # set. Expanded fine-tuning must beat this score to replace it.
            initial_validation = run_epoch(model, splits["validation"], config, device, 0, rank, world)
            best = initial_validation["loss"]
            if rank == 0:
                state = {"schema": "phase2-local-mask-decomposition-v1", "config": config,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": 0, "best": best, "bad": 0, "history": [], "signature": signature,
                         "numpy_rng": rng.bit_generator.state, "initial_validation": initial_validation}
                save_checkpoint(run / "best.pt", state)
                log(output, "initial_validation", f"expanded fine-tuning baseline | validation loss={best:.6f}; old best retained as epoch 0", metrics=initial_validation)
            if world > 1:
                dist.barrier()
        wrapped = None
        last_trainable = None
        started = time.perf_counter()
        for epoch in range(start + 1, setting["epochs"] + 1):
            if bad >= setting["patience"]:
                break
            trainable = not masks_only and epoch > setting.get("locator_frozen_epochs", 15)
            if trainable != last_trainable or wrapped is None:
                if world > 1:
                    dist.barrier()
                # Recreate DDP when the trainable parameter set changes; its
                # reducer cannot gain newly unfrozen parameters in-place.
                wrapped = None
                model.set_locator_trainable(trainable)
                wrapped = (DistributedDataParallel(
                    model, device_ids=[local] if cuda else None, broadcast_buffers=False,
                    gradient_as_bucket_view=True, find_unused_parameters=True,
                ) if world > 1 else model)
                last_trainable = trainable
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            if rank == 0:
                accumulation = math.ceil(setting["target_global_batch_size"] / (setting["batch_size"] * world))
                log(output, "epoch_start", f"decomposition epoch={epoch}/{setting['epochs']} | task={config.get('task', 'full')} | train/val/test={len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} | slots={config['model']['max_slots']} | GPU ranks={world} | micro={setting['batch_size']} accumulate={accumulation} effective_global_batch={setting['batch_size'] * world * accumulation} | locator_trainable={trainable} | location_active={not masks_only and epoch > setting['mask_warmup_epochs']}")
            def callback(phase):
                def emit(number, total, metrics):
                    log(output, "batch", f"{phase} epoch={epoch} batch={number}/{total} loss={metrics['loss']:.5f} mask_dice={metrics.get('mask_dice', 0):.4f} count_acc={metrics.get('count_accuracy', 0):.4f}",
                        epoch=epoch, phase=phase, batch=number, metrics=metrics)
                return emit if rank == 0 else None
            training_metrics = run_epoch(wrapped, splits["train"], config, device, epoch, rank, world,
                                         optimizer, rng, callback("train"))
            # Evaluate through the unwrapped module: unequal rank partitions
            # must not trigger DDP forward collectives during evaluation.
            validation = run_epoch(model, splits["validation"], config, device, epoch, rank, world,
                                   progress=callback("validation"))
            # All post-warmup epochs share the same loss. Warmup checkpoints
            # cannot become the final best just because they omit localization.
            eligible = masks_only or epoch > setting["mask_warmup_epochs"]
            improved = eligible and validation["loss"] < best
            if improved:
                best, bad = validation["loss"], 0
            elif eligible:
                bad += 1
            memory = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0,
                                  device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            record = {"epoch": epoch, "train": training_metrics, "validation": validation,
                      "seconds": time.perf_counter() - tick, "peak_allocated_gib": float(memory)}
            history.append(record)
            if rank == 0:
                state = {"schema": "phase2-local-mask-decomposition-v1" if masks_only else "phase2-response-decomposition-v2", "config": config,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best": best if math.isfinite(best) else None,
                         "bad": bad, "history": history, "signature": signature,
                         "numpy_rng": rng.bit_generator.state, "initial_validation": initial_validation}
                save_checkpoint(last, state)
                if improved:
                    save_checkpoint(run / "best.pt", state)
                write_json(run / "training.json", {"history": history, "best_validation_loss": best if math.isfinite(best) else None})
                log(output, "epoch_done", f"epoch={epoch} train={training_metrics['loss']:.5f} val={validation['loss']:.5f} mask_dice={validation.get('mask_dice',0):.4f} best={best:.5f} bad={bad}/{setting['patience']} peak={float(memory):.2f} GiB", **record)
            if world > 1:
                dist.barrier()
        if not (run / "best.pt").is_file():
            raise RuntimeError("no post-warmup checkpoint; epochs must exceed mask_warmup_epochs")
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        test = run_epoch(model, splits["test"], config, device, saved["epoch"], rank, world)
        report = {"seed": seed, "best_epoch": saved["epoch"], "test": test,
                  "best_validation_loss": saved["best"], "initial_validation": initial_validation,
                  "records": {name: len(value) for name, value in splits.items()},
                  "elapsed_seconds": time.perf_counter() - started}
        if rank == 0:
            write_json(run / "training.json", {**report, "history": history})
            reports[str(seed)] = report
            log(output, "done", f"decomposition done best_epoch={saved['epoch']} test={test}")
    if rank == 0:
        write_json(output / "training_summary.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "train", "all"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    masks_only = config.get("task") == "decomposition_only"
    if args.command in ("index", "all"):
        if masks_only:
            if config["label_preparation"].get("reuse_from"):
                from expand_masks import prepare_expanded
                index = prepare_expanded(config)
            else:
                from mask_only_data import prepare_mask_dataset
                index = prepare_mask_dataset(config)
        else:
            from dataset_update.dataset_union import open_dataset
            dataset = open_dataset(resolve(config["data_root"]))
            index = build_decomposition_index(config, dataset, resolve(config["teacher_experiment_root"]), output)
    else:
        name = "mask_records.json" if masks_only else "decomposition_records.json"
        index = json.loads((output / name).read_text())
    if args.command in ("train", "all"):
        train(config, output, index)
        if masks_only and config["training"].get("evaluate_after_training", False):
            from evaluate_masks import evaluate
            for seed in config["training"]["seeds"]:
                evaluate(config, output, index, seed=seed,
                         repetitions=config["training"].get("evaluation_bootstrap_repetitions", 1000))


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
