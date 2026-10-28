#!/usr/bin/env python3
"""Index existing layered teachers and train the one-pass implicit-slot student."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
PHASE2_ROOT = HERE.parents[2]
PACKAGE_ROOT = HERE.parents[3]
PHASE1_ROOT = PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity"
for path in (HERE.parent, PHASE1_ROOT / "src", PHASE2_ROOT / "src", PHASE2_ROOT / "vendor/rind-dataset"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slot_data import WindowSlotRecords, build_slot_index, slot_batch_count, slot_batches
from slot_model import ImplicitContributionSlotField, slot_training_loss
from rind_phase2.data import Phase2Dataset


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _index_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Configuration fields that determine cached teacher-record contents."""
    return {
        "data_root": config["data_root"],
        "teacher_experiment_root": config["teacher_experiment_root"],
        "layer_filter": config["layer_filter"],
        "model": {"max_slots": config["model"]["max_slots"]},
        "normalization_support": config["training"]["normalization_support"],
    }


def _progress(root: Path, event: str, message: str, **fields: Any) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": stamp, "event": event, **fields}, allow_nan=False) + "\n")


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {"data_root", "teacher_experiment_root", "output_root", "layer_filter", "model", "training", "loss"}
    if not required <= set(config):
        raise ValueError(f"missing config fields: {sorted(required - set(config))}")
    model = config["model"]
    training = config["training"]
    layer_filter = config["layer_filter"]
    threshold = layer_filter.get("maximum_training_physical_cost")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("layer_filter.maximum_training_physical_cost must be nonnegative")
    if model.get("max_slots") != 8:
        raise ValueError("the audited first version requires model.max_slots=8")
    if model.get("architecture") != "multiscale_boundary" or model.get("attention_heads") != 8:
        raise ValueError("implicit slots require the Phase-I multiscale_boundary eight-head backbone")
    for key in ("hidden", "fourier_frequencies", "slot_decoder_layers", "slot_attention_heads"):
        if type(model.get(key)) is not int or model[key] < (0 if key == "fourier_frequencies" else 1):
            raise ValueError(f"model.{key} is invalid")
    if model["hidden"] % model["slot_attention_heads"]:
        raise ValueError("slot_attention_heads must divide hidden")
    for key in ("epochs", "patience", "batch_size", "candidate_chunk", "num_threads", "prefetch_batches"):
        if type(training.get(key)) is not int or training[key] < (0 if key == "prefetch_batches" else 1):
            raise ValueError(f"training.{key} is invalid")
    target_batch = training.get("target_global_batch_size", training["batch_size"])
    if type(target_batch) is not int or target_batch < 1:
        raise ValueError("training.target_global_batch_size must be a positive integer")
    if type(training.get("activation_checkpoint_candidates", True)) is not bool:
        raise ValueError("training.activation_checkpoint_candidates must be boolean")
    if type(training.get("ddp_no_sync", False)) is not bool:
        raise ValueError("training.ddp_no_sync must be boolean")
    if training.get("device") not in ("cpu", "cuda") or training.get("amp_dtype", "none") not in ("none", "bfloat16", "float16"):
        raise ValueError("invalid device or AMP dtype")
    devices = training.get("gpu_devices", [0])
    if not isinstance(devices, list) or not devices or len(set(devices)) != len(devices) or any(type(x) is not int or x < 0 for x in devices):
        raise ValueError("gpu_devices must contain distinct nonnegative integers")
    if training.get("normalization_support") not in ("world", "geometry"):
        raise ValueError("normalization_support must be world or geometry")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training.seeds must contain distinct nonnegative integers")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        if not isinstance(training.get(key), (int, float)) or not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} is invalid")
    if training["learning_rate"] <= 0 or training["grad_clip"] <= 0:
        raise ValueError("learning_rate and grad_clip must be positive")
    for key, value in config["loss"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{key} is invalid")
    if type(config["loss"].get("sinkhorn_iterations")) is not int or config["loss"]["sinkhorn_iterations"] < 1:
        raise ValueError("loss.sinkhorn_iterations must be a positive integer")
    if config["loss"].get("sinkhorn_temperature", 0) <= 0:
        raise ValueError("loss.sinkhorn_temperature must be positive")
    return config


def context(config: dict[str, Any]) -> tuple[Phase2Dataset, Path, Path]:
    data_root = Path(config["data_root"])
    teacher_root = Path(config["teacher_experiment_root"])
    output_root = Path(config["output_root"])
    if not data_root.is_absolute():
        data_root = PHASE2_ROOT / data_root
    if not teacher_root.is_absolute():
        teacher_root = PHASE2_ROOT / teacher_root
    if not output_root.is_absolute():
        output_root = PHASE2_ROOT / output_root
    return Phase2Dataset(data_root), teacher_root, output_root


def _backbone_config(config: dict[str, Any], dataset: Phase2Dataset) -> dict[str, Any]:
    model = config["model"]
    return {
        "hidden": model["hidden"],
        "fourier_frequencies": model["fourier_frequencies"],
        "coordinate_scale": float(dataset.manifest["global_size"]),
        "use_obstacle": bool(model.get("use_obstacle", False)),
        "pixel_coordinates": bool(model.get("pixel_coordinates", True)),
        "normalization_support": config["training"]["normalization_support"],
        "architecture": model["architecture"],
        "attention_heads": model["attention_heads"],
    }


def build_model(config: dict[str, Any], dataset: Phase2Dataset, device: Any,
                initialize: bool = True) -> ImplicitContributionSlotField:
    import torch
    model_config = config["model"]
    model = ImplicitContributionSlotField(
        max_slots=model_config["max_slots"],
        slot_decoder_layers=model_config["slot_decoder_layers"],
        slot_attention_heads=model_config["slot_attention_heads"],
        slot_dropout=float(model_config.get("slot_dropout", 0.0)),
        activation_checkpoint_candidates=bool(
            config["training"].get("activation_checkpoint_candidates", True)
        ),
        backbone_config=_backbone_config(config, dataset),
    ).to(device)
    checkpoint_value = config["training"].get("initial_phase1_checkpoint")
    if initialize and checkpoint_value:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PHASE2_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Phase-I initialization is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_phase1_backbone(checkpoint)
    return model


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    if device.type == "cuda":
        return {key: value.pin_memory().to(device, non_blocking=True) for key, value in batch.items()}
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model: Any, records: WindowSlotRecords, config: dict[str, Any], device: Any,
              optimizer: Any = None, rng: np.random.Generator | None = None,
              progress: Any = None, rank: int = 0, world_size: int = 1) -> dict[str, float]:
    import torch
    training = config["training"]
    loss_config = config["loss"]
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    count = 0
    amp_dtype = training.get("amp_dtype", "none")
    amp_enabled = device.type == "cuda" and amp_dtype != "none"
    torch_amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    micro_batch = int(training["batch_size"])
    total_batches = slot_batch_count(records, micro_batch, world_size)
    target_global_batch = int(training.get("target_global_batch_size", micro_batch * world_size))
    accumulation_steps = (
        max(1, math.ceil(target_global_batch / (micro_batch * world_size)))
        if optimizer is not None else 1
    )
    for batch_index, batch in enumerate(slot_batches(
            records, micro_batch, rng, training.get("prefetch_batches", 0),
            rank, world_size), 1):
        batch = _move(batch, device)
        group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
        group_end = min(group_start + accumulation_steps - 1, total_batches)
        group_size = group_end - group_start + 1
        optimizer_step = optimizer is not None and batch_index == group_end
        sync_context = (
            model.no_sync()
            if (optimizer is not None and world_size > 1 and not optimizer_step and
                training.get("ddp_no_sync", False))
            else nullcontext()
        )
        with sync_context:
            with torch.set_grad_enabled(optimizer is not None):
                with torch.autocast(device_type=device.type, dtype=torch_amp_dtype, enabled=amp_enabled):
                    output = model(
                        batch["response"], batch["obstacle"], batch["window"],
                        batch["candidate_xy"], batch["support"], training["candidate_chunk"],
                    )
                losses = slot_training_loss(
                    output, batch["teacher_probability"], batch["teacher_core"],
                    batch["teacher_intensity"], batch["layer_mask"], batch["slot_supervised"],
                    batch["teacher_occupancy"], batch["count_target"],
                    core_weight=loss_config["core_weight"],
                    intensity_weight=loss_config["intensity_weight"],
                    presence_weight=loss_config["presence_weight"],
                    aggregate_weight=loss_config["aggregate_weight"],
                    count_weight=loss_config["count_weight"],
                    consistency_weight=loss_config["consistency_weight"],
                    sinkhorn_temperature=loss_config["sinkhorn_temperature"],
                    sinkhorn_iterations=int(loss_config["sinkhorn_iterations"]),
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("non-finite implicit-slot loss")
                if optimizer is not None:
                    (losses["loss"] / group_size).backward()
        if optimizer_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_count = len(batch["response"])
        count += batch_count
        for key in ("loss", "slot", "aggregate", "count", "consistency", "count_accuracy"):
            totals[key] = totals.get(key, 0.0) + float(losses[key].detach()) * batch_count
        if progress is not None:
            progress(batch_index, batch_count, {key: value / count for key, value in totals.items()})
    if world_size > 1:
        import torch.distributed as dist
        keys = sorted(totals)
        packed = torch.tensor([totals[key] for key in keys] + [float(count)],
                              dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        count = int(packed[-1].item())
        totals = {key: float(packed[index].item()) for index, key in enumerate(keys)}
    return {key: value / count for key, value in totals.items()}


def _checkpoint(path: Path, value: dict[str, Any]) -> None:
    import torch
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def train(config: dict[str, Any], dataset: Phase2Dataset, teacher_root: Path,
          output_root: Path, index: dict[str, Any], runtime_gpus: int | None) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    settings = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    configured_gpu_ids = list(settings.get("gpu_devices", [0]))
    gpu_ids = configured_gpu_ids
    if runtime_gpus is not None and runtime_gpus > 1 and not distributed:
        raise RuntimeError("multi-GPU slot training must be launched through run_implicit_slots.sh/torchrun")
    if settings["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if distributed:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"local rank {local_rank} exceeds visible CUDA devices")
            device = torch.device(f"cuda:{local_rank}")
            gpu_ids = list(range(world_size))
        else:
            # The launcher owns physical-device selection through
            # CUDA_VISIBLE_DEVICES.  A selected physical GPU (for example 3)
            # is therefore always logical cuda:0 in a one-process run.
            if os.environ.get("RIND_PHYSICAL_GPU_IDS"):
                gpu_ids = [0]
                device = torch.device("cuda:0")
            else:
                if not configured_gpu_ids or max(configured_gpu_ids) >= torch.cuda.device_count():
                    raise RuntimeError(f"requested CUDA devices {configured_gpu_ids} are unavailable")
                device = torch.device(f"cuda:{configured_gpu_ids[0]}")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
    else:
        gpu_ids, device = [], torch.device("cpu")
    if distributed and device.type != "cuda":
        raise RuntimeError("distributed slot training requires CUDA")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.set_num_threads(max(1, settings["num_threads"] // world_size))
    cache_root = output_root / "training_cache"
    splits = {name: WindowSlotRecords(index["records"][name], dataset, cache_root, name,
                                      config["model"]["max_slots"], settings["normalization_support"])
              for name in ("train", "validation", "test")}
    if splits["train"].scene_ids & splits["validation"].scene_ids or splits["train"].scene_ids & splits["test"].scene_ids or splits["validation"].scene_ids & splits["test"].scene_ids:
        raise ValueError("scene leakage across slot splits")
    reports = {}
    for seed in settings["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if gpu_ids:
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        base_model = build_model(config, dataset, device, initialize=True)
        if settings.get("channels_last", True):
            base_model.to(memory_format=torch.channels_last)
        model = (DistributedDataParallel(
            base_model, device_ids=[local_rank], output_device=local_rank,
            # PyTorch 2.4's reducer can hit expect_autograd_hooks_ when
            # static_graph is combined with no_sync gradient accumulation and
            # non-reentrant activation checkpointing.  The model still has a
            # fixed parameter graph; leaving static_graph disabled uses DDP's
            # supported checkpoint/no_sync path without changing gradients.
            broadcast_buffers=False, gradient_as_bucket_view=True,
        ) if distributed else base_model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
            fused=bool(settings.get("fused_optimizer", False) and device.type == "cuda"),
        )
        run_root = output_root / "models" / f"seed_{seed}"
        if rank == 0:
            run_root.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()
        signature = _digest({"config": config, "index": index, "seed": seed})
        history, start, best, bad = [], 0, float("inf"), 0
        if (run_root / "last.pt").is_file():
            saved = torch.load(run_root / "last.pt", map_location="cpu", weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("resume checkpoint does not match config/index")
            base_model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history, start, best, bad = saved["history"], saved["epoch"], saved["best"], saved["bad"]
            rng.bit_generator.state = saved["numpy_rng"]
        parameters = sum(parameter.numel() for parameter in base_model.parameters())
        accumulation_steps = max(1, math.ceil(
            int(settings.get("target_global_batch_size", settings["batch_size"] * world_size)) /
            (settings["batch_size"] * world_size)
        ))
        effective_global_batch = settings["batch_size"] * world_size * accumulation_steps
        if rank == 0:
            physical_gpu_ids = os.environ.get("RIND_PHYSICAL_GPU_IDS", "")
            _progress(output_root, "train_start",
                      f"implicit slots | windows train/validation/test={len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} | slots={config['model']['max_slots']} | parameters={parameters:,} | DDP GPUs={world_size if distributed else gpu_ids or ['cpu']} physical={physical_gpu_ids or 'default'} | per-GPU micro-batch={settings['batch_size']}, global micro-batch={settings['batch_size']*world_size}, accumulation={accumulation_steps}, effective global batch={effective_global_batch} | candidate chunk={settings['candidate_chunk']}, activation checkpoint={settings.get('activation_checkpoint_candidates', True)}, no_sync={settings.get('ddp_no_sync', False)} | resume epoch={start}",
                      records=index["counts"], max_slots=config["model"]["max_slots"], parameters=parameters,
                      gpu_ids=gpu_ids, world_size=world_size, per_gpu_batch=settings["batch_size"],
                      global_batch=settings["batch_size"]*world_size,
                      accumulation_steps=accumulation_steps,
                      effective_global_batch=effective_global_batch,
                      candidate_chunk=settings["candidate_chunk"], physical_gpu_ids=physical_gpu_ids,
                      activation_checkpoint_candidates=settings.get("activation_checkpoint_candidates", True),
                      ddp_no_sync=settings.get("ddp_no_sync", False),
                      resume_epoch=start)
        started = time.perf_counter()
        every = int(settings.get("progress_every_batches", 10))
        for epoch in range(start + 1, settings["epochs"] + 1):
            if bad >= settings["patience"]:
                break
            tick = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            def callback(phase: str):
                def report(batch_index: int, batch_count: int, running: dict[str, float]) -> None:
                    if batch_index == 1 or batch_index % every == 0:
                        _progress(output_root, "batch_progress",
                                  f"{phase} epoch={epoch}/{settings['epochs']} batch={batch_index} records={batch_count} loss={running['loss']:.6f} count_acc={running['count_accuracy']:.4f}",
                                  phase=phase, epoch=epoch, batch=batch_index, running=running)
                return report
            train_metrics = run_epoch(model, splits["train"], config, device, optimizer, rng,
                                      callback("train") if rank == 0 else None, rank, world_size)
            validation_metrics = run_epoch(model, splits["validation"], config, device,
                                           progress=callback("validation") if rank == 0 else None,
                                           rank=rank, world_size=world_size)
            peak_memory = torch.tensor([
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
            ], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            peak_allocated_gib, peak_reserved_gib = map(float, peak_memory.tolist())
            improved = validation_metrics["loss"] < best
            best = min(best, validation_metrics["loss"])
            bad = 0 if improved else bad + 1
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
                            "seconds": time.perf_counter() - tick,
                            "peak_allocated_gib": peak_allocated_gib,
                            "peak_reserved_gib": peak_reserved_gib})
            state = {
                "schema": "phase2-implicit-contribution-slots-v1", "model": base_model.state_dict(),
                "model_config": config["model"], "backbone_config": _backbone_config(config, dataset),
                "epoch": epoch, "best": best, "bad": bad, "history": history,
                "optimizer": optimizer.state_dict(), "signature": signature,
                "numpy_rng": rng.bit_generator.state,
            }
            if rank == 0:
                if improved:
                    _checkpoint(run_root / "best.pt", state)
                _checkpoint(run_root / "last.pt", state)
                _json(run_root / "training.json", {"best_validation_loss": best, "history": history})
                _progress(output_root, "epoch_done",
                          f"epoch {epoch} done | train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f} count_acc={validation_metrics['count_accuracy']:.4f} best={best:.6f} bad={bad}/{settings['patience']} | peak allocated/reserved={peak_allocated_gib:.2f}/{peak_reserved_gib:.2f} GiB",
                          epoch=epoch, train=train_metrics, validation=validation_metrics,
                          best_validation_loss=best, bad_epochs=bad,
                          peak_allocated_gib=peak_allocated_gib,
                          peak_reserved_gib=peak_reserved_gib)
            if distributed:
                dist.barrier()
        if distributed:
            dist.barrier()
        saved = torch.load(run_root / "best.pt", map_location="cpu", weights_only=True)
        base_model.load_state_dict(saved["model"])
        test_metrics = run_epoch(model, splits["test"], config, device,
                                 rank=rank, world_size=world_size)
        report = {"seed": seed, "best_epoch": saved["epoch"], "best_validation_loss": saved["best"],
                  "test": test_metrics, "records": index["counts"],
                  "elapsed_seconds": time.perf_counter() - started}
        if rank == 0:
            _json(run_root / "training.json", {**report, "history": history})
            reports[str(seed)] = report
            _progress(output_root, "train_done",
                      f"implicit slots done | seed={seed} best epoch={saved['epoch']} test loss={test_metrics['loss']:.6f} count_acc={test_metrics['count_accuracy']:.4f}",
                      **report)
        if distributed:
            dist.barrier()
    if rank == 0:
        _json(output_root / "training_summary.json", reports)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "train", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=int)
    args = parser.parse_args()
    if args.gpus is not None and args.gpus < 1:
        parser.error("--gpus must be positive")
    config = read_config(args.config)
    dataset, teacher_root, output_root = context(config)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command in ("index", "all"):
        index = build_slot_index(config, dataset, teacher_root, output_root)
        _json(output_root / "protocol.json", {
            "schema": "phase2-implicit-contribution-slots-v1",
            "config": config,
            "index_contract": _index_contract(config),
            "student_input": "original multi-source response plus window; no residual masks, layer order, source IDs, or scene geometry",
            "matching": "GPU Sinkhorn one-to-one teacher-layer/slot matching; observation tokens are shared and non-exclusive",
            "overflow": index["overflow_policy"],
        })
        print("Indexed:", index["counts"], "overflow:",
              {name: index["statistics"][name].get("overflow_windows", 0) for name in index["statistics"]})
    else:
        index = json.loads((output_root / "slot_records.json").read_text())
        protocol = json.loads((output_root / "protocol.json").read_text())
        stored_contract = protocol.get("index_contract")
        if stored_contract is None and isinstance(protocol.get("config"), dict):
            stored_contract = _index_contract(protocol["config"])
        if (protocol.get("schema") != "phase2-implicit-contribution-slots-v1" or
                stored_contract != _index_contract(config)):
            raise ValueError("data/model indexing fields differ from the slot protocol; rerun index")
    if args.command in ("train", "all"):
        train(config, dataset, teacher_root, output_root, index, args.gpus)


def _cleanup_distributed() -> None:
    """Release NCCL cleanly on both success and Python-level failures."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Preserve the original training exception; torchrun will terminate
        # sibling ranks if a process has already failed.
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_distributed()
