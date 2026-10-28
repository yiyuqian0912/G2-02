#!/usr/bin/env python3
"""Cache-only decomposition metrics. Never instantiate a Phase1 locator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from train_decomposition import load_config, make_model, resolve, write_json, digest, semantic_config
from mask_only_data import MaskRecords
from mask_validation import validate_mask_cache
from decomposition_data import decomposed_batches
from decomposition_model import decomposition_training_loss
from evaluate_slots import _summarize, _scene_balanced, _bootstrap, _stratify


def metric_rows(result, batch, assignment):
    """Compare masks at 0.5 after Hungarian alignment; presence evaluated separately."""
    free = (~batch["obstacle"])[:, None]
    gather = assignment[..., None, None].expand_as(batch["teacher_mask"])
    predicted = result["mask_probability"].gather(1, gather) >= .5
    target = batch["teacher_mask"].bool()
    intersection = (predicted & target & free).sum((-1, -2)).double()
    area_p = (predicted & free).sum((-1, -2)).double()
    area_t = (target & free).sum((-1, -2)).double()
    real = batch["layer_mask"].bool()
    positive = real.sum(1).clamp_min(1)
    def mean_layers(value):
        return (value * real).sum(1) / positive
    presence = result["presence_probability"].gather(1, assignment) >= .5
    a = result["intensity"].gather(1, assignment)
    tp = (presence & real).sum(1)
    predicted_count = presence.sum(1)
    true_count = real.sum(1)
    count_probability = result["count_logits"].argmax(1)
    free_pixels = free[:, 0].sum((-1, -2)).clamp_min(1)
    error = (result["reconstruction"] - batch["response"]).abs()
    contribution = (result["mask_probability"].gather(1, gather)
                    * result["intensity"].gather(1, assignment)[..., None, None]
                    * result["presence_probability"].gather(1, assignment)[..., None, None])
    predicted_residual = batch["response"][:, None] - (contribution * real[..., None, None]).cumsum(1)
    target_residual = batch["response"][:, None] - (
        target * (batch["teacher_intensity"] * real)[..., None, None]).cumsum(1)
    metric = {
        "mask_dice": mean_layers((2 * intersection) / (area_p + area_t).clamp_min(1)),
        "mask_iou": mean_layers(intersection / (area_p + area_t - intersection).clamp_min(1)),
        "mask_precision": mean_layers(intersection / area_p.clamp_min(1)),
        "mask_recall": mean_layers(intersection / area_t.clamp_min(1)),
        "strength_mae": mean_layers((a - batch["teacher_intensity"]).abs()),
        "presence_precision": tp / predicted_count.clamp_min(1),
        "presence_recall": tp / true_count.clamp_min(1),
        "count_accuracy": (predicted_count == true_count).double(),
        "count_mae": (predicted_count - true_count).abs().double(),
        "count_distribution_accuracy": (count_probability == true_count).double(),
        "reconstruction_mae": (error * free[:, 0]).sum((-1, -2)) / free_pixels,
        "reconstruction_relative_l1": (error * free[:, 0]).sum((-1, -2)) /
            (batch["response"] * free[:, 0]).sum((-1, -2)).clamp_min(1e-12),
        "residual_mae": mean_layers(((predicted_residual - target_residual).abs()
                                      * free).sum((-1, -2)) / free_pixels[:, None]),
        "negative_residual_fraction": mean_layers(((predicted_residual < -1e-6)
                                                    & free).sum((-1, -2)) / free_pixels[:, None]),
    }
    values = torch.stack(list(metric.values()), 1).detach().cpu().numpy()
    scenes, views, windows = (batch[key].cpu().numpy() for key in ("scene_id", "view_id", "window"))
    true_count, predicted_count = true_count.cpu().numpy(), predicted_count.cpu().numpy()
    return [{"scene_id": int(scenes[i]), "view_id": int(views[i]), "size": int(windows[i, 2]),
             "layer_count": int(true_count[i]), "predicted_count": int(predicted_count[i]),
             "metrics": dict(zip(metric, map(float, values[i])))} for i in range(len(values))]


def evaluate(config, output, index, seed=0, repetitions=1000):
    if config.get("task") != "decomposition_only":
        raise ValueError("evaluate_masks requires task=decomposition_only")
    validate_mask_cache(index, output / "mask_cache", config["model"]["max_slots"])
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = config["training"].get("device", "cuda") == "cuda"
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(max(1, config["training"].get("num_threads", 8) // world))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    tick = time.perf_counter()
    saved = torch.load(output / "models" / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=True)
    if saved["schema"] != "phase2-local-mask-decomposition-v1":
        raise ValueError("not a mask-only checkpoint")
    # Use the checkpoint's exact architecture, rather than silently accepting config edits.
    if saved["config"]["model"] != config["model"]:
        raise ValueError("checkpoint/model config mismatch")
    signed_index = {key: value for key, value in index.items() if key != "statistics"}
    if saved["signature"] != digest({"config": semantic_config(saved["config"]), "index": signed_index}):
        raise ValueError("checkpoint/cache mismatch")
    model = make_model(saved["config"], index["global_size"], device, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    records = MaskRecords(index["records"]["test"], output / "mask_cache", "test", config["model"]["max_slots"])
    rows = []
    with torch.inference_mode():
        for number, batch in enumerate(decomposed_batches(
                records, config["training"]["batch_size"], prefetch=config["training"].get("prefetch_batches", 2),
                rank=rank, world_size=world, training=False), 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=cuda and config["training"]["amp_dtype"] == "bfloat16"):
                result = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            assignment = decomposition_training_loss(result, batch, **config["loss"])["assignment"]
            rows.extend(metric_rows(result, batch, assignment))
            if rank == 0 and (number == 1 or number % 20 == 0):
                print(f"mask evaluation rank0: batches={number} local windows={len(rows)}", flush=True)
    if world > 1:
        collected = [None] * world
        dist.all_gather_object(collected, rows)
        rows = [row for part in collected for row in part]
    if rank != 0:
        return None
    if len(rows) != len(records) or len({(row['scene_id'], row['view_id']) for row in rows}) != len(rows):
        raise ValueError("evaluation duplicates or misses test windows")
    rows.sort(key=lambda row: (row["scene_id"], row["view_id"]))
    report = {
        "task": "decomposition_only", "best_epoch": saved["epoch"], "seed": seed,
        "windows": len(rows), "scenes": len(records.scene_ids),
        "candidate_renders": 0, "location_evaluated": False,
        "label_semantics": "minimum-positive/BFS effective layers; physical feasibility unchecked",
        "window_weighted": _summarize(rows), "scene_balanced": _scene_balanced(rows),
        "scene_bootstrap_95ci": _bootstrap(rows, list(rows[0]["metrics"]) if rows else [], repetitions, seed),
        "by_layer_count": _stratify(rows, "layer_count"), "by_window_size": _stratify(rows, "size"),
        "elapsed_seconds": time.perf_counter() - tick,
        "mask_threshold": .5, "presence_threshold": .5,
        "matching": "same exact Hungarian mask/strength/presence assignment as training; no location scores",
    }
    destination = output / "evaluation"
    write_json(destination / "decomposition_metrics.json", report)
    (destination / "windows.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    lines = ["# 局部分层解耦评估", "", f"最佳 epoch：{saved['epoch']}；测试视窗：{len(rows)}；场景：{len(records.scene_ids)}。", "",
             "仅评估分层掩码、强度、有效层数与响应重建；没有渲染或评估位置概率场。", "",
             "标签是既有最低正残差/BFS规则产生的有效层；重建正确不保证每层可由一个物理源实现。", "",
             "掩码指标按训练时 Hungarian 匹配计算，单独报告槽位激活与层数，避免把已匹配的掩码质量当成端到端检出率。", "",
             "| 指标 | 场景等权均值 | 95% 场景 bootstrap 区间 |", "|---|---:|---|" ]
    for key, value in report["scene_balanced"].items():
        ci = report["scene_bootstrap_95ci"].get(key)
        interval = f"[{ci['lower']:.6f}, {ci['upper']:.6f}]" if ci else "—"
        lines.append(f"| {key} | {value['mean']:.6f} | {interval} |")
    (destination / "decomposition_metrics.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(f"mask evaluation saved: {destination}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve(config["output_root"])
    evaluate(config, output, json.loads((output / "mask_records.json").read_text()),
             args.seed, args.bootstrap_repetitions)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
