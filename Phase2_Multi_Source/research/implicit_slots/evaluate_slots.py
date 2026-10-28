#!/usr/bin/env python3
"""Comprehensive permutation-invariant evaluation of the Phase-II slot student."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
PHASE2_ROOT = HERE.parents[2]
PACKAGE_ROOT = HERE.parents[3]
PHASE1_ROOT = PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity"
for path in (HERE.parent, PHASE1_ROOT / "src", PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slot_data import WindowSlotRecords, slot_batches
from train_slots import build_model, context, read_config

TINY = np.finfo(np.float64).tiny


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def highest_density_region(probability: np.ndarray, mass: float) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("probability must be finite, nonnegative, and one-dimensional")
    total = float(value.sum())
    if total <= 0 or not 0 < mass < 1:
        raise ValueError("probability mass and HDR mass are invalid")
    value = value / total
    order = np.argsort(-value, kind="stable")
    boundary = min(int(np.searchsorted(np.cumsum(value[order]), mass, side="left")), len(order) - 1)
    cutoff = float(value[order[boundary]])
    tolerance = max(np.finfo(np.float64).eps * max(1.0, abs(cutoff)) * 8, 1e-15)
    return value >= cutoff - tolerance


def distribution_metrics(q: np.ndarray, p: np.ndarray, candidate: np.ndarray,
                         valid: np.ndarray) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if q.shape != p.shape or q.ndim != 1 or candidate.shape != (len(q), 2) or valid.shape != q.shape:
        raise ValueError("distribution metric inputs are misaligned")
    if (not np.isfinite(q).all() or not np.isfinite(p).all() or
            np.any(q < 0) or np.any(p < 0) or q.sum() <= 0 or p.sum() <= 0):
        raise ValueError("distribution metric inputs are invalid")
    q, p = q / q.sum(), p / p.sum()
    q_positive, p_positive = q > 0, p > 0
    entropy = float(-np.sum(q[q_positive] * np.log(q[q_positive])))
    cross_entropy = float(-np.sum(q[q_positive] * np.log(np.maximum(p[q_positive], TINY))))
    uniform_ce = math.log(len(q))
    mixture = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q[q_positive] * np.log(q[q_positive] / mixture[q_positive])))
    js += 0.5 * float(np.sum(p[p_positive] * np.log(p[p_positive] / mixture[p_positive])))
    tv = 0.5 * float(np.abs(q - p).sum())
    q_peak, p_peak = int(np.argmax(q)), int(np.argmax(p))
    q_mean = np.sum(candidate * q[:, None], axis=0)
    p_mean = np.sum(candidate * p[:, None], axis=0)
    available = uniform_ce - entropy
    result = {
        "cross_entropy": cross_entropy,
        "teacher_entropy": entropy,
        "student_entropy": float(-np.sum(p[p_positive] * np.log(p[p_positive]))),
        "forward_kl": max(0.0, cross_entropy - entropy),
        "jensen_shannon": js,
        "total_variation": tv,
        "probability_overlap": 1.0 - tv,
        "uniform_cross_entropy": uniform_ce,
        "information_recovery": ((uniform_ce - cross_entropy) / available
                                 if available > 1e-12 else 0.0),
        "invalid_candidate_mass": float(p[~valid].sum()),
        "top1_grid_agreement": float(q_peak == p_peak),
        "peak_coordinate_distance": float(np.linalg.norm(candidate[q_peak] - candidate[p_peak])),
        "mean_coordinate_distance": float(np.linalg.norm(q_mean - p_mean)),
    }
    for mass in (0.50, 0.75, 0.90):
        label = str(int(mass * 100))
        teacher = highest_density_region(q, mass)
        student = highest_density_region(p, mass)
        intersection = int(np.count_nonzero(teacher & student))
        teacher_area, student_area = int(teacher.sum()), int(student.sum())
        union = teacher_area + student_area - intersection
        result[f"hdr_precision_{label}"] = intersection / student_area
        result[f"hdr_recall_{label}"] = intersection / teacher_area
        result[f"hdr_dice_{label}"] = 2 * intersection / (teacher_area + student_area)
        result[f"hdr_iou_{label}"] = intersection / union
        result[f"student_mass_in_teacher_hdr_{label}"] = float(p[teacher].sum())
    return result


def exact_assignment(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost injective teacher-row to student-slot assignment, R<=8."""
    value = np.asarray(cost, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] > value.shape[1] or not np.isfinite(value).all():
        raise ValueError("assignment cost must be finite [T,R] with T<=R")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for teacher in range(value.shape[0]):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, assignment) in states.items():
            for slot in range(value.shape[1]):
                if mask & (1 << slot):
                    continue
                new_mask = mask | (1 << slot)
                proposal = (total + float(value[teacher, slot]), assignment + (slot,))
                if new_mask not in updated or proposal[0] < updated[new_mask][0]:
                    updated[new_mask] = proposal
        states = updated
    best = min(states.values(), key=lambda item: item[0])
    return np.asarray(best[1], dtype=np.int64)


def _matching_cost(q: np.ndarray, core: np.ndarray, intensity: np.ndarray,
                   p: np.ndarray, log_p: np.ndarray, predicted_intensity: np.ndarray,
                   presence_logits: np.ndarray, loss_config: dict[str, Any]) -> np.ndarray:
    teachers, slots = len(q), len(p)
    result = np.empty((teachers, slots), dtype=np.float64)
    for teacher in range(teachers):
        positive = q[teacher] > 0
        ce = -np.sum(q[teacher, positive, None] * log_p[:, positive].T, axis=0)
        core_mass = np.sum(p[:, core[teacher]], axis=1)
        core_loss = -np.log(np.maximum(core_mass, TINY))
        intensity_cost = np.abs(float(intensity[teacher]) - predicted_intensity)
        positive_presence = np.logaddexp(0.0, -presence_logits)
        result[teacher] = (
            ce + float(loss_config["core_weight"]) * core_loss
            + float(loss_config["intensity_weight"]) * intensity_cost
            + float(loss_config["presence_weight"]) * positive_presence
        )
    return result


def _topk_nms(probability: np.ndarray, candidate: np.ndarray, valid: np.ndarray,
              count: int, radius: float = 32.0) -> np.ndarray:
    order = np.argsort(-np.asarray(probability), kind="stable")
    selected = []
    for index in order:
        if not valid[index]:
            continue
        point = candidate[index]
        if all(np.linalg.norm(point - other) >= radius for other in selected):
            selected.append(point)
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.float64)


def _matched_distances(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray | None:
    if len(predicted) != len(truth) or not len(truth):
        return None
    distance = np.linalg.norm(predicted[:, None, :] - truth[None, :, :], axis=-1)
    best = min(itertools.permutations(range(len(truth))),
               key=lambda order: sum(distance[row, column] for row, column in enumerate(order)))
    return np.asarray([distance[row, column] for row, column in enumerate(best)], dtype=np.float64)


def _informative_truth(dataset: Any, scene: int, view: int) -> np.ndarray:
    observation = dataset.get_observation(scene, view)
    free = ~np.asarray(observation["obstacle"], dtype=bool)
    channels = np.asarray(dataset.get_channels(scene, view), dtype=np.float64)
    sources = np.asarray(dataset.get_source_params(scene), dtype=np.float64)
    keep = []
    for index, channel in enumerate(channels):
        visible = channel[free] > 1e-12
        if visible.any() and not visible.all():
            keep.append(sources[index, :2])
    return np.asarray(keep, dtype=np.float64).reshape(-1, 2)


def _describe(values: list[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {
        "count": int(len(array)), "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def _summarize(rows: list[dict[str, Any]], field: str = "metrics") -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row[field]) for row in rows)))
    return {key: _describe([float(row[field][key]) for row in rows]) for key in keys}


def _scene_balanced(rows: list[dict[str, Any]], field: str = "metrics") -> dict[str, Any]:
    groups: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        groups[int(row["scene_id"])].append(row[field])
    if not groups:
        return {}
    keys = sorted(set.intersection(*(set(item) for values in groups.values() for item in values)))
    return {key: _describe([
        float(np.mean([item[key] for item in groups[scene]])) for scene in groups
    ]) for key in keys}


def _bootstrap(rows: list[dict[str, Any]], keys: list[str], repetitions: int,
               seed: int, field: str = "metrics") -> dict[str, Any]:
    groups: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        groups[int(row["scene_id"])].append(row[field])
    scenes = sorted(groups)
    if len(scenes) < 2 or repetitions < 1:
        return {}
    rng = np.random.default_rng(seed)
    result = {}
    for key in keys:
        estimates = np.empty(repetitions, dtype=np.float64)
        for iteration in range(repetitions):
            sampled = rng.integers(0, len(scenes), size=len(scenes))
            estimates[iteration] = np.mean([
                np.mean([item[key] for item in groups[scenes[index]]]) for index in sampled
            ])
        result[key] = {
            "lower": float(np.quantile(estimates, 0.025)),
            "upper": float(np.quantile(estimates, 0.975)),
            "repetitions": repetitions, "scene_clusters": len(scenes),
        }
    return result


def _stratify(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _summarize(values) for name, values in sorted(groups.items())}


def _window_row(batch: dict[str, Any], output: dict[str, Any], index: int,
                candidate: np.ndarray, dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    scene, view = int(batch["scene_id"][index]), int(batch["view_id"][index])
    layer_mask = batch["layer_mask"][index].cpu().numpy().astype(bool)
    supervised = bool(batch["slot_supervised"][index])
    q_all = batch["teacher_probability"][index].cpu().numpy().astype(np.float64)
    core_all = batch["teacher_core"][index].cpu().numpy().astype(bool)
    teacher_intensity_all = batch["teacher_intensity"][index].cpu().numpy().astype(np.float64)
    valid = batch["valid"][index].cpu().numpy().astype(bool)
    p = output["slot_probability"][index].detach().cpu().numpy().astype(np.float64)
    log_p = output["slot_log_probability"][index].detach().cpu().numpy().astype(np.float64)
    presence = output["presence_probability"][index].detach().cpu().numpy().astype(np.float64)
    presence_logits = output["presence_logits"][index].detach().cpu().numpy().astype(np.float64)
    predicted_intensity = output["intensity"][index].detach().cpu().numpy().astype(np.float64)
    count_probability = output["count_logits"][index].detach().float().softmax(-1).cpu().numpy()
    count_target = int(batch["count_target"][index])
    count_prediction = int(np.argmax(count_probability))
    teacher_occupancy = batch["teacher_occupancy"][index].cpu().numpy().astype(np.float64)
    student_occupancy = output["occupancy"][index].detach().cpu().numpy().astype(np.float64)
    aggregate = distribution_metrics(teacher_occupancy, student_occupancy, candidate, valid)
    intersection = float(np.sum(teacher_occupancy * student_occupancy))
    aggregate["soft_occupancy_dice"] = (
        (2 * intersection + 1e-8) /
        (float(np.square(teacher_occupancy).sum() + np.square(student_occupancy).sum()) + 1e-8)
    )
    aggregate["raw_occupancy_l1"] = float(np.mean(np.abs(teacher_occupancy - student_occupancy)))
    aggregate["count_correct"] = float(count_prediction == count_target)
    aggregate["count_absolute_error"] = float(abs(count_prediction - count_target))
    aggregate["count_nll"] = float(-math.log(max(float(count_probability[count_target]), TINY)))
    layers = []
    matched_slots: set[int] = set()
    if supervised:
        q = q_all[layer_mask]
        core = core_all[layer_mask]
        teacher_intensity = teacher_intensity_all[layer_mask]
        assignment = exact_assignment(_matching_cost(
            q, core, teacher_intensity, p, log_p, predicted_intensity,
            presence_logits, config["loss"],
        ))
        matched_slots = set(map(int, assignment))
        for teacher, slot in enumerate(assignment):
            metrics = distribution_metrics(q[teacher], p[slot], candidate, valid)
            metrics["intensity_absolute_error"] = float(abs(
                teacher_intensity[teacher] - predicted_intensity[slot]))
            metrics["matched_presence_probability"] = float(presence[slot])
            layers.append({"teacher_layer": teacher, "student_slot": int(slot), "metrics": metrics})
    truth = _informative_truth(dataset, scene, view)
    coordinate = None
    if len(truth):
        student_points = _topk_nms(student_occupancy, candidate, valid, len(truth))
        teacher_points = _topk_nms(teacher_occupancy, candidate, valid, len(truth))
        student_distance = _matched_distances(student_points, truth)
        teacher_distance = _matched_distances(teacher_points, truth)
        if student_distance is not None and teacher_distance is not None:
            coordinate = {
                "informative_true_source_count": float(len(truth)),
                "student_mean_source_distance": float(student_distance.mean()),
                "teacher_mean_source_distance": float(teacher_distance.mean()),
                "student_minus_teacher_distance": float(student_distance.mean() - teacher_distance.mean()),
            }
            for radius in (32, 64, 128):
                coordinate[f"student_recall_at_{radius}"] = float(np.mean(student_distance <= radius))
                coordinate[f"teacher_recall_at_{radius}"] = float(np.mean(teacher_distance <= radius))
    presence_truth = np.asarray([slot in matched_slots for slot in range(len(presence))], dtype=bool)
    presence_prediction = presence >= 0.5
    tp = int(np.count_nonzero(presence_truth & presence_prediction))
    fp = int(np.count_nonzero(~presence_truth & presence_prediction))
    fn = int(np.count_nonzero(presence_truth & ~presence_prediction))
    return {
        "scene_id": scene, "view_id": view,
        "window_size": int(batch["response"].shape[-1]),
        "teacher_layer_count": count_target, "predicted_layer_count": count_prediction,
        "slot_supervised": supervised, "metrics": aggregate, "layers": layers,
        "coordinate": coordinate,
        "presence_counts": {"tp": tp, "fp": fp, "fn": fn} if supervised else None,
    }


def _markdown(report: dict[str, Any]) -> str:
    def mean(section: dict[str, Any], key: str) -> str:
        value = section.get(key)
        return "n/a" if not value else f"{value['mean']:.6f}"
    aggregate = report["aggregate_occupancy"]["scene_balanced"]
    layer = report["matched_layers"]["scene_balanced"]
    count = report["count_metrics"]
    coordinate = report["coordinate_metrics"]["scene_balanced"]
    return "\n".join([
        "# Phase2 隐式槽位学生综合评估", "",
        f"- 检查点：`{report['checkpoint']}`（epoch {report['checkpoint_epoch']}）",
        f"- 测试窗口：{report['coverage']['windows']}；独立场景：{report['coverage']['scenes']}",
        f"- 有效层数量准确率 / MAE：{count['accuracy']:.4f} / {count['mae']:.4f}",
        f"- 整合场 HDR Dice@90 / Recall / Precision：{mean(aggregate, 'hdr_dice_90')} / {mean(aggregate, 'hdr_recall_90')} / {mean(aggregate, 'hdr_precision_90')}",
        f"- 整合场 JS / overlap / invalid mass：{mean(aggregate, 'jensen_shannon')} / {mean(aggregate, 'probability_overlap')} / {mean(aggregate, 'invalid_candidate_mass')}",
        f"- 匹配单层 KL / 信息恢复率 / HDR Dice@90：{mean(layer, 'forward_kl')} / {mean(layer, 'information_recovery')} / {mean(layer, 'hdr_dice_90')}",
        f"- 匹配单层强度 MAE / presence：{mean(layer, 'intensity_absolute_error')} / {mean(layer, 'matched_presence_probability')}",
        f"- 学生 / 教师到信息性真源的平均距离：{mean(coordinate, 'student_mean_source_distance')} / {mean(coordinate, 'teacher_mean_source_distance')} 像素", "",
        "## 判读", "",
        "1. 先看整合场 HDR Dice/Recall/Precision，它们直接评价最终源可能区域的几何形状。",
        "2. 再看匹配单层 KL、信息恢复率与 Dice，判断各贡献槽位是否复现教师层。",
        "3. count accuracy 评价有效残差层数，不等同于真实物理源数量。",
        "4. 真源距离仅作诊断；残差有效贡献与真实源不是严格一一对应。",
        "5. occupancy 是多槽位软占据场，不是完整多源联合后验。", "",
        "完整定义和置信区间见 `comprehensive_metrics.json`；逐窗口结果见 `windows.jsonl` 与 `windows.csv`。", "",
    ])


def evaluate(config: dict[str, Any], seed: int, bootstrap_repetitions: int) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("slot evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if distributed:
        dist.init_process_group("nccl", init_method="env://")
    dataset, _, output_root = context(config)
    index = json.loads((output_root / "slot_records.json").read_text())
    checkpoint_path = output_root / "models" / f"seed_{seed}" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(config, dataset, device, initialize=False)
    model.load_state_dict(checkpoint["model"])
    if config["training"].get("channels_last", True):
        model.to(memory_format=torch.channels_last)
    model.eval()
    records = WindowSlotRecords(
        index["records"]["test"], dataset, output_root / "training_cache", "test",
        config["model"]["max_slots"], config["training"]["normalization_support"],
    )
    candidate = np.asarray(records.candidate, dtype=np.float64)
    evaluation_root = output_root / "evaluation" / "implicit_slots"
    shard_root = evaluation_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for stale in shard_root.glob(f"seed_{seed}_rank_*.jsonl"):
            stale.unlink()
    if distributed:
        dist.barrier()
    shard_path = shard_root / f"seed_{seed}_rank_{rank}.jsonl"
    amp_dtype = config["training"].get("amp_dtype", "none")
    torch_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    count = 0
    with shard_path.open("w") as stream, torch.inference_mode():
        for batch_index, batch in enumerate(slot_batches(
                records, config["training"]["batch_size"], None,
                config["training"].get("prefetch_batches", 0), rank, world_size), 1):
            batch = {key: value.pin_memory().to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch_dtype, enabled=amp_dtype != "none"):
                output = model(
                    batch["response"], batch["obstacle"], batch["window"],
                    batch["candidate_xy"], batch["support"],
                    config["training"]["candidate_chunk"],
                )
            for item in range(len(batch["response"])):
                row = _window_row(batch, output, item, candidate, dataset, config)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
            if rank == 0 and (batch_index == 1 or batch_index % 25 == 0):
                print(f"evaluation batch={batch_index} windows_on_rank0={count}", flush=True)
    if distributed:
        dist.barrier()
    if rank != 0:
        if distributed:
            # Keep every rank alive until rank 0 has merged shards and written
            # the final report, then tear NCCL down in a coordinated order.
            dist.barrier()
            dist.destroy_process_group()
        return None
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for shard in sorted(shard_root.glob(f"seed_{seed}_rank_*.jsonl")):
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            deduplicated[(int(row["scene_id"]), int(row["view_id"]))] = row
    rows = [deduplicated[key] for key in sorted(deduplicated)]
    layer_rows = [
        {"scene_id": row["scene_id"], "view_id": row["view_id"],
         "window_size": row["window_size"], **layer}
        for row in rows for layer in row["layers"]
    ]
    coordinate_rows = [
        {"scene_id": row["scene_id"], "view_id": row["view_id"],
         "window_size": row["window_size"], "metrics": row["coordinate"]}
        for row in rows if row["coordinate"] is not None
    ]
    confusion = np.zeros((config["model"]["max_slots"] + 1,) * 2, dtype=np.int64)
    for row in rows:
        confusion[row["teacher_layer_count"], row["predicted_layer_count"]] += 1
    presence = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        if row["presence_counts"]:
            for key in presence:
                presence[key] += int(row["presence_counts"][key])
    precision = presence["tp"] / max(1, presence["tp"] + presence["fp"])
    recall = presence["tp"] / max(1, presence["tp"] + presence["fn"])
    report = {
        "schema": "phase2-implicit-slot-comprehensive-evaluation-v1",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": seed, "world_size": world_size, "elapsed_seconds": time.perf_counter() - started,
        "coverage": {
            "windows": len(rows), "scenes": len({row["scene_id"] for row in rows}),
            "matched_layers": len(layer_rows),
            "slot_supervised_windows": sum(bool(row["slot_supervised"]) for row in rows),
            "overflow_windows": sum(not bool(row["slot_supervised"]) for row in rows),
            "coordinate_windows": len(coordinate_rows),
        },
        "count_metrics": {
            "accuracy": float(np.mean([row["teacher_layer_count"] == row["predicted_layer_count"] for row in rows])),
            "mae": float(np.mean([abs(row["teacher_layer_count"] - row["predicted_layer_count"]) for row in rows])),
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        },
        "presence_metrics": {**presence, "precision_at_0.5": precision, "recall_at_0.5": recall,
                             "f1_at_0.5": 2 * precision * recall / max(TINY, precision + recall)},
        "aggregate_occupancy": {
            "window_weighted": _summarize(rows), "scene_balanced": _scene_balanced(rows),
            "by_window_size": _stratify(rows, "window_size"),
            "by_teacher_layer_count": _stratify(rows, "teacher_layer_count"),
        },
        "matched_layers": {
            "layer_weighted": _summarize(layer_rows), "scene_balanced": _scene_balanced(layer_rows),
            "by_window_size": _stratify(layer_rows, "window_size"),
        },
        "coordinate_metrics": {
            "window_weighted": _summarize(coordinate_rows),
            "scene_balanced": _scene_balanced(coordinate_rows),
        },
        "scene_cluster_bootstrap_95ci": {
            "aggregate": _bootstrap(rows, ["hdr_dice_90", "hdr_recall_90", "hdr_precision_90",
                                                   "jensen_shannon", "probability_overlap",
                                                   "invalid_candidate_mass"],
                                    bootstrap_repetitions, seed + 1001),
            "matched_layers": _bootstrap(layer_rows, ["forward_kl", "information_recovery",
                                                        "hdr_dice_90", "intensity_absolute_error"],
                                         bootstrap_repetitions, seed + 2001),
        },
        "metric_definitions": {
            "count": "teacher retained effective residual-layer count, not generating physical-source count",
            "matching": "exact minimum-cost injective matching using the same CE/core/intensity/presence terms as training",
            "aggregate_occupancy": "normalized shape comparison of 1-product(1-presence*slot_probability); not a joint posterior",
            "hdr": "tie-inclusive highest-density regions at equal probability mass",
            "information_recovery": "(uniform CE - student CE)/(uniform CE - teacher entropy)",
            "coordinate": "NMS peaks compared with sources having nonconstant true visibility in the local window; truth is evaluation-only",
        },
        "parameters": {"bootstrap_repetitions": bootstrap_repetitions,
                       "candidate_count": int(index["candidate_grid"]["candidate_count"])},
    }
    _write_json(evaluation_root / "comprehensive_metrics.json", report)
    with (evaluation_root / "windows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (evaluation_root / "windows.csv").open("w", newline="") as stream:
        fields = ["scene_id", "view_id", "window_size", "teacher_layer_count",
                  "predicted_layer_count", "count_correct", "hdr_dice_90",
                  "hdr_recall_90", "hdr_precision_90", "jensen_shannon",
                  "probability_overlap", "invalid_candidate_mass"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "scene_id": row["scene_id"], "view_id": row["view_id"],
                "window_size": row["window_size"], "teacher_layer_count": row["teacher_layer_count"],
                "predicted_layer_count": row["predicted_layer_count"],
                **{key: row["metrics"][key] for key in fields[5:]},
            })
    (evaluation_root / "REPORT.zh-CN.md").write_text(_markdown(report))
    print(f"evaluation saved: {evaluation_root}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return report


def _cleanup_distributed() -> None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.seed < 0 or args.bootstrap_repetitions < 0:
        parser.error("seed and bootstrap repetitions must be nonnegative")
    evaluate(read_config(args.config), args.seed, args.bootstrap_repetitions)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_distributed()
