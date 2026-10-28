#!/usr/bin/env python3
"""Cache-only evaluation of complete local decomposition and Phase1 localization.

The main prediction starts with the original multi-source observation. Oracle
localization feeds teacher masks through the very same locator in a separately
labelled diagnostic; it is never reported as end-to-end student performance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve()
for path in (HERE.parent, HERE.parents[2], HERE.parents[2] / "src",
             HERE.parents[2] / "vendor/rind-dataset", HERE.parents[3] / "src",
             HERE.parents[3] / "Phase1_Single_Source_Single_Intensity/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decomposition_data import DecompositionRecords, decomposed_batches
from decomposition_model import decomposition_training_loss
from train_decomposition import load_config, make_model, resolve, write_json
from evaluate_slots import _summarize, _scene_balanced, _bootstrap, _stratify


def gpu_distribution_metrics(q, p, candidate, valid):
    """Vectorized equivalent of evaluate_slots.distribution_metrics on-device."""
    if q.ndim != 2 or p.shape != q.shape or valid.shape != q.shape:
        raise ValueError("distribution rows must be aligned [L,N]")
    q, p = q.double(), p.double()
    q = q / q.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    tiny = torch.finfo(torch.float64).tiny
    log_q, log_p = q.clamp_min(tiny).log(), p.clamp_min(tiny).log()
    entropy = -(q * log_q).sum(-1)
    cross_entropy = -(q * log_p).sum(-1)
    mixture = (q + p) * 0.5
    log_mixture = mixture.clamp_min(tiny).log()
    js = 0.5 * (q * (log_q - log_mixture)).sum(-1)
    js = js + 0.5 * (p * (log_p - log_mixture)).sum(-1)
    uniform = q.new_full((len(q),), math.log(q.shape[-1]))
    available = uniform - entropy
    variation = 0.5 * (q - p).abs().sum(-1)
    if candidate.ndim == 2:
        candidate = candidate[None].expand(len(q), -1, -1)
    candidate = candidate.double()
    rows = torch.arange(len(q), device=q.device)
    q_peak, p_peak = q.argmax(-1), p.argmax(-1)
    result = {
        "cross_entropy": cross_entropy, "teacher_entropy": entropy,
        "student_entropy": -(p * log_p).sum(-1),
        "forward_kl": (cross_entropy - entropy).clamp_min(0),
        "jensen_shannon": js, "total_variation": variation,
        "probability_overlap": 1 - variation, "uniform_cross_entropy": uniform,
        "information_recovery": torch.where(available > 1e-12,
                                                (uniform - cross_entropy) / available.clamp_min(1e-12),
                                                torch.zeros_like(available)),
        "invalid_candidate_mass": (p * ~valid).sum(-1),
        "top1_grid_agreement": (q_peak == p_peak).double(),
        "peak_coordinate_distance": (candidate[rows, q_peak] - candidate[rows, p_peak]).norm(dim=-1),
        "mean_coordinate_distance": ((candidate * q[..., None]).sum(1)
                                     - (candidate * p[..., None]).sum(1)).norm(dim=-1),
    }
    q_sorted = q.sort(dim=-1, descending=True).values
    p_sorted = p.sort(dim=-1, descending=True).values
    q_cumulative, p_cumulative = q_sorted.cumsum(-1), p_sorted.cumsum(-1)
    for mass in (0.50, 0.75, 0.90):
        q_cut = q_sorted.gather(1, (q_cumulative < mass).sum(-1).clamp_max(q.shape[-1] - 1)[:, None])
        p_cut = p_sorted.gather(1, (p_cumulative < mass).sum(-1).clamp_max(q.shape[-1] - 1)[:, None])
        tolerance = 8 * torch.finfo(torch.float64).eps
        teacher, student = q >= q_cut - tolerance, p >= p_cut - tolerance
        intersection = (teacher & student).sum(-1).double()
        teacher_area, student_area = teacher.sum(-1).double(), student.sum(-1).double()
        label = str(int(100 * mass))
        result[f"hdr_precision_{label}"] = intersection / student_area.clamp_min(1)
        result[f"hdr_recall_{label}"] = intersection / teacher_area.clamp_min(1)
        result[f"hdr_dice_{label}"] = 2 * intersection / (teacher_area + student_area).clamp_min(1)
        result[f"hdr_iou_{label}"] = intersection / (teacher_area + student_area - intersection).clamp_min(1)
        result[f"student_mass_in_teacher_hdr_{label}"] = (p * teacher).sum(-1)
    return result


def ratio_of_means(summary):
    if not summary or any(summary.get(name) is None for name in (
            "uniform_cross_entropy", "cross_entropy", "teacher_entropy")):
        return None
    uniform = summary["uniform_cross_entropy"]["mean"]
    entropy = summary["teacher_entropy"]["mean"]
    return ((uniform - summary["cross_entropy"]["mean"]) / (uniform - entropy)
            if uniform - entropy > 1e-12 else None)


def _metric_rows(metric_dict):
    if not metric_dict:
        return []
    keys = list(metric_dict)
    matrix = torch.stack([metric_dict[key].double() for key in keys], dim=-1).cpu().numpy()
    return [dict(zip(keys, map(float, values))) for values in matrix]


def _summary(rows):
    weighted, scenes = _summarize(rows), _scene_balanced(rows)
    return {
        "row_weighted": weighted, "scene_balanced": scenes,
        "information_recovery_ratio_of_means": {
            "row_weighted": ratio_of_means(weighted), "scene_balanced": ratio_of_means(scenes),
        },
    }


def _batch_rows(batch, output, losses, oracle_probability):
    assignment = losses["assignment"]
    real = batch["layer_mask"].bool()
    localizable = batch["localization_valid"].bool() & real
    batch_size, slots, height, width = batch["teacher_mask"].shape
    mask = output["mask_probability"].gather(1, assignment[..., None, None].expand(-1, -1, height, width))
    target = batch["teacher_mask"].float()
    free = ~batch["obstacle"]
    free_count = free.sum((-1, -2)).clamp_min(1)
    thresholded = mask >= 0.5
    intersection = (thresholded & batch["teacher_mask"] & free[:, None]).sum((-1, -2)).float()
    target_area = (batch["teacher_mask"] & free[:, None]).sum((-1, -2)).float()
    predicted_area = (thresholded & free[:, None]).sum((-1, -2)).float()
    dice = (2 * intersection + 1e-8) / (target_area + predicted_area + 1e-8)
    iou = (intersection + 1e-8) / (target_area + predicted_area - intersection + 1e-8)
    rho = output["presence_probability"].gather(1, assignment)
    strength = output["intensity"].gather(1, assignment)
    real_count = real.sum(-1).clamp_min(1)
    predicted_count = (output["presence_probability"] >= 0.5).sum(-1)
    count_mode = output["count_logits"].argmax(-1)
    contribution = mask * (rho * strength * real)[..., None, None]
    target_contribution = target * (batch["teacher_intensity"] * real)[..., None, None]
    predicted_residual = batch["response"][:, None] - contribution.cumsum(1)
    target_residual = batch["response"][:, None] - target_contribution.cumsum(1)
    residual_mae = (((predicted_residual - target_residual).abs() * free[:, None]).sum((-1, -2))
                    / free_count[:, None] * real).sum(-1) / real_count
    negative_residual = ((torch.relu(-predicted_residual) * free[:, None]).sum((-1, -2))
                         / free_count[:, None] * real).sum(-1) / real_count
    reconstruction_mae = ((output["reconstruction"] - batch["response"]).abs() * free).sum((-1, -2)) / free_count
    reconstruction_max = ((output["reconstruction"] - batch["response"]).abs() * free).flatten(1).max(-1).values
    predicted_present = rho >= 0.5
    metrics = _metric_rows({
        "mask_dice": (dice * real).sum(-1) / real_count,
        "mask_iou": (iou * real).sum(-1) / real_count,
        "mask_recall": (intersection / target_area.clamp_min(1) * real).sum(-1) / real_count,
        "mask_precision": (intersection / predicted_area.clamp_min(1) * real).sum(-1) / real_count,
        "intensity_mae": ((strength - batch["teacher_intensity"]).abs() * real).sum(-1) / real_count,
        "matched_presence": (rho * real).sum(-1) / real_count,
        "count_correct": (predicted_count == real.sum(-1)).float(),
        "count_mae": (predicted_count - real.sum(-1)).abs().float(),
        "count_mode_correct": (count_mode == real.sum(-1)).float(),
        "reconstruction_mae": reconstruction_mae, "reconstruction_max_error": reconstruction_max,
        "residual_mae": residual_mae, "negative_residual_mae": negative_residual,
    })
    active_batch, active_teacher = torch.where(localizable)
    predicted_probability = output["slot_probability"][active_batch, assignment[active_batch, active_teacher]]
    target_probability = batch["teacher_probability"][active_batch, active_teacher]
    predicted_metrics = _metric_rows(gpu_distribution_metrics(
        target_probability, predicted_probability, batch["candidate_xy"][active_batch], batch["valid"][active_batch]))
    oracle_metrics = _metric_rows(gpu_distribution_metrics(
        target_probability, oracle_probability, batch["candidate_xy"][active_batch], batch["valid"][active_batch]))
    # Compare complete fields against complete targets. Invalid localization
    # layers stay present; the report separately selects all-reliable windows.
    all_q = batch["teacher_probability"] * real[:, :, None]
    teacher_occupancy = 1 - (1 - all_q).prod(1)
    aggregate_metrics = _metric_rows(gpu_distribution_metrics(
        teacher_occupancy, output["occupancy"], batch["candidate_xy"], batch["valid"]))
    layers_by_window = [[] for _ in range(batch_size)]
    for position, (b, t) in enumerate(zip(active_batch.cpu().tolist(), active_teacher.cpu().tolist())):
        layers_by_window[b].append({
            "teacher_step": t + 1, "student_slot": int(assignment[b, t]),
            "predicted": predicted_metrics[position], "oracle": oracle_metrics[position],
            "kl_gap_vs_oracle": predicted_metrics[position]["forward_kl"] - oracle_metrics[position]["forward_kl"],
        })
    rows = []
    for b in range(batch_size):
        rows.append({
            "scene_id": int(batch["scene_id"][b]), "view_id": int(batch["view_id"][b]),
            "window_size": height, "teacher_layer_count": int(real[b].sum()),
            "localizable_layer_count": int(localizable[b].sum()),
            "all_layers_localizable": bool(torch.equal(localizable[b], real[b])),
            "predicted_layer_count": int(predicted_count[b]),
            "count_mode": int(count_mode[b]), "metrics": metrics[b],
            "aggregate_metrics": aggregate_metrics[b], "localization_layers": layers_by_window[b],
            "presence": {
                "tp": int((predicted_present[b] & real[b]).sum()),
                "fp": int((predicted_present[b] & ~real[b]).sum()),
                "fn": int((~predicted_present[b] & real[b]).sum()),
            },
        })
    return rows


def _markdown(report):
    def mean(section, key):
        value = section.get("scene_balanced", {}).get(key)
        return "n/a" if value is None else f"{value['mean']:.6f}"
    decomp = report["decomposition"]
    predicted, oracle = report["predicted_mask_localization"], report["oracle_mask_localization"]
    aggregate = report["aggregate_all_layers"]
    return "\n".join([
        "# Phase2 完整响应解耦评估", "",
        f"- 最佳检查点 epoch：{report['checkpoint_epoch']}",
        f"- 独立场景 / 窗口 / 可定位层：{report['coverage']['scenes']} / {report['coverage']['windows']} / {report['coverage']['localizable_layers']}",
        f"- 局部二值掩码 Dice / IoU：{mean(decomp, 'mask_dice')} / {mean(decomp, 'mask_iou')}",
        f"- 完整有效层数 accuracy / MAE：{mean(decomp, 'count_correct')} / {mean(decomp, 'count_mae')}",
        f"- 响应重建 MAE / 过程残差 MAE：{mean(decomp, 'reconstruction_mae')} / {mean(decomp, 'residual_mae')}",
        f"- 预测掩码进入 Phase1：KL={mean(predicted, 'forward_kl')}；HDR Dice@90={mean(predicted, 'hdr_dice_90')}",
        f"- 教师真掩码进入 Phase1（独立上限诊断）：KL={mean(oracle, 'forward_kl')}；HDR Dice@90={mean(oracle, 'hdr_dice_90')}",
        f"- 端到端定位信息恢复率（scene-balanced ratio-of-means）：{predicted['information_recovery_ratio_of_means']['scene_balanced']}",
        f"- 完整整合场 HDR Dice@90：{mean(aggregate, 'hdr_dice_90')}", "",
        "预测掩码定位指标从原始多源响应出发；oracle 指标使用真实教师层掩码，不能作为端到端学生成绩。",
        "完整分解保留全部算术层；逐层主定位指标只计入通过物理质量阈值的层。完整整合场包含所有层，物理质量较差层的影响由可靠窗口子集单列检查。",
        "数量标签是完整有效分解层数，不等同于真实生成源数。原始数据和物理重渲染不参与本评估，输入仅为已验证的分解缓存。", "",
        "详情见 comprehensive_metrics.json 和 windows.jsonl。", "",
    ])


def evaluate(config, seed=0, bootstrap_repetitions=1000):
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cuda = config["training"].get("device", "cuda") == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    root = resolve(config["output_root"])
    index = json.loads((root / "decomposition_records.json").read_text())
    saved_path = root / "models" / f"seed_{seed}" / "best.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    if saved.get("schema") != "phase2-response-decomposition-v2":
        raise ValueError("checkpoint is not a mask-supervised decomposition model")
    if saved["config"]["model"] != config["model"]:
        raise ValueError("evaluation model configuration differs from the checkpoint")
    model = make_model(config, index.get("global_size", 1024), device, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    records = DecompositionRecords(index["records"]["test"], root / "decomposition_cache", "test",
                                   config["model"]["max_slots"], "world")
    destination = root / "evaluation" / "decomposition"
    shards = destination / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for stale in shards.glob(f"seed_{seed}_rank_*.jsonl"):
            stale.unlink()
    if world > 1:
        dist.barrier()
    started = time.perf_counter()
    with (shards / f"seed_{seed}_rank_{rank}.jsonl").open("w") as stream, torch.inference_mode():
        for number, batch in enumerate(decomposed_batches(
                records, config["training"]["batch_size"],
                prefetch=config["training"].get("prefetch_batches", 2),
                rank=rank, world_size=world, training=False), 1):
            batch = {key: value.pin_memory().to(device, non_blocking=True)
                     if cuda else value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=cuda and config["training"].get("amp_dtype") == "bfloat16"):
                output = model(batch["response"], batch["obstacle"], batch["window"],
                               batch["candidate_xy"], batch["support"],
                               candidate_chunk=config["training"]["candidate_chunk"], compute_location=True)
                active_b, active_t = torch.where(batch["layer_mask"] & batch["localization_valid"])
                if len(active_b):
                    # Oracle diagnostic is a separate call on true masks, not
                    # a replacement for the original-observation prediction.
                    oracle = model.locate_masks(
                        batch["teacher_mask"][active_b, active_t].float()[:, None],
                        batch["obstacle"][active_b], batch["window"][active_b],
                        batch["candidate_xy"][active_b], batch["support"][active_b],
                        config["training"]["candidate_chunk"],
                    )["slot_probability"][:, 0]
                else:
                    oracle = batch["teacher_probability"].new_zeros((0, records.candidate_count))
            losses = decomposition_training_loss(output, batch, **config["loss"])
            for row in _batch_rows(batch, output, losses, oracle):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if rank == 0 and (number == 1 or number % 25 == 0):
                print(f"decomposition evaluation | rank0 batches={number} checkpoint_epoch={saved['epoch']}", flush=True)
    if world > 1:
        dist.barrier()
    if rank != 0:
        dist.barrier()
        return None
    rows, identities = [], set()
    for shard in sorted(shards.glob(f"seed_{seed}_rank_*.jsonl")):
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            identity = (row["scene_id"], row["view_id"])
            if identity in identities:
                raise ValueError(f"evaluation duplicated a real test window: {identity}")
            identities.add(identity)
            rows.append(row)
    if len(rows) != len(records):
        raise ValueError(f"evaluation coverage mismatch: {len(rows)} vs {len(records)}")
    rows.sort(key=lambda row: (row["scene_id"], row["view_id"]))
    predicted_rows, oracle_rows, aggregate_rows = [], [], []
    for row in rows:
        base = {key: row[key] for key in ("scene_id", "view_id", "window_size", "teacher_layer_count")}
        aggregate_rows.append({**base, "metrics": row["aggregate_metrics"],
                               "all_layers_localizable": row["all_layers_localizable"]})
        for layer in row["localization_layers"]:
            predicted_rows.append({**base, "metrics": layer["predicted"]})
            oracle_rows.append({**base, "metrics": layer["oracle"]})
    confusion = np.zeros((config["model"]["max_slots"] + 1,) * 2, dtype=np.int64)
    presence = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        confusion[row["teacher_layer_count"], row["predicted_layer_count"]] += 1
        for key in presence:
            presence[key] += row["presence"][key]
    precision = presence["tp"] / max(1, presence["tp"] + presence["fp"])
    recall = presence["tp"] / max(1, presence["tp"] + presence["fn"])
    report = {
        "schema": "phase2-complete-decomposition-evaluation-v2",
        "checkpoint": str(saved_path.resolve()), "checkpoint_epoch": saved["epoch"],
        "seed": seed, "world_size": world, "elapsed_seconds": time.perf_counter() - started,
        "coverage": {"windows": len(rows), "scenes": len({row['scene_id'] for row in rows}),
                     "full_layers": sum(row["teacher_layer_count"] for row in rows),
                     "localizable_layers": len(predicted_rows),
                     "all_layers_localizable_windows": sum(row["all_layers_localizable"] for row in rows)},
        "decomposition": {**_summary(rows),
                          "by_window_size": _stratify(rows, "window_size"),
                          "by_full_layer_count": _stratify(rows, "teacher_layer_count")},
        "predicted_mask_localization": _summary(predicted_rows),
        "oracle_mask_localization": _summary(oracle_rows),
        "aggregate_all_layers": _summary(aggregate_rows),
        "aggregate_all_layers_localizable_windows": _summary([
            row for row in aggregate_rows if row["all_layers_localizable"]]),
        "count_confusion_rows_true_columns_predicted": confusion.tolist(),
        "presence": {**presence, "precision_at_0.5": precision, "recall_at_0.5": recall,
                     "f1_at_0.5": 2 * precision * recall / max(1e-15, precision + recall)},
        "scene_cluster_bootstrap_95ci": {
            "decomposition": _bootstrap(rows, ["mask_dice", "reconstruction_mae", "count_correct"],
                                         bootstrap_repetitions, seed + 3001),
            "predicted_localization": _bootstrap(predicted_rows, ["forward_kl", "hdr_dice_90"],
                                                  bootstrap_repetitions, seed + 4001),
        },
        "definitions": {
            "count": "all arithmetic effective layers; generating-source count is not used",
            "end_to_end": "original multi-source y -> predicted layer mask -> shared Phase1 locator",
            "oracle": "teacher layer mask -> same shared locator; separate diagnostic, never end-to-end",
            "matching": "exact GPU Hungarian using the mask/intensity/presence objective; no soft label spreading",
            "information_recovery": "main: ratio of summary mean gain to mean available information; per-row ratios also retained",
            "aggregate": "complete teacher q set compared to all predicted slots; all-localizable subset reported separately",
            "truth_access": "cache-only evaluation; no raw source IDs, raw dataset, geometry, or rendering needed",
        },
    }
    write_json(destination / "comprehensive_metrics.json", report)
    with (destination / "windows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "REPORT.zh-CN.md").write_text(_markdown(report))
    print(f"decomposition evaluation saved: {destination}", flush=True)
    if world > 1:
        dist.barrier()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.seed < 0 or args.bootstrap_repetitions < 0:
        parser.error("seed and bootstrap repetitions must be nonnegative")
    evaluate(load_config(args.config), args.seed, args.bootstrap_repetitions)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
