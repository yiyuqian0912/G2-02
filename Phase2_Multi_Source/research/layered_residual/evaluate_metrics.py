"""Comprehensive, checkpoint-safe evaluation for the Phase-II layer student."""
from __future__ import print_function

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
PHASE2_ROOT = PACKAGE_ROOT / "Phase2_Multi_Source"
PHASE1_ROOT = PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity"
for item in (PHASE1_ROOT / "src", PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset", HERE.parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from layered_residual import combine_layer_probabilities
from train_layers import (LayerRecords, _informative_truth, _matching_distances,
                          _topk_nms, context, load_verified_index, read_config)


TINY = np.finfo(np.float64).tiny


def _normalize_probability_rows(probability, label):
    """Restore exact row normalization after torch float32 batch transport."""
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("{} probability rows must be finite and nonnegative".format(label))
    totals = value.sum(axis=1, dtype=np.float64)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("{} probability row has no finite positive mass".format(label))
    return value / totals[:, None]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _credible_indices(probability, mass):
    order = np.argsort(-probability, kind="stable")
    count = int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")) + 1
    return order[:min(count, len(order))]


def _highest_density_region(probability, mass):
    """Tie-inclusive high-density region containing at least ``mass`` probability."""
    probability = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-probability, kind="stable")
    boundary_index = min(
        int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")),
        len(order) - 1)
    cutoff = probability[order[boundary_index]]
    tolerance = max(np.finfo(np.float64).eps * max(1.0, abs(cutoff)) * 8.0, 1e-15)
    return probability >= cutoff - tolerance


def _distribution_metrics(q, p, candidate, valid, support_count):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (q.shape != p.shape or q.ndim != 1 or candidate.shape != (len(q), 2) or
            valid.shape != q.shape or not np.isfinite(q).all() or
            not np.isfinite(p).all() or np.any(q < 0) or np.any(p < 0) or
            not np.isclose(q.sum(), 1.0, atol=1e-7) or
            not np.isclose(p.sum(), 1.0, atol=1e-7)):
        raise ValueError("invalid distributions passed to comprehensive evaluation")
    q_positive = q > 0
    p_positive = p > 0
    entropy = float(-np.sum(q[q_positive] * np.log(q[q_positive])))
    cross_entropy = float(-np.sum(q[q_positive] * np.log(np.maximum(p[q_positive], TINY))))
    uniform_ce = math.log(int(support_count))
    mixture = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q[q_positive] * np.log(q[q_positive] / mixture[q_positive])))
    js += 0.5 * float(np.sum(p[p_positive] * np.log(p[p_positive] / mixture[p_positive])))
    tv = 0.5 * float(np.abs(q - p).sum())
    squared_l2 = float(np.square(q - p).sum())
    bc = float(np.sqrt(q * p).sum())
    denominator = float(np.linalg.norm(q) * np.linalg.norm(p))
    q_peak = int(np.argmax(q))
    p_peak = int(np.argmax(p))
    q_order = np.argsort(-q, kind="stable")
    student_rank = int(np.flatnonzero(q_order == p_peak)[0]) + 1
    teacher_rank_in_student = int(np.flatnonzero(np.argsort(-p, kind="stable") == q_peak)[0]) + 1
    q_mean = np.sum(candidate * q[:, None], axis=0)
    p_mean = np.sum(candidate * p[:, None], axis=0)
    available = uniform_ce - entropy
    gain = uniform_ce - cross_entropy
    result = {
        "cross_entropy": cross_entropy,
        "teacher_entropy": entropy,
        "student_entropy": float(-np.sum(p[p_positive] * np.log(p[p_positive]))),
        "forward_kl": max(0.0, cross_entropy - entropy),
        "uniform_cross_entropy": uniform_ce,
        "available_information_vs_uniform": available,
        "gain_over_uniform": gain,
        "normalized_information_recovery": gain / available if available > 1e-12 else 0.0,
        "perplexity": float(math.exp(min(cross_entropy, 700.0))),
        "perplexity_reduction_vs_uniform": float(math.exp(min(gain, 700.0))),
        "jensen_shannon": js,
        "total_variation": tv,
        "overlap": 1.0 - tv,
        "squared_l2_distance": squared_l2,
        "bhattacharyya_coefficient": bc,
        "hellinger_distance": math.sqrt(max(0.0, 1.0 - bc)),
        "cosine_similarity": float(np.dot(q, p) / denominator) if denominator > 0 else 0.0,
        "student_mass_on_teacher_valid_candidates": float(p[valid].sum()),
        "student_mass_on_teacher_invalid_candidates": float(p[~valid].sum()),
        "top1_grid_agreement": float(q_peak == p_peak),
        "teacher_peak_probability": float(q[q_peak]),
        "student_peak_probability": float(p[p_peak]),
        "student_probability_at_teacher_peak": float(p[q_peak]),
        "teacher_probability_at_student_peak": float(q[p_peak]),
        "teacher_peak_rank_in_student": float(teacher_rank_in_student),
        "reciprocal_rank_of_teacher_peak": 1.0 / teacher_rank_in_student,
        "student_peak_rank_in_teacher": float(student_rank),
        "peak_coordinate_distance": float(np.linalg.norm(candidate[q_peak] - candidate[p_peak])),
        "probability_mean_coordinate_distance": float(np.linalg.norm(q_mean - p_mean)),
    }
    for mass in (0.5, 0.9, 0.95):
        label = str(int(mass * 100))
        q_set = _credible_indices(q, mass)
        p_set = _credible_indices(p, mass)
        result["student_mass_on_teacher_{}pct_set".format(label)] = float(p[q_set].sum())
        result["teacher_{}pct_set_fraction".format(label)] = float(len(q_set) / len(q))
        result["teacher_mass_on_student_{}pct_set".format(label)] = float(q[p_set].sum())
        result["student_{}pct_set_fraction".format(label)] = float(len(p_set) / len(p))
        teacher_region = _highest_density_region(q, mass)
        student_region = _highest_density_region(p, mass)
        intersection = int(np.count_nonzero(teacher_region & student_region))
        teacher_area = int(np.count_nonzero(teacher_region))
        student_area = int(np.count_nonzero(student_region))
        union = teacher_area + student_area - intersection
        result["support_precision_{}".format(label)] = float(intersection / student_area)
        result["support_recall_{}".format(label)] = float(intersection / teacher_area)
        result["support_dice_{}".format(label)] = float(
            2 * intersection / (teacher_area + student_area))
        result["support_iou_{}".format(label)] = float(intersection / union)
        result["student_mass_in_teacher_region_{}".format(label)] = float(
            p[teacher_region].sum())
        result["teacher_mass_in_student_region_{}".format(label)] = float(
            q[student_region].sum())
        result["teacher_region_fraction_{}".format(label)] = float(teacher_area / len(q))
        result["student_region_fraction_{}".format(label)] = float(student_area / len(p))
    for count in (10, 100):
        actual = min(count, len(q))
        q_set = set(np.argsort(-q, kind="stable")[:actual].tolist())
        p_set = set(np.argsort(-p, kind="stable")[:actual].tolist())
        intersection = len(q_set & p_set)
        result["top_{}_overlap_fraction".format(count)] = float(intersection / actual)
        result["top_{}_jaccard".format(count)] = float(intersection / len(q_set | p_set))
    return result


def _describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)), "max": float(values.max()),
    }


def _summarize(rows):
    if not rows:
        return None
    keys = sorted(set.intersection(*(set(row["metrics"]) for row in rows)))
    return {key: _describe([row["metrics"][key] for row in rows]) for key in keys}


def _cluster_bootstrap(rows, keys, repetitions, seed):
    """Percentile CI from resampling independent scenes, preserving their rows."""
    if not rows or repetitions < 1:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["scene_id"])].append(row["metrics"])
    scenes = sorted(groups)
    if len(scenes) < 2:
        return None
    rng = np.random.default_rng(seed)
    result = {}
    for key in keys:
        per_scene = [np.asarray([item[key] for item in groups[scene]], dtype=np.float64)
                     for scene in scenes]
        estimates = np.empty(repetitions, dtype=np.float64)
        for iteration in range(repetitions):
            selection = rng.integers(0, len(scenes), size=len(scenes))
            values = np.concatenate([per_scene[index] for index in selection])
            estimates[iteration] = values.mean()
        result[key] = {"lower": float(np.quantile(estimates, 0.025)),
                       "upper": float(np.quantile(estimates, 0.975)),
                       "repetitions": int(repetitions),
                       "independent_scene_clusters": int(len(scenes))}
    return result


def _strata(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _summarize(group) for name, group in sorted(groups.items())}


def _coordinate_metrics(probability, teacher_probability, candidate, valid, truth,
                        radius, thresholds):
    count = len(truth)
    if count == 0:
        return None
    student_xy = _topk_nms(probability, candidate, valid, count, radius)
    teacher_xy = _topk_nms(teacher_probability, candidate, valid, count, radius)
    student = _matching_distances(student_xy, truth)
    teacher = _matching_distances(teacher_xy, truth)
    if student is None or teacher is None:
        return None
    result = {
        "true_informative_source_count": float(count),
        "student_mean_matched_distance": float(student.mean()),
        "student_median_matched_distance": float(np.median(student)),
        "student_rmse_matched_distance": float(np.sqrt(np.mean(student ** 2))),
        "student_max_matched_distance": float(student.max()),
        "teacher_mean_matched_distance": float(teacher.mean()),
        "teacher_median_matched_distance": float(np.median(teacher)),
        "teacher_rmse_matched_distance": float(np.sqrt(np.mean(teacher ** 2))),
        "teacher_max_matched_distance": float(teacher.max()),
        "student_minus_teacher_mean_distance": float(student.mean() - teacher.mean()),
    }
    for threshold in thresholds:
        label = "{:g}".format(float(threshold))
        result["student_recall_at_{}".format(label)] = float(np.mean(student <= threshold))
        result["teacher_recall_at_{}".format(label)] = float(np.mean(teacher <= threshold))
        result["student_all_sources_within_{}".format(label)] = float(np.all(student <= threshold))
        result["teacher_all_sources_within_{}".format(label)] = float(np.all(teacher <= threshold))
    return result


def _cost_bin(value, threshold):
    if value <= 1e-12:
        return "exact_0"
    if value <= 1e-3:
        return "(0,0.001]"
    if value <= threshold:
        return "(0.001,{:g}]".format(threshold)
    return "above_{:g}".format(threshold)


def _training_quality(output, index, seed):
    path = output / "training_summary.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text()).get(str(seed))
    if report is None:
        return None
    entropy = index["target_distribution_statistics"]["test"]["teacher_entropy"]["mean"]
    uniform = index["target_distribution_statistics"]["test"]["uniform_cross_entropy"]["mean"]
    test_ce = float(report["test_cross_entropy"])
    available = uniform - entropy
    gain = uniform - test_ce
    return {
        "best_epoch": int(report["best_epoch"]),
        "best_validation_cross_entropy": float(report["best_validation_cross_entropy"]),
        "test_cross_entropy": test_ce,
        "test_teacher_entropy": entropy,
        "test_forward_kl": max(0.0, test_ce - entropy),
        "test_uniform_cross_entropy": uniform,
        "test_available_information_vs_uniform": available,
        "test_gain_over_uniform": gain,
        "test_normalized_information_recovery": gain / available if available > 1e-12 else 0.0,
        "test_perplexity": float(math.exp(min(test_ce, 700.0))),
        "test_perplexity_reduction_vs_uniform": float(math.exp(min(gain, 700.0))),
        "test_minus_validation_cross_entropy": test_ce - float(report["best_validation_cross_entropy"]),
    }


def _markdown(report):
    def mean(section, key):
        value = section.get(key) if section else None
        return "n/a" if value is None else "{:.6f}".format(value["mean"])
    layer = report["distribution_metrics"]["layer_all_informative"]
    combined = report["distribution_metrics"]["combined_max_field"]
    coordinate = report["coordinate_metrics"]["overall"]
    coverage = report["coverage"]
    lines = [
        "# Phase II 学生模型综合评价", "",
        "- 检查点：`{}`（epoch {}）".format(report["checkpoint"], report["checkpoint_epoch"]),
        "- 测试窗口：{} / {}；信息层：{}；高置信层：{}".format(
            coverage["windows_evaluated"], coverage["test_windows_total"],
            coverage["informative_layers_evaluated"], coverage["high_confidence_layers_evaluated"]),
        "- 单层 Shape Dice@90 / Recall / Precision / IoU：{} / {} / {} / {}".format(
            mean(layer, "support_dice_90"), mean(layer, "support_recall_90"),
            mean(layer, "support_precision_90"), mean(layer, "support_iou_90")),
        "- 单层 CE / forward KL / JS / overlap：{} / {} / {} / {}".format(
            mean(layer, "cross_entropy"), mean(layer, "forward_kl"),
            mean(layer, "jensen_shannon"), mean(layer, "overlap")),
        "- 单层相对均匀基线增益 / 信息恢复率：{} / {}".format(
            mean(layer, "gain_over_uniform"), mean(layer, "normalized_information_recovery")),
        "- 多层 max Shape Dice@90 / Recall / Precision / IoU：{} / {} / {} / {}".format(
            mean(combined, "support_dice_90"), mean(combined, "support_recall_90"),
            mean(combined, "support_precision_90"), mean(combined, "support_iou_90")),
        "- 多层 max 汇总 CE / JS / overlap：{} / {} / {}".format(
            mean(combined, "cross_entropy"), mean(combined, "jensen_shannon"),
            mean(combined, "overlap")),
        "- 多层 max 汇总峰值坐标距离：{} 像素".format(
            mean(combined, "peak_coordinate_distance")),
    ]
    if coordinate:
        lines.append("- 学生 / 教师匹配真源平均距离：{} / {} 像素".format(
            mean(coordinate, "student_mean_matched_distance"),
            mean(coordinate, "teacher_mean_matched_distance")))
    lines += ["", "## 判读顺序", "",
              "1. 先看 90% 最高密度区域的 Dice、Recall、Precision 与 IoU，评价源可行区域的几何形状。",
              "2. Recall 衡量教师认为可能的区域是否被学生覆盖；Precision 惩罚学生额外预测的错误区域。",
              "3. 再看 CE、forward KL、JS 与概率 overlap，评价区域内部概率是否校准。",
              "4. 用 normalized information recovery 判断学生恢复了教师可用信息的比例。",
              "5. 用峰值距离、匹配真源距离和 Recall@阈值检查定位能力。",
              "6. 对照窗口大小、层序号、教师物理代价和真源数量分层结果，定位失败子集。",
              "", "完整数值与定义见 `comprehensive_metrics.json`；逐窗口结果见 `comprehensive_windows.jsonl`。", ""]
    return "\n".join(lines)


def evaluate(config, dataset, teacher_root, output, index, seed, device_name,
             max_windows, bootstrap_repetitions):
    import torch
    from rind_phase1.model import student_distribution
    from rind_phase1.train import collate_records, load_model

    checkpoint_path = output / "models" / "seed_{}".format(seed) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("best checkpoint is missing: {}".format(checkpoint_path))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model, checkpoint = load_model(checkpoint_path)
    model.to(device).eval()
    groups = defaultdict(list)
    for row in index["evaluation_test_records"]:
        groups[(int(row["scene_id"]), int(row["view_id"]))].append(row)
    group_items = sorted(groups.items())
    if max_windows is not None:
        group_items = group_items[:max_windows]
    layer_rows = []
    combined_rows = []
    coordinate_rows = []
    window_rows = []
    skipped_zero = skipped_one = 0
    support_mode = config["training"]["normalization_support"]
    cost_threshold = float(config["layer_filter"]["maximum_training_physical_cost"])
    candidate_chunk = int(config["evaluation"]["candidate_chunk"])
    started = time.perf_counter()
    with torch.inference_mode():
        for position, ((scene, view), rows) in enumerate(group_items, 1):
            rows = sorted(rows, key=lambda item: int(item["step"]))
            informative = [row for row in rows if row["informative"]]
            skipped_zero += sum(bool(row["all_zero"]) for row in rows)
            skipped_one += sum(bool(row["all_one"]) for row in rows)
            if not informative:
                window_rows.append({"scene_id": scene, "view_id": view,
                                    "status": "no_informative_layer", "window_size": int(rows[0]["size"])})
                continue
            records = LayerRecords(
                informative,dataset,teacher_root,support_mode,
                config["training"].get("use_scene_geometry",False))
            samples = [records[index_value] for index_value in range(len(records))]
            batch = collate_records(samples)
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded = model.encode_observation(
                batch["response"],batch["obstacle"],batch["window"],
                batch.get("geometry_types"),batch.get("geometry_params"),
                batch.get("geometry_mask"))
            energy = torch.cat([
                model.score_encoded(encoded, batch["candidate_xy"][:, start:start + candidate_chunk])
                for start in range(0, batch["candidate_xy"].shape[1], candidate_chunk)
            ], dim=1)
            log_p, probability = student_distribution(energy.float(), batch["support"])
            if not torch.isfinite(log_p[batch["support"]]).all() or not torch.isfinite(probability).all():
                raise ValueError("student produced non-finite probabilities at scene={}, view={}".format(scene, view))
            # collate_records intentionally transports teacher targets as
            # float32.  Re-normalize in float64 before the strict shared max
            # combiner; this removes only round-off from batch transport.
            q = _normalize_probability_rows(
                batch["teacher_prob"].cpu().numpy(), "teacher")
            p = _normalize_probability_rows(
                probability.cpu().numpy(), "student")
            candidate = batch["candidate_xy"].cpu().numpy().astype(np.float64)
            valid = batch["valid"].cpu().numpy().astype(bool)
            support = batch["support"].cpu().numpy().astype(bool)
            for array in (candidate, valid, support):
                if not all(np.array_equal(array[0], other) for other in array[1:]):
                    raise ValueError("layers in a window do not use the same candidate grid/support")
            per_layer = []
            for layer_index, row in enumerate(informative):
                metrics = _distribution_metrics(q[layer_index], p[layer_index], candidate[layer_index],
                                                valid[layer_index], int(support[layer_index].sum()))
                item = {"scene_id": scene, "view_id": view, "window_size": int(row["size"]),
                        "step": int(row["step"]), "intensity": float(row["intensity"]),
                        "minimum_physical_cost": float(row["minimum_physical_cost"]),
                        "physical_cost_bin": _cost_bin(float(row["minimum_physical_cost"]), cost_threshold),
                        "confidence": "high" if row["high_confidence"] else "low",
                        "metrics": metrics}
                layer_rows.append(item)
                per_layer.append(item)
            teacher_max = combine_layer_probabilities(list(q))
            student_max = combine_layer_probabilities(list(p))
            combined_metrics = _distribution_metrics(
                teacher_max, student_max, candidate[0], valid[0], int(support[0].sum()))
            truth = _informative_truth(dataset, scene, view)
            coordinate = _coordinate_metrics(
                student_max, teacher_max, candidate[0], valid[0], truth,
                float(config["evaluation"]["nms_radius"]), config["evaluation"]["distance_thresholds"])
            combined_item = {"scene_id": scene, "view_id": view,
                             "window_size": int(informative[0]["size"]),
                             "informative_layer_count": len(informative),
                             "high_confidence_layer_count": sum(bool(row["high_confidence"]) for row in informative),
                             "informative_true_source_count": len(truth), "metrics": combined_metrics}
            combined_rows.append(combined_item)
            if coordinate is not None:
                coordinate_rows.append({**{key: value for key, value in combined_item.items() if key != "metrics"},
                                        "metrics": coordinate})
            window_rows.append({**combined_item, "status": "evaluated", "coordinate": coordinate,
                                "layers": per_layer})
            if position == 1 or position % 100 == 0 or position == len(group_items):
                print("comprehensive evaluation {}/{} windows".format(position, len(group_items)), flush=True)

    high = [row for row in layer_rows if row["confidence"] == "high"]
    low = [row for row in layer_rows if row["confidence"] == "low"]
    ci_keys = ["cross_entropy", "forward_kl", "gain_over_uniform",
               "normalized_information_recovery", "jensen_shannon", "total_variation",
               "overlap", "support_dice_90", "support_recall_90", "support_precision_90",
               "support_iou_90", "student_mass_in_teacher_region_90",
               "teacher_mass_in_student_region_90", "peak_coordinate_distance"]
    report = {
        "schema": "phase1-layer-student-comprehensive-evaluation-v2",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": int(seed), "device": str(device), "elapsed_seconds": time.perf_counter() - started,
        "target_semantics": config["target_semantics"],
        "combine_rule": config["evaluation"]["combine_rule"],
        "truth_policy": "true sources are used only for coordinate evaluation, never prediction",
        "training_quality": _training_quality(output, index, seed),
        "coverage": {
            "test_windows_total": len(group_items),
            "windows_evaluated": sum(row["status"] == "evaluated" for row in window_rows),
            "windows_without_informative_layers": sum(row["status"] == "no_informative_layer" for row in window_rows),
            "windows_with_coordinate_metrics": len(coordinate_rows),
            "informative_layers_evaluated": len(layer_rows), "high_confidence_layers_evaluated": len(high),
            "low_confidence_layers_evaluated": len(low), "all_zero_layers": int(skipped_zero),
            "all_one_layers": int(skipped_one), "independent_test_scenes": len(set(
                row["scene_id"] for row in combined_rows)),
        },
        "distribution_metrics": {
            "layer_all_informative": _summarize(layer_rows),
            "layer_high_confidence": _summarize(high),
            "layer_low_confidence": _summarize(low),
            "combined_max_field": _summarize(combined_rows),
        },
        "scene_cluster_bootstrap_95ci": {
            "layer_all_informative": _cluster_bootstrap(layer_rows, ci_keys, bootstrap_repetitions, seed + 1001),
            "combined_max_field": _cluster_bootstrap(combined_rows, ci_keys, bootstrap_repetitions, seed + 2001),
        },
        "layer_stratification": {
            "by_window_size": _strata(layer_rows, "window_size"),
            "by_layer_step": _strata(layer_rows, "step"),
            "by_teacher_physical_cost": _strata(layer_rows, "physical_cost_bin"),
            "by_confidence": _strata(layer_rows, "confidence"),
        },
        "combined_stratification": {
            "by_window_size": _strata(combined_rows, "window_size"),
            "by_informative_layer_count": _strata(combined_rows, "informative_layer_count"),
            "by_informative_true_source_count": _strata(combined_rows, "informative_true_source_count"),
        },
        "coordinate_metrics": {
            "overall": _summarize(coordinate_rows),
            "by_window_size": _strata(coordinate_rows, "window_size"),
            "by_informative_layer_count": _strata(coordinate_rows, "informative_layer_count"),
            "by_informative_true_source_count": _strata(coordinate_rows, "informative_true_source_count"),
        },
        "metric_definitions": {
            "forward_kl": "KL(teacher || student) = cross_entropy - teacher_entropy; lower is better",
            "normalized_information_recovery": "(uniform_CE - model_CE) / (uniform_CE - teacher_entropy); 1 is teacher-perfect, 0 is uniform",
            "jensen_shannon": "symmetric bounded distribution divergence; lower is better",
            "total_variation": "half L1 distance; lower is better", "overlap": "1 - total_variation; higher is better",
            "credible_set_mass": "student probability captured by the teacher's smallest highest-probability credible set",
            "support_geometry": "tie-inclusive highest-density regions at equal probability mass; Recall measures teacher-region coverage, Precision penalizes student-only area, Dice and IoU summarize shape overlap",
            "combined_max_field": "normalized pointwise maximum across independently predicted residual-layer fields; a hotspot summary, not a joint posterior",
            "coordinate_matching": "NMS peaks are matched one-to-one with informative true sources by minimum total Euclidean distance",
            "probability_mean_coordinate_distance": "distance between distribution means; interpret cautiously for multimodal fields",
        },
        "evaluation_parameters": {
            "normalization_support": support_mode, "candidate_count": int(index["candidate_grid"]["candidate_count"]),
            "nms_radius": float(config["evaluation"]["nms_radius"]),
            "distance_thresholds": config["evaluation"]["distance_thresholds"],
            "teacher_high_confidence_cost_threshold": cost_threshold,
            "scene_bootstrap_repetitions": int(bootstrap_repetitions), "max_windows": max_windows,
        },
    }
    destination = output / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "comprehensive_metrics.json", report)
    with (destination / "comprehensive_windows.jsonl").open("w") as stream:
        for row in window_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "comprehensive_metrics.md").write_text(_markdown(report))
    print("Saved comprehensive metrics to {}".format(destination), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("--max-windows must be positive")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be nonnegative")
    config = read_config(args.config)
    dataset, teacher_root, output = context(config)
    index = load_verified_index(config, teacher_root, output)
    evaluate(config, dataset, teacher_root, output, index, args.seed, args.device,
             args.max_windows, args.bootstrap_repetitions)


if __name__ == "__main__":
    main()
