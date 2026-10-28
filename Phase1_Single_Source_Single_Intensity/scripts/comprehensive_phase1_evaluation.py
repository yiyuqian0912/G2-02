#!/usr/bin/env python3
"""Comprehensive evaluation of saved Phase-I teacher/student probability fields.

This command never renders teachers and never runs the student network.  It
loads the predictions written by the locked test evaluation and measures
distribution calibration, highest-density-region geometry, physical validity,
and single-source coordinate localization.  Ground-truth source coordinates
are used only by the coordinate metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rind_phase1.data import Phase1Dataset
from rind_phase1.teacher import load_teacher_record


DEFAULT_OUTPUT = Path("outputs/experiments/phase1_100k_multiscale_h20_memmap")
TINY = np.finfo(np.float64).tiny
CORE_MASSES = (0.5, 0.75, 0.9, 0.95)
PRIMARY_CORE_MASS = 0.75


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize(probability: Any, label: str) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{label} must be a finite nonnegative vector")
    total = float(value.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label} has no positive mass")
    return value / total


def _credible_indices(probability: np.ndarray, mass: float) -> np.ndarray:
    order = np.argsort(-probability, kind="stable")
    count = int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")) + 1
    return order[: min(count, len(order))]


def _highest_density_region(probability: np.ndarray, mass: float) -> np.ndarray:
    """Tie-inclusive highest-density region containing at least ``mass``."""
    order = np.argsort(-probability, kind="stable")
    boundary = min(
        int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")),
        len(order) - 1,
    )
    cutoff = float(probability[order[boundary]])
    tolerance = max(np.finfo(np.float64).eps * max(1.0, abs(cutoff)) * 8.0, 1e-15)
    return probability >= cutoff - tolerance


def _polarized_core(probability: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Drop the low-probability tail and renormalize the retained HDR mass."""
    core = np.where(np.asarray(region, dtype=bool), probability, 0.0)
    return _normalize(core, "polarized core probability")


def _entropy(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-np.sum(probability[positive] * np.log(probability[positive])))


def _distribution_metrics(
    teacher: np.ndarray,
    student: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    q = _normalize(teacher, "teacher probability")
    p = _normalize(student, "student probability")
    xy = np.asarray(candidate, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if xy.shape != (len(q), 2) or valid.shape != q.shape or support.shape != q.shape:
        raise ValueError("candidate coordinates and supports must align with probabilities")
    if not support.any() or np.any(p[~support] > 1e-12) or np.any(q[~valid] > 1e-12):
        raise ValueError("teacher/student probability lies outside its declared support")

    q_positive = q > 0
    p_positive = p > 0
    teacher_entropy = _entropy(q)
    cross_entropy = float(-np.sum(q[q_positive] * np.log(np.maximum(p[q_positive], TINY))))
    uniform_ce = math.log(int(support.sum()))
    mixture = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q[q_positive] * np.log(q[q_positive] / mixture[q_positive])))
    js += 0.5 * float(np.sum(p[p_positive] * np.log(p[p_positive] / mixture[p_positive])))
    total_variation = 0.5 * float(np.abs(q - p).sum())
    coefficient = float(np.sqrt(q * p).sum())
    denominator = float(np.linalg.norm(q) * np.linalg.norm(p))
    q_peak, p_peak = int(np.argmax(q)), int(np.argmax(p))
    teacher_order = np.argsort(-q, kind="stable")
    student_order = np.argsort(-p, kind="stable")
    teacher_peak_rank = int(np.flatnonzero(student_order == q_peak)[0]) + 1
    student_peak_rank = int(np.flatnonzero(teacher_order == p_peak)[0]) + 1
    q_mean = np.sum(xy * q[:, None], axis=0)
    p_mean = np.sum(xy * p[:, None], axis=0)
    available = uniform_ce - teacher_entropy
    gain = uniform_ce - cross_entropy
    result = {
        "cross_entropy": cross_entropy,
        "teacher_entropy": teacher_entropy,
        "student_entropy": _entropy(p),
        "forward_kl": max(0.0, cross_entropy - teacher_entropy),
        "uniform_cross_entropy": uniform_ce,
        "available_information_vs_uniform": available,
        "gain_over_uniform": gain,
        "normalized_information_recovery": gain / available if available > 1e-12 else 0.0,
        "perplexity": float(math.exp(min(cross_entropy, 700.0))),
        "perplexity_reduction_vs_uniform": float(math.exp(min(gain, 700.0))),
        "jensen_shannon": js,
        "total_variation": total_variation,
        "overlap": 1.0 - total_variation,
        "squared_l2_distance": float(np.square(q - p).sum()),
        "bhattacharyya_coefficient": coefficient,
        "hellinger_distance": math.sqrt(max(0.0, 1.0 - coefficient)),
        "cosine_similarity": float(np.dot(q, p) / denominator) if denominator > 0 else 0.0,
        "student_mass_on_teacher_valid_candidates": float(p[valid].sum()),
        "student_mass_on_teacher_invalid_candidates": float(p[~valid].sum()),
        "top1_grid_agreement": float(q_peak == p_peak),
        "teacher_peak_probability": float(q[q_peak]),
        "student_peak_probability": float(p[p_peak]),
        "student_probability_at_teacher_peak": float(p[q_peak]),
        "teacher_probability_at_student_peak": float(q[p_peak]),
        "teacher_peak_rank_in_student": float(teacher_peak_rank),
        "reciprocal_rank_of_teacher_peak": 1.0 / teacher_peak_rank,
        "student_peak_rank_in_teacher": float(student_peak_rank),
        "peak_coordinate_distance": float(np.linalg.norm(xy[q_peak] - xy[p_peak])),
        "probability_mean_coordinate_distance": float(np.linalg.norm(q_mean - p_mean)),
    }
    for mass in CORE_MASSES:
        label = str(int(mass * 100))
        q_set = _credible_indices(q, mass)
        p_set = _credible_indices(p, mass)
        result[f"student_mass_on_teacher_{label}pct_set"] = float(p[q_set].sum())
        result[f"teacher_mass_on_student_{label}pct_set"] = float(q[p_set].sum())
        result[f"teacher_{label}pct_set_fraction"] = float(len(q_set) / len(q))
        result[f"student_{label}pct_set_fraction"] = float(len(p_set) / len(p))
        teacher_region = _highest_density_region(q, mass)
        student_region = _highest_density_region(p, mass)
        intersection = int(np.count_nonzero(teacher_region & student_region))
        teacher_area = int(np.count_nonzero(teacher_region))
        student_area = int(np.count_nonzero(student_region))
        union = teacher_area + student_area - intersection
        result[f"support_precision_{label}"] = float(intersection / student_area)
        result[f"support_recall_{label}"] = float(intersection / teacher_area)
        result[f"support_dice_{label}"] = float(2 * intersection / (teacher_area + student_area))
        result[f"support_iou_{label}"] = float(intersection / union)
        result[f"student_mass_in_teacher_region_{label}"] = float(p[teacher_region].sum())
        result[f"teacher_mass_in_student_region_{label}"] = float(q[student_region].sum())
        result[f"teacher_region_fraction_{label}"] = float(teacher_area / len(q))
        result[f"student_region_fraction_{label}"] = float(student_area / len(p))
        # This is the requested evaluation-time polarization.  Each field is
        # truncated to its own highest-density core and renormalized.  It does
        # not change the saved model, logits, or original probabilities.
        q_core = _polarized_core(q, teacher_region)
        p_core = _polarized_core(p, student_region)
        core_mixture = 0.5 * (q_core + p_core)
        q_core_positive = q_core > 0
        p_core_positive = p_core > 0
        core_js = 0.5 * float(
            np.sum(q_core[q_core_positive] * np.log(q_core[q_core_positive] / core_mixture[q_core_positive]))
        )
        core_js += 0.5 * float(
            np.sum(p_core[p_core_positive] * np.log(p_core[p_core_positive] / core_mixture[p_core_positive]))
        )
        core_tv = 0.5 * float(np.abs(q_core - p_core).sum())
        result[f"polarized_core_jensen_shannon_{label}"] = core_js
        result[f"polarized_core_overlap_{label}"] = 1.0 - core_tv
    for count in (10, 100):
        actual = min(count, len(q))
        q_set = set(teacher_order[:actual].tolist())
        p_set = set(student_order[:actual].tolist())
        intersection = len(q_set & p_set)
        result[f"top_{count}_overlap_fraction"] = float(intersection / actual)
        result[f"top_{count}_jaccard"] = float(intersection / len(q_set | p_set))
    return result


def _physical_metrics(
    teacher: np.ndarray,
    student: np.ndarray,
    cost: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    invalid_penalty: float,
) -> dict[str, float]:
    q, p = _normalize(teacher, "teacher probability"), _normalize(student, "student probability")
    cost = np.asarray(cost, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if cost.shape != q.shape or valid.shape != q.shape or not np.isfinite(cost[valid]).all():
        raise ValueError("physical costs must be finite on valid candidates")
    compatible = valid & (cost <= threshold)
    raw_peak = int(np.argmax(p))
    valid_peak = int(np.argmax(np.where(valid, p, -np.inf)))
    return {
        "teacher_compatible_mass": float(q[compatible].sum()),
        "student_compatible_mass": float(p[compatible].sum()),
        "student_invalid_mass": float(p[~valid].sum()),
        "student_valid_mass": float(p[valid].sum()),
        "teacher_expected_valid_cost": float(q[valid] @ cost[valid]),
        "student_expected_valid_cost_conditional": (
            float(p[valid] @ cost[valid]) / float(p[valid].sum()) if p[valid].sum() > 0 else invalid_penalty
        ),
        "student_expected_penalized_cost": float(p[valid] @ cost[valid]) + invalid_penalty * float(p[~valid].sum()),
        "student_raw_peak_is_valid": float(valid[raw_peak]),
        "student_raw_peak_is_compatible": float(compatible[raw_peak]),
        "student_valid_peak_is_compatible": float(compatible[valid_peak]),
        "student_valid_peak_physical_cost": float(cost[valid_peak]),
    }


def _coordinate_metrics(
    teacher: np.ndarray,
    student: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    truth: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    q, p = _normalize(teacher, "teacher probability"), _normalize(student, "student probability")
    xy = np.asarray(candidate, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != (2,):
        raise ValueError("Phase-I coordinate evaluation requires one true source [2]")
    distances = np.linalg.norm(xy - truth[None, :], axis=1)
    student_peak = int(np.argmax(p))
    student_valid_peak = int(np.argmax(np.where(valid, p, -np.inf)))
    teacher_peak = int(np.argmax(q))
    nearest_valid = int(np.argmin(np.where(valid, distances, np.inf)))
    result = {
        "student_peak_source_distance": float(distances[student_peak]),
        "student_valid_peak_source_distance": float(distances[student_valid_peak]),
        "teacher_peak_source_distance": float(distances[teacher_peak]),
        "nearest_valid_grid_source_distance": float(distances[nearest_valid]),
        "student_expected_source_distance": float(p @ distances),
        "teacher_expected_source_distance": float(q @ distances),
        "student_peak_is_physics_valid": float(valid[student_peak]),
    }
    for threshold in thresholds:
        label = f"{threshold:g}"
        within = distances <= threshold
        result[f"student_recall_at_{label}"] = float(distances[student_peak] <= threshold)
        result[f"student_valid_recall_at_{label}"] = float(distances[student_valid_peak] <= threshold)
        result[f"teacher_recall_at_{label}"] = float(distances[teacher_peak] <= threshold)
        result[f"student_mass_within_{label}"] = float(p[within].sum())
        result[f"teacher_mass_within_{label}"] = float(q[within].sum())
    return result


def _describe(values: list[float], scene_ids: list[int] | None = None) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    array = array[finite]
    if not len(array):
        return None
    result = {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }
    if scene_ids is not None:
        ids = np.asarray(scene_ids, dtype=np.int64)[finite]
        groups: dict[int, list[float]] = defaultdict(list)
        for scene, value in zip(ids, array):
            groups[int(scene)].append(float(value))
        result["scene_mean"] = float(np.mean([np.mean(group) for group in groups.values()]))
        result["independent_scenes"] = int(len(groups))
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    keys = sorted(set.intersection(*(set(row["metrics"]) for row in rows)))
    scenes = [int(row["scene_id"]) for row in rows]
    return {key: _describe([float(row["metrics"][key]) for row in rows], scenes) for key in keys}


def _strata(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _summarize(group) for name, group in sorted(groups.items())}


def _cluster_bootstrap(
    rows: list[dict[str, Any]], keys: list[str], repetitions: int, seed: int
) -> dict[str, Any] | None:
    """Resample independent scenes while preserving all windows in each scene."""
    if not rows or repetitions < 1:
        return None
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["scene_id"])].append(row["metrics"])
    scenes = sorted(grouped)
    if len(scenes) < 2:
        return None
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(scenes), size=(repetitions, len(scenes)), dtype=np.int32)
    result = {}
    for key in keys:
        sums = np.asarray([sum(item[key] for item in grouped[scene]) for scene in scenes], dtype=np.float64)
        counts = np.asarray([len(grouped[scene]) for scene in scenes], dtype=np.float64)
        estimates = sums[selections].sum(axis=1) / counts[selections].sum(axis=1)
        result[key] = {
            "estimate": float(np.mean([row["metrics"][key] for row in rows])),
            "lower": float(np.quantile(estimates, 0.025)),
            "upper": float(np.quantile(estimates, 0.975)),
            "repetitions": int(repetitions),
            "independent_scene_clusters": int(len(scenes)),
        }
    return result


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    ci = report["scene_cluster_bootstrap_95ci"] or {}

    def mean(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value['mean']:.6f}"

    def interval(key: str) -> str:
        value = ci.get(key)
        return "n/a" if value is None else f"[{value['lower']:.6f}, {value['upper']:.6f}]"

    lines = [
        "# Phase1 100k 综合评价",
        "",
        f"- Checkpoint epoch：{report['checkpoint_epoch']}",
        f"- 测试视窗：{report['coverage']['observations_evaluated']:,} / {report['coverage']['test_observations_total']:,}",
        f"- 独立测试场景：{report['coverage']['independent_test_scenes']:,}",
        "- 真源坐标只用于评价，没有进入预测。",
        "",
        "## 主要结果",
        "",
        "| 指标 | 均值 | 场景 bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    primary = [
        ("cross_entropy", "Cross entropy"),
        ("forward_kl", "Forward KL"),
        ("normalized_information_recovery", "Information recovery"),
        ("jensen_shannon", "Jensen–Shannon"),
        ("overlap", "Probability overlap"),
        ("support_dice_75", "Core Dice@75"),
        ("support_recall_75", "Core Recall@75"),
        ("support_precision_75", "Core Precision@75"),
        ("support_iou_75", "Core IoU@75"),
        ("polarized_core_overlap_75", "Polarized probability overlap@75"),
        ("polarized_core_jensen_shannon_75", "Polarized Jensen–Shannon@75"),
        ("student_mass_in_teacher_region_75", "Student mass in teacher HDR@75"),
        ("top_100_overlap_fraction", "Top-100 candidate overlap"),
        ("student_mass_on_teacher_invalid_candidates", "Invalid-candidate mass"),
        ("student_peak_source_distance", "Raw peak source distance"),
        ("student_valid_peak_source_distance", "Valid-filtered peak source distance"),
        ("student_recall_at_32", "Raw peak Recall@32"),
    ]
    for key, label in primary:
        lines.append(f"| {label} | {mean(key)} | {interval(key)} |")
    lines += [
        "",
        "## 判读顺序",
        "",
        "1. Core Dice/Recall/Precision/IoU@75 是主结论：它们只比较教师和学生各自承载 75% 概率质量的最小高密度区域。",
        "2. Polarized overlap/JS@75 先剔除两个分布的低概率尾部，再分别归一化后比较核心区域内的概率形状。",
        "3. HDR@50/@90/@95 仍写入 JSON，用于检查结论是否对核心质量阈值敏感。",
        "4. CE、KL、原始 JS、overlap 和 information recovery 保留为全分布校准审计，不作为核心区域形状的主指标。",
        "5. Invalid mass 与物理代价判断学生是否把概率放到了不可能位置。",
        "6. 这个流程只做后处理评价，不修改 checkpoint，也不需要重新训练。",
        "",
    ]
    return "\n".join(lines)


def evaluate(
    output_root: Path,
    variant: str,
    seed: int,
    run_label: str | None,
    bootstrap_repetitions: int,
    max_observations: int | None,
    compatibility_threshold: float | None,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    protocol = _read_json(root / "protocol.json")
    config = protocol["config"]
    variant_root = root / variant
    seed_root = variant_root / (
        f"seed_{seed}" if run_label is None else f"seed_{seed}_{run_label}"
    )
    checkpoint_path = seed_root / "best.pt"
    prediction_root = seed_root / "predictions"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint is missing: {checkpoint_path}")
    if not prediction_root.is_dir():
        raise FileNotFoundError(
            f"saved test predictions are missing: {prediction_root}; run evaluate_phase1_100k.sh full first"
        )
    training = _read_json(seed_root / "training.json")
    best_row = min(training["history"], key=lambda row: float(row["validation_ce"]))
    records = _read_json(variant_root / "records.json")["test"]
    if max_observations is not None:
        records = records[:max_observations]
    raw_data_root = Path(config["data_root"]).expanduser()
    data_root = raw_data_root if raw_data_root.is_absolute() else PROJECT_ROOT / raw_data_root
    dataset = Phase1Dataset(data_root.resolve())
    threshold = float(
        config["evaluation"]["compatibility_threshold"]
        if compatibility_threshold is None
        else compatibility_threshold
    )
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("compatibility threshold must be finite and nonnegative")
    thresholds = (8.0, 16.0, 32.0)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, relative in enumerate(records, 1):
        teacher = load_teacher_record(variant_root / relative)
        scene, view = int(teacher["scene_id"]), int(teacher["view_id"])
        prediction_path = prediction_root / f"s{scene}_v{view}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"saved prediction is missing: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = {key: archive[key].copy() for key in archive.files}
        candidate = np.asarray(teacher["candidate_xy"], dtype=np.float64)
        if (
            int(prediction["scene_id"]) != scene
            or int(prediction["view_id"]) != view
            or not np.array_equal(np.asarray(prediction["candidate_xy"]), candidate)
            or str(np.asarray(prediction["config_id"]).item()) != str(teacher["config_id"])
        ):
            raise ValueError(f"teacher/prediction alignment failed for scene={scene}, view={view}")
        q = _normalize(teacher["teacher_prob"], "teacher probability")
        p = _normalize(prediction["student_prob"], "student probability")
        valid = np.asarray(teacher["valid"], dtype=bool)
        if "support" in prediction:
            support = np.asarray(prediction["support"], dtype=bool)
        else:
            # Older saved prediction files predate the explicit support field.
            # Reconstruct it only from the frozen protocol, never by guessing
            # from zero probabilities (which would change the metric domain).
            policy = config["training"].get("normalization_support", "geometry")
            support = np.ones_like(valid) if policy == "world" else valid.copy()
        distribution = _distribution_metrics(q, p, candidate, valid, support)
        physics = teacher.get("metadata", {}).get("settings", {}).get("physics", {})
        invalid_penalty = float(physics.get("alpha", 1.0)) + 2.0 * float(physics.get("beta", 0.0))
        physical = _physical_metrics(
            q, p, teacher["physical_cost"], valid, threshold, invalid_penalty
        )
        scene_truth = np.asarray(dataset.get_scene(scene)["drivers"], dtype=np.float64)
        if scene_truth.shape != (1, 2):
            raise ValueError(f"Phase-I scene {scene} does not contain exactly one source")
        coordinate = _coordinate_metrics(q, p, candidate, valid, scene_truth[0], thresholds)
        rows.append(
            {
                "scene_id": scene,
                "view_id": view,
                "window_size": int(teacher["window"][2]),
                "metrics": {**distribution, **physical, **coordinate},
            }
        )
        if position == 1 or position % 250 == 0 or position == len(records):
            print(f"comprehensive evaluation {position}/{len(records)} observations", flush=True)

    ci_keys = [
        "cross_entropy",
        "forward_kl",
        "normalized_information_recovery",
        "jensen_shannon",
        "overlap",
        "support_dice_75",
        "support_recall_75",
        "support_precision_75",
        "support_iou_75",
        "polarized_core_overlap_75",
        "polarized_core_jensen_shannon_75",
        "student_mass_in_teacher_region_75",
        "teacher_mass_in_student_region_75",
        "top_100_overlap_fraction",
        "student_mass_on_teacher_invalid_candidates",
        "student_compatible_mass",
        "student_expected_penalized_cost",
        "student_peak_source_distance",
        "student_valid_peak_source_distance",
        "student_recall_at_8",
        "student_recall_at_16",
        "student_recall_at_32",
    ]
    report = {
        "schema": "phase1-comprehensive-evaluation-v2-core-polarized",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(best_row["epoch"]),
        "seed": int(seed),
        "run_label": run_label,
        "elapsed_seconds": time.perf_counter() - started,
        "truth_policy": "true source coordinates are used only for evaluation, never prediction",
        "evaluation_policy": (
            "evaluation-only polarization: independently retain and renormalize each distribution's "
            "highest-density core; saved checkpoint and raw probabilities are unchanged"
        ),
        "coverage": {
            "test_observations_total": len(_read_json(variant_root / "records.json")["test"]),
            "observations_evaluated": len(rows),
            "independent_test_scenes": len({row["scene_id"] for row in rows}),
            "missing_predictions": 0,
        },
        "metrics": _summarize(rows),
        "scene_cluster_bootstrap_95ci": _cluster_bootstrap(
            rows, ci_keys, bootstrap_repetitions, seed + 3107
        ),
        "stratification": {"by_window_size": _strata(rows, "window_size")},
        "metric_definitions": {
            "forward_kl": "KL(teacher || student) = cross entropy - teacher entropy; lower is better",
            "normalized_information_recovery": "(uniform CE - model CE) / (uniform CE - teacher entropy); higher is better",
            "overlap": "1 - total variation; higher is better",
            "credible_set_mass": "probability captured by the other distribution's smallest highest-probability credible set",
            "support_geometry": "tie-inclusive highest-density regions at equal target mass; Dice/IoU summarize overlap, Recall measures teacher coverage, Precision penalizes student-only area",
            "polarized_core": "drop probability outside each distribution's own highest-density core, renormalize retained values, then compare; the primary core mass is 75%",
            "raw_peak_coordinate": "argmax over the student's declared world support; invalid predictions remain visible",
            "valid_filtered_peak_coordinate": "argmax after evaluation-only physical validity filtering; diagnostic upper bound, not the deployed prediction",
            "source_recall": "fraction of observations whose selected peak lies within the stated Euclidean pixel distance of the true source",
        },
        "evaluation_parameters": {
            "normalization_support": config["training"].get("normalization_support", "geometry"),
            "candidate_spacing": float(config["teacher"]["spacing"]),
            "compatibility_threshold": threshold,
            "distance_thresholds": list(thresholds),
            "credible_masses": list(CORE_MASSES),
            "primary_core_mass": PRIMARY_CORE_MASS,
            "scene_bootstrap_repetitions": int(bootstrap_repetitions),
            "max_observations": max_observations,
        },
    }
    destination = seed_root / "evaluation"
    _write_json(destination / "comprehensive_metrics.json", report)
    with (destination / "comprehensive_observations.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (destination / "comprehensive_metrics.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Comprehensive JSON: {destination / 'comprehensive_metrics.json'}")
    print(f"Comprehensive Markdown: {destination / 'comprehensive_metrics.md'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", default="response")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--compatibility-threshold", type=float)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be nonnegative")
    if args.max_observations is not None and args.max_observations < 1:
        parser.error("--max-observations must be positive")
    return args


def main() -> None:
    args = parse_args()
    evaluate(
        args.output_root,
        args.variant,
        args.seed,
        args.run_label,
        args.bootstrap_repetitions,
        args.max_observations,
        args.compatibility_threshold,
    )


if __name__ == "__main__":
    main()
