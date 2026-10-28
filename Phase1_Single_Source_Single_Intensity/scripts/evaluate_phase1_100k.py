#!/usr/bin/env python3
"""Audit the current 100k experiment without rebuilding teacher targets.

The report separates three quantities on the same discrete candidate support:

* teacher entropy H(q), the irreducible ambiguity in the physical target;
* uniform cross entropy, the no-information baseline;
* student cross entropy and the implied KL(q || p).

When ``test_metrics.json`` exists, the script also imports the locked test-set
metrics produced by ``rind_phase1.experiments evaluation``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_OUTPUT = Path(
    "outputs/experiments/phase1_100k_multiscale_h20_memmap"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("statistics require nonempty finite values")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def scene_mean(rows: list[dict[str, Any]], key: str) -> float:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[int(row["scene_id"])].append(finite_float(row[key], key))
    if not groups:
        raise ValueError("cannot compute a scene mean for an empty split")
    return float(np.mean([np.mean(values) for values in groups.values()]))


def support_count(row: dict[str, Any], policy: str) -> int:
    if policy == "world":
        value = row["candidate_count"]
    elif policy == "geometry":
        value = row["valid_candidates"]
    else:
        raise ValueError(f"unknown normalization support: {policy}")
    count = int(value)
    if count < 1:
        raise ValueError("each observation must have nonempty student support")
    return count


def split_statistics(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty split")
    entropies = np.asarray(
        [finite_float(row["teacher_entropy"], "teacher_entropy") for row in rows],
        dtype=np.float64,
    )
    supports = np.asarray([support_count(row, policy) for row in rows], dtype=np.int64)
    uniform_ce = np.log(supports.astype(np.float64))
    maximum_gain = uniform_ce - entropies
    if np.any(maximum_gain < -1e-6):
        raise ValueError("teacher entropy exceeds the declared uniform support entropy")
    return {
        "observations": len(rows),
        "scenes": len({int(row["scene_id"]) for row in rows}),
        "normalization_support": policy,
        "teacher_entropy_nats": quantiles(entropies),
        "teacher_entropy_scene_mean_nats": scene_mean(rows, "teacher_entropy"),
        "teacher_effective_candidates": quantiles(np.exp(entropies)),
        "exp_mean_teacher_entropy": float(math.exp(float(entropies.mean()))),
        "student_support_candidates": quantiles(supports),
        "uniform_cross_entropy_nats": quantiles(uniform_ce),
        "maximum_recoverable_information_nats": quantiles(maximum_gain),
    }


def checkpoint_statistics(
    training: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    history = training.get("history", [])
    if not history:
        raise ValueError("training.json contains no completed epochs")
    best_row = min(history, key=lambda row: float(row["validation_ce"]))
    last_row = history[-1]
    entropy = validation["teacher_entropy_nats"]["mean"]
    uniform_ce = validation["uniform_cross_entropy_nats"]["mean"]

    def assess(row: dict[str, Any]) -> dict[str, Any]:
        ce = finite_float(row["validation_ce"], "validation_ce")
        kl = ce - entropy
        capacity = uniform_ce - entropy
        recovery = (uniform_ce - ce) / capacity if capacity > 0 else None
        return {
            "epoch": int(row["epoch"]),
            "train_cross_entropy_nats": finite_float(row["train_ce"], "train_ce"),
            "validation_cross_entropy_nats": ce,
            "validation_forward_kl_estimate_nats": kl,
            "validation_gain_over_uniform_nats": uniform_ce - ce,
            "validation_relative_ce_reduction": (uniform_ce - ce) / uniform_ce,
            "validation_information_recovery": recovery,
            "probability_concentration_over_uniform": math.exp(uniform_ce - ce),
            "student_cross_entropy_effective_candidates": math.exp(ce),
            "comparison_identity": "mean CE = mean H(q) + mean KL(q||p)",
        }

    return {
        "completed_epochs": len(history),
        "configured_or_last_known_best_validation_ce": finite_float(
            training.get("best_validation_ce", best_row["validation_ce"]),
            "best_validation_ce",
        ),
        "best": assess(best_row),
        "latest": assess(last_row),
        "note": (
            "Teacher statistics use one equal weight per prepared observation. "
            "Distributed validation may repeat at most a small tail per size/count bucket; "
            "therefore checkpoint CE minus this entropy is an estimate differing only by that tail weighting."
        ),
    }


def metric_mean(block: dict[str, Any], key: str) -> float | None:
    metric = block.get("metrics", {}).get(key)
    if not metric:
        return None
    value = metric.get("scene_mean", metric.get("observation_mean"))
    return None if value is None else float(value)


def locked_test_statistics(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"]
    baseline = report["uniform_baseline"]
    teacher_entropy = metric_mean(aggregate, "teacher_entropy")
    forward_kl = metric_mean(aggregate, "forward_kl")
    uniform_kl = metric_mean(baseline, "forward_kl")
    test_ce = (
        teacher_entropy + forward_kl
        if teacher_entropy is not None and forward_kl is not None
        else None
    )
    uniform_ce = (
        teacher_entropy + uniform_kl
        if teacher_entropy is not None and uniform_kl is not None
        else None
    )
    recovery = (
        1.0 - forward_kl / uniform_kl
        if forward_kl is not None and uniform_kl not in (None, 0.0)
        else None
    )
    selected = {
        key: metric_mean(aggregate, key)
        for key in (
            "forward_kl",
            "jensen_shannon",
            "l1_distance",
            "teacher_entropy",
            "student_entropy",
            "student_compatible_mass",
            "teacher_compatible_mass",
            "student_invalid_mass",
            "student_expected_penalized_cost",
            "student_top1_physical_cost",
            "student_top1_compatible",
        )
    }
    selected = {key: value for key, value in selected.items() if value is not None}
    by_size = {}
    for size, block in report.get("by_window_size", {}).items():
        h = metric_mean(block, "teacher_entropy")
        kl = metric_mean(block, "forward_kl")
        by_size[size] = {
            "observations": block.get("observations"),
            "teacher_entropy": h,
            "forward_kl": kl,
            "cross_entropy": h + kl if h is not None and kl is not None else None,
            "student_compatible_mass": metric_mean(block, "student_compatible_mass"),
            "student_invalid_mass": metric_mean(block, "student_invalid_mass"),
        }
    return {
        "available": True,
        "best_epoch": int(report["best_epoch"]),
        "observations": aggregate.get("observations"),
        "scenes": aggregate.get("scenes"),
        "scene_balanced_test_cross_entropy_nats": test_ce,
        "scene_balanced_uniform_cross_entropy_nats": uniform_ce,
        "scene_balanced_information_recovery": recovery,
        "metrics": selected,
        "by_window_size": by_size,
        "source": "test_metrics.json; test set remains locked until training completes",
    }


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    val = report["teacher_splits"].get("validation")
    checkpoint = report.get("checkpoint")
    lines = [
        "# Phase1 100k 当前模型评估",
        "",
        f"- 实验目录：`{report['output_root']}`",
        f"- 教师版本：`{report['variant']}`",
        f"- 学生归一化范围：`{report['normalization_support']}`",
        "- 对数底：自然对数，单位为 nats。",
        "",
    ]
    if val:
        lines += [
            "## 验证集教师难度",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 视窗数 | {fmt(val['observations'])} |",
            f"| 独立场景数 | {fmt(val['scenes'])} |",
            f"| 平均教师熵 H(q) | {fmt(val['teacher_entropy_nats']['mean'])} |",
            f"| 教师熵中位数 | {fmt(val['teacher_entropy_nats']['median'])} |",
            f"| exp(mean H(q)) | {fmt(val['exp_mean_teacher_entropy'])} |",
            f"| 平均均匀基线 CE | {fmt(val['uniform_cross_entropy_nats']['mean'])} |",
            f"| 最大可恢复信息 | {fmt(val['maximum_recoverable_information_nats']['mean'])} |",
            "",
        ]
    if checkpoint:
        best = checkpoint["best"]
        latest = checkpoint["latest"]
        lines += [
            "## 训练状态",
            "",
            "| 指标 | 最佳 epoch | 最新 epoch |",
            "|---|---:|---:|",
            f"| Epoch | {best['epoch']} | {latest['epoch']} |",
            f"| Validation CE | {fmt(best['validation_cross_entropy_nats'])} | {fmt(latest['validation_cross_entropy_nats'])} |",
            f"| KL(q‖p) 估计 | {fmt(best['validation_forward_kl_estimate_nats'])} | {fmt(latest['validation_forward_kl_estimate_nats'])} |",
            f"| 相对均匀 CE 降幅 | {fmt(best['validation_relative_ce_reduction'])} | {fmt(latest['validation_relative_ce_reduction'])} |",
            f"| 信息恢复率 | {fmt(best['validation_information_recovery'])} | {fmt(latest['validation_information_recovery'])} |",
            f"| 概率集中倍数 | {fmt(best['probability_concentration_over_uniform'])} | {fmt(latest['probability_concentration_over_uniform'])} |",
            "",
            "关系：`CE(q,p) = H(q) + KL(q||p)`。信息恢复率为 "
            "`(CE_uniform - CE_student) / (CE_uniform - H(q))`。",
            "",
        ]
    test = report["locked_test"]
    if test.get("available"):
        lines += [
            "## 锁定测试集",
            "",
            f"- 最佳 epoch：{test['best_epoch']}；视窗：{test['observations']}；场景：{test['scenes']}。",
            f"- Test CE：{fmt(test['scene_balanced_test_cross_entropy_nats'])}。",
            f"- Test information recovery：{fmt(test['scene_balanced_information_recovery'])}。",
            "",
            "| 指标（scene-balanced） | 数值 |",
            "|---|---:|",
        ]
        lines += [f"| {key} | {fmt(value)} |" for key, value in test["metrics"].items()]
        lines.append("")
    else:
        lines += [
            "## 锁定测试集",
            "",
            "尚未生成 `test_metrics.json`。训练完成后运行完整评估命令；当前报告没有读取测试集。",
            "",
        ]
    lines += [
        "## 解释限制",
        "",
        "- 教师熵低表示物理教师目标更集中，不等同于学生表现好。",
        "- KL 越接近 0 越好；信息恢复率越接近 1 越好。",
        "- 训练集教师统计只覆盖 Phase1 基础记录；追加的 Phase2 残差层不写入该 `teachers.json`。",
        "- 最终模型结论应采用锁定测试集，而不是用验证集最佳值代替测试结果。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", default="response")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label")
    parser.add_argument(
        "--require-test",
        action="store_true",
        help="Fail unless the locked test evaluation has already produced test_metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    variant_root = root / args.variant
    protocol = read_json(root / "protocol.json")
    policy = protocol["config"]["training"].get("normalization_support", "geometry")
    teacher_index = read_json(variant_root / "teachers.json")
    rows = teacher_index["records"]
    splits = {
        split: split_statistics([row for row in rows if row["split"] == split], policy)
        for split in ("train", "validation", "test")
        if any(row["split"] == split for row in rows)
    }

    seed_root = variant_root / (
        f"seed_{args.seed}" if args.run_label is None else f"seed_{args.seed}_{args.run_label}"
    )
    training_path = seed_root / "training.json"
    checkpoint = None
    if training_path.is_file():
        if "validation" not in splits:
            raise ValueError("validation teacher statistics are missing")
        checkpoint = checkpoint_statistics(read_json(training_path), splits["validation"])

    test_path = seed_root / "test_metrics.json"
    if args.require_test and not test_path.is_file():
        raise FileNotFoundError(
            f"locked test metrics are missing: {test_path}; finish training and run full evaluation"
        )
    locked_test = (
        locked_test_statistics(read_json(test_path))
        if test_path.is_file()
        else {"available": False}
    )
    report = {
        "schema_version": 1,
        "output_root": str(root),
        "variant": args.variant,
        "seed": args.seed,
        "run_label": args.run_label,
        "normalization_support": policy,
        "teacher_temperature": protocol["config"]["teacher"]["temperature"],
        "teacher_splits": splits,
        "checkpoint": checkpoint,
        "locked_test": locked_test,
    }
    report_dir = seed_root / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "current_evaluation.json"
    markdown_path = report_dir / "current_evaluation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown(report), encoding="utf-8")

    print(markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
