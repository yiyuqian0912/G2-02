"""Audit Phase2 teacher quality and student learning against uniform baselines."""
from __future__ import print_function

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
PHASE2_ROOT = PACKAGE_ROOT / "Phase2_Multi_Source"
PHASE1_ROOT = PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity"
for item in (PHASE1_ROOT / "src", PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset", HERE.parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _indices(length, maximum):
    if maximum <= 0 or length <= maximum:
        return list(range(length))
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _skill(cross_entropy, entropy, baseline_cross_entropy):
    available = baseline_cross_entropy - entropy
    if not math.isfinite(available) or available <= 1e-8:
        return None
    return float((baseline_cross_entropy - cross_entropy) / available)


def audit_split(root, rows, maximum, progress_every):
    selected = _indices(len(rows), maximum)
    entropy = []
    uniform_ce = []
    effective = []
    final_mae = []
    layer_cost = []
    layer_conflicts = []
    sizes = defaultdict(lambda: {"entropy": [], "uniform_ce": []})

    for position, index in enumerate(selected, 1):
        row = rows[index]
        directory = root / row["directory"]
        with np.load(directory / "combined.npz", allow_pickle=False) as archive:
            q = np.asarray(archive["probability"], dtype=np.float64)
        if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
            raise ValueError("malformed teacher probability: {}".format(directory))
        if not np.isclose(q.sum(), 1.0, atol=1e-10):
            raise ValueError("unnormalized teacher probability: {}".format(directory))
        positive = q > 0
        h = float(-np.sum(q[positive] * np.log(q[positive])))
        u = float(math.log(len(q)))
        entropy.append(h)
        uniform_ce.append(u)
        effective.append(math.exp(h))
        size = str(int(row["size"]))
        sizes[size]["entropy"].append(h)
        sizes[size]["uniform_ce"].append(u)

        report = json.loads((directory / "report.json").read_text())
        final_mae.append(float(report["final_mae"]))
        for layer in report.get("layers", []):
            cost = layer.get("minimum_physical_cost")
            if cost is not None and math.isfinite(float(cost)):
                layer_cost.append(float(cost))
            trace = layer.get("trace", {})
            layer_conflicts.append(len(trace.get("conflict", [])))

        if progress_every and position % progress_every == 0:
            print("  audited {}/{} records".format(position, len(selected)), flush=True)

    h_mean = float(np.mean(entropy)) if entropy else None
    u_mean = float(np.mean(uniform_ce)) if uniform_ce else None
    return {
        "records_available": int(len(rows)),
        "records_audited": int(len(selected)),
        "subsampled": bool(len(selected) != len(rows)),
        "teacher_entropy": _summary(entropy),
        "uniform_cross_entropy": _summary(uniform_ce),
        "uniform_information_gap_mean": None if h_mean is None else float(u_mean - h_mean),
        "effective_candidate_count": _summary(effective),
        "final_reconstruction_mae": _summary(final_mae),
        "layer_minimum_physical_cost": _summary(layer_cost),
        "layer_cost_le_0_01_fraction": (
            None if not layer_cost else float(np.mean(np.asarray(layer_cost) <= 0.01))
        ),
        "layer_conflict_regions": _summary(layer_conflicts),
        "by_window_size": {
            size: {
                "count": len(values["entropy"]),
                "teacher_entropy_mean": float(np.mean(values["entropy"])),
                "uniform_cross_entropy_mean": float(np.mean(values["uniform_ce"])),
            }
            for size, values in sorted(sizes.items(), key=lambda item: int(item[0]))
        },
    }


def checkpoint_audit(root, seed):
    import torch

    directory = root / "models" / "seed_{}".format(seed)
    result = {}
    for name in ("best.pt", "last.pt"):
        path = directory / name
        if not path.exists():
            result[name] = {"exists": False}
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        nonfinite = [
            key for key, value in checkpoint["model"].items()
            if torch.is_tensor(value) and not torch.isfinite(value).all().item()
        ]
        result[name] = {
            "exists": True,
            "epoch": int(checkpoint["epoch"]),
            "best_validation_cross_entropy": float(checkpoint["best_validation_ce"]),
            "parameters_finite": not nonfinite,
            "nonfinite_parameter_names": nonfinite,
        }
    return result


def _distribution_row(q, p, log_p, support):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    log_p = np.asarray(log_p, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if (q.shape != p.shape or log_p.shape != q.shape or support.shape != q.shape or
            not support.any() or not np.isclose(q.sum(), 1.0, atol=1e-8) or
            not np.isclose(p.sum(), 1.0, atol=1e-6)):
        raise ValueError("misaligned or unnormalized student evaluation arrays")
    positive_q = q > 0
    positive_p = p > 0
    cross_entropy = float(-np.sum(q[positive_q] * log_p[positive_q]))
    entropy = float(-np.sum(q[positive_q] * np.log(q[positive_q])))
    student_entropy = float(-np.sum(p[positive_p] * np.log(p[positive_p])))
    mixture = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q[positive_q] * np.log(q[positive_q] / mixture[positive_q])))
    js += 0.5 * float(np.sum(p[positive_p] * np.log(p[positive_p] / mixture[positive_p])))
    tv = 0.5 * float(np.abs(q - p).sum())
    uniform_ce = float(math.log(np.count_nonzero(support)))
    order = np.argsort(-q, kind="stable")
    cutoff = int(np.searchsorted(np.cumsum(q[order]), 0.9, side="left")) + 1
    teacher_90 = order[:cutoff]
    return {
        "cross_entropy": cross_entropy,
        "teacher_entropy": entropy,
        "forward_kl": float(cross_entropy - entropy),
        "jensen_shannon": js,
        "total_variation": tv,
        "distribution_overlap": float(1.0 - tv),
        "student_entropy": student_entropy,
        "entropy_mismatch": float(student_entropy - entropy),
        "uniform_cross_entropy": uniform_ce,
        "uniform_normalized_skill": _skill(cross_entropy, entropy, uniform_ce),
        "student_mass_on_teacher_90pct_set": float(p[teacher_90].sum()),
        "teacher_90pct_set_fraction": float(len(teacher_90) / np.count_nonzero(support)),
    }


def _aggregate_student_rows(rows):
    if not rows:
        return None
    keys = [key for key, value in rows[0].items()
            if isinstance(value, (int, float)) and key != "size"]
    return {key: _summary([row[key] for row in rows if row.get(key) is not None])
            for key in keys}


def _student_grade(metrics):
    if not metrics:
        return "insufficient_data"
    values = {
        "skill": metrics["uniform_normalized_skill"]["mean"],
        "js": metrics["jensen_shannon"]["mean"],
        "tv": metrics["total_variation"]["mean"],
        "mass90": metrics["student_mass_on_teacher_90pct_set"]["mean"],
    }
    if any(value is None or not math.isfinite(value) for value in values.values()):
        return "insufficient_data"
    thresholds = (
        ("excellent", 0.8, 0.05, 0.15, 0.90),
        ("good", 0.7, 0.07, 0.20, 0.85),
        ("initially_usable", 0.5, 0.10, 0.25, 0.75),
    )
    for label, skill, js, tv, mass90 in thresholds:
        if (values["skill"] >= skill and values["js"] <= js and
                values["tv"] <= tv and values["mass90"] >= mass90):
            return label
    if (values["skill"] <= 0.1 or values["js"] > 0.20 or
            values["tv"] > 0.40 or values["mass90"] < 0.60):
        return "failed"
    return "below_initial_target"


def student_distribution_audit(config, dataset, root, manifest, seed, maximum,
                               device_name, batch_size, progress_every):
    import torch
    from rind_phase1.model import student_distribution
    from rind_phase1.train import batches, load_model
    from train_combined import CombinedRecords

    checkpoint_path = root / "models" / "seed_{}".format(seed) / "best.pt"
    if not checkpoint_path.exists():
        return {"status": "not_run", "reason": "best checkpoint does not exist"}
    if device_name == "none":
        return {"status": "not_run", "reason": "student evaluation disabled"}
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return {"status": "failed", "reason": "CUDA requested but unavailable"}
    device = torch.device(device_name)
    model, checkpoint = load_model(checkpoint_path)
    model.to(device).eval()
    result = {"status": "complete", "device": str(device),
              "checkpoint_epoch": int(checkpoint["epoch"]), "splits": {}}

    with torch.inference_mode():
        for split in ("validation", "test"):
            source_rows = manifest["records"][split]
            indexes = _indices(len(source_rows), maximum)
            selected = [source_rows[index] for index in indexes]
            records = CombinedRecords(
                selected, dataset, root,
                config["training"].get("normalization_support", "geometry"),
                config["teacher"].get("tolerance", 1e-12))
            rows = []
            processed = 0
            try:
                for batch in batches(records, batch_size, rng=None, prefetch=0):
                    size = int(batch["response"].shape[-1])
                    batch = {key: value.to(device) for key, value in batch.items()}
                    energy = model(
                        batch["response"], batch["obstacle"], batch["window"],
                        batch["candidate_xy"])
                    log_probability, probability = student_distribution(
                        energy.float(), batch["support"])
                    q = batch["teacher_prob"].cpu().numpy()
                    p = probability.cpu().numpy()
                    log_p = log_probability.cpu().numpy()
                    support = batch["support"].cpu().numpy()
                    for index in range(len(q)):
                        row = {"size": size}
                        row.update(_distribution_row(q[index], p[index], log_p[index], support[index]))
                        rows.append(row)
                    processed += len(q)
                    if progress_every and processed % progress_every < len(q):
                        print("  student {}: evaluated {}/{} records".format(
                            split, processed, len(records)), flush=True)
            except Exception as error:
                result["status"] = "failed"
                result["splits"][split] = {
                    "status": "failed", "records_processed": processed,
                    "error_type": type(error).__name__, "error": str(error)}
                break
            buckets = defaultdict(list)
            for row in rows:
                buckets[str(row["size"])].append(row)
            overall = _aggregate_student_rows(rows)
            result["splits"][split] = {
                "status": "complete", "records_available": len(source_rows),
                "records_evaluated": len(rows), "subsampled": len(rows) != len(source_rows),
                "internal_grade": _student_grade(overall),
                "overall": overall,
                "by_window_size": {
                    size: {"count": len(values), "metrics": _aggregate_student_rows(values)}
                    for size, values in sorted(buckets.items(), key=lambda item: int(item[0]))
                },
            }
    return result


def training_audit(root, split_reports, seed, reported_validation_ce):
    path = root / "models" / "seed_{}".format(seed) / "training.json"
    result = {"training_json_exists": path.exists()}
    if path.exists():
        training = json.loads(path.read_text())
        history = training.get("history", [])
        if history:
            best_row = min(history, key=lambda row: row["validation_cross_entropy"])
            minimum_train = min(float(row["train_cross_entropy"]) for row in history)
            result.update({
                "completed_epochs": len(history),
                "best_epoch": int(best_row["epoch"]),
                "best_validation_cross_entropy": float(best_row["validation_cross_entropy"]),
                "minimum_train_cross_entropy": minimum_train,
            })
            train_stats = split_reports.get("train", {})
            if train_stats.get("teacher_entropy"):
                h = train_stats["teacher_entropy"]["mean"]
                u = train_stats["uniform_cross_entropy"]["mean"]
                result["minimum_train_uniform_normalized_skill"] = _skill(minimum_train, h, u)
            if reported_validation_ce is None:
                reported_validation_ce = float(best_row["validation_cross_entropy"])

    if reported_validation_ce is not None and split_reports.get("validation", {}).get("teacher_entropy"):
        h = split_reports["validation"]["teacher_entropy"]["mean"]
        u = split_reports["validation"]["uniform_cross_entropy"]["mean"]
        result["evaluated_validation_cross_entropy"] = float(reported_validation_ce)
        result["validation_uniform_normalized_skill"] = _skill(reported_validation_ce, h, u)
        result["validation_forward_kl_estimate"] = float(reported_validation_ce - h)
        result["validation_gain_over_uniform_ce"] = float(u - reported_validation_ce)
    return result


def coverage_audit(manifest):
    if manifest.get("split_statistics"):
        result = {}
        for split, stats in manifest["split_statistics"].items():
            eligible = int(stats["eligible_windows_seen"])
            complete = int(stats["completed_training_records"])
            result[split] = dict(stats)
            result[split]["teacher_completion_fraction_of_eligible"] = (
                None if eligible == 0 else float(complete / eligible)
            )
        return result

    result = {}
    audit = manifest.get("audit", [])
    for split in ("train", "validation", "test"):
        rows = [row for row in audit if row.get("split") == split]
        statuses = Counter(row.get("status") for row in rows)
        complete = len(manifest.get("records", {}).get(split, []))
        incomplete = statuses.get("incomplete", 0)
        eligible = complete + incomplete
        result[split] = {
            "raw_windows_seen": len(rows),
            "eligible_windows_seen": eligible,
            "completed_training_records": complete,
            "incomplete_teachers": incomplete,
            "teacher_completion_fraction_of_eligible": (
                None if eligible == 0 else float(complete / eligible)
            ),
            "status_counts": dict(statuses),
            "note": "derived from legacy records.json without split_statistics",
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records-per-split", type=int, default=0,
                        help="0 audits every record; positive values take a deterministic subsample")
    parser.add_argument("--reported-validation-ce", type=float)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--student-device", choices=("auto", "cpu", "cuda", "none"),
                        default="auto")
    parser.add_argument("--student-batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.max_records_per_split < 0 or args.progress_every < 0 or
            args.student_batch_size < 1):
        parser.error("record/progress limits must be nonnegative and student batch size must be positive")

    from train_combined import context, read_config
    config = read_config(args.config.resolve())
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = PHASE2_ROOT / root
    manifest_path = root / "records.json"
    if not manifest_path.exists():
        raise FileNotFoundError("teacher manifest does not exist: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text())

    print("Experiment:", root)
    print("Record counts:", manifest["counts"])
    split_reports = {}
    for split in ("train", "validation", "test"):
        print("Auditing {}...".format(split), flush=True)
        split_reports[split] = audit_split(
            root, manifest["records"][split], args.max_records_per_split, args.progress_every
        )

    student = {"status": "not_run", "reason": "student setup not reached"}
    if (root / "models" / "seed_{}".format(args.seed) / "best.pt").exists():
        print("Evaluating student best.pt on held-out records...", flush=True)
        dataset, context_root = context(config)
        if context_root.resolve() != root.resolve():
            raise ValueError("config output root changed during student setup")
        student = student_distribution_audit(
            config, dataset, root, manifest, args.seed, args.max_records_per_split,
            args.student_device, args.student_batch_size, args.progress_every)

    report = {
        "experiment_root": str(root.resolve()),
        "record_counts": manifest["counts"],
        "teacher_coverage": coverage_audit(manifest),
        "teacher_distribution_and_physics": split_reports,
        "training": training_audit(
            root, split_reports, args.seed, args.reported_validation_ce
        ),
        "checkpoints": checkpoint_audit(root, args.seed),
        "student_distribution_evaluation": student,
        "interpretation": {
            "uniform_normalized_skill": "1 matches the teacher, 0 equals uniform, below 0 is worse than uniform",
            "formal_success_requires": "finite completed training, held-out distribution metrics, strong baselines, and teacher coverage",
            "current_target": "equal-weight marginal mixture over recovered effective-contribution locations; not a joint multi-source posterior",
        },
    }
    output = args.output or root / "diagnostics" / "teacher_training_audit.json"
    _json(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
