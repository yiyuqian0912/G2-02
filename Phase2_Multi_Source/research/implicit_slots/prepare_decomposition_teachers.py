#!/usr/bin/env python3
"""Append teachers for new scene configurations while preserving the old book/splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
PHASE2 = HERE.parents[2]
for path in (PHASE2, HERE.parent, PHASE2 / "research/layered_residual", PHASE2 / "src",
             PHASE2 / "vendor/rind-dataset", PHASE2.parent / "Phase1_Single_Source_Single_Intensity/src"):
    sys.path.insert(0, str(path))

from dataset_update import open_dataset
from layered_residual import recover, save_run
from train_decomposition import load_config, resolve, write_json, digest


def prepare(config):
    import torch
    from rind_phase2.regions import check_geometry_backend
    # Compile once on the coordinator before any teacher threads reach the
    # JIT call graph, and validate canonical closed-boundary geometry.
    print(f"Teacher geometry preflight: {check_geometry_backend()}", flush=True)
    settings = config.get("new_teacher", {})
    dataset = open_dataset(resolve(config["data_root"]))
    output = resolve(config["teacher_experiment_root"])
    output.mkdir(parents=True, exist_ok=True)
    base_root = resolve(settings.get("base_teacher_root", "outputs/full_training_h20_v2"))
    base_manifest = json.loads((base_root / "records.json").read_text())
    if not hasattr(dataset, "scene_splits"):
        raise ValueError("prepare-new requires an installed virtual union with geometry-grouped scene_splits")
    splits = dataset.scene_splits
    # Old cached records already encode their original train/val/test roles.
    membership = {int(scene): split for split, values in splits.items() for scene in values}
    records = {name: [] for name in ("train", "validation", "test")}
    for split, rows in base_manifest["records"].items():
        for item in rows:
            row = dict(item)
            if membership.get(int(row["scene_id"])) != split:
                raise ValueError("union changed a base teacher scene's split")
            row["directory"] = str((base_root / row["directory"]).resolve())
            records[split].append(row)
    if not torch.cuda.is_available():
        raise RuntimeError("new teacher rendering requires CUDA")
    gpu_count = torch.cuda.device_count()
    requested = int(settings.get("base_scene_count", 10000))
    seed = int(settings.get("seed", 20261028))
    teacher = dict(settings.get("physics", {}))
    teacher.setdefault("spacing", 8)
    teacher.setdefault("temperature", 0.05)
    teacher.setdefault("alpha", 1.0)
    teacher.setdefault("beta", 0.0)
    teacher.setdefault("boundary_lambda", 2.0)
    teacher.setdefault("boundary_sigma", 1.0)
    teacher.setdefault("max_layers", 32)
    if int(dataset.manifest.get("new_scene_offset", requested)) != requested:
        raise ValueError("new_teacher.base_scene_count differs from union's preserved base scene count")
    sample_row = next((row for values in base_manifest["records"].values() for row in values), None)
    if sample_row is None:
        raise ValueError("base teacher manifest has no complete records")
    baseline_report = json.loads((base_root / sample_row["directory"] / "report.json").read_text())
    baseline_physics = baseline_report.get("phase1_cost", {})
    if settings.get("inherit_base_physics", True):
        for key in ("spacing", "temperature"):
            teacher[key] = baseline_report[key]
        for key in ("alpha", "beta", "boundary_lambda", "boundary_sigma"):
            if key in baseline_physics:
                teacher[key] = baseline_physics[key]
        print(f"Inherited original Phase1 teacher scoring/grid: {teacher}", flush=True)
    for key in ("spacing", "temperature"):
        if not np.isclose(float(baseline_report[key]), float(teacher[key]), rtol=0, atol=1e-12):
            raise ValueError(f"new teacher {key} must match base teacher {baseline_report[key]}")
    for key in ("alpha", "beta", "boundary_lambda", "boundary_sigma"):
        if key in baseline_physics and not np.isclose(float(baseline_physics[key]), float(teacher[key]), rtol=0, atol=1e-12):
            raise ValueError(f"new teacher {key} must match base teacher {baseline_physics[key]}")
    contract = {"dataset_manifest": dataset.manifest, "base_teacher_manifest_sha256":
                hashlib.sha256((base_root / "records.json").read_bytes()).hexdigest(),
                "selection": settings, "physics": teacher}
    signature = digest(contract)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()).get("signature") != signature:
        raise ValueError("new teacher output has a different dataset/physics/selection contract")
    write_json(protocol_path, {"signature": signature, "contract": contract, "scene_splits": splits})
    audit, stats = [], {}
    tolerance = 1e-12
    sizes = set(settings.get("window_sizes", [64, 128]))
    maximum_views = int(settings.get("max_views_per_scene", 8))

    def execute(job):
        split, scene, view, size, gpu = job
        directory = output / "teachers" / split / f"scene{scene}-view{view}"
        path = directory / "report.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get("append_contract") != signature:
                raise ValueError(f"stale teacher cache at {path}")
            if report.get("status") != "complete" or (directory / "combined.npz").is_file():
                return report, directory, True
        result = recover(
            dataset, scene, view, backend="cuda", gpu_device=f"cuda:{gpu}",
            gpu_batch_size=int(settings.get("gpu_batch_size", 4096)),
            gpu_min_batch_size=256, gpu_pixel_chunk=4096,
            gpu_pair_budget=134217728, gpu_verify_candidates=0,
            shadow_screening=True, tolerance=tolerance, **teacher)
        report = dict(result[0])
        report["append_contract"] = signature
        report["window"] = dataset.get_observation(scene, view)["window"].tolist()
        report["world_size"] = int(dataset.manifest["global_size"])
        save_run(directory, report, *result[1:], plots=False)
        return report, directory, False

    started = time.perf_counter()
    for split in records:
        targets = settings.get("record_counts", {"train": 100000, "validation": 10000, "test": 10000})
        target = targets[split]
        scenes = np.asarray([int(value) for value in splits[split] if int(value) >= requested], dtype=np.int64)
        rng = np.random.default_rng(seed + (0 if split == "train" else 1 if split == "validation" else 2))
        rng.shuffle(scenes)
        def candidates():
            for scene in scenes:
                views = [(view, size) for view, size in dataset.iter_scene_views(int(scene)) if size in sizes]
                rng.shuffle(views)
                for view, size in views[:maximum_views]:
                    yield int(scene), int(view), int(size)
        stream = iter(candidates())
        accepted = checked = failures = skipped = reused = 0
        exhausted = False
        print(f"prepare-new {split}: appended scene configurations={len(scenes)}, target completed new windows={target}, sizes={sorted(sizes)}, max views/scene={maximum_views}, GPUs={gpu_count}", flush=True)
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            while not exhausted and (target == "all" or accepted < int(target)):
                jobs = []
                desired = gpu_count if target == "all" else min(gpu_count, int(target) - accepted)
                while len(jobs) < desired:
                    try:
                        scene, view, size = next(stream)
                    except StopIteration:
                        exhausted = True
                        break
                    checked += 1
                    observation = dataset.get_observation(scene, view)
                    free = ~observation["obstacle"]
                    values = observation["response"][free]
                    if not len(values) or not np.any(values > tolerance) or np.all(values > tolerance):
                        skipped += 1
                        continue
                    jobs.append((split, scene, view, size, len(jobs)))
                for job, (report, directory, cache_hit) in zip(jobs, executor.map(execute, jobs)):
                    _, scene, view, size, _ = job
                    reused += int(cache_hit)
                    row = {"scene_id": scene, "view_id": view, "size": size,
                           "directory": str(directory.resolve()), "window": report["window"]}
                    if report["status"] == "complete":
                        records[split].append(row)
                        accepted += 1
                    else:
                        failures += 1
                    audit.append({**row, "status": report["status"], "reason": report.get("failure_reason")})
                    print(f"teacher {split} scene={scene} view={view}: {report['status']} layers={report['layers_recovered']} completed={accepted}/{target} reuse={cache_hit}", flush=True)
                stats[split] = {"new_completed": accepted, "checked": checked, "skipped": skipped,
                                "incomplete": failures, "reused": reused, "target": target}
                write_json(output / "records.json", {"records": records, "statistics": stats,
                           "counts": {key: len(value) for key, value in records.items()},
                           "append_contract": signature})
        if target != "all" and accepted < int(target):
            raise RuntimeError(f"{split}: completed only {accepted}/{target}; raise max_views_per_scene or lower target")
    write_json(output / "append_audit.json", {"statistics": stats, "windows": audit,
               "seconds": time.perf_counter() - started})
    print(f"new teacher book ready: {output / 'records.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))
