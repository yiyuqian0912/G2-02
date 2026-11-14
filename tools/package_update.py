#!/usr/bin/env python3
"""Build a source-only server package from canonical repository paths."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "RIND_Phase2_Training_Minimal"
EXTENSIONS = {".py", ".sh", ".json", ".toml", ".lock", ".md", ".txt", ".html", ".css", ".js", ".png"}
EXCLUDE = {".git", ".venv", ".tools", ".python", "__pycache__", ".pytest_cache", "data", "outputs", "runtime_cache", "dist", ".local", "pilot"}


def source_files(root=ROOT):
    result = set()
    for name in ("README.md", "README.zh-CN.md", "setup.sh", "requirements.txt"):
        result.add(root / name)
    result.update(root.glob("*.sh"))
    for directory in (root / "docs", root / "tools"):
        result.update(p for p in directory.rglob("*") if p.is_file())
    for phase in ("Phase1_Single_Source_Single_Intensity", "Phase2_Multi_Source"):
        base = root / phase
        result.update(p for p in base.iterdir() if p.is_file())
        for directory in ("src", "configs", "scripts", "tests", "docs", "vendor", "part_e", "student_baseline", "research", "tools", "dataset_update"):
            if (base / directory).is_dir():
                result.update(p for p in (base / directory).rglob("*") if p.is_file())
    selected = []
    for path in result:
        relative = path.relative_to(root)
        if (path.is_symlink() or path.suffix not in EXTENSIONS or
                any(part in EXCLUDE for part in relative.parts[:-1]) or
                any(part.startswith("run_") and part != "run_scripts" for part in relative.parts[:-1]) or
                path.name.endswith((".tar.gz", ".zip"))):
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"unexpected large source file: {relative}")
        selected.append(path)
    return sorted(selected)


def build(output, root=ROOT):
    output = Path(output).expanduser().resolve()
    files = source_files(root)
    records = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    required = ("run_single_source_rl.sh","run_single_source_rl_tmux.sh",
                "Phase1_Single_Source_Single_Intensity/configs/active_rl_frontier16_minimum.json",
                "Phase1_Single_Source_Single_Intensity/configs/active_rl_frontier16_minimum_full.json",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/scene_schedule.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/runtime.py",
                "Phase1_Single_Source_Single_Intensity/tests/test_rl_runtime.py",
                "Phase1_Single_Source_Single_Intensity/tests/test_rl_scene_schedule.py",
                "Phase1_Single_Source_Single_Intensity/docs/FRONTIER16_MINIMUM_RL.zh-CN.md",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/frontier.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/trajectory_view.py",
                "Phase1_Single_Source_Single_Intensity/tests/test_frontier_rl.py",
                "Phase1_Single_Source_Single_Intensity/configs/active_rl.json",
                "Phase1_Single_Source_Single_Intensity/docs/SINGLE_SOURCE_ACTIVE_RL.zh-CN.md",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/core.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/experiment.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/locator.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/prepare.py",
                "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/deploy.py",
                "Phase1_Single_Source_Single_Intensity/tests/test_active_rl.py","Phase2_Multi_Source/research/implicit_slots/gpu_mask_inputs.py",
                "Phase2_Multi_Source/research/implicit_slots/pairwise_ops.py",
                "Phase2_Multi_Source/research/implicit_slots/test_pairwise_memory.py",
                "Phase2_Multi_Source/research/implicit_slots/test_effective_assignment.py",
                "Phase2_Multi_Source/docs/V6_RUNTIME_AUDIT.zh-CN.md",
                "Phase2_Multi_Source/research/implicit_slots/test_gpu_mask_inputs.py",
                "Phase2_Multi_Source/research/implicit_slots/test_sequential_fast.py",
                "Phase2_Multi_Source/docs/V6_BEST16_RUN.zh-CN.md",
                "Phase2_Multi_Source/research/implicit_slots/model_comparison.py",
                "Phase2_Multi_Source/research/implicit_slots/test_model_comparison.py","Phase2_Multi_Source/research/implicit_slots/test_optimizer_normalization.py","run_implicit_masks_tmux.sh",
                "Phase2_Multi_Source/research/implicit_slots/optimization_batches.py","run_sequential_masks_tmux.sh",
                "Phase2_Multi_Source/configs/masks_sequential.json",
                "Phase2_Multi_Source/research/implicit_slots/sequential_count_shape.py",
                "Phase2_Multi_Source/research/implicit_slots/sequential_objective.py",
                "Phase2_Multi_Source/research/implicit_slots/configure_sequential.py",
                "Phase2_Multi_Source/research/implicit_slots/evaluate_sequential.py",
                "Phase2_Multi_Source/research/implicit_slots/test_sequential.py",
                "Phase2_Multi_Source/docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md","run_failure_diagnostics_tmux.sh",
                "Phase2_Multi_Source/docs/FAILURE_DIAGNOSTICS.zh-CN.md",
                "Phase2_Multi_Source/research/implicit_slots/failure_diagnostics.py",
                "Phase2_Multi_Source/research/implicit_slots/evaluation_snapshot.py",
                "Phase2_Multi_Source/research/implicit_slots/test_failure_diagnostics.py",
"Phase2_Multi_Source/research/implicit_slots/gpu_runtime.py",
                "Phase2_Multi_Source/research/implicit_slots/test_gpu_runtime.py",
"run_disentangled_masks_tmux.sh",
                "Phase2_Multi_Source/configs/masks_disentangled.json",
                "Phase2_Multi_Source/docs/DISENTANGLED_TRAINING.zh-CN.md",
                "Phase2_Multi_Source/research/implicit_slots/disentangled_model.py",
                "Phase2_Multi_Source/research/implicit_slots/disentangled_objective.py",
                "Phase2_Multi_Source/research/implicit_slots/inventory_selection.py",
                "Phase2_Multi_Source/research/implicit_slots/configure_disentangled.py",
                "Phase2_Multi_Source/research/implicit_slots/test_disentangled.py",
                "Phase2_Multi_Source/research/implicit_slots/test_inventory_training.py",
"Phase2_Multi_Source/research/implicit_slots/inventory_complex.py",
                "Phase2_Multi_Source/research/implicit_slots/test_complex_inventory.py",
                "Phase2_Multi_Source/docs/COMPLEX_WINDOW_INVENTORY.zh-CN.md",
                "Phase2_Multi_Source/configs/masks_count_first_train_template.json",
                "run_count_masks.sh", "run_count_masks_tmux.sh",
                "Phase2_Multi_Source/configs/masks_count_first.json",
                "Phase2_Multi_Source/docs/COUNT_FIRST_TRAINING.zh-CN.md",
                "Phase2_Multi_Source/research/implicit_slots/count_objective.py",
                "Phase2_Multi_Source/research/implicit_slots/expand_complex_masks.py",
                "Phase2_Multi_Source/research/implicit_slots/test_count_first.py",
                "run_active_observation.sh",
                "Phase2_Multi_Source/docs/ACTIVE_OBSERVATION.zh-CN.md",
                "Phase2_Multi_Source/research/active_observation/observation_env.py",
                "Phase2_Multi_Source/research/active_observation/active_policies.py",
                "Phase2_Multi_Source/research/active_observation/demo.py",
                "Phase2_Multi_Source/research/active_observation/test_observation_env.py",
                "setup.sh", "requirements.txt", "run_masks_tmux.sh", "run_masks_expanded_tmux.sh",
                "run_slot_search_tmux.sh", "Phase2_Multi_Source/configs/slot_search.json",
                "run_refined_masks_tmux.sh", "run_refined_masks.sh",
                "run_balanced_masks_tmux.sh", "run_balanced_masks.sh",
                "Phase2_Multi_Source/configs/masks_balanced_finetune.json",
                "Phase2_Multi_Source/research/implicit_slots/balanced_sampling.py",
                "Phase2_Multi_Source/research/implicit_slots/extend_evaluation.py",
                "Phase2_Multi_Source/configs/masks_boundary_refined.json",
                "Phase2_Multi_Source/research/implicit_slots/boundary_refined_model.py",
                "Phase2_Multi_Source/research/implicit_slots/reuse_mask_cache.py",
                "Phase2_Multi_Source/research/implicit_slots/architecture_search.py",
                "Phase2_Multi_Source/configs/masks_expanded.json", "Phase2_Multi_Source/research/implicit_slots/expand_masks.py",
                "Phase1_Single_Source_Single_Intensity/pyproject.toml", "Phase1_Single_Source_Single_Intensity/part_e/evaluate.py")
    for name in required:
        if name not in records:
            raise ValueError(f"package source missing: {name}")
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = json.dumps({"schema": "rind-source-package-v1", "commit": commit,
                           "working_tree_dirty": dirty, "files_sha256": records,
                           "raw_data_included": False, "checkpoints_included": False}, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable gzip header and tar metadata; source hashes describe dirty builds.
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for path in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(f"{PREFIX}/{path.relative_to(root)}")
                info.size, info.mode, info.mtime = len(data), 0o755 if path.suffix == ".sh" else 0o644, 0
                tar.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo(f"{PREFIX}/PACKAGE_MANIFEST.json")
            info.size, info.mode = len(manifest), 0o644
            tar.addfile(info, io.BytesIO(manifest))
    with tarfile.open(temporary) as tar:
        for item in tar.getmembers():
            relative = Path(item.name).relative_to(PREFIX)
            if item.issym() or ".." in relative.parts or any(p in EXCLUDE for p in relative.parts):
                raise ValueError(f"unsafe package member: {item.name}")
    temporary.replace(output)
    print(f"Source files: {len(files)}; data/caches/checkpoints/environments excluded")
    print(f"Package: {output}")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/RIND_Phase2_Update.tar.gz")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
