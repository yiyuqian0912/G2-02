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
    required = ("setup.sh", "requirements.txt", "run_masks_tmux.sh", "run_masks_expanded_tmux.sh",
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
