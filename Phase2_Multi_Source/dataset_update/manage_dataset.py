#!/usr/bin/env python3
"""Install the new RIND archive and build a split-safe virtual appended release."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import numpy as np

try:
    from .dataset_union import COMMON_ARRAYS, ComponentDataset, open_dataset, validate_manifest
except ImportError:
    from dataset_union import COMMON_ARRAYS, ComponentDataset, open_dataset, validate_manifest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _required_files(manifest):
    reference = ("reference_sources", "valid_origin_counts") if manifest.get(
        "candidate_layout") == "scene-source-sets" else (
            "view_candidates", "candidate_counts", "candidate_numbers")
    return ["manifest.json"] + [f"{name}.npy" for name in (*COMMON_ARRAYS, *reference)]


def validate_installed(root: Path) -> dict:
    dataset = ComponentDataset(root)
    m, n, w = dataset.manifest, dataset.num_scenes, dataset.global_size
    k, v = int(m["max_drivers"]), int(m["view_capacity"])
    o = dataset.obstacle_types.shape[1]
    b = (w * w + 7) // 8
    specs = {
        "drivers": ("float64", (n, k, 2)), "driver_counts": ("uint8", (n,)),
        "driver_strengths": ("float64", (n, k)), "obstacle_counts": ("uint8", (n,)),
        "obstacle_types": ("uint8", (n, o)), "obstacle_params": ("float64", (n, o, 32)),
        "obstacle_group_ids": ("uint16", (n, o)), "obstacle_bits": ("uint8", (n, b)),
        "visibility_bits": ("uint8", (n, k, b)), "local_views": ("uint16", (n, v, 3)),
        "view_counts": ("uint16", (n,)),
    }
    if m.get("candidate_layout") == "scene-source-sets" and not dataset.is_strength_overlay:
        specs.update({
            "reference_sources": ("float64", (n, int(m["candidate_capacity"]), k, 2)),
            "valid_origin_counts": ("uint32", (n, len(m["window_sizes"]))),
        })
    arrays = {}
    for name, (dtype, shape) in specs.items():
        value = np.load(dataset.array_root(name) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid array {name}: {value.shape}/{value.dtype}, expected {shape}/{dtype}")
        arrays[name] = value
    sizes = Counter()
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        counts = np.asarray(dataset.driver_counts[start:stop])
        active = np.arange(k)[None] < counts[:, None]
        strength = np.asarray(dataset.driver_strengths[start:stop])
        if (np.any(counts < 1) or np.any(counts > k) or
                not np.isfinite(strength).all() or np.any(strength < 0) or
                np.any(dataset.view_counts[start:stop] < 1) or
                np.any(dataset.view_counts[start:stop] > v)):
            raise ValueError("Invalid source/view counts or strengths")
        mode = m.get("driver_strength_mode", "uniform-0-1")
        if mode == "fixed-one" and not np.array_equal(strength, active.astype(np.float64)):
            raise ValueError("Unit-source archive contains nonunit active strengths")
        if mode == "uniform-0-1" and (np.any(strength[active] <= 0) or np.any(strength[active] >= 1)):
            raise ValueError("Random-source archive contains strengths outside (0,1)")
        for scene in range(start, stop):
            views = dataset.local_views[scene, :int(dataset.view_counts[scene])].astype(np.int64)
            if (np.any(views[:, 2] <= 0) or np.any(views[:, 0] + views[:, 2] > w) or
                    np.any(views[:, 1] + views[:, 2] > w)):
                raise ValueError(f"Out-of-world view in scene {scene}")
            unique_sizes, numbers = np.unique(views[:, 2], return_counts=True)
            sizes.update(dict(zip(map(int, unique_sizes), map(int, numbers))))
    return {"num_scenes": n, "num_views": len(dataset), "window_size_counts": dict(sizes),
            "driver_strength_mode": m.get("driver_strength_mode", "uniform-0-1")}


def randomize_strengths(base_root: Path, destination: Path, seed: int = 20261028) -> dict:
    """Make deterministic random-strength derived data without copying bitmaps.

    Visibility is source-specific and independent of intensity. Reweighting
    the stored channels therefore produces the exact additive observation for
    the new source strengths; no visibility rendering is needed.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Strength seed must be a nonnegative integer")
    base = ComponentDataset(base_root)
    if base.manifest.get("driver_strength_mode") != "fixed-one":
        raise ValueError("Random-strength overlay requires the verified unit-strength base")
    destination = destination.expanduser().resolve()
    base_manifest_hash = hashlib.sha256((base.root / "manifest.json").read_bytes()).hexdigest()
    if destination.exists():
        current = json.loads((destination / "manifest.json").read_text())
        if (current.get("dataset_layout") != "rind-strength-overlay-v1" or
                current.get("strength_overlay_seed") != seed or
                current.get("base_manifest_sha256") != base_manifest_hash or
                Path(current["base_root"]).resolve() != base.root):
            raise ValueError("Strength overlay destination exists with a different seed/base")
        provenance = json.loads((destination / "strength_provenance.json").read_text())
        if sha256_file(destination / "driver_strengths.npy") != provenance["strength_array_sha256"]:
            raise ValueError("Derived random strength array checksum changed")
        return validate_installed(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rind-strength-", dir=destination.parent))
    try:
        n, k = base.num_scenes, int(base.manifest["max_drivers"])
        strengths = np.lib.format.open_memmap(staging / "driver_strengths.npy", mode="w+",
                                             dtype=np.float64, shape=(n, k))
        rng = np.random.default_rng(seed)
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            active = np.arange(k)[None, :] < np.asarray(base.driver_counts[start:stop])[:, None]
            values = rng.random((stop - start, k), dtype=np.float64)
            while np.any(values[active] <= 0):
                invalid = active & (values <= 0)
                values[invalid] = rng.random(int(invalid.sum()), dtype=np.float64)
            strengths[start:stop] = np.where(active, values, 0.0)
        strengths.flush()
        del strengths
        manifest = dict(base.manifest)
        manifest.update({
            "dataset_name": "RIND random-strength overlay of unit random-window release",
            "dataset_layout": "rind-strength-overlay-v1",
            "variant": "multi-source-random-strength-random-windows-derived",
            "driver_strength_mode": "uniform-0-1", "strength_overlay_seed": seed,
            "base_root": str(base.root), "base_manifest_sha256": base_manifest_hash,
            "reference_policy": "generating-configuration-only", "candidate_capacity": 1,
            "candidate_method": "generating configuration only; unit-source witnesses disabled",
            "candidate_scope": "evaluation-only; never teacher proposals",
            "informative_rule": "stored source-free windows; variation rechecked after strength reweighting",
        })
        write_json(staging / "manifest.json", manifest)
        strength_digest = sha256_file(staging / "driver_strengths.npy")
        write_json(staging / "strength_provenance.json", {
            "schema": "rind-strength-overlay-v1", "seed": seed, "rng": "numpy.PCG64",
            "active_strengths": "independent Uniform(0,1); exact zero rejected",
            "inactive_strengths": 0.0, "base_root": str(base.root),
            "base_manifest_sha256": base_manifest_hash,
            "strength_array_sha256": strength_digest,
            "response_rule": "sum_j strength_j * stored_per_source_visibility_j",
            "visibility_rerendered": False,
            "unit_reference_witnesses": "disabled; only generating source configuration returned to evaluation callers",
            "unit_source_files_modified": False,
        })
        summary = validate_installed(staging)
        os.replace(staging, destination)
        return {**summary, "seed": seed, "strength_array_sha256": strength_digest,
                "destination": str(destination)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def install_archive(archive_path: Path, destination: Path, expected_sha256: str | None = None) -> dict:
    destination = destination.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    digest = sha256_file(archive_path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("Archive SHA256 mismatch")
    if destination.exists():
        record_path = destination / "installation.json"
        if not record_path.is_file() or json.loads(record_path.read_text()).get("archive_sha256") != digest:
            raise ValueError("Destination already exists with a different/unrecorded archive; choose a new destination")
        return validate_installed(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rind-new-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            manifests = []
            for item in archive.infolist():
                if PurePosixPath(item.filename).name == "manifest.json" and item.file_size < 1024 * 1024:
                    value = json.loads(archive.read(item))
                    try:
                        validate_manifest(value)
                    except ValueError:
                        continue
                    manifests.append((PurePosixPath(item.filename).parent, value))
            if len(manifests) != 1:
                raise ValueError("ZIP must contain exactly one supported RIND dataset")
            prefix, manifest = manifests[0]
            required = [str(prefix / name) for name in _required_files(manifest)]
            available = archive.namelist()
            expected_hashes = {}
            for item in archive.infolist():
                if PurePosixPath(item.filename).name == "SHA256SUMS" and item.file_size < 8 * 1024 * 1024:
                    hash_root = PurePosixPath(item.filename).parent
                    for line in archive.read(item).decode().splitlines():
                        if not line.strip() or line.startswith("#"):
                            continue
                        checksum, relative = line.split(maxsplit=1)
                        expected_hashes[str(hash_root / relative.lstrip("*"))] = checksum.lower()
            if not all(name in expected_hashes for name in required):
                raise ValueError("Archive checksums do not cover all required dataset files")
            free = shutil.disk_usage(destination.parent).free
            required_bytes = sum(archive.getinfo(name).file_size for name in required)
            if free < required_bytes + 1024**3:
                raise ValueError(f"Insufficient disk: extraction needs {required_bytes / 1024**3:.1f} GiB + 1 GiB margin")
            for number, name in enumerate(required, 1):
                member = PurePosixPath(name)
                item = archive.getinfo(name)
                if (available.count(name) != 1 or member.is_absolute() or ".." in member.parts or
                        "\\" in name or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError(f"Unsafe archive member: {name}")
                print(f"install [{number}/{len(required)}] {member.name} ({item.file_size / 1024**3:.2f} GiB)", flush=True)
                content_hash = hashlib.sha256()
                with archive.open(item) as source, (staging / member.name).open("wb") as target:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        target.write(chunk)
                        content_hash.update(chunk)
                if content_hash.hexdigest() != expected_hashes[name]:
                    raise ValueError(f"Checksum mismatch: {member.name}")
        summary = validate_installed(staging)
        write_json(staging / "installation.json", {
            "archive": archive_path.name, "archive_sha256": digest, **summary,
        })
        os.replace(staging, destination)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_union(base_root: Path, appended_root: Path, destination: Path,
                teacher_root: Path, seed: int = 20261027) -> dict:
    base, appended = ComponentDataset(base_root), ComponentDataset(appended_root)
    if base.global_size != appended.global_size:
        raise ValueError("Cannot append different world sizes")
    protocol = json.loads((teacher_root / "protocol.json").read_text())
    old_splits = protocol.get("scene_splits")
    if not isinstance(old_splits, dict) or set(old_splits) != {"train", "validation", "test"}:
        raise ValueError("Teacher protocol must contain complete scene_splits")
    expected_ids = set(range(base.num_scenes))
    assigned = [int(scene) for split in old_splits.values() for scene in split]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected_ids:
        raise ValueError("Old protocol must assign every base scene exactly once")
    geometry_split = {}
    group_ids = []
    for split, scenes in old_splits.items():
        for scene in scenes:
            fingerprint = base.geometry_group(int(scene))
            if fingerprint in geometry_split and geometry_split[fingerprint] != split:
                raise ValueError("Old scene split already leaks an identical geometry; resolve before appending")
            geometry_split[fingerprint] = split
    splits = {name: list(map(int, old_splits[name])) for name in ("train", "validation", "test")}
    for scene in range(base.num_scenes):
        group_ids.append(base.geometry_group(scene))
    repeated = 0
    for local in range(appended.num_scenes):
        fingerprint = appended.geometry_group(local)
        if fingerprint in geometry_split:
            repeated += 1
            split = geometry_split[fingerprint]
        else:
            uniform = int.from_bytes(hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()[:8], "big") / 2**64
            split = "train" if uniform < 0.8 else "validation" if uniform < 0.9 else "test"
            geometry_split[fingerprint] = split
        splits[split].append(base.num_scenes + local)
        group_ids.append(fingerprint)
        if (local + 1) % 10000 == 0:
            print(f"geometry split [{local + 1}/{appended.num_scenes}] repeated_geometry_configs={repeated}", flush=True)
    components = []
    for dataset in (base, appended):
        components.append({
            "root": str(dataset.root), "num_scenes": dataset.num_scenes,
            "manifest_sha256": hashlib.sha256((dataset.root / "manifest.json").read_bytes()).hexdigest(),
            "driver_strength_mode": dataset.manifest.get("driver_strength_mode", "uniform-0-1"),
            "view_sampling": dataset.manifest.get("view_sampling", "source-free-quadtree"),
        })
    strength_modes = {component["driver_strength_mode"] for component in components}
    manifest = {
        "schema_version": 3, "dataset_layout": "rind-virtual-union-v1",
        "dataset_name": "RIND virtual appended multi-source dataset",
        "num_scenes": base.num_scenes + appended.num_scenes,
        "global_size": base.global_size, "response_type": "additive-driver-intensity",
        "cell_sampling": "cell-center", "array_indexing": "yx", "bit_order": "little",
        "driver_strength_mode": next(iter(strength_modes)) if len(strength_modes) == 1 else "mixed-components",
        "components": components,
        "old_scene_ids_preserved": True, "new_scene_offset": base.num_scenes,
        "geometry_group_split": True,
    }
    destination = destination.expanduser().resolve()
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text())
        split_record = json.loads((destination / "scene_splits.json").read_text())
        if existing != manifest or split_record["scene_splits"] != splits:
            raise ValueError("Union destination exists with different metadata; choose a new union directory")
        return existing
    destination.mkdir(parents=True)
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "scene_splits.json", {
        "scene_splits": splits, "seed": seed,
        "base_teacher_protocol": str((teacher_root / "protocol.json").resolve()),
        "policy": "preserve base split; all identical obstacle geometries share split; hash new geometry groups 80/10/10",
    })
    np.save(destination / "geometry_groups.npy", np.asarray(group_ids, dtype="S64"))
    write_json(destination / "append_report.json", {
        "base_scenes": base.num_scenes, "appended_scenes": appended.num_scenes,
        "scene_configurations_total": manifest["num_scenes"],
        "independent_geometry_groups": len(geometry_split),
        "appended_configurations_with_seen_geometry": repeated,
        "split_scene_counts": {name: len(ids) for name, ids in splits.items()},
        "base_cache_policy": "old teacher paths and old scene IDs are unchanged; render all appended configurations separately",
        "variants": components,
    })
    # Smoke-test both ends through the virtual reader; rasters remain mmap.
    merged = open_dataset(destination)
    merged.get_observation(0, 0)
    merged.get_observation(merged.num_scenes - 1, 0)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--destination", type=Path, required=True)
    install.add_argument("--sha256")
    randomize = subparsers.add_parser("randomize")
    randomize.add_argument("--base-root", type=Path, required=True)
    randomize.add_argument("--destination", type=Path, required=True)
    randomize.add_argument("--seed", type=int, default=20261028)
    merge = subparsers.add_parser("append")
    merge.add_argument("--base-root", type=Path, required=True)
    merge.add_argument("--appended-root", type=Path, required=True)
    merge.add_argument("--destination", type=Path, required=True)
    merge.add_argument("--teacher-root", type=Path, required=True)
    merge.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.command == "install":
        result = install_archive(args.archive, args.destination, args.sha256)
    elif args.command == "randomize":
        result = randomize_strengths(args.base_root, args.destination, args.seed)
    else:
        result = build_union(args.base_root, args.appended_root, args.destination, args.teacher_root, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
