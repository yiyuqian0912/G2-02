"""Install only the data from a local RIND release ZIP, then clean the archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import numpy as np
from rind_dataset import RINDDataset
from rind_dataset.constants import PARAM_WIDTH
from rind_phase1.data import PROJECT_ROOT, default_data_root


ARRAYS = ("drivers", "driver_counts", "driver_strengths", "obstacle_counts",
          "obstacle_types", "obstacle_params", "obstacle_group_ids", "obstacle_bits",
          "visibility_bits", "local_views", "view_counts", "reference_positions")
OPTIONAL_FILES = ("generation_stats.json", "reference_generation_stats.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    if (manifest.get("schema_version") != 3 or manifest.get("max_drivers") != 1 or
            manifest.get("driver_strength_mode") != "fixed-one" or
            manifest.get("candidate_layout") != "single-position-list"):
        raise ValueError("Expected the single-source schema-v3 release with unit strength and references")
    for field in ("num_scenes", "global_size", "view_capacity", "candidate_capacity"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError(f"Invalid manifest field: {field}")


def validate_installed(root):
    """Check the installed format without loading full rasters into memory."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _valid_manifest(manifest)
    ds = RINDDataset(root)
    s, w, v = ds.num_scenes, ds.global_size, manifest["view_capacity"]
    k = ds.obstacle_types.shape[1]
    byte_count = (w * w + 7) // 8
    shapes = {"drivers": (s, 1, 2), "driver_counts": (s,), "driver_strengths": (s, 1),
              "obstacle_counts": (s,), "obstacle_types": (s, k),
              "obstacle_params": (s, k, PARAM_WIDTH), "obstacle_group_ids": (s, k),
              "obstacle_bits": (s, byte_count), "visibility_bits": (s, 1, byte_count),
              "local_views": (s, v, 3), "view_counts": (s,),
              "reference_positions": (s, v, manifest["candidate_capacity"], 2)}
    for name, shape in shapes.items():
        array = getattr(ds, name)
        expected_dtype = ("float64" if name in ("drivers", "driver_strengths", "obstacle_params", "reference_positions")
                          else "uint16" if name in ("local_views", "view_counts", "obstacle_group_ids")
                          else "uint8")
        if array.shape != shape or array.dtype != np.dtype(expected_dtype):
            raise ValueError(f"Invalid array shape or dtype: {name}")
    if (not np.all(ds.driver_counts == 1) or not np.all(ds.driver_strengths == 1) or
            not np.isfinite(ds.drivers).all() or np.any(ds.drivers < 0) or np.any(ds.drivers > w) or
            np.any(ds.view_counts <= 0) or np.any(ds.view_counts > v) or
            np.any(ds.obstacle_counts > k)):
        raise ValueError("Invalid source values or scene/view counts")
    return {"num_scenes": s, "num_views": len(ds),
            "references_per_view": manifest["candidate_capacity"]}


def _dataset_members(archive):
    matches = []
    for info in archive.infolist():
        if PurePosixPath(info.filename).name != "manifest.json" or info.file_size > 1024 * 1024:
            continue
        try:
            manifest = json.loads(archive.read(info))
            _valid_manifest(manifest)
        except (ValueError, UnicodeDecodeError):
            continue
        matches.append((PurePosixPath(info.filename).parent, manifest))
    if len(matches) != 1:
        raise ValueError("ZIP must contain exactly one single-source dataset with reference positions")
    prefix, manifest = matches[0]
    names = ["manifest.json"] + [f"{name}.npy" for name in ARRAYS]
    available = archive.namelist()
    names += [name for name in OPTIONAL_FILES if str(prefix / name) in available]
    members = [str(prefix / name) for name in names]
    for member in members:
        if (".." in PurePosixPath(member).parts or "\\" in member or member.startswith("/") or
                available.count(member) != 1):
            raise ValueError(f"Missing, duplicated, or invalid archive member: {member}")
        info = archive.getinfo(member)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Archive member must be a regular file: {member}")
    return members, manifest


def _member_hashes(archive, members):
    for info in archive.infolist():
        if PurePosixPath(info.filename).name != "SHA256SUMS" or info.file_size > 4 * 1024 * 1024:
            continue
        prefix = PurePosixPath(info.filename).parent
        hashes = {}
        for line in archive.read(info).decode("utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*")
            if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()) or
                    ".." in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute()):
                raise ValueError("Invalid SHA256SUMS entry")
            hashes[str(prefix / relative)] = digest.lower()
        if all(member in hashes for member in members):
            return hashes
    raise ValueError("Release ZIP needs SHA256SUMS covering every dataset file")


def choose_archive(project_root):
    candidates = sorted(Path(project_root).glob("*.zip"))
    if len(candidates) == 1:
        return candidates[0]
    matching = []
    for path in candidates:
        try:
            with zipfile.ZipFile(path) as archive:
                _dataset_members(archive)
            matching.append(path)
        except (ValueError, zipfile.BadZipFile, OSError):
            continue
    if len(matching) == 1:
        return matching[0]
    raise ValueError("Place one single-source ZIP in the project root, or pass its path explicitly")


def install_data(archive_path, destination, *, keep_archive=False, expected_sha256=None):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        counts = validate_installed(destination)
        print(f"Data already installed: {destination} ({counts['num_scenes']} scenes). ZIP retained.")
        return destination
    archive_path = Path(archive_path).expanduser().resolve()
    sidecar = Path(str(archive_path) + ".sha256")
    if expected_sha256 is None and sidecar.is_file():
        expected_sha256 = sidecar.read_text(encoding="utf-8").split()[0]
    archive_digest = file_sha256(archive_path)
    if expected_sha256 is not None and archive_digest != expected_sha256.lower():
        raise ValueError("ZIP SHA256 mismatch; archive retained")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rind-install-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members, _ = _dataset_members(archive)
            hashes = _member_hashes(archive, members)
            for number, member in enumerate(members, start=1):
                output = staging / PurePosixPath(member).name
                print(f"[{number}/{len(members)}] Extracting and verifying {output.name}", flush=True)
                digest = hashlib.sha256()
                with archive.open(member) as source, output.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != hashes[member]:
                    raise ValueError(f"Dataset checksum mismatch: {output.name}; ZIP retained")
        counts = validate_installed(staging)
        record = {"archive_name": archive_path.name, "archive_sha256": archive_digest,
                  "schema_version": 3, **counts}
        lockfile = PROJECT_ROOT / "uv.lock"
        if lockfile.is_file():
            record["environment_lock_sha256"] = file_sha256(lockfile)
        (staging / "installation.json").write_text(json.dumps(record, indent=2) + "\n")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if not keep_archive:
        archive_path.unlink()
        if sidecar.is_file():
            sidecar.unlink()
        print(f"Removed verified ZIP: {archive_path.name}")
    print(f"Ready: {destination}\n{counts['num_scenes']} scenes, {counts['num_views']} views, "
          f"{counts['references_per_view']} references per view.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", help="ZIP path; otherwise find it in the project root")
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--keep-archive", action="store_true", help="Keep ZIP after a successful installation")
    parser.add_argument("--sha256", help="Optional expected SHA256 of the complete ZIP")
    args = parser.parse_args()
    try:
        destination = Path(args.data_root).expanduser().resolve()
        if destination.exists():
            validate_installed(destination)
            print(f"Data already installed: {destination}")
            return
        archive = args.archive if args.archive else choose_archive(PROJECT_ROOT)
        install_data(archive, destination, keep_archive=args.keep_archive,
                     expected_sha256=args.sha256)
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Data installation failed: {exc}\n")


if __name__ == "__main__":
    main()
