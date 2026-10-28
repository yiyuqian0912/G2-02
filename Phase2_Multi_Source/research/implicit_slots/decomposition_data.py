"""Verified complete residual books and cache-only batches for decomposition.

Every retained window keeps all arithmetic decomposition layers. A layer's
``localization_valid`` flag controls its Phase1 loss only, never its presence
in the mask/intensity/reconstruction targets. Training does not need raw data
or teacher NPZ files once this cache has been indexed.
"""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from slot_data import _digest_array, _load_layer, highest_density_mask


SCHEMA = "phase2-decomposition-book-v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _exact_observation(dataset: Any, directory: Path, scene: int, view: int,
                       size: int, tolerance: float,
                       stored_window: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_path = directory / "combined.npz"
    if combined_path.is_file():
        with np.load(combined_path, allow_pickle=False) as archive:
            response = np.asarray(archive["original"], dtype=np.float64)
            obstacle = np.asarray(archive["obstacle"], dtype=bool)
        # Original teacher metadata stores window coordinates in the layer
        # report/manifest only indirectly; the raw observation supplies W.
        if dataset is not None:
            observation = dataset.get_observation(scene, view)
            window = np.asarray(observation["window"], dtype=np.float64)
            if not np.array_equal(obstacle, np.asarray(observation["obstacle"], dtype=bool)):
                raise ValueError("stored obstacle does not match raw observation")
            # get_observation intentionally transports float32. Account for
            # its quantization, never using it for exact residual arithmetic.
            raw32 = np.asarray(observation["response"], dtype=np.float32)
            if not np.array_equal(response.astype(np.float32), raw32):
                raise ValueError("stored original does not match raw observation")
        elif stored_window is not None:
            window = np.asarray(stored_window, dtype=np.float64)
        else:
            raise ValueError("raw dataset unavailable and no saved window coordinates")
    else:
        if dataset is None:
            raise ValueError("raw dataset unavailable and combined.npz original is missing")
        observation = dataset.get_observation(scene, view)
        window = np.asarray(observation["window"], dtype=np.float64)
        x, y, side = map(int, window)
        if not hasattr(dataset, "get_region"):
            raise ValueError("exact original missing: combined.npz or float64 get_region required")
        exact = dataset.get_region(scene, x, y, side)
        response = np.asarray(exact["response"], dtype=np.float64)
        obstacle = np.asarray(exact["obstacle"], dtype=bool)
    if (response.shape != (size, size) or obstacle.shape != response.shape or
            window.shape != (3,) or window[2] != size or
            not np.isfinite(response).all() or not np.isfinite(window).all()):
        raise ValueError("invalid original response, obstacle or window")
    if np.any(response < -tolerance) or np.any(np.abs(response[obstacle]) > tolerance):
        raise ValueError("invalid negative response or nonzero obstacle response")
    return response, obstacle, window


def _read_complete_window(config: dict[str, Any], dataset: Any, directory: Path,
                          row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    options = config.get("decomposition", {})
    tolerance = float(options.get("arithmetic_tolerance", 1e-9))
    threshold = float(config.get("layer_filter", {}).get("maximum_training_physical_cost", 0.01))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("arithmetic_tolerance must be finite and positive")
    scene, view, size = int(row["scene_id"]), int(row["view_id"]), int(row["size"])
    original, obstacle, window = _exact_observation(
        dataset, directory, scene, view, size, tolerance, row.get("window", report.get("window")))
    free = ~obstacle
    if not free.any():
        return {"skip": "no_free_pixels"}
    if not np.any(original[free] > 1e-12):
        return {"skip": "all_dark"}
    if np.all(original[free] > 1e-12):
        return {"skip": "all_lit"}
    recovered = int(report.get("layers_recovered", 0))
    details = {int(item["step"]): item for item in report.get("layers", [])}
    if recovered < 1 or set(details) != set(range(1, recovered + 1)):
        raise ValueError("complete report has inconsistent layer count or order")
    residual = original.copy()
    layers = []
    valid_reference = candidate_reference = None
    for step in range(1, recovered + 1):
        path = directory / f"layer-{step:02d}.npz"
        layer = _load_layer(path, include_cost=True)
        with np.load(path, allow_pickle=False) as archive:
            raw_mask = np.asarray(archive["layer_response"])
            if not np.isin(raw_mask, (0, 1)).all():
                raise ValueError(f"layer {step}: response is not binary")
            if "residual_before" not in archive or "residual_after" not in archive:
                raise ValueError(f"layer {step}: missing residual process labels")
            before = np.asarray(archive["residual_before"], dtype=np.float64)
            after = np.asarray(archive["residual_after"], dtype=np.float64)
        mask = np.asarray(layer["response"], dtype=bool)
        intensity = details[step].get("intensity")
        if (not isinstance(intensity, (int, float)) or not math.isfinite(intensity)
                or intensity <= 0):
            raise ValueError(f"layer {step}: invalid effective intensity")
        if mask.shape != original.shape or np.any(mask[obstacle]) or not mask[free].any():
            raise ValueError(f"layer {step}: empty mask or mask outside free space")
        if (before.shape != original.shape or after.shape != original.shape or
                not np.isfinite(before).all() or not np.isfinite(after).all()):
            raise ValueError(f"layer {step}: invalid residual arrays")
        if np.max(np.abs(before - residual), initial=0) > tolerance:
            raise ValueError(f"layer {step}: residual_before is not previous residual_after")
        updated = residual - float(intensity) * mask
        if np.min(updated[free], initial=0) < -tolerance:
            raise ValueError(f"layer {step}: subtraction creates negative residual")
        if np.max(np.abs(updated - after), initial=0) > tolerance:
            raise ValueError(f"layer {step}: residual_after violates subtraction")
        residual = after.copy()
        if not layer["valid"].any():
            raise ValueError(f"layer {step}: empty candidate validity support")
        if np.any(layer["probability"][~layer["valid"]] > 1e-14):
            raise ValueError(f"layer {step}: teacher mass outside physical support")
        if not np.isfinite(layer["probability"]).all() or np.any(layer["probability"] < 0):
            raise ValueError(f"layer {step}: invalid teacher probabilities")
        if not np.isfinite(layer["cost"][layer["valid"]]).all():
            raise ValueError(f"layer {step}: invalid physical costs")
        if candidate_reference is None:
            candidate_reference, valid_reference = layer["candidate"], layer["valid"]
        elif (not np.array_equal(candidate_reference, layer["candidate"]) or
              not np.array_equal(valid_reference, layer["valid"])):
            raise ValueError("layers do not share one candidate grid and validity support")
        minimum_cost = float(layer["cost"][layer["valid"]].min())
        informative = bool(mask[free].any() and not mask[free].all())
        layers.append({
            "step": step, "intensity": float(intensity), "mask": mask,
            "probability": layer["probability"], "minimum_physical_cost": minimum_cost,
            "informative": informative,
            "localization_valid": bool(informative and minimum_cost <= threshold),
        })
    final_error = float(np.max(np.abs(residual[free]), initial=0))
    if final_error > tolerance:
        raise ValueError("complete report has nonzero final residual")
    reconstructed = sum(item["intensity"] * item["mask"] for item in layers)
    reconstruction_error = float(np.max(np.abs(reconstructed - original), initial=0))
    if reconstruction_error > tolerance:
        raise ValueError("full layer set fails original-response reconstruction")
    return {
        "original": original, "obstacle": obstacle, "window": window,
        "candidate": candidate_reference, "valid": valid_reference, "layers": layers,
        "final_residual_max": final_error, "reconstruction_max_error": reconstruction_error,
    }


def build_decomposition_index(config: dict[str, Any], dataset: Any,
                              teacher_root: Path, output_root: Path) -> dict[str, Any]:
    """Verify complete books and build contiguous data, mask and q memmaps.

    Invalid/overflow windows are excluded as whole windows and audited. No
    layer is silently discarded. The old teachers and old slot cache remain
    untouched; this schema uses ``decomposition_cache`` and its own index.
    """
    teacher_root, output_root = Path(teacher_root), Path(output_root)
    source_path = teacher_root / "records.json"
    source = json.loads(source_path.read_text())
    max_slots = int(config["model"].get("max_slots", 32))
    if max_slots < 1:
        raise ValueError("max_slots must be positive")
    cache_root = output_root / "decomposition_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    statistics, excluded = {}, []
    source_scenes: dict[str, set[int]] = {}
    source_digest = hashlib.sha256(source_path.read_bytes())
    reference_grid = None
    dtype_for = {
        "response": "<f4", "obstacle_bits": "u1", "mask_bits": "u1",
        "probability": "<f4", "core_bits": "u1", "valid": "u1",
    }
    for split in records:
        incoming = source["records"].get(split, [])
        source_scenes[split] = {int(item["scene_id"]) for item in incoming}
        identities = set()
        counts: dict[str, int] = defaultdict(int)
        counts["raw_windows"] = len(incoming)
        offsets = {"response": 0, "obstacle_bits": 0, "mask_bits": 0}
        layer_count = 0
        streams = {key: (cache_root / f"{split}_{key}.tmp").open("wb") for key in dtype_for}
        try:
            for position, window in enumerate(incoming, 1):
                scene, view = int(window["scene_id"]), int(window["view_id"])
                if (scene, view) in identities:
                    raise ValueError(f"duplicate teacher window in {split}: {(scene, view)}")
                identities.add((scene, view))
                directory = teacher_root / window["directory"]
                report_path = directory / "report.json"
                report = json.loads(report_path.read_text())
                if report.get("status") != "complete" or report.get("partial_combined_distribution"):
                    reason = "incomplete_teacher"
                    counts[reason] += 1
                    excluded.append({"split": split, "scene_id": scene, "view_id": view, "reason": reason})
                    continue
                try:
                    data = _read_complete_window(config, dataset, directory, window, report)
                except (ValueError, KeyError, FileNotFoundError, IndexError) as error:
                    counts["invalid_book"] += 1
                    excluded.append({"split": split, "scene_id": scene, "view_id": view,
                                     "reason": "invalid_book", "detail": str(error)})
                    continue
                if "skip" in data:
                    reason = data["skip"]
                    counts[reason] += 1
                    excluded.append({"split": split, "scene_id": scene, "view_id": view, "reason": reason})
                    continue
                layers = data["layers"]
                counts[f"full_layer_count_{len(layers)}"] += 1
                if len(layers) > max_slots:
                    counts["overflow_windows"] += 1
                    excluded.append({"split": split, "scene_id": scene, "view_id": view,
                                     "reason": "overflow", "layer_count": len(layers)})
                    continue
                candidate_digest = _digest_array(data["candidate"])
                if reference_grid is None:
                    reference_grid = {"sha256": candidate_digest,
                                      "candidate_count": len(data["candidate"])}
                    np.save(cache_root / "candidate_xy.npy", data["candidate"].astype(np.float32))
                elif reference_grid["sha256"] != candidate_digest:
                    raise ValueError("complete teacher books use different candidate grids")
                metadata_layers = []
                mask_bytes = (int(window["size"]) ** 2 + 7) // 8
                for layer in layers:
                    probability = np.asarray(layer["probability"], dtype=np.float32)
                    probability.tofile(streams["probability"])
                    np.packbits(highest_density_mask(layer["probability"], 0.75), bitorder="little").tofile(streams["core_bits"])
                    np.packbits(layer["mask"].ravel(), bitorder="little").tofile(streams["mask_bits"])
                    metadata_layers.append({
                        key: layer[key] for key in ("step", "intensity", "minimum_physical_cost",
                                                    "informative", "localization_valid")
                    } | {"cache_layer_index": layer_count, "mask_offset": offsets["mask_bits"]})
                    offsets["mask_bits"] += mask_bytes
                    layer_count += 1
                row = {
                    "split": split, "scene_id": scene, "view_id": view, "size": int(window["size"]),
                    "directory": window["directory"], "layers": metadata_layers,
                    "layer_count": len(layers), "localizable_layer_count": sum(layer["localization_valid"] for layer in layers),
                    "cache_window_index": len(records[split]), "response_offset": offsets["response"],
                    "obstacle_offset": offsets["obstacle_bits"], "window": data["window"].tolist(),
                    "final_residual_max": data["final_residual_max"],
                    "reconstruction_max_error": data["reconstruction_max_error"],
                }
                data["original"].astype("<f4").ravel().tofile(streams["response"])
                np.packbits(data["obstacle"].ravel(), bitorder="little").tofile(streams["obstacle_bits"])
                np.asarray(data["valid"], dtype=np.uint8).tofile(streams["valid"])
                offsets["response"] += int(window["size"]) ** 2
                offsets["obstacle_bits"] += mask_bytes
                records[split].append(row)
                counts["windows"] += 1
                counts["layers"] += len(layers)
                counts["localizable_layers"] += row["localizable_layer_count"]
                # Hash the validated content, independent of path or mtime.
                source_digest.update(report_path.read_bytes())
                source_digest.update(json.dumps(metadata_layers, sort_keys=True).encode())
                for value in (data["original"], data["obstacle"], data["window"], data["candidate"], data["valid"]):
                    source_digest.update(_digest_array(value).encode())
                for layer in layers:
                    source_digest.update(_digest_array(layer["mask"]).encode())
                    source_digest.update(_digest_array(layer["probability"]).encode())
                if position == 1 or position % 100 == 0 or position == len(incoming):
                    print(f"index complete books | {split} checked={position}/{len(incoming)} accepted={len(records[split])} layers={layer_count}", flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        for key, dtype in dtype_for.items():
            extension = "f32" if dtype == "<f4" else "u8"
            (cache_root / f"{split}_{key}.tmp").replace(cache_root / f"{split}_{key}.{extension}")
        statistics[split] = dict(counts)
    names = list(records)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if source_scenes[names[left]] & source_scenes[names[right]]:
                raise ValueError(f"scene leakage: {names[left]} vs {names[right]}")
    if reference_grid is None:
        raise ValueError("no valid complete decomposition books")
    file_contract = {}
    for path in sorted(cache_root.iterdir()):
        if path.is_file() and path.suffix in (".f32", ".u8", ".npy"):
            file_contract[path.name] = {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    index = {
        "schema": SCHEMA, "teacher_manifest": str(source_path.resolve()),
        "dataset_root": str(getattr(dataset, "root", config.get("data_root", "unavailable"))),
        "global_size": int(getattr(dataset, "manifest", {}).get("global_size", config.get("global_size", 1024))),
        "corpus_sha256": source_digest.hexdigest(), "max_slots": max_slots,
        "candidate_grid": reference_grid, "records": records,
        "counts": {split: len(rows) for split, rows in records.items()},
        "statistics": statistics, "excluded_windows": excluded,
        "contract": {
            "all_arithmetic_layers_kept": True, "raw_data_required_for_training": False,
            "overflow_policy": "exclude whole window; never truncate layers",
            "localization_threshold": float(config.get("layer_filter", {}).get("maximum_training_physical_cost", 0.01)),
            "arithmetic_tolerance": float(config.get("decomposition", {}).get("arithmetic_tolerance", 1e-9)),
            "normalization_support": config.get("training", {}).get("normalization_support", "world"),
            "cache_files": file_contract,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "decomposition_records.json", index)
    return index


class DecompositionRecords:
    """Complete window targets backed only by verified contiguous memmaps."""

    def __init__(self, entries: list[dict[str, Any]], cache_root: Path, split: str,
                 max_slots: int, normalization_support: str = "world") -> None:
        self.entries = list(entries)
        self.cache_root, self.split = Path(cache_root), split
        self.max_slots, self.normalization_support = int(max_slots), normalization_support
        if normalization_support not in ("world", "geometry"):
            raise ValueError("invalid normalization support")
        self.candidate = np.load(self.cache_root / "candidate_xy.npy", mmap_mode="r")
        self.candidate_count = len(self.candidate)
        self.candidate_counts = [self.candidate_count] * len(entries)
        self.window_sizes = [int(row["size"]) for row in entries]
        self.scene_ids = {int(row["scene_id"]) for row in entries}
        self._maps = {}
        for key, dtype, extension in (
            ("response", "<f4", "f32"), ("obstacle_bits", "u1", "u8"),
            ("mask_bits", "u1", "u8"), ("probability", "<f4", "f32"),
            ("core_bits", "u1", "u8"), ("valid", "u1", "u8"),
        ):
            path = self.cache_root / f"{split}_{key}.{extension}"
            if not path.is_file():
                raise FileNotFoundError(f"complete-book cache missing: {path}")
            self._maps[key] = (np.memmap(path, mode="r", dtype=dtype)
                               if path.stat().st_size else np.empty(0, dtype=dtype))
        self.core_bytes = (self.candidate_count + 7) // 8
        for row in self.entries:
            if len(row["layers"]) > self.max_slots:
                raise ValueError("complete window exceeds max_slots; no silent truncation")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        size, count = int(row["size"]), len(row["layers"])
        pixels, bit_bytes = size * size, (size * size + 7) // 8
        offset = int(row["response_offset"])
        response = np.array(self._maps["response"][offset:offset + pixels], dtype=np.float32).reshape(size, size)
        offset = int(row["obstacle_offset"])
        obstacle = np.unpackbits(self._maps["obstacle_bits"][offset:offset + bit_bytes], bitorder="little")[:pixels].reshape(size, size).astype(bool)
        valid_offset = int(row["cache_window_index"]) * self.candidate_count
        valid = np.array(self._maps["valid"][valid_offset:valid_offset + self.candidate_count], dtype=bool)
        teacher_mask = np.zeros((self.max_slots, size, size), dtype=bool)
        teacher_probability = np.zeros((self.max_slots, self.candidate_count), dtype=np.float32)
        teacher_core = np.zeros_like(teacher_probability, dtype=bool)
        teacher_intensity = np.zeros(self.max_slots, dtype=np.float32)
        layer_mask = np.arange(self.max_slots) < count
        localization_valid = np.zeros(self.max_slots, dtype=bool)
        for slot, detail in enumerate(row["layers"]):
            layer_index = int(detail["cache_layer_index"])
            offset = int(detail["mask_offset"])
            teacher_mask[slot] = np.unpackbits(self._maps["mask_bits"][offset:offset + bit_bytes], bitorder="little")[:pixels].reshape(size, size).astype(bool)
            offset = layer_index * self.candidate_count
            teacher_probability[slot] = self._maps["probability"][offset:offset + self.candidate_count]
            teacher_intensity[slot] = detail["intensity"]
            localization_valid[slot] = bool(detail["localization_valid"])
            offset = layer_index * self.core_bytes
            teacher_core[slot] = np.unpackbits(self._maps["core_bits"][offset:offset + self.core_bytes], bitorder="little")[:self.candidate_count].astype(bool)
        q = teacher_probability[localization_valid]
        occupancy = (1.0 - np.prod(1.0 - q, axis=0) if len(q)
                     else np.zeros(self.candidate_count, dtype=np.float32))
        return {
            "response": response, "obstacle": obstacle,
            "window": np.asarray(row["window"], dtype=np.float32),
            "candidate_xy": np.array(self.candidate, dtype=np.float32),
            "valid": valid, "support": valid.copy() if self.normalization_support == "geometry" else np.ones_like(valid),
            "teacher_mask": teacher_mask, "teacher_probability": teacher_probability,
            "teacher_core": teacher_core, "teacher_intensity": teacher_intensity,
            "layer_mask": layer_mask, "localization_valid": localization_valid,
            "teacher_occupancy": occupancy.astype(np.float32),
            "count_target": np.int64(count),
            "scene_id": np.int64(row["scene_id"]), "view_id": np.int64(row["view_id"]),
        }


def collate_decomposition_records(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not records or len({tuple(row["response"].shape) for row in records}) != 1:
        raise ValueError("decomposition batches require one native window size")
    bool_keys = {"obstacle", "valid", "support", "teacher_mask", "teacher_core", "layer_mask", "localization_valid"}
    int_keys = {"count_target", "scene_id", "view_id"}
    return {
        key: torch.as_tensor(np.stack([np.asarray(row[key]) for row in records]),
                             dtype=torch.bool if key in bool_keys else torch.long if key in int_keys else torch.float32)
        for key in records[0]
    }


def _batch_groups(records: DecompositionRecords, batch_size: int,
                  rng: np.random.Generator | None, rank: int, world_size: int,
                  training: bool) -> list[list[int]]:
    if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid decomposition batching arguments")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(records.window_sizes, records.candidate_counts)):
        buckets[(int(key[0]), int(key[1]))].append(index)
    groups = []
    for indices in buckets.values():
        if rng is not None:
            rng.shuffle(indices)
        if training:
            global_batch = batch_size * world_size
            for start in range(0, len(indices), global_batch):
                current = indices[start:start + global_batch]
                if len(current) < global_batch:
                    current = current + [current[item % len(current)] for item in range(global_batch - len(current))]
                groups.append(current[rank::world_size])
        else:
            # Inference uses the unwrapped module. Ranks may have different
            # batch counts; each real window appears exactly once globally.
            local = indices[rank::world_size]
            groups.extend(local[start:start + batch_size] for start in range(0, len(local), batch_size))
    if rng is not None:
        rng.shuffle(groups)
    return groups


def batch_count(records: DecompositionRecords, batch_size: int, world_size: int = 1,
                rank: int = 0, training: bool = True) -> int:
    return len(_batch_groups(records, batch_size, None, rank, world_size, training))


def decomposed_batches(records: DecompositionRecords, batch_size: int,
                       rng: np.random.Generator | None = None, prefetch: int = 0,
                       rank: int = 0, world_size: int = 1,
                       training: bool = True) -> Iterator[dict[str, torch.Tensor]]:
    groups = _batch_groups(records, batch_size, rng, rank, world_size, training)
    def load(group: list[int]) -> dict[str, torch.Tensor]:
        return collate_decomposition_records([records[index] for index in group])
    if prefetch < 1:
        for group in groups:
            yield load(group)
        return
    with ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="decomposition-batch") as pool:
        pending: deque = deque()
        iterator = iter(groups)
        for _ in range(min(prefetch, len(groups))):
            pending.append(pool.submit(load, next(iterator)))
        while pending:
            future = pending.popleft()
            group = next(iterator, None)
            if group is not None:
                pending.append(pool.submit(load, group))
            yield future.result()
