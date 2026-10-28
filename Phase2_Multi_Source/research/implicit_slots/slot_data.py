"""Teacher-book indexing and native-size batches for implicit slots."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def _digest_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_layer(path: Path, include_cost: bool = True) -> dict[str, np.ndarray]:
    """Load a layer and expand legacy sparse grids to the common full grid."""
    with np.load(path, allow_pickle=False) as archive:
        response = np.asarray(archive["layer_response"], dtype=bool)
        candidate_raw = np.asarray(archive["candidate_xy"], dtype=np.float64)
        grid_index = np.asarray(archive["grid_index"], dtype=np.int64)
        x_axis = np.asarray(archive["x_axis"], dtype=np.float64)
        y_axis = np.asarray(archive["y_axis"], dtype=np.float64)
        valid_raw = np.asarray(archive["valid"], dtype=bool)
        probability_raw = np.asarray(archive["probability"], dtype=np.float64)
        cost_raw = np.asarray(archive["physical_cost"], dtype=np.float64) if include_cost else None
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    candidate = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.array_equal(candidate_raw, candidate[grid_index]):
        raise ValueError(f"candidate/grid mismatch: {path}")
    count = len(candidate)
    valid = np.zeros(count, dtype=bool)
    probability = np.zeros(count, dtype=np.float64)
    valid[grid_index] = valid_raw
    probability[grid_index] = probability_raw
    if not np.isclose(probability.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f"layer probability is not normalized: {path}")
    result = {"response": response, "candidate": candidate, "valid": valid, "probability": probability}
    if include_cost:
        cost = np.full(count, np.inf, dtype=np.float64)
        cost[grid_index] = cost_raw
        result["cost"] = cost
    return result


def highest_density_mask(probability: np.ndarray, mass: float = 0.75) -> np.ndarray:
    order = np.argsort(-probability, kind="stable")
    boundary = min(int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")), len(order) - 1)
    cutoff = float(probability[order[boundary]])
    tolerance = max(np.finfo(np.float64).eps * max(1.0, abs(cutoff)) * 8, 1e-15)
    return probability >= cutoff - tolerance


def build_slot_index(config: dict[str, Any], dataset: Any, teacher_root: Path, output_root: Path) -> dict[str, Any]:
    manifest_path = teacher_root / "records.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"teacher records are missing: {manifest_path}")
    source = json.loads(manifest_path.read_text())
    threshold = float(config["layer_filter"]["maximum_training_physical_cost"])
    max_slots = int(config["model"]["max_slots"])
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    statistics: dict[str, dict[str, Any]] = {}
    reference_grid = None
    split_scenes: dict[str, set[int]] = {}
    cache_root = output_root / "training_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for split in records:
        counts = defaultdict(int)
        split_scenes[split] = set()
        identities = set()
        probability_temp = cache_root / f"{split}_probability.tmp"
        core_temp = cache_root / f"{split}_core_bits.tmp"
        valid_temp = cache_root / f"{split}_valid.tmp"
        cached_layers = cached_windows = 0
        with probability_temp.open("wb") as probability_stream, core_temp.open("wb") as core_stream, valid_temp.open("wb") as valid_stream:
            for window in source["records"][split]:
                scene, view = int(window["scene_id"]), int(window["view_id"])
                identity = (scene, view)
                if identity in identities:
                    raise ValueError(f"duplicate window in {split}: {identity}")
                identities.add(identity)
                directory = teacher_root / window["directory"]
                report = json.loads((directory / "report.json").read_text())
                if report.get("status") != "complete" or report.get("partial_combined_distribution"):
                    counts["incomplete_windows"] += 1
                    continue
                observation = dataset.get_observation(scene, view)
                free = ~np.asarray(observation["obstacle"], dtype=bool)
                response = np.asarray(observation["response"], dtype=np.float64)
                if not free.any() or not np.any(response[free] > 1e-12) or np.all(response[free] > 1e-12):
                    counts["constant_or_empty_windows"] += 1
                    continue
                detail_by_step = {int(item["step"]): item for item in report.get("layers", [])}
                retained = []
                retained_arrays = []
                candidate_digest = None
                window_valid = None
                for step in range(1, int(report["layers_recovered"]) + 1):
                    path = directory / f"layer-{step:02d}.npz"
                    layer = _load_layer(path, include_cost=True)
                    if layer["response"].shape != free.shape or np.any(layer["response"][~free]):
                        raise ValueError(f"layer response is outside free window: {path}")
                    digest = _digest_array(layer["candidate"])
                    candidate_digest = digest if candidate_digest is None else candidate_digest
                    if digest != candidate_digest:
                        raise ValueError(f"layers use different candidate grids: {directory}")
                    if reference_grid is None:
                        reference_grid = {"sha256": digest, "candidate_count": int(len(layer["candidate"]))}
                        np.save(cache_root / "candidate_xy.npy", layer["candidate"].astype(np.float32))
                    elif reference_grid["sha256"] != digest:
                        raise ValueError("teacher cache does not use one common candidate grid")
                    if window_valid is None:
                        window_valid = layer["valid"]
                    elif not np.array_equal(window_valid, layer["valid"]):
                        raise ValueError(f"layers use different physical validity masks: {directory}")
                    all_zero = not bool(np.any(layer["response"][free]))
                    all_one = bool(np.all(layer["response"][free]))
                    minimum_cost = float(layer["cost"][layer["valid"]].min())
                    detail = detail_by_step.get(step, {})
                    intensity = detail.get("intensity")
                    if not isinstance(intensity, (int, float)) or not math.isfinite(intensity) or intensity <= 0:
                        raise ValueError(f"missing positive layer intensity: {path}")
                    informative = not all_zero and not all_one
                    high_confidence = informative and minimum_cost <= threshold
                    counts["layers_total"] += 1
                    counts["layers_informative"] += int(informative)
                    counts["layers_high_confidence"] += int(high_confidence)
                    if high_confidence:
                        retained.append({"step": step, "file": path.name, "intensity": float(intensity),
                                         "minimum_physical_cost": minimum_cost,
                                         "cache_layer_index": cached_layers + len(retained_arrays)})
                        retained_arrays.append(layer["probability"])
                if not retained:
                    counts["windows_without_retained_layers"] += 1
                    continue
                for probability in retained_arrays:
                    probability.astype("<f4", copy=False).tofile(probability_stream)
                    np.packbits(highest_density_mask(probability, 0.75), bitorder="little").tofile(core_stream)
                np.asarray(window_valid, dtype=np.uint8).tofile(valid_stream)
                cached_layers += len(retained)
                overflow = len(retained) > max_slots
                counts["windows"] += 1
                counts["overflow_windows"] += int(overflow)
                counts[f"layer_count_{len(retained)}"] += 1
                split_scenes[split].add(scene)
                records[split].append({
                    "split": split, "scene_id": scene, "view_id": view,
                    "size": int(window["size"]), "directory": window["directory"],
                    "layers": retained, "layer_count": len(retained), "overflow": overflow,
                    "cache_window_index": cached_windows,
                })
                cached_windows += 1
        probability_path = cache_root / f"{split}_probability.f32"
        core_path = cache_root / f"{split}_core_bits.u8"
        valid_path = cache_root / f"{split}_valid.u8"
        probability_temp.replace(probability_path)
        core_temp.replace(core_path)
        valid_temp.replace(valid_path)
        counts["cached_layers"] = cached_layers
        counts["cached_windows"] = cached_windows
        statistics[split] = dict(counts)
    if split_scenes["train"] & split_scenes["validation"] or split_scenes["train"] & split_scenes["test"] or split_scenes["validation"] & split_scenes["test"]:
        raise ValueError("scene leakage across slot splits")
    index = {
        "schema": "phase2-implicit-contribution-slots-v1",
        "teacher_manifest": str(manifest_path.resolve()),
        "max_slots": max_slots,
        "overflow_policy": "aggregate occupancy and capped count only; no truncated slot matching",
        "cache": {
            "root": str(cache_root.resolve()), "probability_dtype": "float32",
            "valid_dtype": "uint8", "core_encoding": "packbits-little-75pct-hdr",
        },
        "records": records,
        "counts": {name: len(rows) for name, rows in records.items()},
        "statistics": statistics,
        "candidate_grid": reference_grid,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slot_records.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


class WindowSlotRecords:
    def __init__(self, entries: list[dict[str, Any]], dataset: Any, cache_root: Path,
                 split: str, max_slots: int, normalization_support: str = "world") -> None:
        self.entries = list(entries)
        self.dataset = dataset
        self.cache_root = Path(cache_root)
        self.split = split
        self.max_slots = int(max_slots)
        self.normalization_support = normalization_support
        self.window_sizes = [int(row["size"]) for row in entries]
        self.candidate = np.load(self.cache_root / "candidate_xy.npy", mmap_mode="r")
        self.candidate_count = len(self.candidate)
        self.candidate_counts = [self.candidate_count] * len(entries)
        self.scene_ids = {int(row["scene_id"]) for row in entries}
        layer_count = 0 if not entries else 1 + max(
            int(layer["cache_layer_index"]) for row in entries for layer in row["layers"]
        )
        self.probability = np.memmap(
            self.cache_root / f"{split}_probability.f32", mode="r", dtype="<f4",
            shape=(layer_count, self.candidate_count),
        )
        self.core_bytes = (self.candidate_count + 7) // 8
        self.core_bits = np.memmap(
            self.cache_root / f"{split}_core_bits.u8", mode="r", dtype=np.uint8,
            shape=(layer_count, self.core_bytes),
        )
        self.valid_cache = np.memmap(
            self.cache_root / f"{split}_valid.u8", mode="r", dtype=np.uint8,
            shape=(len(entries), self.candidate_count),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        observation = self.dataset.get_observation(row["scene_id"], row["view_id"])
        layer_indices = np.asarray([detail["cache_layer_index"] for detail in row["layers"]], dtype=np.int64)
        probabilities = np.asarray(self.probability[layer_indices], dtype=np.float32)
        intensities = np.asarray([detail["intensity"] for detail in row["layers"]], dtype=np.float32)
        valid = np.asarray(self.valid_cache[int(row["cache_window_index"])], dtype=bool)
        occupancy = 1.0 - np.prod(1.0 - probabilities, axis=0)
        teacher_probability = np.zeros((self.max_slots, self.candidate_count), dtype=np.float32)
        teacher_core = np.zeros_like(teacher_probability, dtype=bool)
        teacher_intensity = np.zeros(self.max_slots, dtype=np.float32)
        layer_mask = np.zeros(self.max_slots, dtype=bool)
        slot_supervised = len(probabilities) <= self.max_slots
        if slot_supervised:
            for slot, (probability, intensity, cache_index) in enumerate(zip(probabilities, intensities, layer_indices)):
                teacher_probability[slot] = probability
                teacher_core[slot] = np.unpackbits(
                    np.asarray(self.core_bits[cache_index]), bitorder="little"
                )[:self.candidate_count].astype(bool, copy=False)
                teacher_intensity[slot] = intensity
                layer_mask[slot] = True
        support = valid.copy() if self.normalization_support == "geometry" else np.ones_like(valid)
        return {
            "response": np.asarray(observation["response"], dtype=np.float32),
            "obstacle": np.asarray(observation["obstacle"], dtype=bool),
            "window": np.asarray(observation["window"], dtype=np.float32),
            "candidate_xy": np.asarray(self.candidate, dtype=np.float32), "valid": valid, "support": support,
            "teacher_probability": teacher_probability, "teacher_core": teacher_core,
            "teacher_intensity": teacher_intensity, "layer_mask": layer_mask,
            "teacher_occupancy": occupancy.astype(np.float32),
            "count_target": np.int64(min(len(probabilities), self.max_slots)),
            "slot_supervised": np.bool_(slot_supervised),
            "scene_id": np.int64(row["scene_id"]), "view_id": np.int64(row["view_id"]),
        }


def collate_slot_records(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not records or len({record["response"].shape for record in records}) != 1:
        raise ValueError("slot batches must contain one native window size")
    keys = {
        "response": torch.float32, "obstacle": torch.bool, "window": torch.float32,
        "candidate_xy": torch.float32, "valid": torch.bool, "support": torch.bool,
        "teacher_probability": torch.float32, "teacher_core": torch.bool,
        "teacher_intensity": torch.float32, "layer_mask": torch.bool,
        "teacher_occupancy": torch.float32, "count_target": torch.long,
        "slot_supervised": torch.bool, "scene_id": torch.long, "view_id": torch.long,
    }
    return {key: torch.as_tensor(np.stack([np.asarray(record[key]) for record in records]), dtype=dtype)
            for key, dtype in keys.items()}


def slot_batch_count(records: WindowSlotRecords, batch_size: int, world_size: int = 1) -> int:
    """Number of homogeneous global batches yielded on every DDP rank."""
    if batch_size < 1 or world_size < 1:
        raise ValueError("invalid batch/distributed settings")
    global_batch = batch_size * world_size
    bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
    for key in zip(records.window_sizes, records.candidate_counts):
        bucket_counts[(int(key[0]), int(key[1]))] += 1
    return sum((count + global_batch - 1) // global_batch for count in bucket_counts.values())


def slot_batches(records: WindowSlotRecords, batch_size: int, rng: np.random.Generator | None = None,
                 prefetch: int = 0, rank: int = 0, world_size: int = 1) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size < 1 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid batch/distributed settings")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(records.window_sizes, records.candidate_counts)):
        buckets[(int(key[0]), int(key[1]))].append(index)
    groups = []
    global_batch = batch_size * world_size
    for bucket in buckets.values():
        if rng is not None:
            rng.shuffle(bucket)
        for start in range(0, len(bucket), global_batch):
            group = bucket[start:start + global_batch]
            if len(group) < global_batch:
                group = group + [group[index % len(group)] for index in range(global_batch - len(group))]
            groups.append(group[rank::world_size])
    if rng is not None:
        rng.shuffle(groups)
    def load(group: list[int]) -> dict[str, torch.Tensor]:
        return collate_slot_records([records[index] for index in group])
    if prefetch < 1:
        for group in groups:
            yield load(group)
        return
    with ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="slot-batch") as pool:
        pending: deque = deque()
        iterator = iter(groups)
        for _ in range(min(prefetch, len(groups))):
            pending.append(pool.submit(load, next(iterator)))
        while pending:
            future = pending.popleft()
            try:
                group = next(iterator)
            except StopIteration:
                group = None
            if group is not None:
                pending.append(pool.submit(load, group))
            yield future.result()
