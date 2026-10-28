"""Cheap local residual labels and cache-only batches; no candidate rendering.

Labels reproduce the current minimum-positive/BFS subtraction rule. They are
algorithmic effective layers, not proof of physical single-source feasibility.
No source identities or per-source visibility channels decide the decomposition.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
PHASE2 = HERE.parents[2]
for path in (PHASE2, PHASE2 / "research/layered_residual", PHASE2 / "src",
             PHASE2 / "vendor/rind-dataset", PHASE2.parent / "Phase1_Single_Source_Single_Intensity/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rind_phase2.regions import response_regions
from layered_residual import minimum_positive_regions, trace_contribution, _constraints

SCHEMA = "rind-local-mask-labels-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_decomposition(response, obstacle, window, obstacle_types, obstacle_params,
                        *, max_layers=32, tolerance=1e-12):
    """Run only the local arithmetic stage, with exactly zero candidate calls."""
    original = np.asarray(response, dtype=np.float64)
    obstacle = np.asarray(obstacle, dtype=bool)
    if (original.ndim != 2 or original.shape != obstacle.shape or
            not np.isfinite(original).all() or np.any(original < -tolerance) or
            np.any(np.abs(original[obstacle]) > tolerance)):
        raise ValueError("invalid local observation")
    free = ~obstacle
    if not free.any():
        return {"status": "skip", "reason": "no_free_pixels"}
    if not np.any(original[free] > tolerance):
        return {"status": "skip", "reason": "all_dark"}
    if np.all(original[free] > tolerance):
        return {"status": "skip", "reason": "all_lit"}
    residual = original.copy()
    masks, strengths = [], []
    conflicts = unknown_regions = 0
    for _ in range(max_layers):
        if float(np.max(residual[free])) <= tolerance:
            break
        regions = response_regions(residual, obstacle, np.asarray(window)[:2],
                                   obstacle_types, obstacle_params, tolerance=tolerance)
        minimum, seeds = minimum_positive_regions(regions.values, tolerance=tolerance)
        if minimum is None:
            break
        seed = min(seeds, key=lambda region: int(regions.pixels[region][0]))
        trace = trace_contribution(regions.values, regions.neighbors, seed, tolerance=tolerance)
        positive, _, _ = _constraints(regions, trace, residual, obstacle, tolerance)
        if not positive.any():
            return {"status": "incomplete", "reason": "empty_layer"}
        updated = residual - minimum * positive
        if float(np.min(updated[free])) < -tolerance:
            return {"status": "incomplete", "reason": "negative_residual"}
        updated[np.abs(updated) <= tolerance] = 0.0
        masks.append(positive)
        strengths.append(float(minimum))
        conflicts += len(trace["conflict"])
        unknown_regions += len(trace["unknown"])
        residual = updated
    if float(np.max(np.abs(residual[free]))) > tolerance:
        return {"status": "incomplete", "reason": "layer_capacity_exceeded"}
    if not masks:
        return {"status": "incomplete", "reason": "no_positive_layer"}
    masks = np.asarray(masks, dtype=bool)
    strengths = np.asarray(strengths, dtype=np.float64)
    error = float(np.max(np.abs(original - np.sum(strengths[:, None, None] * masks, axis=0))))
    if error > max(1e-9, max_layers * tolerance * 2):
        raise ValueError("complete local decomposition does not reconstruct original response")
    return {"status": "complete", "masks": masks, "intensities": strengths,
            "reconstruction_max_error": error, "conflict_regions": conflicts,
            "unknown_regions": unknown_regions, "candidate_renders": 0,
            "physical_single_source_checked": False}


def _splits(dataset, config):
    if hasattr(dataset, "scene_splits"):
        return dataset.scene_splits
    protocol_path = config.get("label_preparation", {}).get("base_split_protocol")
    if protocol_path:
        from train_decomposition import resolve
        return json.loads(resolve(protocol_path).read_text())["scene_splits"]
    # Split identical geometries together even for a standalone release.
    result = {name: [] for name in ("train", "validation", "test")}
    seed = config["label_preparation"].get("seed", 20261028)
    for scene in range(dataset.num_scenes):
        key = dataset.geometry_group(scene)
        value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
        result["train" if value < .8 else "validation" if value < .9 else "test"].append(scene)
    return result


def prepare_mask_dataset(config, dataset=None):
    """Build/reuse tiny per-window labels then contiguous mask training files."""
    from train_decomposition import resolve
    from rind_phase2.regions import check_geometry_backend
    started = time.perf_counter()
    if dataset is None:
        from dataset_update import open_dataset
        dataset = open_dataset(resolve(config["data_root"]))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    options = config["label_preparation"]
    splits = _splits(dataset, config)
    membership = {}
    for split, scenes in splits.items():
        for scene in scenes:
            if int(scene) in membership:
                raise ValueError("scene assigned to several label splits")
            membership[int(scene)] = split
    import inspect
    import rind_phase2.regions as region_module
    import rind_phase2.region_geometry as geometry_module
    contract = {"schema": SCHEMA, "dataset_manifest": dataset.manifest,
                "selection": options, "max_slots": config["model"]["max_slots"],
                "algorithm": "minimum-positive-region BFS; nonnegative residual subtraction",
                "split_sha256": _digest(splits),
                "tracer_sha256": hashlib.sha256(inspect.getsource(trace_contribution).encode()).hexdigest(),
                "regions_sha256": hashlib.sha256(Path(region_module.__file__).read_bytes()).hexdigest(),
                "geometry_sha256": hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest(),
                "label_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = _digest(contract)
    protocol = output / "mask_protocol.json"
    if protocol.is_file() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError("label contract changed; choose a new output_root to preserve previous labels")
    _write_json(protocol, {"signature": signature, **contract})
    print("Local-label geometry preflight:", check_geometry_backend(), flush=True)
    sizes = set(options.get("window_sizes", [16, 32, 64, 128]))
    maximum_views = int(options.get("max_views_per_scene", 8))
    if maximum_views < 1:
        raise ValueError("max_views_per_scene must be positive")
    labels = {name: [] for name in ("train", "validation", "test")}
    statistics = {}
    for split_index, split in enumerate(labels):
        rng = np.random.default_rng(int(options.get("seed", 20261028)) + split_index)
        scenes = np.asarray(splits[split], dtype=np.int64).copy()
        rng.shuffle(scenes)
        target = options["record_counts"][split]
        if target != "all" and (type(target) is not int or target < 1):
            raise ValueError("record_counts must be positive integers or all")
        counts = defaultdict(int)
        for round_index in range(maximum_views):
            for scene_raw in scenes:
                if target != "all" and len(labels[split]) >= int(target):
                    break
                scene = int(scene_raw)
                views = [(int(view), int(size)) for view, size in dataset.iter_scene_views(scene) if int(size) in sizes]
                scene_rng = np.random.default_rng(np.random.SeedSequence([
                    int(options.get("seed", 20261028)), split_index, scene]))
                scene_rng.shuffle(views)
                if round_index >= len(views):
                    continue
                view, size = views[round_index]
                directory = output / "local_labels" / split
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"scene{scene}-view{view}.npz"
                counts["checked"] += 1
                if path.is_file():
                    with np.load(path, allow_pickle=False) as stored:
                        if str(stored["signature"]) != signature:
                            raise ValueError("local label cache belongs to a different protocol")
                        window = stored["window"].tolist()
                        count = int(len(stored["intensities"]))
                    counts["reused"] += 1
                else:
                    geometry = dataset.get_scene(scene)
                    window = np.asarray(geometry["local_views"][view], dtype=np.int64)
                    x, y, side = map(int, window)
                    exact = dataset.get_region(scene, x, y, side)
                    free_response = np.asarray(exact["response"])[~np.asarray(exact["obstacle"], dtype=bool)]
                    if not len(free_response) or not np.any(free_response > 1e-12) or np.all(free_response > 1e-12):
                        reason = "no_free_pixels" if not len(free_response) else "all_dark" if not np.any(free_response > 1e-12) else "all_lit"
                        counts[reason] += 1
                        if counts["checked"] % 500 == 0:
                            print(f"local label scan {split}: completed={len(labels[split])}/{target} counts={dict(counts)}", flush=True)
                        continue
                    result = local_decomposition(
                        exact["response"], exact["obstacle"], window,
                        geometry["obstacle_types"], geometry["obstacle_params"],
                        max_layers=config["model"]["max_slots"],
                        tolerance=options.get("tolerance", 1e-12))
                    if result["status"] != "complete":
                        counts[result.get("reason", "incomplete")] += 1
                        continue
                    count = len(result["intensities"])
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(
                        temporary, signature=np.asarray(signature), response=exact["response"],
                        obstacle=exact["obstacle"], window=window,
                        masks=result["masks"], intensities=result["intensities"],
                        reconstruction_max_error=result["reconstruction_max_error"],
                        conflict_regions=result["conflict_regions"], unknown_regions=result["unknown_regions"])
                    temporary.replace(path)
                    counts["generated"] += 1
                labels[split].append({"scene_id": scene, "view_id": view, "size": size,
                                      "file": str(path.resolve()), "window": list(window),
                                      "layer_count": count})
                if len(labels[split]) == 1 or len(labels[split]) % 100 == 0:
                    print(f"local labels {split}: completed={len(labels[split])}/{target} checked={counts['checked']} | candidate renders=0", flush=True)
            if target != "all" and len(labels[split]) >= int(target):
                break
        statistics[split] = dict(counts)
        if target != "all" and len(labels[split]) < int(target):
            raise RuntimeError(f"{split}: only {len(labels[split])}/{target} complete windows; increase max_views_per_scene")
        print(f"local labels {split} done: actual={len(labels[split])}/{target} counts={dict(counts)}", flush=True)
    index = build_mask_cache(config, labels, output, dataset.manifest["global_size"], signature, statistics)
    print(f"local labels ready | actual counts={index['counts']} | candidate renders=0 | {time.perf_counter() - started:.1f}s | {output / 'mask_records.json'}", flush=True)
    return index


def build_mask_cache(config, labels, output, global_size=1024, signature="", statistics=None):
    output = Path(output)
    cache = output / "mask_cache"
    cache.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in ("train", "validation", "test")}
    scene_membership = {}
    for split, rows in labels.items():
        for row in rows:
            scene = int(row["scene_id"])
            if scene in scene_membership and scene_membership[scene] != split:
                raise ValueError("scene leakage across local label splits")
            scene_membership[scene] = split
    fingerprint = hashlib.sha256(signature.encode())
    for split in records:
        offsets = {"response": 0, "obstacle": 0, "masks": 0}
        streams = {name: (cache / f"{split}_{name}.tmp").open("wb") for name in offsets}
        try:
            for row in labels[split]:
                with np.load(row["file"], allow_pickle=False) as z:
                    response = np.asarray(z["response"], dtype=np.float64)
                    obstacle = np.asarray(z["obstacle"], dtype=bool)
                    masks = np.asarray(z["masks"], dtype=bool)
                    strengths = np.asarray(z["intensities"], dtype=np.float64)
                    window = np.asarray(z["window"], dtype=np.float32)
                if (response.shape != (int(row["size"]), int(row["size"])) or
                        obstacle.shape != response.shape or not np.isfinite(response).all() or
                        window.shape != (3,) or int(window[2]) != int(row["size"]) or
                        not len(strengths) or len(strengths) > config["model"]["max_slots"] or
                        masks.shape != (len(strengths), *response.shape) or
                        not np.isfinite(strengths).all() or np.any(strengths <= 0) or
                        np.any(masks[:, obstacle]) or
                        not np.allclose(np.sum(strengths[:, None, None] * masks, axis=0), response, rtol=0, atol=1e-9)):
                    raise ValueError(f"invalid complete local mask labels: {row['file']}")
                item = {**row, "response_offset": offsets["response"],
                        "obstacle_offset": offsets["obstacle"], "mask_offset": offsets["masks"],
                        "intensities": strengths.tolist(), "window": window.tolist()}
                item.pop("file", None)  # Training remains portable after deleting local label NPZs.
                response.astype("<f4").ravel().tofile(streams["response"])
                packed_obstacle = np.packbits(obstacle.ravel(), bitorder="little")
                packed_masks = np.packbits(masks.reshape(len(masks), -1), bitorder="little", axis=1)
                packed_obstacle.tofile(streams["obstacle"])
                packed_masks.tofile(streams["masks"])
                offsets["response"] += response.size
                offsets["obstacle"] += packed_obstacle.size
                offsets["masks"] += packed_masks.size
                records[split].append(item)
                for array in (response, obstacle, masks, strengths):
                    fingerprint.update(np.ascontiguousarray(array).tobytes())
        finally:
            for stream in streams.values():
                stream.close()
        for key in offsets:
            (cache / f"{split}_{key}.tmp").replace(cache / f"{split}_{key}.{'f32' if key == 'response' else 'u8'}")
    index = {"schema": SCHEMA, "task": "decomposition_only", "global_size": int(global_size),
             "records": records, "counts": {key: len(value) for key, value in records.items()},
             "corpus_sha256": fingerprint.hexdigest(), "statistics": statistics or {},
             "max_slots": config["model"]["max_slots"],
             "candidate_renders": 0, "probability_fields_saved": False,
             "physical_single_source_checked": False}
    _write_json(output / "mask_records.json", index)
    return index


class MaskRecords:
    """Only response, mask, intensity, existence and identity tensors."""
    def __init__(self, entries, cache_root, split, max_slots):
        self.entries = entries
        self.max_slots = max_slots
        self.window_sizes = [int(row["size"]) for row in entries]
        self.candidate_counts = [0] * len(entries)
        self.scene_ids = {int(row["scene_id"]) for row in entries}
        self.maps = {}
        for name, dtype, extension in (("response", "<f4", "f32"), ("obstacle", "u1", "u8"), ("masks", "u1", "u8")):
            path = Path(cache_root) / f"{split}_{name}.{extension}"
            self.maps[name] = np.memmap(path, mode="r", dtype=dtype) if path.stat().st_size else np.empty(0, dtype=dtype)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        side = int(row["size"])
        pixels = side * side
        nbytes = (pixels + 7) // 8
        count = len(row["intensities"])
        offset = row["response_offset"]
        response = np.array(self.maps["response"][offset:offset + pixels]).reshape(side, side)
        offset = row["obstacle_offset"]
        obstacle = np.unpackbits(self.maps["obstacle"][offset:offset + nbytes], bitorder="little")[:pixels].reshape(side, side).astype(bool)
        masks = np.zeros((self.max_slots, side, side), dtype=bool)
        offset = row["mask_offset"]
        packed = self.maps["masks"][offset:offset + count * nbytes].reshape(count, nbytes)
        masks[:count] = np.unpackbits(packed, axis=1, bitorder="little")[:, :pixels].reshape(count, side, side)
        strengths = np.zeros(self.max_slots, dtype=np.float32)
        strengths[:count] = row["intensities"]
        return {"response": response, "obstacle": obstacle, "window": np.asarray(row["window"], dtype=np.float32),
                "teacher_mask": masks, "teacher_intensity": strengths,
                "layer_mask": np.arange(self.max_slots) < count, "count_target": np.int64(count),
                "scene_id": np.int64(row["scene_id"]), "view_id": np.int64(row["view_id"])}
