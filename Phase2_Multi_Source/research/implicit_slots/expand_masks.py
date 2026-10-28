"""Append cheap local labels; preserve the previous train prefix and locked evaluation."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

import mask_only_data as labels
from mask_validation import validate_mask_cache


def _digest_arrays(arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def _base_and_contract(config, dataset):
    from train_decomposition import resolve
    base = resolve(config["label_preparation"]["reuse_from"]).resolve()
    output = resolve(config["output_root"]).resolve()
    if base == output:
        raise ValueError("expanded output_root must differ from reuse_from")
    index = json.loads((base / "mask_records.json").read_text())
    validate_mask_cache(index, base / "mask_cache", config["model"]["max_slots"])
    protocol = json.loads((base / "mask_protocol.json").read_text())
    # Preserve the current corpus and split assignment. The old 20k config
    # already reads the combined 110k data release.
    if protocol["dataset_manifest"] != dataset.manifest:
        raise ValueError("dataset release differs from base cache; install/use the same merged dataset")
    splits = labels._splits(dataset, config)
    if labels._digest(splits) != protocol["split_sha256"]:
        raise ValueError("scene split differs from base cache")
    import rind_phase2.regions as regions
    import rind_phase2.region_geometry as geometry
    expected = {
        "label_code_sha256": hashlib.sha256(Path(labels.__file__).read_bytes()).hexdigest(),
        "tracer_sha256": hashlib.sha256(inspect.getsource(labels.trace_contribution).encode()).hexdigest(),
        "regions_sha256": hashlib.sha256(Path(regions.__file__).read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest(),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"base label algorithm differs ({key}); do not mix label rules")
    options, previous = config["label_preparation"], protocol["selection"]
    for key, default in (("seed", 20261028), ("tolerance", 1e-12)):
        if options.get(key, default) != previous.get(key, default):
            raise ValueError(f"label_preparation.{key} differs from base labels")
    for split in ("validation", "test"):
        if options["record_counts"][split] != index["counts"][split]:
            raise ValueError(f"{split} count must remain locked at {index['counts'][split]}")
    target = options["record_counts"]["train"]
    if target != "all" and (type(target) is not int or target < index["counts"]["train"]):
        raise ValueError("expanded training target must include every base training window")
    if int(options.get("max_views_per_scene", 32)) < 1 or int(options.get("chunk_size", 64)) < 1:
        raise ValueError("max_views_per_scene and chunk_size must be positive")
    membership = {}
    for split, scenes in splits.items():
        for scene in scenes:
            if int(scene) in membership:
                raise ValueError("scene appears in multiple dataset splits")
            membership[int(scene)] = split
    for split, rows in index["records"].items():
        if any(membership.get(row["scene_id"]) != split for row in rows):
            raise ValueError("base records no longer match the dataset scene split")
    contract = {"schema": "rind-expanded-local-labels-v1", "base_corpus_sha256": index["corpus_sha256"],
                "base_records_sha256": labels._digest(index["records"]),
                "base_protocol_signature": protocol["signature"], "max_slots": index["max_slots"],
                "selection": options, "dataset_manifest": dataset.manifest}
    return base, output, index, protocol, splits, contract


def prepare_expanded(config, dataset=None, *, stop_after_chunks=None):
    """Prevent two custom tmux sessions from writing the same append cache."""
    import fcntl
    from train_decomposition import resolve
    output = resolve(config["output_root"]).resolve()
    if output == resolve(config["label_preparation"]["reuse_from"]).resolve():
        raise ValueError("expanded output_root must differ from reuse_from")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".expansion.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is preparing this expanded output_root") from exc
        return _prepare_expanded(config, dataset, stop_after_chunks=stop_after_chunks)


def _prepare_expanded(config, dataset=None, *, stop_after_chunks=None):
    """Use compressed chunks for resumable labels; one final contiguous append.

    stop_after_chunks is a test hook for interruption/restart validation.
    No previous local_labels/*.npz files are required.
    """
    from train_decomposition import resolve
    from rind_phase2.regions import check_geometry_backend
    if dataset is None:
        from dataset_update import open_dataset
        dataset = open_dataset(resolve(config["data_root"]))
    started = time.perf_counter()
    base, output, old, old_protocol, splits, contract = _base_and_contract(config, dataset)
    signature = labels._digest(contract)
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "expansion_protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text())["signature"] != signature:
        raise ValueError("expansion configuration changed; choose a new output_root")
    labels._write_json(protocol_file, {"signature": signature, **contract})
    final_index = output / "mask_records.json"
    if final_index.exists():
        result = json.loads(final_index.read_text())
        if result.get("expansion", {}).get("signature") != signature:
            raise ValueError("existing expanded index belongs to a different preparation")
        validate_mask_cache(result, output / "mask_cache", config["model"]["max_slots"])
        print(f"expanded labels already complete: {result['counts']}; candidate renders=0", flush=True)
        return result
    print("Local-label geometry preflight:", check_geometry_backend(), flush=True)
    directory = output / "append_chunks"
    directory.mkdir(exist_ok=True)
    chunks = sorted(directory.glob("chunk-*.npz"))
    seen = {(row["scene_id"], row["view_id"]) for row in old["records"]["train"]}
    added, cursor, statistics, chunk_hashes = [], [0, 0], Counter(), []
    for number, path in enumerate(chunks):
        if path.name != f"chunk-{number:06d}.npz":
            raise ValueError("append chunk sequence contains a gap")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
        if meta["signature"] != signature:
            raise ValueError("append chunk protocol mismatch")
        for row in meta["records"]:
            identity = (row["scene_id"], row["view_id"])
            if identity in seen:
                raise ValueError("duplicate base/new training window in append chunks")
            seen.add(identity)
            added.append(row)
        cursor, statistics = meta["cursor"], Counter(meta["statistics"])
        chunk_hashes.append(meta["arrays_sha256"])
    options = config["label_preparation"]
    target = options["record_counts"]["train"]
    base_count = len(old["records"]["train"])
    seed, tolerance = int(options.get("seed", 20261028)), float(options.get("tolerance", 1e-12))
    sizes = set(options.get("window_sizes", [16, 32, 64, 128]))
    scenes = np.asarray(splits["train"], dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(scenes)
    pending, arrays = [], {"response": [], "obstacle": [], "masks": []}
    # Cache at most max_views metadata rows per scene. Avoid re-filtering and
    # reshuffling all 256 release windows on every residual-label scan round.
    view_orders = {}
    lengths = {key: 0 for key in arrays}
    last_saved_checked = statistics["checked"]
    chunks_written = 0
    print(f"expand train | reused={base_count}, required actual total={target}, validation/test locked={old['counts']['validation']}/{old['counts']['test']} | candidate renders=0", flush=True)

    def flush(next_cursor):
        nonlocal pending, arrays, lengths, chunks_written, last_saved_checked
        payload = {key: np.concatenate(values) if values else np.empty(0, dtype='<f4' if key == 'response' else 'u1')
                   for key, values in arrays.items()}
        content_hash = _digest_arrays(payload.values())
        meta = {"signature": signature, "records": pending, "cursor": list(next_cursor),
                "statistics": dict(statistics), "arrays_sha256": content_hash}
        path = directory / f"chunk-{len(chunks):06d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(meta, allow_nan=False)), **payload)
        temporary.replace(path)
        chunks.append(path)
        added.extend(pending)
        chunk_hashes.append(content_hash)
        pending, arrays = [], {"response": [], "obstacle": [], "masks": []}
        lengths = {key: 0 for key in arrays}
        chunks_written += 1
        last_saved_checked = statistics["checked"]
        if stop_after_chunks is not None and chunks_written >= stop_after_chunks:
            raise InterruptedError("test interruption after committed chunk")

    def enough():
        return target != "all" and base_count + len(added) + len(pending) >= target

    for round_index in range(int(cursor[0]), int(options.get("max_views_per_scene", 32))):
        if enough():
            break
        first_scene = int(cursor[1]) if round_index == int(cursor[0]) else 0
        for scene_offset in range(first_scene, len(scenes)):
            if enough():
                break
            scene = int(scenes[scene_offset])
            if scene not in view_orders:
                views = [(int(view), int(size)) for view, size in dataset.iter_scene_views(scene) if int(size) in sizes]
                np.random.default_rng(np.random.SeedSequence([seed, 0, scene])).shuffle(views)
                view_orders[scene] = np.asarray(views[:int(options.get("max_views_per_scene", 32))], dtype=np.int32).reshape(-1, 2)
            views = view_orders[scene]
            if round_index >= len(views):
                continue
            view, side = map(int, views[round_index])
            if (scene, view) in seen:
                continue
            statistics["checked"] += 1
            geometry = dataset.get_scene(scene)
            window = np.asarray(geometry["local_views"][view], dtype=np.int64)
            exact = dataset.get_region(scene, int(window[0]), int(window[1]), int(window[2]))
            result = labels.local_decomposition(
                exact["response"], exact["obstacle"], window,
                geometry["obstacle_types"], geometry["obstacle_params"],
                max_layers=config["model"]["max_slots"], tolerance=tolerance)
            if result["status"] != "complete":
                statistics[result["reason"]] += 1
            else:
                masks, response, obstacle = result["masks"], np.asarray(exact["response"]), np.asarray(exact["obstacle"], bool)
                payload = {"response": response.astype('<f4').ravel(),
                           "obstacle": np.packbits(obstacle.ravel(), bitorder="little"),
                           "masks": np.packbits(masks.reshape(len(masks), -1), axis=1, bitorder="little").ravel()}
                row = {"scene_id": scene, "view_id": view, "size": side, "window": window.tolist(),
                       "layer_count": len(masks), "intensities": result["intensities"].tolist(),
                       "response_offset": lengths["response"], "obstacle_offset": lengths["obstacle"],
                       "mask_offset": lengths["masks"], "conflict_regions": result["conflict_regions"],
                       "unknown_regions": result["unknown_regions"],
                       "reconstruction_max_error": result["reconstruction_max_error"]}
                for key, array in payload.items():
                    arrays[key].append(array)
                    lengths[key] += array.size
                pending.append(row)
                seen.add((scene, view))
                statistics["generated"] += 1
            next_cursor = [round_index, scene_offset + 1]
            if (len(pending) >= int(options.get("chunk_size", 64)) or
                    statistics["checked"] - last_saved_checked >= 500 or enough()):
                flush(next_cursor)
                print(f"expand train | total={base_count + len(added)}/{target}, new={len(added)}, checked={statistics['checked']}, rejected={statistics['checked'] - statistics['generated']} | candidate renders=0 | {time.perf_counter() - started:.1f}s", flush=True)
        # Following rounds start at the beginning; the saved cursor already
        # encodes the last completed scene in its original round.
    if pending or statistics["checked"] != last_saved_checked:
        flush([int(options.get("max_views_per_scene", 32)), 0])
    if target != "all" and base_count + len(added) < target:
        raise RuntimeError(f"only {base_count + len(added)}/{target} complete windows; max_views_per_scene exhausted")
    result = _assemble(output, base, old, chunks, added, signature, statistics, chunk_hashes)
    component_counts, source_counts = Counter(), Counter()
    metadata_cache = {}
    for row in result["records"]["train"]:
        scene = row["scene_id"]
        if scene not in metadata_cache:
            info = dataset.source_metadata(scene) if hasattr(dataset, "source_metadata") else {}
            k = int(dataset.driver_counts[scene]) if hasattr(dataset, "driver_counts") else -1
            metadata_cache[scene] = (info.get("component_root", "unspecified"), k)
        component, k = metadata_cache[scene]
        component_counts[component] += 1
        if k >= 0:
            source_counts[str(k)] += 1
    result["expansion"].update(train_windows_by_component=dict(component_counts),
                               train_windows_by_true_source_count=dict(source_counts),
                               train_scene_configurations=len(metadata_cache),
                               train_windows_by_size=dict(Counter(str(row["size"]) for row in result["records"]["train"])),
                               train_windows_by_effective_layer_count=dict(Counter(str(len(row["intensities"])) for row in result["records"]["train"])),
                               preparation_seconds=time.perf_counter() - started)
    derived_protocol = {key: value for key, value in old_protocol.items() if key != "signature"}
    derived_protocol.update(selection=options, expanded_from=old_protocol["signature"])
    labels._write_json(output / "mask_protocol.json", {"signature": labels._digest(derived_protocol), **derived_protocol})
    labels._write_json(output / "mask_records.json", result)
    print(f"expanded labels ready | counts={result['counts']} | reused={base_count}, newly generated={len(added)} | components={dict(component_counts)} | candidate renders=0", flush=True)
    return result


def _assemble(output, base, old, chunks, added, signature, statistics, chunk_hashes):
    cache = output / "mask_cache"
    cache.mkdir(exist_ok=True)
    records = copy.deepcopy(old["records"])
    for split in ("train", "validation", "test"):
        for key, suffix in (("response", "f32"), ("obstacle", "u8"), ("masks", "u8")):
            name = f"{split}_{key}.{suffix}"
            shutil.copyfile(base / "mask_cache" / name, cache / (name + '.tmp'))
    streams = {key: (cache / f"train_{key}.{'f32' if key == 'response' else 'u8'}.tmp").open('ab')
               for key in ("response", "obstacle", "masks")}
    offsets = {key: stream.tell() // (4 if key == 'response' else 1) for key, stream in streams.items()}
    try:
        for path in chunks:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                payload = {key: np.asarray(z[key]) for key in streams}
            if _digest_arrays(payload.values()) != meta["arrays_sha256"]:
                raise ValueError(f"append chunk checksum mismatch: {path}")
            for source in meta["records"]:
                row = dict(source)
                for field, key in (("response_offset", "response"), ("obstacle_offset", "obstacle"), ("mask_offset", "masks")):
                    row[field] += offsets[key]
                records["train"].append(row)
            for key, array in payload.items():
                array.tofile(streams[key])
                offsets[key] += array.size
    finally:
        for stream in streams.values():
            stream.close()
    for path in cache.glob('*.tmp'):
        path.replace(path.with_suffix(''))
    index = {**copy.deepcopy(old), "records": records,
             "counts": {key: len(rows) for key, rows in records.items()},
             "statistics": {**copy.deepcopy(old.get("statistics", {})), "train": dict(statistics)},
             "corpus_sha256": labels._digest({"base": old['corpus_sha256'], "chunks": chunk_hashes}),
             "expansion": {"signature": signature, "base_root": str(base),
                           "base_train_windows": len(old['records']['train']), "new_train_windows": len(added),
                           "frozen_evaluation_record_sha256": labels._digest({key: records[key] for key in ('validation', 'test')})}}
    validate_mask_cache(index, cache, old["max_slots"])
    return index
