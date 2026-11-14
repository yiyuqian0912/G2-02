"""Read-only source caches and shared, fixed-cohort architecture-search views."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

from mask_validation import validate_mask_cache


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"existing search artifact differs; choose a new output_root: {path}")
    write_json(path, value)


def capacity_audit(index, capacities):
    """Capacity eligibility uses ALL train/development labels, never test labels."""
    splits = {}
    for split in ("train", "validation"):
        rows = index["records"][split]
        hist = Counter(len(row["intensities"]) for row in rows)
        splits[split] = {"windows": len(rows), "scenes": len({r["scene_id"] for r in rows}),
                         "max_layers": max(hist), "layer_histogram": dict(sorted(hist.items()))}
    candidates = []
    for slots in capacities:
        overflow = {split: sum(len(r["intensities"]) > slots for r in index["records"][split])
                    for split in splits}
        candidates.append({"slots": slots, "eligible": not any(overflow.values()),
                           "overflow_windows": overflow,
                           "coverage": {split: 1 - n / splits[split]["windows"] for split, n in overflow.items()}})
    return {"splits": splits, "candidates": candidates,
            "policy": "zero train/development overflow; no target truncation or window filtering; test not used"}


def stratified_subset(rows, count, seed):
    """Proportional strata by label count and native window size, fixed seed."""
    if count >= len(rows):
        return list(rows)
    if count < 1:
        raise ValueError("pilot sample count must be positive")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(len(row["intensities"]), row["size"])].append(i)
    keys = sorted(groups)
    if count < len(keys):
        raise ValueError("pilot record count must cover every layer-count/window-size stratum")
    quotas = {k: count * len(groups[k]) // len(rows) for k in keys}
    remaining = count - sum(quotas.values())
    order = sorted(keys, key=lambda k: (-(count * len(groups[k]) % len(rows)), k))
    for key in order[:remaining]:
        quotas[key] += 1
    # Keep every native window size represented, including rare large windows;
    # otherwise the GPU batch probe could underestimate full-fit memory needs.
    for key in keys:
        if quotas[key] == 0:
            donor = max((k for k in keys if quotas[k] > 1),
                        key=lambda k: (quotas[k] - count * len(groups[k]) / len(rows), quotas[k]))
            quotas[donor] -= 1
            quotas[key] = 1
    rng = random.Random(seed)
    selected = []
    for key in keys:
        selected.extend(rng.sample(groups[key], quotas[key]))
    return [rows[i] for i in sorted(selected)]


def make_pilot(source, index, output, counts, seed):
    """Copy packed bytes ONCE for all trials; never render or change any label."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = {split: stratified_subset(index["records"][split], counts[split], seed + i)
                for i, split in enumerate(("train", "validation"))}
    manifest = {"seed": seed, "strata": ["effective_layer_count", "window_size"],
                "allocation": "proportional; minimum one per nonempty stratum",
                "source_corpus_sha256": index["corpus_sha256"],
                "windows": {split: [{"scene_id": r["scene_id"], "view_id": r["view_id"],
                                      "size": r["size"], "layers": len(r["intensities"])} for r in rows]
                            for split, rows in selected.items()}}
    immutable_json(output / "sample_manifest.json", manifest)
    final = output / "mask_records.json"
    if final.exists():
        result = json.loads(final.read_text())
        if result.get("selection_sha256") != digest(manifest):
            raise ValueError("pilot sample manifest differs")
        validate_mask_cache(result, output / "mask_cache", result["max_slots"], splits=("train", "validation"))
        return result
    temporary = output / "mask_cache.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    records = {"train": [], "validation": [], "test": []}
    content = hashlib.sha256(digest(manifest).encode())
    fields = (("response", "f32", "response_offset", 4),
              ("obstacle", "u8", "obstacle_offset", 1), ("masks", "u8", "mask_offset", 1))
    for split, rows in selected.items():
        offsets = {name: 0 for name, *_ in fields}
        streams = {name: (source / "mask_cache" / f"{split}_{name}.{suffix}").open("rb")
                   for name, suffix, *_ in fields}
        targets = {name: (temporary / f"{split}_{name}.{suffix}").open("wb") for name, suffix, *_ in fields}
        try:
            for row in rows:
                item = copy.deepcopy(row)
                pixels, packed = row["size"] ** 2, (row["size"] ** 2 + 7) // 8
                lengths = {"response": pixels, "obstacle": packed, "masks": len(row["intensities"]) * packed}
                for name, _, field, bytes_per in fields:
                    streams[name].seek(row[field] * bytes_per)
                    data = streams[name].read(lengths[name] * bytes_per)
                    if len(data) != lengths[name] * bytes_per:
                        raise ValueError("source cache was truncated during pilot copy")
                    targets[name].write(data)
                    content.update(data)
                    item[field] = offsets[name]
                    offsets[name] += lengths[name]
                records[split].append(item)
        finally:
            for stream in (*streams.values(), *targets.values()):
                stream.close()
    cache = output / "mask_cache"
    if cache.exists():
        shutil.rmtree(cache)
    temporary.replace(cache)
    result = {**copy.deepcopy(index), "records": records,
              "counts": {k: len(v) for k, v in records.items()}, "statistics": {},
              "corpus_sha256": content.hexdigest(), "selection_sha256": digest(manifest)}
    result.pop("expansion", None)
    validate_mask_cache(result, cache, result["max_slots"], splits=("train", "validation"))
    write_json(final, result)
    return result


def bind_cache(output, source, index, slots):
    """A model-capacity view shares immutable cache bytes via a runtime symlink."""
    output, source = Path(output), Path(source).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "mask_cache"
    target = (source / "mask_cache").resolve()
    if cache.exists() or cache.is_symlink():
        if not cache.is_symlink() or cache.resolve() != target:
            raise ValueError(f"cache view points to another dataset: {cache}")
    else:
        cache.symlink_to(target, target_is_directory=True)
    result = {**copy.deepcopy(index), "max_slots": slots,
              "cache_view": {"source": str(source), "source_max_slots": index["max_slots"],
                             "source_corpus_sha256": index["corpus_sha256"]}}
    immutable_json(output / "mask_records.json", result)
    return result
