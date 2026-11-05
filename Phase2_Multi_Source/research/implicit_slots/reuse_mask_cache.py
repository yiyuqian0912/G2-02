"""Bind complete existing response/layer labels, without raw data or rendering."""
import json
from pathlib import Path

from mask_validation import validate_mask_cache
from search_cache import bind_cache, digest, immutable_json


def prepare_cached_labels(config):
    from train_decomposition import resolve
    source, output = resolve(config["cache_source_root"]).resolve(), resolve(config["output_root"]).resolve()
    if source == output:
        raise ValueError("new model output must differ from immutable source cache")
    path = source / "mask_records.json"
    if not path.is_file():
        raise FileNotFoundError(f"Complete cached labels missing: {path}. Set cache_source_root to the completed masks_only_100k output; no raw data is required.")
    index = json.loads(path.read_text())
    slots = config["model"]["max_slots"]
    validate_mask_cache(index, source / "mask_cache", index["max_slots"])
    for split, rows in index["records"].items():
        if any(len(row["intensities"]) > slots for row in rows):
            raise ValueError(f"{split} labels exceed {slots} slots; no truncation/filtering is permitted")
    protocol = {"schema": "rind-reused-complete-labels-v1", "source_root": str(source),
                "source_index_sha256": digest(index), "counts": index["counts"],
                "candidate_renders": 0, "raw_data_required": False,
                "teacher_labels_modified": False}
    immutable_json(output / "cache_reuse_protocol.json", protocol)
    result = bind_cache(output, source, index, slots)
    validate_mask_cache(result, output / "mask_cache", slots)
    print(f"Reused complete labels | train/validation/test={index['counts']} | source={source} | candidate renders=0 | raw data not required", flush=True)
    return result
