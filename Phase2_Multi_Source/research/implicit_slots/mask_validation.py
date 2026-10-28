"""Validate mask-only cache metadata before allocating a model or GPU batch."""
from pathlib import Path
import math


def validate_mask_cache(index, cache_root, max_slots):
    """Check complete splits, identities, offsets and exact file lengths.

    Does not re-read bulk arrays. Numerical reconstruction is checked when the
    cache is built, and the training loop rejects nonfinite losses/gradients.
    """
    if index.get("schema") != "rind-local-mask-labels-v1" or index.get("task") != "decomposition_only":
        raise ValueError("mask-only training requires a mask-only cache")
    if index.get("max_slots") != max_slots:
        raise ValueError("cache/model max_slots mismatch")
    world = index.get("global_size")
    if type(world) is not int or world < 1:
        raise ValueError("invalid cached world size")
    if index.get("candidate_renders") != 0 or index.get("probability_fields_saved") is not False:
        raise ValueError("mask-only cache unexpectedly contains position supervision")
    membership = {}
    for split in ("train", "validation", "test"):
        rows = index.get("records", {}).get(split)
        if not isinstance(rows, list) or not rows or index.get("counts", {}).get(split) != len(rows):
            raise ValueError(f"empty or inconsistent {split} records")
        seen = set()
        offsets = {"response": 0, "obstacle": 0, "masks": 0}
        for row in rows:
            scene, view = row["scene_id"], row["view_id"]
            if type(scene) is not int or type(view) is not int or min(scene, view) < 0:
                raise ValueError("invalid scene/view identity")
            if (scene, view) in seen:
                raise ValueError(f"duplicate window in {split}")
            seen.add((scene, view))
            if scene in membership and membership[scene] != split:
                raise ValueError("scene leakage across mask splits")
            membership[scene] = split
            side, window = row["size"], row["window"]
            if (type(side) is not int or side < 1 or not isinstance(window, list) or len(window) != 3 or
                    any(not math.isfinite(float(v)) or float(v) != int(v) for v in window) or
                    window[2] != side or min(window[:2]) < 0 or
                    max(window[0] + side, window[1] + side) > world):
                raise ValueError("invalid cached window coordinates")
            strengths = row["intensities"]
            if (not 1 <= len(strengths) <= max_slots or
                    any(not math.isfinite(float(a)) or a <= 0 for a in strengths) or
                    row.get("layer_count", len(strengths)) != len(strengths)):
                raise ValueError("invalid cached layer count/strength")
            for field, key in (("response_offset", "response"), ("obstacle_offset", "obstacle"), ("mask_offset", "masks")):
                if type(row.get(field)) is not int or row[field] != offsets[key]:
                    raise ValueError(f"non-contiguous {split} {field}")
            pixels, nbytes = side * side, (side * side + 7) // 8
            offsets["response"] += pixels
            offsets["obstacle"] += nbytes
            offsets["masks"] += len(strengths) * nbytes
        for key, elements in offsets.items():
            path = Path(cache_root) / f"{split}_{key}.{'f32' if key == 'response' else 'u8'}"
            expected = elements * (4 if key == "response" else 1)
            if not path.is_file() or path.stat().st_size != expected:
                raise ValueError(f"cache file missing or truncated: {path}; expected {expected} bytes")
