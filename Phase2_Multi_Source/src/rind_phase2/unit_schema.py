"""Explicit on-disk contract for this dataset variant."""
import json
from pathlib import Path

import numpy as np

ARRAYS = ("drivers", "driver_counts", "driver_strengths", "obstacle_counts",
          "obstacle_types", "obstacle_params", "obstacle_group_ids", "obstacle_bits",
          "visibility_bits", "local_views", "view_counts", "reference_sources",
          "valid_origin_counts")
OPTIONAL_FILES = ("generation_stats.json", "validation_report.json", "build_provenance.json")


def validate_manifest(m):
    if (not isinstance(m, dict) or m.get("schema_version") != 3 or
            m.get("variant") != "multi-source-unit-random" or
            m.get("view_sampling") != "random-informative" or
            m.get("driver_strength_mode") != "fixed-one" or
            m.get("candidate_layout") != "scene-source-sets" or
            m.get("response_type") != "additive-driver-intensity"):
        raise ValueError("Expected unit-strength multi-source data with informative random windows")
    for key in ("num_scenes", "max_drivers", "view_capacity", "candidate_capacity", "obstacle_capacity"):
        if type(m.get(key)) is not int or m[key] <= 0:
            raise ValueError(f"Invalid manifest field: {key}")
    source_range = m.get("driver_count_range")
    if (not isinstance(source_range, list) or len(source_range) != 2 or
            any(type(value) is not int for value in source_range) or
            not 1 <= source_range[0] <= source_range[1] == m["max_drivers"]):
        raise ValueError("Invalid source count range")
    if (m["max_drivers"] > 4 or m["obstacle_capacity"] > 255 or
            m["candidate_capacity"] != 10 or m["view_capacity"] > 65535 or
            m.get("global_size") != 1024 or m.get("window_sizes") != [64, 128, 256, 512] or
            m.get("source_free") is not True or m.get("bit_order") != "little" or
            m.get("cell_sampling") != "cell-center" or m.get("array_indexing") != "yx"):
        raise ValueError("Unsupported window, coordinate, source or sampling semantics")


def array_specs(m):
    n, k, v, c, w, p = (m["num_scenes"], m["max_drivers"], m["view_capacity"],
                        m["candidate_capacity"], m["global_size"], m["obstacle_capacity"])
    bits = (w * w + 7) // 8
    return {"drivers": ("float64", (n, k, 2)), "driver_counts": ("uint8", (n,)),
            "driver_strengths": ("float64", (n, k)), "obstacle_counts": ("uint8", (n,)),
            "obstacle_types": ("uint8", (n, p)), "obstacle_params": ("float64", (n, p, 32)),
            "obstacle_group_ids": ("uint16", (n, p)), "obstacle_bits": ("uint8", (n, bits)),
            "visibility_bits": ("uint8", (n, k, bits)), "local_views": ("uint16", (n, v, 3)),
            "view_counts": ("uint16", (n,)), "reference_sources": ("float64", (n, c, k, 2)),
            "valid_origin_counts": ("uint32", (n, 4))}


def validate_arrays(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    validate_manifest(m)
    arrays = {}
    for name, (dtype, shape) in array_specs(m).items():
        a = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if a.shape != shape or a.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid array shape or dtype: {name}")
        arrays[name] = a
    for start in range(0, m["num_scenes"], 128):
        a = {name: values[start:start + 128] for name, values in arrays.items()
             if name not in ("visibility_bits", "obstacle_bits", "obstacle_params")}
        counts = a["driver_counts"]
        active = np.arange(m["max_drivers"])[None, :] < counts[:, None]
        if (np.any(counts < m["driver_count_range"][0]) or np.any(counts > m["max_drivers"]) or
                not np.array_equal(a["driver_strengths"], active.astype(np.float64)) or
                not np.isfinite(a["drivers"]).all() or np.any(a["drivers"] < 0) or
                np.any(a["drivers"] > m["global_size"]) or
                np.any(a["view_counts"] != m["view_capacity"]) or
                np.any(a["obstacle_counts"] > m["obstacle_capacity"]) or
                not np.isfinite(a["reference_sources"]).all() or
                np.any(a["reference_sources"] < 0) or np.any(a["reference_sources"] > m["global_size"]) or
                not np.array_equal(a["reference_sources"][:, 0], a["drivers"])):
            raise ValueError("Invalid scene, strength, window or reference values")
        views = a["local_views"].astype(np.int64)
        if (not np.isin(views[:, :, 2], m["window_sizes"]).all() or
                np.any(views[:, :, 0] + views[:, :, 2] > m["global_size"]) or
                np.any(views[:, :, 1] + views[:, :, 2] > m["global_size"])):
            raise ValueError("Window outside the world or unsupported window size")
    return {"num_scenes": m["num_scenes"], "num_views": int(arrays["view_counts"].sum()),
            "references_per_view": m["candidate_capacity"]}
