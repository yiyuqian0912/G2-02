"""Append verified Phase-II residual layers as Phase-I training examples.

Only the old training split is imported.  Validation and test remain the new
single-source experiment's scene-disjoint splits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from rind_dataset import RINDDataset

from .data import PROJECT_ROOT


def _file_hash(path):
    block = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _resolve_root(raw):
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _phase2_project_root(teacher_root):
    for parent in (teacher_root, *teacher_root.parents):
        if (parent / "src/rind_phase2").is_dir() or (parent / "data/raw/phase2-multi-source").is_dir():
            return parent
    return (PROJECT_ROOT.parent / "Phase2_Multi_Source").resolve()


def _resolve_data_root(teacher_root):
    configured = os.environ.get("RIND_MULTI_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    project = _phase2_project_root(teacher_root)
    protocol = teacher_root / "protocol.json"
    if protocol.is_file():
        try:
            raw = json.loads(protocol.read_text()).get("config", {}).get("data_root")
            if raw:
                candidate = Path(raw).expanduser()
                return candidate.resolve() if candidate.is_absolute() else (project / candidate).resolve()
        except (OSError, ValueError):
            pass
    return (project / "data/raw/phase2-multi-source").resolve()


def _load_layer(path, include_cost=False):
    """Load a saved residual layer and expand a legacy sparse world grid."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"layer_response", "candidate_xy", "grid_index", "x_axis", "y_axis",
                    "valid", "probability"}
        if include_cost:
            required.add("physical_cost")
        missing = required - set(archive.files)
        if missing:
            raise ValueError("layer file {} is missing {}".format(path, sorted(missing)))
        response = np.asarray(archive["layer_response"])
        candidate_raw = np.asarray(archive["candidate_xy"], dtype=np.float64)
        grid_index = np.asarray(archive["grid_index"], dtype=np.int64)
        x_axis = np.asarray(archive["x_axis"], dtype=np.float64)
        y_axis = np.asarray(archive["y_axis"], dtype=np.float64)
        valid_raw = np.asarray(archive["valid"], dtype=bool)
        probability_raw = np.asarray(archive["probability"], dtype=np.float64)
        cost_raw = (np.asarray(archive["physical_cost"], dtype=np.float64)
                    if include_cost else None)
    if response.dtype != np.bool_ or response.ndim != 2:
        raise ValueError("layer_response must be a two-dimensional boolean mask: {}".format(path))
    if (x_axis.ndim != 1 or y_axis.ndim != 1 or not len(x_axis) or not len(y_axis) or
            not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all()):
        raise ValueError("invalid candidate axes: {}".format(path))
    full_count = len(x_axis) * len(y_axis)
    count = len(grid_index)
    if (candidate_raw.shape != (count, 2) or valid_raw.shape != (count,) or
            probability_raw.shape != (count,) or
            (include_cost and cost_raw.shape != (count,)) or
            np.any(grid_index < 0) or np.any(grid_index >= full_count) or
            len(np.unique(grid_index)) != count):
        raise ValueError("misaligned layer arrays: {}".format(path))
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    candidate = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.array_equal(candidate_raw, candidate[grid_index]):
        raise ValueError("candidate coordinates do not match grid_index: {}".format(path))
    if (not valid_raw.any() or not np.isfinite(probability_raw).all() or
            np.any(probability_raw < 0) or np.any(probability_raw[~valid_raw] != 0) or
            not np.isclose(probability_raw.sum(), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("invalid layer probability: {}".format(path))
    valid = np.zeros(full_count, dtype=bool)
    probability = np.zeros(full_count, dtype=np.float64)
    valid[grid_index] = valid_raw
    probability[grid_index] = probability_raw
    result = dict(response=response.astype(np.float32), candidate=candidate,
                  valid=valid, probability=probability,
                  candidate_count=int(full_count))
    if include_cost:
        if not np.isfinite(cost_raw[valid_raw]).all():
            raise ValueError("nonfinite valid physical cost: {}".format(path))
        result["minimum_cost"] = float(cost_raw[valid_raw].min())
    return result


def _compatible_report(report, expected):
    if report.get("status") != "complete" or report.get("partial_combined_distribution"):
        return False, "incomplete_window"
    if not np.isclose(float(report.get("temperature", np.nan)),
                      float(expected["temperature"]), rtol=0, atol=1e-12):
        return False, "temperature_differs"
    physics = report.get("phase1_cost", {})
    for key in ("alpha", "beta", "boundary_lambda", "boundary_sigma"):
        if not np.isclose(float(physics.get(key, np.nan)), float(expected[key]),
                          rtol=0, atol=1e-12):
            return False, "{}_differs".format(key)
    return True, None


def build_layer_cache_index(cache_roots, output_path, expected, maximum_cost=0.01):
    """Create or verify a portable index of compatible old *training* layers."""
    roots = [_resolve_root(root) for root in cache_roots]
    source_state = []
    for root in roots:
        manifest = root / "records.json"
        source_state.append(dict(root=str(root), exists=manifest.is_file(),
                                 records_sha256=_file_hash(manifest) if manifest.is_file() else None))
    contract = dict(schema="phase2-layer-append-v1", sources=source_state,
                    expected={key: float(expected[key]) for key in
                              ("temperature", "alpha", "beta", "boundary_lambda", "boundary_sigma")},
                    maximum_cost=float(maximum_cost), split_policy="old-train-to-new-train-only")
    output_path = Path(output_path)
    if output_path.is_file():
        saved = json.loads(output_path.read_text())
        if saved.get("contract") == contract:
            return saved

    entries, sources = [], []
    for state in source_state:
        root = Path(state["root"])
        if not state["exists"]:
            sources.append({**state, "status": "missing", "accepted_layers": 0})
            continue
        manifest = json.loads((root / "records.json").read_text())
        records = manifest.get("records", {}).get("train")
        if not isinstance(records, list):
            raise ValueError("Phase-II cache has no records.train list: {}".format(root))
        data_root = _resolve_data_root(root)
        if not (data_root / "manifest.json").is_file():
            raise FileNotFoundError("Phase-II data for appended cache is missing: {}".format(data_root))
        dataset = RINDDataset(data_root)
        counts = dict(windows=len(records), accepted_windows=0, accepted_layers=0,
                      incomplete_windows=0, incompatible_windows=0,
                      high_cost_layers=0, constant_layers=0)
        reasons = {}
        for window in records:
            directory = root / window["directory"]
            report_path = directory / "report.json"
            if not report_path.is_file():
                counts["incomplete_windows"] += 1
                continue
            report = json.loads(report_path.read_text())
            compatible, reason = _compatible_report(report, expected)
            if not compatible:
                counts["incomplete_windows" if reason == "incomplete_window" else "incompatible_windows"] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            scene, view = int(window["scene_id"]), int(window["view_id"])
            observation = dataset.get_view(scene, view)
            obstacle = np.asarray(observation["obstacle"], dtype=bool)
            if int(observation["view"][2]) != int(window["size"]):
                raise ValueError("Phase-II window size mismatch: {}".format(directory))
            accepted_in_window = 0
            for step in range(1, int(report.get("layers_recovered", 0)) + 1):
                path = directory / "layer-{:02d}.npz".format(step)
                layer = _load_layer(path, include_cost=True)
                if layer["response"].shape != obstacle.shape or np.any(layer["response"].astype(bool)[obstacle]):
                    raise ValueError("layer response and observation are misaligned: {}".format(path))
                free_values = layer["response"].astype(bool)[~obstacle]
                if not len(free_values) or not free_values.any() or free_values.all():
                    counts["constant_layers"] += 1
                    continue
                if layer["minimum_cost"] > maximum_cost:
                    counts["high_cost_layers"] += 1
                    continue
                entries.append(dict(source_root=str(root), data_root=str(data_root),
                                    scene_id=scene, view_id=view, size=int(window["size"]),
                                    step=step, layer_path=str(path.relative_to(root)),
                                    candidate_count=layer["candidate_count"],
                                    minimum_physical_cost=layer["minimum_cost"]))
                accepted_in_window += 1
            if accepted_in_window:
                counts["accepted_windows"] += 1
                counts["accepted_layers"] += accepted_in_window
        sources.append({**state, "status": "accepted", "data_root": str(data_root),
                        "counts": counts, "rejection_reasons": reasons})
    index = dict(contract=contract, entries=entries, sources=sources,
                 accepted_layers=len(entries))
    _json(output_path, index)
    return index


class LayerCacheRecords:
    """Lazy dataset exposing old residual layers through the Phase-I contract."""
    def __init__(self, entries, normalization_support="world", cache_in_memory=False):
        if normalization_support not in ("world", "geometry"):
            raise ValueError("unknown normalization support")
        self.entries = list(entries)
        self.normalization_support = normalization_support
        self.window_sizes = [int(row["size"]) for row in self.entries]
        self.candidate_counts = [int(row["candidate_count"]) for row in self.entries]
        self.scene_ids = {"phase2:{}:{}".format(row["source_root"], row["scene_id"])
                          for row in self.entries}
        self.config_ids = {"phase2-residual-layer:" + row["source_root"] for row in self.entries}
        self._datasets = {}
        self.cache = [None] * len(self.entries) if cache_in_memory else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]
        row = self.entries[index]
        data_root = row["data_root"]
        if data_root not in self._datasets:
            self._datasets[data_root] = RINDDataset(data_root)
        observation = self._datasets[data_root].get_view(int(row["scene_id"]), int(row["view_id"]))
        layer = _load_layer(Path(row["source_root"]) / row["layer_path"])
        valid = layer["valid"]
        result = dict(response=layer["response"],
                      obstacle=np.asarray(observation["obstacle"], dtype=bool),
                      window=np.asarray(observation["view"], dtype=np.int64),
                      candidate_xy=layer["candidate"], valid=valid,
                      support=(valid.copy() if self.normalization_support == "geometry"
                               else np.ones_like(valid)),
                      teacher_prob=layer["probability"])
        if self.cache is not None:
            self.cache[index] = result
        return result
