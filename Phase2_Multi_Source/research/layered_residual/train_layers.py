"""Train one Phase-I-style locator per binary residual layer and test multi-layer reuse."""
from __future__ import print_function

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
PHASE2_ROOT = PACKAGE_ROOT / "Phase2_Multi_Source"
PHASE1_ROOT = PACKAGE_ROOT / "Phase1_Single_Source_Single_Intensity"
for item in (PHASE1_ROOT / "src", PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset", HERE.parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    block = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def _source_hashes():
    paths = (
        HERE,
        HERE.parent / "layered_residual.py",
        PHASE1_ROOT / "src/rind_phase1/model.py",
        PHASE1_ROOT / "src/rind_phase1/train.py",
        PHASE2_ROOT / "src/rind_phase2/data.py",
    )
    return {str(path.relative_to(PACKAGE_ROOT)): _file_digest(path) for path in paths}


def _array_digest(array):
    value = np.ascontiguousarray(array)
    block = hashlib.sha256()
    block.update(str(value.dtype).encode())
    block.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    block.update(value.tobytes())
    return block.hexdigest()


def _load_layer(path, include_cost=False):
    """Load one saved layer and expand legacy sparse grids to the full world grid."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"layer_response", "candidate_xy", "grid_index", "x_axis", "y_axis",
                    "valid", "probability"}
        if include_cost:
            required.add("physical_cost")
        missing = required - set(archive.files)
        if missing:
            raise ValueError("layer file {} is missing {}".format(path, sorted(missing)))
        response_raw = np.asarray(archive["layer_response"])
        candidate_raw = np.asarray(archive["candidate_xy"], dtype=np.float64)
        grid_index = np.asarray(archive["grid_index"], dtype=np.int64)
        x_axis = np.asarray(archive["x_axis"], dtype=np.float64)
        y_axis = np.asarray(archive["y_axis"], dtype=np.float64)
        valid_raw = np.asarray(archive["valid"], dtype=bool)
        probability_raw = np.asarray(archive["probability"], dtype=np.float64)
        cost_raw = (np.asarray(archive["physical_cost"], dtype=np.float64)
                    if include_cost else None)
    if response_raw.dtype != np.bool_ or response_raw.ndim != 2:
        raise ValueError("layer_response must be a two-dimensional boolean mask: {}".format(path))
    if (x_axis.ndim != 1 or y_axis.ndim != 1 or not len(x_axis) or not len(y_axis) or
            not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all()):
        raise ValueError("invalid grid axes: {}".format(path))
    full_count = len(x_axis) * len(y_axis)
    count = len(grid_index)
    if (candidate_raw.shape != (count, 2) or valid_raw.shape != (count,) or
            probability_raw.shape != (count,) or
            (include_cost and cost_raw.shape != (count,)) or
            np.any(grid_index < 0) or np.any(grid_index >= full_count) or
            len(np.unique(grid_index)) != count):
        raise ValueError("misaligned candidate arrays: {}".format(path))
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    candidate = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.array_equal(candidate_raw, candidate[grid_index]):
        raise ValueError("candidate coordinates do not match grid_index: {}".format(path))
    if (not np.isfinite(probability_raw).all() or np.any(probability_raw < 0) or
            not valid_raw.any() or
            (include_cost and not np.isfinite(cost_raw[valid_raw]).all()) or
            not np.isclose(probability_raw.sum(), 1.0, rtol=0, atol=1e-10) or
            np.any(probability_raw[~valid_raw] != 0)):
        raise ValueError("invalid physical target arrays: {}".format(path))
    valid = np.zeros(full_count, dtype=bool)
    probability = np.zeros(full_count, dtype=np.float64)
    valid[grid_index] = valid_raw
    probability[grid_index] = probability_raw
    result = {
        "response": response_raw.astype(np.float32), "candidate": candidate,
        "valid": valid, "probability": probability, "grid_index": grid_index,
        "x_axis": x_axis, "y_axis": y_axis, "stored_candidate_count": count,
        "full_candidate_count": full_count,
    }
    if include_cost:
        cost = np.full(full_count, np.inf, dtype=np.float64)
        cost[grid_index] = cost_raw
        result["cost"] = cost
    return result


def _progress(root, event, message, **fields):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] {}".format(stamp, message), flush=True)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": stamp, "event": event, **fields}, allow_nan=False) + "\n")


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = {"data_root", "teacher_experiment_root", "output_root", "layer_filter",
                "training", "evaluation", "target_semantics"}
    if not required <= set(config):
        raise ValueError("missing config fields: {}".format(sorted(required - set(config))))
    layer_filter = config["layer_filter"]
    maximum = layer_filter.get("maximum_training_physical_cost")
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum < 0:
        raise ValueError("maximum_training_physical_cost must be finite and nonnegative")
    for key in ("skip_all_zero", "skip_all_one"):
        if type(layer_filter.get(key)) is not bool:
            raise ValueError("layer_filter.{} must be boolean".format(key))
    training = config["training"]
    for key in ("epochs", "patience", "batch_size", "hidden", "fourier_frequencies",
                "candidate_chunk", "progress_every_batches", "num_threads", "prefetch_batches"):
        if type(training.get(key)) is not int or training[key] < (0 if key == "fourier_frequencies" else 1):
            raise ValueError("training.{} has an invalid value".format(key))
    if (not training.get("seeds") or len(set(training["seeds"])) != len(training["seeds"]) or
            any(type(seed) is not int or seed < 0 for seed in training["seeds"])):
        raise ValueError("training.seeds must contain distinct nonnegative integers")
    if training.get("device") not in ("cpu", "cuda"):
        raise ValueError("training.device must be cpu or cuda")
    if training.get("normalization_support") not in ("world", "geometry"):
        raise ValueError("training.normalization_support must be world or geometry")
    if training.get("architecture") not in ("baseline", "large"):
        raise ValueError("training.architecture must be baseline or large")
    if training.get("architecture") == "large" and training["hidden"] < 64:
        raise ValueError("large architecture requires hidden >= 64")
    heads = training.get("attention_heads", 1)
    if type(heads) is not int or heads < 1 or training["hidden"] % heads:
        raise ValueError("attention_heads must divide hidden")
    if training.get("amp_dtype", "none") not in ("none", "bfloat16", "float16"):
        raise ValueError("training.amp_dtype is invalid")
    devices = training.get("gpu_devices", [0])
    if (not isinstance(devices, list) or not devices or len(set(devices)) != len(devices) or
            any(type(x) is not int or x < 0 for x in devices)):
        raise ValueError("training.gpu_devices must contain distinct nonnegative integers")
    for key in ("use_obstacle", "pixel_coordinates", "fused_optimizer",
                "deterministic_algorithms", "cudnn_benchmark"):
        if type(training.get(key)) is not bool:
            raise ValueError("training.{} must be boolean".format(key))
    if type(training.get("use_scene_geometry",False)) is not bool:
        raise ValueError("training.use_scene_geometry must be boolean")
    geometry_heads=training.get("geometry_attention_heads",4)
    geometry_cross_heads=training.get("geometry_cross_attention_heads",heads)
    geometry_layers=training.get("geometry_layers",2)
    geometry_dropout=training.get("geometry_dropout",0.0)
    geometry_fusion=training.get("geometry_fusion","concat")
    if type(geometry_heads) is not int or geometry_heads<1:
        raise ValueError("training.geometry_attention_heads has an invalid value")
    if type(geometry_cross_heads) is not int or geometry_cross_heads<1:
        raise ValueError("training.geometry_cross_attention_heads has an invalid value")
    if type(geometry_layers) is not int or geometry_layers<0:
        raise ValueError("training.geometry_layers has an invalid value")
    if (training["hidden"]%geometry_heads or
            training["hidden"]%geometry_cross_heads):
        raise ValueError("geometry attention head counts must divide hidden")
    if (not isinstance(geometry_dropout,(int,float)) or
            not math.isfinite(geometry_dropout) or geometry_dropout<0 or
            geometry_dropout>=1):
        raise ValueError("training.geometry_dropout must be in [0,1)")
    if geometry_fusion not in ("concat","residual"):
        raise ValueError("training.geometry_fusion must be concat or residual")
    if geometry_fusion=="residual" and not training.get("use_scene_geometry",False):
        raise ValueError("residual geometry fusion requires use_scene_geometry=true")
    initial_checkpoint=training.get("initial_checkpoint")
    if initial_checkpoint is not None and (not isinstance(initial_checkpoint,str) or
                                           not initial_checkpoint.strip()):
        raise ValueError("training.initial_checkpoint must be a nonempty path string")
    if geometry_fusion=="residual" and not initial_checkpoint:
        raise ValueError("residual geometry fusion requires an initial blind checkpoint")
    geometry_warmup=training.get("geometry_warmup_epochs",0)
    if (type(geometry_warmup) is not int or geometry_warmup<0 or
            geometry_warmup>=training["epochs"]):
        raise ValueError("geometry_warmup_epochs must be nonnegative and below epochs")
    for key in ("learning_rate", "grad_clip"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError("training.{} must be positive".format(key))
    for key in ("base_learning_rate","geometry_learning_rate"):
        value=training.get(key,training["learning_rate"])
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("training.{} must be positive".format(key))
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    evaluation = config["evaluation"]
    if evaluation.get("combine_rule") != "normalized_elementwise_max":
        raise ValueError("only normalized_elementwise_max is supported")
    if type(evaluation.get("candidate_chunk")) is not int or evaluation["candidate_chunk"] < 1:
        raise ValueError("evaluation.candidate_chunk must be positive")
    if not math.isfinite(evaluation.get("nms_radius", -1)) or evaluation["nms_radius"] < 0:
        raise ValueError("evaluation.nms_radius must be nonnegative")
    thresholds = evaluation.get("distance_thresholds")
    if (not isinstance(thresholds, list) or not thresholds or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in thresholds)):
        raise ValueError("evaluation.distance_thresholds must contain positive finite values")
    return config


def context(config):
    from rind_phase2.data import Phase2Dataset
    data = Path(config["data_root"])
    teacher = Path(config["teacher_experiment_root"])
    output = Path(config["output_root"])
    if not data.is_absolute():
        data = PHASE2_ROOT / data
    if not teacher.is_absolute():
        teacher = PHASE2_ROOT / teacher
    if not output.is_absolute():
        output = PHASE2_ROOT / output
    return Phase2Dataset(data), teacher, output


def build_layer_index(config, dataset, teacher_root, output):
    manifest_path = teacher_root / "records.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "completed Phase2 teacher cache is missing: {}. Run prepare-teachers first.".format(manifest_path))
    source = json.loads(manifest_path.read_text())
    if (not isinstance(source.get("records"), dict) or
            any(not isinstance(source["records"].get(name), list)
                for name in ("train", "validation", "test"))):
        raise ValueError("teacher records.json does not contain the three required splits")
    threshold = float(config["layer_filter"]["maximum_training_physical_cost"])
    records = {name: [] for name in ("train", "validation", "test")}
    evaluation_test = []
    counts = {}
    target_statistics = {}
    split_scenes = {}
    reference_grid = None
    for split in ("train", "validation", "test"):
        statistics = defaultdict(int)
        target_entropy = []
        target_uniform_ce = []
        target_effective_candidates = []
        target_max_probability = []
        split_scenes[split] = set()
        identities = set()
        for window in source["records"][split]:
            identity = (int(window["scene_id"]), int(window["view_id"]))
            if identity in identities:
                raise ValueError("duplicate teacher window in {}: {}".format(split, identity))
            identities.add(identity)
            directory = teacher_root / window["directory"]
            report = json.loads((directory / "report.json").read_text())
            if report.get("status") != "complete" or report.get("partial_combined_distribution"):
                statistics["incomplete_windows"] += 1
                continue
            observation = dataset.get_observation(int(window["scene_id"]), int(window["view_id"]))
            if int(observation["window"][2]) != int(window["size"]):
                raise ValueError("teacher window size mismatch: {}".format(directory))
            free = ~np.asarray(observation["obstacle"], dtype=bool)
            observed = np.asarray(observation["response"], dtype=np.float64)[free]
            if not len(observed):
                raise ValueError("teacher manifest contains a window without free pixels: {}".format(directory))
            if not np.any(observed > 1e-12) or np.all(observed > 1e-12):
                raise ValueError("teacher manifest contains an all-dark/all-lit window: {}".format(directory))
            if (int(report.get("scene_id", -1)) != int(window["scene_id"]) or
                    int(report.get("view_id", -1)) != int(window["view_id"])):
                raise ValueError("teacher report identity mismatch: {}".format(directory))
            split_scenes[split].add(int(window["scene_id"]))
            statistics["complete_windows"] += 1
            detail_by_step = {int(item["step"]): item for item in report.get("layers", [])}
            if len(detail_by_step) != int(report["layers_recovered"]):
                raise ValueError("teacher report layer metadata is incomplete: {}".format(directory))
            window_grid = None
            for step in range(1, int(report["layers_recovered"]) + 1):
                path = directory / "layer-{:02d}.npz".format(step)
                if not path.exists():
                    raise FileNotFoundError("missing layer file: {}".format(path))
                layer = _load_layer(path, include_cost=True)
                response = layer["response"].astype(bool)
                valid = layer["valid"]
                probability = layer["probability"]
                cost = layer["cost"]
                candidate = layer["candidate"]
                if response.shape != free.shape or np.any(response[~free]):
                    raise ValueError("layer mask is misaligned with the free observation area: {}".format(path))
                grid_signature = _array_digest(candidate)
                if window_grid is None:
                    window_grid = grid_signature
                elif window_grid != grid_signature:
                    raise ValueError("layers from one window use different candidate grids: {}".format(directory))
                if reference_grid is None:
                    reference_grid = {
                        "sha256": grid_signature,
                        "candidate_count": int(len(candidate)),
                        "x_count": int(len(layer["x_axis"])),
                        "y_count": int(len(layer["y_axis"])),
                    }
                elif reference_grid["sha256"] != grid_signature:
                    raise ValueError("teacher layers do not use one common full-field candidate grid")
                statistics["legacy_sparse_layers_expanded"] += int(
                    layer["stored_candidate_count"] != layer["full_candidate_count"])
                minimum = float(cost[valid].min())
                all_zero = not bool(np.any(response[free]))
                all_one = bool(np.all(response[free])) if free.any() else False
                keep_zero = not config["layer_filter"]["skip_all_zero"]
                keep_one = not config["layer_filter"]["skip_all_one"]
                informative = (keep_zero or not all_zero) and (keep_one or not all_one)
                high_confidence = informative and minimum <= threshold
                detail = detail_by_step.get(step, {})
                intensity = detail.get("intensity")
                if not isinstance(intensity, (int, float)) or not math.isfinite(intensity) or intensity <= 0:
                    raise ValueError("missing positive layer intensity in {} step {}".format(directory, step))
                entry = {
                    "split": split, "scene_id": int(window["scene_id"]),
                    "view_id": int(window["view_id"]), "size": int(window["size"]),
                    "step": step, "directory": window["directory"],
                    "layer_file": path.name, "minimum_physical_cost": minimum,
                    "intensity": float(intensity),
                    "all_zero": all_zero, "all_one": all_one,
                    "informative": informative, "high_confidence": high_confidence,
                }
                statistics["layers_total"] += 1
                statistics["layers_all_zero"] += int(all_zero)
                statistics["layers_all_one"] += int(all_one)
                statistics["layers_informative"] += int(informative)
                statistics["layers_high_confidence"] += int(high_confidence)
                if high_confidence:
                    records[split].append(entry)
                    positive = probability > 0
                    entropy = float(-np.sum(probability[positive] * np.log(probability[positive])))
                    support_count = (int(valid.sum()) if config["training"]["normalization_support"] == "geometry"
                                     else int(len(probability)))
                    target_entropy.append(entropy)
                    target_uniform_ce.append(math.log(support_count))
                    target_effective_candidates.append(math.exp(entropy))
                    target_max_probability.append(float(probability.max()))
                if split == "test":
                    evaluation_test.append(entry)
        counts[split] = dict(statistics)
        if not records[split]:
            raise ValueError("no high-confidence informative layers in {}".format(split))
        def describe(values):
            values = np.asarray(values, dtype=np.float64)
            return {"count": int(len(values)), "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p10": float(np.quantile(values, 0.1)),
                    "p90": float(np.quantile(values, 0.9)),
                    "min": float(values.min()), "max": float(values.max())}
        target_statistics[split] = {
            "teacher_entropy": describe(target_entropy),
            "uniform_cross_entropy": describe(target_uniform_ce),
            "uniform_information_gap": describe(
                np.asarray(target_uniform_ce) - np.asarray(target_entropy)),
            "effective_candidate_count": describe(target_effective_candidates),
            "maximum_probability": describe(target_max_probability),
        }
        print("index {} | training layers={} | {}".format(split, len(records[split]), dict(statistics)), flush=True)
    if (split_scenes["train"] & split_scenes["validation"] or
            split_scenes["train"] & split_scenes["test"] or
            split_scenes["validation"] & split_scenes["test"]):
        raise ValueError("scene leakage in teacher manifest")
    index = {
        "schema": "phase1-layer-student-v2",
        "target_semantics": config["target_semantics"],
        "teacher_manifest": str(manifest_path.resolve()),
        "teacher_manifest_sha256": _file_digest(manifest_path),
        "teacher_record_counts": source.get("counts"),
        "records": records,
        "evaluation_test_records": evaluation_test,
        "counts": {name: len(records[name]) for name in records},
        "layer_statistics": counts,
        "target_distribution_statistics": target_statistics,
        "filter": config["layer_filter"],
        "candidate_grid": reference_grid,
    }
    _json(output / "layer_records.json", index)
    protocol = {
        "config": config, "config_sha256": _digest(config),
        "layer_index_sha256": _digest(index),
        "source_sha256": _source_hashes(),
        "scene_policy": "reuse the teacher manifest scene-disjoint train/validation/test split",
        "training_unit": "one informative high-confidence binary residual layer",
        "test_policy": "all saved test layers are evaluated; all-zero/all-one layers remain coverage failures",
        "combination": "preserve each layer distribution; normalized elementwise max only for summary-field comparison",
    }
    existing = output / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError("output_root contains another layer protocol; choose a new output_root")
    _json(existing, protocol)
    return index


def load_verified_index(config, teacher_root, output):
    index_path = output / "layer_records.json"
    protocol_path = output / "protocol.json"
    if not index_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("layer index/protocol is missing; run the index command first")
    index = json.loads(index_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if index.get("schema") != "phase1-layer-student-v2":
        raise ValueError("legacy layer index detected; rerun index before training")
    if protocol.get("config") != config or protocol.get("config_sha256") != _digest(config):
        raise ValueError("configuration differs from protocol.json; choose a new output_root or rerun index")
    if protocol.get("layer_index_sha256") != _digest(index):
        raise ValueError("layer_records.json differs from the indexed protocol")
    if protocol.get("source_sha256") != _source_hashes():
        raise ValueError("training/model source changed after indexing; choose a new output_root or rerun index")
    manifest = teacher_root / "records.json"
    if (not manifest.exists() or
            index.get("teacher_manifest_sha256") != _file_digest(manifest)):
        raise ValueError("teacher records changed after indexing; choose a new output_root or rerun index")
    return index


class LayerRecords:
    def __init__(self, entries, dataset, teacher_root, normalization_support="world",
                 use_scene_geometry=False):
        self.entries = list(entries)
        self.dataset = dataset
        self.teacher_root = Path(teacher_root)
        self.normalization_support = normalization_support
        self.use_scene_geometry = bool(use_scene_geometry)
        self.window_sizes = [int(row["size"]) for row in self.entries]
        self.scene_ids = {int(row["scene_id"]) for row in self.entries}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        observation = self.dataset.get_observation(row["scene_id"], row["view_id"])
        path = self.teacher_root / row["directory"] / row["layer_file"]
        layer = _load_layer(path)
        response = layer["response"]
        candidate = layer["candidate"]
        valid = layer["valid"]
        probability = layer["probability"]
        support = valid.copy() if self.normalization_support == "geometry" else np.ones_like(valid)
        result = {
            "response": response, "obstacle": observation["obstacle"],
            "window": observation["window"], "candidate_xy": candidate,
            "valid": valid, "support": support, "teacher_prob": probability,
        }
        if self.use_scene_geometry:
            result.update(self.dataset.get_obstacle_tokens(row["scene_id"]))
        return result


def _atomic_checkpoint(path, checkpoint):
    import torch
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _geometry_parameter(name):
    return name.startswith(("geometry_encoder.", "geometry_attention.",
                            "geometry_correction_head."))


def _resolve_initial_checkpoint(value, seed):
    if not value:
        return None
    path = Path(str(value).format(seed=seed)).expanduser()
    return path if path.is_absolute() else PHASE2_ROOT / path


def _load_blind_initialization(model, path):
    """Load the shared blind locator while leaving geometry modules new."""
    import torch
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError("initial blind checkpoint is missing: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("initial checkpoint has no model state: {}".format(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or not missing or not all(_geometry_parameter(name) for name in missing):
        raise ValueError(
            "blind checkpoint is not architecture-compatible; missing={}, unexpected={}".format(
                missing, unexpected))
    return checkpoint


def train(config, dataset, teacher_root, output, index, runtime_gpu_ids=None):
    import torch
    from rind_phase1.interfaces import INTERFACE_VERSION
    from rind_phase1.model import SourceEnergyField
    from rind_phase1.train import batches, load_model, run_epoch

    settings = dict(config["training"])
    if runtime_gpu_ids is not None:
        settings["gpu_devices"] = list(runtime_gpu_ids)
    torch.set_num_threads(settings["num_threads"])
    support = settings["normalization_support"]
    use_scene_geometry=settings.get("use_scene_geometry",False)
    splits = {name: LayerRecords(index["records"][name], dataset, teacher_root, support,
                                 use_scene_geometry)
              for name in ("train", "validation", "test")}
    if (splits["train"].scene_ids & splits["validation"].scene_ids or
            splits["train"].scene_ids & splits["test"].scene_ids or
            splits["validation"].scene_ids & splits["test"].scene_ids):
        raise ValueError("scene leakage between layer splits")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    gpu_ids = list(settings.get("gpu_devices", [0])) if settings["device"] == "cuda" else []
    if gpu_ids and max(gpu_ids) >= torch.cuda.device_count():
        raise ValueError("requested GPU IDs {} but only {} are visible".format(gpu_ids, torch.cuda.device_count()))
    device = torch.device("cuda:{}".format(gpu_ids[0]) if gpu_ids else "cpu")
    if gpu_ids:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)
    reports = {}
    for seed in settings["seeds"]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if gpu_ids:
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(settings.get("deterministic_algorithms", False))
        rng = np.random.default_rng(seed)
        model_config = {
            "hidden": settings["hidden"],
            "fourier_frequencies": settings["fourier_frequencies"],
            "coordinate_scale": float(dataset.manifest["global_size"]),
            "use_obstacle": settings.get("use_obstacle", False),
            "use_scene_geometry": use_scene_geometry,
            "geometry_parameter_width": int(dataset._base.obstacle_params.shape[-1]),
            "geometry_type_count": 4,
            "geometry_attention_heads": settings.get("geometry_attention_heads", 4),
            "geometry_cross_attention_heads": settings.get(
                "geometry_cross_attention_heads",settings.get("attention_heads",1)),
            "geometry_layers": settings.get("geometry_layers", 2),
            "geometry_dropout": settings.get("geometry_dropout", 0.0),
            "geometry_fusion": settings.get("geometry_fusion", "concat"),
            "pixel_coordinates": settings.get("pixel_coordinates", True),
            "normalization_support": support,
            "architecture": settings.get("architecture", "baseline"),
            "attention_heads": settings.get("attention_heads", 1),
        }
        base_model = SourceEnergyField(**model_config).to(device)
        initial_path = _resolve_initial_checkpoint(settings.get("initial_checkpoint"), seed)
        initial_checkpoint = _load_blind_initialization(base_model, initial_path)
        model = (torch.nn.DataParallel(base_model, device_ids=gpu_ids, output_device=gpu_ids[0])
                 if len(gpu_ids) > 1 else base_model)
        fused = bool(settings.get("fused_optimizer", False) and device.type == "cuda")
        base_parameters=[]; geometry_parameters=[]
        for name,parameter in base_model.named_parameters():
            (geometry_parameters if _geometry_parameter(name) else base_parameters).append(parameter)
        if settings.get("geometry_fusion") == "residual" and not geometry_parameters:
            raise ValueError("residual geometry training has no geometry parameters")
        base_lr=float(settings.get("base_learning_rate",settings["learning_rate"]))
        geometry_lr=float(settings.get("geometry_learning_rate",settings["learning_rate"]))
        parameter_groups=[{"params":base_parameters,"lr":base_lr,"name":"blind_base"}]
        if geometry_parameters:
            parameter_groups.append({"params":geometry_parameters,"lr":geometry_lr,
                                     "name":"geometry_residual"})
        optimizer = torch.optim.AdamW(parameter_groups,weight_decay=settings["weight_decay"],
                                      fused=fused)
        directory = output / "models" / "seed_{}".format(seed)
        directory.mkdir(parents=True, exist_ok=True)
        signature = _digest({"protocol": json.loads((output / "protocol.json").read_text()),
                             "records": index, "seed": seed})
        history, best, bad, start, steps = [], float("inf"), 0, 0, 0
        if (directory / "last.pt").exists():
            checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
            if checkpoint.get("run_signature") != signature:
                raise ValueError("resume checkpoint protocol/targets/seed mismatch")
            base_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
            cuda_rng = checkpoint.get("cuda_rng", [])
            if gpu_ids and len(cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng)
            rng.bit_generator.state = checkpoint["numpy_rng"]
            history = checkpoint["history"]
            best = checkpoint["best_validation_ce"]
            bad = checkpoint["bad_epochs"]
            start = checkpoint["epoch"]
            steps = checkpoint["global_step"]
        def number_batches(records):
            return sum(int(np.ceil(sum(size == current for size in records.window_sizes) /
                                   settings["batch_size"]))
                       for current in sorted(set(records.window_sizes)))
        epoch_steps = number_batches(splits["train"])
        validation_steps = number_batches(splits["validation"])
        test_steps = number_batches(splits["test"])
        candidate_chunk = None if len(gpu_ids) > 1 else settings["candidate_chunk"]
        every = settings.get("progress_every_batches", 10)
        def callback(phase, epoch, total):
            def report(batch_index, batch_records, running_ce):
                if batch_index == 1 or batch_index % every == 0 or batch_index == total:
                    _progress(output, "batch_progress",
                              "{} | seed={}, epoch={}/{}, batch={}/{}, running CE={:.6f}".format(
                                  phase, seed, epoch, settings["epochs"], batch_index, total, running_ce),
                              phase=phase, seed=seed, epoch=epoch, batch=batch_index,
                              total_batches=total, running_cross_entropy=running_ce)
            return report
        started = time.perf_counter()
        uniform_ce = (math.log(index["candidate_grid"]["candidate_count"])
                      if support == "world" else None)
        train_entropy = index["target_distribution_statistics"]["train"]["teacher_entropy"]["mean"]
        validation_entropy = index["target_distribution_statistics"]["validation"]["teacher_entropy"]["mean"]
        validation_uniform = index["target_distribution_statistics"]["validation"]["uniform_cross_entropy"]["mean"]
        warmup_epochs=int(settings.get("geometry_warmup_epochs",0))
        if warmup_epochs<0:
            raise ValueError("geometry_warmup_epochs must be nonnegative")
        _progress(output, "train_start",
                  "layer student | records train/validation/test={}/{}/{} | candidates={} | uniform CE={} | teacher entropy train/validation={:.6f}/{:.6f} | GPUs={} | resume epoch={} | initialization={} | base/geometry LR={}/{} | geometry warmup={}".format(
                      len(splits["train"]), len(splits["validation"]), len(splits["test"]),
                      index["candidate_grid"]["candidate_count"],
                      "{:.6f}".format(uniform_ce) if uniform_ce is not None else "support-dependent",
                      train_entropy, validation_entropy, gpu_ids or ["cpu"], start,
                      str(initial_path) if initial_path else "random",base_lr,geometry_lr,warmup_epochs),
                  seed=seed, resume_epoch=start,
                  records=index["counts"], target_semantics=config["target_semantics"])
        if start==0 and initial_checkpoint is not None:
            initial_validation_ce=run_epoch(
                model,splits["validation"],settings["batch_size"],candidate_chunk=candidate_chunk,
                progress_callback=callback("initial validation",0,validation_steps),
                amp_dtype=settings.get("amp_dtype","none"),
                prefetch_batches=settings.get("prefetch_batches",0))
            best=initial_validation_ce
            initial_state={
                "interface_version":INTERFACE_VERSION,"model_config":model_config,
                "model":base_model.state_dict(),"epoch":0,"seed":seed,
                "records_manifest":str((output/"layer_records.json").resolve()),
                "target_semantics":config["target_semantics"],
                "scene_splits":{name:sorted(records.scene_ids) for name,records in splits.items()},
                "optimizer":optimizer.state_dict(),"history":history,
                "best_validation_ce":best,"bad_epochs":0,"global_step":0,
                "run_signature":signature,"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if gpu_ids else [],
                "numpy_rng":rng.bit_generator.state,
                "initial_checkpoint":str(initial_path.resolve()),
                "initial_checkpoint_epoch":int(initial_checkpoint.get("epoch",-1)),
            }
            _atomic_checkpoint(directory/"best.pt",initial_state)
            _progress(output,"initial_validation",
                      "warm-start baseline | validation CE={:.6f}; epoch 0 is eligible for best checkpoint".format(best),
                      validation_cross_entropy=best,initial_checkpoint=str(initial_path))
        for epoch in range(start + 1, settings["epochs"] + 1):
            if bad >= settings["patience"]:
                break
            base_trainable=epoch>warmup_epochs
            for parameter in base_parameters:
                parameter.requires_grad_(base_trainable)
            if epoch==start+1 or epoch==warmup_epochs+1:
                _progress(output,"trainability",
                          "epoch {} | blind base {} | geometry residual trainable".format(
                              epoch,"trainable" if base_trainable else "frozen"),
                          epoch=epoch,blind_base_trainable=base_trainable)
            tick = time.perf_counter()
            if gpu_ids:
                for gpu_id in gpu_ids:
                    torch.cuda.reset_peak_memory_stats(gpu_id)
            train_ce = run_epoch(
                model, splits["train"], settings["batch_size"], optimizer, rng,
                settings["grad_clip"], candidate_chunk, callback("train", epoch, epoch_steps),
                amp_dtype=settings.get("amp_dtype", "none"),
                prefetch_batches=settings.get("prefetch_batches", 0))
            validation_ce = run_epoch(
                model, splits["validation"], settings["batch_size"], candidate_chunk=candidate_chunk,
                progress_callback=callback("validation", epoch, validation_steps),
                amp_dtype=settings.get("amp_dtype", "none"),
                prefetch_batches=settings.get("prefetch_batches", 0))
            steps += epoch_steps
            improved = validation_ce < best
            best = min(best, validation_ce)
            bad = 0 if improved else bad + 1
            epoch_seconds=time.perf_counter()-tick
            throughput=len(splits["train"])/max(epoch_seconds,np.finfo(float).eps)
            peak_gib=([torch.cuda.max_memory_allocated(gpu_id)/(1024**3) for gpu_id in gpu_ids]
                      if gpu_ids else [])
            history.append({"epoch": epoch, "train_cross_entropy": train_ce,
                            "validation_cross_entropy": validation_ce,
                            "train_forward_kl": train_ce-train_entropy,
                            "validation_forward_kl": validation_ce-validation_entropy,
                            "validation_gain_over_uniform": validation_uniform-validation_ce,
                            "seconds": epoch_seconds,
                            "training_layers_per_second": throughput,
                            "peak_gpu_memory_gib": peak_gib})
            common = {
                "interface_version": INTERFACE_VERSION, "model_config": model_config,
                "model": base_model.state_dict(), "epoch": epoch, "seed": seed,
                "records_manifest": str((output / "layer_records.json").resolve()),
                "target_semantics": config["target_semantics"],
                "scene_splits": {name: sorted(records.scene_ids) for name, records in splits.items()},
                "optimizer": optimizer.state_dict(), "history": history,
                "best_validation_ce": best, "bad_epochs": bad, "global_step": steps,
                "run_signature": signature, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if gpu_ids else [],
                "numpy_rng": rng.bit_generator.state,
                "initial_checkpoint":str(initial_path.resolve()) if initial_path else None,
                "initial_checkpoint_epoch":(int(initial_checkpoint.get("epoch",-1))
                                             if initial_checkpoint is not None else None),
            }
            if improved:
                _atomic_checkpoint(directory / "best.pt", common)
            _atomic_checkpoint(directory / "last.pt", common)
            _json(directory / "training.json", {"seed": seed, "history": history,
                  "best_validation_cross_entropy": best,
                  "selection": "minimum validation CE on high-confidence residual layers"})
            _progress(output, "epoch_done",
                      "epoch {}/{} | train CE={:.6f}, validation CE={:.6f} | train/validation KL={:.6f}/{:.6f} | validation gain over uniform={:.6f} | {:.1f} train layers/s | peak GPU GiB={} | best={:.6f}, bad={}/{}".format(
                          epoch, settings["epochs"], train_ce, validation_ce,
                          train_ce-train_entropy, validation_ce-validation_entropy,
                          validation_uniform-validation_ce,throughput,
                          [round(value,2) for value in peak_gib],best,bad,
                          settings["patience"]), epoch=epoch, train_cross_entropy=train_ce,
                      validation_cross_entropy=validation_ce,training_layers_per_second=throughput,
                      peak_gpu_memory_gib=peak_gib,best_validation_cross_entropy=best)
        best_model, checkpoint = load_model(directory / "best.pt")
        best_model.to(device)
        test_model = (torch.nn.DataParallel(best_model, device_ids=gpu_ids, output_device=gpu_ids[0])
                      if len(gpu_ids) > 1 else best_model)
        test_ce = run_epoch(
            test_model, splits["test"], settings["batch_size"], candidate_chunk=candidate_chunk,
            progress_callback=callback("test", checkpoint["epoch"], test_steps),
            amp_dtype=settings.get("amp_dtype", "none"),
            prefetch_batches=settings.get("prefetch_batches", 0))
        report = {"seed": seed, "history": history, "best_epoch": checkpoint["epoch"],
                  "best_validation_cross_entropy": checkpoint["best_validation_ce"],
                  "test_cross_entropy": test_ce, "records": index["counts"],
                  "uniform_cross_entropy": uniform_ce,
                  "validation_improvement_over_uniform": (
                      None if uniform_ce is None else uniform_ce-checkpoint["best_validation_ce"]),
                  "test_improvement_over_uniform": None if uniform_ce is None else uniform_ce-test_ce,
                  "elapsed_seconds": time.perf_counter() - started}
        _json(directory / "training.json", report)
        reports[str(seed)] = report
        _progress(output, "train_done", "layer student done | seed={}, test CE={:.6f}".format(seed, test_ce),
                  seed=seed, test_cross_entropy=test_ce)
    _json(output / "training_summary.json", reports)
    return reports


def _distribution_metrics(q, p, log_p):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    log_p = np.asarray(log_p, dtype=np.float64)
    positive_q = q > 0
    positive_p = p > 0
    entropy = float(-np.sum(q[positive_q] * np.log(q[positive_q])))
    cross_entropy = float(-np.sum(q[positive_q] * log_p[positive_q]))
    mixture = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q[positive_q] * np.log(q[positive_q] / mixture[positive_q])))
    js += 0.5 * float(np.sum(p[positive_p] * np.log(p[positive_p] / mixture[positive_p])))
    tv = 0.5 * float(np.abs(q - p).sum())
    order = np.argsort(-q, kind="stable")
    cutoff = int(np.searchsorted(np.cumsum(q[order]), 0.9, side="left")) + 1
    return {"cross_entropy": cross_entropy, "teacher_entropy": entropy,
            "forward_kl": cross_entropy - entropy, "jensen_shannon": js,
            "total_variation": tv, "overlap": 1.0 - tv,
            "student_mass_on_teacher_90pct_set": float(p[order[:cutoff]].sum())}


def _summary(rows):
    if not rows:
        return None
    result = {"count": len(rows)}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                       "p90": float(np.quantile(values, 0.9))}
    return result


def _topk_nms(probability, candidate, valid, count, radius):
    order = np.argsort(-np.where(valid, probability, -np.inf), kind="stable")
    selected = []
    for index in order:
        if not np.isfinite(probability[index]) or not valid[index]:
            continue
        point = candidate[index]
        if all(np.linalg.norm(point - candidate[old]) >= radius for old in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return candidate[selected]


def _matching_distances(predicted, truth):
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if len(truth) == 0 or len(predicted) < len(truth):
        return None
    best = None
    for permutation in itertools.permutations(range(len(predicted)), len(truth)):
        distances = np.linalg.norm(predicted[list(permutation)] - truth, axis=1)
        if best is None or distances.sum() < best.sum():
            best = distances
    return best


def _informative_truth(dataset, scene, view, tolerance=1e-12):
    observation = dataset.get_observation(scene, view)
    free = ~observation["obstacle"]
    channels = np.asarray(dataset.get_channels(scene, view), dtype=np.float64)
    sources = np.asarray(dataset.get_source_params(scene), dtype=np.float64)
    selected = []
    for index, channel in enumerate(channels):
        values = channel[free]
        if len(values) and np.any(values > tolerance) and not np.all(values > tolerance):
            selected.append(sources[index, :2])
    return np.asarray(selected, dtype=np.float64).reshape(-1, 2)


def evaluate(config, dataset, teacher_root, output, index, seed=0, device_name="auto"):
    import torch
    from rind_phase1.model import student_distribution
    from rind_phase1.train import collate_records, load_model
    from layered_residual import combine_layer_probabilities

    checkpoint_path = output / "models" / "seed_{}".format(seed) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("layer student checkpoint is missing: {}".format(checkpoint_path))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model, checkpoint = load_model(checkpoint_path)
    model.to(device).eval()
    all_entries = index["evaluation_test_records"]
    groups = defaultdict(list)
    for row in all_entries:
        groups[(row["scene_id"], row["view_id"])].append(row)
    layer_metrics, high_metrics, mixed_metrics = [], [], []
    coordinate_rows, window_rows = [], []
    skipped_zero = skipped_one = 0
    candidate_chunk = int(config["evaluation"]["candidate_chunk"])
    amp_name = config["training"].get("amp_dtype", "none")
    amp_enabled = device.type == "cuda" and amp_name != "none"
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    started = time.perf_counter()
    with torch.inference_mode():
        for position, ((scene, view), rows) in enumerate(sorted(groups.items()), 1):
            informative = [row for row in rows if row["informative"]]
            skipped_zero += sum(row["all_zero"] for row in rows)
            skipped_one += sum(row["all_one"] for row in rows)
            if not informative:
                window_rows.append({"scene_id": scene, "view_id": view, "status": "no_informative_layer"})
                continue
            records = LayerRecords(
                informative,dataset,teacher_root,config["training"]["normalization_support"],
                config["training"].get("use_scene_geometry",False))
            samples = [records[index_value] for index_value in range(len(records))]
            batch = collate_records(samples)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                encoded = model.encode_observation(
                    batch["response"],batch["obstacle"],batch["window"],
                    batch.get("geometry_types"),batch.get("geometry_params"),
                    batch.get("geometry_mask"))
                energy_parts = []
                for start in range(0, batch["candidate_xy"].shape[1], candidate_chunk):
                    energy_parts.append(model.score_encoded(
                        encoded, batch["candidate_xy"][:, start:start + candidate_chunk]))
                energy = torch.cat(energy_parts, dim=1)
            log_p, p = student_distribution(energy.float(), batch["support"])
            q = batch["teacher_prob"].cpu().numpy()
            p = p.cpu().numpy()
            log_p = log_p.cpu().numpy()
            candidate = batch["candidate_xy"].cpu().numpy()
            valid = batch["valid"].cpu().numpy()
            per_window_layers = []
            for layer_index, row in enumerate(informative):
                metric = _distribution_metrics(q[layer_index], p[layer_index], log_p[layer_index])
                layer_metrics.append(metric)
                if row["high_confidence"]:
                    high_metrics.append(metric)
                per_window_layers.append({
                    "step": int(row["step"]),
                    "minimum_physical_cost": float(row["minimum_physical_cost"]),
                    "intensity": float(row["intensity"]),
                    "high_confidence": bool(row["high_confidence"]),
                    "metrics": metric,
                })
            for array in (candidate, valid):
                if not all(np.array_equal(array[0], other) for other in array[1:]):
                    raise ValueError("layers from one window do not share a candidate grid")
            teacher_max = combine_layer_probabilities(list(q))
            student_max = combine_layer_probabilities(list(p))
            mixed = _distribution_metrics(
                teacher_max, student_max, np.log(np.maximum(student_max, np.finfo(np.float64).tiny)))
            mixed_metrics.append(mixed)
            truth = _informative_truth(dataset, scene, view)
            coordinate = None
            if len(truth):
                student_xy = _topk_nms(student_max, candidate[0], valid[0], len(truth),
                                       config["evaluation"]["nms_radius"])
                teacher_xy = _topk_nms(teacher_max, candidate[0], valid[0], len(truth),
                                       config["evaluation"]["nms_radius"])
                student_distance = _matching_distances(student_xy, truth)
                teacher_distance = _matching_distances(teacher_xy, truth)
                if student_distance is not None and teacher_distance is not None:
                    coordinate = {"student_mean_distance": float(student_distance.mean()),
                                  "teacher_mean_distance": float(teacher_distance.mean())}
                    for threshold in config["evaluation"]["distance_thresholds"]:
                        coordinate["student_recall_at_{}".format(int(threshold))] = float(
                            np.mean(student_distance <= threshold))
                        coordinate["teacher_recall_at_{}".format(int(threshold))] = float(
                            np.mean(teacher_distance <= threshold))
                    coordinate_rows.append(coordinate)
            window_rows.append({"scene_id": scene, "view_id": view, "status": "evaluated",
                                "layers": len(informative), "high_confidence_layers": int(sum(
                                    row["high_confidence"] for row in informative)),
                                "informative_true_sources": len(truth), "mixed": mixed,
                                "coordinate": coordinate,
                                "layer_metrics": per_window_layers})
            if position == 1 or position % 100 == 0 or position == len(groups):
                print("evaluate windows {}/{}".format(position, len(groups)), flush=True)
    report = {
        "schema": "phase1-layer-student-evaluation-v1",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_config": checkpoint["model_config"],
        "device": str(device), "target_semantics": config["target_semantics"],
        "combine_rule": config["evaluation"]["combine_rule"],
        "source_truth_used_for_prediction": False,
        "source_truth_used_for_coordinate_evaluation_only": True,
        "layer_distribution_metrics_all_informative": _summary(layer_metrics),
        "layer_distribution_metrics_high_confidence": _summary(high_metrics),
        "mixed_max_field_metrics": _summary(mixed_metrics),
        "coordinate_metrics_on_informative_true_sources": _summary(coordinate_rows),
        "coverage": {"test_windows_in_teacher_manifest": len(groups),
                     "windows_evaluated": sum(row["status"] == "evaluated" for row in window_rows),
                     "windows_without_informative_layers": sum(
                         row["status"] == "no_informative_layer" for row in window_rows),
                     "windows_with_coordinate_metrics": len(coordinate_rows),
                     "all_zero_layers": int(skipped_zero), "all_one_layers": int(skipped_one),
                     "informative_layers_evaluated": len(layer_metrics),
                     "high_confidence_layers_evaluated": len(high_metrics),
                     "low_confidence_layers_evaluated": len(layer_metrics)-len(high_metrics)},
        "candidate_grid": index["candidate_grid"],
        "elapsed_seconds": time.perf_counter() - started,
        "windows": window_rows,
    }
    destination = output / "evaluation" / "multi_source_test.json"
    _json(destination, report)
    print("Saved:", destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "train", "evaluate", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    runtime_gpu_ids = None
    if args.gpus is not None:
        if args.gpus < 1:
            parser.error("--gpus must be positive")
        runtime_gpu_ids = list(range(args.gpus))
    dataset, teacher_root, output = context(config)
    output.mkdir(parents=True, exist_ok=True)
    _progress(output, "run_start", "command={} | output={} | teacher={}".format(
        args.command, output, teacher_root), command=args.command,
        output=str(output), teacher_root=str(teacher_root))
    if args.command in ("index", "all"):
        index = build_layer_index(config, dataset, teacher_root, output)
    else:
        index = load_verified_index(config, teacher_root, output)
    if args.command in ("train", "all"):
        train(config, dataset, teacher_root, output, index, runtime_gpu_ids)
    if args.command in ("evaluate", "all"):
        evaluate(config, dataset, teacher_root, output, index, args.seed, args.evaluation_device)
    _progress(output, "run_done", "command={} complete".format(args.command), command=args.command)


if __name__ == "__main__":
    main()
