"""Fine-tune the Phase-I-style layer locator on sets of views from one scene.

This is multi-view distillation, not joint multi-source enumeration.  Every
network forward remains one binary residual-layer localization problem.  The
set objective is applied only after teacher-associated layers are aligned.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

PHASE2_ROOT = Path(__file__).resolve().parents[2]

from rind_phase1.interfaces import INTERFACE_VERSION
from rind_phase1.model import student_distribution
from rind_phase1.multiview import LayerField, associate_layers, layer_association_score
from rind_phase1.set_training import multiview_distillation_loss
from rind_phase1.train import collate_records, load_model
from rind_phase2.data import Phase2Dataset
from train_layers import LayerRecords, _load_layer


def _canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def _resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PHASE2_ROOT / path


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = ("name", "data_root", "teacher_experiment_root", "layer_index_root",
                "output_root", "initial_checkpoint", "association", "training")
    if any(key not in config for key in required):
        raise ValueError("view-set config is missing a required field")
    association = config["association"]
    for key in ("minimum_score", "intensity_log_scale",
                "exact_intensity_minimum_score", "intensity_relative_tolerance",
                "informativeness_scale"):
        if key not in association:
            raise ValueError("association.{} is required".format(key))
    training = config["training"]
    for key in ("seeds", "epochs", "patience", "scenes_per_batch",
                "max_views_per_track", "layer_loss_weight", "set_loss_weight",
                "redundancy_discount", "learning_rate", "weight_decay", "grad_clip",
                "candidate_chunk", "num_threads", "device",
                "maximum_layer_ce_increase"):
        if key not in training:
            raise ValueError("training.{} is required".format(key))
    if training.get("normalization_support", "world") != "world":
        raise ValueError("view-set fusion currently requires normalization_support=world")
    if (not training["seeds"] or training["epochs"] < 1 or
            training["patience"] < 1 or training["scenes_per_batch"] < 1 or
            training["max_views_per_track"] < 2 or training["candidate_chunk"] < 1):
        raise ValueError("invalid positive training counts")
    if training["device"] not in ("cpu", "cuda"):
        raise ValueError("training.device must be cpu or cuda")
    if training.get("amp_dtype", "none") not in ("none", "float16", "bfloat16"):
        raise ValueError("training.amp_dtype must be none, float16 or bfloat16")
    if not 0 <= float(training["redundancy_discount"]) <= 1:
        raise ValueError("training.redundancy_discount must be in [0,1]")
    if (not math.isfinite(float(training["maximum_layer_ce_increase"])) or
            float(training["maximum_layer_ce_increase"]) < 0):
        raise ValueError("training.maximum_layer_ce_increase must be nonnegative")
    if (int(training.get("prefetch_workers", 0)) < 0 or
            int(training.get("prefetch_batches", 0)) < 0):
        raise ValueError("prefetch counts must be nonnegative")
    return config


def resolve(config):
    return {
        "data": _resolve_path(config["data_root"]),
        "teacher": _resolve_path(config["teacher_experiment_root"]),
        "layer_index": _resolve_path(config["layer_index_root"]),
        "output": _resolve_path(config["output_root"]),
    }


def _association_arguments(config):
    settings = config["association"]
    return {
        "minimum_score": float(settings["minimum_score"]),
        "intensity_log_scale": float(settings["intensity_log_scale"]),
        "exact_intensity_minimum_score": float(
            settings["exact_intensity_minimum_score"]),
        "intensity_relative_tolerance": float(
            settings["intensity_relative_tolerance"]),
        "informativeness_scale": float(settings["informativeness_scale"]),
    }


def build_view_set_index(config, teacher_root, layer_index_root, output):
    source_path = layer_index_root / "layer_records.json"
    if not source_path.is_file():
        raise FileNotFoundError("layer_records.json is missing; build the v4 layer index first")
    source = json.loads(source_path.read_text())
    if source.get("schema") != "phase1-layer-student-v2":
        raise ValueError("view-set training requires the audited v2 layer index")
    arguments = _association_arguments(config)
    result = {}
    statistics = {}
    for split in ("train", "validation", "test"):
        entries = source["records"][split]
        by_scene = defaultdict(list)
        for index, row in enumerate(entries):
            by_scene[int(row["scene_id"])].append((index, row))
        scene_sets = []
        layer_count = track_count = pair_count = 0
        track_lengths = []
        minimum_scores = []
        for position, scene_id in enumerate(sorted(by_scene), 1):
            indexed_rows = sorted(
                by_scene[scene_id], key=lambda item: (item[1]["view_id"], item[1]["step"]))
            fields = []
            global_indices = []
            for global_index, row in indexed_rows:
                layer_path = teacher_root / row["directory"] / row["layer_file"]
                probability = _load_layer(layer_path)["probability"]
                fields.append(LayerField(
                    int(row["view_id"]), int(row["step"]),
                    float(row["intensity"]), probability))
                global_indices.append(global_index)
            local_tracks = associate_layers(fields, **arguments)
            tracks = []
            for local_track in local_tracks:
                if len(local_track) < 2:
                    continue
                scores = [layer_association_score(
                    fields[first], fields[second],
                    arguments["intensity_log_scale"],
                    arguments["informativeness_scale"])
                    for offset, first in enumerate(local_track)
                    for second in local_track[offset + 1:]]
                track = {
                    "entry_indices": [global_indices[index] for index in local_track],
                    "view_ids": [int(fields[index].view_id) for index in local_track],
                    "minimum_pair_score": float(min(scores)),
                    "mean_pair_score": float(np.mean(scores)),
                }
                tracks.append(track)
                layer_count += len(local_track)
                pair_count += len(scores)
                track_count += 1
                track_lengths.append(len(local_track))
                minimum_scores.append(track["minimum_pair_score"])
            if tracks:
                scene_sets.append({"scene_id": scene_id, "tracks": tracks})
            if position == 1 or position % 500 == 0 or position == len(by_scene):
                print("index {} scenes {}/{} | retained scenes={} tracks={}".format(
                    split, position, len(by_scene), len(scene_sets), track_count), flush=True)
        if not scene_sets:
            raise ValueError("no multi-view contribution tracks found in {}".format(split))
        result[split] = scene_sets
        statistics[split] = {
            "source_layers": len(entries),
            "source_scenes": len(by_scene),
            "retained_scenes": len(scene_sets),
            "multi_view_tracks": track_count,
            "layer_appearances_in_tracks": layer_count,
            "cross_view_pairs": pair_count,
            "track_length_mean": float(np.mean(track_lengths)),
            "track_length_min": int(min(track_lengths)),
            "track_length_max": int(max(track_lengths)),
            "minimum_pair_score_mean": float(np.mean(minimum_scores)),
        }
    index = {
        "schema": "phase2-view-set-index-v1",
        "source_layer_index": str(source_path.resolve()),
        "source_layer_index_sha256": _file_digest(source_path),
        "candidate_grid": source["candidate_grid"],
        "association": config["association"],
        "scene_sets": result,
        "statistics": statistics,
        "semantics": (
            "teacher-associated tracks of Phase1-style binary residual layers; "
            "no source IDs or obstacle geometry used for association"),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "view_set_index.json", index)
    print(json.dumps(statistics, indent=2), flush=True)
    return index


def load_view_set_index(config, layer_index_root, output):
    path = output / "view_set_index.json"
    if not path.is_file():
        raise FileNotFoundError("view_set_index.json is missing; run index first")
    index = json.loads(path.read_text())
    source_path = layer_index_root / "layer_records.json"
    if (index.get("schema") != "phase2-view-set-index-v1" or
            index.get("association") != config["association"] or
            not source_path.is_file() or
            index.get("source_layer_index_sha256") != _file_digest(source_path)):
        raise ValueError("view-set index does not match the current layer index/config")
    return index, json.loads(source_path.read_text())


def _make_units(scene_sets, *, training, rng, max_views, max_scenes=0):
    selected_scenes = list(scene_sets[:max_scenes] if max_scenes else scene_sets)
    units = []
    for scene in selected_scenes:
        tracks = scene["tracks"]
        chosen = [tracks[int(rng.integers(len(tracks)))]] if training else tracks
        for track in chosen:
            indices = list(track["entry_indices"])
            if len(indices) > max_views:
                if training:
                    chosen_positions = sorted(rng.choice(
                        len(indices), size=max_views, replace=False).tolist())
                    indices = [indices[position] for position in chosen_positions]
                else:
                    indices = indices[:max_views]
            units.append({"scene_id": int(scene["scene_id"]), "indices": indices})
    if training:
        rng.shuffle(units)
    return units


class _ChunkedStudent:
    """Factory namespace so torch is imported only by training commands."""
    @staticmethod
    def create(base_model, candidate_chunk):
        import torch

        class Chunked(torch.nn.Module):
            def __init__(self, base, chunk):
                super().__init__()
                self.base = base
                self.chunk = int(chunk)

            def forward(self, response, obstacle, window, candidate_xy):
                encoded = self.base.encode_observation(response, obstacle, window)
                parts = [self.base.score_encoded(encoded, candidate_xy[:, start:start + self.chunk])
                         for start in range(0, candidate_xy.shape[1], self.chunk)]
                return torch.cat(parts, dim=1)

        return Chunked(base_model, candidate_chunk)


def _predict_samples(model, samples, device, amp_dtype):
    import torch
    by_size = defaultdict(list)
    for position, sample in enumerate(samples):
        by_size[int(np.asarray(sample["window"])[2])].append(position)
    log_probabilities = [None] * len(samples)
    teachers = [None] * len(samples)
    amp_enabled = device.type == "cuda" and amp_dtype != "none"
    torch_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    for positions in by_size.values():
        batch = collate_records([samples[position] for position in positions])
        if device.type == "cuda":
            batch = {key: value.pin_memory().to(device, non_blocking=True)
                     for key, value in batch.items()}
        else:
            batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=amp_enabled):
            energy = model(batch["response"], batch["obstacle"],
                           batch["window"], batch["candidate_xy"])
        if not torch.isfinite(energy[batch["support"]]).all():
            if not amp_enabled:
                raise ValueError("view-set forward produced nonfinite energy")
            with torch.autocast(device_type=device.type, enabled=False):
                energy = model(batch["response"], batch["obstacle"],
                               batch["window"], batch["candidate_xy"])
            if not torch.isfinite(energy[batch["support"]]).all():
                raise ValueError("view-set FP32 retry produced nonfinite energy")
        log_probability, _ = student_distribution(energy.float(), batch["support"])
        for row, position in enumerate(positions):
            log_probabilities[position] = log_probability[row]
            teachers[position] = batch["teacher_prob"][row]
    return log_probabilities, teachers


def _pad_tracks(units, log_probabilities, teachers):
    import torch
    width = log_probabilities[0].shape[0]
    max_views = max(len(unit["indices"]) for unit in units)
    zero_log = torch.zeros(width, dtype=log_probabilities[0].dtype,
                           device=log_probabilities[0].device)
    zero_teacher = torch.zeros(width, dtype=teachers[0].dtype,
                               device=teachers[0].device)
    student_rows = []
    teacher_rows = []
    masks = []
    cursor = 0
    for unit in units:
        count = len(unit["indices"])
        student = log_probabilities[cursor:cursor + count]
        teacher = teachers[cursor:cursor + count]
        cursor += count
        student_rows.append(torch.stack(student + [zero_log] * (max_views - count)))
        teacher_rows.append(torch.stack(teacher + [zero_teacher] * (max_views - count)))
        masks.append([True] * count + [False] * (max_views - count))
    return (torch.stack(student_rows), torch.stack(teacher_rows),
            torch.tensor(masks, dtype=torch.bool, device=zero_log.device))


def _mean_rows_by_scene(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    keys = [key for key in rows[0] if key != "scene_id"]
    per_scene = [{key: float(np.mean([item[key] for item in values])) for key in keys}
                 for values in grouped.values()]
    result = {key: float(np.mean([row[key] for row in per_scene])) for key in keys}
    result.update({"scenes": len(grouped), "tracks": len(rows)})
    return result


def _track_pairs(tracks):
    return {tuple(sorted((first, second)))
            for track in tracks
            for offset, first in enumerate(track)
            for second in track[offset + 1:]}


def evaluate_student_association(model, records, entries, scene_sets, config,
                                 settings, device, *, max_scenes=0):
    """Evaluate the association step that will actually run at inference."""
    import torch
    by_scene = defaultdict(list)
    for index, row in enumerate(entries):
        by_scene[int(row["scene_id"])].append(index)
    selected = list(scene_sets[:max_scenes] if max_scenes else scene_sets)
    arguments = _association_arguments(config)
    totals = {"true_positive_pairs": 0, "predicted_pairs": 0,
              "teacher_pairs": 0}
    scene_rows = []
    model.eval()
    with torch.inference_mode():
        for position, scene in enumerate(selected, 1):
            scene_id = int(scene["scene_id"])
            indices = sorted(by_scene[scene_id],
                             key=lambda index: (entries[index]["view_id"],
                                                entries[index]["step"]))
            samples = [records[index] for index in indices]
            log_probabilities, _ = _predict_samples(
                model, samples, device, settings.get("amp_dtype", "none"))
            fields = [LayerField(
                int(entries[index]["view_id"]), int(entries[index]["step"]),
                float(entries[index]["intensity"]),
                log_probability.detach().exp().cpu().numpy())
                for index, log_probability in zip(indices, log_probabilities)]
            predicted_local = associate_layers(fields, **arguments)
            predicted_tracks = [[indices[local] for local in track]
                                for track in predicted_local if len(track) >= 2]
            teacher_tracks = [list(track["entry_indices"])
                              for track in scene["tracks"]]
            predicted_pairs = _track_pairs(predicted_tracks)
            teacher_pairs = _track_pairs(teacher_tracks)
            true_positive = len(predicted_pairs & teacher_pairs)
            precision = (true_positive / len(predicted_pairs)
                         if predicted_pairs else 1.0)
            recall = true_positive / len(teacher_pairs)
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            scene_rows.append({
                "precision": precision, "recall": recall, "f1": f1,
                "exact_pair_partition": float(predicted_pairs == teacher_pairs),
                "predicted_tracks": float(len(predicted_tracks)),
                "teacher_tracks": float(len(teacher_tracks)),
            })
            totals["true_positive_pairs"] += true_positive
            totals["predicted_pairs"] += len(predicted_pairs)
            totals["teacher_pairs"] += len(teacher_pairs)
            if position == 1 or position % 100 == 0 or position == len(selected):
                print("student association scenes {}/{}".format(
                    position, len(selected)), flush=True)
    macro = {key: float(np.mean([row[key] for row in scene_rows]))
             for key in scene_rows[0]}
    global_precision = (totals["true_positive_pairs"] / totals["predicted_pairs"]
                        if totals["predicted_pairs"] else 1.0)
    global_recall = totals["true_positive_pairs"] / totals["teacher_pairs"]
    return {
        "scenes": len(scene_rows),
        "pair_counts": totals,
        "micro_pair_precision": global_precision,
        "micro_pair_recall": global_recall,
        "micro_pair_f1": (2 * global_precision * global_recall /
                          (global_precision + global_recall)
                          if global_precision + global_recall else 0.0),
        "macro_scene_metrics": macro,
        "reference_semantics": (
            "teacher-field associations are proxy labels; true source IDs are not used"),
    }


def _loaded_unit_batches(units, records, batch_size, workers, prefetch):
    """Load and decompress future track batches while CUDA handles this one."""
    unit_batches = [units[start:start + batch_size]
                    for start in range(0, len(units), batch_size)]

    def indices(batch_units):
        return [index for unit in batch_units for index in unit["indices"]]

    if workers <= 0 or prefetch <= 0:
        for batch_units in unit_batches:
            yield batch_units, [records[index] for index in indices(batch_units)]
        return
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="rind-view-set") as pool:
        pending = deque()
        iterator = iter(unit_batches)

        def submit(batch_units):
            return batch_units, [pool.submit(records.__getitem__, index)
                                 for index in indices(batch_units)]

        for _ in range(min(prefetch, len(unit_batches))):
            pending.append(submit(next(iterator)))
        while pending:
            batch_units, futures = pending.popleft()
            try:
                following = next(iterator)
            except StopIteration:
                following = None
            if following is not None:
                pending.append(submit(following))
            yield batch_units, [future.result() for future in futures]


def run_split(model, records, scene_sets, settings, device, *, optimizer=None,
              rng=None, max_scenes=0, progress_label="split"):
    import torch
    training = optimizer is not None
    model.train(training)
    if rng is None:
        rng = np.random.default_rng(0)
    units = _make_units(
        scene_sets, training=training, rng=rng,
        max_views=int(settings["max_views_per_track"]), max_scenes=max_scenes)
    if not units:
        raise ValueError("view-set split has no training units")
    rows = []
    batch_size = int(settings["scenes_per_batch"])
    progress_every = int(settings.get("progress_every_batches", 10))
    context = torch.enable_grad if training else torch.inference_mode
    loaded = _loaded_unit_batches(
        units, records, batch_size, int(settings.get("prefetch_workers", 0)),
        int(settings.get("prefetch_batches", 0)))
    with context():
        for batch_number, (batch_units, samples) in enumerate(loaded, 1):
            log_probabilities, teachers = _predict_samples(
                model, samples, device, settings.get("amp_dtype", "none"))
            student, teacher, mask = _pad_tracks(
                batch_units, log_probabilities, teachers)
            result = multiview_distillation_loss(
                student, teacher, mask,
                layer_loss_weight=float(settings["layer_loss_weight"]),
                set_loss_weight=float(settings["set_loss_weight"]),
                redundancy_discount=float(settings["redundancy_discount"]))
            if training:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["grad_clip"]),
                    error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                student_probability = student.exp()
                individual_entropy = -(student_probability * student).sum(-1)
                individual_entropy = (
                    individual_entropy.masked_fill(~mask, 0).sum(-1) /
                    mask.sum(-1).to(individual_entropy.dtype))
                teacher_fused = result["teacher_fused_probability"]
                student_fused = result["student_fused_probability"]
                teacher_entropy = -(teacher_fused * result[
                    "teacher_fused_log_probability"]).sum(-1)
                student_entropy = -(student_fused * result[
                    "student_fused_log_probability"]).sum(-1)
                overlap = 1.0 - 0.5 * torch.abs(
                    teacher_fused - student_fused).sum(-1)
                for offset, unit in enumerate(batch_units):
                    rows.append({
                        "scene_id": unit["scene_id"],
                        "objective": float(result["per_track_objective"][offset]),
                        "layer_cross_entropy": float(
                            result["per_track_layer_cross_entropy"][offset]),
                        "set_cross_entropy": float(
                            result["per_track_set_cross_entropy"][offset]),
                        "teacher_fused_entropy": float(teacher_entropy[offset]),
                        "student_fused_entropy": float(student_entropy[offset]),
                        "student_entropy_reduction": float(
                            individual_entropy[offset] - student_entropy[offset]),
                        "fused_probability_overlap": float(overlap[offset]),
                        "views": float(mask[offset].sum()),
                    })
            total_batches = math.ceil(len(units) / batch_size)
            if (batch_number == 1 or batch_number % progress_every == 0 or
                    batch_number == total_batches):
                current = _mean_rows_by_scene(rows)
                print("{} batch {}/{} | scenes={} tracks={} | objective={:.6f} "
                      "layer CE={:.6f} set CE={:.6f} overlap={:.4f}".format(
                          progress_label, batch_number, total_batches,
                          current["scenes"], current["tracks"], current["objective"],
                          current["layer_cross_entropy"], current["set_cross_entropy"],
                          current["fused_probability_overlap"]), flush=True)
    return _mean_rows_by_scene(rows)


def _checkpoint_state(base_model, optimizer, model_config, epoch, seed, history,
                      best_set_ce, bad_epochs, signature, rng, initial_checkpoint,
                      initial_validation):
    import torch
    return {
        "interface_version": INTERFACE_VERSION,
        "model_config": model_config,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "history": history,
        "best_validation_set_cross_entropy": best_set_ce,
        "bad_epochs": bad_epochs,
        "run_signature": signature,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "initial_validation": initial_validation,
        "training_semantics": "single-layer Phase1 prediction plus teacher-aligned multi-view distillation",
    }


def train(config, paths, dataset, view_index, layer_index, *, gpu_count=None,
          device_override=None, epoch_override=None, max_scenes=0):
    import torch
    settings = dict(config["training"])
    if device_override:
        settings["device"] = device_override
    if epoch_override:
        settings["epochs"] = int(epoch_override)
    torch.set_num_threads(int(settings["num_threads"]))
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if settings["device"] == "cuda":
        available = torch.cuda.device_count()
        count = int(gpu_count or len(settings.get("gpu_devices", [0])))
        if count < 1 or count > available:
            raise ValueError("requested {} GPUs but {} are visible".format(count, available))
        gpu_ids = list(range(count))
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
    else:
        gpu_ids = []
        device = torch.device("cpu")
    source_entries = layer_index["records"]
    records = {split: LayerRecords(
        source_entries[split], dataset, paths["teacher"], "world", False)
        for split in ("train", "validation", "test")}
    scene_sets = view_index["scene_sets"]
    reports = {}
    for seed in settings["seeds"]:
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if gpu_ids:
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        initial = _resolve_path(str(config["initial_checkpoint"]).format(seed=seed))
        base_model, initial_state = load_model(initial)
        if initial_state["model_config"].get("use_scene_geometry", False):
            raise ValueError("view-set training currently targets the geometry-blind v4 model")
        base_model.to(device)
        model_config = initial_state["model_config"]
        wrapped = _ChunkedStudent.create(base_model, int(settings["candidate_chunk"]))
        model = (torch.nn.DataParallel(wrapped, device_ids=gpu_ids, output_device=gpu_ids[0])
                 if len(gpu_ids) > 1 else wrapped)
        optimizer = torch.optim.AdamW(
            base_model.parameters(), lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            fused=bool(settings.get("fused_optimizer", False) and gpu_ids))
        directory = paths["output"] / "models" / "seed_{}".format(seed)
        directory.mkdir(parents=True, exist_ok=True)
        signature = _canonical_digest({
            "config": config, "view_index": view_index,
            "seed": seed, "gpu_count": len(gpu_ids),
            "device": settings["device"], "max_scenes": max_scenes,
        })
        last_path = directory / "last.pt"
        history = []
        start = 0
        best_set_ce = float("inf")
        bad = 0
        initial_validation = None
        if last_path.is_file():
            state = torch.load(last_path, map_location="cpu", weights_only=True)
            if state.get("run_signature") != signature:
                raise ValueError("existing view-set checkpoint uses another run protocol")
            base_model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            start = int(state["epoch"])
            best_set_ce = float(state["best_validation_set_cross_entropy"])
            bad = int(state["bad_epochs"])
            initial_validation = state["initial_validation"]
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if gpu_ids and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            print("resume seed={} epoch={} best set CE={:.6f}".format(
                seed, start, best_set_ce), flush=True)
        else:
            initial_validation = run_split(
                model, records["validation"], scene_sets["validation"], settings,
                device, rng=np.random.default_rng(seed), max_scenes=max_scenes,
                progress_label="initial validation")
            best_set_ce = initial_validation["set_cross_entropy"]
            state = _checkpoint_state(
                base_model, optimizer, model_config, 0, seed, history,
                best_set_ce, 0, signature, rng, initial, initial_validation)
            state["validation"] = initial_validation
            _atomic_checkpoint(directory / "best.pt", state)
            _atomic_checkpoint(last_path, state)
            print("epoch 0 baseline | validation set CE={:.6f}, layer CE={:.6f}".format(
                best_set_ce, initial_validation["layer_cross_entropy"]), flush=True)
        started = time.perf_counter()
        for epoch in range(start + 1, int(settings["epochs"]) + 1):
            if bad >= int(settings["patience"]):
                break
            train_metrics = run_split(
                model, records["train"], scene_sets["train"], settings, device,
                optimizer=optimizer, rng=rng, max_scenes=max_scenes,
                progress_label="train epoch {}".format(epoch))
            validation = run_split(
                model, records["validation"], scene_sets["validation"], settings,
                device, rng=np.random.default_rng(seed), max_scenes=max_scenes,
                progress_label="validation epoch {}".format(epoch))
            layer_guard = (validation["layer_cross_entropy"] <=
                           initial_validation["layer_cross_entropy"] +
                           float(settings["maximum_layer_ce_increase"]))
            improved = layer_guard and validation["set_cross_entropy"] < best_set_ce
            if improved:
                best_set_ce = validation["set_cross_entropy"]
            bad = 0 if improved else bad + 1
            history.append({"epoch": epoch, "train": train_metrics,
                            "validation": validation})
            state = _checkpoint_state(
                base_model, optimizer, model_config, epoch, seed, history,
                best_set_ce, bad, signature, rng, initial, initial_validation)
            state["validation"] = validation
            if improved:
                _atomic_checkpoint(directory / "best.pt", state)
            _atomic_checkpoint(last_path, state)
            _write_json(directory / "training.json", {
                "seed": seed, "history": history,
                "initial_validation": initial_validation,
                "best_validation_set_cross_entropy": best_set_ce,
                "bad_epochs": bad})
            print("epoch {}/{} done | train={:.6f} validation={:.6f} "
                  "set CE={:.6f} overlap={:.4f} layer_guard={} "
                  "best set CE={:.6f} bad={}/{}".format(
                      epoch, settings["epochs"], train_metrics["objective"],
                      validation["objective"], validation["set_cross_entropy"],
                      validation["fused_probability_overlap"], layer_guard,
                      best_set_ce, bad, settings["patience"]), flush=True)
        best_model, checkpoint = load_model(directory / "best.pt")
        best_model.to(device)
        best_wrapped = _ChunkedStudent.create(
            best_model, int(settings["candidate_chunk"]))
        best_parallel = (torch.nn.DataParallel(
            best_wrapped, device_ids=gpu_ids, output_device=gpu_ids[0])
            if len(gpu_ids) > 1 else best_wrapped)
        test_metrics = run_split(
            best_parallel, records["test"], scene_sets["test"], settings,
            device, rng=np.random.default_rng(seed), max_scenes=max_scenes,
            progress_label="test")
        report = {
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "selection": "minimum validation set CE subject to the layer-CE guard",
            "best_validation_set_cross_entropy": float(
                checkpoint["best_validation_set_cross_entropy"]),
            "initial_validation": checkpoint["initial_validation"],
            "validation": checkpoint.get("validation"),
            "test": test_metrics,
            "history": history,
            "initial_checkpoint": str(initial.resolve()),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "set_index_statistics": view_index["statistics"],
        }
        _write_json(directory / "training.json", report)
        reports[str(seed)] = report
    _write_json(paths["output"] / "training_summary.json", reports)
    return reports


def evaluate(config, paths, dataset, view_index, layer_index, *, seed=0,
             gpu_count=None, device_override=None, max_scenes=0):
    import torch
    settings = dict(config["training"])
    if device_override:
        settings["device"] = device_override
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if settings["device"] == "cuda":
        count = int(gpu_count or len(settings.get("gpu_devices", [0])))
        if count < 1 or count > torch.cuda.device_count():
            raise ValueError("invalid GPU count")
        gpu_ids = list(range(count))
        device = torch.device("cuda:0")
    else:
        gpu_ids = []
        device = torch.device("cpu")
    model_path = paths["output"] / "models" / "seed_{}".format(seed) / "best.pt"
    base, checkpoint = load_model(model_path)
    base.to(device)
    wrapped = _ChunkedStudent.create(base, int(settings["candidate_chunk"]))
    model = (torch.nn.DataParallel(wrapped, device_ids=gpu_ids, output_device=gpu_ids[0])
             if len(gpu_ids) > 1 else wrapped)
    records = {split: LayerRecords(
        layer_index["records"][split], dataset, paths["teacher"], "world", False)
        for split in ("validation", "test")}
    report = {
        "schema": "phase2-view-set-evaluation-v1",
        "checkpoint": str(model_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "source_truth_used": False,
        "association_at_training": "teacher fields and effective intensity; no true source IDs",
        "validation": run_split(
            model, records["validation"], view_index["scene_sets"]["validation"],
            settings, device, rng=np.random.default_rng(seed), max_scenes=max_scenes,
            progress_label="evaluate validation"),
        "test": run_split(
            model, records["test"], view_index["scene_sets"]["test"],
            settings, device, rng=np.random.default_rng(seed), max_scenes=max_scenes,
            progress_label="evaluate test"),
        "student_driven_association_validation": evaluate_student_association(
            model, records["validation"], layer_index["records"]["validation"],
            view_index["scene_sets"]["validation"], config, settings, device,
            max_scenes=max_scenes),
        "student_driven_association_test": evaluate_student_association(
            model, records["test"], layer_index["records"]["test"],
            view_index["scene_sets"]["test"], config, settings, device,
            max_scenes=max_scenes),
        "set_index_statistics": view_index["statistics"],
    }
    _write_json(paths["output"] / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "train", "evaluate", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="deterministic per-split cap for smoke tests; 0 uses all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.gpus is not None and args.gpus < 1:
        parser.error("--gpus must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_scenes < 0:
        parser.error("--max-scenes must be nonnegative")
    config = read_config(args.config)
    paths = resolve(config)
    dataset = Phase2Dataset(paths["data"])
    if args.command in ("index", "all"):
        build_view_set_index(
            config, paths["teacher"], paths["layer_index"], paths["output"])
    if args.command in ("train", "evaluate"):
        view_index, layer_index = load_view_set_index(
            config, paths["layer_index"], paths["output"])
    elif args.command == "all":
        view_index, layer_index = load_view_set_index(
            config, paths["layer_index"], paths["output"])
    else:
        return
    if args.command in ("train", "all"):
        train(config, paths, dataset, view_index, layer_index,
              gpu_count=args.gpus, device_override=args.device,
              epoch_override=args.epochs, max_scenes=args.max_scenes)
    if args.command == "evaluate":
        report = evaluate(config, paths, dataset, view_index, layer_index,
                          seed=args.seed, gpu_count=args.gpus,
                          device_override=args.device, max_scenes=args.max_scenes)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
