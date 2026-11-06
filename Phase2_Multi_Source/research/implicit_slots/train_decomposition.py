#!/usr/bin/env python3
"""Train a mask-supervised decomposition student followed by shared Phase1 localization."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
PHASE2_ROOT = HERE.parents[2]
for item in (HERE.parent, PHASE2_ROOT, PHASE2_ROOT / "src",
             PHASE2_ROOT / "vendor/rind-dataset",
             PHASE2_ROOT.parent / "Phase1_Single_Source_Single_Intensity/src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from decomposition_data import (DecompositionRecords, build_decomposition_index,
                                decomposed_batches, batch_count)
from decomposition_model import ResponseDecompositionField, decomposition_training_loss, slot_activation


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PHASE2_ROOT / path


def load_config(path):
    config = json.loads(Path(path).read_text())
    if "base_config" in config:
        base = load_config(resolve(config.pop("base_config")))
        base.update(config)
        config = base
    if config.get("task") not in (None, "full", "decomposition_only"):
        raise ValueError("task must be decomposition_only or full (or omitted for full training)")
    required = ["data_root", "output_root", "model", "training", "loss"]
    if config.get("task") != "decomposition_only":
        required.append("teacher_experiment_root")
    for key in required:
        if key not in config:
            raise ValueError(f"configuration is missing {key}")
    architecture = config["model"].get("decomposition_architecture", "legacy")
    if architecture not in ("legacy", "boundary_refined_slots", "disentangled_slots_v5", "sequential_count_shape_v6"):
        raise ValueError("unknown decomposition_architecture")
    if (architecture == 'disentangled_slots_v5') != (config['loss'].get('objective') == 'disentangled_v5'):
        raise ValueError('disentangled model and objective must be enabled together')
    if architecture == 'sequential_count_shape_v6' and (config['loss'].get('objective') != 'sequential_count_shape_v6' or config.get('task') != 'decomposition_only'):
        raise ValueError('v6 requires its equal two-task objective and decomposition_only')
    if config['loss'].get('objective') == 'sequential_count_shape_v6' and architecture != 'sequential_count_shape_v6':
        raise ValueError('v6 objective requires the sequential model')
    if architecture == 'disentangled_slots_v5' and config.get('task') != 'decomposition_only':
        raise ValueError('v5 currently supports decomposition-only training')
    if architecture == 'sequential_count_shape_v6':
        allowed={'objective','count_weight','mask_weight','global_count_weight','slot_count_weight','location_weight','match_location_weight','count_class_weights','shape_class_weights'}
        if set(config['loss'])-allowed or config['loss'].get('count_weight')!=.5 or config['loss'].get('mask_weight')!=.5:
            raise ValueError('v6 requires exactly two equal tasks (.5/.5), without reconstruction/strength loss')
        for field in ('count_class_weights','shape_class_weights'):
            weights=config['loss'].get(field)
            if weights is not None and (len(weights)!=config['model']['max_slots']+1 or
                    any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in weights)):
                raise ValueError('invalid v6 training class weights: '+field)
        if config['model'].get('residual_gradient','none') not in ('none','straight_through'):
            raise ValueError('unknown residual gradient protocol')
        if config['training'].get('loss_normalization','microbatch') not in ('microbatch','optimizer_step'):
            raise ValueError('unknown loss normalization protocol')
        if config['training'].get('learning_rate_schedule') not in (None,'warmup_cosine'):
            raise ValueError('unsupported v6 learning-rate schedule')
        if (not isinstance(config['training'].get('new_module_lr_multiplier',3.),(int,float)) or
                not math.isfinite(config['training'].get('new_module_lr_multiplier',3.)) or
                config['training'].get('new_module_lr_multiplier',3.)<=0):
            raise ValueError('new-module LR multiplier must be finite and positive')
        if config['model'].get('use_obstacle') or config['model'].get('use_local_obstacle'):
            raise ValueError('v6 model inputs are response/window only; geometry input is disabled')
        if any(key in config['training'] for key in ('teacher_forcing_epochs','autonomous_every_batches')):
            raise ValueError('teacher forcing/process supervision is not permitted in v6')
    if "cache_source_root" in config and config.get("task") != "decomposition_only":
        raise ValueError("cache_source_root requires decomposition_only")
    for key in ("epochs", "patience", "batch_size", "target_global_batch_size", "candidate_chunk"):
        if type(config["training"].get(key)) is not int or config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive integer")
    if config["training"].get("amp_dtype") not in ("none", "bfloat16"):
        raise ValueError("AMP supports none or bfloat16; float16 without a scaler is disallowed")
    if not 1 <= config["model"]["max_slots"] <= 32:
        raise ValueError("max_slots must lie in [1,32]")
    for key in ("learning_rate", "grad_clip"):
        value = config["training"].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if config.get("task") == "decomposition_only":
        for key in ("location_weight", "match_location_weight"):
            if config["loss"].get(key, 0) != 0:
                raise ValueError(f"loss.{key} must be zero for decomposition_only")
        if "label_preparation" not in config:
            raise ValueError("mask-only configuration requires label_preparation")
    if config['training'].get('gpu_inputs','off') not in ('off','auto','resident','stream'):
        raise ValueError('unknown GPU input mode')
    if type(config['training'].get('defer_constraint_verification',False)) is not bool:
        raise ValueError('defer_constraint_verification must be boolean')
    if type(config['training'].get('adaptive_checkpoint_min_layers',0)) is not int or config['training'].get('adaptive_checkpoint_min_layers',0)<0:
        raise ValueError('invalid adaptive checkpoint threshold')
    runtime_batch=os.environ.get('RIND_BATCH_SIZE')
    if runtime_batch is not None:
        if not runtime_batch.isdigit() or int(runtime_batch)<1:raise ValueError('RIND_BATCH_SIZE must be positive integer')
        config['training']['batch_size']=int(runtime_batch)
    return config


def semantic_config(config):
    value = copy.deepcopy(config)
    for key in ("batch_size", "candidate_chunk", "prefetch_batches", "num_threads",
                "progress_every_batches", "gpu_devices", "activation_checkpoint_candidates",
                "activation_checkpoint_locator_encoder", "channels_last", "device",
                "gpu_inputs", "gpu_cache_fraction", "gpu_cache_reserve_gib", "gpu_upload_chunk_mib",
                "defer_constraint_verification", "adaptive_checkpoint_min_layers"):
        value["training"].pop(key, None)
    for name in ('evaluation_expansion', 'training_expansion'):
        if value.get(name):
            for key in ('workers', 'chunk_size'):
                value[name].pop(key, None)
    return value


def log(root, event, message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **fields}
    print(f"[{row['time']}] {message}", flush=True)
    with (Path(root) / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def make_model(config, world_size, device, initialize):
    model_config = config["model"]
    training = config["training"]
    masks_only = config.get("task") == "decomposition_only"
    backbone = {
        "hidden": model_config["hidden"], "fourier_frequencies": model_config["fourier_frequencies"],
        "coordinate_scale": float(world_size), "use_obstacle": model_config["use_obstacle"],
        "pixel_coordinates": True, "normalization_support": "world",
        "architecture": "multiscale_boundary", "attention_heads": 8,
    }
    architecture = model_config.get("decomposition_architecture", "legacy")
    model_class, extra = ResponseDecompositionField, {}
    if architecture in ("boundary_refined_slots", "disentangled_slots_v5", "sequential_count_shape_v6"):
        from boundary_refined_model import BoundaryRefinedDecompositionField
        model_class = BoundaryRefinedDecompositionField
        if architecture == 'disentangled_slots_v5':
            from disentangled_model import DisentangledDecompositionField
            model_class = DisentangledDecompositionField
        extra = {"refinement_rounds": model_config.get("refinement_rounds", 3),
                 "head_roles": tuple(model_config.get("head_roles", [4, 2, 2])),
                 "mask_feature_dim": model_config.get("mask_feature_dim", 128)}
    if architecture == 'sequential_count_shape_v6':
        from sequential_count_shape import SequentialCountShapeField
        model_class = SequentialCountShapeField
        extra['roundoff_ulps'] = model_config.get('roundoff_ulps',8)
        extra['residual_gradient'] = model_config.get('residual_gradient','none')
        extra['adaptive_checkpoint_min_layers'] = training.get('adaptive_checkpoint_min_layers',0)
    model = model_class(
        max_slots=model_config["max_slots"], backbone_config=backbone,
        decomposition_hidden=model_config["decomposition_hidden"],
        slot_decoder_layers=model_config.get("slot_decoder_layers", 2),
        slot_attention_heads=model_config.get("slot_attention_heads", 8),
        use_local_obstacle=model_config.get("use_local_obstacle", False),
        activation_checkpoint_candidates=training.get("activation_checkpoint_candidates", True),
        activation_checkpoint_locator_encoder=training.get("activation_checkpoint_locator_encoder", True),
        activation_checkpoint_decomposer=training.get("activation_checkpoint_decomposer", architecture == "legacy"),
        enable_localization=not masks_only,
        **extra,
    ).to(device)
    model.stable_count_output = bool(config['loss'].get('stable_count_loss', False))
    if initialize and not masks_only:
        import torch
        checkpoint_path = resolve(training["initial_phase1_checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Phase1 checkpoint missing: {checkpoint_path}")
        model.load_phase1_backbone(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    warm_start = training.get("initial_decomposition_checkpoint")
    if initialize and warm_start:
        import torch
        saved = torch.load(resolve(warm_start), map_location="cpu", weights_only=True)
        if saved.get("schema") != "phase2-local-mask-decomposition-v1":
            raise ValueError("initial_decomposition_checkpoint must be a mask-only checkpoint")
        frozen=config.get('source_training_protocol',{}).get('warm_start')
        if architecture=='sequential_count_shape_v6' and frozen:
            from evaluation_snapshot import sha256_file
            if (saved['epoch']!=frozen['epoch'] or saved['signature']!=frozen['model_signature'] or
                    sha256_file(resolve(warm_start))!=frozen['snapshot_sha256']):
                raise ValueError('frozen warm-start checkpoint changed; refusing to initialize from different weights')
        old_architecture = saved['config']['model'].get('decomposition_architecture','legacy')
        migrate = (architecture == 'disentangled_slots_v5' and old_architecture == 'boundary_refined_slots'
                   and training.get('allow_v3_warm_start') is True)
        sequential_migrate = (architecture == 'sequential_count_shape_v6' and old_architecture == 'disentangled_slots_v5'
                              and training.get('allow_v5_warm_start') is True)
        if old_architecture != architecture and not migrate and not sequential_migrate:
            raise ValueError("decomposition warm-start architecture differs; explicit v3-to-v5 migration required")
        for key in ("max_slots", "decomposition_hidden", "slot_decoder_layers", "slot_attention_heads", "use_local_obstacle"):
            if saved["config"]["model"].get(key) != model_config.get(key):
                raise ValueError(f"decomposition warm-start model.{key} differs")
        state = {key: value for key, value in saved["model"].items() if not key.startswith("locator.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        permitted = lambda key: key.startswith('locator.') or (migrate and key.startswith('global_count_head.')) or (sequential_migrate and key.startswith(('state_encoder.','context_feature.','residual_pixels.')))
        if unexpected or any(not permitted(key) for key in missing):
            raise ValueError("decomposition warm-start weights do not match")
        if sequential_migrate:
            parameters=dict(model.named_parameters())
            model.warm_start_load_report={
                'checkpoint':str(resolve(warm_start)), 'source_epoch':int(saved['epoch']),
                'source_architecture':old_architecture, 'target_architecture':architecture,
                'inherited_parameters':sum(p.numel() for name,p in parameters.items() if name in state),
                'new_parameters':sum(p.numel() for name,p in parameters.items() if name in missing),
                'new_parameter_names':sorted(name for name in missing if name in parameters),
                'optimizer_inherited':False}
        if migrate:
            expected_missing = {name for name in model.state_dict() if name.startswith('global_count_head.')}
            if set(missing) != expected_missing:
                raise ValueError('v3-to-v5 migration may initialize only the new count head')
    return model


def run_epoch(model, records, config, device, epoch, rank, world_size, optimizer=None,
              rng=None, progress=None):
    import torch
    import torch.distributed as dist
    training = config["training"]
    learning = optimizer is not None
    sampling = training.get("sampling") if learning else None
    model.train(learning)
    sequential = config['loss'].get('objective') == 'sequential_count_shape_v6'
    primary = torch.zeros(33,4,dtype=torch.int64,device=device) if sequential else None
    objective_totals = torch.zeros(4,dtype=torch.float64,device=device) if sequential else None
    totals = {}
    records_seen = 0
    micro = training["batch_size"]
    batches = batch_count(records, micro, world_size=world_size, rank=rank, training=learning, sampling=sampling)
    observed_groups = torch.zeros(4, dtype=torch.int64, device=device) if sampling else None
    count_tracking = bool(training.get('track_count_metrics') or training.get('model_selection'))
    confusion = torch.zeros((33, 33), dtype=torch.int64, device=device) if count_tracking else None
    count_mode_correct = torch.zeros((), dtype=torch.int64, device=device) if count_tracking else None
    accumulation = max(1, math.ceil(training["target_global_batch_size"] / (micro * world_size))) if learning else 1
    compute_location = (config.get("task") != "decomposition_only" and
                        epoch > int(training.get("mask_warmup_epochs", 5)))
    loss_config = dict(config["loss"])
    if not compute_location:
        loss_config["location_weight"] = 0.0
    if learning:
        optimizer.zero_grad(set_to_none=True)
    raw_batches=decomposed_batches(records,micro,rng=rng,prefetch=training.get("prefetch_batches",2),
                                  rank=rank,world_size=world_size,training=learning,sampling=sampling)
    if sequential and learning and training.get('loss_normalization')=='optimizer_step':
        from optimization_batches import optimizer_batches
        batch_stream=optimizer_batches(raw_batches,accumulation,loss_config,device,world_size)
    else:
        batch_stream=((number,batch,None) for number,batch in enumerate(raw_batches,1))
    for number,batch,normalizers in batch_stream:
        batch = {key: (value if value.device == device else
                       value.pin_memory().to(device, non_blocking=True) if device.type == "cuda"
                       else value.to(device)) for key, value in batch.items()}
        if observed_groups is not None:
            r = batch['layer_mask'].sum(1)
            observed_groups += torch.stack(((r == 1).sum(), (r == 2).sum(), ((r >= 3) & (r <= 4)).sum(), (r >= 5).sum()))
        start = ((number - 1) // accumulation) * accumulation + 1
        end = min(start + accumulation - 1, batches)
        divisor = end - start + 1
        synchronization = (model.no_sync() if learning and hasattr(model, "no_sync") and number != end
                           else nullcontext())
        with synchronization, torch.set_grad_enabled(learning):
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and training["amp_dtype"] == "bfloat16"):
                forward_options = ({'verify_constraints': False} if sequential and
                                   training.get('defer_constraint_verification', False) else {})
                result = model(batch["response"], batch["obstacle"], batch["window"],
                               batch.get("candidate_xy"), batch.get("support"),
                               candidate_chunk=training["candidate_chunk"],
                               compute_location=compute_location, **forward_options)
            runtime_loss_config=dict(loss_config)
            if normalizers is not None:runtime_loss_config['optimizer_normalizers']=normalizers
            losses=decomposition_training_loss(result,batch,**runtime_loss_config)
            if objective_totals is not None:
                objective_totals+=torch.stack([losses[k] for k in
                    ('_count_numerator','_count_denominator','_shape_numerator','_shape_denominator')]).detach().double()
            if primary is not None:
                from sequential_objective import primary_counts
                primary += primary_counts(result,batch)
            if confusion is not None:
                with torch.no_grad():
                    true_r = batch['layer_mask'].sum(1).long()
                    predicted_r = slot_activation(result).sum(1).long()
                    confusion += torch.bincount(true_r * 33 + predicted_r, minlength=33 * 33).reshape(33, 33)
                    count_mode_correct += (losses['_count_log_probability'].argmax(1) == true_r).sum()
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(f"nonfinite loss in batch {number}; scene IDs={batch['scene_id'].tolist()}")
            if learning:
                backward_loss=(losses['optimization_loss'] if normalizers is not None else losses['loss']/divisor)
                if not bool(torch.isfinite(backward_loss)):raise FloatingPointError('nonfinite optimizer-step objective')
                backward_loss.backward()
                if number == end:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("nonfinite gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        size = len(batch["response"])
        records_seen += size
        for key, value in losses.items():
            if key == "assignment" or key=='optimization_loss' or key.startswith('_') or value.numel() != 1:
                continue
            totals[key] = totals.get(key, 0.0) + value.detach().double() * size
        if progress and (number == 1 or number % training.get("progress_every_batches", 20) == 0):
            snapshot={key: float(value / records_seen) for key, value in totals.items()}
            if primary is not None:
                snapshot["layer_accuracy"]=float(primary[:,3].sum()/primary[:,2].sum().clamp_min(1))
                snapshot['count_task']=float(objective_totals[0]/objective_totals[1].clamp_min(1e-12))
                snapshot['shape_task']=float(objective_totals[2]/objective_totals[3].clamp_min(1e-12))
                snapshot['loss']=.5*(snapshot['count_task']+snapshot['shape_task'])
            progress(number,batches,snapshot)
    # Keep scalar metric accumulation on-device; copy one vector per epoch,
    # rather than synchronizing dozens of GPU scalars on every batch.
    keys=sorted(totals)
    if world_size > 1:
        key_sets=[None]*world_size
        dist.all_gather_object(key_sets,keys)
        keys=sorted(set().union(*(set(value) for value in key_sets)))
    packed=torch.stack([totals.get(key,torch.zeros((),device=device,dtype=torch.float64)) for key in keys]+
                       [torch.tensor(float(records_seen),device=device,dtype=torch.float64)])
    if world_size>1:dist.all_reduce(packed)
    values=packed.cpu().tolist();records_seen=int(values[-1])
    if not records_seen:raise ValueError('empty split')
    metrics={key:value/records_seen for key,value in zip(keys,values[:-1])}
    if objective_totals is not None:
        if world_size>1:dist.all_reduce(objective_totals)
        ct,cd,st,sd=objective_totals.cpu().tolist()
        metrics['count_task']=ct/cd;metrics['shape_task']=st/sd
        metrics['loss']=.5*(metrics['count_task']+metrics['shape_task']);metrics['mask']=metrics['shape_task']
    if confusion is not None:
        from count_objective import summarize_confusion
        if world_size > 1:
            dist.all_reduce(confusion)
            dist.all_reduce(count_mode_correct)
        metrics.update(summarize_confusion(confusion.cpu().numpy()))
        metrics['count_distribution_accuracy'] = int(count_mode_correct) / records_seen
        metrics['windows_processed'] = records_seen
    if primary is not None:
        from sequential_objective import summarize_primary
        if world_size > 1:dist.all_reduce(primary)
        metrics.update(summarize_primary(primary.cpu()))
    if observed_groups is not None:
        if world_size > 1:
            dist.all_reduce(observed_groups)
        metrics['sampled_windows_including_padding'] = records_seen
        metrics.update({f'draws_R_{name}': int(n) for name, n in zip(('1', '2', '3_4', '5plus'), observed_groups.tolist())})
    return metrics


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp.pt")
    import torch
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, output, index, *, epoch_limit=None):
    if config.get('training_expansion', {}).get('mode') == 'inventory':
        raise ValueError('inventory-only configuration cannot train; choose the training size first')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    setting = config["training"]
    if epoch_limit is not None and (type(epoch_limit) is not int or not 1 <= epoch_limit <= setting["epochs"]):
        raise ValueError("epoch_limit must be within the configured training horizon")
    horizon = setting["epochs"] if epoch_limit is None else epoch_limit
    evaluate_test = setting.get("evaluate_test_after_training", True)
    if type(evaluate_test) is not bool:
        raise ValueError("evaluate_test_after_training must be boolean")
    split_names = ("train", "validation", "test") if evaluate_test else ("train", "validation")
    masks_only = config.get("task") == "decomposition_only"
    if masks_only:
        from mask_validation import validate_mask_cache
        validate_mask_cache(index, output / "mask_cache", config["model"]["max_slots"], splits=split_names)
    cuda = setting.get("device", "cuda") == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set device=cpu for a smoke run")
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(max(1, setting.get("num_threads", 8) // world))
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if masks_only:
        from mask_only_data import MaskRecords
        record_type, cache = MaskRecords, output / "mask_cache"
    else:
        record_type, cache = DecompositionRecords, output / "decomposition_cache"
    splits = {name: record_type(index["records"][name], cache, name,
                                        config["model"]["max_slots"])
              for name in split_names}
    if setting.get('sampling'):
        if not masks_only:
            raise ValueError('layer-group sampling currently requires decomposition_only')
        from balanced_sampling import plan as sampling_plan
        _, sampling_summary = sampling_plan(splits['train'], setting['sampling'])
        if rank == 0:
            write_json(output / 'sampling_plan.json', sampling_summary)
            log(output, 'sampling', 'Training layer-group sampling: ' + json.dumps(sampling_summary))
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if first not in splits or second not in splits:
            continue
        if splits[first].scene_ids & splits[second].scene_ids:
            raise ValueError("scene leakage across splits")
    reports = {}
    for seed in setting["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        run = output / "models" / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        last = run / "last.pt"
        model = make_model(config, index.get("global_size", 1024), device, initialize=not last.is_file())
        if rank == 0:
            log(output, "model", f"architecture={config['model'].get('decomposition_architecture', 'legacy')} | parameters={sum(p.numel() for p in model.parameters()):,} | rounds={getattr(model, 'refinement_rounds', 1)} | train/validation/test="
                + "/".join(str(len(splits[n])) if n in splits else "locked" for n in ("train", "validation", "test"))
                + f" | GPU ranks={world} | AMP={setting['amp_dtype']} | locator={model.locator is not None}")
            if hasattr(model,'warm_start_load_report'):
                inherited=model.warm_start_load_report
                write_json(run/'warm_start_load.json',inherited)
                log(output,'warm_start',f"loaded source best epoch={inherited['source_epoch']} | "
                    f"inherited={inherited['inherited_parameters']:,}, new={inherited['new_parameters']:,} parameters | "
                    'new optimizer; original weights preserved')
        # Keep parameters registered throughout the schedule. Locator weights
        # are frozen with its explicit method before wrapping below.
        model.set_locator_trainable(False)
        parameter_groups = [
            {"params": [p for name, p in model.named_parameters() if not name.startswith("locator.")],
             "lr": setting["learning_rate"]},
        ]
        if config['loss'].get('objective')=='sequential_count_shape_v6':
            new_prefix=('state_encoder.','context_feature.','residual_pixels.')
            base_lr=setting['learning_rate'];new_lr=base_lr*setting.get('new_module_lr_multiplier',3.)
            parameter_groups=[
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(new_prefix)],
                 'lr':base_lr,'base_lr':base_lr},
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(new_prefix)],
                 'lr':new_lr,'base_lr':new_lr}]
        if model.locator is not None:
            parameter_groups.append({"params": list(model.locator.parameters()),
                                     "lr": setting.get("locator_learning_rate", 1e-5)})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=setting["weight_decay"],
                                      fused=cuda and setting.get("fused_optimizer", True))
        signed_index = {key: value for key, value in index.items() if key != "statistics"} if masks_only else index
        signature = digest({"config": semantic_config(config), "index": signed_index})
        history, start, best, bad = [], 0, float("inf"), 0
        initial_validation = None
        selection_best = None
        minimum_validation_loss = float('inf')
        selection_options = setting.get('model_selection')
        from count_objective import selection_key, eligible as selection_eligible
        if selection_options and not setting.get('initial_decomposition_checkpoint'):
            raise ValueError('count-first fine-tuning requires an explicit initial checkpoint')
        if selection_options and not setting.get('evaluate_initial_checkpoint', False):
            raise ValueError('count-first fine-tuning requires evaluate_initial_checkpoint=true')
        if last.is_file():
            saved = torch.load(last, map_location="cpu", weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("checkpoint/config/cache mismatch; choose a new output_root")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history, start, best, bad = saved["history"], saved["epoch"], saved["best"], saved["bad"]
            best = float("inf") if best is None else best
            rng.bit_generator.state = saved["numpy_rng"]
            initial_validation = saved.get("initial_validation")
            minimum_validation_loss = saved.get('minimum_validation_loss', best)
            selection_best = saved.get('selection_key')
            if selection_best is None:
                if selection_options:
                    raise ValueError('count-first checkpoint lacks selection state')
                selection_best = (-best,)
            selection_best = tuple(selection_best)
            if rank==0:log(output,'resume',f'resuming new experiment after epoch={start}; model and optimizer restored from {last}')
        if cuda and config['loss'].get('objective')=='sequential_count_shape_v6':
            from sequential_objective import verify_effective_assignment
            checked=verify_effective_assignment(device)
            write_json(output/'runtime/matching'/f'seed{seed}-rank{rank}.json',checked)
            log(output,'matching_preflight',f'rank={rank} effective-layer matching parity passed on {device}')
        if cuda and masks_only and setting.get('gpu_inputs', 'off') != 'off':
            from gpu_mask_inputs import enable_gpu_inputs
            for split_name, records in splits.items():
                pipeline = enable_gpu_inputs(records, device, setting)
                if pipeline is not None:
                    write_json(output/'runtime/gpu_inputs'/f'seed{seed}-rank{rank}-{split_name}.json',pipeline)
                    log(output,'gpu_inputs',f'rank={rank} split={split_name} input={pipeline["mode"]} '
                        f'packed={pipeline["packed_cache_bytes"]/2**30:.2f} GiB; GPU bit decode; '
                        f'{pipeline["reason"]}')
        if os.environ.get('RIND_AUTO_BATCH','0') == '1':
            from gpu_runtime import calibrate_batch
            calibrate_batch(model,splits['train'],config,device,rank,world,output,seed)
        if not last.is_file() and masks_only and setting.get("initial_decomposition_checkpoint") and setting.get("evaluate_initial_checkpoint", False):
            # Keep the old best model as epoch 0 on the exact same validation
            # set. Expanded fine-tuning must beat this score to replace it.
            initial_validation = run_epoch(model, splits["validation"], config, device, 0, rank, world)
            best = initial_validation["loss"]
            minimum_validation_loss = best
            selection_best = selection_key(initial_validation, selection_options)
            if rank == 0:
                state = {"schema": "phase2-local-mask-decomposition-v1", "config": config,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": 0, "best": best, "bad": 0, "history": [], "signature": signature,
                         "selection_key": list(selection_best), "minimum_validation_loss": minimum_validation_loss,
                         "selection_options": selection_options,
                         "numpy_rng": rng.bit_generator.state, "initial_validation": initial_validation}
                save_checkpoint(run / "best.pt", state)
                if selection_options:
                    save_checkpoint(run / "loss_best.pt", state)
                log(output, "initial_validation", f"fine-tuning baseline | validation loss={best:.6f}; selection={selection_options or 'validation_loss'}; initial best retained as epoch 0", metrics=initial_validation)
            if world > 1:
                dist.barrier()
        wrapped = None
        last_trainable = None
        started = time.perf_counter()
        for epoch in range(start + 1, horizon + 1):
            if bad >= setting["patience"]:
                break
            if config['loss'].get('objective')=='sequential_count_shape_v6' and setting.get('learning_rate_schedule')=='warmup_cosine':
                from sequential_objective import learning_rate_scale
                factor=learning_rate_scale(epoch,setting['epochs'],min(setting.get('warmup_epochs',2),setting['epochs']-1),setting.get('minimum_lr_ratio',.1))
                for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
                if rank==0:log(output,'learning_rate',f"epoch={epoch} learning rates={[g['lr'] for g in optimizer.param_groups]}")
            trainable = not masks_only and epoch > setting.get("locator_frozen_epochs", 15)
            if trainable != last_trainable or wrapped is None:
                if world > 1:
                    dist.barrier()
                # Recreate DDP when the trainable parameter set changes; its
                # reducer cannot gain newly unfrozen parameters in-place.
                wrapped = None
                model.set_locator_trainable(trainable)
                wrapped = (DistributedDataParallel(
                    model, device_ids=[local] if cuda else None, broadcast_buffers=False,
                    gradient_as_bucket_view=True, find_unused_parameters=not masks_only,
                ) if world > 1 else model)
                last_trainable = trainable
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            if rank == 0:
                accumulation = math.ceil(setting["target_global_batch_size"] / (setting["batch_size"] * world))
                log(output, "epoch_start", f"decomposition epoch={epoch}/{setting['epochs']} | task={config.get('task', 'full')} | train/val/test={len(splits['train'])}/{len(splits['validation'])}/{len(splits['test']) if evaluate_test else 'locked'} | slots={config['model']['max_slots']} | GPU ranks={world} | micro={setting['batch_size']} accumulate={accumulation} effective_global_batch={setting['batch_size'] * world * accumulation} | locator_trainable={trainable} | location_active={not masks_only and epoch > setting['mask_warmup_epochs']}")
            def callback(phase):
                def emit(number, total, metrics):
                    detail = ''
                    if config['loss'].get('objective') == 'disentangled_v5':
                        detail = (f" | count/global={metrics['count']:.4f}/{metrics['count_global']:.4f}"
                                  f" shape={metrics['mask']:.4f} strength={metrics['intensity']:.4f}"
                                  f" recon supervised/deploy={metrics['reconstruction_supervised_normalized']:.4f}/"
                                  f"{metrics['reconstruction_deployment_normalized']:.4f}"
                                  f" strict={metrics['strict_decomposition_success']:.4f}")
                    if config['loss'].get('objective') == 'sequential_count_shape_v6':
                        detail = (f" | equal tasks count/shape={metrics['count_task']:.4f}/{metrics['shape_task']:.4f}"
                                  f" layer_acc={metrics.get('layer_accuracy',0):.4f} autonomous=True")
                    if config['loss'].get('objective')=='sequential_count_shape_v6':
                        message=f"{phase} epoch={epoch} batch={number}/{total} loss={metrics['loss']:.5f} count_acc={metrics['count_accuracy']:.4f}"+detail
                    else:
                        message=f"{phase} epoch={epoch} batch={number}/{total} loss={metrics['loss']:.5f} mask_dice={metrics.get('mask_dice',0):.4f} count_acc={metrics.get('count_accuracy',0):.4f}"+detail
                    log(output,"batch",message,epoch=epoch,phase=phase,batch=number,metrics=metrics)
                return emit if rank == 0 else None
            training_metrics = run_epoch(wrapped, splits["train"], config, device, epoch, rank, world,
                                         optimizer, rng, callback("train"))
            # Evaluate through the unwrapped module: unequal rank partitions
            # must not trigger DDP forward collectives during evaluation.
            validation = run_epoch(model, splits["validation"], config, device, epoch, rank, world,
                                   progress=callback("validation"))
            # All post-warmup epochs share the same loss. Warmup checkpoints
            # cannot become the final best just because they omit localization.
            eligible = masks_only or epoch > setting["mask_warmup_epochs"]
            loss_improved = validation['loss'] < minimum_validation_loss
            minimum_validation_loss = min(minimum_validation_loss, validation['loss'])
            candidate_key = selection_key(validation, selection_options)
            allowed = selection_eligible(validation, initial_validation, selection_options)
            improved = eligible and allowed and (selection_best is None or candidate_key > selection_best)
            if improved:
                best, bad = validation["loss"], 0
                selection_best = candidate_key
            elif eligible:
                bad += 1
            memory = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0,
                                  device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            record = {"epoch": epoch, "train": training_metrics, "validation": validation,
                      "seconds": time.perf_counter() - tick, "peak_allocated_gib": float(memory)}
            if selection_options:
                record.update(selection_key=list(candidate_key), selection_eligible=allowed,
                              selected_as_best=bool(improved))
            history.append(record)
            if rank == 0:
                state = {"schema": "phase2-local-mask-decomposition-v1" if masks_only else "phase2-response-decomposition-v2", "config": config,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best": best if math.isfinite(best) else None,
                         "bad": bad, "history": history, "signature": signature,
                         "selection_key": list(selection_best), "minimum_validation_loss": minimum_validation_loss,
                         "selection_options": selection_options,
                         "numpy_rng": rng.bit_generator.state, "initial_validation": initial_validation}
                save_checkpoint(last, state)
                if improved:
                    save_checkpoint(run / "best.pt", state)
                if selection_options and loss_improved:
                    save_checkpoint(run / "loss_best.pt", state)
                write_json(run / "training.json", {"history": history, "best_validation_loss": best if math.isfinite(best) else None})
                if selection_options:
                    groups = ', '.join(f"{name}={100*value['accuracy']:.2f}% (n={value['windows']})"
                        for name, value in validation['count_by_group'].items() if value['accuracy'] is not None)
                    log(output, 'count_selection', f"validation macro count={100*validation['count_group_macro']:.2f}% | {groups} | eligible={allowed} selected={improved}")
                if config['loss'].get('objective')=='sequential_count_shape_v6':
                    log(output,'primary_metrics',f"validation autonomous count_acc={validation['count_accuracy']:.4f} layer_acc={validation['layer_accuracy']:.4f} | count/shape group macro={validation['count_group_macro']:.4f}/{validation['shape_group_macro']:.4f}",by_group=validation['primary_by_group'])
                quality=(f"count_acc={validation['count_accuracy']:.4f} layer_acc={validation['layer_accuracy']:.4f}" if
                         config['loss'].get('objective')=='sequential_count_shape_v6' else f"mask_dice={validation.get('mask_dice',0):.4f}")
                log(output, "epoch_done", f"epoch={epoch} train={training_metrics['loss']:.5f} val={validation['loss']:.5f} {quality} best={best:.5f} bad={bad}/{setting['patience']} peak={float(memory):.2f} GiB", **record)
            if world > 1:
                dist.barrier()
        if not (run / "best.pt").is_file():
            raise RuntimeError("no post-warmup checkpoint; epochs must exceed mask_warmup_epochs")
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        test = (run_epoch(model, splits["test"], config, device, saved["epoch"], rank, world)
                if evaluate_test else None)
        report = {"seed": seed, "best_epoch": saved["epoch"], "test": test,
                  "test_evaluated": evaluate_test,
                  "trained_through_epoch": history[-1]["epoch"] if history else start,
                  "best_validation_loss": minimum_validation_loss, "initial_validation": initial_validation,
                  "selected_validation_loss": saved['best'], "selection_options": selection_options,
                  "selected_validation_count_group_macro": (saved['history'][-1]['validation'].get('count_group_macro')
                        if saved['history'] else (initial_validation or {}).get('count_group_macro')),
                  "selected_validation_count_accuracy": (saved['history'][-1]['validation']['count_accuracy']
                        if saved['history'] else (initial_validation or {}).get('count_accuracy')),
                  "records": {name: len(value) for name, value in splits.items()},
                  "elapsed_seconds": time.perf_counter() - started}
        if config['loss'].get('objective')=='sequential_count_shape_v6':
            selected=(saved['history'][-1]['validation'] if saved['history'] else initial_validation or {})
            report['selected_validation_layer_accuracy']=selected.get('layer_accuracy')
            report['selected_validation_shape_group_macro']=selected.get('shape_group_macro')
        if rank == 0:
            write_json(run / "training.json", {**report, "history": history})
            reports[str(seed)] = report
            display_test = ({k:v for k,v in test.items() if isinstance(v,(int,float))} if test is not None else None)
            log(output, "done", f"decomposition done best_epoch={saved['epoch']} test={display_test}")
    if rank == 0:
        write_json(output / "training_summary.json", reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "train", "all"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    masks_only = config.get("task") == "decomposition_only"
    inventory_only = config.get('training_expansion', {}).get('mode') == 'inventory'
    if inventory_only and args.command == 'train':
        raise ValueError('inventory-only configuration cannot train; finish the inventory and choose a size first')
    if args.command in ("index", "all"):
        if masks_only:
            if config.get('training_expansion'):
                if not inventory_only:
                    checkpoint = resolve(config['training']['initial_decomposition_checkpoint'])
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f'initial trained checkpoint missing: {checkpoint}; preserve the source experiment')
                from expand_complex_masks import prepare_complex
                index = prepare_complex(config)
                if inventory_only:
                    return  # Even command=all is inventory only; no model training.
            elif config.get('evaluation_expansion'):
                checkpoint = resolve(config['training']['initial_decomposition_checkpoint'])
                if not checkpoint.is_file():
                    raise FileNotFoundError(f'initial best checkpoint missing: {checkpoint}; keep the completed original experiment for warm-start and reference evaluation')
                from extend_evaluation import prepare_extended_evaluation
                index = prepare_extended_evaluation(config)
            elif config.get("cache_source_root"):
                from reuse_mask_cache import prepare_cached_labels
                index = prepare_cached_labels(config)
            elif config["label_preparation"].get("reuse_from"):
                from expand_masks import prepare_expanded
                index = prepare_expanded(config)
            else:
                from mask_only_data import prepare_mask_dataset
                index = prepare_mask_dataset(config)
        else:
            from dataset_update.dataset_union import open_dataset
            dataset = open_dataset(resolve(config["data_root"]))
            index = build_decomposition_index(config, dataset, resolve(config["teacher_experiment_root"]), output)
    else:
        name = "mask_records.json" if masks_only else "decomposition_records.json"
        index = json.loads((output / name).read_text())
    if args.command in ("train", "all"):
        train(config, output, index)
        if masks_only and config["training"].get("evaluate_after_training", False):
            if config['loss'].get('objective') == 'sequential_count_shape_v6':
                from evaluate_sequential import evaluate_with_reference
            else:
                from evaluate_masks import evaluate_with_reference
            for seed in config["training"]["seeds"]:
                if config['training'].get('evaluate_validation_metrics', False):
                    evaluate_with_reference(config, output, index, seed=seed,
                        repetitions=config['training'].get('evaluation_bootstrap_repetitions', 1000), split='validation')
                evaluate_with_reference(config, output, index, seed=seed,
                         repetitions=config["training"].get("evaluation_bootstrap_repetitions", 1000))


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
