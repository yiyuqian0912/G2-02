#!/usr/bin/env python3
"""Cache-only decomposition metrics. Never instantiate a Phase1 locator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from train_decomposition import load_config, make_model, resolve, write_json, digest, semantic_config
from mask_only_data import MaskRecords
from mask_validation import validate_mask_cache
from decomposition_data import decomposed_batches
from decomposition_model import decomposition_training_loss, slot_activation, contribution_gate
from evaluate_slots import _summarize, _scene_balanced, _bootstrap, _stratify


def pad_missing_predictions(result, slots):
    """Represent capacity overflow with absent predictions, never drop targets.

    Used for locked-test evaluation only. Extra rows have zero mask, strength
    and existence; reconstruction and actual predicted count stay unchanged.
    """
    current = result["mask_probability"].shape[1]
    if slots < current:
        raise ValueError("cannot truncate predictions")
    if slots == current:
        return result
    padded = dict(result)
    for key, fill in (("mask_probability", 0), ("mask_logits", -30),
                      ("presence_probability", 0), ("presence_logits", -30), ("intensity", 0),
                      *((('active_slots',False),) if 'active_slots' in result else ())):
        value = result[key]
        shape = (value.shape[0], slots - current, *value.shape[2:])
        padded[key] = torch.cat((value, value.new_full(shape, fill)), dim=1)
    padded["count_logits"] = torch.nn.functional.pad(result["count_logits"], (0, slots - current), value=-1e9)
    for key in ('global_count_logits','slot_count_logits'):
        if key in result:padded[key]=torch.nn.functional.pad(result[key],(0,slots-current),value=-1e9)
    if "refinement_outputs" in result:
        padded["refinement_outputs"] = [pad_missing_predictions(value, slots)
                                        for value in result["refinement_outputs"]]
    return padded


def metric_rows(result, batch, assignment, thresholds=None):
    """Compare masks at 0.5 after Hungarian alignment; presence evaluated separately."""
    free = (~batch["obstacle"])[:, None]
    gather = assignment[..., None, None].expand_as(batch["teacher_mask"])
    predicted = result["mask_probability"].gather(1, gather) >= .5
    target = batch["teacher_mask"].bool()
    intersection = (predicted & target & free).sum((-1, -2)).double()
    area_p = (predicted & free).sum((-1, -2)).double()
    area_t = (target & free).sum((-1, -2)).double()
    real = batch["layer_mask"].bool()
    positive = real.sum(1).clamp_min(1)
    def mean_layers(value):
        return (value * real).sum(1) / positive
    presence = slot_activation(result).gather(1, assignment)
    a = result["intensity"].gather(1, assignment)
    tp = (presence & real).sum(1)
    predicted_count = presence.sum(1)
    true_count = real.sum(1)
    count_probability = result["count_logits"].argmax(1)
    free_pixels = free[:, 0].sum((-1, -2)).clamp_min(1)
    error = (result["reconstruction"] - batch["response"]).abs()
    contribution = (result["mask_probability"].gather(1, gather)
                    * result["intensity"].gather(1, assignment)[..., None, None]
                    * contribution_gate(result).gather(1, assignment)[..., None, None])
    predicted_residual = batch["response"][:, None] - (contribution * real[..., None, None]).cumsum(1)
    target_residual = batch["response"][:, None] - (
        target * (batch["teacher_intensity"] * real)[..., None, None]).cumsum(1)
    thresholds = thresholds or {"mask_iou": .9, "relative_strength": .1, "relative_reconstruction": .05}
    iou = intersection / (area_p + area_t - intersection).clamp_min(1)
    strength_relative = (a - batch["teacher_intensity"]).abs() / batch["teacher_intensity"].clamp_min(1e-12)
    reconstruction_relative = (error * free[:, 0]).sum((-1, -2)) / (
        batch["response"] * free[:, 0]).sum((-1, -2)).clamp_min(1e-12)
    all_masks = ((iou >= thresholds["mask_iou"]) | ~real).all(1)
    all_strengths = ((strength_relative <= thresholds["relative_strength"]) | ~real).all(1)
    presence_exact = (presence == real).all(1)
    metric = {
        "mask_dice": mean_layers((2 * intersection) / (area_p + area_t).clamp_min(1)),
        "mask_iou": mean_layers(intersection / (area_p + area_t - intersection).clamp_min(1)),
        "mask_precision": mean_layers(intersection / area_p.clamp_min(1)),
        "mask_recall": mean_layers(intersection / area_t.clamp_min(1)),
        "strength_mae": mean_layers((a - batch["teacher_intensity"]).abs()),
        "strength_relative_mae": mean_layers(strength_relative),
        "strength_within_10pct": mean_layers((strength_relative<=.1).double()),
        "presence_precision": tp / predicted_count.clamp_min(1),
        "presence_recall": tp / true_count.clamp_min(1),
        "count_accuracy": (predicted_count == true_count).double(),
        "count_mae": (predicted_count - true_count).abs().double(),
        "count_distribution_accuracy": (count_probability == true_count).double(),
        "strict_decomposition_success": (presence_exact & all_masks & all_strengths &
            (reconstruction_relative <= thresholds["relative_reconstruction"])).double(),
        "all_layers_iou_pass": all_masks.double(),
        "all_layers_strength_pass": all_strengths.double(),
        "reconstruction_mae": (error * free[:, 0]).sum((-1, -2)) / free_pixels,
        "reconstruction_relative_l1": (error * free[:, 0]).sum((-1, -2)) /
            (batch["response"] * free[:, 0]).sum((-1, -2)).clamp_min(1e-12),
        "residual_mae": mean_layers(((predicted_residual - target_residual).abs()
                                      * free).sum((-1, -2)) / free_pixels[:, None]),
        "negative_residual_fraction": mean_layers(((predicted_residual < -1e-6)
                                                    & free).sum((-1, -2)) / free_pixels[:, None]),
    }
    values = torch.stack(list(metric.values()), 1).detach().cpu().numpy()
    scenes, views, windows = (batch[key].cpu().numpy() for key in ("scene_id", "view_id", "window"))
    true_count, predicted_count = true_count.cpu().numpy(), predicted_count.cpu().numpy()
    return [{"scene_id": int(scenes[i]), "view_id": int(views[i]), "size": int(windows[i, 2]),
             "layer_count": int(true_count[i]), "predicted_count": int(predicted_count[i]),
             "metrics": dict(zip(metric, map(float, values[i])))} for i in range(len(values))]


def evaluate(config, output, index, seed=0, repetitions=1000, *, split="test", reference_checkpoint=None,
             checkpoint_kind='best', diagnostics_only=False, diagnostics_root=None):
    if config.get("task") != "decomposition_only":
        raise ValueError("evaluate_masks requires task=decomposition_only")
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    slots = config["model"]["max_slots"]
    overflow_allowed = config.get("evaluation", {}).get("capacity_overflow") == "missing_predictions"
    active_splits = ("train", "validation", "test") if split == "test" else ("train", "validation")
    validate_mask_cache(index, output / "mask_cache", slots, splits=active_splits,
                        allow_overflow_splits=("test",) if overflow_allowed and split == "test" else ())
    thresholds = config.get("evaluation", {}).get("success_thresholds", {
        "mask_iou": .9, "relative_strength": .1, "relative_reconstruction": .05})
    if (set(thresholds) != {"mask_iou", "relative_strength", "relative_reconstruction"} or
            any(not np.isfinite(v) or v < 0 for v in thresholds.values()) or thresholds["mask_iou"] > 1):
        raise ValueError("invalid strict success thresholds")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = config["training"].get("device", "cuda") == "cuda"
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(max(1, config["training"].get("num_threads", 8) // world))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    tick = time.perf_counter()
    if checkpoint_kind not in ('best','last'):raise ValueError('checkpoint_kind must be best or last')
    checkpoint = (resolve(reference_checkpoint) if reference_checkpoint is not None
                  else output / "models" / f"seed_{seed}" / (checkpoint_kind+'.pt'))
    diagnostic_base=(resolve(diagnostics_root) if diagnostics_root is not None else output/'diagnostics')
    from evaluation_snapshot import load_snapshot
    saved,checkpoint_metadata=load_snapshot(checkpoint,diagnostic_base,rank,world)
    checkpoint_metadata.update(checkpoint_kind='initial_reference' if reference_checkpoint is not None else checkpoint_kind,
                               split=split,seed=seed,experiment_root=str(output.resolve()))
    if rank==0:print(f"Evaluation snapshot | role={checkpoint_metadata['checkpoint_kind']} epoch={saved['epoch']} split={split} "
                     f"sha256={checkpoint_metadata['source_checkpoint_sha256'][:16]}",flush=True)
    if saved["schema"] != "phase2-local-mask-decomposition-v1":
        raise ValueError("not a mask-only checkpoint")
    # Use the checkpoint's exact architecture, rather than silently accepting config edits.
    if saved["config"]["model"] != config["model"] and not (
            reference_checkpoint is not None and config['model'].get('decomposition_architecture')=='disentangled_slots_v5'
            and config['training'].get('allow_v3_warm_start') is True
            and saved['config']['model'].get('decomposition_architecture')=='boundary_refined_slots'):
        raise ValueError("checkpoint/model config mismatch")
    training_index = index
    if reference_checkpoint is not None:
        expected = config['training'].get('initial_decomposition_checkpoint')
        if not expected or checkpoint.resolve() != resolve(expected).resolve():
            raise ValueError('reference must be the explicitly configured initial checkpoint')
        training_index = json.loads((resolve(config['cache_source_root']) / 'mask_records.json').read_text())
        extension = index.get('training_expansion') or index.get('evaluation_extension', {})
        verified=extension.get('source_index_sha256')==digest(training_index)
        if not verified and index.get('cache_view'):
            reuse=json.loads((output/'cache_reuse_protocol.json').read_text())
            verified=(reuse['source_index_sha256']==digest(training_index) and
                      index['records']==training_index['records'] and index['counts']==training_index['counts'])
        if not verified:raise ValueError('reference evaluation requires the verified original label index')
    signed_index = {key: value for key, value in training_index.items() if key != "statistics"}
    if saved["signature"] != digest({"config": semantic_config(saved["config"]), "index": signed_index}):
        raise ValueError("checkpoint/cache mismatch")
    model = make_model(saved["config"], index["global_size"], device, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    rows_in = index["records"][split]
    overflow = sum(len(row["intensities"]) > slots for row in rows_in)
    target_slots = max(slots, max(len(row["intensities"]) for row in rows_in))
    records = MaskRecords(rows_in, output / "mask_cache", split, target_slots)
    rows = []
    with torch.inference_mode():
        for number, batch in enumerate(decomposed_batches(
                records, config["training"]["batch_size"], prefetch=config["training"].get("prefetch_batches", 2),
                rank=rank, world_size=world, training=False), 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=cuda and config["training"]["amp_dtype"] == "bfloat16"):
                result = model.decompose(batch["response"], batch["obstacle"], batch["window"])
            result = pad_missing_predictions(result, target_slots)
            # Matching is part of the trained experiment, not a runtime metric
            # knob. Editing the launch config must not silently change which
            # predicted layer gets compared with which teacher layer.
            if config['model'].get('decomposition_architecture')=='disentangled_slots_v5':
                from disentangled_objective import evaluation_assignment
                assignment=evaluation_assignment(result,batch)
            else:
                assignment = decomposition_training_loss(result, batch, **saved["config"]["loss"])["assignment"]
            batch_rows=metric_rows(result,batch,assignment,thresholds)
            from failure_diagnostics import diagnostic_rows
            detailed=diagnostic_rows(result,batch,assignment,
                fixed_matching=config['model'].get('decomposition_architecture')=='disentangled_slots_v5')
            for row,details in zip(batch_rows,detailed):row['diagnostics']=details
            rows.extend(batch_rows)
            if rank == 0 and (number == 1 or number % 20 == 0):
                print(f"mask evaluation rank0: batches={number} local windows={len(rows)}", flush=True)
    if world > 1:
        collected = [None] * world
        dist.all_gather_object(collected, rows)
        rows = [row for part in collected for row in part]
    if rank != 0:
        return None
    if len(rows) != len(records) or len({(row['scene_id'], row['view_id']) for row in rows}) != len(rows):
        raise ValueError("evaluation duplicates or misses windows")
    rows.sort(key=lambda row: (row["scene_id"], row["view_id"]))
    cohorts = {}
    legacy_count = index.get('evaluation_extension', {}).get('legacy_counts', {}).get(split)
    if legacy_count is not None:
        legacy_ids = {(r['scene_id'], r['view_id']) for r in rows_in[:legacy_count]}
        for row in rows:
            row['cohort'] = 'legacy' if (row['scene_id'], row['view_id']) in legacy_ids else 'added'
        for name in ('legacy', 'added'):
            selected = [row for row in rows if row['cohort'] == name]
            cohorts[name] = {'windows': len(selected), 'scenes': len({r['scene_id'] for r in selected}),
                             'window_weighted': _summarize(selected), 'scene_balanced': _scene_balanced(selected),
                             'by_layer_count': _stratify(selected, 'layer_count'),
                             'scene_bootstrap_95ci': _bootstrap(selected, list(selected[0]['metrics']) if selected else [], repetitions, seed)}
    report = {
        "checkpoint": checkpoint_metadata,
        "task": "decomposition_only", "evaluated_epoch":saved['epoch'],
        "best_epoch": saved["epoch"] if checkpoint_kind=='best' else None, "seed": seed,
        "split": split, "model_slots": slots, "capacity_overflow_windows": overflow,
        "capacity_overflow_policy": "missing_predictions; full denominator; no target truncation",
        "success_thresholds": thresholds,
        "windows": len(rows), "scenes": len(records.scene_ids),
        "candidate_renders": 0, "location_evaluated": False,
        "label_semantics": "minimum-positive/BFS effective layers; physical feasibility unchecked",
        "window_weighted": _summarize(rows), "scene_balanced": _scene_balanced(rows),
        "scene_bootstrap_95ci": _bootstrap(rows, list(rows[0]["metrics"]) if rows else [], repetitions, seed),
        "by_layer_count": _stratify(rows, "layer_count"), "by_window_size": _stratify(rows, "size"),
        "elapsed_seconds": time.perf_counter() - tick,
        "mask_threshold": .5, "presence_threshold": None if 'active_slots' in result else .5,
        "activation_rule": 'predicted_count_argmax_then_top_slots' if 'active_slots' in result else 'presence>=0.5',
        "evaluation_matching": ('shape_strength_v5' if config['model'].get('decomposition_architecture')=='disentangled_slots_v5' else 'training_assignment'),
        "matching": ('fixed v5 shape/strength Hungarian for model and reference' if
                     config['model'].get('decomposition_architecture')=='disentangled_slots_v5' else
                     'same exact Hungarian mask/strength/presence assignment as training; no location scores'),
        "matching_loss_config": saved["config"]["loss"],
        'checkpoint_role': ('initial_reference' if reference_checkpoint is not None else 'training_last_snapshot' if checkpoint_kind=='last' else 'selected_model'),
        'by_cohort': cohorts,
    }
    if config['training'].get('track_count_metrics') or config['training'].get('model_selection'):
        from count_objective import summarize_confusion
        confusion = np.zeros((33, 33), dtype=np.int64)
        for row in rows:
            confusion[row['layer_count'], row['predicted_count']] += 1
        report['count_selection_metrics'] = summarize_confusion(confusion)
    destination = output / "evaluation"
    if len(config["training"].get("seeds", [0])) > 1:
        destination = destination / f"seed_{seed}"
    if reference_checkpoint is not None:
        destination = destination / 'initial_reference'
    if diagnostics_only:
        destination=(diagnostic_base/output.name/f'seed_{seed}'/
            f"{checkpoint_kind}_{split}_epoch{saved['epoch']}_{checkpoint_metadata['source_checkpoint_sha256'][:12]}")
    destination.mkdir(parents=True,exist_ok=True)
    stem = "decomposition_metrics" if split == "test" else "validation_metrics"
    write_json(destination / f"{stem}.json", report)
    (destination / ("windows.jsonl" if split == "test" else "validation_windows.jsonl")).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    lines = ["# 局部分层解耦评估", "", f"评估checkpoint：{checkpoint_metadata['checkpoint_kind']}；epoch={saved['epoch']}；{split}视窗：{len(rows)}；场景：{len(records.scene_ids)}。", "",
             f"模型槽数：{slots}；超过容量的窗口：{overflow}。超容量窗口保留，缺少的层按未预测处理。", "",
             f"严格整窗成功：所有实际启用槽位与教师层一一对应，每层IoU≥{thresholds['mask_iou']}、强度相对误差≤{thresholds['relative_strength']}，总响应相对L1≤{thresholds['relative_reconstruction']}。", "",
             "仅评估分层掩码、强度、有效层数与响应重建；没有渲染或评估位置概率场。", "",
             "标签是既有最低正残差/BFS规则产生的有效层；重建正确不保证每层可由一个物理源实现。", "",
             "掩码指标按训练时 Hungarian 匹配计算，单独报告槽位激活与层数，避免把已匹配的掩码质量当成端到端检出率。", "",
             "| 指标 | 场景等权均值 | 95% 场景 bootstrap 区间 |", "|---|---:|---|" ]
    for key, value in report["scene_balanced"].items():
        ci = report["scene_bootstrap_95ci"].get(key)
        interval = f"[{ci['lower']:.6f}, {ci['upper']:.6f}]" if ci else "—"
        lines.append(f"| {key} | {value['mean']:.6f} | {interval} |")
    (destination / f"{stem}.zh-CN.md").write_text("\n".join(lines) + "\n")
    from failure_diagnostics import aggregate,write_artifacts,portable_bundle
    details=[row['diagnostics'] for row in rows]
    import hashlib
    audit_metadata={**checkpoint_metadata,'cache_corpus_sha256':index.get('corpus_sha256'),
                    'evaluated_window_list_sha256':digest([{k:r[k] for k in ('scene_id','view_id','window')} for r in rows_in]),
                    'diagnostic_code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('evaluate_masks.py','failure_diagnostics.py','evaluation_snapshot.py','decomposition_model.py')},
                    'original_success_thresholds':thresholds,
                    'original_matching':report['evaluation_matching'],'geometry_independence_verified':False}
    analysis=aggregate(details,repetitions,seed,audit_metadata)
    audit_directory=destination if diagnostics_only else destination/(stem+'_diagnostics')
    write_artifacts(audit_directory,details,analysis)
    write_json(audit_directory/'checkpoint_metadata.json',audit_metadata)
    if diagnostics_only:
        bundle=portable_bundle(audit_directory,diagnostic_base,audit_metadata)
        print(f'Diagnostic download bundle: {bundle}',flush=True)
    print(f"mask evaluation saved: {destination}", flush=True)
    return report


def evaluate_with_reference(config, output, index, seed=0, repetitions=1000, *, split='test'):
    report = evaluate(config, output, index, seed, repetitions, split=split)
    if not config['training'].get('compare_initial_checkpoint', False):
        return report
    reference = evaluate(config, output, index, seed, repetitions, split=split,
                         reference_checkpoint=config['training']['initial_decomposition_checkpoint'])
    if report is None:
        return None
    destination = output / 'evaluation'
    if len(config['training'].get('seeds', [0])) > 1:
        destination = destination / f'seed_{seed}'
    summary = {'windows': report['windows'], 'reference_epoch': reference['best_epoch'],
               'selected_finetune_epoch': report['best_epoch'], 'metrics': {}, 'by_layer_count': {}}
    for key, value in report['scene_balanced'].items():
        before, after = reference['scene_balanced'][key]['mean'], value['mean']
        summary['metrics'][key] = {'before': before, 'after': after, 'change': after - before}
    for r, values in report['by_layer_count'].items():
        summary['by_layer_count'][r] = {key: {'before': reference['by_layer_count'][r][key]['mean'],
                                            'after': value['mean'],
                                            'change': value['mean'] - reference['by_layer_count'][r][key]['mean']}
                                       for key, value in values.items()}
    if report.get('count_selection_metrics') and reference.get('count_selection_metrics'):
        before, after = (v['count_selection_metrics']['count_group_macro'] for v in (reference,report))
        summary['count_group_macro'] = {'before':before, 'after':after, 'change':after-before}
    stem = 'finetuning_comparison' if split == 'test' else 'validation_finetuning_comparison'
    write_json(destination / f'{stem}.json', summary)
    lines = ['# 同一扩充测试集：微调前后对比', '', f"窗口数：{report['windows']}。初始checkpoint epoch={reference['best_epoch']}；微调所选epoch={report['best_epoch']}（0表示保留初始权重）。", '',
             '测试只在训练完成后运行，不参与模型选择。旧窗口和新增窗口的独立统计见两个指标JSON的by_cohort。', '',
             '| 指标 | 微调前 | 微调后 | 后减前 |', '|---|---:|---:|---:|']
    for key, values in summary['metrics'].items():
        lines.append(f"| {key} | {values['before']:.6f} | {values['after']:.6f} | {values['change']:+.6f} |")
    (destination / f'{stem}.zh-CN.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--split',choices=('validation','test'))
    parser.add_argument('--diagnostics-only',action='store_true')
    parser.add_argument('--diagnostics-root',type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve(config["output_root"])
    index=json.loads((output/'mask_records.json').read_text())
    if config['loss'].get('objective')=='sequential_count_shape_v6':
        from evaluate_sequential import evaluate_with_reference as sequential_evaluate
        if args.checkpoint=='last' and not args.diagnostics_only:
            parser.error('last snapshot requires --diagnostics-only; final evaluation uses best')
        splits=([args.split] if args.split else ['validation'] if args.diagnostics_only else ['validation','test'])
        for split in splits:
            sequential_evaluate(config,output,index,args.seed,args.bootstrap_repetitions,split=split,checkpoint_kind=args.checkpoint)
        return
    if args.diagnostics_only:
        evaluate(config,output,index,args.seed,args.bootstrap_repetitions,split=args.split or 'validation',
                 checkpoint_kind=args.checkpoint,diagnostics_only=True,diagnostics_root=args.diagnostics_root)
        return
    if args.checkpoint!='best':
        parser.error('last checkpoint diagnostics require --diagnostics-only; formal evaluation remains best')
    splits=([args.split] if args.split else
            (['validation','test'] if config['training'].get('evaluate_validation_metrics',False) else ['test']))
    for split in splits:
        evaluate_with_reference(config,output,index,args.seed,args.bootstrap_repetitions,split=split)



if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
