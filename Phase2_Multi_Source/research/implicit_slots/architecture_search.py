#!/usr/bin/env python3
"""Cache-only, validation-selected slot/width/depth/LR search and automatic fit.

One independent pilot per idle GPU; the selected architecture then uses DDP.
No new physical teacher, source identities, or locked-test score is used to
choose a trial. Optimum means best within this finite search and budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
PHASE2 = HERE.parents[2]
for path in (HERE.parent, PHASE2, PHASE2 / "src", PHASE2 / "vendor/rind-dataset",
             PHASE2 / "research/layered_residual", PHASE2.parent / "Phase1_Single_Source_Single_Intensity/src"):
    sys.path.insert(0, str(path))

from search_cache import (digest, write_json, immutable_json, capacity_audit,
                          make_pilot, bind_cache)
from mask_validation import validate_mask_cache


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PHASE2 / path).resolve()


def load_search(path):
    cfg = json.loads(Path(path).read_text())
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("search seed must be a nonnegative integer")
    if not cfg["source_candidates"] or any(not isinstance(p, str) or not p for p in cfg["source_candidates"]):
        raise ValueError("source_candidates must contain cache directories")
    if type(cfg.get("pilot_global_batch_size", 128)) is not int or cfg.get("pilot_global_batch_size", 128) < 1:
        raise ValueError("pilot_global_batch_size must be a positive integer")
    for key in ("slot_counts", "hidden_widths", "decoder_depths", "learning_rates", "epoch_budgets"):
        values = cfg[key]
        if not values or len(set(values)) != len(values) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"invalid search.{key}")
    for key in ("slot_counts", "hidden_widths", "decoder_depths", "epoch_budgets"):
        if any(type(v) is not int for v in cfg[key]):
            raise ValueError(f"search.{key} must contain integers")
    if any(s > 32 for s in cfg["slot_counts"]) or cfg["epoch_budgets"] != sorted(cfg["epoch_budgets"]):
        raise ValueError("slots must be <=32; budgets must strictly increase")
    if type(cfg["attention_heads"]) is not int or cfg["attention_heads"] < 1 or any(w < 16 or w % cfg["attention_heads"] for w in cfg["hidden_widths"]):
        raise ValueError("hidden widths must be >=16 and divisible by heads")
    for key in ("max_trials", "reduction_factor"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "reduction_factor" else 1):
            raise ValueError(f"invalid search.{key}")
    for key in ("train", "validation"):
        if type(cfg["pilot_records"][key]) is not int or cfg["pilot_records"][key] < 1:
            raise ValueError("pilot record counts must be positive integers")
    weights = cfg["score_weights"]
    if (set(weights) != {"count_accuracy", "mask_iou", "reconstruction_quality"} or
            any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()) or
            not math.isclose(sum(weights.values()), 1)):
        raise ValueError("score weights must be nonnegative and sum to one")
    if not 0 <= cfg["score_tolerance"] <= 1:
        raise ValueError("invalid score tolerance")
    auto = cfg["autobatch"]
    if (type(auto["enabled"]) is not bool or not 0 < auto["memory_fraction"] < 1 or
            type(auto["steps"]) is not int or auto["steps"] < 1 or
            any(type(b) is not int or b < 1 for b in auto["batch_sizes"]) or not auto["batch_sizes"]):
        raise ValueError("invalid autobatch settings")
    final = cfg["final_training"]
    if any(type(final[k]) is not int or final[k] < 1 for k in ("epochs", "patience")) or not final["seeds"] or any(type(s) is not int or s < 0 for s in final["seeds"]):
        raise ValueError("invalid final training settings")
    thresholds = cfg["success_thresholds"]
    if (set(thresholds) != {"mask_iou", "relative_strength", "relative_reconstruction"} or
            any(not math.isfinite(v) or v < 0 for v in thresholds.values()) or thresholds["mask_iou"] > 1):
        raise ValueError("invalid strict success thresholds")
    if type(cfg["bootstrap_repetitions"]) is not int or cfg["bootstrap_repetitions"] < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    return cfg


def trial_grid(cfg, eligible):
    grid = list(itertools.product(eligible, cfg["hidden_widths"], cfg["decoder_depths"], cfg["learning_rates"]))
    rng = random.Random(cfg["seed"])
    rng.shuffle(grid)
    # A controlled slot sweep at the smallest common architecture is always
    # included; the remainder explores model capacity and optimization rate.
    reference_lr = min(cfg["learning_rates"], key=lambda x: abs(x - 1e-4))
    anchors = list(dict.fromkeys(
        [(s, min(cfg["hidden_widths"]), min(cfg["decoder_depths"]), reference_lr) for s in eligible] +
        [(s, max(cfg["hidden_widths"]), max(cfg["decoder_depths"]), reference_lr) for s in eligible]))
    if cfg["max_trials"] < len(anchors):
        raise ValueError("max_trials must include the controlled sweep of every eligible slot count")
    chosen = (anchors + [row for row in grid if row not in anchors])[:cfg["max_trials"]]
    return [{"id": f"trial_{i:03d}_s{s}_d{w}_l{depth}_lr{lr:g}", "slots": s,
             "hidden": w, "depth": depth, "learning_rate": lr} for i, (s, w, depth, lr) in enumerate(chosen)]


def trial_config(template, cfg, trial, output, *, final=False):
    result = copy.deepcopy(template)
    result.update(name=trial["id"] + ("_full" if final else "_pilot"), task="decomposition_only",
                  output_root=str(output), evaluation={"capacity_overflow": "missing_predictions",
                  "success_thresholds": cfg["success_thresholds"]})
    result["model"].update(max_slots=trial["slots"], decomposition_hidden=trial["hidden"],
                           slot_decoder_layers=trial["depth"], slot_attention_heads=cfg["attention_heads"])
    t = result["training"]
    for key in ("initial_decomposition_checkpoint", "initial_phase1_checkpoint", "evaluate_initial_checkpoint"):
        t.pop(key, None)
    # Every candidate and the final fit start from scratch; changing slot count
    # never slices old learned queries or silently reuses incompatible weights.
    t.update(seeds=cfg["final_training"]["seeds"] if final else [0],
             epochs=cfg["final_training"]["epochs"] if final else max(cfg["epoch_budgets"]),
             patience=cfg["final_training"]["patience"] if final else max(cfg["epoch_budgets"]) + 1,
             learning_rate=trial["learning_rate"], evaluate_after_training=False,
             evaluate_test_after_training=False, mask_warmup_epochs=0, locator_frozen_epochs=0)
    if not final:
        t["target_global_batch_size"] = cfg.get("pilot_global_batch_size", 128)
    result["loss"].update(location_weight=0.0, match_location_weight=0.0)
    return result


def prepare(cfg, *, device="cuda"):
    from train_decomposition import load_config
    output = resolve(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "search_protocol.json"
    previous = json.loads(protocol_file.read_text()) if protocol_file.exists() else None
    # Lock the original cache across interruptions, even if another expansion
    # finishes while this search is running.
    source = Path(previous["source_cache"]) if previous else next(
        (resolve(v) for v in cfg["source_candidates"] if (resolve(v) / "mask_records.json").is_file()), None)
    if source is None:
        raise FileNotFoundError("No completed mask_records.json in source_candidates. Prepare masks_only_v1 or masks_only_100k first; raw data and physical teachers are not required here.")
    index = json.loads((source / "mask_records.json").read_text())
    validate_mask_cache(index, source / "mask_cache", index["max_slots"], splits=("train", "validation"))
    template = load_config(resolve(cfg["template_config"]))
    if template.get("task") != "decomposition_only":
        raise ValueError("search template must be decomposition_only")
    source_signature = digest({k: v for k, v in index.items() if k != "statistics"})
    protocol = {"schema": "rind-slot-architecture-search-v1", "config": cfg, "template": template, "device": device,
                "source_cache": str(source), "source_index_sha256": source_signature,
                "code_sha256": {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (
                    HERE, HERE.with_name("search_cache.py"), HERE.with_name("decomposition_model.py"),
                    HERE.with_name("train_decomposition.py"), HERE.with_name("evaluate_masks.py"),
                    HERE.with_name("mask_only_data.py"), HERE.with_name("mask_validation.py"),
                    HERE.with_name("decomposition_data.py"), HERE.with_name("slot_model.py"))}}
    immutable_json(protocol_file, protocol)
    audit = capacity_audit(index, cfg["slot_counts"])
    write_json(output / "capacity_audit.json", audit)
    eligible = [r["slots"] for r in audit["candidates"] if r["eligible"]]
    if not eligible:
        raise ValueError("No candidate capacity fits all train/development labels; see capacity_audit.json. Labels were not filtered.")
    trials = trial_grid(cfg, eligible)
    immutable_json(output / "trials.json", trials)
    pilot_root = output / "pilot_data"
    pilot = make_pilot(source, index, pilot_root, cfg["pilot_records"], cfg["seed"])
    for trial in trials:
        root = output / "trials" / trial["id"]
        bind_cache(root, pilot_root, pilot, trial["slots"])
        config = trial_config(template, cfg, trial, root)
        config["training"]["device"] = device
        if device == "cpu":
            config["training"].update(device="cpu", amp_dtype="none", batch_size=2, prefetch_batches=0)
        immutable_json(root / "config.json", config)
    write_json(output / "plan.json", {"source_cache": str(source), "source_counts": index["counts"],
               "pilot_counts": pilot["counts"], "eligible_slots": eligible, "trial_count": len(trials),
               "epoch_budgets": cfg["epoch_budgets"], "test_used_for_selection": False})
    print(f"SEARCH PLAN | cache={source} | full train/development={index['counts']['train']}/{index['counts']['validation']} | common pilot={pilot['counts']} | eligible slots={eligible} | trials={len(trials)}", flush=True)
    return output, source, index, trials, template


def select_gpus(args):
    if args.device == "cpu":
        return [None]
    if args.gpu_ids:
        ids = args.gpu_ids.split(",")
        if any(not i.isdigit() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid --gpu-ids")
        return ids
    data = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in processes.splitlines() if "," in line}
    ids = []
    for line in data.splitlines():
        i, uuid, memory, use = (v.strip() for v in line.split(","))
        if uuid not in occupied and int(memory) <= args.max_used_memory_mib and int(use) <= args.max_utilization:
            ids.append(i)
    if args.gpus != "auto":
        if not args.gpus.isdigit() or int(args.gpus) < 1:
            raise ValueError("--gpus must be auto or a positive integer")
        ids = ids[:int(args.gpus)]
    if not ids:
        raise RuntimeError("No idle GPU. No CPU fallback or occupation of busy GPUs.")
    return ids


def quality_score(report, cfg):
    means = {k: v["mean"] for k, v in report["scene_balanced"].items()}
    quality = {"count_accuracy": means["count_accuracy"], "mask_iou": means["mask_iou"],
               "reconstruction_quality": max(0, 1 - means["reconstruction_relative_l1"])}
    value = sum(cfg["score_weights"][k] * v for k, v in quality.items())
    if not math.isfinite(value):
        raise ValueError("nonfinite validation score")
    return value


def child_env(gpus, device):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else ",".join(gpus)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return env


def worker_command(config_path, search_path, *, budget=None, final=False, evaluate_only=False, gpus=None):
    command = [sys.executable, "-u"]
    if gpus and len(gpus) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}"]
    command += [str(HERE), "worker", "--config", str(search_path), "--trial-config", str(config_path)]
    if budget is not None:
        command += ["--budget", str(budget)]
    if final:
        command += ["--final"]
    if evaluate_only:
        command += ["--evaluate-only"]
    return command


def run_rung(active, budget, output, cfg, args, gpus):
    pending = list(active)
    running = {}
    successes = []
    failures = []
    try:
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                trial = pending.pop(0)
                root = output / "trials" / trial["id"]
                result = root / f"result_epoch_{budget}.json"
                if result.exists():
                    successes.append(json.loads(result.read_text()))
                    continue
                stream = (root / f"epoch_{budget}.log").open("a", buffering=1)
                proc = subprocess.Popen(worker_command(root / "config.json", args.config, budget=budget),
                                        env=child_env([gpu] if gpu is not None else [], args.device),
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                running[gpu] = (proc, stream, trial)
                print(f"START | {trial['id']} | cumulative epochs={budget} | GPU={gpu} | log={root / f'epoch_{budget}.log'}", flush=True)
            for gpu, (proc, stream, trial) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                root = output / "trials" / trial["id"]
                result_file = root / f"result_epoch_{budget}.json"
                if code == 0 and result_file.exists():
                    result = json.loads(result_file.read_text())
                    successes.append(result)
                    print(f"DONE | {trial['id']} | validation quality={result['score']:.6f} | params={result['parameters']:,}", flush=True)
                else:
                    failure = {"id": trial["id"], "exit_code": code, "log": str(root / f"epoch_{budget}.log")}
                    failures.append(failure)
                    print(f"FAILED | {failure}", flush=True)
                del running[gpu]
            write_json(output / "search_status.json", {"budget": budget,
                       "running": [r[2]["id"] for r in running.values()], "pending": len(pending),
                       "completed": len(successes), "failed": failures})
            if running:
                time.sleep(.5)
    finally:
        # Ctrl-C to tmux must stop all independently launched trial workers too.
        import signal
        for proc, stream, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            stream.close()
    if not successes:
        raise RuntimeError(f"All trials failed at budget {budget}; inspect per-trial logs; no winner was invented")
    successes.sort(key=lambda r: (-r["score"], -r["strict_success"], r["parameters"], r["id"]))
    write_json(output / f"rung_{budget}.json", {"results": successes, "failures": failures})
    return successes


def run_search(output, trials, cfg, args, gpus):
    selected = output / "selected.json"
    if selected.exists():
        return json.loads(selected.read_text())
    active = trials
    for i, budget in enumerate(cfg["epoch_budgets"]):
        results = run_rung(active, budget, output, cfg, args, gpus)
        if i + 1 < len(cfg["epoch_budgets"]):
            keep = max(1, math.ceil(len(results) / cfg["reduction_factor"]))
            ids = {r["id"] for r in results[:keep]}
            active = [trial for trial in trials if trial["id"] in ids]
    near = [r for r in results if r["score"] >= results[0]["score"] - cfg["score_tolerance"]]
    winner = min(near, key=lambda r: (r["parameters"], r["slots"], -r["score"]))
    result = {"trial": next(t for t in trials if t["id"] == winner["id"]), "validation": winner,
              "score_tolerance": cfg["score_tolerance"], "test_used_for_selection": False,
              "selection_scope": "finite grid/subset, fixed seed, fixed pilot cohorts and successive-halving budgets"}
    immutable_json(selected, result)
    lines = ["# 槽位与模型参数搜索", "", f"选择：`{winner['id']}`；参数：{winner['parameters']:,}；验证质量分数：{winner['score']:.6f}。", "",
             "在预先规定的分数容差内优先较少参数。未使用测试集；这是有限预算搜索结果，不是全局最优保证。", "",
             "| 配置 | 验证质量 | 严格整窗成功率 | 参数量 |", "|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['id']} | {row['score']:.6f} | {row['strict_success']:.6f} | {row['parameters']:,} |")
    (output / "search_report.zh-CN.md").write_text("\n".join(lines) + "\n")
    return result


def probe_batch(config, cfg, output, device):
    """Benchmark GPU forward/loss/backward; the probe never changes fit weights."""
    import torch
    from mask_only_data import MaskRecords
    from decomposition_data import decomposed_batches
    from train_decomposition import make_model
    from decomposition_model import decomposition_training_loss
    auto = cfg["autobatch"]
    saved = output / "batch_probe.json"
    if saved.exists():
        return json.loads(saved.read_text())["selected_batch"]
    if device.type != "cuda" or not auto["enabled"]:
        return config["training"]["batch_size"]
    index = json.loads((output / "mask_records.json").read_text())
    rows = index["records"]["train"] + index["records"]["validation"]
    side = max(r["size"] for r in rows)
    rows = [r for r in rows if r["size"] == side]
    model = make_model(config, index["global_size"], device, initialize=False).train()
    options = sorted(b for b in set(auto["batch_sizes"]) if config["training"]["target_global_batch_size"] % b == 0)
    if not options:
        raise ValueError("pilot autobatch sizes must include a divisor of target_global_batch_size")
    measurements = []
    for micro in options:
        # Probe from a single split so offsets always refer to the right file.
        split = "train" if any(r["size"] == side for r in index["records"]["train"]) else "validation"
        candidates = [r for r in index["records"][split] if r["size"] == side]
        entries = [candidates[i % len(candidates)] for i in range(micro)]
        records = MaskRecords(entries, output / "mask_cache", split, config["model"]["max_slots"])
        batch = result = loss = None
        try:
            batch = next(iter(decomposed_batches(records, micro, prefetch=0, training=False)))
            batch = {k: v.to(device) for k, v in batch.items()}
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for step in range(auto["steps"] + 1):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["training"]["amp_dtype"] == "bfloat16"):
                    result = model.decompose(batch["response"], batch["obstacle"], batch["window"])
                loss = decomposition_training_loss(result, batch, **config["loss"])["loss"]
                loss.backward()
                torch.cuda.synchronize(device)
                if step:
                    times.append(time.perf_counter() - tick)
            # Adam moments are not created by this no-update probe; reserve
            # their FP32 bytes and leave the configured VRAM margin.
            peak = torch.cuda.max_memory_allocated(device) + 8 * sum(p.numel() for p in model.parameters())
            total = torch.cuda.get_device_properties(device).total_memory
            within = peak <= auto["memory_fraction"] * total
            measurements.append({"batch": micro, "windows_per_second": micro / (sum(times) / len(times)),
                                 "estimated_peak_bytes_with_adam": peak, "within_memory_budget": within})
            if not within:
                break
        except torch.cuda.OutOfMemoryError:
            measurements.append({"batch": micro, "out_of_memory": True})
            break
        finally:
            batch = result = loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    feasible = [m for m in measurements if m.get("within_memory_budget")]
    if not feasible:
        raise RuntimeError("No batch fits GPU memory budget; reduce autobatch.batch_sizes")
    selected = max(feasible, key=lambda m: m["windows_per_second"])["batch"]
    write_json(saved, {"selected_batch": selected, "gpu": torch.cuda.get_device_name(device),
                       "worst_native_window_size": side, "measurements": measurements,
                       "scope": "single-GPU compute probe; no optimizer update; disk/DDP costs excluded"})
    return selected


def worker(args, cfg):
    import torch
    import torch.distributed as dist
    from train_decomposition import load_config, train
    from evaluate_masks import evaluate
    config = load_config(args.trial_config)
    output = resolve(config["output_root"])
    index = json.loads((output / "mask_records.json").read_text())
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local}" if config["training"]["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, config["training"].get("num_threads", 8) // max(world, 2)))
    if args.final:
        selected = json.loads((output.parent / "selected.json").read_text())
        probe = output.parent / "trials" / selected["trial"]["id"] / "batch_probe.json"
        cap = json.loads(probe.read_text())["selected_batch"] if probe.exists() else config["training"]["batch_size"]
        # Keep effective batch close to the requested value across GPU counts.
        per_rank = max(1, math.ceil(config["training"]["target_global_batch_size"] / (world * 16)) * 16)
        config["training"]["batch_size"] = max(b for b in range(1, min(cap, per_rank) + 1) if per_rank % b == 0)
    else:
        config["training"]["batch_size"] = probe_batch(config, cfg, output, device)
    try:
        if not args.evaluate_only:
            train(config, output, index, epoch_limit=args.budget if not args.final else None)
        seeds = config["training"]["seeds"]
        for seed in seeds:
            report = evaluate(config, output, index, seed=seed,
                              repetitions=cfg["bootstrap_repetitions"] if args.final else 0,
                              split="test" if args.final else "validation")
            if rank == 0 and not args.final:
                saved = torch.load(output / f"models/seed_{seed}/best.pt", map_location="cpu", weights_only=True)
                summary = json.loads((output / "training_summary.json").read_text())[str(seed)]
                means = {k: v["mean"] for k, v in report["scene_balanced"].items()}
                history = json.loads((output / f"models/seed_{seed}/training.json").read_text())["history"]
                result = {"id": output.name, "slots": config["model"]["max_slots"],
                          "score": quality_score(report, cfg), "strict_success": means["strict_decomposition_success"],
                          "parameters": sum(v.numel() for v in saved["model"].values()),
                          "best_epoch": saved["epoch"], "trained_through_epoch": summary["trained_through_epoch"],
                          "budget": args.budget, "validation": means,
                          "training_seconds": sum(r["seconds"] for r in history),
                          "test_evaluated": False}
                write_json(output / f"result_epoch_{args.budget}.json", result)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def fit(output, source, index, template, cfg, args, gpus, *, evaluate_only=False):
    if not (output / "selected.json").exists():
        raise ValueError("Run search/all first; no validation-selected architecture exists")
    selected = json.loads((output / "selected.json").read_text())
    trial = selected["trial"]
    root = output / "best_full"
    bind_cache(root, source, index, trial["slots"])
    final_config = trial_config(template, cfg, trial, root, final=True)
    final_config["training"]["device"] = args.device
    if args.device == "cpu":
        final_config["training"].update(device="cpu", amp_dtype="none", batch_size=2, prefetch_batches=0)
    immutable_json(root / "config.json", final_config)
    if (root / "fit_complete.json").exists() and not evaluate_only:
        print(f"Full fit already completed: {root}", flush=True)
        return
    gpu_args = [] if args.device == "cpu" else gpus
    command = worker_command(root / "config.json", args.config, final=True, evaluate_only=evaluate_only, gpus=gpu_args)
    write_json(output / "final_status.json", {"state": "running", "selected_trial": trial["id"], "gpu_ids": gpu_args})
    # The tmux worker captures all output. Full training is foreground so Ctrl-C
    # reaches torchrun and all DDP ranks as well as the coordinator.
    subprocess.run(command, env=child_env(gpu_args, args.device), check=True)
    results = {}
    for seed in final_config["training"]["seeds"]:
        relative = Path("best_full/evaluation")
        if len(final_config["training"]["seeds"]) > 1:
            relative = relative / f"seed_{seed}"
        relative = relative / "decomposition_metrics.json"
        report = json.loads((output / relative).read_text())
        results[str(seed)] = {"report": str(relative), "best_epoch": report["best_epoch"],
                             "test_windows": report["windows"], "capacity_overflow_windows": report["capacity_overflow_windows"],
                             "scene_balanced": report["scene_balanced"]}
    write_json(output / "final_summary.json", {"selected_trial": trial, "records": index["counts"],
               "test_used_for_selection": False, "test_results": results})
    write_json(root / "fit_complete.json", {"selected_trial": trial["id"], "seeds": final_config["training"]["seeds"],
               "test_evaluated_after_selection": True})
    write_json(output / "final_status.json", {"state": "complete", "output": str(root)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "search", "fit", "all", "evaluate", "worker"))
    parser.add_argument("--config", type=Path, default=PHASE2 / "configs/slot_search.json")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--max-used-memory-mib", type=int, default=2048)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--trial-config", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    cfg = load_search(args.config)
    if args.command == "worker":
        if args.trial_config is None or (not args.final and args.budget is None):
            parser.error("worker requires --trial-config and --budget (or --final)")
        worker(args, cfg)
        return
    if args.max_used_memory_mib < 0 or not 0 <= args.max_utilization <= 100:
        parser.error("invalid GPU selection thresholds")
    root = resolve(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".search.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is running this search output_root") from exc
        output, source, index, trials, template = prepare(cfg, device=args.device)
        if args.command == "plan":
            return
        gpus = select_gpus(args)
        print(f"GPU PLAN | independent pilot workers={len(gpus)} | devices={gpus} | final DDP ranks={len(gpus)}", flush=True)
        if args.command in ("search", "all"):
            run_search(output, trials, cfg, args, gpus)
        if args.command in ("fit", "all", "evaluate"):
            if args.command == "all" and not args.gpu_ids:
                gpus = select_gpus(args)
                print(f"FULL FIT GPU RECHECK | available devices={gpus}", flush=True)
            fit(output, source, index, template, cfg, args, gpus, evaluate_only=args.command == "evaluate")


if __name__ == "__main__":
    main()
