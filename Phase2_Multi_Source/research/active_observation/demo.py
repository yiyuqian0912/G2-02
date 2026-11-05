"""Run observable-only window policies on installed RIND data.

This validates acquisition and accounting, not source localization accuracy.
No trained locator/policy checkpoint is silently substituted or simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from dataset_update import open_dataset
from .active_policies import choose_action
from .observation_env import BudgetedWindowEnv, Window


class RindResponseOracle:
    def __init__(self, dataset, scene):
        self.dataset, self.scene = dataset, int(scene)
        self.calls = 0
        self.seconds = 0.

    def __call__(self, window):
        start = time.perf_counter()
        # get_region reconstructs sum_j a_j V_j from stored per-source bits.
        # Only this response crosses the acquisition interface; no GT channels.
        item = self.dataset.get_region(self.scene, window.x, window.y, window.size)
        response = np.array(item["response"], dtype=np.float64, copy=True)
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return response


def rollout(dataset, scene, initial, budget, policy, seed, max_steps):
    oracle = RindResponseOracle(dataset, scene)
    world = int(dataset.global_size)
    sizes = tuple(2**i for i in range(4, world.bit_length()) if 2**i <= world)
    env = BudgetedWindowEnv(oracle, world_size=world, budget=budget,
                            sizes=sizes, max_steps=max_steps)
    state = env.reset(initial)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    while not state.terminated:
        action = choose_action(state, env.legal_actions(), policy=policy, rng=rng)
        state = env.step(action)
    elapsed = time.perf_counter() - started
    assert state.spent == int(state.coverage.sum()) <= budget
    report = {
        "scene_id": int(scene), "seed": seed, "policy": policy,
        "budget_pixels": budget, "observed_unique_pixels": state.spent,
        "budget_utilization": state.spent / budget, "transitions": state.steps,
        "termination_reason": state.termination_reason,
        "oracle_calls": oracle.calls, "oracle_seconds": oracle.seconds,
        "policy_and_step_seconds_excluding_initial": elapsed,
        "history": [step.to_dict() for step in state.history],
        "localization_evaluated": False,
    }
    return state, report


def plot_episode(state, report, output, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#cad3dc")
    world = state.world_size
    im = axes[0].imshow(np.ma.masked_invalid(state.response), cmap=cmap,
                        origin="upper", extent=(0, world, world, 0), vmin=0)
    fig.colorbar(im, ax=axes[0], label="Observed response")
    axes[0].set_title("Acquired response only | gray = unknown")
    axes[1].imshow(state.coverage, origin="upper", cmap="Blues", vmin=0, vmax=1,
                   extent=(0, world, world, 0))
    rows = [r for r in state.history if r.window is not None]
    centers = np.array([(r.window.x + r.window.size/2,
                         r.window.y + r.window.size/2) for r in rows])
    axes[1].plot(centers[:, 0], centers[:, 1], "o-", color="tomato", markersize=3)
    for r in (rows[0], rows[-1]):
        w = r.window
        axes[1].add_patch(Rectangle((w.x, w.y), w.size, w.size, fill=False,
                                    edgecolor="black", linewidth=1.2))
    axes[1].set_title("Unique coverage and window-center path")
    if truth is not None:
        for ax in axes[:2]:
            ax.scatter(truth[:, 0], truth[:, 1], marker="+", c="lime", s=60)
        axes[1].text(.01, .01, "Green +: truth, evaluation annotation only",
                     transform=axes[1].transAxes, fontsize=8)
    for ax in axes[:2]:
        ax.set(xlim=(0, world), ylim=(world, 0), xlabel="World x", ylabel="World y")
    axes[2].step([r.step for r in state.history], [r.spent_pixels for r in state.history],
                  where="post", label="Unique pixels (initial included)")
    axes[2].axhline(state.budget, ls="--", color="gray", label="Budget")
    axes[2].set(xlabel="Action step", ylabel="Cumulative observed pixels",
                title=f"{state.spent:,}/{state.budget:,} pixels | {state.termination_reason}")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"RIND scene {report['scene_id']} | {report['policy']} | acquisition baseline, no locator")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", default=[0])
    parser.add_argument("--initial-window", type=int, nargs=3, metavar=("X", "Y", "SIZE"),
                        default=[480, 480, 64])
    parser.add_argument("--budgets", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--policies", nargs="+", choices=("random", "grow", "boundary"),
                        default=["random", "grow", "boundary"])
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--annotate-truth", action="store_true",
                        help="Source positions on figures after rollout; never policy inputs")
    parser.add_argument("--output", type=Path, default=Path("outputs/active_observation_demo"))
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    dataset = open_dataset(data_root)
    world = int(dataset.global_size)
    initial = Window(*args.initial_window)
    if any(len(values) != len(set(values)) for values in
           (args.scenes, args.budgets, args.policies)):
        parser.error("scenes, budgets and policies must not contain duplicates")
    if any(s < 0 or s >= dataset.num_scenes for s in args.scenes):
        parser.error(f"scene IDs must be in [0,{dataset.num_scenes})")
    if (initial.size < 16 or initial.size & (initial.size - 1) or min(initial.x, initial.y) < 0
            or initial.x + initial.size > world or initial.y + initial.size > world):
        parser.error("initial-window must be an in-bounds power-of-two square, size >= 16")
    if any(b < initial.area or b > world**2 for b in args.budgets):
        parser.error("every budget must cover the initial window and be <= world area")
    if args.max_steps < 1 or args.seed < 0:
        parser.error("max-steps must be positive and seed nonnegative")
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output is nonempty; choose a new directory to preserve previous runs")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = data_root / "manifest.json"
    run_config = {
        "schema": "rind-active-acquisition-demo-v1", "data_root": str(data_root),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "scenes": args.scenes, "initial_window": vars(initial), "budgets": args.budgets,
        "policies": args.policies, "seed": args.seed, "max_steps": args.max_steps,
        "annotate_truth": args.annotate_truth, "budget_unit": "unique_acquired_pixels_including_initial",
        "inference_inputs": ["observed_response", "coverage", "window_history", "remaining_budget"],
        "all_dark_or_lit_skipped": False, "source_containing_windows_skipped": False,
        "localization_evaluated": False,
    }
    (args.output / "parameters.json").write_text(json.dumps(run_config, indent=2) + "\n")
    for scene in args.scenes:
        for budget in args.budgets:
            for policy in args.policies:
                # Independent of loop order, source truth, and observation values.
                sequence = np.random.SeedSequence([args.seed, scene, budget,
                                                  ("random", "grow", "boundary").index(policy)])
                seed = int(sequence.generate_state(1)[0])
                state, record = rollout(dataset, scene, initial, budget, policy, seed, args.max_steps)
                name = f"scene_{scene}_budget_{budget}_{policy}"
                np.savez_compressed(args.output / f"{name}.npz", response=state.response,
                                    coverage=state.coverage)
                truth = dataset.get_source_params(scene) if args.annotate_truth else None
                plot_episode(state, record, args.output / f"{name}.png", truth)
                record["artifact_prefix"] = name
                (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                print(f"scene={scene} policy={policy} pixels={state.spent:,}/{budget:,} "
                      f"steps={state.steps} oracle={record['oracle_seconds']:.3f}s "
                      f"stop={state.termination_reason}", flush=True)
    (args.output / "summary.json").write_text(json.dumps({**run_config, "episodes": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
