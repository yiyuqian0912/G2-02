"""Completed joint physical costs -> soft targets with explicit sampling mass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from rind_phase1_multi.data import PROJECT_ROOT, Phase1MultiDataset
from rind_phase1_multi.physics import JointResponseEvaluator
from rind_phase1_multi.search import evaluate_joint_support, sample_joint_support, uniform_source_grid


class NoValidCandidatesError(ValueError):
    pass


class IncompleteSearchError(ValueError):
    pass


def _validate_temperature(temperature):
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def teacher_probabilities(physical_cost, valid, *, temperature, base_mass):
    """q_i proportional to mass_i * exp(-C_i/tau); preserve original order.

    Fixed-K equal-mass sampling reduces to the usual softmax. Stratification
    uses pi_K/N_K so unequal proposal counts do not silently define the K prior.
    Invalid entries have zero probability, regardless of their stored costs.
    """
    costs = np.asarray(physical_cost, dtype=np.float64)
    mask, mass = np.asarray(valid), np.asarray(base_mass, dtype=np.float64)
    if (costs.ndim != 1 or mask.shape != costs.shape or mask.dtype != np.dtype(bool) or
            mass.shape != costs.shape or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError("Costs, boolean validity and finite nonnegative base_mass must align")
    _validate_temperature(temperature)
    if not np.isfinite(costs[mask]).all() or np.any(costs[mask] < 0):
        raise ValueError("Valid physical costs must be finite and nonnegative")
    active = mask & (mass > 0)
    if not active.any():
        raise NoValidCandidatesError("No valid positive-mass candidate on the declared joint support")
    selected = costs[active]
    # Subtract before division, so tiny tau cannot underflow every candidate.
    with np.errstate(over="ignore", under="ignore"):
        log_weights = np.log(mass[active]) - (selected - selected.min()) / float(temperature)
        weights = np.exp(log_weights - log_weights.max())
    probabilities = np.zeros_like(costs)
    probabilities[active] = weights / weights.sum(dtype=np.float64)
    return probabilities


def target_from_search(result, *, temperature):
    if not result.complete or not result.evaluated.all():
        raise IncompleteSearchError(
            f"Incomplete joint support ({int(result.evaluated.sum())}/{len(result.support)} evaluated): "
            f"{result.failure_reason or 'unfinished'}; no final teacher target")
    if len(result.support) != len(result.physical_cost):
        raise ValueError("Joint source sets and physical costs are misaligned")
    return teacher_probabilities(result.physical_cost, result.valid, temperature=temperature,
                                 base_mass=result.support.base_mass)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generate_joint_teacher(dataset, scene_id, view_id, *, spacing, temperature, source_counts,
                           samples_per_count, count_prior=None, seed, backend="auto",
                           max_evaluations=None, max_source_renders=None, run_id=None):
    """Sample/evaluate complete configurations without reading truth or witnesses.

    Completion refers to this finite sampled support, not exhaustive coverage
    of the continuous inverse set. All research sampling settings are explicit.
    """
    _validate_temperature(temperature)
    evaluator = JointResponseEvaluator(dataset, scene_id, view_id, backend=backend,
                                       max_source_renders=max_source_renders)
    grid = uniform_source_grid(evaluator.domain, evaluator, spacing=spacing)
    support = sample_joint_support(grid, source_counts=source_counts, samples_per_count=samples_per_count,
                                   count_prior=count_prior, seed=seed)
    result = evaluate_joint_support(support, evaluator, max_evaluations=max_evaluations)
    probabilities = target_from_search(result, temperature=temperature)
    provenance = {"manifest": dataset.manifest, "manifest_sha256": _digest(dataset.manifest)}
    installation = Path(dataset.root) / "installation.json"
    if installation.is_file():
        provenance["archive_sha256"] = json.loads(installation.read_text())["archive_sha256"]
    code_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("data.py", "physics.py", "search.py", "teacher.py")}
    vendor = PROJECT_ROOT / "vendor/rind-dataset"
    for name in ("SOURCE_SNAPSHOT.json", "rind_dataset/dataset.py", "rind_dataset/runtime.py",
                 "rind_dataset/render.py", "rind_dataset/geometry.py", "rind_dataset/constants.py"):
        code_hashes[f"vendor/{name}"] = hashlib.sha256((vendor / name).read_bytes()).hexdigest()
    settings = {"dataset": provenance, "support": support.metadata, "physics": evaluator.cost_config,
                "temperature": float(temperature), "search_mode": support.metadata["search_mode"],
                "max_evaluations": None if max_evaluations is None else int(max_evaluations),
                "max_source_renders": None if max_source_renders is None else int(max_source_renders),
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "code_sha256": code_hashes}
    settings["uv_lock_sha256"] = hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()
    config_id = _digest(settings)
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a nonempty string")
    run_id = run_id or f"scene{scene_id}-view{view_id}-{config_id[:12]}"
    metadata = {"record_schema_version": 1, "scene_id": int(scene_id), "view_id": int(view_id),
                "config_id": config_id, "run_id": run_id, "complete": True,
                "completion_scope": "declared finite Monte Carlo support; coverage is not exhaustive",
                "dataset_root": str(dataset.root), "settings": settings,
                "num_joint_evaluations": result.num_joint_evaluations,
                "num_source_renders": result.num_source_renders, "num_cache_hits": result.num_cache_hits,
                "visibility_cache_bytes": evaluator.visibility_cache_bytes,
                "elapsed_seconds": result.elapsed_seconds, "reference_solutions_used_as_targets": False}
    return {"sources": support.sources.copy(), "source_counts": support.source_counts.copy(),
            "base_mass": support.base_mass.copy(), "multiplicity": support.multiplicity.copy(),
            "physical_cost": result.physical_cost.copy(), "valid": result.valid.copy(),
            "evaluated": result.evaluated.copy(), "teacher_prob": probabilities,
            "window": np.array(evaluator.domain.window, dtype=np.int64),
            "position_xy": grid.positions.copy(), "position_valid": grid.valid.copy(),
            "position_grid_index": grid.grid_index.copy(), "x_axis": grid.x_axis.copy(),
            "y_axis": grid.y_axis.copy(), "metadata": metadata}


RECORD_FIELDS = ("sources", "source_counts", "base_mass", "multiplicity", "physical_cost", "valid",
                 "evaluated", "teacher_prob", "window", "position_xy", "position_valid",
                 "position_grid_index", "x_axis", "y_axis")


def _validate_record(record):
    if record["metadata"].get("record_schema_version") != 1:
        raise ValueError("Unsupported joint teacher record schema")
    counts, sources = np.asarray(record["source_counts"]), np.asarray(record["sources"])
    if (counts.ndim != 1 or not len(counts) or counts.dtype.kind not in "iu" or sources.ndim != 3 or
            sources.shape[0] != len(counts) or sources.shape[2] != 3 or
            np.any(counts <= 0) or np.any(counts > sources.shape[1])):
        raise ValueError("Teacher record has malformed padded source sets/counts")
    for name in ("base_mass", "multiplicity", "physical_cost", "valid", "evaluated", "teacher_prob"):
        if np.asarray(record[name]).shape != counts.shape:
            raise ValueError(f"Teacher record has misaligned {name}")
    positions = np.asarray(record["position_xy"])
    indices = np.asarray(record["position_grid_index"])
    position_valid = np.asarray(record["position_valid"])
    if (positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all() or
            indices.shape != positions.shape or indices.dtype.kind not in "iu" or
            position_valid.shape != (len(positions),) or position_valid.dtype != np.dtype(bool)):
        raise ValueError("Teacher record has malformed position-grid arrays")
    for name in ("x_axis", "y_axis"):
        axis = np.asarray(record[name])
        if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Teacher record grid axes must be finite and strictly ascending")
    if (np.any(indices < 0) or np.any(indices[:, 0] >= len(record["y_axis"])) or
            np.any(indices[:, 1] >= len(record["x_axis"])) or
            not np.array_equal(positions[:, 0], record["x_axis"][indices[:, 1]]) or
            not np.array_equal(positions[:, 1], record["y_axis"][indices[:, 0]])):
        raise ValueError("Teacher record coordinates and grid ordering are misaligned")
    evaluated = np.asarray(record["evaluated"])
    if (evaluated.dtype != np.dtype(bool) or not evaluated.all() or
            not record["metadata"].get("complete")):
        raise IncompleteSearchError("Only completed support can be saved/loaded as a teacher target")
    expected = teacher_probabilities(record["physical_cost"], record["valid"],
                                     temperature=record["metadata"]["settings"]["temperature"],
                                     base_mass=record["base_mass"])
    probabilities = np.asarray(record["teacher_prob"])
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-12) or
            not np.allclose(probabilities, expected, rtol=1e-12, atol=0)):
        raise ValueError("Teacher record probabilities do not match its aligned physical costs/mass")
    for i, count in enumerate(counts):
        if np.any(sources[i, int(count):] != 0):
            raise ValueError("Inactive padded source rows must be zero")
        if record["valid"][i] and not np.isfinite(sources[i, :int(count)]).all():
            raise ValueError("A valid source set contains nonfinite parameters")


def save_teacher_record(record, path):
    """Compressed numeric NPZ with JSON provenance; never requires pickle."""
    _validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(record[name]) for name in RECORD_FIELDS}
    payload["metadata_json"] = np.asarray(json.dumps(record["metadata"], sort_keys=True, allow_nan=False))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


def load_teacher_record(path):
    with np.load(path, allow_pickle=False) as archive:
        record = {name: archive[name].copy() for name in RECORD_FIELDS}
        record["metadata"] = json.loads(str(archive["metadata_json"].item()))
    _validate_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/phase1.json")
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--counts", type=int, nargs="+", help="Candidate K values; never inferred from truth")
    parser.add_argument("--count-prior", type=float, nargs="+", help="Required for multiple K; aligned positive probabilities")
    parser.add_argument("--samples-per-count", type=int, nargs="+", help="One common count, or one per K")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("auto", "numba", "reference"), default="auto")
    parser.add_argument("--max-evaluations", type=int, help="Unique joint-cost evaluation budget")
    parser.add_argument("--max-source-renders", type=int, help="Unique position-visibility render budget")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--diagnostics", action="store_true", help="Check witnesses and write JSON diagnostics")
    parser.add_argument("--plot", action="store_true", help="Also write a PNG; diagnostics extra required")
    parser.add_argument("--low-cost-threshold", type=float)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        settings = config["teacher"]
        if (settings.get("baseline") != "joint_source_sets_response_mae" or
                settings.get("alpha") != 1 or settings.get("beta") != 0 or
                settings.get("boundary_lambda") != 0 or settings.get("adaptive_search") or
                settings.get("quadtree_prior") is not False or
                settings.get("candidate_domain") != "world_outside_observation_and_obstacles" or
                settings.get("joint_proposal_policy") != "stratified_uniform_joint_monte_carlo"):
            raise ValueError("This baseline implements joint Monte Carlo, response MAE, and world-outside-window only")
        required = {"spacing": args.spacing if args.spacing is not None else settings.get("candidate_spacing"),
                    "temperature": args.temperature if args.temperature is not None else settings.get("temperature"),
                    "source_counts": args.counts if args.counts is not None else settings.get("source_counts"),
                    "samples_per_count": (args.samples_per_count if args.samples_per_count is not None
                                          else settings.get("samples_per_count"))}
        if any(value is None for value in required.values()):
            raise ValueError("Set --spacing, --temperature, --counts and --samples-per-count, or fill explicit config values")
        if isinstance(required["samples_per_count"], list) and len(required["samples_per_count"]) == 1:
            required["samples_per_count"] = required["samples_per_count"][0]
        prior = args.count_prior if args.count_prior is not None else settings.get("source_count_prior")
        seed = args.seed if args.seed is not None else config["seed"]
        if args.low_cost_threshold is not None and not (args.diagnostics or args.plot):
            raise ValueError("--low-cost-threshold requires --diagnostics or --plot")
        if args.plot:
            import importlib.util
            if importlib.util.find_spec("matplotlib") is None:
                raise ImportError("Plotting requires uv sync --locked --extra diagnostics")
        root = args.data_root if args.data_root is not None else (
            None if os.environ.get("RIND_MULTI_DATA_ROOT") else PROJECT_ROOT / config["data"]["dataset_root"])
        dataset = Phase1MultiDataset(root)
        record = generate_joint_teacher(dataset, args.scene_id, args.view_id, **required, count_prior=prior,
                                        seed=seed, backend=args.backend, max_evaluations=args.max_evaluations,
                                        max_source_renders=args.max_source_renders, run_id=args.run_id)
        output = args.output or PROJECT_ROOT / "outputs/teachers" / f"{record['metadata']['run_id']}.npz"
        if args.diagnostics or args.plot:
            from rind_phase1_multi.diagnostics import inspect_teacher, plot_teacher
            report = inspect_teacher(dataset, record, low_cost_threshold=args.low_cost_threshold, backend=args.backend)
            output.parent.mkdir(parents=True, exist_ok=True)
            report_path = output.with_suffix(".diagnostics.json")
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if args.plot:
                plot_teacher(dataset, record, report, output.with_suffix(".png"))
            print(f"Diagnostics: {report_path}")
        save_teacher_record(record, output)
        print(f"Ready: {output}\n{len(record['source_counts'])} joint hypotheses; "
              f"{record['metadata']['num_source_renders']} unique visibility renders; "
              f"minimum MAE={record['physical_cost'][record['valid']].min():.6g}. "
              "Completed sampled support; joint coverage is not exhaustive.")
    except (OSError, ValueError, KeyError, ImportError, AssertionError, RuntimeError) as exc:
        parser.exit(1, f"Joint teacher generation failed: {exc}\n")


if __name__ == "__main__":
    main()
