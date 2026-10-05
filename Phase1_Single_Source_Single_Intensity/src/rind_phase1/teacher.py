"""Completed uniform physical costs -> soft teacher targets and portable records."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rind_phase1.data import PROJECT_ROOT, Phase1Dataset
from rind_phase1.physics import PhysicalEvaluator
from rind_phase1.search import CandidateDomain, uniform_search


class NoValidCandidatesError(ValueError):
    pass


class IncompleteSearchError(ValueError):
    pass


def _validate_temperature(temperature):
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def teacher_probabilities(physical_cost, valid, *, temperature):
    """Stable Gibbs probabilities in the original order; invalid entries are zero."""
    costs, mask = np.asarray(physical_cost, dtype=np.float64), np.asarray(valid)
    if costs.ndim != 1 or mask.shape != costs.shape or mask.dtype != np.dtype(bool):
        raise ValueError("Costs and boolean validity must be aligned [N] arrays")
    _validate_temperature(temperature)
    if not mask.any():
        raise NoValidCandidatesError("No valid candidate on this uniform support; choose an explicit finer spacing")
    selected = costs[mask]
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Valid physical costs must be finite and nonnegative")
    # Subtract before division, so even tiny tau leaves at least one weight = 1.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(selected - selected.min()) / float(temperature))
    probabilities = np.zeros_like(costs)
    probabilities[mask] = weights / weights.sum(dtype=np.float64)
    return probabilities


def target_from_search(result, *, temperature):
    if not result.complete or not result.evaluated.all():
        raise IncompleteSearchError("Search is incomplete; no final teacher distribution was constructed")
    if result.search_mode != "uniform":
        raise ValueError("Only completed uniform support is currently implemented")
    if len(result.grid.candidate_xy) != len(result.physical_cost):
        raise ValueError("Candidate coordinates and physical costs are misaligned")
    return teacher_probabilities(result.physical_cost, result.valid, temperature=temperature)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generate_uniform_teacher(dataset, scene_id, view_id, *, spacing, temperature,
                             quadtree_prior=False, backend="auto", max_evaluations=None,
                             run_id=None):
    """Build a target without consulting reference positions or source truth labels."""
    _validate_temperature(temperature)
    evaluator = PhysicalEvaluator(dataset, scene_id, view_id, backend=backend)
    domain = CandidateDomain(tuple(evaluator.observation["window"]), evaluator.world_size, quadtree_prior)
    result = uniform_search(domain, evaluator, spacing=spacing, max_evaluations=max_evaluations)
    probabilities = target_from_search(result, temperature=temperature)
    provenance = {"manifest": dataset.manifest, "manifest_sha256": _digest(dataset.manifest)}
    installation = dataset.root / "installation.json"
    if installation.is_file():
        provenance["archive_sha256"] = json.loads(installation.read_text())["archive_sha256"]
    code_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("data.py", "physics.py", "search.py", "teacher.py")}
    snapshot = PROJECT_ROOT / "vendor/rind-dataset/SOURCE_SNAPSHOT.json"
    if snapshot.is_file():
        code_hashes["reader_snapshot"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    settings = {"dataset": provenance, "support": result.grid.metadata(),
                "physics": evaluator.cost_config, "temperature": float(temperature),
                "search_mode": result.search_mode, "max_evaluations": max_evaluations,
                "code_sha256": code_hashes}
    config_id = _digest(settings)
    run_id = run_id or f"scene{scene_id}-view{view_id}-{config_id[:12]}"
    metadata = {"record_schema_version": 1, "scene_id": int(scene_id), "view_id": int(view_id),
                "config_id": config_id, "run_id": run_id, "complete": True,
                "dataset_root": str(dataset.root), "settings": settings,
                "num_physics_evaluations": result.num_physics_evaluations,
                "num_cache_hits": result.num_cache_hits, "elapsed_seconds": result.elapsed_seconds,
                "reference_solutions_used_as_targets": False}
    return {"scene_id": int(scene_id), "view_id": int(view_id),
            "candidate_xy": result.grid.candidate_xy.copy(), "valid": result.valid.copy(),
            "evaluated": result.evaluated.copy(), "physical_cost": result.physical_cost.copy(),
            "teacher_prob": probabilities, "temperature": float(temperature),
            "config_id": config_id, "run_id": run_id, "metadata": metadata,
            "window": np.asarray(domain.window, dtype=np.int64),
            "grid_index": result.grid.grid_index.copy(), "x_axis": result.grid.x_axis.copy(),
            "y_axis": result.grid.y_axis.copy(),
            "search_level": np.zeros(len(probabilities), dtype=np.uint8)}


RECORD_FIELDS = ("scene_id", "view_id", "candidate_xy", "valid", "evaluated", "physical_cost",
                 "teacher_prob", "temperature", "config_id", "run_id", "window", "grid_index",
                 "x_axis", "y_axis", "search_level")


def save_teacher_record(record, path):
    """One compressed NPZ, numeric/string arrays only; loadable with allow_pickle=False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(record[key]) for key in RECORD_FIELDS}
    payload["metadata_json"] = np.asarray(json.dumps(record["metadata"], sort_keys=True, allow_nan=False))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


def load_teacher_record(path):
    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key].copy() for key in RECORD_FIELDS}
        record["metadata"] = json.loads(str(archive["metadata_json"].item()))
    for key in ("scene_id", "view_id", "temperature", "config_id", "run_id"):
        record[key] = record[key].item()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/phase1.json")
    parser.add_argument("--spacing", type=float, help="Explicit world-coordinate spacing; overrides config")
    parser.add_argument("--temperature", type=float, help="Explicit tau; overrides config")
    parser.add_argument("--prior", choices=("parent", "world"),
                        help="World outside the window (default config), or optional parent-prior ablation")
    parser.add_argument("--backend", choices=("auto", "numba", "reference"), default="auto")
    parser.add_argument("--max-evaluations", type=int, help="Optional render budget; incomplete runs fail")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, help="Teacher NPZ; defaults to outputs/teachers/<run_id>.npz")
    parser.add_argument("--plot", action="store_true", help="Write a diagnostic PNG (diagnostics extra required)")
    parser.add_argument("--low-cost-threshold", type=float, help="Optional explicit near-zero threshold")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        settings = config["teacher"]
        if (settings.get("alpha") != 1 or settings.get("beta") != 0
                or settings.get("boundary_lambda") != 0 or settings.get("adaptive_search")):
            raise ValueError("This baseline implements alpha=1, beta=0, lambda=0, uniform search only")
        spacing = args.spacing if args.spacing is not None else settings.get("candidate_spacing")
        temperature = args.temperature if args.temperature is not None else settings.get("temperature")
        if spacing is None or temperature is None:
            raise ValueError("Spacing and temperature are unresolved; specify --spacing and --temperature or fill config")
        if args.low_cost_threshold is not None and (not np.isfinite(args.low_cost_threshold)
                                                   or args.low_cost_threshold < 0):
            raise ValueError("low_cost_threshold must be finite and nonnegative")
        prior = settings["quadtree_prior"] if args.prior is None else args.prior == "parent"
        root = args.data_root if args.data_root is not None else (
            None if os.environ.get("RIND_DATA_ROOT") else PROJECT_ROOT / config["data"]["dataset_root"])
        if args.plot:
            import importlib.util
            if importlib.util.find_spec("matplotlib") is None:
                raise ImportError("Plotting requires uv sync --locked --extra diagnostics")
        dataset = Phase1Dataset(root)
        record = generate_uniform_teacher(dataset, args.scene_id, args.view_id, spacing=spacing,
                                          temperature=temperature, quadtree_prior=prior, backend=args.backend,
                                          max_evaluations=args.max_evaluations, run_id=args.run_id)
        from rind_phase1.diagnostics import inspect_teacher, plot_teacher
        diagnostics = inspect_teacher(dataset, record, low_cost_threshold=args.low_cost_threshold,
                                      backend=args.backend)
        output = args.output or PROJECT_ROOT / "outputs/teachers" / f"{record['run_id']}.npz"
        save_teacher_record(record, output)
        report_path = output.with_suffix(".diagnostics.json")
        report_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
        if args.plot:
            plot_teacher(dataset, record, diagnostics, output.with_suffix(".png"))
        print(f"Ready: {output}\n{len(record['candidate_xy'])} uniform candidates; "
              f"{int(record['valid'].sum())} valid; {diagnostics['zero_cost_candidates']} exact grid matches; "
              f"entropy={diagnostics['teacher_entropy_nats']:.6g} nats.\nDiagnostics: {report_path}")
    except (OSError, ValueError, KeyError, ImportError, AssertionError) as exc:
        parser.exit(1, f"Teacher generation failed: {exc}\n")


if __name__ == "__main__":
    main()
