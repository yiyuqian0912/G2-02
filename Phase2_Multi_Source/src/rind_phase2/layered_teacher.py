"""Local minimum + one-hop neighbor, iterative single-source proposal teacher.

This is a bounded heuristic proposal search, not a complete inverse solver.
Truth source parameters/channels and reference witnesses are never consulted.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np

from rind_phase2.data import PROJECT_ROOT, Phase2Dataset
from rind_phase2.physics import JointResponseEvaluator
from rind_phase2.regions import response_regions
from rind_phase2.search import (uniform_source_grid, make_joint_support, evaluate_joint_support,
                               NoCandidatePositionsError)
from rind_phase2.teacher import record_from_search, save_teacher_record


class NoLayeredSolutionError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


@dataclass
class LayerState:
    residual: np.ndarray
    prediction: np.ndarray
    sources: list
    trace: list


def _runs(indices):
    ids = np.sort(np.asarray(indices, dtype=int))
    if not len(ids):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1]
    ends = np.r_[starts[1:], len(ids)]
    return [[int(ids[a]), int(b-a)] for a,b in zip(starts,ends)]


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def generate_layered_teacher(dataset, scene_id, view_id, *, spacing, temperature, seed,
                             max_sources=4, beam_width=16, samples_per_pattern=2,
                             region_tolerance=1e-12, local_mismatch_tolerance=0.0,
                             max_final_mae=0.01, backend="auto", max_source_renders=None,
                             max_evaluations=None, run_id=None):
    """Implement one-hop subtraction literally, then check the full observation.

    The candidate's illumination beyond the selected one-hop areas is NOT
    subtracted silently. It is retained in prediction and checked at completion.
    Equal intensity on disconnected regions never merges their identities.
    """
    for name, value in (("max_sources", max_sources), ("beam_width", beam_width),
                        ("samples_per_pattern", samples_per_pattern)):
        _positive_int(name, value)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(region_tolerance) or region_tolerance < 0:
        raise ValueError("region_tolerance must be finite and nonnegative")
    if not np.isfinite(local_mismatch_tolerance) or not 0 <= local_mismatch_tolerance < .5:
        raise ValueError("local_mismatch_tolerance must be in [0,.5)")
    if not np.isfinite(max_final_mae) or max_final_mae < 0:
        raise ValueError("max_final_mae must be finite and nonnegative")
    max_sources, beam_width, samples_per_pattern, seed = map(int, (max_sources, beam_width, samples_per_pattern, seed))
    started = time.perf_counter()
    evaluator = JointResponseEvaluator(dataset, scene_id, view_id, backend=backend,
                                       max_source_renders=max_source_renders)
    grid = uniform_source_grid(evaluator.domain, evaluator, spacing=spacing)
    ids = np.flatnonzero(grid.valid)
    if not len(ids):
        raise NoCandidatePositionsError("No geometrically valid source-grid positions")
    observed = evaluator.observed_response
    obstacle = np.asarray(evaluator.observation["obstacle"], dtype=bool)
    if np.any(observed < -region_tolerance):
        raise ValueError("Layered positive-source method requires nonnegative response")
    if np.any(np.abs(observed[obstacle]) > region_tolerance):
        raise ValueError("Obstacle pixels must have zero response")
    rng = np.random.default_rng(seed)
    report = dict(method="minimum_region_one_hop", scene_id=int(scene_id), view_id=int(view_id),
                  source_truth_used=False, subtraction_scope="anchor and selected one-hop neighbors only",
                  equal_intensity_disconnected_regions="separate IDs, separate subsequent steps",
                  anchor_ties="first region in deterministic row-major connected-component order",
                  no_positive_response=bool(not np.any(observed[~obstacle] > region_tolerance)),
                  source_grid_candidates=len(ids), levels=[], beam_pruned=0, pattern_candidates_not_drawn=0,
                  rejected_intensity_above_one=0, rejected_no_local_candidates=0,
                  rejected_final_reconstruction=0, incomplete_at_source_limit=0)
    beam = [LayerState(observed.copy(), np.zeros_like(observed), [], [])]
    completed = []
    for depth in range(max_sources):
        children = []
        log = dict(depth=depth+1, parent_states=len(beam), anchor_regions=[], compatible_positions=0,
                   neighbor_patterns=0, children=0)
        for state in beam:
            regions = response_regions(state.residual, obstacle, evaluator.domain.window[:2],
                                        evaluator.obstacle_types, evaluator.obstacle_params,
                                        tolerance=region_tolerance)
            positive = np.flatnonzero(regions.values > region_tolerance)
            if not len(positive):
                # A fully dark observation has no observable source under this method.
                if state.sources:
                    completed.append(state)
                continue
            minimum = float(regions.values[positive].min())
            anchor = int(positive[np.flatnonzero(np.abs(regions.values[positive]-minimum)
                                               <= region_tolerance)[0]])
            intensity = float(regions.values[anchor])
            if intensity > 1.0:
                report["rejected_intensity_above_one"] += 1
                continue
            adjacent = regions.neighbors[anchor]
            local = [anchor, *adjacent]
            patterns = {}
            for index in ids:
                unit = evaluator.visibility_at(grid.positions[index]).ravel()
                if 1-float(unit[regions.pixels[anchor]].mean()) > local_mismatch_tolerance:
                    continue
                selected = []
                mismatch = float((~unit[regions.pixels[anchor]]).sum())
                valid = True
                for neighbor in adjacent:
                    fraction = float(unit[regions.pixels[neighbor]].mean())
                    if fraction <= local_mismatch_tolerance:
                        mismatch += fraction*len(regions.pixels[neighbor])
                    elif (1-fraction <= local_mismatch_tolerance and
                          regions.values[neighbor] + region_tolerance >= intensity):
                        selected.append(neighbor)
                        mismatch += (1-fraction)*len(regions.pixels[neighbor])
                    else:
                        valid = False
                        break
                if valid:
                    error = mismatch/sum(len(regions.pixels[z]) for z in local)
                    patterns.setdefault(tuple(selected), []).append((int(index), error))
            log["compatible_positions"] += sum(map(len, patterns.values()))
            log["neighbor_patterns"] += len(patterns)
            log["anchor_regions"].append(dict(region_id=anchor, intensity=intensity,
                anchor_pixels=len(regions.pixels[anchor]), adjacent_regions=adjacent,
                region_count=len(regions.values), excluded_obstacle_edges=regions.excluded_edges))
            if not patterns:
                report["rejected_no_local_candidates"] += 1
            for selected, candidates in sorted(patterns.items()):
                # Preserve discrete neighbor alternatives before bounded beam selection.
                take = min(samples_per_pattern, len(candidates))
                chosen = rng.choice(len(candidates), size=take, replace=False)
                report["pattern_candidates_not_drawn"] += len(candidates)-take
                removed = np.concatenate([regions.pixels[z] for z in (anchor, *selected)])
                residual = state.residual.copy()
                residual.ravel()[removed] -= intensity
                residual[np.abs(residual) <= region_tolerance] = 0
                if np.any(residual < -region_tolerance):
                    raise AssertionError("Layer subtraction produced a negative residual")
                for choice in chosen:
                    index, error = candidates[int(choice)]
                    xy = grid.positions[index]
                    unit = evaluator.visibility_at(xy)
                    source = [float(xy[0]), float(xy[1]), intensity]
                    trace = dict(step=depth+1, source=source, anchor_region_id=anchor,
                        anchor_intensity=intensity, anchor_runs=_runs(regions.pixels[anchor]),
                        adjacent_region_ids=adjacent, selected_neighbor_ids=list(selected),
                        removed_runs=_runs(removed), local_mismatch=float(error),
                        same_intensity_other_region_ids=[int(z) for z in positive if z != anchor
                            and abs(regions.values[z]-intensity) <= region_tolerance],
                        illumination_outside_removed_pixels=int(unit.sum()-unit.ravel()[removed].sum()),
                        remaining_response_sum=float(residual.sum()))
                    child = LayerState(residual.copy(), state.prediction+intensity*unit,
                                       state.sources+[source], state.trace+[trace])
                    if np.max(residual[~obstacle], initial=0) <= region_tolerance:
                        mae = float(np.mean(np.abs(child.prediction-observed)))
                        if mae <= max_final_mae:
                            completed.append(child)
                        else:
                            report["rejected_final_reconstruction"] += 1
                    else:
                        children.append(child)
        log["children"] = len(children)
        report["levels"].append(log)
        # Global overshoot cannot be repaired by later positive sources; residual
        # magnitude measures unfinished work. Ranking is explicitly heuristic.
        children.sort(key=lambda z: (float(np.maximum(z.prediction-observed, 0).mean()),
                                     float(z.residual.mean())))
        report["beam_pruned"] += max(0, len(children)-beam_width)
        beam = children[:beam_width]
        if not beam:
            break
    report["incomplete_at_source_limit"] = len(beam)
    report["num_source_renders"] = evaluator.num_source_renders
    report["elapsed_search_seconds"] = time.perf_counter()-started
    if not completed:
        raise NoLayeredSolutionError(
            "No completed layered proposal passed full-response validation; see search report. "
            "No fallback to joint random sampling was performed.", report)
    # A proposal target over retained configurations, NOT a source-count posterior.
    support = make_joint_support([np.array(z.sources) for z in completed],
        base_mass=np.ones(len(completed))/len(completed), metadata={
            "search_mode": "layered_minimum_region_one_hop", "grid": grid.metadata(), "seed": int(seed),
            "max_sources": max_sources, "beam_width": beam_width, "samples_per_pattern": samples_per_pattern,
            "region_tolerance": region_tolerance, "local_mismatch_tolerance": local_mismatch_tolerance,
            "max_final_mae": max_final_mae,
            "sampling_measure": "equal empirical mass per retained completed path; aggregate exact duplicates",
            "source_count_prior": None, "strength_rule": "minimum positive connected-region response, <=1",
            "not_a_prior_corrected_posterior": True, "coverage": "bounded sampled paths; not exhaustive"})
    result = evaluate_joint_support(support, evaluator, max_evaluations=max_evaluations)
    # Pre-sampling unit renders are part of this search's true physical cost.
    result.num_source_renders = evaluator.num_source_renders
    result.elapsed_seconds = time.perf_counter()-started
    record = record_from_search(dataset, scene_id, view_id, evaluator, grid, result,
                               temperature=temperature, max_source_renders=max_source_renders,
                               max_evaluations=max_evaluations, run_id=run_id)
    if not np.all(record['physical_cost'] <= max_final_mae + 1e-12):
        raise AssertionError("Full evaluator disagrees with layered acceptance")
    record["metadata"]["completion_scope"] = "all retained finite layered proposals evaluated; coverage not exhaustive"
    report["completed_paths"] = len(completed)
    report["unique_joint_hypotheses"] = len(support)
    report["approximate_local_matching"] = local_mismatch_tolerance > 0
    report["max_retained_mae"] = float(record['physical_cost'].max())
    # make_joint_support preserves first-occurrence order; attach one aligned trace.
    traces = {}
    from rind_phase2.search import source_set_key
    for z in completed:
        traces.setdefault(source_set_key(z.sources), z.trace)
    record["metadata"]["layered_search"] = report
    record["metadata"]["layer_traces"] = [traces[source_set_key(support.source_set(i))] for i in range(len(support))]
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-id', type=int, required=True)
    parser.add_argument('--view-id', type=int, required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/teacher_layered.json')
    for name, typ in [('spacing',float),('temperature',float),('seed',int),('max-sources',int),
                      ('beam-width',int),('samples-per-pattern',int),('region-tolerance',float),
                      ('local-mismatch-tolerance',float),('max-final-mae',float),
                      ('max-source-renders',int),('max-evaluations',int)]:
        parser.add_argument('--'+name,type=typ)
    parser.add_argument('--backend', choices=['auto','numba','reference'],default='auto')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    output=args.output or PROJECT_ROOT/'outputs/teachers'/f'layered-scene{args.scene_id}-view{args.view_id}.npz'
    try:
        settings=json.loads(args.config.read_text())
        for name in list(settings):
            value=getattr(args,name,None)
            if value is not None:settings[name]=value
        dataset=Phase2Dataset(args.data_root)
        record=generate_layered_teacher(dataset,args.scene_id,args.view_id,backend=args.backend,**settings)
        if args.output is None:
            output = output.with_name(f"layered-{record['metadata']['run_id']}.npz")
        save_teacher_record(record,output)
        report_path=output.with_suffix('.layers.json')
        report_path.write_text(json.dumps(dict(search=record['metadata']['layered_search'],
            traces=record['metadata']['layer_traces']),indent=2)+'\n')
        print(f'Ready: {output}\nLayer traces: {report_path}\n'
              f'{len(record["source_counts"])} retained configurations; '
              f'MAE range [{record["physical_cost"].min():.6g}, {record["physical_cost"].max():.6g}]. '
              'Layered proposal compatibility target; not an exhaustive posterior.')
    except NoLayeredSolutionError as exc:
        failure=output.with_suffix('.failed.json');failure.parent.mkdir(parents=True,exist_ok=True)
        failure.write_text(json.dumps(exc.report,indent=2)+'\n')
        parser.exit(1,f'{exc}\nFailure report: {failure}. No new teacher record written; any existing record is unchanged.\n')
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'Layered teacher failed: {exc}\n')

if __name__=='__main__':
    main()
