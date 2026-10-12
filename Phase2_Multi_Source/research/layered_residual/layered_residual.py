"""Conservative residual-layer recovery with Phase-I single-source teachers."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if "Phase2_Multi_Source" in str(HERE) else Path.cwd().parent
for item in (REPO/"Phase1_Single_Source_Single_Intensity/src",
             REPO/"Phase2_Multi_Source/src", REPO/"Phase2_Multi_Source/vendor/rind-dataset"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from rind_phase1.physics import boundary_weight_map, edge_cost, response_cost
from rind_phase1.search import CandidateDomain, uniform_grid
from rind_phase1.teacher import teacher_probabilities
from rind_phase2.data import Phase2Dataset
from rind_phase2.physics import JointResponseEvaluator
from rind_phase2.regions import response_regions
from rind_phase2.search import (uniform_source_grid, sample_joint_support,
                                evaluate_joint_support)


def _close(a, b, tol):
    return abs(float(a)-float(b)) <= tol


def minimum_positive_regions(values, *, tolerance=1e-12):
    """Select the lowest strictly positive residual regions.

    Zero-response shadow/background regions are never contribution layers.
    Values at or below ``tolerance`` are excluded from both the minimum and
    the seed list.
    """
    values=np.asarray(values,dtype=np.float64)
    ids=np.flatnonzero(values>tolerance)
    if not len(ids):
        return None,[]
    minimum=float(values[ids].min())
    seeds=[int(index) for index in ids if _close(values[index],minimum,tolerance)]
    return minimum,seeds


def combine_layer_probabilities(probabilities):
    """Return an equal-weight mixture of aligned, normalized layer masses.

    Every recovered layer contributes exactly ``1 / number_of_layers`` of the
    final categorical target.  The result means: choose one recovered effective
    contribution uniformly, then sample its position from that layer's Phase-I
    distribution.  It is not a multi-source joint posterior or an occupancy
    probability.
    """
    arrays=[np.asarray(p,dtype=np.float64) for p in probabilities]
    if not arrays or any(p.ndim!=1 or p.shape!=arrays[0].shape for p in arrays):
        raise ValueError('aligned nonempty one-dimensional layer probabilities required')
    stack=np.stack(arrays)
    if not np.isfinite(stack).all() or np.any(stack<0):
        raise ValueError('layer probabilities must be finite and nonnegative')
    totals=stack.sum(axis=1,dtype=np.float64)
    if not np.allclose(totals,1.,rtol=0,atol=1e-10):
        raise ValueError('every layer probability must sum to one')
    combined=np.mean(stack,axis=0,dtype=np.float64)
    total=combined.sum(dtype=np.float64)
    if not np.isfinite(total) or total<=0:
        raise ValueError('equal-layer mixture has no probability mass')
    return combined/total


def screen_candidates_by_observed_shadow(evaluator, grid, observed, obstacle, *, tolerance=1e-12):
    """Reject candidates that illuminate any observed free zero-response pixel.

    With positive source strengths and additive binary visibility, a zero total
    response at a free pixel proves that every real source is invisible there.
    This screen is therefore exact for the current noiseless Phase-II model.
    Obstacle pixels are excluded because their stored zero is not a visibility
    observation.  Rendering is performed once per geometrically valid uniform
    candidate and is reused by all later layer scores through the evaluator's
    visibility cache.
    """
    observed=np.asarray(observed,dtype=np.float64)
    obstacle=np.asarray(obstacle,dtype=bool)
    if observed.ndim!=2 or obstacle.shape!=observed.shape:
        raise ValueError('aligned 2D observed response and obstacle mask required')
    if not np.isfinite(observed).all() or tolerance<0 or not np.isfinite(tolerance):
        raise ValueError('finite response and nonnegative finite tolerance required')
    shadow=(~obstacle)&(observed<=tolerance)
    geometry_valid=np.asarray(evaluator.positions_valid(grid.candidate_xy),dtype=bool)
    if geometry_valid.shape!=(len(grid.candidate_xy),):
        raise ValueError('candidate geometry validity is misaligned')
    retained=geometry_valid.copy()
    for index in np.flatnonzero(geometry_valid):
        visibility=np.asarray(evaluator.visibility_at(grid.candidate_xy[index]))
        if visibility.shape!=observed.shape:
            raise ValueError('candidate visibility shape does not match observation')
        if np.any(visibility[shadow]>tolerance):
            retained[index]=False
    geometry_count=int(np.count_nonzero(geometry_valid));retained_count=int(np.count_nonzero(retained))
    return retained,{'enabled':True,'rule':'reject candidate if its unit visibility is positive at any free observed-zero pixel',
                     'shadow_pixels':int(np.count_nonzero(shadow)),
                     'geometry_valid_candidates':geometry_count,
                     'retained_candidates':retained_count,
                     'rejected_candidates':geometry_count-retained_count,
                     'retained_fraction':(retained_count/geometry_count if geometry_count else 0.)}


def trace_contribution(values, neighbors, seed, *, tolerance=1e-12):
    """Track only whether the current minimum contribution is still present.

    On a decrease, the target exits when the drop equals its strength, or when
    the destination response is too small to contain it. Otherwise the decrease
    is attributed to other contributions and target propagation continues.
    Target-exit branches stop; other queued branches continue.
    """
    values = np.asarray(values, dtype=np.float64)
    a = float(values[seed])
    present = np.zeros(len(values), dtype=bool)
    absent = np.zeros(len(values), dtype=bool)
    present[seed] = True
    queue = deque([int(seed)])
    transitions = []
    while queue:
        region = queue.popleft()
        for nxt in neighbors[region]:
            delta = float(values[nxt]-values[region])
            if delta >= -tolerance:
                target_present, reason = True, ('response_increase' if delta > tolerance
                                                 else 'equal_response')
            else:
                drop = -delta
                if _close(drop, a, tolerance):
                    target_present, reason = False, 'drop_equals_current_layer'
                elif values[nxt] < a-tolerance:
                    target_present, reason = False, 'destination_below_current_layer'
                else:
                    target_present, reason = True, 'other_contribution_exits'
            transitions.append({"from":int(region), "to":int(nxt), "delta":delta,
                                "drop":max(0.,-delta), "target_present":target_present,
                                "reason":reason})
            if target_present:
                if not present[nxt]:
                    present[nxt] = True
                    queue.append(int(nxt))
            else:
                absent[nxt] = True
    positive, negative, conflict, unknown = [], [], [], []
    for region, (seen, stopped) in enumerate(zip(present, absent)):
        if seen and not stopped:
            positive.append(region)
        elif not seen and stopped:
            negative.append(region)
        elif seen and stopped:
            conflict.append(region)
        else:
            unknown.append(region)
    return {"strength":a, "seed":int(seed), "positive":positive, "negative":negative,
            "unknown":unknown, "conflict":conflict, "present_reached":np.flatnonzero(present).tolist(),
            "exit_reached":np.flatnonzero(absent).tolist(), "transitions":transitions,
            "state_overflow":False}


def _phase1_layer_teacher(evaluator, grid, target, *, temperature, alpha, beta,
                          boundary_lambda, boundary_sigma, candidate_valid=None):
    """Apply the Phase-I teacher directly to one inferred layer response.

    Phase I does not require an exact-render candidate. Every candidate in the
    supplied shared support (geometry plus optional observed-shadow screening)
    is scored against ``target`` and retained according to its physical cost.
    """
    weights = boundary_weight_map(target, boundary_lambda=boundary_lambda,
                                  boundary_sigma=boundary_sigma)
    if candidate_valid is None:
        valid=np.asarray(evaluator.positions_valid(grid.candidate_xy),dtype=bool)
    else:
        valid=np.asarray(candidate_valid,dtype=bool)
        if valid.shape!=(len(grid.candidate_xy),):
            raise ValueError('candidate_valid must align with the uniform grid')
    costs = np.full(len(valid), np.inf)
    for index in np.flatnonzero(valid):
        unit = evaluator.visibility_at(grid.candidate_xy[index])
        costs[index] = alpha*response_cost(target, unit, weights)
        if beta > 0:
            costs[index] += beta*edge_cost(target, unit)
    probability = teacher_probabilities(costs, valid, temperature=temperature)
    return valid, costs, probability


def _phase1_layers_from_one_uniform_pass(
        evaluator, grid, targets, observed, obstacle, *, temperature, alpha, beta,
        boundary_lambda, boundary_sigma, shadow_screening=True, tolerance=1e-12):
    """Render every uniform candidate once and score it against every layer.

    Candidate visibility depends only on scene geometry and source position, so
    it must not be regenerated for each recovered residual layer.  This routine
    makes candidates the outer loop: one unit-visibility render is obtained for
    a position, optionally checked against the original observed shadow, and
    then compared with every layer target before moving to the next position.
    Layer-specific softmax normalization is applied after this single pass.

    The unavoidable ``number of candidates x number of layers`` work here is
    only array comparison.  The expensive physical visibility render count is
    ``number of geometrically valid candidates``, independent of layer count.
    """
    targets=[np.asarray(target,dtype=np.float64) for target in targets]
    if not targets:
        raise ValueError('at least one layer target is required')
    observed=np.asarray(observed,dtype=np.float64)
    obstacle=np.asarray(obstacle,dtype=bool)
    if any(target.shape!=observed.shape for target in targets) or obstacle.shape!=observed.shape:
        raise ValueError('layer targets, observed response, and obstacle mask must align')
    weights=[boundary_weight_map(target,boundary_lambda=boundary_lambda,
                                 boundary_sigma=boundary_sigma) for target in targets]
    geometry_valid=np.asarray(evaluator.positions_valid(grid.candidate_xy),dtype=bool)
    if geometry_valid.shape!=(len(grid.candidate_xy),):
        raise ValueError('candidate geometry validity is misaligned')
    retained=geometry_valid.copy()
    costs=np.full((len(targets),len(geometry_valid)),np.inf,dtype=np.float64)
    shadow=(~obstacle)&(observed<=tolerance)
    for index in np.flatnonzero(geometry_valid):
        unit=np.asarray(evaluator.visibility_at(grid.candidate_xy[index]))
        if unit.shape!=observed.shape:
            raise ValueError('candidate visibility shape does not match observation')
        if shadow_screening and np.any(unit[shadow]>tolerance):
            retained[index]=False
            continue
        for layer_index,(target,weight) in enumerate(zip(targets,weights)):
            value=alpha*response_cost(target,unit,weight)
            if beta>0:
                value+=beta*edge_cost(target,unit)
            costs[layer_index,index]=value
    geometry_count=int(np.count_nonzero(geometry_valid))
    retained_count=int(np.count_nonzero(retained))
    if not retained_count:
        return retained,costs,[],{
            'performed':True,'enabled':bool(shadow_screening),
            'rule':('reject candidate if its unit visibility is positive at any free observed-zero pixel'
                    if shadow_screening else 'disabled'),
            'shadow_pixels':int(np.count_nonzero(shadow)),
            'geometry_valid_candidates':geometry_count,'retained_candidates':0,
            'rejected_candidates':geometry_count,'retained_fraction':0.}
    probabilities=[teacher_probabilities(row,retained,temperature=temperature) for row in costs]
    rejected=geometry_count-retained_count
    screening={'performed':True,'enabled':bool(shadow_screening),
               'rule':('reject candidate if its unit visibility is positive at any free observed-zero pixel'
                       if shadow_screening else 'disabled'),
               'shadow_pixels':int(np.count_nonzero(shadow)),
               'geometry_valid_candidates':geometry_count,
               'retained_candidates':retained_count,
               'rejected_candidates':rejected,
               'retained_fraction':retained_count/geometry_count if geometry_count else 0.}
    return retained,costs,probabilities,screening


def _constraints(regions, trace, residual, obstacle, tolerance):
    positive = np.zeros(residual.shape, bool)
    negative = np.zeros(residual.shape, bool)
    for rid in trace['positive']:
        positive.ravel()[regions.pixels[rid]] = True
    for rid in trace['negative']:
        negative.ravel()[regions.pixels[rid]] = True
    negative |= (~obstacle) & (residual < trace['strength']-tolerance)
    unknown = (~obstacle) & ~positive & ~negative
    return positive, negative, unknown


def recover(dataset, scene_id, view_id, *, spacing=16, temperature=.01,
            tolerance=1e-12, alpha=1., beta=0., boundary_lambda=0., boundary_sigma=1.,
            max_layers=32, backend='auto', joint_samples=0, seed=20261012,
            shadow_screening=True):
    started = time.perf_counter()
    evaluator = JointResponseEvaluator(dataset, scene_id, view_id, backend=backend)
    domain = CandidateDomain(tuple(evaluator.observation['window']), evaluator.world_size, False)
    grid = uniform_grid(domain, spacing)
    obstacle = np.asarray(evaluator.observation['obstacle'], bool)
    residual = evaluator.observed_response.copy()
    original = residual.copy()
    # Decompose the response first. Candidate visibility is independent of the
    # residual layers, so the uniform full-field pass is deferred until all
    # layer masks are known and can be scored together.
    screening={'performed':False,'enabled':bool(shadow_screening),
               'rule':'not run because layer decomposition did not complete',
               'shadow_pixels':int(np.count_nonzero((~obstacle)&(original<=tolerance))),
               'geometry_valid_candidates':0,'retained_candidates':0,
               'rejected_candidates':0,'retained_fraction':0.}
    reconstructed = np.zeros_like(residual)
    layers, status, reason = [], 'running', None
    for step in range(1, max_layers+1):
        if np.max(residual[~obstacle], initial=0) <= tolerance:
            status = 'complete'; break
        regions = response_regions(residual, obstacle, domain.window[:2], evaluator.obstacle_types,
                                   evaluator.obstacle_params, tolerance=tolerance)
        minimum,seeds=minimum_positive_regions(regions.values,tolerance=tolerance)
        if minimum is None:
            status='complete';break
        seed_region = min(seeds, key=lambda x:int(regions.pixels[x][0]))
        trace = trace_contribution(regions.values, regions.neighbors, seed_region,
                                   tolerance=tolerance)
        positive, negative, unknown = _constraints(regions, trace, residual, obstacle, tolerance)
        entry = {"step":step, "intensity":minimum, "seed_region":seed_region,
                 "equal_minimum_regions":seeds, "trace":trace,
                 "residual_before":residual.copy(),
                 "region_labels":regions.labels.copy(), "positive":positive,
                 "negative":negative, "unknown":unknown}
        if trace['state_overflow']:
            status, reason = 'incomplete', 'symbolic_state_limit_exceeded'
            layers.append(entry); break
        # The BFS-positive support is this round's complete layer response.
        # Negative, conflicting, and unvisited regions are not assigned to
        # this effective contribution.  They remain in the residual and are
        # handled in later rounds.  This is the same deterministic separation
        # rule used for disconnected equal-valued regions.
        layer_response = positive.copy()
        if not np.any(layer_response):
            status, reason = 'incomplete', 'empty_layer_response'
            layers.append(entry); break
        updated = residual-minimum*layer_response
        if np.any(updated < -tolerance):
            status, reason = 'incomplete', 'layer_response_would_create_negative_residual'
            entry['full_mask'] = layer_response; layers.append(entry); break
        updated[np.abs(updated) <= tolerance] = 0
        entry.update(full_mask=layer_response, layer_response=layer_response,
                     effective_above_physical_source_limit=minimum > 1+tolerance,
                     residual_after=updated.copy())
        layers.append(entry)
        residual = updated
        reconstructed += minimum*layer_response
    else:
        # A residual cleared by the final allowed layer is complete.  The old
        # implementation labelled this case max_layers_reached because the
        # termination condition was only checked at the start of an iteration.
        if np.max(residual[~obstacle], initial=0) <= tolerance:
            status, reason = 'complete', None
        else:
            status, reason = 'incomplete', 'max_layers_reached'
    if status == 'running':
        status, reason = 'incomplete', 'search_stopped'
    decomposed=[x for x in layers if 'layer_response' in x]
    # One full-field candidate pass assigns costs to every successfully
    # decomposed layer. Incomplete decompositions are rejected before any
    # expensive candidate rendering is performed.
    if status=='complete' and decomposed:
        valid,cost_matrix,probabilities,screening=_phase1_layers_from_one_uniform_pass(
            evaluator,grid,[x['layer_response'].astype(np.float64) for x in decomposed],
            original,obstacle,temperature=temperature,alpha=alpha,beta=beta,
            boundary_lambda=boundary_lambda,boundary_sigma=boundary_sigma,
            shadow_screening=shadow_screening,tolerance=tolerance)
        if not probabilities:
            status='incomplete'
            reason=('shadow_screen_removed_all_candidates' if shadow_screening
                    else 'no_valid_uniform_candidates')
        else:
            for entry,costs,probability in zip(decomposed,cost_matrix,probabilities):
                entry.update(candidate_xy=grid.candidate_xy.copy(),
                             grid_index=grid.grid_index.copy(),x_axis=grid.x_axis.copy(),
                             y_axis=grid.y_axis.copy(),valid=valid.copy(),physical_cost=costs,
                             probability=probability,probability_sum=float(probability.sum()),
                             minimum_physical_cost=float(costs[valid].min()),
                             exact_grid_matches=int(np.count_nonzero(
                                 valid & np.isclose(costs,0.,atol=1e-15))))
    combined = None
    successful = [x for x in layers if 'probability' in x]
    if successful:
        combined = combine_layer_probabilities([x['probability'] for x in successful])
    report = {"method":"decompose_all_residual_layers_then_one_uniform_visibility_pass",
              "scene_id":int(scene_id), "view_id":int(view_id), "status":status,
              "failure_reason":reason, "layers_attempted":len(layers),
              "layers_recovered":len(successful), "partial_combined_distribution":status!='complete',
              "final_mae":float(np.mean(np.abs(reconstructed-original))),
              "residual_max":float(np.max(residual)), "spacing":float(spacing),
              "combined_distribution_rule":"equal-weight mixture across recovered layer probabilities",
              "temperature":float(temperature), "phase1_cost":{"alpha":alpha,"beta":beta,
              "boundary_lambda":boundary_lambda,"boundary_sigma":boundary_sigma},
              "layer_selection":"minimum residual strictly above tolerance; zero-response regions excluded",
              "source_truth_used":False, "elapsed_seconds":time.perf_counter()-started,
              "shadow_candidate_screening":screening,
              "uniform_candidate_sampling_passes":int(bool(screening['performed'])),
              "uniform_candidate_positions":int(screening['geometry_valid_candidates']),
              "layer_probability_assignments":len(successful),
              "layer_candidate_score_comparisons":int(
                  screening['retained_candidates']*len(successful)),
              # Backward-compatible alias. These are cheap layer comparisons,
              # not repeated physical renders or repeated candidate sampling.
              "candidate_evaluations":int(screening['retained_candidates']*len(successful)),
              "candidate_score_evaluations_without_shadow_screen":int(
                  screening['geometry_valid_candidates']*len(successful)),
              "candidate_score_evaluations_saved_by_shadow_screen":int(
                  screening['rejected_candidates']*len(successful)),
              "unique_visibility_renders":evaluator.num_source_renders,
              "layer_membership_conflicts":int(sum(len(x['trace']['conflict']) for x in layers)),
              "proof_status":{"subtraction_nonnegative":"checked exactly within tolerance",
              "layer_response":"unambiguous BFS-positive support; disconnected, conflicting, and unvisited regions are processed later",
              "phase1_teacher":"one shared uniform visibility pass; every retained candidate is compared with every recovered layer, then each layer is normalized separately",
              "BFS_transition_interpretation":"conditional heuristic; ambiguity retained",
              "decomposition_correctness":"not proved"}}
    report['combined_probability_sum'] = None if combined is None else float(combined.sum())
    report['combined_layer_weights'] = ([] if not successful else
                                        [1./len(successful)]*len(successful))
    # Truth is read only after inference has finished and only for diagnostics.
    truth_masks = np.asarray(dataset.get_channels(scene_id, view_id)) > 0
    truth_diag = []
    for layer in successful:
        mask = layer['full_mask']
        ious = [float(np.logical_and(mask, truth).sum()/max(1, np.logical_or(mask, truth).sum()))
                for truth in truth_masks]
        truth_diag.append({"best_single_generating_source_iou":max(ious, default=0.),
                           "best_generating_source_index":int(np.argmax(ious)) if ious else None})
    report['truth_diagnostics_not_used_for_inference'] = {
        "generating_source_count":int(len(truth_masks)), "layers":truth_diag}
    if joint_samples and successful:
        baseline_evaluator = JointResponseEvaluator(dataset, scene_id, view_id, backend=backend)
        baseline_grid = uniform_source_grid(baseline_evaluator.domain, baseline_evaluator,
                                            spacing=spacing)
        phase1_pool = grid.candidate_xy[evaluator.positions_valid(grid.candidate_xy)]
        phase2_pool = baseline_grid.positions[baseline_grid.valid]
        if not np.array_equal(phase1_pool, phase2_pool):
            raise AssertionError("Phase-I and Phase-II valid candidate pools are not aligned")
        count = len(successful)
        baseline_support = sample_joint_support(
            baseline_grid, source_counts=[count], samples_per_count=[int(joint_samples)], seed=seed)
        baseline = evaluate_joint_support(baseline_support, baseline_evaluator)
        report['joint_sampling_comparison'] = {
            "same_candidate_pool":True, "source_count":count,
            "samples_requested":int(joint_samples), "unique_hypotheses":len(baseline_support),
            "joint_evaluations":baseline.num_joint_evaluations,
            "unique_visibility_renders":baseline.num_source_renders,
            "elapsed_seconds":baseline.elapsed_seconds,
            "minimum_mae":float(baseline.physical_cost[baseline.valid].min()),
            "layered_final_mae":report['final_mae'],
            "comparison_limit":"existing joint sampler uses iid strengths in [0,1); it is Monte Carlo, not exhaustive",
            "effective_layer_above_one":bool(any(x['intensity'] > 1+tolerance for x in successful))}
    return report, layers, combined, grid, original, obstacle, residual


def save_run(output, report, layers, combined, grid, original, obstacle, residual, *, plots=True):
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    summary=dict(report); summary['layers']=[]
    for layer in layers:
        numeric={}; detail={}
        for key,value in layer.items():
            if isinstance(value,np.ndarray): numeric[key]=value
            elif key != 'trace': detail[key]=value
        numeric['trace_json']=np.asarray(json.dumps(layer['trace']))
        np.savez_compressed(output/f"layer-{layer['step']:02d}.npz",**numeric)
        summary['layers'].append(detail|{"trace":layer['trace']})
    successful=[x for x in layers if 'probability' in x]
    valid=np.array([],dtype=bool) if not successful else successful[0]['valid']
    layer_weights=(np.array([],dtype=np.float64) if not successful else
                   np.full(len(successful),1./len(successful),dtype=np.float64))
    np.savez_compressed(output/'combined.npz', probability=np.array([]) if combined is None else combined,
                        candidate_xy=grid.candidate_xy,grid_index=grid.grid_index,x_axis=grid.x_axis,
                        y_axis=grid.y_axis,valid=valid,layer_weights=layer_weights,
                        original=original,obstacle=obstacle,residual=residual)
    summary['plots_saved']=bool(plots)
    (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    if plots:
        plot_run(output)


def plot_run(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output=Path(output); report=json.loads((output/'report.json').read_text())
    with np.load(output/'combined.npz') as z: combined={k:z[k] for k in z.files}
    layers=[]
    expected = [output/f'layer-{i:02d}.npz' for i in range(1,report['layers_attempted']+1)]
    for path in expected:
        if not path.exists():
            continue
        with np.load(path) as z: layers.append({k:z[k] for k in z.files})
    maps=[('Combined equal-layer mixture',combined['probability'])]+[
        (f"Layer {i+1}, a={report['layers'][i]['intensity']:.6g}",x.get('probability',np.array([])))
        for i,x in enumerate(layers)]
    maps=[x for x in maps if x[1].size]
    if not maps: return
    fig,axes=plt.subplots(1,len(maps),figsize=(5.2*len(maps),5),squeeze=False)
    window=report.get('window'); world=int(report.get('world_size',1024))
    for ax,(title,p) in zip(axes[0],maps):
        arr=np.full((len(combined['y_axis']),len(combined['x_axis'])),np.nan)
        arr.flat[combined['grid_index']]=p
        cmap=plt.get_cmap('magma').copy(); cmap.set_bad('#bfc3c9')
        im=ax.imshow(arr,origin='upper',extent=(0,world,world,0),cmap=cmap,vmin=0,interpolation='nearest')
        if window:
            ax.add_patch(Rectangle((window[0],window[1]),window[2],window[2],fill=False,
                                   edgecolor='cyan',linewidth=1.7))
        ax.set_title(title); ax.set(xlabel='World x',ylabel='World y',xlim=(0,world),ylim=(world,0))
        fig.colorbar(im,ax=ax,label='Discrete probability mass')
    fig.suptitle(f"{report['status']} | recovered {report['layers_recovered']}/{report['layers_attempted']} layers")
    fig.tight_layout(); fig.savefig(output/'probability.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(len(layers),4,figsize=(14,3.4*len(layers)),squeeze=False)
    residual_max=max(float(x['residual_before'].max()) for x in layers)
    for row,x in enumerate(layers):
        items=((x['residual_before'],'Residual before','viridis',0,residual_max),
               (x['positive'].astype(float)-x['negative'].astype(float),'Constraints (+1/-1/0)','coolwarm',-1,1),
               (x.get('layer_response',x.get('full_mask',x['unknown'])),
                'Layer response (white=1)','gray',0,1),
               (x.get('residual_after',x['residual_before']),'Residual after','viridis',0,residual_max))
        for ax,(image,title,cmap,vmin,vmax) in zip(axes[row],items):
            ax.imshow(image,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_title(f"L{row+1}: {title}")
    fig.tight_layout(); fig.savefig(output/'layers.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-id',type=int,required=True); p.add_argument('--view-id',type=int,required=True)
    p.add_argument('--data-root'); p.add_argument('--spacing',type=float,default=16)
    p.add_argument('--temperature',type=float,default=.01); p.add_argument('--tolerance',type=float,default=1e-12)
    p.add_argument('--alpha',type=float,default=1); p.add_argument('--beta',type=float,default=0)
    p.add_argument('--boundary-lambda',type=float,default=0); p.add_argument('--boundary-sigma',type=float,default=1)
    p.add_argument('--max-layers',type=int,default=32); p.add_argument('--backend',choices=['auto','numba','reference'],default='auto')
    p.add_argument('--joint-samples',type=int,default=0)
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); dataset=Phase2Dataset(a.data_root)
    result=recover(dataset,a.scene_id,a.view_id,spacing=a.spacing,temperature=a.temperature,
        tolerance=a.tolerance,alpha=a.alpha,beta=a.beta,boundary_lambda=a.boundary_lambda,
        boundary_sigma=a.boundary_sigma,max_layers=a.max_layers,backend=a.backend,
        joint_samples=a.joint_samples,seed=a.seed)
    report=dict(result[0]); report['window']=dataset.get_observation(a.scene_id,a.view_id)['window'].tolist()
    report['world_size']=int(dataset.manifest['global_size'])
    save_run(a.output,report,*result[1:])
    print(json.dumps({k:report[k] for k in ('status','failure_reason','layers_recovered','layers_attempted','final_mae','elapsed_seconds')},indent=2))

if __name__=='__main__': main()
