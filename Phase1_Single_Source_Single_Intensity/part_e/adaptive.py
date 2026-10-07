"""Budgeted coarse-to-fine search on a declared uniform final lattice.

Final probabilities use only common-spacing points, never coarse trace points.
No global coverage guarantee is made; evaluate coverage against uniform search.
"""
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class AdaptiveResult:
    candidate_xy: np.ndarray
    valid: np.ndarray
    physical_cost: np.ndarray
    covered: np.ndarray
    num_physics_evaluations: int
    elapsed_seconds: float
    complete: bool
    budget_exhausted: bool
    evaluated_indices: np.ndarray


def adaptive_search(grid, evaluator, *, budget, coarse_stride=4,
                    retain_fraction=.5, exploration_fraction=.1, seed=0):
    """Refine promising coarse blocks; audit misses on this same grid.

    A block completes only after every final-grid point in it is evaluated.
    Partially evaluated blocks and scouting points stay in trace only.
    The budget counts new physical renders, excluding invalid/cache queries.
    """
    if type(budget) is not int or budget < 1 or type(coarse_stride) is not int or coarse_stride < 1:
        raise ValueError('positive integer budget and coarse_stride required')
    if not 0 < retain_fraction <= 1 or not 0 <= exploration_fraction <= 1:
        raise ValueError('invalid retention/exploration fractions')
    xy = np.asarray(grid.candidate_xy)
    indices = np.asarray(grid.grid_index)
    if xy.shape != (len(indices),2) or len(indices) == 0:
        raise ValueError('nonempty ordered uniform grid required')
    rows, cols = np.divmod(indices,len(grid.x_axis))
    blocks = {}
    for i,(row,col) in enumerate(zip(rows,cols)):
        blocks.setdefault((int(row)//coarse_stride,int(col)//coarse_stride),[]).append(i)
    results, trace, renders = {}, [], 0
    started = time.perf_counter()
    def query(index):
        nonlocal renders
        if index in results: return results[index]
        if renders >= budget: return None
        value = evaluator.evaluate(xy[index])
        if value.num_physics_evaluations not in (0,1):
            raise ValueError('scalar evaluator must count zero/one render')
        if value.valid and (not np.isfinite(value.physical_cost) or value.physical_cost < 0):
            raise ValueError('valid costs must be finite and nonnegative')
        if not value.valid and value.num_physics_evaluations:
            raise ValueError('invalid positions must not be rendered')
        renders += value.num_physics_evaluations
        results[index] = value
        trace.append(index)
        return value
    scouts = []
    for key, members in sorted(blocks.items()):
        center = np.mean(xy[members],axis=0)
        index = members[int(np.argmin(np.sum((xy[members]-center)**2,axis=1)))]
        value = query(index)
        if value is None: break
        scouts.append((float(value.physical_cost) if value.valid else np.inf,key))
    # If broad scouting is unfinished, no final support can be declared.
    scouting_complete = len(scouts) == len(blocks)
    covered = np.zeros(len(xy),dtype=bool)
    all_selected_complete = scouting_complete
    if scouting_complete:
        ranked = sorted(scouts,key=lambda item:(item[0],item[1]))
        kept = [key for _,key in ranked[:max(1,int(np.ceil(len(ranked)*retain_fraction)))]]
        remaining = [key for _,key in ranked if key not in kept]
        rng = np.random.default_rng(seed)
        if remaining:
            count = min(len(remaining),int(np.ceil(len(blocks)*exploration_fraction)))
            kept += [remaining[i] for i in rng.permutation(len(remaining))[:count]]
        for key in kept:
            members = blocks[key]
            completed = True
            for index in members:
                if query(index) is None:
                    completed = False
                    break
            if completed:
                covered[members] = True
            else:
                all_selected_complete = False
                break
    chosen = np.flatnonzero(covered)
    valid = np.array([results[i].valid for i in chosen],dtype=bool)
    costs = np.array([results[i].physical_cost if results[i].valid else np.inf for i in chosen])
    return AdaptiveResult(xy[chosen].copy(),valid,costs,covered,renders,
                          time.perf_counter()-started,all_selected_complete and bool(len(chosen)),
                          not all_selected_complete and renders >= budget,np.array(trace,dtype=int))


def teacher_probabilities(result, *, temperature):
    if not result.complete:
        raise ValueError('unfinished selected blocks cannot form a final adaptive teacher')
    if not np.isfinite(temperature) or temperature <= 0 or not result.valid.any():
        raise ValueError('positive temperature and nonempty valid support required')
    cost = result.physical_cost[result.valid]
    with np.errstate(over='ignore',under='ignore'):
        weights = np.exp(-(cost-cost.min())/temperature)
    probability = np.zeros(len(result.valid))
    probability[result.valid] = weights/weights.sum()
    return probability
