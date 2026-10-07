"""Run matched physical edge and uniform/adaptive comparisons on one real view."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .physical_cost import BoundaryEvaluator
from .adaptive import adaptive_search, teacher_probabilities
from .evaluate import edge_ablation, search_comparison, plot_record
from .record_io import save
from . import checks


def compare(dataset, config, output):
    from rind_phase1.search import CandidateDomain, uniform_grid, uniform_search
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    scene,view = config['scene_id'],config['view_id']
    sample = dataset.get_observation(scene,view)
    domain = CandidateDomain(tuple(sample['window']),int(dataset.manifest['global_size']),False)
    grid = uniform_grid(domain,config['candidate_spacing'])
    tau = config['temperature']
    # Warm compiled renderer outside measured search; never add truth to targets.
    dataset.rerender(scene,sample['window'],dataset.get_scene(scene)['drivers'][0],
                     backend=config.get('backend','auto'))
    def uniform_record(beta):
        evaluator = BoundaryEvaluator(dataset,scene,view,alpha=1.,beta=beta,
            boundary_lambda=config.get('boundary_lambda',0.),
            boundary_sigma=config.get('boundary_sigma',1.),backend=config.get('backend','auto'))
        result = uniform_search(domain,evaluator,spacing=config['candidate_spacing'])
        settings = {'dataset':{'manifest':dataset.manifest},'support':grid.metadata(),
                    'physics':{**evaluator.settings,'edge_computed':beta>0,
                               'edge_definition':'internal four-neighbor symmetric mean distance/window diagonal',
                               'empty_edge_convention':'empty to nonempty=0; nonempty to empty=1'},
                    'temperature':tau,'search_mode':'uniform'}
        config_id = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        valid,cost = result.valid,result.physical_cost
        with np.errstate(over='ignore',under='ignore'):
            weights = np.exp(-(cost[valid]-cost[valid].min())/tau)
        probability = np.zeros(len(valid)); probability[valid] = weights/weights.sum()
        record = {'scene_id':scene,'view_id':view,'window':sample['window'],
                  'candidate_xy':grid.candidate_xy,'valid':valid,'evaluated':result.evaluated,
                  'physical_cost':cost,'teacher_prob':probability,'temperature':tau,'config_id':config_id,
                  'metadata':{'complete':result.complete,'settings':settings,
                              'num_physics_evaluations':result.num_physics_evaluations,
                              'elapsed_seconds':result.elapsed_seconds}}
        checks.teacher(record)
        return record
    response = uniform_record(0.)
    edge = uniform_record(config.get('edge_beta',.1))
    save(response,output/'response_teacher.npz')
    save(edge,output/'edge_teacher.npz')
    evaluator = BoundaryEvaluator(dataset,scene,view,alpha=1.,beta=0.,
        boundary_lambda=config.get('boundary_lambda',0.),boundary_sigma=config.get('boundary_sigma',1.),
        backend=config.get('backend','auto'))
    adaptive = adaptive_search(grid,evaluator,budget=config.get('adaptive_budget',len(grid.candidate_xy)),
        coarse_stride=config.get('coarse_stride',2),retain_fraction=config.get('retain_fraction',.5),
        exploration_fraction=config.get('exploration_fraction',.1),seed=config.get('seed',0))
    coverage = {k:response[k] for k in ('scene_id','view_id','window','candidate_xy','valid','config_id')}
    coverage.update(covered=adaptive.covered,metadata={'search_mode':'adaptive','complete':adaptive.complete,
                    'num_physics_evaluations':adaptive.num_physics_evaluations,'elapsed_seconds':adaptive.elapsed_seconds})
    save(coverage,output/'adaptive_coverage.npz')
    report = {'edge_ablation':edge_ablation(response,edge,threshold=config['compatibility_threshold']),
              'adaptive_complete':adaptive.complete,'adaptive_budget_exhausted':adaptive.budget_exhausted,
              'student_edge_ablation':'pending matched real trained checkpoints',
              'timing_protocol':'single CPU run; renderer warmed using one excluded diagnostic render; not a controlled repeated benchmark'}
    if adaptive.complete:
        report['search_comparison'] = search_comparison(response,adaptive.covered,
            threshold=config['compatibility_threshold'],adaptive_evaluations=adaptive.num_physics_evaluations,
            adaptive_seconds=adaptive.elapsed_seconds)
        target = {**response,'candidate_xy':adaptive.candidate_xy,'valid':adaptive.valid,
                  'physical_cost':adaptive.physical_cost,'teacher_prob':teacher_probabilities(adaptive,temperature=tau),
                  'evaluated':np.ones(len(adaptive.valid),dtype=bool)}
        adaptive_settings = {**response['metadata']['settings'],'search_mode':'adaptive',
                             'retention':{k:config.get(k) for k in ('adaptive_budget','coarse_stride','retain_fraction','exploration_fraction','seed')}}
        target['config_id'] = hashlib.sha256(json.dumps(adaptive_settings,sort_keys=True).encode()).hexdigest()
        target['metadata'] = {'complete':True,'settings':adaptive_settings,
            'num_physics_evaluations':adaptive.num_physics_evaluations,'elapsed_seconds':adaptive.elapsed_seconds}
        checks.teacher(target)
        save(target,output/'adaptive_teacher.npz')
        plot_record(target,None,output/'adaptive_teacher.svg')
    else:
        report['search_comparison'] = {'status':'incomplete; no final adaptive teacher constructed'}
    plot_record(response,None,output/'response_teacher.svg',sample['response'])
    plot_record(edge,None,output/'edge_teacher.svg')
    (output/'comparisons.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('data_root','teacher_source'):
        config[key] = str((args.config.parent/config[key]).resolve())
    source = Path(config['teacher_source'])
    sys.path[:0] = [str(source/'src'),str(source/'vendor/rind-dataset')]
    from rind_phase1.data import Phase1Dataset
    report = compare(Phase1Dataset(config['data_root']),config,args.output)
    print(f"Comparisons saved to {args.output}; adaptive_complete={report['adaptive_complete']}")


if __name__ == '__main__': main()
