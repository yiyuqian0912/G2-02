"""Reproducible Phase I experiment stages. Run from the repository checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .data import PROJECT_ROOT, Phase1Dataset
from .interfaces import INTERFACE_VERSION, training_record
from .teacher import generate_uniform_teacher, load_teacher_record, save_teacher_record


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def source_hashes():
    paths = list((PROJECT_ROOT/'src/rind_phase1').glob('*.py'))
    paths += list((PROJECT_ROOT/'part_e').glob('*.py'))
    return {str(p.relative_to(PROJECT_ROOT)):file_hash(p) for p in sorted(paths)}


def read_config(path):
    c = json.loads(Path(path).read_text())
    counts = c['scene_counts']
    if set(counts) != {'train','validation','test'} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('three positive scene counts required')
    sizes = c['window_sizes']
    if not sizes or len(set(sizes)) != len(sizes) or any(x not in (16,32,64,128,256,512) for x in sizes):
        raise ValueError('unique native window sizes required')
    if type(c['views_per_size']) is not int or not 1 <= c['views_per_size'] <= 3:
        raise ValueError('views_per_size must be 1..3')
    t = c['training']
    if not t['seeds'] or len(set(t['seeds'])) != len(t['seeds']):
        raise ValueError('distinct training seeds required')
    for k in ('epochs','patience','batch_size','num_threads','hidden'):
        if type(t[k]) is not int or t[k] <= 0:
            raise ValueError(f'positive integer {k} required')
    for k in ('learning_rate','grad_clip'):
        if not np.isfinite(t[k]) or t[k] <= 0:
            raise ValueError(f'positive {k} required')
    from .protocol import validate_research_config
    validate_research_config(c)
    return c


def context(c):
    root = (PROJECT_ROOT/c['output_root']).resolve()
    dataset = Phase1Dataset(PROJECT_ROOT/c['data_root'])
    return root,dataset


def plan(c, root, dataset):
    """Freeze selection without looking at responses, costs, sources, or test metrics."""
    counts = c['scene_counts']
    if sum(counts.values()) > dataset.num_scenes:
        raise ValueError('requested scene count exceeds release')
    rng = np.random.default_rng(c['split_seed'])
    shuffled = rng.permutation(dataset.num_scenes).tolist()
    splits, observations, offset = {}, {}, 0
    for split in ('train','validation','test'):
        splits[split] = shuffled[offset:offset+counts[split]]
        offset += counts[split]
        observations[split] = []
        for scene in splits[split]:
            # Stored window metadata only; no response, source, or geometry based selection.
            views = dataset._base.local_views[scene,:int(dataset._base.view_counts[scene])]
            for size in c['window_sizes']:
                available = np.flatnonzero(views[:,2] == size)
                if len(available) < c['views_per_size']:
                    raise ValueError(f'missing size {size} in scene {scene}')
                chosen = rng.choice(available,size=c['views_per_size'],replace=False)
                observations[split].extend(dict(scene_id=int(scene),view_id=int(v),size=int(size)) for v in sorted(chosen))
    split_manifest = None
    if c.get('protocol_version',1)>=2:
        from .protocol import fixed_selection
        splits,observations,split_manifest=fixed_selection(c,dataset)
    protocol = dict(interface_version=INTERFACE_VERSION,config=c,config_sha256=digest(c),
                    dataset_manifest_sha256=digest(dataset.manifest),source_sha256=source_hashes(),
                    scene_splits=splits,observations=observations,
                    selection='seeded scene permutation and seeded uniform selection within each size; no response filtering')
    if split_manifest is not None:
        protocol['split_manifest_sha256']=digest(split_manifest)
        protocol['selection']='fixed scene pools; stable per-scene/size windows, independent of training count'
    path = root/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('experiment directory already has a different config/data/code/selection; use a new output_root')
    else:
        write_json(path,protocol)
    return protocol


def prepare(c, root, dataset, protocol, variant):
    settings = {**c['teacher'],**c['variants'][variant]}
    entries = {k:[] for k in protocol['observations']}
    rows = []
    directory = root/variant
    total = sum(map(len,protocol['observations'].values()))
    started = time.perf_counter()
    for split, samples in protocol['observations'].items():
        for item in samples:
            scene,view = item['scene_id'],item['view_id']
            path = directory/'teachers'/f's{scene}_v{view}.npz'
            if path.exists():
                record = load_teacher_record(path)
                if record['metadata'].get('experiment_protocol') != digest(protocol):
                    raise ValueError(f'teacher cache protocol mismatch: {path}')
            else:
                record = generate_uniform_teacher(dataset,scene,view,**settings)
                record['metadata']['experiment_protocol'] = digest(protocol)
                temp = path.with_suffix('.tmp.npz')
                save_teacher_record(record,temp)
                temp.replace(path)
            training_record(record,dataset.get_observation(scene,view))
            valid,cost,q = record['valid'],record['physical_cost'],record['teacher_prob']
            low = valid & (cost <= c['evaluation']['compatibility_threshold'])
            rows.append(dict(scene_id=scene,view_id=view,split=split,size=item['size'],
                valid_candidates=int(valid.sum()),minimum_cost=float(cost[valid].min()),
                compatible_candidates=int(low.sum()),teacher_compatible_mass=float(q[low].sum()),
                teacher_entropy=float(-(q[q>0]*np.log(q[q>0])).sum()),
                sha256=file_hash(path),path=str(path.relative_to(directory))))
            entries[split].append(str(path.relative_to(directory)))
            if len(rows)%32 == 0 or len(rows)==total:
                print(f'prepare {variant}: {len(rows)}/{total}, {time.perf_counter()-started:.1f}s',flush=True)
    write_json(directory/'records.json',entries)
    # Hashes allow resume/evaluation to detect modified targets, not just a renamed manifest.
    write_json(directory/'teachers.json',dict(records=rows,records_sha256=digest(rows),
        protocol_sha256=digest(protocol),elapsed_seconds=time.perf_counter()-started,
        observations_without_compatible_candidates=sum(r['compatible_candidates']==0 for r in rows)))
    return entries


def prepared(c, root, protocol, variant):
    directory=root/variant
    index=json.loads((directory/'teachers.json').read_text())
    if index['protocol_sha256'] != digest(protocol):
        raise ValueError('prepared targets belong to another protocol')
    entries=json.loads((directory/'records.json').read_text())
    expected={k:[] for k in ('train','validation','test')}
    for row in index['records']:
        if file_hash(directory/row['path']) != row['sha256']:
            raise ValueError('teacher record was modified after preparation')
        expected[row['split']].append(row['path'])
    if entries != expected:
        raise ValueError('records manifest changed after preparation')
    return directory,index


def atomic_checkpoint(path, checkpoint):
    import torch
    temp=path.with_suffix('.tmp.pt')
    torch.save(checkpoint,temp)
    temp.replace(path)


def train(c,root,dataset,protocol,variant,seed):
    import torch
    from .model import SourceEnergyField
    from .train import load_splits,run_epoch
    directory,index=prepared(c,root,protocol,variant)
    settings=c['training']
    torch.set_num_threads(settings['num_threads'])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    device=settings['device']
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    # load_splits checks all IDs; test observations are not scored during training.
    splits=load_splits(directory/'records.json',dataset,settings.get('normalization_support','geometry'))
    model_config=dict(hidden=settings['hidden'],fourier_frequencies=settings['fourier_frequencies'],
                      coordinate_scale=float(dataset.manifest['global_size']),
                      use_obstacle=settings.get('use_obstacle',True),pixel_coordinates=settings.get('pixel_coordinates',False),
                      normalization_support=settings.get('normalization_support','geometry'))
    model=SourceEnergyField(**model_config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    run=directory/f'seed_{seed}'
    run.mkdir(parents=True,exist_ok=True)
    signature=digest(dict(protocol=digest(protocol),targets=index['records_sha256'],seed=seed))
    history,best,bad_epochs,start,steps=[],float('inf'),0,0,0
    if (run/'last.pt').exists():
        cp=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if cp['run_signature'] != signature:
            raise ValueError('resume checkpoint protocol/targets/seed mismatch')
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['torch_rng'])
        rng.bit_generator.state=cp['numpy_rng']
        history,best,bad_epochs,start,steps=cp['history'],cp['best_validation_ce'],cp['bad_epochs'],cp['epoch'],cp['global_step']
        if device=='cuda' and 'cuda_rng' in cp:
            torch.cuda.set_rng_state_all(cp['cuda_rng'])
        if device=='mps' and 'mps_rng' in cp:
            torch.mps.set_rng_state(cp['mps_rng'])
    epoch_steps=sum(int(np.ceil(sum(s==size for s in splits['train'].window_sizes)/settings['batch_size'])) for size in c['window_sizes'])
    print(f'train {variant} seed={seed}: {len(splits["train"])} train / {len(splits["validation"])} validation, resume epoch={start}',flush=True)
    for epoch in range(start,settings['epochs']):
        if bad_epochs >= settings['patience']:
            break
        tick=time.perf_counter()
        train_ce=run_epoch(model,splits['train'],settings['batch_size'],optimizer,rng,grad_clip=settings['grad_clip'],candidate_chunk=settings.get('candidate_chunk'))
        val_ce=run_epoch(model,splits['validation'],settings['batch_size'],candidate_chunk=settings.get('candidate_chunk'))
        steps+=epoch_steps
        improved=val_ce < best
        bad_epochs=0 if improved else bad_epochs+1
        best=min(best,val_ce)
        row=dict(epoch=epoch+1,train_ce=train_ce,validation_ce=val_ce,seconds=time.perf_counter()-tick)
        history.append(row)
        cp=dict(interface_version=INTERFACE_VERSION,model_config=model_config,model=model.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch+1,global_step=steps,seed=seed,run_signature=signature,
            protocol_sha256=digest(protocol),records_sha256=index['records_sha256'],
            scene_splits=protocol['scene_splits'],teacher_configs=sorted(splits['train'].config_ids),
            best_validation_ce=best,bad_epochs=bad_epochs,history=history,
            torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
            software=dict(torch=str(torch.__version__),numpy=np.__version__,device=device,num_threads=settings['num_threads']))
        if device=='cuda': cp['cuda_rng']=torch.cuda.get_rng_state_all()
        if device=='mps': cp['mps_rng']=torch.mps.get_rng_state()
        if improved: atomic_checkpoint(run/'best.pt',cp)
        atomic_checkpoint(run/'last.pt',cp)
        write_json(run/'training.json',dict(seed=seed,history=history,best_validation_ce=best,
                   selection='minimum validation cross entropy; test not evaluated during training'))
        print(f'{variant} seed={seed} epoch={epoch+1}: train={train_ce:.5f} val={val_ce:.5f} best={best:.5f} ({row["seconds"]:.1f}s)',flush=True)
    if not (run/'best.pt').exists():
        raise ValueError('training produced no checkpoint')
    return run


def evaluation(c,root,dataset,protocol,variant,seed):
    import torch
    # Part E lives in the checkout; no second copy of scientific metrics.
    from part_e.evaluate import evaluate_record,aggregate,window_summary,plot_record,unseen_scene_check,unseen_obstacle_combinations
    from .train import load_model
    from .predict import predict
    directory,index=prepared(c,root,protocol,variant)
    run=directory/f'seed_{seed}'
    model,cp=load_model(run/'best.pt')
    signature=digest(dict(protocol=digest(protocol),targets=index['records_sha256'],seed=seed))
    if cp.get('run_signature') != signature:
        raise ValueError('checkpoint does not match frozen protocol and targets')
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if last['epoch'] < c['training']['epochs'] and last['bad_epochs'] < c['training']['patience']:
        raise ValueError('training is not finished; test evaluation is locked')
    torch.set_num_threads(c['training']['num_threads'])
    model.to(c['training']['device'])
    rows,baselines,records,common_response,assisted=[],[],[],[],[]
    entries=json.loads((directory/'records.json').read_text())
    for i,filename in enumerate(entries['test']):
        record=load_teacher_record(directory/filename)
        sample=dataset.get_observation(record['scene_id'],record['view_id'])
        prediction=predict(model,record,sample,chunk=c['evaluation']['prediction_chunk'])
        row=evaluate_record(record,prediction,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing'])
        # Uniform over exactly the same declared student normalization support.
        support=prediction.get('support',record['valid'])
        uniform={**prediction,'student_prob':support.astype(float)/support.sum()}
        baseline=evaluate_record(record,uniform,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing'])
        response=sample['response']
        row['response_group']='uniform' if np.all(response==response.flat[0]) else 'nonuniform'
        rows.append(row);baselines.append(baseline);records.append(record)
        from .sampling import probability_from_energy
        assisted_prediction={**prediction,'support':record['valid'],'normalization_support':'geometry',
                             'student_prob':probability_from_energy(prediction['energy'],record['valid'])}
        assisted.append(evaluate_record(record,assisted_prediction,threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing']))
        if variant != 'response':
            reference=load_teacher_record(root/'response'/filename)
            if not all(np.array_equal(reference[k],record[k]) for k in ('candidate_xy','valid','window')):
                raise ValueError('edge ablation requires identical response-baseline support')
            # Re-evaluation against an explicitly declared common response teacher.
            reference_prediction={**prediction,'config_id':reference['config_id']}
            common_response.append(evaluate_record(reference,reference_prediction,
                threshold=c['evaluation']['compatibility_threshold'],spacing=c['teacher']['spacing']))
        path=run/'predictions'/f's{record["scene_id"]}_v{record["view_id"]}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**prediction)
        if i < c['evaluation']['plot_count']:
            plot_record(record,prediction,run/f'example_{i}.svg',response)
    # Categorical obstacle-type multisets are a limited, explicitly named diagnostic.
    signatures={}
    for scene in protocol['scene_splits']['train']+protocol['scene_splits']['test']:
        signatures[str(scene)]=dataset.get_scene(scene)['obstacle_types'].astype(int).tolist()
    report=dict(variant=variant,seed=seed,checkpoint_sha256=file_hash(run/'best.pt'),best_epoch=cp['epoch'],
        run_signature=signature,observations=rows,aggregate=aggregate(rows),by_window_size=window_summary(rows),
        by_response_group={g:aggregate([r for r in rows if r['response_group']==g]) for g in ('uniform','nonuniform')},
        normalization_support=model.normalization_support,use_obstacle=model.use_obstacle,
        geometry_assisted_secondary=aggregate(assisted),common_response_reference=aggregate(common_response or rows),
        uniform_baseline=aggregate(baselines),uniform_baseline_by_window_size=window_summary(baselines),
        unseen_scene=unseen_scene_check(records,protocol['scene_splits'],cp['scene_splits']['train']),
        obstacle_combinations=unseen_obstacle_combinations(rows,signatures,cp['scene_splits']['train']),
        interpretation='Discrete common grid; normalization_support declares whether hidden geometry masks student probabilities. Invalid mass and penalized cost are explicit. Constant/nonconstant response is not a count of independent geometric boundaries.')
    write_json(run/'test_metrics.json',report)
    print(f'evaluated {variant} seed={seed}: {len(rows)} observations',flush=True)
    return report


def summarize(c,root,variant):
    reports=[json.loads((root/variant/f'seed_{seed}'/'test_metrics.json').read_text()) for seed in c['training']['seeds']]
    common=set.intersection(*(set(r['aggregate']['metrics']) for r in reports))
    metrics={}
    for key in sorted(common):
        values=[r['aggregate']['metrics'][key]['scene_mean'] for r in reports]
        metrics[key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,per_seed=values)
    result=dict(variant=variant,seeds=c['training']['seeds'],scene_balanced_metrics=metrics,
                best_epochs=[r['best_epoch'] for r in reports],
                uniform_baseline=reports[0]['uniform_baseline'],
                note='Standard deviation across training seeds, not a confidence interval over independent datasets.')
    write_json(root/variant/'summary.json',result)
    lines=[f'# {variant}: first-round experiment','',f'Seeds: {c["training"]["seeds"]}; best epochs: {result["best_epochs"]}.','',
           '| Metric (scene mean) | Mean | Seed std |','|---|---:|---:|']
    for key in ('forward_kl','jensen_shannon','l1_distance','student_compatible_mass','student_expected_physical_cost','student_invalid_mass','student_expected_penalized_cost','student_entropy','teacher_entropy'):
        if key in metrics:
            m=metrics[key];lines.append(f'| {key} | {m["mean"]:.6f} | {m["std"]:.6f} |')
    lines+=['','These results apply only to the frozen scene subset, window sizes and candidate lattice. See per-seed reports for uniform baseline, window groups, and grid coverage.']
    (root/variant/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def compare_search(c,root,dataset,protocol,variant):
    """Compare adaptive retention against the exact same uniform reference costs."""
    from .physics import PhysicalEvaluator
    from .search import CandidateDomain,uniform_grid
    from part_e.adaptive import adaptive_search
    from part_e.evaluate import search_comparison
    directory,_=prepared(c,root,protocol,variant)
    paths=json.loads((directory/'records.json').read_text())['test']
    settings=c['search_comparison']
    rows=[]
    for filename in paths[:settings['max_observations']]:
        reference=load_teacher_record(directory/filename)
        scene,view=reference['scene_id'],reference['view_id']
        domain=CandidateDomain(tuple(reference['window']),int(dataset.manifest['global_size']),False)
        grid=uniform_grid(domain,c['teacher']['spacing'])
        kwargs={k:v for k,v in {**c['teacher'],**c['variants'][variant]}.items() if k not in ('spacing','temperature')}
        evaluator=PhysicalEvaluator(dataset,scene,view,**kwargs)
        result=adaptive_search(grid,evaluator,budget=len(grid.candidate_xy),
            coarse_stride=settings['coarse_stride'],retain_fraction=settings['retain_fraction'],
            exploration_fraction=settings['exploration_fraction'],seed=settings['seed'])
        row=dict(scene_id=scene,view_id=view,complete=result.complete)
        if result.complete:
            row.update(search_comparison(reference,result.covered,
                threshold=c['evaluation']['compatibility_threshold'],
                adaptive_evaluations=result.num_physics_evaluations,adaptive_seconds=result.elapsed_seconds))
        rows.append(row)
    report=dict(observations=rows,protocol_sha256=digest(protocol),
        note='Held-out fixed subset. Render-count and compatible-mass coverage comparison; one-run timing is not a controlled speed benchmark. Adaptive probabilities are conditional on retained support; no area weighting.')
    write_json(directory/'search_comparison.json',report)
    return report


def sampling_probe(c,root,dataset,protocol,variant,seed):
    """Validation-only physical check on an interleaved, finer candidate grid."""
    from .train import load_model
    from .predict import predict
    from part_e.evaluate import evaluate_record,aggregate
    spacing=c['evaluation'].get('probe_spacing')
    if spacing is None: raise ValueError('set evaluation.probe_spacing')
    if spacing>=c['teacher']['spacing']: raise ValueError('probe spacing must be finer than training spacing')
    directory,_=prepared(c,root,protocol,variant)
    model,_=load_model(directory/f'seed_{seed}'/'best.pt')
    rows=[]
    for item in protocol['observations']['validation'][:c['evaluation'].get('probe_observations',16)]:
        settings={**c['teacher'],**c['variants'][variant],'spacing':spacing}
        record=generate_uniform_teacher(dataset,item['scene_id'],item['view_id'],**settings)
        sample=dataset.get_observation(item['scene_id'],item['view_id'])
        result=predict(model,record,sample,chunk=c['evaluation']['prediction_chunk'])
        rows.append(evaluate_record(record,result,threshold=c['evaluation']['compatibility_threshold'],spacing=spacing))
    write_json(directory/f'seed_{seed}'/'sampling_probe.json',dict(split='validation',training_spacing=c['teacher']['spacing'],probe_spacing=spacing,
        observations=rows,aggregate=aggregate(rows),note='New physical targets on a finer grid. Does not establish pixel-scale convergence.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','prepare','train','evaluate','summarize','search','probe','all'])
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/experiment_consistent.json')
    parser.add_argument('--variant',default='response')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    c=read_config(args.config)
    if args.variant not in c['variants']:
        parser.error('unknown variant')
    root,dataset=context(c)
    protocol=plan(c,root,dataset)
    seeds=c['training']['seeds'] if args.seed is None else [args.seed]
    if not set(seeds)<=set(c['training']['seeds']):
        parser.error('seed must be declared in the frozen config')
    if args.stage=='plan':
        print(f'Frozen protocol: {root/"protocol.json"}',flush=True)
    if args.stage=='probe':
        for seed in seeds: sampling_probe(c,root,dataset,protocol,args.variant,seed)
    if args.stage=='search': compare_search(c,root,dataset,protocol,args.variant)
    if args.stage in ('prepare','all'): prepare(c,root,dataset,protocol,args.variant)
    if args.stage in ('train','all'):
        for seed in seeds: train(c,root,dataset,protocol,args.variant,seed)
    if args.stage in ('evaluate','all'):
        for seed in seeds: evaluation(c,root,dataset,protocol,args.variant,seed)
    if args.stage in ('summarize','all') and all((root/args.variant/f'seed_{seed}'/'test_metrics.json').exists() for seed in c['training']['seeds']):
        summarize(c,root,args.variant)
    if args.stage=='all' and c['evaluation'].get('run_search_comparison',False):
        compare_search(c,root,dataset,protocol,args.variant)


if __name__=='__main__': main()
