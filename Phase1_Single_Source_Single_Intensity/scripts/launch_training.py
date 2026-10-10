"""Terminal parameter overrides; preserves frozen experiment configurations."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run-name', default='round1_terminal', help='Experiment name; use a new name when changing a started run')
    p.add_argument('--config', type=Path, help='Optional JSON template for a new run')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='FIELD=JSON',
                   help='Override an existing field, e.g. --set training.weight_decay=0.001; repeatable')
    p.add_argument('--variant', choices=['response','edge'], default='response', help='Run response before the paired edge ablation')
    p.add_argument('--epochs', type=positive_int)
    p.add_argument('--batch-size', type=positive_int)
    p.add_argument('--lr', type=positive_float, help='AdamW learning rate')
    p.add_argument('--patience', type=positive_int, help='Stop after this many epochs without validation improvement')
    p.add_argument('--seeds', type=nonnegative_int, nargs='+', help='Training seeds, e.g. --seeds 0 1 2')
    p.add_argument('--device', choices=['cpu','cuda'])
    p.add_argument('--threads', type=positive_int, help='PyTorch and Numba CPU threads')
    p.add_argument('--hidden', type=positive_int)
    p.add_argument('--fourier-frequencies', type=nonnegative_int)
    p.add_argument('--scene-counts', type=positive_int, nargs=3, metavar=('TRAIN','VAL','TEST'))
    p.add_argument('--window-sizes', type=int, nargs='+', choices=[16,32,64,128,256,512])
    p.add_argument('--spacing', type=positive_float, help='Candidate grid spacing in world coordinates')
    p.add_argument('--temperature', type=positive_float, help='Teacher softmax temperature')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Show/check effective settings; do not write files or train')
    mode.add_argument('--configure-only', action='store_true', help='Save settings without starting training')
    return p


def configure(args, project=PROJECT):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.run_name):
        raise ValueError('run-name must contain only letters, digits, underscores or hyphens')
    filename = 'experiment_terminal.json' if args.run_name == 'round1_terminal' else f'experiment_{args.run_name}.json'
    path = project/'configs'/filename
    original = json.loads(path.read_text()) if path.exists() else None
    template = (project/args.config) if args.config is not None else project/('configs/experiment_consistent.json' if (project/'configs/experiment_consistent.json').exists() else 'configs/experiment_round1.json')
    config = json.loads(json.dumps(original)) if original is not None and args.config is None else json.loads(template.read_text())
    if original is None or args.config is not None:
        config.update(name=args.run_name,output_root=f'outputs/experiments/{args.run_name}')
    for argument, field in [('epochs','epochs'),('batch_size','batch_size'),('lr','learning_rate'),
                            ('patience','patience'),('seeds','seeds'),('device','device'),
                            ('threads','num_threads'),('hidden','hidden'),('fourier_frequencies','fourier_frequencies')]:
        if getattr(args,argument) is not None:
            config['training'][field] = getattr(args,argument)
    if args.scene_counts is not None:
        config['scene_counts'] = dict(zip(('train','validation','test'),args.scene_counts))
    if args.window_sizes is not None:
        config['window_sizes'] = args.window_sizes
    for key in ('spacing','temperature'):
        if getattr(args,key) is not None:
            config['teacher'][key] = getattr(args,key)
    for override in args.overrides:
        key, separator, raw = override.partition('=')
        if not separator:
            raise ValueError('--set requires FIELD=JSON')
        if key in ('name','output_root'):
            raise ValueError('Use --run-name to set the experiment name and output directory')
        parts = key.split('.')
        node = config
        for part in parts[:-1]:
            if not isinstance(node,dict) or part not in node:
                raise ValueError(f'unknown config field: {key}')
            node = node[part]
        if not isinstance(node,dict) or parts[-1] not in node:
            raise ValueError(f'unknown config field: {key}')
        value = json.loads(raw)
        old = node[parts[-1]]
        numeric = type(old) in (int,float) and type(value) in (int,float)
        if type(value) is not type(old) and not numeric:
            raise ValueError(f'{key}: expected {type(old).__name__}')
        node[parts[-1]] = value
    t = config['training']
    for key in ('epochs','patience','batch_size','num_threads','hidden'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if type(t['fourier_frequencies']) is not int or t['fourier_frequencies'] < 0:
        raise ValueError('fourier_frequencies must be a nonnegative integer')
    if t['device'] not in ('cpu','cuda'):
        raise ValueError('supported training devices: cpu, cuda')
    if not t['seeds'] or any(type(x) is not int or x < 0 for x in t['seeds']):
        raise ValueError('seeds must be nonempty nonnegative integers')
    for node, keys, positive in [(t,('learning_rate','grad_clip'),True),
                                 (t,('weight_decay',),False),
                                 (config['teacher'],('spacing','temperature','boundary_sigma'),True),
                                 (config['teacher'],('alpha','boundary_lambda'),False)]:
        for key in keys:
            v = node[key]
            if type(v) not in (int,float) or not math.isfinite(v) or v < 0 or (positive and v == 0):
                raise ValueError(f'invalid {key}: {v}')
    if set(config['scene_counts']) != {'train','validation','test'} or any(type(v) is not int or v < 1 for v in config['scene_counts'].values()):
        raise ValueError('scene_counts requires positive integer train/validation/test counts')
    if not config['window_sizes'] or any(type(v) is not int or v not in (16,32,64,128,256,512) for v in config['window_sizes']):
        raise ValueError('unsupported window_sizes')
    if type(config['views_per_size']) is not int or not 1 <= config['views_per_size'] <= 3:
        raise ValueError('views_per_size must be 1..3')
    if type(config['split_seed']) is not int or config['split_seed'] < 0:
        raise ValueError('split_seed must be nonnegative integer')
    for variant in config['variants'].values():
        if type(variant['beta']) not in (int,float) or not math.isfinite(variant['beta']) or variant['beta'] < 0:
            raise ValueError('beta must be finite and nonnegative')
    e = config['evaluation']
    if not math.isfinite(e['compatibility_threshold']) or e['compatibility_threshold'] < 0:
        raise ValueError('compatibility_threshold must be finite and nonnegative')
    for key, minimum in [('prediction_chunk',1),('plot_count',0)]:
        if type(e[key]) is not int or e[key] < minimum:
            raise ValueError(f'invalid evaluation.{key}')
    if len(set(config['training']['seeds'])) != len(config['training']['seeds']):
        raise ValueError('training seeds must be unique')
    if len(set(config['window_sizes'])) != len(config['window_sizes']):
        raise ValueError('window sizes must be unique')
    if config['training']['hidden'] < 2:
        raise ValueError('hidden must be at least 2')
    if config.get('protocol_version',1)>=2:
        from rind_phase1.protocol import validate_research_config
        validate_research_config(config)
    output = (project/config['output_root']).resolve()
    if config != original and output.exists() and any(output.iterdir()):
        raise ValueError('This output directory already contains a run. Use a new --run-name to change parameters.')
    return path,config,output


def main():
    p = parser()
    args = p.parse_args()
    os.chdir(PROJECT)
    try:
        path, config, output = configure(args)
        os.environ['NUMBA_NUM_THREADS'] = str(config['training']['num_threads'])
        import torch
        from rind_phase1.data import Phase1Dataset
        from part_e.evaluate import evaluate_record
        dataset = Phase1Dataset(config['data_root'])
        if args.variant == 'edge' and not (output/'response/records.json').is_file():
            raise ValueError('Run --variant response with this run-name before the edge ablation')
        if config.get('protocol_version',1)>=2:
            from rind_phase1.protocol import fixed_selection
            fixed_selection(config,dataset)
        if sum(config['scene_counts'].values()) > dataset.num_scenes:
            raise ValueError('scene counts exceed installed dataset')
        if config['training']['device']=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available; select --device cpu')
    except (ValueError, ImportError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(f'Config: {path.relative_to(PROJECT)}\nOutput: {output}',flush=True)
    print(json.dumps(config,indent=2),flush=True)
    print('Observations: '+', '.join(f'{split}={count*len(config["window_sizes"])*config["views_per_size"]}' for split,count in config['scene_counts'].items()),flush=True)
    if args.check:
        print('Setup OK. No files written; training not started.')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or json.loads(path.read_text()) != config:
        temp = path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(config,indent=2)+'\n')
        temp.replace(path)
    if args.configure_only:
        print('Settings saved. Run again with the same --run-name to start.')
        return
    output.mkdir(parents=True,exist_ok=True)
    print(f'Starting/resuming. Log: {output/"training.log"}',flush=True)
    with (output/'training.log').open('a',buffering=1) as log:
        process = subprocess.Popen([sys.executable,'-u','-m','rind_phase1.experiments','all',
                                    '--config',str(path),'--variant',args.variant],
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line)
            code = process.wait()
        except KeyboardInterrupt:
            # Terminal SIGINT also reaches the child in the same foreground group.
            process.wait()
            raise SystemExit(130)
    if code:
        raise SystemExit(code)
    print(f'Finished. Results: {output/args.variant/"RESULTS.md"}')


if __name__=='__main__': main()
