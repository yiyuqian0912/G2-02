"""Explicit, reproducible Section 16.2 trials; no teacher physics or candidate search."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from data import validate_record_splits
from run_data import (METADATA_FIELDS, comparison_group, evaluation_fingerprint,
                      load_record_file, make_mock_splits, metadata_for)
from train import TrainConfig, run_training


def _config(values):
    if not isinstance(values, dict):
        raise ValueError('config must be a JSON object')
    try:
        return TrainConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid trial config: {exc}') from exc


def _validate_metadata(metadata):
    for key in ('dataset_version', 'scene_split_version', 'area_weight_convention'):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    for key in ('candidate_search_config', 'teacher_cost_config'):
        if not isinstance(metadata[key], dict) or not metadata[key]:
            raise ValueError(f'{key} must be a nonempty object')
    if type(metadata['candidate_count']) is not int or metadata['candidate_count'] <= 0:
        raise ValueError('candidate_count metadata must be a positive nominal count')
    cost = metadata['teacher_cost_config']
    if 'alpha' in cost or 'beta' in cost:
        if not all(key in cost and type(cost[key]) in (int, float) and math.isfinite(cost[key])
                   and cost[key] >= 0 for key in ('alpha', 'beta')) or cost['alpha'] + cost['beta'] <= 0:
            raise ValueError('teacher alpha and beta must both be finite, nonnegative, and not both zero')
    json.dumps(metadata, allow_nan=False)


def prepare_trials(manifest, base_dir=Path('.')):
    """Validate every trial and its data before any training or output writes."""
    if not isinstance(manifest, dict) or set(manifest) - {'base_config', 'trials'}:
        raise ValueError('manifest accepts only base_config and trials')
    base = manifest.get('base_config', {})
    _config(base)
    trials = manifest.get('trials')
    if not isinstance(trials, list) or not trials:
        raise ValueError('trials must be a nonempty list')
    prepared, names = [], set()
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) - {'name', 'config', 'mock', 'records'}:
            raise ValueError('trial accepts only name, config, mock, and records')
        name = trial.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('trial name must be 1-64 letters, digits, underscores or hyphens, starting alphanumeric')
        if name.casefold() in names:
            raise ValueError('trial names must be unique (case insensitive)')
        names.add(name.casefold())
        overrides = trial.get('config', {})
        _config(overrides)
        values = {**base, **overrides}
        if ('mock' in trial) == ('records' in trial):
            raise ValueError('each trial must specify exactly one of mock:true or records')
        if 'mock' in trial:
            if trial['mock'] is not True:
                raise ValueError('mock must be true')
            config = _config(values)
            defaults = metadata_for(TrainConfig())
            if any(getattr(config, key) != defaults[key] for key in METADATA_FIELDS if key != 'candidate_count'):
                raise ValueError('mock provenance cannot be relabeled as upstream data or alpha/beta tuning')
            splits = validate_record_splits(make_mock_splits(config))
            source = {'mode': 'mock', 'target_mode': 'physical_cost'}
        else:
            if not isinstance(trial['records'], str) or not trial['records']:
                raise ValueError('records must be a nonempty file path')
            path = (Path(base_dir) / trial['records']).resolve()
            splits, metadata = load_record_file(path, require_metadata=True)
            _validate_metadata(metadata)
            for key in METADATA_FIELDS:
                if key in values and values[key] != metadata[key]:
                    raise ValueError(f'{name}: {key} override disagrees with upstream records; provide a matching record file')
                values[key] = metadata[key]
            config = _config(values)
            source = {'mode': 'upstream', 'records': str(path)}
        requested_temperature = 'teacher_temperature' in base or 'teacher_temperature' in overrides
        supplied_probability = any('teacher_prob' in record for records in splits.values() for record in records)
        if requested_temperature and supplied_probability:
            raise ValueError(f'{name}: temperature trials require cost-only records; never discard supplied teacher_prob')
        signature = evaluation_fingerprint(splits['validation'], config.teacher_temperature)
        counts = [int(record['valid_mask'].sum()) for record in splits['validation']]
        prepared.append({'name': name, 'config': config, 'splits': splits, 'source': source,
                         'validation_fingerprint': signature, 'comparison_group': comparison_group(signature, config),
                         'valid_candidate_count_min': min(counts), 'valid_candidate_count_max': max(counts)})
    return prepared


def rank_results(results):
    """No global rank: only identical validation targets/provenance can be ranked."""
    groups = {}
    for row in results:
        row['rank_within_group'] = None
        if row['status'] == 'completed':
            groups.setdefault(row['comparison_group'], []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        key = lambda row: (row['forward_kl'], row['l1_distance'], row['entropy_mismatch'])
        ordered = sorted(group, key=key)
        previous, rank = None, 0
        for index, row in enumerate(ordered, 1):
            if key(row) != previous:
                rank, previous = index, key(row)
            row['rank_within_group'] = rank
    return results


RESULT_FIELDS = ['name', 'status', 'comparison_group', 'rank_within_group', 'forward_kl', 'l1_distance',
                 'entropy_mismatch', 'parameter_count', 'runtime_seconds', 'hidden_dim', 'candidate_count',
                 'teacher_temperature', 'alpha', 'beta', 'valid_candidate_count_min', 'valid_candidate_count_max',
                 'validation_fingerprint', 'evaluation_checkpoint', 'error']


def _save_results(results, output):
    rank_results(results)
    (output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def run_experiments(manifest, output_dir, base_dir=Path('.')):
    trials = prepare_trials(manifest, base_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('experiment output directory must be empty; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    results = []
    for trial in trials:
        config = trial['config']
        row = {key: trial[key] for key in ('name', 'comparison_group', 'validation_fingerprint',
                                           'valid_candidate_count_min', 'valid_candidate_count_max')}
        row.update(status='running', rank_within_group=None, hidden_dim=config.hidden_dim,
                   candidate_count=config.candidate_count, teacher_temperature=config.teacher_temperature,
                   alpha=config.teacher_cost_config.get('alpha'), beta=config.teacher_cost_config.get('beta'))
        try:
            trial_output = output / trial['name']
            summary = run_training(config, trial['splits'], trial_output)
            (trial_output / 'trial.json').write_text(json.dumps({'name': trial['name'], 'source': trial['source'],
                'metadata': metadata_for(config), 'config': asdict(config),
                'validation_fingerprint': trial['validation_fingerprint'],
                'comparison_group': trial['comparison_group']}, indent=2))
            row.update({key: summary['validation'][key] for key in ('forward_kl', 'l1_distance', 'entropy_mismatch')})
            row.update(status='completed', parameter_count=summary['parameter_count'],
                       runtime_seconds=summary['runtime_seconds'],
                       evaluation_checkpoint=str(Path(trial['name']) / summary['evaluation_checkpoint']))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            results.append(row)
            _save_results(results, output)
            raise
        results.append(row)
        _save_results(results, output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiments'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = run_experiments(manifest, args.output, args.manifest.resolve().parent)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
