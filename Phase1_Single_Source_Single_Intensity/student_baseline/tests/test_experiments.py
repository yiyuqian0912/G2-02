from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import pytest
import torch

from data import collate_records
from experiments import prepare_trials, rank_results, run_experiments
from run_data import (comparison_group, evaluation_fingerprint, make_mock_splits, metadata_for)
from train import TrainConfig, run_training


def manifest():
    return {'base_config': {'epochs': 1, 'hidden_dim': 16, 'candidate_count': 16},
            'trials': [{'name': 'small', 'mock': True}, {'name': 'large', 'mock': True, 'config': {'hidden_dim': 32}}]}


def write_upstream(tmp_path, filename='teacher.pt', supplied=False, alpha=1., beta=2., candidates=16):
    config = TrainConfig(hidden_dim=16, epochs=1, candidate_count=candidates,
        dataset_version='fixture-v1', scene_split_version='fixture-splits-v1',
        candidate_search_config={'type': 'fixture', 'nominal_count': candidates},
        teacher_cost_config={'type': 'fixture', 'alpha': alpha, 'beta': beta},
        area_weight_convention='fixture equal area')
    splits = make_mock_splits(config)
    if supplied:
        for records in splits.values():
            for r in records:
                r['teacher_prob'] = collate_records([r])['teacher_prob'][0]
    path = tmp_path / filename
    torch.save({'splits': splits, 'metadata': metadata_for(config)}, path)
    return path, config, splits


@pytest.mark.parametrize('change', ['duplicate', 'case_duplicate', 'traversal', 'unknown', 'both_sources',
                                   'neither_source', 'bad_mock', 'bad_config', 'fractional_count', 'empty'])
def test_manifest_errors(change):
    m = manifest()
    if change == 'duplicate': m['trials'][1]['name'] = 'small'
    if change == 'case_duplicate': m['trials'][1]['name'] = 'SMALL'
    if change == 'traversal': m['trials'][0]['name'] = '../escape'
    if change == 'unknown': m['auto_search'] = True
    if change == 'both_sources': m['trials'][0]['records'] = 'x.pt'
    if change == 'neither_source': m['trials'][0].pop('mock')
    if change == 'bad_mock': m['trials'][0]['mock'] = False
    if change == 'bad_config': m['base_config']['imaginary_option'] = 1
    if change == 'fractional_count': m['base_config']['candidate_count'] = 2.5
    if change == 'empty': m['trials'] = []
    with pytest.raises(ValueError): prepare_trials(m)


def test_temperature_trials_change_targets_and_separate_groups():
    m = manifest()
    m['trials'][1]['config']['teacher_temperature'] = 0.3
    a, b = prepare_trials(m)
    assert all('teacher_prob' not in r for records in a['splits'].values() for r in records)
    qa = collate_records(a['splits']['validation'], a['config'].teacher_temperature)['teacher_prob']
    qb = collate_records(b['splits']['validation'], b['config'].teacher_temperature)['teacher_prob']
    assert not torch.allclose(qa, qb)
    assert a['comparison_group'] != b['comparison_group']
    m['trials'][1]['config'].pop('teacher_temperature')
    a, b = prepare_trials(m)
    assert a['comparison_group'] == b['comparison_group']


def test_temperature_rejects_supplied_probabilities_and_preflights_all_trials(tmp_path):
    path, _, _ = write_upstream(tmp_path, supplied=True)
    m = {'base_config': {'epochs': 1}, 'trials': [
        {'name': 'ok', 'mock': True},
        {'name': 'invalid', 'records': path.name, 'config': {'teacher_temperature': 0.2}}]}
    with pytest.raises(ValueError, match='cost-only'):
        run_experiments(m, tmp_path / 'out', tmp_path)
    assert not (tmp_path / 'out').exists()
    m['trials'][1]['config'].clear()
    prepared = prepare_trials(m, tmp_path)
    assert 'teacher_prob' in prepared[1]['splits']['train'][0]


def test_upstream_metadata_and_no_candidate_resampling(tmp_path):
    path, config, original = write_upstream(tmp_path)
    m = {'trials': [{'name': 'upstream', 'records': path.name, 'config': {'hidden_dim': 16, 'epochs': 1}}]}
    trial = prepare_trials(m, tmp_path)[0]
    assert metadata_for(trial['config']) == metadata_for(config)
    assert torch.equal(trial['splits']['train'][0]['candidate_xy'], original['train'][0]['candidate_xy'])
    m['trials'][0]['config']['candidate_count'] = 32
    with pytest.raises(ValueError, match='matching record file'): prepare_trials(m, tmp_path)
    m['trials'][0]['config'].pop('candidate_count')
    m['trials'][0]['config']['teacher_cost_config'] = {'type': 'fixture', 'alpha': 2., 'beta': 1.}
    with pytest.raises(ValueError, match='matching record file'): prepare_trials(m, tmp_path)


def test_missing_and_invalid_upstream_metadata(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save(make_mock_splits(TrainConfig()), path)
    m = {'trials': [{'name': 'a', 'records': path.name}]}
    with pytest.raises(ValueError, match='require metadata'): prepare_trials(m, tmp_path)
    path, _, _ = write_upstream(tmp_path, filename='legacy.pt', alpha=-1)
    with pytest.raises(ValueError, match='alpha and beta'): prepare_trials(m, tmp_path)


def test_alpha_beta_and_candidate_files_form_separate_comparisons(tmp_path):
    a, _, _ = write_upstream(tmp_path, 'a.pt')
    b, _, _ = write_upstream(tmp_path, 'b.pt', alpha=2., beta=1.)
    c, _, _ = write_upstream(tmp_path, 'c.pt', candidates=25)
    trials = prepare_trials({'trials': [{'name': p.stem, 'records': p.name} for p in (a, b, c)]}, tmp_path)
    assert len({t['comparison_group'] for t in trials}) == 3
    assert [t['config'].candidate_count for t in trials] == [16, 16, 25]


def test_fingerprints_cover_observation_coordinates_masks_and_probabilities():
    config = TrainConfig(candidate_count=16)
    records = make_mock_splits(config)['validation']
    base = evaluation_fingerprint(records, 0.1)
    for key in ('response', 'candidate_xy', 'physical_cost'):
        changed = deepcopy(records)
        changed[0][key].reshape(-1)[0] += 0.5
        assert evaluation_fingerprint(changed, 0.1) != base
    changed = deepcopy(records)
    changed[0]['valid_mask'] = torch.ones(16, dtype=torch.bool)
    changed[0]['valid_mask'][0] = False
    assert evaluation_fingerprint(changed, 0.1) != base
    assert comparison_group(base, config) != comparison_group(base, replace(config, teacher_cost_config={'alpha': 1, 'beta': 2}))


def test_ranking_never_crosses_groups():
    rows = [{'name': name, 'status': status, 'comparison_group': group, 'forward_kl': value,
             'l1_distance': value, 'entropy_mismatch': value}
            for name, status, group, value in [('a', 'completed', 'same', 0.2), ('b', 'completed', 'same', 0.1),
                                              ('c', 'completed', 'other', 0.), ('d', 'failed', 'same', 0.)]]
    rank_results(rows)
    assert [r['rank_within_group'] for r in rows] == [2, 1, None, None]


def test_experiment_outputs_and_seed_reproducibility(tmp_path):
    m = manifest()
    results = run_experiments(m, tmp_path / 'experiment')
    assert len(results) == 2 and all(r['status'] == 'completed' for r in results)
    assert {r['rank_within_group'] for r in results} == {1, 2}
    assert results[0]['parameter_count'] < results[1]['parameter_count']
    for name in ('small', 'large'):
        output = tmp_path / 'experiment' / name
        for filename in ('config.json', 'summary.json', 'trial.json', 'metrics.jsonl', 'last.pt', 'best.pt'):
            assert (output / filename).is_file()
    assert (tmp_path / 'experiment' / 'results.csv').is_file()
    same = prepare_trials(m)[0]
    repeat = run_training(same['config'], same['splits'], tmp_path / 'repeat')
    first = json.loads((tmp_path / 'experiment' / 'small' / 'summary.json').read_text())
    assert first['validation'] == repeat['validation']
    a = torch.load(tmp_path / 'experiment' / 'small' / 'last.pt', weights_only=False)
    b = torch.load(tmp_path / 'repeat' / 'last.pt', weights_only=False)
    assert all(torch.equal(value, b['model'][key]) for key, value in a['model'].items())
    with pytest.raises(ValueError, match='empty'): run_experiments(m, tmp_path / 'experiment')


def test_upstream_provenance_saved_in_checkpoint_and_trial(tmp_path):
    path, config, _ = write_upstream(tmp_path)
    m = {'trials': [{'name': 'record-fixture', 'records': path.name, 'config': {'epochs': 1, 'hidden_dim': 16}}]}
    run_experiments(m, tmp_path / 'output', tmp_path)
    trial = json.loads((tmp_path / 'output' / 'record-fixture' / 'trial.json').read_text())
    saved = torch.load(tmp_path / 'output' / 'record-fixture' / 'last.pt', weights_only=False)
    assert trial['metadata'] == metadata_for(config)
    assert saved['config']['teacher_cost_config'] == {'type': 'fixture', 'alpha': 1., 'beta': 2.}


def test_run_training_resume_matches_uninterrupted_and_checks_data(tmp_path):
    config = TrainConfig(hidden_dim=16, candidate_count=16, epochs=1)
    splits = make_mock_splits(config)
    run_training(config, splits, tmp_path / 'resumed')
    checkpoint = tmp_path / 'resumed' / 'last.pt'
    changed = deepcopy(splits)
    changed['train'][0]['physical_cost'][0] += 1
    with pytest.raises(ValueError, match='targets differ'):
        run_training(replace(config, epochs=2), changed, tmp_path / 'resumed', checkpoint)
    resumed = run_training(replace(config, epochs=2), splits, tmp_path / 'resumed', checkpoint)
    uninterrupted = run_training(replace(config, epochs=2), splits, tmp_path / 'uninterrupted')
    assert resumed['validation'] == uninterrupted['validation']
    assert resumed['global_step'] == uninterrupted['global_step'] == 6
    a = torch.load(checkpoint, weights_only=False)
    b = torch.load(tmp_path / 'uninterrupted' / 'last.pt', weights_only=False)
    assert all(torch.equal(value, b['model'][key]) for key, value in a['model'].items())
