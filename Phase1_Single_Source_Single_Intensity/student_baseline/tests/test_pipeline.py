import random
import numpy as np
import pytest
import torch
from data import collate_records, normalize_teacher_distribution, preprocess_record, validate_scene_splits
from mock_data import make_mock_records
from model import FourierXY, SourceEnergyField, student_distribution
from train import (TrainConfig, build_model, distribution_metrics, evaluate_source_grid,
                   load_checkpoint, save_checkpoint, seed_everything, teacher_student_loss, train_step, validate)

torch.set_num_threads(1)


@pytest.mark.parametrize('b,k,h,w', [(1, 1, 8, 8), (2, 7, 17, 23), (3, 11, 32, 28), (2, 64, 128, 128)])
def test_shapes(b, k, h, w):
    model = SourceEnergyField(hidden=16)
    xy = torch.randn(b, k, 2)
    output = model.predict(torch.rand(b, 1, h, w), xy)
    assert FourierXY()(xy).shape == (b, k, 26)
    assert output['energy'].shape == (b, k)
    assert torch.allclose(output['student_probability'].sum(-1), torch.ones(b))
    perm = torch.randperm(k)
    response = torch.rand(b, 1, h, w)
    assert torch.allclose(model(response, xy[:, perm]), model(response, xy)[:, perm], atol=1e-6)


def test_masked_loss_and_metrics():
    energy = torch.tensor([[1., 2., 999.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    q = torch.tensor([[0.25, 0.75, 0.]])
    log_p, p = student_distribution(energy, mask)
    assert p[0, 2] == 0
    loss = teacher_student_loss(energy, q, mask)
    assert torch.allclose(loss, teacher_student_loss(energy[:, :2], q[:, :2]))
    loss.backward()
    assert torch.isfinite(energy.grad).all() and energy.grad[0, 2] == 0
    metrics = distribution_metrics(energy, q, mask)
    assert all(torch.isfinite(v).all() for v in metrics.values())
    assert torch.allclose(metrics['cross_entropy'] - metrics['teacher_entropy'], metrics['forward_kl'])


def test_teacher_normalization():
    cost = torch.tensor([[10000., 10000., float('nan')]])
    mask = torch.tensor([[True, True, False]])
    q = normalize_teacher_distribution(cost, 0.1, torch.tensor([[1., 3., 0.]]), mask)
    assert torch.allclose(q, torch.tensor([[0.25, 0.75, 0.]]))
    assert torch.allclose(normalize_teacher_distribution(cost, 0.1, valid_mask=mask), torch.tensor([[0.5, 0.5, 0.]]))


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_bad_temperature(temperature):
    with pytest.raises(ValueError):
        normalize_teacher_distribution(torch.zeros(1, 3), temperature)


@pytest.mark.parametrize('area', [0., -1., float('nan'), float('inf')])
def test_bad_area(area):
    with pytest.raises(ValueError):
        normalize_teacher_distribution(torch.zeros(1, 1), 0.1, torch.tensor([[area]]))


def test_preprocessing_and_padding():
    a, b = make_mock_records(2, 9)
    a['coordinate_space'] = 'world'
    a['window'] = {'x0': 10., 'y0': -4., 'L': 2.}
    expected = a['candidate_xy'].clone()
    a['candidate_xy'] = expected * 2 + torch.tensor([10., -4.])
    assert torch.allclose(preprocess_record(a)['candidate_xy'], expected)
    for key in ('candidate_xy', 'physical_cost', 'teacher_prob', 'area_weight'):
        b[key] = b[key][:4]
    b.pop('teacher_prob')
    batch = collate_records([a, b])
    assert batch['candidate_xy'].shape == (2, 9, 2)
    assert torch.equal(batch['candidate_xy'][0], expected)
    assert not batch['valid_mask'][1, 4:].any()
    assert batch['teacher_prob'][1, 4:].sum() == 0
    assert batch['scene_id'] == [a['scene_id'], b['scene_id']]
    # Supplied q is used as-is even when areas differ.
    a['area_weight'] = torch.arange(1., 10.)
    assert torch.equal(collate_records([a])['teacher_prob'][0], a['teacher_prob'])


@pytest.mark.parametrize('mutation', ['all_invalid', 'bad_prob', 'negative_prob', 'missing_window', 'bad_response', 'missing_target', 'bad_shape', 'masked_mass'])
def test_invalid_record(mutation):
    r = make_mock_records(1, 4)[0]
    if mutation == 'all_invalid': r['valid_mask'] = torch.zeros(4, dtype=torch.bool)
    elif mutation == 'bad_prob': r['teacher_prob'] *= 2
    elif mutation == 'negative_prob': r['teacher_prob'][0] = -1
    elif mutation == 'missing_window': r.pop('coordinate_space')
    elif mutation == 'bad_response': r['response'][0, 0, 0] = float('nan')
    elif mutation == 'missing_target': r.pop('teacher_prob'); r.pop('physical_cost')
    elif mutation == 'bad_shape': r['physical_cost'] = torch.zeros(3)
    elif mutation == 'masked_mass': r['valid_mask'] = torch.tensor([False, True, True, True])
    with pytest.raises(ValueError): preprocess_record(r)


def test_scene_splits():
    records = make_mock_records(2)
    validate_scene_splits({'train': records[:1], 'validation': records[1:]})
    with pytest.raises(ValueError): validate_scene_splits({'train': records, 'validation': records[:1]})


def test_all_invalid_distribution():
    with pytest.raises(ValueError): student_distribution(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError): normalize_teacher_distribution(torch.zeros(1, 2), 0.1, valid_mask=torch.zeros(1, 2, dtype=torch.bool))


def test_dense_chunk_equivalence():
    model = SourceEnergyField(hidden=16)
    response, xy = torch.rand(1, 1, 16, 16), torch.rand(29, 2)
    a = evaluate_source_grid(model, response, xy, 7)
    b = evaluate_source_grid(model, response, xy, 29)
    assert model.training
    assert torch.allclose(a, b, atol=1e-6)
    assert student_distribution(a[None])[1].sum().item() == pytest.approx(1.)
    assert evaluate_source_grid(model, response, xy[:0]).shape == (0,)


def test_checkpoint_exact_resume(tmp_path):
    seed_everything(42)
    config = TrainConfig(hidden_dim=16)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    batch = collate_records(make_mock_records(3, 16))
    for _ in range(3): train_step(model, batch, optimizer)
    scheduler.step()
    before = validate(model, [batch])
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, scheduler, 1, 3, config, before['cross_entropy'])
    draws = (random.random(), np.random.rand(), torch.rand(1))
    train_step(model, batch, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    fresh = build_model(config)
    opt = torch.optim.AdamW(fresh.parameters(), lr=config.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    state = load_checkpoint(path, fresh, opt, sched)
    assert state['global_step'] == 3 and state['epoch'] == 1
    assert sched.state_dict() == scheduler.state_dict()
    assert validate(fresh, [batch]) == before
    assert random.random() == draws[0] and np.random.rand() == draws[1]
    assert torch.equal(torch.rand(1), draws[2])
    train_step(fresh, batch, opt)
    assert all(torch.equal(v, expected[k]) for k, v in fresh.state_dict().items())


def test_random_mock_and_cost_only_step():
    records = make_mock_records(2, 13, random_targets=True)
    assert torch.equal(records[0]['response'], make_mock_records(2, 13, random_targets=True)[0]['response'])
    batch = collate_records(records)
    batch.pop('teacher_prob')
    model = SourceEnergyField(hidden=16)
    optimizer = torch.optim.AdamW(model.parameters())
    assert np.isfinite(train_step(model, batch, optimizer)['loss'])


def test_full_record_split_validation():
    from data import validate_record_splits
    splits = {name: make_mock_records(1, prefix=name) for name in ('train', 'validation', 'test')}
    validate_record_splits(splits)
    splits['test'][0]['response'] = torch.zeros(1, 12, 12)
    with pytest.raises(ValueError, match='fixed response size'): validate_record_splits(splits)
    with pytest.raises(ValueError): validate_record_splits({'train': []})


def test_diagnostics_scatter_and_absent_cost(tmp_path):
    from diagnostics import save_diagnostics
    records = make_mock_records(1, 13)
    records[0].pop('physical_cost')
    paths = save_diagnostics(SourceEnergyField(hidden=16), records, tmp_path)
    assert len(paths) == 1 and (tmp_path / 'record-0.png').stat().st_size > 1000
    raw = torch.load(tmp_path / 'record-0.pt', weights_only=True)
    assert not raw['physical_cost_available']
    assert raw['student_probability'].shape == (13,)


def test_mock_overfit_acceptance(tmp_path):
    from verify_pipeline import run_verification
    report = run_verification(tmp_path, max_steps=3000)
    assert report['records'] == 12 and report['overfit_passed']
    assert set(report['per_family']) == {'one_peak', 'two_peaks', 'ridge'}
    records = make_mock_records(12)
    assert len({r['response'].numpy().tobytes() for r in records}) == 12
    assert len({r['teacher_prob'].numpy().tobytes() for r in records}) == 12


def test_cli_train_resume(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'epochs': 1, 'hidden_dim': 16, 'candidate_count': 16}))
    output = tmp_path / 'run'
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / 'train.py'), '--config', str(config), '--output', str(output)]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    first = torch.load(output / 'last.pt', weights_only=False)
    assert first['epoch'] == 1 and first['global_step'] == 3
    config.write_text(json.dumps({'epochs': 2, 'hidden_dim': 16, 'candidate_count': 16}))
    subprocess.run(command + ['--resume', str(output / 'last.pt')], cwd=root, check=True, capture_output=True, text=True)
    second = torch.load(output / 'last.pt', weights_only=False)
    assert second['epoch'] == 2 and second['global_step'] == 6
    logs = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in logs] == [1, 2]
    assert 'forward_kl' in logs[-1]['validation']
