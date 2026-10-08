"""Reproducible acceptance experiment: bounded mock overfit plus saved diagnostics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from data import collate_records
from diagnostics import save_diagnostics, save_dense_grid
from mock_data import make_mock_records
from train import (TrainConfig, build_model, evaluate_source_grid, load_checkpoint, make_loader,
                   save_checkpoint, seed_everything, train_step, validate)


def run_verification(output, max_steps=3000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = TrainConfig()
    torch.set_num_threads(config.num_threads)
    seed_everything(config.random_seed)
    records = make_mock_records(12, config.candidate_count, config.image_size, config.random_seed)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = make_loader(records, config)
    initial = validate(model, loader)
    history = []
    for step in range(1, max_steps + 1):
        # Cycle through all twelve fixed records in four-record minibatches.
        start = ((step - 1) * config.batch_size) % len(records)
        batch = collate_records(records[start:start + config.batch_size])
        loss = train_step(model, batch, optimizer)['loss']
        if step % 25 == 0 or step == max_steps:
            metrics = validate(model, loader)
            history.append({'step': step, 'loss': loss, **metrics})
            print(json.dumps(history[-1]), flush=True)
            if metrics['forward_kl'] < 0.02 and metrics['l1_distance'] < 0.1:
                break
    final = validate(model, loader)
    families = {family: validate(model, make_loader([r for r in records if r['family'] == family], config))
                for family in ('one_peak', 'two_peaks', 'ridge')}
    checkpoint_path = output / 'mock-student.pt'
    save_checkpoint(checkpoint_path, model, optimizer, epoch=step // 3, global_step=step, config=config,
                    best_val_loss=final['cross_entropy'])
    fresh = build_model(config)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    load_checkpoint(checkpoint_path, fresh, fresh_optimizer)
    reloaded = validate(fresh, loader)
    round_trip = final == reloaded
    batch = collate_records(records[:4])
    train_step(model, batch, optimizer)
    train_step(fresh, batch, fresh_optimizer)
    resume_equal = all(torch.equal(v, fresh.state_dict()[k]) for k, v in model.state_dict().items())
    # Restore the accepted checkpoint before writing figures.
    load_checkpoint(checkpoint_path, model, optimizer)
    plots = save_diagnostics(model, records[:3], output / 'diagnostics')
    dense_plots = [str(save_dense_grid(model, r, output / 'dense' / r['family'], chunk=257)) for r in records[:3]]
    dense = torch.load(output / 'dense' / 'ridge' / 'dense.pt', weights_only=True)
    response = records[2]['response'][None]
    unchunked = evaluate_source_grid(model, response, dense['candidate_xy'], chunk=len(dense['candidate_xy']))
    dense_error = (dense['energy'] - unchunked).abs().max().item()
    report = {'config': asdict(config), 'records': 12, 'steps': step, 'initial': initial, 'final': final,
              'per_family': families, 'overfit_passed': final['forward_kl'] < 0.02 and final['l1_distance'] < 0.1,
              'checkpoint_round_trip_exact': round_trip, 'continued_training_exact': resume_equal,
              'dense_max_chunk_error': dense_error, 'dense_probability_sum': dense['student_probability'].sum().item(),
              'diagnostic_plots': plots, 'dense_plots': dense_plots,
              'real_teacher_integration': 'pending upstream records and area convention'}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'history.json').write_text(json.dumps(history, indent=2))
    assert report['overfit_passed'], 'Mock overfit gate failed; do not start full training'
    assert round_trip and resume_equal
    assert dense_error < 1e-5
    assert abs(report['dense_probability_sum'] - 1) < 1e-5
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/verification'))
    parser.add_argument('--max-steps', type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(run_verification(args.output, args.max_steps), indent=2))
