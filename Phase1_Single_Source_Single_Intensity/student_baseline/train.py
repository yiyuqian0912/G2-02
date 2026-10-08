"""Training and evaluation utilities for the Phase I student baseline."""
import argparse
from dataclasses import asdict, dataclass, field
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import collate_records, normalize_teacher_distribution, validate_record_splits, validate_teacher_probability
from model import SourceEnergyField, student_distribution


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    fourier_frequencies: int = 6
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    teacher_temperature: float = 0.1
    candidate_count: int = 64
    checkpoint_interval: int = 1
    validation_interval: int = 1
    random_seed: int = 0
    epochs: int = 10
    device: str = 'cpu'
    num_threads: int = 1
    image_size: int = 16
    dataset_version: str = 'structured-mock-v1'
    scene_split_version: str = 'mock-disjoint-v1'
    candidate_search_config: dict = field(default_factory=lambda: {'type': 'uniform-mock-grid'})
    teacher_cost_config: dict = field(default_factory=lambda: {'type': 'synthetic-log-density'})
    area_weight_convention: str = 'equal-area mock candidates; upstream convention pending'

    def __post_init__(self):
        for name in ('hidden_dim', 'fourier_frequencies', 'batch_size', 'candidate_count',
                     'checkpoint_interval', 'validation_interval', 'random_seed', 'epochs',
                     'num_threads', 'image_size'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.hidden_dim < 2 or self.fourier_frequencies < 0 or self.image_size < 4:
            raise ValueError('hidden_dim >= 2, fourier_frequencies >= 0, image_size >= 4 required')
        for name in ('batch_size', 'candidate_count', 'checkpoint_interval', 'validation_interval', 'epochs', 'num_threads'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('learning_rate', 'teacher_temperature'):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.weight_decay) not in (int, float) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config=None):
    config = TrainConfig() if config is None else config
    if isinstance(config, dict):
        config = TrainConfig(**config)
    return SourceEnergyField(config.hidden_dim, config.fourier_frequencies).to(config.device)


def teacher_student_loss(energy, teacher_prob, valid_mask=None):
    log_p, _ = student_distribution(energy, valid_mask)
    mask = torch.ones_like(energy, dtype=torch.bool) if valid_mask is None else valid_mask
    validate_teacher_probability(teacher_prob, mask)
    safe_log_p = log_p.masked_fill(~mask, 0)
    return -(teacher_prob.detach() * safe_log_p).sum(-1).mean()


def distribution_metrics(energy, teacher_prob, valid_mask=None):
    log_p, p = student_distribution(energy, valid_mask)
    mask = torch.ones_like(energy, dtype=torch.bool) if valid_mask is None else valid_mask
    q = validate_teacher_probability(teacher_prob, mask)
    safe_log_p = log_p.masked_fill(~mask, 0)
    ce = -(q * safe_log_p).sum(-1)
    hq = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(-1)
    hp = -(p * safe_log_p).sum(-1)
    return {'cross_entropy': ce, 'forward_kl': ce - hq, 'l1_distance': (q - p).abs().sum(-1),
            'student_entropy': hp, 'teacher_entropy': hq, 'entropy_mismatch': (hp - hq).abs()}


def _prepare_batch(model, batch, temperature):
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    q = batch.get('teacher_prob')
    if q is None:
        q = normalize_teacher_distribution(batch['physical_cost'], temperature,
                                           batch.get('area_weight'), batch.get('valid_mask'))
    return batch, q


def train_step(model, batch, optimizer, teacher_temperature=0.1):
    model.train()
    batch, q = _prepare_batch(model, batch, teacher_temperature)
    energy = model(batch['response'], batch['candidate_xy'])
    loss = teacher_student_loss(energy, q, batch.get('valid_mask'))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.detach())}


def train_epoch(model, loader, optimizer, teacher_temperature=0.1):
    total, count, steps = 0.0, 0, 0
    for batch in loader:
        n = len(batch['response'])
        total += train_step(model, batch, optimizer, teacher_temperature)['loss'] * n
        count += n
        steps += 1
    if not count:
        raise ValueError('empty training loader')
    return {'loss': total / count, 'steps': steps}


@torch.no_grad()
def validate(model, loader, teacher_temperature=0.1):
    was_training = model.training
    model.eval()
    totals, count = {}, 0
    try:
        for batch in loader:
            batch, q = _prepare_batch(model, batch, teacher_temperature)
            energy = model(batch['response'], batch['candidate_xy'])
            for name, values in distribution_metrics(energy, q, batch.get('valid_mask')).items():
                totals[name] = totals.get(name, 0.0) + values.sum().item()
            count += len(energy)
        if not count:
            raise ValueError('empty validation loader')
        return {name: value / count for name, value in totals.items()}
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_source_grid(model, response, xy_grid, chunk=4096):
    if response.ndim != 4 or response.shape[:2] != (1, 1):
        raise ValueError('response must be [1,1,H,W]')
    if xy_grid.ndim != 2 or xy_grid.shape[-1] != 2 or chunk < 1:
        raise ValueError('xy_grid must be [N,2] and chunk positive')
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        all_energy = [model(response.to(device), xy_grid[start:start + chunk].to(device)[None])[0].cpu()
                      for start in range(0, len(xy_grid), chunk)]
        return torch.cat(all_energy) if all_energy else torch.empty(0)
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, scheduler=None, epoch=0, global_step=0, config=None, best_val_loss=math.inf, data_fingerprints=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {'torch': torch.get_rng_state(), 'python': random.getstate(),
           'numpy': np.random.get_state()}
    if torch.cuda.is_available():
        rng['cuda'] = torch.cuda.get_rng_state_all()
    if next(model.parameters()).device.type == 'mps':
        rng['mps'] = torch.mps.get_rng_state()
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict() if scheduler is not None else None,
                  'epoch': epoch, 'global_step': global_step,
                  'config': asdict(config) if isinstance(config, TrainConfig) else config,
                  'best_val_loss': best_val_loss, 'rng_state': rng, 'data_fingerprints': data_fingerprints}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_rng=True):
    # Checkpoints contain Python/NumPy RNG state: only load locally trusted files.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if restore_rng:
        rng = checkpoint['rng_state']
        torch.set_rng_state(rng['torch'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        if 'cuda' in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
        if 'mps' in rng and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng['mps'])
    return checkpoint


def make_loader(records, config, shuffle=False):
    return DataLoader(records, batch_size=config.batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_records, teacher_temperature=config.teacher_temperature))


def _check_resume_config(config, checkpoint):
    previous = checkpoint['config']
    allowed = {'epochs', 'device', 'num_threads', 'checkpoint_interval', 'validation_interval'}
    changed = [key for key in previous if key not in allowed and previous[key] != asdict(config)[key]]
    if changed:
        raise ValueError(f'resume configuration differs: {changed}')
    if config.epochs <= checkpoint['epoch']:
        raise ValueError('epochs must exceed the saved epoch when resuming')


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def run_training(config, splits, output_dir, resume=None):
    """Train on supplied records, returning final validation metrics and provenance.

    Epochs is the total desired epoch count. Rankings use the last checkpoint;
    best.pt remains available for separate validation-selected evaluation.
    """
    from diagnostics import save_diagnostics
    from run_data import split_fingerprints
    config = TrainConfig(**config) if isinstance(config, dict) else config
    splits = validate_record_splits(splits)
    fingerprints = split_fingerprints(splits, config.teacher_temperature)
    output_dir = Path(output_dir)
    previous = None
    if resume is not None:
        previous = torch.load(resume, map_location='cpu', weights_only=False)
        _check_resume_config(config, previous)
        if previous.get('data_fingerprints') is not None and previous['data_fingerprints'] != fingerprints:
            raise ValueError('resume records or teacher targets differ from the saved run')
        # Keep the run history and previous best model together.
        if Path(resume).resolve().parent != output_dir.resolve():
            raise ValueError('resume output_dir must be the checkpoint directory')
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('output_dir must be empty for a new run; use resume for an existing run')
    torch.set_num_threads(config.num_threads)
    seed_everything(config.random_seed)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start, global_step, best = 0, 0, math.inf
    if previous is not None:
        checkpoint = load_checkpoint(resume, model, optimizer)
        start, global_step, best = checkpoint['epoch'], checkpoint['global_step'], checkpoint['best_val_loss']
        # Resuming an older checkpoint must not append duplicate future epochs.
        log_path = output_dir / 'metrics.jsonl'
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if rows and rows[-1]['epoch'] > start:
                raise ValueError('resume from the latest checkpoint to preserve run history')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    loaders = {name: make_loader(records, config, name == 'train') for name, records in splits.items()}
    device = next(model.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    for epoch in range(start + 1, config.epochs + 1):
        metrics = {'epoch': epoch, 'train': train_epoch(model, loaders['train'], optimizer, config.teacher_temperature)}
        global_step += metrics['train']['steps']
        checkpoint_args = dict(epoch=epoch, global_step=global_step, config=config, data_fingerprints=fingerprints)
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            metrics['validation'] = validate(model, loaders['validation'], config.teacher_temperature)
            val_loss = metrics['validation']['cross_entropy']
            if val_loss < best:
                best = val_loss
                save_checkpoint(output_dir / 'best.pt', model, optimizer, best_val_loss=best, **checkpoint_args)
            save_diagnostics(model, splits['validation'][:3], output_dir / f'diagnostics-{epoch:04}', config.teacher_temperature)
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            save_checkpoint(output_dir / 'last.pt', model, optimizer, best_val_loss=best, **checkpoint_args)
        with (output_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    _synchronize(device)
    summary = {'validation': metrics['validation'], 'evaluation_checkpoint': 'last.pt',
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'runtime_seconds': time.perf_counter() - started,
               'start_epoch': start, 'epoch': config.epochs, 'global_step': global_step,
               'best_val_loss': best, 'data_fingerprints': fingerprints,
               'config': asdict(config)}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    from run_data import load_record_file, make_mock_splits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--records', type=Path, help='Trusted torch file with train/validation/test record lists')
    parser.add_argument('--output', type=Path, default=Path('outputs/train'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.config:
        config = TrainConfig(**json.loads(args.config.read_text()))
    elif args.resume:
        config = TrainConfig(**torch.load(args.resume, map_location='cpu', weights_only=False)['config'])
    else:
        config = TrainConfig()
    if args.records:
        splits, metadata = load_record_file(args.records)
        defaults = TrainConfig()
        keys = ('dataset_version', 'scene_split_version', 'candidate_search_config',
                'teacher_cost_config', 'area_weight_convention')
        if any(not getattr(config, key) or getattr(config, key) == getattr(defaults, key) for key in keys):
            raise ValueError('real records require explicit dataset and upstream metadata in --config')
        if any(key in metadata and metadata[key] != getattr(config, key) for key in (*keys, 'candidate_count')):
            raise ValueError('configuration differs from upstream record metadata')
        if config.teacher_temperature != defaults.teacher_temperature and any(
                'teacher_prob' in record for records in splits.values() for record in records):
            raise ValueError('changing teacher_temperature requires cost-only records; supplied teacher_prob is authoritative')
    else:
        splits = make_mock_splits(config)
    run_training(config, splits, args.output, args.resume)


if __name__ == '__main__':
    main()
