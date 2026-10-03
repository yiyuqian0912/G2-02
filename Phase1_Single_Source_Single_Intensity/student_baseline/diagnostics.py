"""Persist raw diagnostics and render candidate/grid comparisons."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import collate_records
from model import student_distribution
from train import distribution_metrics, evaluate_source_grid


def _plot_candidates(ax, xy, values, title):
    xy, values = xy.numpy(), values.numpy()
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if len(xs) * len(ys) == len(xy) and len(np.unique(xy, axis=0)) == len(xy):
        grid = np.full((len(ys), len(xs)), np.nan)
        grid[np.searchsorted(ys, xy[:, 1]), np.searchsorted(xs, xy[:, 0])] = values
        artist = ax.pcolormesh(xs, ys, grid, shading='nearest')
    else:
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=18)
    ax.set(title=title, xlabel='normalized x', ylabel='normalized y', aspect='equal')
    plt.colorbar(artist, ax=ax, shrink=0.75)


@torch.no_grad()
def save_diagnostics(model, records, directory, temperature=0.1):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, record in enumerate(records):
        batch = collate_records([record], temperature)
        energy = evaluate_source_grid(model, batch['response'], batch['candidate_xy'][0])
        _, probability = student_distribution(energy[None], batch['valid_mask'])
        metrics = {k: v.item() for k, v in distribution_metrics(energy[None], batch['teacher_prob'], batch['valid_mask']).items()}
        raw = {**batch, 'energy': energy, 'student_probability': probability[0], 'metrics': metrics,
               'physical_cost_available': 'physical_cost' in record}
        torch.save(raw, directory / f'record-{i}.pt')
        mask = batch['valid_mask'][0]
        xy = batch['candidate_xy'][0, mask]
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), constrained_layout=True)
        axes[0].imshow(batch['response'][0, 0].numpy(), origin='lower', cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Response')
        for ax, values, title in zip(axes[1:], [batch['physical_cost'][0], batch['teacher_prob'][0], energy, probability[0]],
                                     ['Physical cost', 'Teacher probability', 'Student energy', 'Student probability']):
            if title == 'Physical cost' and 'physical_cost' not in record:
                ax.text(0.5, 0.5, 'Not supplied', ha='center', va='center')
                ax.set_title(title)
            else:
                _plot_candidates(ax, xy, values[mask], title)
        fig.suptitle(f"{record.get('family', 'teacher')} | H(q)={metrics['teacher_entropy']:.3f}, H(p)={metrics['student_entropy']:.3f}, KL={metrics['forward_kl']:.4f}")
        path = directory / f'record-{i}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_dense_grid(model, record, directory, side=64, chunk=4096):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, x = torch.meshgrid(torch.linspace(-1, 2, side), torch.linspace(-1, 2, side), indexing='ij')
    xy = torch.stack((x.flatten(), y.flatten()), -1)
    batch = collate_records([record])
    energy = evaluate_source_grid(model, batch['response'], xy, chunk)
    # Normalize once over the complete grid, never independently per chunk.
    _, probability = student_distribution(energy[None])
    torch.save({'candidate_xy': xy, 'energy': energy, 'student_probability': probability[0]}, directory / 'dense.pt')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    _plot_candidates(axes[0], xy, energy, 'Dense student energy')
    _plot_candidates(axes[1], xy, probability[0], 'Dense student probability')
    fig.savefig(directory / 'dense.png', dpi=140)
    plt.close(fig)
    return directory / 'dense.png'
