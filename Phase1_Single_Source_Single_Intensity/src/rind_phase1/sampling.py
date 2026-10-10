"""Shared dense scoring and sampling-frequency checks; independent of plotting."""
import numpy as np
import torch
from .interfaces import validate_observation


def sampling_report(world_size, spacing, frequencies):
    if not np.isfinite(spacing) or spacing <= 0 or type(frequencies) is not int or frequencies < 0:
        raise ValueError('positive spacing and nonnegative integer frequencies required')
    period=2*world_size/(2**(frequencies-1)) if frequencies else None
    return dict(shortest_period=period,spacing=float(spacing),
                samples_per_shortest_period=period/spacing if period else None,
                resolved=period is None or period/spacing >= 4)


def require_resolved_sampling(world_size,spacing,frequencies):
    report=sampling_report(world_size,spacing,frequencies)
    if not report['resolved']:
        raise ValueError(f'Fourier/sampling mismatch: shortest period {report["shortest_period"]:g}, spacing {spacing:g}; require >=4 samples per period. Reduce fourier_frequencies or spacing.')
    return report


def grid_coordinates(world_size,spacing):
    if type(world_size) is not int or world_size < 1 or type(spacing) is not int or spacing < 1 or world_size % spacing:
        raise ValueError('integer spacing must divide world_size')
    axis=np.arange(spacing/2,world_size,spacing,dtype=np.float64)
    xx,yy=np.meshgrid(axis,axis)
    return np.column_stack((xx.ravel(),yy.ravel())),len(axis)


def score_grid(model, observation, world_size, spacing=1, chunk=1024):
    validate_observation(observation)
    if type(chunk) is not int or chunk < 1: raise ValueError('positive chunk required')
    xy,size=grid_coordinates(world_size,spacing)
    device=next(model.parameters()).device
    response=torch.as_tensor(observation['response'][None],dtype=torch.float32,device=device)
    obstacle=torch.as_tensor(observation['obstacle'][None],dtype=torch.bool,device=device)
    window=torch.as_tensor(observation['window'][None],dtype=torch.float32,device=device)
    result=np.empty(len(xy),np.float32)
    was_training=model.training
    model.eval()
    try:
        with torch.inference_mode():
            encoded=model.encode_observation(response,obstacle,window)
            for start in range(0,len(xy),chunk):
                candidate=torch.as_tensor(xy[None,start:start+chunk],dtype=torch.float32,device=device)
                result[start:start+candidate.shape[1]]=model.score_encoded(encoded,candidate)[0].cpu().numpy()
    finally:
        model.train(was_training)
    if not np.isfinite(result).all(): raise ValueError('nonfinite energy')
    return result.reshape(size,size),xy


def probability_from_energy(energy,support):
    energy,support=np.asarray(energy),np.asarray(support)
    if support.dtype != bool or support.shape != energy.shape or not support.any() or not np.isfinite(energy[support]).all():
        raise ValueError('finite energies and nonempty aligned boolean support required')
    weights=np.exp(-(energy[support].astype(np.float64)-float(energy[support].min())))
    probability=np.zeros(energy.shape,np.float64)
    probability[support]=weights/weights.sum()
    return probability
