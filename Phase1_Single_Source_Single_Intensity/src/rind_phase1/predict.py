"""Canonical world-coordinate inference on a completed teacher support."""
import numpy as np
import torch
from .interfaces import training_record
from .train import collate_records
from .model import student_distribution


def predict(model, teacher, observation, chunk=4096, normalization_support=None):
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('chunk must be positive')
    record = training_record(teacher, observation)
    batch = collate_records([record])
    device = next(model.parameters()).device
    batch = {k:v.to(device) for k,v in batch.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoded = model.encode_observation(batch['response'],batch['obstacle'],batch['window'])
            energy = torch.cat([model.score_encoded(encoded,batch['candidate_xy'][:,i:i+chunk])
                                for i in range(0,len(record['candidate_xy']),chunk)],dim=1)
            policy = normalization_support or model.normalization_support
            if policy not in ('world','geometry'): raise ValueError('unknown normalization support')
            support = torch.ones_like(batch['valid']) if policy == 'world' else batch['valid']
            _, prob = student_distribution(energy.double(),support)
    finally:
        model.train(was_training)
    return {**{k:teacher[k] for k in ('scene_id','view_id','window','candidate_xy','valid','config_id')},
            'interface_version':model.interface_version,
            'normalization_support':policy,'support':support[0].cpu().numpy(),
            'energy':energy[0].cpu().numpy(), 'student_prob':prob[0].cpu().numpy()}
