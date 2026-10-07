"""Explicit format bridge to Thomas's unchanged student baseline."""
import importlib.util
from pathlib import Path
import numpy as np
from . import checks


def to_thomas_record(teacher, observation):
    checks.teacher(teacher)
    response = checks.observation(teacher, observation)
    x, y, size = map(int, teacher['window'])
    return {'scene_id': teacher['scene_id'], 'view_id': teacher['view_id'],
            'response': response[None].astype(np.float32),
            'candidate_xy': np.asarray(teacher['candidate_xy']).copy(),
            'window': {'x0': x, 'y0': y, 'L': size}, 'coordinate_space': 'world',
            'valid_mask': np.asarray(teacher['valid']).copy(),
            'teacher_prob': np.asarray(teacher['teacher_prob']).copy(),
            'physical_cost': np.asarray(teacher['physical_cost']).copy()}


def predict_thomas(model, teacher, observation, *, chunk=4096):
    import torch
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('chunk must be a positive integer')
    response = checks.observation(teacher, observation)
    checks.teacher(teacher)
    x, y, size = teacher['window']
    # Preserve D's training normalization, rather than substituting world/1024.
    xy = (np.asarray(teacher['candidate_xy']) - [x, y]) / size
    device = next(model.parameters()).device
    image = torch.as_tensor(response[None, None], dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            energy = np.concatenate([model(image, torch.as_tensor(xy[i:i+chunk][None],
                dtype=torch.float32, device=device))[0].cpu().numpy()
                for i in range(0, len(xy), chunk)])
    finally:
        model.train(was_training)
    valid = teacher['valid']
    weights = np.exp(-(energy[valid].astype(float) - energy[valid].min()))
    prob = np.zeros(len(xy))
    prob[valid] = weights / weights.sum()
    return {**{k: teacher[k] for k in ('scene_id', 'view_id', 'candidate_xy', 'valid', 'window', 'config_id')},
            'energy': energy, 'student_prob': prob}


def load_thomas_model(baseline_dir, checkpoint_path, *, trusted_checkpoint=False):
    """Safe tensor-only loading by default; D full checkpoints require explicit trust."""
    import torch
    path = Path(baseline_dir) / 'model.py'
    spec = importlib.util.spec_from_file_location('part_e_thomas_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=not trusted_checkpoint)
    config = checkpoint.get('config', {})
    model = module.SourceEnergyField(config.get('hidden_dim', 128), config.get('fourier_frequencies', 6))
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model.eval()
    return model, checkpoint
