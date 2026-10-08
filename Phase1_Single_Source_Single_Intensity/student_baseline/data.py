"""Teacher record validation and batching; no geometry dependencies."""
import math
import torch


def validate_teacher_probability(prob, mask):
    if prob.shape != mask.shape or not torch.isfinite(prob).all() or (prob < 0).any():
        raise ValueError('teacher_prob must be finite, nonnegative and match candidates')
    if (prob[~mask] != 0).any():
        raise ValueError('invalid candidates must have zero teacher probability')
    if not torch.allclose(prob.sum(-1), torch.ones_like(prob.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError('teacher_prob must sum to one')
    return prob


def normalize_teacher_distribution(physical_cost, temperature, area_weight=None, valid_mask=None, eps=1e-12):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if physical_cost.ndim != 2 or physical_cost.shape[-1] == 0:
        raise ValueError('physical_cost must be [B,K]')
    mask = torch.ones_like(physical_cost, dtype=torch.bool) if valid_mask is None else valid_mask
    if mask.shape != physical_cost.shape or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('invalid candidate mask')
    if not torch.isfinite(physical_cost[mask]).all():
        raise ValueError('valid physical costs must be finite')
    # Subtract the valid minimum before division to improve numerical stability.
    cost = physical_cost.masked_fill(~mask, torch.inf)
    log_w = -(cost - cost.amin(-1, keepdim=True)) / temperature
    if area_weight is not None:
        if area_weight.shape != cost.shape or not torch.isfinite(area_weight[mask]).all() or (area_weight[mask] <= 0).any():
            raise ValueError('valid area weights must be finite and positive')
        log_w = log_w + area_weight.masked_fill(~mask, 1).log()
    return log_w.masked_fill(~mask, -torch.inf).softmax(-1)


def preprocess_record(record):
    r = dict(record)
    for key in ('scene_id', 'view_id'):
        if key not in r or not isinstance(r[key], (str, int)):
            raise ValueError(f'{key} must be a string or integer')
    response = torch.as_tensor(r['response'], dtype=torch.float32)
    xy = torch.as_tensor(r['candidate_xy'], dtype=torch.float32)
    if response.ndim != 3 or response.shape[0] != 1 or min(response.shape) < 1 or not torch.isfinite(response).all():
        raise ValueError('response must be finite [1,H,W]')
    if xy.ndim != 2 or xy.shape[-1] != 2 or not len(xy) or not torch.isfinite(xy).all():
        raise ValueError('candidate_xy must be finite [K,2]')
    space = r.get('coordinate_space', 'world')
    if space == 'world':
        window = r.get('window', {})
        if not all(k in window and math.isfinite(window[k]) for k in ('x0', 'y0', 'L')) or window['L'] <= 0:
            raise ValueError('world coordinates require finite window={x0,y0,L}, L>0')
        xy = (xy - xy.new_tensor([window['x0'], window['y0']])) / window['L']
    elif space != 'normalized':
        raise ValueError('coordinate_space must be world or normalized')
    mask = torch.as_tensor(r.get('valid_mask', torch.ones(len(xy), dtype=torch.bool)))
    if mask.dtype != torch.bool or mask.shape != (len(xy),) or not mask.any():
        raise ValueError('valid_mask must be bool [K] and nonempty')
    r.update(response=response, candidate_xy=xy, coordinate_space='normalized', valid_mask=mask)
    for key in ('teacher_prob', 'physical_cost', 'area_weight'):
        if key in r:
            value = torch.as_tensor(r[key], dtype=torch.float32)
            if value.shape != (len(xy),) or not torch.isfinite(value[mask]).all():
                raise ValueError(f'{key} must be finite on valid candidates and have shape [K]')
            if key == 'area_weight' and (value[mask] <= 0).any():
                raise ValueError('valid area weights must be positive')
            r[key] = value
    if 'teacher_prob' in r:
        validate_teacher_probability(r['teacher_prob'], mask)
    elif 'physical_cost' not in r:
        raise ValueError('teacher_prob or physical_cost is required')
    return r


def collate_records(records, teacher_temperature=0.1):
    records = [preprocess_record(r) for r in records]
    if not records or len({tuple(r['response'].shape) for r in records}) != 1:
        raise ValueError('batch needs records with identical response shapes')
    b, k = len(records), max(len(r['candidate_xy']) for r in records)
    batch = {'response': torch.stack([r['response'] for r in records]),
             'candidate_xy': torch.zeros(b, k, 2), 'valid_mask': torch.zeros(b, k, dtype=torch.bool),
             'teacher_prob': torch.zeros(b, k), 'physical_cost': torch.full((b, k), torch.nan),
             'area_weight': torch.ones(b, k), 'scene_id': [], 'view_id': []}
    for i, r in enumerate(records):
        n = len(r['candidate_xy'])
        for key in ('candidate_xy', 'valid_mask', 'physical_cost', 'area_weight'):
            if key in r:
                batch[key][i, :n] = r[key]
        q = r.get('teacher_prob')
        if q is None:
            q = normalize_teacher_distribution(r['physical_cost'][None], teacher_temperature,
                r['area_weight'][None] if 'area_weight' in r else None, r['valid_mask'][None])[0]
        batch['teacher_prob'][i, :n] = q
        for key in ('scene_id', 'view_id'):
            batch[key].append(r[key])
    return batch


def validate_scene_splits(splits):
    seen = set()
    for name, records in splits.items():
        ids = {r['scene_id'] for r in records}
        if seen & ids:
            raise ValueError(f'scene leakage in {name}: {seen & ids}')
        seen |= ids


def validate_record_splits(splits):
    """Validate the full input before training, including fixed-size observations."""
    if not isinstance(splits, dict) or not {'train', 'validation', 'test'} <= splits.keys():
        raise ValueError('records must contain train, validation, and test lists')
    normalized = {}
    shapes = set()
    for name, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f'{name} must be a nonempty record list')
        normalized[name] = [preprocess_record(record) for record in records]
        shapes.update(tuple(r['response'].shape) for r in normalized[name])
    if len(shapes) != 1:
        raise ValueError('all records must use a fixed response size')
    validate_scene_splits(normalized)
    return normalized
