"""Canonical NumPy handoff. No Torch, renderer, or hidden geometry dependency."""
import numpy as np

INTERFACE_VERSION = 'phase1-v1'


def validate_observation(sample):
    for key in ('scene_id', 'view_id', 'response', 'obstacle', 'window'):
        if key not in sample:
            raise ValueError(f'missing observation field: {key}')
    window = np.asarray(sample['window'])
    if window.shape != (3,) or window.dtype.kind not in 'iu' or window[2] <= 0 or (window[:2] < 0).any():
        raise ValueError('window must be integer [x,y,size] with positive size')
    response, obstacle = np.asarray(sample['response']), np.asarray(sample['obstacle'])
    if response.shape != (window[2], window[2]) or not np.isfinite(response).all():
        raise ValueError('response must be finite native [size,size]')
    if obstacle.dtype != np.dtype(bool) or obstacle.shape != response.shape:
        raise ValueError('obstacle must be boolean and match response')
    if np.any(response[obstacle] != 0):
        raise ValueError('obstacle pixels must have zero response')
    return sample


def training_record(teacher, observation):
    """Join by IDs/window; preserve world support and authoritative teacher targets."""
    validate_observation(observation)
    for key in ('scene_id', 'view_id', 'window'):
        if not np.array_equal(teacher[key], observation[key]):
            raise ValueError(f'teacher/observation mismatch: {key}')
    xy = np.asarray(teacher['candidate_xy'])
    valid = np.asarray(teacher['valid'])
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('candidate_xy must be finite [N,2]')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('duplicate candidates')
    if valid.dtype != np.dtype(bool) or valid.shape != (len(xy),) or not valid.any():
        raise ValueError('valid must be boolean [N] with nonempty support')
    evaluated = np.asarray(teacher['evaluated'])
    if evaluated.dtype != np.dtype(bool) or evaluated.shape != valid.shape or not evaluated.all() or teacher['metadata'].get('complete') is not True:
        raise ValueError('teacher support must be completely evaluated')
    from .teacher import teacher_probabilities
    expected = teacher_probabilities(teacher['physical_cost'], valid, temperature=float(teacher['temperature']))
    q = np.asarray(teacher['teacher_prob'])
    if q.shape != valid.shape or not np.isfinite(q).all() or (q < 0).any() or np.any(q[~valid] != 0) or not np.allclose(q, expected, atol=1e-7, rtol=1e-6):
        raise ValueError('teacher_prob must match equal-mass Gibbs target')
    if 'area_weight' in teacher:
        raise ValueError('area_weight is not part of the Phase I contract')
    return {**{k: observation[k] for k in ('scene_id','view_id','response','obstacle','window')},
            **{k: teacher[k] for k in ('candidate_xy','valid','teacher_prob','physical_cost','temperature','config_id')},
            'interface_version': INTERFACE_VERSION}
