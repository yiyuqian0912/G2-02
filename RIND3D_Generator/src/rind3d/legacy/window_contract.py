# LEGACY DIAGNOSTICS ONLY: frozen 32/64/128/256 observation protocol.
"""The single authoritative dataset-window size and quota contract.

Generic field/ROI readers deliberately remain able to read arbitrary extents.
That utility capability must never define which windows enter this dataset.
"""
from collections import Counter
import numbers

WINDOW_SIDES=(32,64,128,256)
WINDOW_PROTOCOL='discrete-powers-of-two-v1'
PILOT_SIDE_COUNTS=dict.fromkeys(WINDOW_SIDES,8)
PRODUCTION_SIDE_COUNTS=dict.fromkeys(WINDOW_SIDES,1024)
PLACEMENT_VERSION='scrambled-sobol-discrete-sides-v2'


def require_window_side(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or int(value) not in WINDOW_SIDES:
        raise ValueError(f'Dataset window side must be one of {WINDOW_SIDES}; got {value!r}')
    return int(value)


def side_counts(values):
    counts=Counter(require_window_side(v) for v in values)
    return {side:counts[side] for side in WINDOW_SIDES}


def require_side_counts(values,expected):
    actual=side_counts(values)
    if actual!=expected:raise ValueError(f'Exact window-side quotas not met: {actual}; expected {expected}')
    return actual


def validate_window_config(cfg):
    if 'size_strata' in cfg:raise ValueError('Range-based size_strata is obsolete; use the four exact window_sides')
    if cfg.get('window_sides')!=list(WINDOW_SIDES):raise ValueError(f'window_sides must equal {list(WINDOW_SIDES)}')
    for side in cfg['window_sides']:require_window_side(side)
    for key,expected in (('pilot_side_counts',PILOT_SIDE_COUNTS),('production_side_counts',PRODUCTION_SIDE_COUNTS)):
        actual=cfg.get(key)
        if not isinstance(actual,dict) or actual!=expected or any(isinstance(k,bool) or not isinstance(k,numbers.Integral) or isinstance(v,bool) or not isinstance(v,numbers.Integral) for k,v in actual.items()):
            raise ValueError(f'{key} must equal {expected}')
    if cfg.get('placement_version')!=PLACEMENT_VERSION:raise ValueError('A corrected discrete-side placement_version is required')
