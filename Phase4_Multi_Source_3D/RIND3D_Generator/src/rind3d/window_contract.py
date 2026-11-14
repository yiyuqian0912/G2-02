"""The single authoritative dataset-window size and quota contract.

Generic field/ROI readers deliberately remain able to read arbitrary extents.
That utility capability must never define which windows enter this dataset.
"""
from collections import Counter
import numbers

WINDOW_SIDES=(64,128,256)
WINDOW_PROTOCOL='discrete-64-128-256-v2'
PLACEMENT_VERSION='stored-index-window-first-v3'


def balanced_quotas(count):
    if isinstance(count,bool) or not isinstance(count,numbers.Integral) or count<1:raise ValueError('Positive integer total window count required')
    base,remainder=divmod(int(count),len(WINDOW_SIDES))
    return {side:base+int(remainder>0 and i>=len(WINDOW_SIDES)-remainder) for i,side in enumerate(WINDOW_SIDES)}


PILOT_SIDE_COUNTS=balanced_quotas(32)
PRODUCTION_SIDE_COUNTS={64:1365,128:1365,256:1366}


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
    if 'size_strata' in cfg:raise ValueError('Range-based size_strata is obsolete; use the three exact window_sides')
    if cfg.get('window_sides')!=list(WINDOW_SIDES):raise ValueError(f'window_sides must equal {list(WINDOW_SIDES)}')
    for side in cfg['window_sides']:require_window_side(side)
    from .profiles import profile_for
    profile=profile_for(cfg)
    for key,expected in (('pilot_side_counts',PILOT_SIDE_COUNTS),('production_side_counts',profile.quotas)):
        actual=cfg.get(key)
        if not isinstance(actual,dict) or actual!=expected or any(isinstance(k,bool) or not isinstance(k,numbers.Integral) or isinstance(v,bool) or not isinstance(v,numbers.Integral) for k,v in actual.items()):
            raise ValueError(f'{key} must equal {expected}')
    if cfg.get('placement_version')!=PLACEMENT_VERSION:raise ValueError('A corrected discrete-side placement_version is required')
