"""Count-first validation policy; no target enters model.forward."""
import math

GROUPS = {'R1': (1, 1), 'R2': (2, 2), 'R3': (3, 3), 'R4': (4, 4), 'R5plus': (5, 32)}


def summarize_confusion(confusion):
    import numpy as np
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (33, 33) or np.any(matrix < 0):
        raise ValueError('count confusion must be a nonnegative 33x33 matrix')
    by_r = {}
    for r in range(1, 33):
        n = int(matrix[r].sum())
        if n:
            by_r[str(r)] = {'windows': n, 'accuracy': float(matrix[r, r] / n)}
    groups = {}
    for name, (lo, hi) in GROUPS.items():
        n = int(matrix[lo:hi + 1].sum())
        correct = sum(int(matrix[r, r]) for r in range(lo, hi + 1))
        groups[name] = {'windows': n, 'accuracy': correct / n if n else None}
    present = [v['accuracy'] for v in groups.values() if v['windows']]
    if not present:
        raise ValueError('count evaluation has no positive-layer windows')
    return {'count_group_macro': sum(present) / len(present),
            'count_by_group': groups, 'count_by_R': by_r, 'count_confusion': matrix.tolist()}


def selection_key(metrics, options=None):
    if not options:
        return (-float(metrics['loss']),)
    if set(options) - {'metric','max_single_layer_drop','max_overall_drop','quality_max_drop'} or not {'metric','max_single_layer_drop','max_overall_drop'} <= set(options):
        raise ValueError('model_selection requires metric/max_single_layer_drop/max_overall_drop')
    if options['metric'] not in ('count_accuracy', 'count_group_macro', 'count_shape_equal'):
        raise ValueError('model_selection.metric must be count_accuracy or count_group_macro')
    if any(not isinstance(options[k], (int, float)) or not math.isfinite(options[k])
           or not 0 <= options[k] <= 1 for k in ('max_single_layer_drop', 'max_overall_drop')):
        raise ValueError('selection degradation limits must lie in [0,1]')
    guards=options.get('quality_max_drop',{})
    if (not isinstance(guards,dict) or set(guards)-{'mask_dice','strength_within_10pct','strict_decomposition_success','layer_accuracy'}
            or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in guards.values())):
        raise ValueError('invalid quality_max_drop guards')
    if options['metric']=='count_shape_equal':
        key=(.5*(float(metrics['count_group_macro'])+float(metrics['shape_group_macro'])),
             .5*(float(metrics['count_accuracy'])+float(metrics['layer_accuracy'])),-float(metrics['loss']))
        if not all(math.isfinite(v) for v in key):raise ValueError('nonfinite equal-task selection key')
        return key
    key = (float(metrics[options['metric']]), float(metrics['count_accuracy']),
           -float(metrics['count_mae']), -float(metrics['loss']))
    if not all(math.isfinite(v) for v in key):
        raise ValueError('nonfinite validation selection key')
    return key


def eligible(metrics, baseline, options=None):
    if not options or baseline is None:
        return True
    for key,drop in options.get('quality_max_drop',{}).items():
        if metrics[key] < baseline[key]-drop-1e-12:return False
    if metrics['count_accuracy'] < baseline['count_accuracy'] - options['max_overall_drop'] - 1e-12:
        return False
    old = baseline['count_by_group']['R1']['accuracy']
    current = metrics['count_by_group']['R1']['accuracy']
    return old is None or (current is not None and
                         current >= old - options['max_single_layer_drop'] - 1e-12)
