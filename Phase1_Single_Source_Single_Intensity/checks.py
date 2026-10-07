"""Strict C/D handoff checks, independent of dataset and PyTorch imports."""
import json
from pathlib import Path
import numpy as np


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        record = {k: archive[k].copy() for k in archive.files}
    if 'metadata_json' in record:
        record['metadata'] = json.loads(str(record.pop('metadata_json').item()))
    for key in ('scene_id', 'view_id', 'temperature', 'config_id', 'run_id'):
        if key in record:
            record[key] = np.asarray(record[key]).item()
    return record


def probability(values, valid, name):
    p = np.asarray(values, dtype=float)
    if p.shape != valid.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f'{name}: expected finite nonnegative aligned [N]')
    if np.any(p[~valid] != 0) or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError(f'{name}: invalid mass must be zero and total mass must be one')
    return p


def teacher(record):
    for key in ('scene_id', 'view_id', 'candidate_xy', 'valid', 'physical_cost',
                'teacher_prob', 'temperature', 'window', 'config_id'):
        if key not in record:
            raise ValueError(f'missing teacher field: {key}')
    xy, valid = np.asarray(record['candidate_xy']), np.asarray(record['valid'])
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('candidate_xy must be finite nonempty [N,2]')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('duplicate candidate coordinates')
    if valid.dtype != np.dtype(bool) or valid.shape != (len(xy),) or not valid.any():
        raise ValueError('valid must be boolean [N] with a valid candidate')
    cost = np.asarray(record['physical_cost'], dtype=float)
    if cost.shape != valid.shape or not np.isfinite(cost[valid]).all() or (cost[valid] < 0).any():
        raise ValueError('valid physical costs must be finite and nonnegative')
    if 'evaluated' not in record or np.asarray(record['evaluated']).dtype != np.dtype(bool) or np.asarray(record['evaluated']).shape != valid.shape or not np.asarray(record['evaluated']).all():
        raise ValueError('a completed evaluated mask is required')
    metadata = record.get('metadata', {})
    if metadata.get('complete') is not True:
        raise ValueError('teacher must explicitly declare complete=True')
    window = np.asarray(record['window'])
    if window.shape != (3,) or window.dtype.kind not in 'iu' or window[2] <= 0 or (window[:2] < 0).any():
        raise ValueError('window must be integer [x,y,L], L>0')
    tau = float(record['temperature'])
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('temperature must be finite and positive')
    q = probability(record['teacher_prob'], valid, 'teacher_prob')
    with np.errstate(over='ignore', under='ignore'):
        w = np.exp(-(cost[valid] - cost[valid].min()) / tau)
    if not np.allclose(q[valid], w / w.sum(), atol=1e-6, rtol=1e-6):
        raise ValueError('teacher probability does not match equal-mass Gibbs target')
    return xy.astype(float), valid, cost, q


def aligned(reference, other, *, config=True):
    for key in ('scene_id', 'view_id'):
        if reference[key] != other[key]:
            raise ValueError(f'mismatched {key}')
    for key in ('candidate_xy', 'valid', 'window'):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f'mismatched {key}; common ordered support required')
    if config and reference['config_id'] != other.get('config_id'):
        raise ValueError('student prediction references a different teacher config_id')


def scene_splits(splits):
    if not {'train', 'validation', 'test'} <= splits.keys():
        raise ValueError('split manifest needs train, validation, test')
    seen = set()
    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{name}: nonempty unique integer scene IDs required')
        if seen.intersection(ids):
            raise ValueError(f'scene leakage in {name}')
        seen.update(ids)
    return {name: set(ids) for name, ids in splits.items()}


def observation(record, sample):
    for key in ('scene_id', 'view_id'):
        if record[key] != sample[key]:
            raise ValueError(f'observation mismatched {key}')
    if not np.array_equal(record['window'], sample['window']):
        raise ValueError('observation window mismatch')
    response = np.asarray(sample['response'])
    size = int(record['window'][2])
    if response.shape != (size, size) or not np.isfinite(response).all():
        raise ValueError('response must be finite [L,L] at original resolution')
    return response


def physical_reconstruction(dataset, scene_id, view_id, source_xy, *, backend='auto', atol=0):
    sample = dataset.get_observation(scene_id, view_id)
    x, y, size = map(int, sample['window'])
    sx, sy = map(float, source_xy)
    world = float(dataset.manifest['global_size'])
    inside_x = x <= sx < x + size or (x + size == world and sx == world)
    inside_y = y <= sy < y + size or (y + size == world and sy == world)
    if inside_x and inside_y:
        raise ValueError('generating source is inside the observation window')
    rendered = dataset.rerender(scene_id, sample['window'], source_xy, backend=backend)
    error = float(np.max(np.abs(np.asarray(rendered, dtype=float) - sample['response'])))
    if error > atol:
        raise ValueError(f'generating-source reconstruction failed: {error}')
    return {'max_absolute_error': error, 'source_free_window': True}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--teacher', type=Path)
    source.add_argument('--pipeline-config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('part_e_outputs/end_to_end'))
    parser.add_argument('--student', type=Path)
    parser.add_argument('--splits', type=Path)
    args = parser.parse_args()
    if args.pipeline_config:
        from .pipeline import run
        config = json.loads(args.pipeline_config.read_text(encoding='utf-8'))
        for key in ('teacher_source','student_source','data_root','checkpoint'):
            if config.get(key):
                config[key] = str((args.pipeline_config.parent/config[key]).resolve())
        report = run(config,args.output)
        print(f"Saved stage checks to {args.output}; complete={report['full_pipeline_complete']}")
        if not report['full_pipeline_complete']:
            raise SystemExit(2)
        return
    record = load_record(args.teacher)
    _, valid, _, _ = teacher(record)
    if args.student:
        prediction = load_record(args.student)
        aligned(record, prediction)
        probability(prediction['student_prob'], valid, 'student_prob')
    if args.splits:
        scene_splits(json.loads(args.splits.read_text(encoding='utf-8')))
    print('PASS: requested handoff checks (physical reconstruction requires dataset).')


if __name__ == '__main__':
    main()
