# Legacy four-side diagnostic/report reproduction only; not the current dataset contract.
"""Offline-only aggregation of certified evidence; never proposes a window.

Cached witnesses below prove only informativeness, not candidate equivalence.
The latter remains exclusively the unchanged full-grid and independent checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rind3d.pack import atomic_json


SIDES = (32, 64, 128, 256)


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k/n
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [max(0., centre-half), min(1., centre+half)]


def describe(values):
    values = list(values)
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None,
            'sum': sum(values)}


def describe_rows(rows):
    proposed = len(rows)
    accepted = sum(r['accepted'] for r in rows)
    decided = sum(r['outcome'] != 'interrupted' for r in rows)
    stages = Counter()
    for r in rows:
        stages.update(r['stage_seconds'])
    return {
        'proposed': proposed, 'decided': decided, 'accepted': accepted,
        'acceptance_rate_all_proposals': accepted/proposed if proposed else None,
        'acceptance_rate_decided': accepted/decided if decided else None,
        'descriptive_wilson95_decided': wilson(accepted, decided),
        'rejections': dict(Counter(r['reason'] for r in rows if r['outcome'] == 'rejected')),
        'interruptions': dict(Counter(r['reason'] for r in rows if r['outcome'] == 'interrupted')),
        'proposal_seconds': describe(r['elapsed_seconds'] for r in rows),
        'decided_proposal_seconds': describe(r['elapsed_seconds'] for r in rows if r['outcome'] != 'interrupted'),
        'stages_seconds': dict(stages),
        'probe_positive_free': sum(r.get('probes', {}).get('positive_free', 0) for r in rows),
        'probe_negative_free': sum(r.get('probes', {}).get('negative_free', 0) for r in rows),
        'probe_solid': sum(r.get('probes', {}).get('solid_count', 0) for r in rows),
        'probe_windows': sum('probes' in r for r in rows),
        'optional_box_unresolved': sum(
            sum(not valid for valid in q['valid']) for r in rows for q in r['box_queries'] if q['valid'] is not None),
    }


def cached_witness_analysis(rows):
    """Re-use already queried, EXACT voxel centres for the SAME true source.

    This deliberately uses later/other-stream points, so it is an offline
    existence test, not an estimate of any deployable guided sampler's gain.
    A miss in this finite cache proves nothing about the full window.
    """
    by_source = defaultdict(dict)
    solid_by_coordinate = {}
    for row in rows:
        source = tuple(row['true_source'])
        probes = row.get('probes')
        if probes is None:
            continue
        for coord, solid, response in zip(probes['coordinates'], probes['solid'], probes['response'], strict=True):
            coord = tuple(coord)
            if coord in solid_by_coordinate and solid_by_coordinate[coord] != solid:
                raise AssertionError('Contradictory certified geometry labels')
            solid_by_coordinate[coord] = solid
            if solid:
                if response:
                    raise AssertionError('Solid probe cannot be lit')
                continue
            if coord in by_source[source] and by_source[source][coord] != response:
                raise AssertionError('Contradictory certified response labels')
            by_source[source][coord] = response
    caches = {}
    for source, evidence in by_source.items():
        caches[source] = {label: np.array([p for p, y in evidence.items() if y == label], dtype=int).reshape(-1, 3)
                          for label in (False, True)}
    results = []
    nearby_pairs = []
    for source, labels in caches.items():
        positive, negative = labels[True], labels[False]
        if not len(positive) or not len(negative):
            continue
        distances, indices = cKDTree(negative).query(positive, p=np.inf)
        nearest = int(np.argmin(distances))
        p, q = positive[nearest], negative[indices[nearest]]
        sides = {}
        for side in SIDES:
            # Count integer origins that contain this exact opposite-label
            # pair, are in bounds, and exclude the true source. We do not
            # propose or evaluate any of these windows.
            lower = np.maximum(np.maximum(p, q)-side+1, 0)
            upper = np.minimum(np.minimum(p, q), 1024-side)
            n = np.maximum(0, upper-lower+1)
            total = int(np.prod(n))
            src = np.asarray(source)*1024
            forbidden_lo = np.maximum(lower, np.floor(src-side).astype(int)+1)
            forbidden_hi = np.minimum(upper, np.floor(src).astype(int))
            forbidden = int(np.prod(np.maximum(0, forbidden_hi-forbidden_lo+1)))
            sides[str(side)] = {'source_excluding_origins': total-forbidden,
                                'fraction_of_all_integer_origins': (total-forbidden)/(1024-side+1)**3,
                                'positive_points_with_negative_within_side': int(np.count_nonzero(distances < side))}
        nearby_pairs.append({'source': list(source), 'minimum_linf_distance_voxels': float(distances[nearest]),
                             'positive_negative_pair': [p.tolist(), q.tolist()], 'by_side': sides})
    for row in rows:
        cache = caches.get(tuple(row['true_source']), {})
        origin = np.asarray(row['origin'])
        witnesses = []
        for label in (True, False):
            points = cache.get(label, np.empty((0, 3), int))
            inside = points[np.all((points >= origin) & (points < origin + row['side']), axis=1)]
            witnesses.append(inside[0].tolist() if len(inside) else None)
        if all(point is not None for point in witnesses):
            if row['reason'].startswith(('GEOMETRY:certified', 'RESPONSE:certified')):
                raise AssertionError('Cached points contradict constant box certificate')
            if row['outcome'] == 'rejected' and row['reason'].startswith('WITNESS:'):
                results.append({'run': row['run'], 'proposal': row['proposal'], 'side': row['side'],
                                'origin': row['origin'], 'source_rank': row['source_rank'],
                                'true_source': row['true_source'], 'sampler_reason': row['reason'],
                                'positive_negative_free_witnesses': witnesses})
    return {'additional_proven_informative_windows': len(results),
            'by_side': {str(side): sum(r['side'] == side for r in results) for side in SIDES},
            'by_source': {str(rank): sum(r['source_rank'] == rank for r in results) for rank in range(3)},
            'examples': results,
            'nearest_opposite_free_pairs': nearby_pairs,
            'free_cached_points_by_source': {str(source): len(points[False]) + len(points[True]) for source, points in caches.items()},
            'caveat': 'Offline lower bound from certified points; later/other-stream points allowed. No new proposal, label, candidate, or production decision.'}


def source_distances(rows):
    result = {}
    for side in SIDES:
        records = []
        for row in rows:
            if row['side'] != side or row['outcome'] == 'interrupted':
                continue
            source = np.asarray(row['true_source']) * 1024
            origin = np.asarray(row['origin'])
            distance = float(np.linalg.norm(np.maximum(np.maximum(origin - source, source - (origin + side)), 0.))) / 1024
            records.append((distance, row))
        if not records:
            continue
        edges = np.quantile([p[0] for p in records], [0, .25, .5, .75, 1])
        bins = []
        for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            part = [r for distance, r in records if a <= distance and (distance <= b if index == 3 else distance < b)]
            bins.append({'distance_range': [float(a), float(b)], **describe_rows(part)})
        result[str(side)] = bins
    return result


def summarize(root, pattern, partial=False):
    rows, runs = [], []
    candidate_rows = []
    for directory in sorted(root.glob(pattern)):
        if not directory.is_dir() or not (directory/'run.json').exists():
            continue
        run = json.loads((directory/'run.json').read_text())
        if not partial and not run['complete']:
            raise ValueError('Diagnostic still running: ' + str(directory))
        run['run'] = directory.name
        runs.append(run)
        path = directory/'proposals.jsonl'
        if path.exists():
            for text in path.read_text().splitlines():
                row = json.loads(text)
                row.update(run=directory.name, source_rank=run['source_rank'], stream=run['proposal_stream'])
                rows.append(row)
        candidate_path = directory/'candidate-checks.json'
        if candidate_path.exists():
            candidate_rows.extend(json.loads(candidate_path.read_text()))
    sampling_seconds = sum(r.get('elapsed_seconds', 0) for r in runs)
    stages = Counter()
    refs = {}
    for row in rows:
        stages.update(row['stage_seconds'])
        for name, value in row['reference_calls'].items():
            part = refs.setdefault(name, {'seconds': 0., 'calls': 0, 'status': Counter(), 'work': Counter()})
            part['seconds'] += value['seconds']
            part['calls'] += value['calls']
            part['status'].update(value['status'])
            part['work'].update(value['work'])
    stages['sampler_setup_and_between_proposals'] = max(0., sampling_seconds - sum(stages.values()))
    per_run = {r['run']: describe_rows([row for row in rows if row['run'] == r['run']]) for r in runs}
    result = {'complete': not partial, 'run_count': len(runs), 'runs': [
        {k: run.get(k) for k in ('run', 'source_rank', 'source_pool_index', 'true_source', 'proposal_stream',
                                'complete', 'elapsed_seconds', 'setup_seconds', 'sampling', 'termination',
                                'post_candidate_validation_seconds', 'post_candidate_validation_complete')}
        for run in runs],
        'overall': describe_rows(rows),
        'by_side': {str(side): describe_rows([r for r in rows if r['side'] == side]) for side in SIDES},
        'by_source_and_side': {str(rank): {str(side): describe_rows([r for r in rows if r['source_rank'] == rank and r['side'] == side]) for side in SIDES} for rank in range(3)},
        'by_run': per_run,
        'sampler_total_seconds': sampling_seconds,
        'stages': {name: {'seconds': value, 'fraction_of_sampler_runtime': value/sampling_seconds if sampling_seconds else None} for name, value in stages.items()},
        'reference_substages': refs,
        'cached_witness_analysis': cached_witness_analysis(rows),
        'distance_to_source_quartiles': source_distances(rows),
        'candidate_checks': {'windows': len(candidate_rows),
                             'count_distribution': dict(Counter(str(len(r['sources'])) for r in candidate_rows)),
                             'status': dict(Counter(r['report']['status'] for r in candidate_rows)),
                             'all_grid_exact': all(r['report']['all_grid_exact'] for r in candidate_rows) if candidate_rows else None,
                             'all_independent_rechecked': all(r['report']['independent_validation_passed'] for r in candidate_rows) if candidate_rows else None}}
    if not partial:
        for run in runs:
            actual = per_run[run['run']]
            report = run['sampling']
            assert actual['proposed'] == report['attempts']
            assert actual['accepted'] == report['accepted']
            assert sum(actual['rejections'].values()) == sum(report['rejections'].values())
            assert sum(actual['interruptions'].values()) == report['interrupted_attempts']
    return result, rows


def write_tables(root, result, rows, prefix):
    atomic_json(root/f'{prefix}.json', result)
    with (root/f'{prefix}-proposals.csv').open('w', newline='') as f:
        fields = ['run', 'source_rank', 'stream', 'proposal', 'side', 'origin', 'true_source',
                  'outcome', 'accepted', 'reason', 'sampler_reason', 'elapsed_seconds',
                  'certificate_seconds', 'contains_seconds', 'response_seconds', 'positive_free', 'negative_free', 'solid_count']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {k: row.get(k) for k in fields[:12]}
            for k in ('origin', 'true_source'):
                item[k] = json.dumps(item[k])
            item.update({k + '_seconds': row['stage_seconds'].get(k, 0) for k in ('certificate', 'contains', 'response')})
            item.update({k: row.get('probes', {}).get(k) for k in ('positive_free', 'negative_free', 'solid_count')})
            writer.writerow(item)
    reasons = sorted({reason for side in result['by_side'].values() for reason in (*side['rejections'], *side['interruptions'])})
    with (root/f'{prefix}-rejections.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reason', *SIDES])
        for reason in reasons:
            writer.writerow([reason, *(result['by_side'][str(side)]['rejections'].get(reason, 0) + result['by_side'][str(side)]['interruptions'].get(reason, 0) for side in SIDES)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/slabs-random-field-diagnostic')
    parser.add_argument('--pattern', default='study-source-*-stream-*')
    parser.add_argument('--prefix', default='summary')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    result, rows = summarize(root, args.pattern, args.partial)
    write_tables(root, result, rows, args.prefix)
    print(json.dumps({'proposed': len(rows), 'by_side': result['by_side'],
                      'stages': result['stages'], 'candidate_checks': result['candidate_checks'],
                      'cached_additional_informative': result['cached_witness_analysis']['by_side']}, indent=2))


if __name__ == '__main__':
    main()
