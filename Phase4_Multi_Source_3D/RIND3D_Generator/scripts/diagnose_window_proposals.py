# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Removable observer of the UNMODIFIED window sampler (Python >=3.12).

This is not a generator or a replacement sampler. Local PEP-669 callbacks see
the real sampler's decisions. Oracle proxies time calls without changing their
arguments, results, order, or exceptions. Diagnostic output is private and is
never consumable as a published scene. Run one process per source/stream.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

from rind3d.legacy.config import load_config
from rind3d.geometry import (Oracle, Budget, BudgetExhausted, Aborted,
                             NumericalUnresolved, load_geometry)
from rind3d.pack import atomic_json
from rind3d.pilot import source_fingerprint, config_hash
from rind3d.search import free_pool, farthest, search_candidates
from rind3d.legacy.windows import sample_windows


def plain(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class ReferenceObserver:
    def __init__(self, reference, observer):
        self.reference, self.observer = reference, observer

    def __getattr__(self, name):
        method = getattr(self.reference, name)
        if name not in ('classify_box', 'classify_points', 'segment_intersect'):
            return method

        def call(*args, **kwargs):
            o = self.observer
            active = o.current
            stage = o.stage
            if name == 'classify_box' and stage == 'certificate':
                o.box_number += 1
                part = 'receiver_box' if o.box_number == 1 else 'source_window_hull'
                key = f'certificate.{part}'
            else:
                key = f'{stage}.{name}'
            started = time.perf_counter()
            started_cpu = time.process_time()
            result = None
            error = None
            try:
                result = method(*args, **kwargs)
                return result
            except Exception as exc:
                error = type(exc).__name__ + ':' + str(exc)
                raise
            finally:
                elapsed = time.perf_counter() - started
                if active is not None:
                    item = active['reference_calls'].setdefault(
                        key, {'calls': 0, 'seconds': 0., 'process_seconds': 0., 'status': {}, 'work': {}})
                    item['calls'] += 1
                    item['seconds'] += elapsed
                    item['process_seconds'] += time.process_time() - started_cpu
                    if result is not None:
                        for status in np.asarray(result.status).ravel():
                            k = str(int(status))
                            item['status'][k] = item['status'].get(k, 0) + 1
                        for work in result.work:
                            for k, value in work.items():
                                if isinstance(value, (int, float)):
                                    item['work'][k] = item['work'].get(k, 0) + value
                    if name == 'classify_box':
                        active['box_queries'].append({
                            'stage': key, 'bounds': plain(args[0]), 'seconds': elapsed,
                            'valid': plain(result.valid) if result is not None else None,
                            'status': plain(result.status) if result is not None else None,
                            'value': plain(result.value) if result is not None else None,
                            'reason': plain(result.reason_code) if result is not None else error})
        return call


class OracleObserver:
    def __init__(self, oracle, observer):
        self.oracle, self.observer = oracle, observer
        self.original_reference = oracle.reference

    def __enter__(self):
        self.oracle.reference = ReferenceObserver(self.original_reference, self.observer)
        return self

    def __exit__(self, *exc):
        self.oracle.reference = self.original_reference

    def __getattr__(self, name):
        method = getattr(self.oracle, name)
        if name not in ('certificate', 'contains', 'response'):
            return method

        def call(*args, **kwargs):
            o = self.observer
            previous = o.stage
            o.stage = name
            if name == 'certificate':
                o.box_number = 0
            started = time.perf_counter()
            started_cpu = time.process_time()
            result = None
            try:
                result = method(*args, **kwargs)
                return result
            finally:
                if o.current is not None:
                    o.current['stage_seconds'][name] += time.perf_counter() - started
                    o.current['stage_process_seconds'][name] += time.process_time() - started_cpu
                    if name == 'certificate' and result is not None:
                        o.current['certificate'] = plain(result)
                o.stage = previous
        return call


class ProposalObserver:
    """Observe only local line events, avoiding global Python tracing overhead."""
    def __init__(self, source, sink=None):
        self.source = plain(np.asarray(source, dtype=np.float64))
        self.sink = sink
        self.rows = []
        self.current = None
        self.stage = 'outside'
        self.box_number = 0
        lines, first = inspect.getsourcelines(sample_windows)
        # Fail closed if the production sampler changes; no guessed hook lines.
        def line(token):
            matches = [first + i for i, text in enumerate(lines) if text.strip().startswith(token)]
            if len(matches) != 1:
                raise RuntimeError('Sampler changed: ambiguous diagnostic hook ' + token)
            return matches[0]
        self.begin_line = line('if w.record() in seen:')
        self.next_line = line('for v in proposals:')
        self.final_line = line('result,report=summary();rng.shuffle(result)')
        self.probes_line = line('if not len(pos) or not len(neg):')

    def monitor(self, code, lineno):
        if code is not sample_windows.__code__:
            return
        frame = sys._getframe(1)
        if frame.f_code is not sample_windows.__code__:
            raise RuntimeError('Cannot observe sampler locals')
        local = frame.f_locals
        if lineno in (self.next_line, self.final_line):
            self.finish(local)
        elif lineno == self.begin_line:
            if self.current is not None:
                raise AssertionError('Unfinished proposal')
            w = local['w']
            self.current = {
                'proposal': int(local['attempts']), 'side': w.size,
                'origin': list(w.origin), 'true_source': self.source,
                'accepted': False, 'outcome': None, 'reason': None,
                'sampler_reason': None, 'stage_seconds': defaultdict(float),
                'stage_process_seconds': defaultdict(float),
                'reference_calls': {}, 'box_queries': [],
                '_started': time.perf_counter(), '_started_cpu': time.process_time(), '_reasons': dict(local['reasons']),
                '_accepted': sum(map(len, local['accepted']))}
        elif lineno == self.probes_line:
            self.current['probes'] = {
                'coordinates': plain(local['coordinates']),
                'solid': plain(local['mask']), 'response': plain(local['y']),
                'positive_free': len(local['pos']), 'negative_free': len(local['neg']),
                'solid_count': int(np.count_nonzero(local['mask']))}

    def finish(self, local=None, exc=None):
        row = self.current
        if row is None:
            return
        row['elapsed_seconds'] = time.perf_counter() - row.pop('_started')
        row['process_seconds'] = time.process_time() - row.pop('_started_cpu')
        before = row.pop('_reasons')
        accepted_before = row.pop('_accepted')
        if exc is not None:
            row['outcome'] = 'interrupted'
            row['exception'] = type(exc).__name__ + ':' + str(exc)
            if isinstance(exc, BudgetExhausted):
                row['reason'] = 'BUDGET:' + exc.kind
            elif isinstance(exc, NumericalUnresolved):
                row['reason'] = 'REFERENCE:unresolved'
            elif isinstance(exc, Aborted):
                row['reason'] = 'REFERENCE:aborted'
            else:
                row['reason'] = 'OTHER:exception:' + type(exc).__name__
        else:
            delta = Counter(local['reasons']) - Counter(before)
            if sum(delta.values()) == 1:
                reason = next(iter(delta))
                row['sampler_reason'] = reason
                row['outcome'] = 'rejected'
                if reason == 'duplicate':
                    row['reason'] = 'DUPLICATE'
                elif reason == 'contains_true_source':
                    row['reason'] = 'SOURCE:inside_window'
                elif reason == 'certified_constant_or_solid':
                    yv, mv = row['certificate']
                    row['reason'] = ('GEOMETRY:certified_all_solid' if mv == 1 else
                                     f'RESPONSE:certified_constant_{yv}')
                elif reason == 'pilot_probes_inconclusive':
                    pos, neg = (row['probes'][k] for k in ('positive_free', 'negative_free'))
                    row['reason'] = ('WITNESS:missing_both' if not pos and not neg else
                                     'WITNESS:missing_positive' if not pos else
                                     'WITNESS:missing_negative')
                elif reason == 'constant_or_no_free':
                    counts = local['counts']
                    row['full_counts'] = counts
                    row['reason'] = ('FREE:insufficient' if counts['free_voxels'] <= 1 else
                                     'RESPONSE:stored_constant')
                else:
                    row['reason'] = 'OTHER:' + reason
            elif not delta and sum(map(len, local['accepted'])) == accepted_before + 1:
                row.update(accepted=True, outcome='accepted', reason='ACCEPTED')
                row['witnesses'] = plain(local['w'].witnesses)
            else:
                raise AssertionError('Cannot attribute a unique sampler decision')
        row['stage_seconds']['proposal_overhead'] = max(
            0., row['elapsed_seconds'] - sum(row['stage_seconds'].values()))
        row['stage_process_seconds']['proposal_overhead'] = max(
            0., row['process_seconds'] - sum(row['stage_process_seconds'].values()))
        row = plain(row)
        self.rows.append(row)
        if self.sink is not None:
            self.sink.write(json.dumps(row, separators=(',', ':')) + '\n')
            self.sink.flush()
        self.current = None

    def run(self, source, cfg, scene_id, oracle, **kwargs):
        tool_id = 5
        sys.monitoring.use_tool_id(tool_id, 'rind3d-window-diagnostic')
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, self.monitor)
        sys.monitoring.set_local_events(tool_id, sample_windows.__code__, sys.monitoring.events.LINE)
        error = None
        started = time.perf_counter()
        started_cpu = time.process_time()
        try:
            with OracleObserver(oracle, self) as proxy:
                result, report = sample_windows(source, cfg, scene_id, 32, oracle=proxy, **kwargs)
        except RuntimeError as exc:
            self.finish(exc=exc)
            error = {'type': type(exc).__name__, 'message': str(exc)}
            result, report = getattr(exc, 'partial_windows', []), getattr(exc, 'sampling_report', {})
        finally:
            sys.monitoring.set_local_events(tool_id, sample_windows.__code__, 0)
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, None)
            sys.monitoring.free_tool_id(tool_id)
        return result, {'sampling': report, 'termination': error,
                        'elapsed_seconds': time.perf_counter() - started,
                        'process_seconds': time.process_time() - started_cpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot-extended.yaml')
    parser.add_argument('--geometry', default='outputs/window-side-audit/pilot/scene-000006/geometry.npz')
    parser.add_argument('--scene', type=int, default=6)
    parser.add_argument('--source-rank', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--stream', type=int, default=302)
    parser.add_argument('--diagnostic-seconds', type=float, default=None,
                        help='Explicit diagnostic-only resource extension; does not edit the config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-accepted', action='store_true',
                        help='After proposal timing, run the unchanged candidate search and recheck')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    original = dict(cfg)
    if args.diagnostic_seconds is not None:
        cfg = dict(cfg, sampling_seconds=args.diagnostic_seconds)
    bundle = load_geometry(args.geometry)
    oracle = Oracle(bundle, cfg['numerical_policy'])
    started = time.perf_counter()
    pool = free_pool(oracle, cfg, args.scene, Budget(cfg['scene_setup_seconds'], cfg['window_search_points']))
    indices = farthest(pool, min(cfg['pilot_source_attempts'], len(pool)), 0)
    source = pool[indices[args.source_rank]].copy()
    oracle.validate_source(source, Budget(cfg['verification_seconds'], cfg['verification_points']))
    meta = {'diagnostic_only': True, 'is_dataset': False, 'scene_id': args.scene,
            'geometry': str(Path(args.geometry).resolve()),
            'geometry_sha256': hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest(),
            'geometry_content_hash': plain(bundle.geometries[0].content_hash),
            'source_rank': args.source_rank, 'source_pool_index': indices[args.source_rank],
            'true_source': source.tolist(), 'pool_size': len(pool),
            'source_policy': 'existing free_pool + farthest(pool,3,0)',
            'proposal_stream': args.stream, 'config': cfg, 'original_config_hash': config_hash(original),
            'resource_extension_seconds': args.diagnostic_seconds,
            'capability': oracle.capability, 'oracle_provenance': oracle.provenance,
            'filter_enabled': oracle.filter is not None, 'fingerprint': source_fingerprint(),
            'setup_seconds': time.perf_counter() - started,
            'started_unix': time.time(), 'complete': False}
    atomic_json(root/'run.json', meta)
    baseline_work = dict(oracle.metrics)
    with (root/'proposals.jsonl').open('w') as sink:
        observer = ProposalObserver(source, sink)
        windows, report = observer.run(source, cfg, args.scene, oracle, proposal_stream=args.stream)
    meta.update(report, complete=True, observed_proposals=len(observer.rows),
                oracle_work={k: oracle.metrics[k] - baseline_work[k] for k in baseline_work},
                windows=[{'origin': list(w.origin), 'side': w.size, 'witnesses': plain(w.witnesses)} for w in windows])
    atomic_json(root/'run.json', meta)
    if args.verify_accepted:
        started = time.perf_counter()
        candidate_rows = []
        for window_id, w in enumerate(windows):
            positions, evidence, candidate_report = search_candidates(
                oracle, source, w, pool, cfg, args.scene, window_id)
            candidate_rows.append({'window_id': window_id, 'origin': list(w.origin),
                                   'side': w.size, 'sources': plain(positions),
                                   'evidence': evidence, 'report': candidate_report})
            atomic_json(root/'candidate-checks.json', plain(candidate_rows))
        meta['post_candidate_validation_seconds'] = time.perf_counter() - started
        meta['post_candidate_validation_complete'] = all(
            r['report']['status'] == 'COMPLETE' and 1 <= len(r['sources']) <= 64
            and r['report']['all_grid_exact'] and r['report']['independent_validation_passed']
            for r in candidate_rows)
        atomic_json(root/'run.json', meta)
    print(json.dumps({k: meta[k] for k in ('source_rank', 'observed_proposals', 'elapsed_seconds', 'sampling', 'termination')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
