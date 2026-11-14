# Explicit legacy four-side diagnostic regression tests.
"""The removable diagnostic must observe real decisions without changing them."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rind3d.legacy.config import load_config
from rind3d.geometry import BudgetExhausted, Aborted, NumericalUnresolved
from rind3d.legacy.windows import sample_windows

spec = importlib.util.spec_from_file_location(
    'proposal_diagnostic', Path(__file__).resolve().parents[1]/'scripts/diagnose_window_proposals.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
summary_spec = importlib.util.spec_from_file_location(
    'proposal_summary', Path(__file__).resolve().parents[1]/'scripts/summarize_window_diagnostic.py')
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


class MockOracle:
    def __init__(self, mode='mixed', error=None):
        self.reference = object()
        self.mode, self.error = mode, error
        self.calls = 0

    def certificate(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        if self.mode == 'decisions':
            return [(0, 1), (1, 0), (-1, 0)][(self.calls - 1) % 3]
        return -1, 0

    def contains(self, points, **kwargs):
        return np.full(len(points), self.mode == 'solid_probes', bool)

    def response(self, source, points, **kwargs):
        if self.mode in ('missing_positive', 'solid_probes'):
            return np.zeros(len(points), bool)
        if self.mode == 'missing_negative':
            return np.ones(len(points), bool)
        return np.arange(len(points)) % 2 == 0


def config():
    cfg = load_config('configs/pilot-extended.yaml')
    return dict(cfg, max_window_attempts=128)


@pytest.mark.parametrize('mode', ['mixed', 'decisions', 'missing_positive', 'missing_negative', 'solid_probes'])
def test_observer_preserves_sampler_results_order_rng_and_counts(mode):
    cfg = config()
    source = np.array([.01, .02, .03])
    expected, report = sample_windows(source, cfg, 6, 32, oracle=MockOracle(mode))
    oracle = MockOracle(mode)
    reference = oracle.reference
    observer = diagnostic.ProposalObserver(source)
    actual, observed = observer.run(source, cfg, 6, oracle)
    assert oracle.reference is reference
    assert actual == expected
    assert observed['sampling'] == report
    assert len(observer.rows) == report['attempts']
    assert sum(r['accepted'] for r in observer.rows) == len(expected)
    assert sum(r['outcome'] == 'rejected' for r in observer.rows) == sum(report['rejections'].values())
    assert all(r['side'] in (32, 64, 128, 256) for r in observer.rows)
    assert all(abs(sum(r['stage_seconds'].values()) - r['elapsed_seconds']) < 1e-8 for r in observer.rows)
    if mode == 'mixed':
        assert report['side_counts'] == {32: 8, 64: 8, 128: 8, 256: 8}
    elif mode in ('missing_positive', 'missing_negative', 'solid_probes'):
        target = {'missing_positive': 'WITNESS:missing_positive',
                  'missing_negative': 'WITNESS:missing_negative',
                  'solid_probes': 'WITNESS:missing_both'}[mode]
        assert target in {r['reason'] for r in observer.rows}
        assert all(not r['accepted'] for r in observer.rows)


@pytest.mark.parametrize('error,reason', [
    (BudgetExhausted('time'), 'BUDGET:time'),
    (BudgetExhausted('point'), 'BUDGET:point'),
    (Aborted('Canonical oracle aborted'), 'REFERENCE:aborted'),
    (NumericalUnresolved('unproved'), 'REFERENCE:unresolved'),
])
def test_work_limits_and_unresolved_are_interruptions_never_physical_rejections(error, reason):
    source = [.01, .02, .03]
    observer = diagnostic.ProposalObserver(source)
    _, result = observer.run(source, config(), 6, MockOracle(error=error))
    assert result['termination']['type'] == type(error).__name__
    assert result['sampling']['interrupted_attempts'] == 1
    assert result['sampling']['rejections'] == {}
    assert observer.rows[0]['outcome'] == 'interrupted'
    assert observer.rows[0]['reason'] == reason


def test_real_source_and_duplicate_gates_run_before_physics(monkeypatch):
    import rind3d.legacy.windows as windows
    # Repeated coordinates yield duplicates at each side on the second cycle.
    monkeypatch.setattr(windows, 'sobol_points', lambda *args: np.zeros((16, 3)))
    source = [.01, .01, .01]
    observer = diagnostic.ProposalObserver(source)
    oracle = MockOracle()
    _, result = observer.run(source, config(), 6, oracle)
    assert result['sampling']['attempts'] == 16
    assert oracle.calls == 0
    assert [r['reason'] for r in observer.rows[:4]] == ['SOURCE:inside_window'] * 4
    assert [r['reason'] for r in observer.rows[4:]] == ['DUPLICATE'] * 12


def evidence_row(proposal, source, coords, labels, *, reason='WITNESS:missing_positive'):
    return {'proposal': proposal, 'source_rank': 0, 'run': 'test', 'side': 32,
            'true_source': source, 'origin': [128, 128, 128],
            'reason': reason, 'outcome': 'rejected',
            'probes': {'coordinates': coords, 'solid': [False]*len(coords), 'response': labels}}


def test_offline_witness_reuse_is_source_specific_and_uses_exact_voxel_centres():
    source = [.01, .02, .03]
    row = evidence_row(1, source, [[130, 131, 132]], [False])
    other_source = evidence_row(2, [.9, .8, .7], [[140, 141, 142]], [True])
    outside = evidence_row(3, source, [[160, 141, 142]], [True])
    outside['origin'] = [160, 128, 128]
    assert summary.cached_witness_analysis([row, other_source, outside])['additional_proven_informative_windows'] == 0
    inside = evidence_row(4, source, [[159, 141, 142]], [True], reason='WITNESS:missing_negative')
    result = summary.cached_witness_analysis([row, other_source, outside, inside])
    assert result['additional_proven_informative_windows'] == 2
    assert result['examples'][0]['positive_negative_free_witnesses'] == [[159, 141, 142], [130, 131, 132]]


def test_offline_cache_fails_on_contradictory_truth_or_constant_certificate():
    source = [.01, .02, .03]
    row = evidence_row(1, source, [[130, 131, 132]], [False])
    opposite = evidence_row(2, source, [[130, 131, 132]], [True])
    with pytest.raises(AssertionError, match='Contradictory certified response'):
        summary.cached_witness_analysis([row, opposite])
    constant = evidence_row(3, source, [[140, 141, 142]], [True], reason='RESPONSE:certified_constant_1')
    with pytest.raises(AssertionError, match='contradict constant box'):
        summary.cached_witness_analysis([row, constant])


@pytest.mark.parametrize('source', [[.24, .24, .24], [.25, .25, .25]])
def test_offline_pair_existence_counts_match_exhaustive_integer_placements(source):
    row = evidence_row(1, source, [[255, 255, 255], [256, 256, 256]], [True, False])
    row['origin'] = [240, 240, 240]
    row['outcome'], row['reason'] = 'accepted', 'ACCEPTED'
    result = summary.cached_witness_analysis([row])
    measured = result['nearest_opposite_free_pairs'][0]['by_side']['32']['source_excluding_origins']
    x, y, z = np.meshgrid(np.arange(225, 256), np.arange(225, 256), np.arange(225, 256))
    origins = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    source = np.asarray(source)*1024
    inside = np.all((source >= origins) & (source < origins+32), axis=1)
    assert measured == int((~inside).sum())
