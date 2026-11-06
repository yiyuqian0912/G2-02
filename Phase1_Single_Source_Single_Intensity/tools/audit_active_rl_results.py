"""Audit downloaded single-source RL results without reading hidden scenes.

The content probes are synthetic sensitivity checks of the saved actor. They
are not new sensor rollouts and do not replace the original test episodes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rind_phase1.active_rl.core import ActorCritic, TileEnv, Window
from rind_phase1.active_rl.locator import policy_inputs
from rind_phase1.active_rl.frontier import policy_names


def read(path):
    return json.loads(Path(path).read_text())


def average(rows, key):
    return float(np.mean([r[key] for r in rows]))


@torch.no_grad()
def sensitivity(root, protocol, metrics):
    if protocol['config'].get('environment')=='frontier':
        return {'available': False, 'reason': 'Long-horizon frontier sensitivity is not run by this v1 CPU audit; quantitative episode auditing remains enabled.'}
    snapshot = root / 'evaluation/policy_snapshots' / (metrics['policy_checkpoint']['sha256'] + '.pt')
    if not snapshot.is_file():
        return {'available': False}
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != metrics['policy_checkpoint']['sha256']:
        raise ValueError('policy snapshot SHA mismatch')
    saved = torch.load(snapshot, map_location='cpu', weights_only=True)
    model = ActorCritic(saved['config']['hidden'])
    model.load_state_dict(saved['policy'])
    model.eval()
    c = protocol['config']
    world = protocol['world']
    grid = world // c['spacing']
    locator = SimpleNamespace(device=torch.device('cpu'), world=world, grid=grid,
                              tile=c['tile_size'], spacing=c['spacing'])
    scenes = protocol['scene_splits']['test']
    variants = ('uniform', 'peak_top_left', 'peak_bottom_right', 'lit_observed', 'checker_observed')
    results = {}
    for budget in c['evaluation_budgets']:
        by_variant = {}
        for variant in variants:
            probability = torch.full((grid * grid,), 1 / (grid * grid))
            if variant.startswith('peak_'):
                probability.fill_(1e-8)
                probability[0 if variant == 'peak_top_left' else -1] = 1
                probability /= probability.sum()
            contexts = [SimpleNamespace(
                env=TileEnv(world, c['tile_size'], Window(*protocol['initial_windows'][str(s)]),
                            budget, c['max_steps']),
                belief=SimpleNamespace(response=torch.zeros(grid, grid), probability=probability))
                for s in scenes]
            while True:
                live = [x for x in contexts if not x.env.done]
                if not live:
                    break
                inputs = list(policy_inputs(live, locator))
                if variant == 'lit_observed':
                    inputs[0][:, 0] = inputs[0][:, 1]
                elif variant == 'checker_observed':
                    checker = (torch.arange(grid)[:, None] + torch.arange(grid)[None, :]) % 2
                    inputs[0][:, 0] = inputs[0][:, 1] * checker
                distribution, _ = model(*inputs)
                for context, action in zip(live, distribution.logits.argmax(1).tolist()):
                    context.env.step(action)
            by_variant[variant] = [x.env.history for x in contexts]
        reference = by_variant['uniform']
        matches = {v: sum(a == b for a, b in zip(reference, paths)) for v, paths in by_variant.items()}
        trace_matches = []
        for scene, history in zip(scenes, reference):
            path = root / 'figures' / f'scene{scene}_b{budget}_ppo.json'
            if path.is_file():
                trace_matches.append({'scene_id': scene, 'same_history': read(path)['history'] == history})
        results[str(budget)] = {
            'scenes': len(scenes), 'exact_path_matches_to_uniform': matches,
            'available_real_trace_comparison': trace_matches,
            'uniform_first_actions': dict(Counter(h[1]['action'] for h in reference)),
        }
    return {'available': True, 'policy_sha256': digest, 'by_budget': results,
            'limitation': 'Synthetic inputs; only saved real traces can be matched directly. '
                          'Same greedy actions do not imply identical action probabilities.'}


def audit(root, output):
    protocol = read(root / 'protocol.json')
    training = read(root / 'training.json')
    metrics = read(root / 'evaluation/metrics.json')
    episodes = read(root / 'evaluation/episodes.json')
    expected = {(s, b, p) for s in protocol['scene_splits']['test']
                for b in protocol['config']['evaluation_budgets']
                for p in policy_names(protocol['config'])}
    actual = [(r['scene_id'], r['budget'], r['policy']) for r in episodes]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('missing or duplicated evaluation episodes')
    groups = {}
    for budget in protocol['config']['evaluation_budgets']:
        policies = {}
        for policy in policy_names(protocol['config']):
            rows = [r for r in episodes if r['budget'] == budget and r['policy'] == policy]
            for row in rows:
                if not (row['spent'] <= budget and row['spent'] == row['curve'][-1]['spent']
                        and abs(row['return_value'] - row['score'] + row['initial_score']) < 1e-5):
                    raise ValueError('inconsistent budget or reward')
            policies[policy] = dict(
                scenes=len(rows), steps=dict(Counter(r['steps'] for r in rows)),
                reasons=dict(Counter(r['reason'] for r in rows)),
                mean={k: average(rows, k) for k in ('mass32', 'map_distance', 'expected_distance',
                                                   'entropy', 'core90_area', 'spent')},
                median_map_distance=float(np.median([r['map_distance'] for r in rows])),
                median_mass32=float(np.median([r['mass32'] for r in rows])),
                hits_by_radius={str(a): sum(r['map_distance'] <= a for r in rows)
                                for a in (16, 32, 64, 128, 256)},
                map_over_256=sum(r['map_distance'] > 256 for r in rows),
                map_over_512=sum(r['map_distance'] > 512 for r in rows),
                improved_score=sum(r['return_value'] > 1e-6 for r in rows),
                worsened_score=sum(r['return_value'] < -1e-6 for r in rows),
                sharpened_but_worsened=sum(r['entropy'] < r['curve'][0]['entropy'] - 1e-6
                                          and r['return_value'] < -1e-6 for r in rows),
                misses_core90=sum(not r['core90_covers_truth'] for r in rows),
                initial_uniform=int(sum(abs(r['curve'][0]['entropy'] - np.log((protocol['world'] /
                                    protocol['config']['spacing']) ** 2)) < 1e-5 for r in rows)),
                source_observed_groups={str(observed): dict(
                    scenes=len(subset), hits32=sum(r['map_distance'] <= 32 for r in subset),
                    mean_mass32=average(subset, 'mass32') if subset else None,
                    mean_map_distance=average(subset, 'map_distance') if subset else None)
                    for observed in (False, True)
                    for subset in [[r for r in rows if r['source_was_observed'] == observed]]})
        groups[str(budget)] = policies
    history = training['history']
    result = dict(
        schema='rind-active-rl-downloaded-audit-v1', source_directory=str(root),
        complete_episodes=len(episodes), independent_test_geometry_groups=metrics['test_geometry_groups'],
        split_sizes={k: len(v) for k, v in protocol['scene_splits'].items()},
        updates=len(history), best_update=metrics['policy_checkpoint']['update'],
        training_world_size=training['world_size'], transitions_per_update=training['transitions_per_update'],
        recorded_update_seconds=sum(r['seconds'] for r in history),
        validation_history=[{'update': r['update'], 'score': r['validation_score']}
                            for r in history if r['validation_score'] is not None],
        groups=groups, reported_paired_intervals={b: g['paired_change_vs_baseline']
                                                for b, g in metrics['groups'].items()},
        sensitivity=sensitivity(root, protocol, metrics))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Audited {len(episodes)} episodes; saved {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.results, args.output)
