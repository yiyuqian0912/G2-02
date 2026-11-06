import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rind_phase1.active_rl.core import Window, belief_metrics
from rind_phase1.active_rl.frontier import FrontierEnv, make_policy, boundary_actions
from rind_phase1.active_rl.locator import Belief, FusionConflictError, policy_inputs
from rind_phase1.active_rl.experiment import metric_rows, ppo_update, evaluate_frontier_rows, episode, summarize
from rind_phase1.model import SourceEnergyField


def locator():
    axis = (torch.arange(8) + .5) * 16
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    return SimpleNamespace(grid=8, world=128, tile=16, spacing=16, device=torch.device('cpu'),
                           xy=torch.stack((xx, yy), -1).reshape(-1, 2),
                           prior=torch.full((64,), -np.log(64)),
                           dataset=SimpleNamespace(_base=SimpleNamespace(drivers=np.array([[[32., 32.]], [[72., 72.]]]))))


class FrontierTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_all_four_neighbour_frontier_budget_and_no_repeats(self):
        env = FrontierEnv(128, 16, Window(32, 32, 16), 64 * 256, 64)
        self.assertEqual(env.spent, 256)
        self.assertEqual(env.frontier, {10, 17, 19, 26})
        rng = np.random.default_rng(9)
        while not env.done:
            old = set(np.flatnonzero(env.coverage))
            action = env.baseline('random', rng)
            tile = action - 1
            self.assertTrue(any(j in old for j in env.neighbours(tile)))
            self.assertEqual(env.step(action), [tile])
            self.assertEqual(env.spent, int(env.coverage.sum()) * 256)
            expected = {j for i in np.flatnonzero(env.coverage) for j in env.neighbours(i)
                        if not env.coverage.flat[j]}
            self.assertEqual(env.frontier, expected)
            self.assertFalse(env.legal()[1][tile + 1])
        self.assertEqual(env.reason, 'budget')
        self.assertTrue(env.coverage.all())
        with self.assertRaises(ValueError): env.step(1)

    def test_corner_and_illegal_jump_or_revisit(self):
        env = FrontierEnv(128, 16, Window(0, 0, 16), 2048, 8)
        self.assertEqual(env.frontier, {1, 8})
        for action in (1, 64, -1):
            with self.assertRaises(ValueError): env.step(action)
        self.assertEqual(env.spent, 256)

    def test_minimum_uses_original_fields_and_is_order_independent(self):
        loc = locator()
        fields = [torch.tensor([.8, .1, .1] + [0.] * 61),
                  torch.tensor([.4, .3, .3] + [0.] * 61),
                  torch.tensor([.2, .7, .1] + [0.] * 61)]
        answers = {(0, i): (p.log(), torch.zeros(1, 1), True) for i, p in enumerate(fields)}
        a = Belief(loc, method='minimum')
        b = Belief(loc, method='minimum')
        for tile in (0, 1, 2): a.update(0, [tile], answers)
        for tile in (2, 0, 1): b.update(0, [tile], answers)
        expected = torch.stack(fields).amin(0)
        expected /= expected.sum()
        torch.testing.assert_close(a.probability, expected)
        torch.testing.assert_close(a.probability, b.probability, rtol=0, atol=0)
        before = a.probability.clone()
        a.update(0, [0, 1], answers)
        torch.testing.assert_close(before, a.probability, rtol=0, atol=0)
        self.assertEqual(a.informative_tiles, 3)
        self.assertEqual(len(a.logs), 0)  # no per-window 4096-vector history needed

    def test_hard_zero_cannot_revive_and_empty_intersection_is_an_error(self):
        loc = locator()
        a = Belief(loc, method='minimum')
        p = torch.tensor([0., .4, .6] + [0.] * 61)
        q = torch.tensor([.9, .05, .05] + [0.] * 61)
        answers = {(0, 0): (p.log(), torch.zeros(1, 1), True),
                   (0, 1): (q.log(), torch.zeros(1, 1), True)}
        a.update(0, [0, 1], answers)
        self.assertEqual(float(a.probability[0]), 0)
        before = a.probability.clone()
        bad = torch.zeros(64); bad[0] = 1
        with self.assertRaises(FusionConflictError):
            a.update(0, [2], {(0, 2): (bad.log(), torch.zeros(1, 1), True)})
        self.assertNotIn(2, a.seen)
        torch.testing.assert_close(a.probability, before, rtol=0, atol=0)

    def test_constant_observations_are_neutral_not_uniform_caps(self):
        loc = locator(); a = Belief(loc, method='minimum')
        p = torch.full((64,), .2 / 63); p[7] = .8
        a.update(0, [0], {(0, 0): (p.log(), torch.ones(1, 1) * .5, True)})
        before = a.probability.clone()
        a.update(0, [1], {(0, 1): (loc.prior, torch.ones(1, 1), False)})
        torch.testing.assert_close(a.probability, before, rtol=0, atol=0)
        self.assertIn(1, a.seen)
        self.assertEqual(a.informative_tiles, 1)

    def test_frontier_features_do_not_read_truth_and_policy_uses_spatial_content(self):
        class Context:
            @property
            def truth(self): raise AssertionError('source leak')
            @property
            def scene(self): raise AssertionError('scene identity leak')
        loc = locator(); c = Context()
        c.env = FrontierEnv(128, 16, Window(32, 32, 16), 4096)
        c.belief = Belief(loc, method='minimum')
        inputs = policy_inputs([c], loc)
        policy = make_policy(dict(environment='frontier', hidden=32))
        d, value = policy(*inputs)
        self.assertEqual(d.probs.shape, (1, 65))
        self.assertTrue((d.probs[~inputs[-1]] == 0).all())
        changed = [x.clone() for x in inputs]
        changed[0][0, 2, 2, 1] = 1
        other, _ = policy(*changed)
        self.assertGreater(float((d.probs - other.probs).abs().max().detach()), 1e-6)
        action = boundary_actions([c], loc)[0]
        self.assertIn(action - 1, c.env.frontier)
        self.assertTrue(torch.isfinite(value).all())

    def test_batched_metrics_match_scalar_and_variable_action_ppo(self):
        loc = locator(); contexts = []
        for i in range(2):
            b = Belief(loc, method='minimum'); b.probability = torch.rand(64).softmax(0)
            contexts.append(SimpleNamespace(scene=i, env=FrontierEnv(128, 16, Window(32, 32, 16), 4096), belief=b))
        rows = metric_rows(contexts, loc)
        for c, row in zip(contexts, rows):
            expected = belief_metrics(c.belief.probability, loc.xy, loc.dataset._base.drivers[c.scene, 0], 16)
            for key in row:
                self.assertAlmostEqual(row[key], expected[key], delta=2e-5)
        model = make_policy(dict(environment='frontier', hidden=32)); inputs = policy_inputs(contexts, loc)
        with torch.no_grad(): distribution, value = model(*inputs); action = distribution.sample(); lp = distribution.log_prob(action)
        buffer = dict(zip(('image', 'scalar', 'actions', 'legal'), inputs))
        buffer.update(action=action, logp=lp, value=value, advantage=torch.tensor([1., -1.]), returns=value + .5)
        config = dict(ppo_epochs=2, minibatch_size=2, clip_ratio=.2, value_weight=.5,
                      action_entropy_bonus=.01, grad_clip=.5, target_kl=.1)
        result = ppo_update(model, buffer, torch.optim.Adam(model.parameters(), lr=.001), config, 1)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_parallel_evaluation_matches_serial_and_all_baselines_are_legal(self):
        loc = locator()
        def query(requests):
            answers = {}
            for s, tile in requests:
                p = torch.sin(torch.arange(64).float() + tile + s).log_softmax(0)
                answers[(s, tile)] = (p, torch.full((1, 1), .5), True)
            return answers
        loc.query = query
        config = dict(environment='frontier', fusion_method='minimum', tile_size=16, spacing=16,
                      fusion_discount=1., max_steps=16, evaluation_budgets=[1024, 2048], budget=2048,
                      evaluation_envs=8, evaluation_curve_stride=2, seed=17, num_envs=2, hidden=32,
                      bootstrap_repetitions=10, plot_scenes=0)
        protocol = dict(scene_splits={'test': [0, 1]}, initial_windows={'0': [32, 32, 16], '1': [64, 64, 16]},
                        geometry_groups={'0': 'a', '1': 'b'}, source_epoch=1, source_sha256='test', inference_fusion='minimum')
        model = make_policy(config).eval()
        rows = evaluate_frontier_rows(model, protocol, config, loc, 0, 1, split='test', baselines=True, initial=model)
        self.assertEqual(len(rows), 20)
        for row in rows:
            seed = int(np.random.SeedSequence([17, row['scene_id'], row['budget'], 19]).generate_state(1)[0])
            serial = episode(model, row['policy'], row['scene_id'], row['budget'], protocol, config, loc, seed)
            self.assertAlmostEqual(row['mass32'], serial['mass32'], delta=1e-6)
            self.assertEqual(row['spent'], serial['spent'])
            self.assertEqual(len(row['acquired_tiles']), len(set(row['acquired_tiles'])))
        with tempfile.TemporaryDirectory() as temporary:
            report = summarize(rows, config, Path(temporary), protocol, {'update': 1})
            self.assertEqual(report['fusion_method'], 'minimum')
            self.assertEqual(set(report['groups']['1024']['policies']), {'ppo', 'random', 'compact', 'boundary', 'initial_policy'})

    def test_frontier_deployment_uses_only_requested_sensor_blocks(self):
        from rind_phase1.active_rl.deploy import UnknownFieldAgent
        from rind_phase1.active_rl.prepare import sha
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            mc=dict(hidden=16,fourier_frequencies=2,coordinate_scale=128.,use_obstacle=False,
                    pixel_coordinates=True,normalization_support='world')
            source=root/'phase1.pt';m=SourceEnergyField(**mc)
            torch.save(dict(interface_version='phase1-v1',model_config=mc,model=m.state_dict(),epoch=0),source)
            config=dict(environment='frontier',fusion_method='minimum',hidden=32,tile_size=16,
                        spacing=16,candidate_chunk=64,locator_batch_size=4,max_steps=8,fusion_discount=1.)
            model=make_policy(config);policy=root/'policy.pt'
            torch.save(dict(config=config,policy=model.state_dict(),locator_snapshot_sha256=sha(source)),policy)
            agent=UnknownFieldAgent(policy,source,[32,32,16],2048)
            first=agent.request();self.assertEqual(first['new_pixels'],256)
            self.assertIsNone(agent.locator.dataset)
            agent.observe({i:np.zeros((16,16),dtype=np.float32) for i in first['tiles']})
            second=agent.request();self.assertIsNotNone(second)
            self.assertEqual(second['new_pixels'],256)
            self.assertTrue(set(second['tiles']).isdisjoint(agent.context.belief.seen))
            observed=np.zeros((16,16),dtype=np.float32);observed[:,:8]=1
            probability=agent.observe({i:observed for i in second['tiles']})
            self.assertEqual(agent.env.spent,512)
            self.assertAlmostEqual(float(probability.sum()),1.,places=6)
            self.assertEqual(agent.context.belief.informative_tiles,1)


if __name__ == '__main__': unittest.main()
