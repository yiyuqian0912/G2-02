import unittest
from pathlib import Path
import tempfile

import numpy as np

from .active_policies import choose_action
from .observation_env import Action, BudgetedWindowEnv, STOP, Window, geometric_actions


class ObservationTests(unittest.TestCase):
    def make_env(self, budget=40, max_steps=20, oracle=None):
        self.calls = []
        self.field = np.arange(64).reshape(8, 8).astype(float)
        def acquire(w):
            self.calls.append(w)
            return self.field[w.slices].copy()
        return BudgetedWindowEnv(oracle or acquire, world_size=8, budget=budget,
                                sizes=(2, 4, 8), max_steps=max_steps)

    def test_initial_count_and_unknown_no_alias(self):
        env = self.make_env()
        old = env.reset(Window(2, 2, 4))
        self.assertEqual(old.spent, 16)
        self.assertTrue(np.isnan(old.response[~old.coverage]).all())
        self.assertFalse(old.response.flags.writeable)
        env.step(Action("right", Window(4, 2, 4)))
        self.assertEqual(old.spent, 16)
        self.assertEqual(old.coverage.sum(), 16)
        self.assertTrue(np.isnan(old.response[2, 6]))
        # Even overriding the flag only changes the detached snapshot.
        old.response.flags.writeable = True
        old.response[:] = 999
        self.assertNotEqual(env.snapshot().response[2, 2], 999)

    def test_overlap_and_zero_cost_reposition(self):
        env = self.make_env()
        env.reset(Window(0, 0, 4))
        state = env.step(Action("right", Window(2, 0, 4)))
        self.assertEqual(state.spent, 24)
        state = env.step(Action("left", Window(0, 0, 4)))
        self.assertEqual(state.spent, 24)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(state.history[-1].new_pixels, 0)
        self.assertFalse(state.history[-1].oracle_called)

    def test_growth_containment_no_clipping(self):
        env = self.make_env(budget=64)
        env.reset(Window(0, 0, 4))
        legal = env.legal_actions()
        growth = [a for a in legal if a.name.startswith("grow")]
        self.assertEqual(growth, [Action("grow_se", Window(0, 0, 8))])
        state = env.step(growth[0])
        self.assertEqual(state.spent, 64)
        self.assertEqual(state.history[-1].new_pixels, 48)
        self.assertEqual(state.termination_reason, "budget_exhausted")

    def test_unaffordable_and_teleport_do_not_query(self):
        env = self.make_env(budget=20)
        env.reset(Window(0, 0, 4))
        for action in (Action("right", Window(2, 0, 4)), Action("teleport", Window(4, 4, 4))):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(env.snapshot().spent, 16)

    def test_bad_initial_budget_does_not_query(self):
        env = self.make_env(budget=4)
        with self.assertRaises(ValueError):
            env.reset(Window(0, 0, 4))
        self.assertEqual(self.calls, [])

    def test_bad_oracle_does_not_commit(self):
        for bad in (np.nan, -1.):
            env = self.make_env(oracle=lambda w: np.full((w.size, w.size), bad))
            with self.assertRaises(ValueError):
                env.reset(Window(0, 0, 4))
            self.assertEqual(env._spent, 0)
            self.assertFalse(env._coverage.any())
        env = self.make_env()
        before = env.reset(Window(0, 0, 4))
        env._oracle = lambda w: np.full((w.size, w.size), 100.)
        with self.assertRaisesRegex(ValueError, "overlap changed"):
            env.step(Action("right", Window(2, 0, 4)))
        np.testing.assert_array_equal(env.snapshot().response, before.response)
        self.assertEqual(env.snapshot().spent, 16)

    def test_lit_dark_and_sources_do_not_filter_acquisitions(self):
        for value in (0., .8, 2.5):
            env = self.make_env(oracle=lambda w: np.full((w.size, w.size), value))
            state = env.reset(Window(0, 0, 4))
            self.assertEqual(state.spent, 16)
            self.assertTrue((state.response[state.coverage] == value).all())

    def test_stop_and_step_limit(self):
        env = self.make_env(max_steps=1)
        env.reset(Window(0, 0, 4))
        state = env.step(Action("right", Window(2, 0, 4)))
        self.assertEqual(state.termination_reason, "max_steps")
        self.assertEqual(env.legal_actions(), ())
        with self.assertRaises(RuntimeError):
            env.step(STOP)
        env = self.make_env()
        env.reset(Window(0, 0, 4))
        state = env.step(STOP)
        self.assertEqual(state.spent, 16)
        self.assertEqual(state.termination_reason, "policy_stop")

    def test_deterministic_policy_and_no_future_dependence(self):
        outputs = []
        for hidden in (0., 1000.):
            env = self.make_env()
            self.field[:] = hidden
            self.field[:4, :4] = np.arange(16).reshape(4, 4)
            state = env.reset(Window(0, 0, 4))
            actions = env.legal_actions()
            outputs.append([choose_action(state, actions, policy=p,
                            rng=np.random.default_rng(10)) for p in ("random", "grow", "boundary")])
        self.assertEqual(*outputs)

    def test_all_policies_preserve_budget(self):
        for policy in ("random", "grow", "boundary"):
            env = self.make_env(budget=40)
            state = env.reset(Window(2, 2, 2))
            rng = np.random.default_rng(1)
            while not state.terminated:
                action = choose_action(state, env.legal_actions(), policy=policy, rng=rng)
                self.assertIn(action, env.legal_actions())
                state = env.step(action)
                self.assertEqual(state.spent, state.coverage.sum())
                self.assertLessEqual(state.spent, 40)
            self.assertLessEqual(state.steps, 20)

    def test_free_reposition_to_affordable_frontier(self):
        env = self.make_env(budget=19, max_steps=40)
        env.reset(Window(0, 0, 2))
        # Fill a path and return; the next frontier can require cached moves.
        for x in (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0):
            name = "right" if x > env.snapshot().window.x else "left"
            env.step(Action(name, Window(x, 0, 2)))
        env.step(Action("down", Window(0, 1, 2)))
        env.step(Action("up", Window(0, 0, 2)))
        state = env.snapshot()
        self.assertEqual(state.spent, 18)
        action = choose_action(state, env.legal_actions(), policy="boundary", rng=np.random.default_rng(1))
        self.assertNotEqual(action, STOP)
        self.assertIn(action, env.legal_actions())
        self.assertEqual(state.new_pixels(action.window), 0)

    def test_invalid_configuration(self):
        for kwargs in ({"budget": 65}, {"budget": True}, {"sizes": (3,)},
                       {"sizes": (4, 2)}, {"max_steps": 0}, {"overlap_atol": np.nan}):
            base = dict(world_size=8, budget=32, sizes=(2, 4, 8))
            base.update(kwargs)
            with self.assertRaises(ValueError):
                BudgetedWindowEnv(lambda w: np.zeros((w.size, w.size)), **base)

    def test_rind_adapter_preserves_addition_and_hides_channels(self):
        from dataset_update.test_dataset_union import make_dataset
        from dataset_update import open_dataset
        from .demo import RindResponseOracle
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            make_dataset(root, [1.])
            counts = np.load(root / "driver_counts.npy")
            counts[0] = 2
            np.save(root / "driver_counts.npy", counts)
            strengths = np.load(root / "driver_strengths.npy")
            strengths[0] = [.3, .5]
            np.save(root / "driver_strengths.npy", strengths)
            bits = np.load(root / "visibility_bits.npy")
            bits[0, 1] = bits[0, 0]
            np.save(root / "visibility_bits.npy", bits)
            dataset = open_dataset(root)
            result = RindResponseOracle(dataset, 0)(Window(0, 0, 4))
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result[:, :2], 0.)
            np.testing.assert_array_equal(result[:, 2:], .8)


if __name__ == "__main__":
    unittest.main()
