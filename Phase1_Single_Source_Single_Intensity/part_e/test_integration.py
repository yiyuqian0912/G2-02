import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from .adaptive import adaptive_search, teacher_probabilities
from .physical_cost import costs, boundary_weights
from .pipeline import execute


class NewIntegrationTests(unittest.TestCase):
    def grid(self):
        xx,yy = np.meshgrid(np.arange(8)+.5,np.arange(8)+.5)
        return SimpleNamespace(candidate_xy=np.column_stack([xx.ravel(),yy.ravel()]),
                               grid_index=np.arange(64),x_axis=np.arange(8))

    def evaluator(self):
        class Evaluator:
            def evaluate(self,xy):
                return SimpleNamespace(valid=True,physical_cost=float(min(np.sum((xy-[1.5,1.5])**2),
                    np.sum((xy-[6.5,6.5])**2))),num_physics_evaluations=1,cache_hit=False)
        return Evaluator()

    def test_full_retention_matches_uniform(self):
        grid = self.grid()
        result = adaptive_search(grid,self.evaluator(),budget=64,retain_fraction=1.)
        self.assertTrue(result.complete)
        self.assertEqual(result.num_physics_evaluations,64)
        np.testing.assert_array_equal(result.candidate_xy,grid.candidate_xy)
        self.assertAlmostEqual(teacher_probabilities(result,temperature=.1).sum(),1.)

    def test_small_budget_cannot_be_teacher(self):
        result = adaptive_search(self.grid(),self.evaluator(),budget=1)
        self.assertFalse(result.complete)
        self.assertLessEqual(result.num_physics_evaluations,1)
        with self.assertRaises(ValueError): teacher_probabilities(result,temperature=.1)

    def test_partial_block_points_excluded(self):
        result = adaptive_search(self.grid(),self.evaluator(),budget=8,retain_fraction=.5,exploration_fraction=0.)
        self.assertFalse(result.complete)
        self.assertEqual(result.covered.sum(),0)
        self.assertGreater(len(result.evaluated_indices),0)

    def test_successful_reduced_support(self):
        result = adaptive_search(self.grid(),self.evaluator(),budget=64,retain_fraction=.5,exploration_fraction=0.)
        self.assertTrue(result.complete)
        self.assertLess(result.num_physics_evaluations,64)
        self.assertEqual(len(result.candidate_xy),result.covered.sum())

    def test_zero_edge_disabled_distinct_from_zero_measured(self):
        image = np.eye(8)
        self.assertIsNone(costs(image,image)['L_edge'])
        self.assertEqual(costs(image,image,beta=1.)['L_edge'],0.)

    def test_missing_edges_penalized(self):
        observed = np.zeros((8,8)); observed[:,4:] = 1
        self.assertEqual(costs(observed,np.zeros((8,8)),beta=1.)['L_edge'],1.)

    def test_weights_observation_only_and_response_baseline(self):
        image = np.zeros((8,8)); image[:,4:] = 1
        weights = boundary_weights(image,boundary_lambda=2.,boundary_sigma=1.)
        self.assertTrue((weights>=1).all())
        self.assertAlmostEqual(costs(image,np.zeros((8,8)))['L_resp'],.5)

    def test_pipeline_records_actual_blocker(self):
        def missing(cfg): raise FileNotFoundError('missing physical data')
        with tempfile.TemporaryDirectory() as output:
            result = execute({},output,missing,None)
            self.assertFalse(result['full_pipeline_complete'])
            self.assertEqual(result['stages'][0]['status'],'blocked')


if __name__ == '__main__': unittest.main()
