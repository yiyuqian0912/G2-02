import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from . import checks
from .adapters import to_thomas_record
from .record_io import save
from .evaluate import (evaluate_record, distributions, components, edge_ablation,
                       search_comparison, unseen_scene_check, unseen_obstacle_combinations,
                       aggregate, run_manifest)


def fixture(scene=2, size=16, edge=False):
    xy = np.array([[8.,8.],[24.,8.],[40.,8.],[56.,8.]])
    valid = np.array([True, True, False, True])
    cost = np.array([0., .1, np.inf, 0.])
    if edge:
        cost = np.array([0., .2, np.inf, .1])
    w = np.exp(-cost[valid]/.1)
    q = np.zeros(4)
    q[valid] = w/w.sum()
    settings = {'dataset': {'version': 'synthetic-fixture'},
                'support': {'world_size': 1024, 'candidate_spacing': 16,
                            'candidate_domain': 'synthetic_subset'},
                'physics': {'alpha': 1., 'beta': 1. if edge else 0.,
                            'boundary_lambda': 0., 'edge_computed': edge},
                'search_mode': 'uniform'}
    return {'scene_id': scene, 'view_id': 0, 'candidate_xy': xy, 'valid': valid,
            'evaluated': np.ones(4, dtype=bool), 'physical_cost': cost,
            'teacher_prob': q, 'temperature': .1, 'window': np.array([128,128,size]),
            'config_id': 'synthetic-edge' if edge else 'synthetic-response',
            'metadata': {'complete': True, 'settings': settings,
                         'num_physics_evaluations': 3, 'elapsed_seconds': .01}}



class HandoffTests(unittest.TestCase):
    def test_invalid_infinity_is_permitted(self):
        checks.teacher(fixture())

    def test_bad_probability_not_silently_normalized(self):
        record = fixture()
        record['teacher_prob'] *= 2
        with self.assertRaises(ValueError): checks.teacher(record)

    def test_invalid_mass_rejected(self):
        record = fixture()
        record['teacher_prob'] = np.ones(4)/4
        with self.assertRaises(ValueError): checks.teacher(record)

    def test_incomplete_rejected(self):
        for field in ('evaluated', 'complete'):
            record = fixture()
            if field == 'evaluated': record[field][1] = False
            else: record['metadata'][field] = False
            with self.assertRaises(ValueError): checks.teacher(record)

    def test_no_valid_candidates_rejected(self):
        record = fixture()
        record['valid'][:] = False
        with self.assertRaises(ValueError): checks.teacher(record)

    def test_probability_cost_mismatch(self):
        record = fixture()
        record['physical_cost'][0] = 1
        with self.assertRaises(ValueError): checks.teacher(record)

    def test_duplicate_coordinates(self):
        record = fixture()
        record['candidate_xy'][1] = record['candidate_xy'][0]
        with self.assertRaises(ValueError): checks.teacher(record)

    def test_misaligned_order(self):
        record = fixture()
        prediction = copy.deepcopy(record)
        prediction['candidate_xy'] = prediction['candidate_xy'][::-1]
        with self.assertRaises(ValueError): checks.aligned(record,prediction)

    def test_leaked_scene(self):
        with self.assertRaises(ValueError): checks.scene_splits({'train':[0], 'validation':[1], 'test':[0]})

    def test_adapter_preserves_coordinates_and_probability(self):
        record = fixture()
        sample = {**record, 'response':np.zeros((16,16))}
        converted = to_thomas_record(record,sample)
        self.assertEqual(converted['response'].shape,(1,16,16))
        self.assertEqual(converted['window'],{'x0':128,'y0':128,'L':16})
        np.testing.assert_array_equal(converted['candidate_xy'],record['candidate_xy'])
        np.testing.assert_array_equal(converted['teacher_prob'],record['teacher_prob'])
        self.assertNotIn('area_weight', converted)

    def test_safe_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'record.npz'
            save(fixture(),path)
            checks.teacher(checks.load_record(path))


class EvaluationTests(unittest.TestCase):
    def test_perfect_student_zero_error(self):
        record = fixture()
        prediction = {**record, 'student_prob':record['teacher_prob']}
        result = evaluate_record(record,prediction,threshold=.01,spacing=16)
        self.assertAlmostEqual(result['forward_kl'],0)
        self.assertEqual(result['l1_distance'],0)
        self.assertEqual(result['sampled_compatible_components'],2)

    def test_physical_expectation_excludes_invalid_inf(self):
        result = evaluate_record(fixture(),threshold=.01)
        self.assertTrue(np.isfinite(result['teacher_expected_physical_cost']))

    def test_zero_prediction_not_clipped(self):
        result = distributions(np.array([.5,.5]),np.array([1.,0.]))
        self.assertIsNone(result['forward_kl'])
        self.assertTrue(result['forward_kl_infinite'])

    def test_pruned_modes_count_as_misses(self):
        result = search_comparison(fixture(),np.array([True,False,False,False]),
            threshold=.01,adaptive_evaluations=1,adaptive_seconds=.01)
        self.assertEqual(result['compatible_cell_recall'],.5)

    def test_fake_coverage_rejected(self):
        with self.assertRaises(ValueError): search_comparison(fixture(),np.ones(4),
            threshold=.01,adaptive_evaluations=1,adaptive_seconds=.01)

    def test_unseen_protocol(self):
        result = unseen_scene_check([fixture()],{'train':[0], 'validation':[1], 'test':[2]},[0])
        self.assertTrue(result['verified'])
        with self.assertRaises(ValueError): unseen_scene_check([fixture()],{'train':[0], 'validation':[1], 'test':[2]},[2])

    def test_edge_ablation_guards(self):
        edge_ablation(fixture(),fixture(edge=True),threshold=.01)
        with self.assertRaises(ValueError): edge_ablation(fixture(),fixture(),threshold=.01)

    def test_non_lattice_topology_rejected(self):
        record = fixture()
        record['candidate_xy'][0,0] += .2
        with self.assertRaises(ValueError): components(record['candidate_xy'],record['valid'],16)

    def test_scene_balancing(self):
        result = aggregate([{'scene_id':1,'score':0.}, {'scene_id':1,'score':0.}, {'scene_id':2,'score':1.}])
        self.assertEqual(result['metrics']['score']['scene_mean'],.5)
        self.assertAlmostEqual(result['metrics']['score']['observation_mean'],1/3)

    def test_reconstruction_and_source_free(self):
        class Dataset:
            manifest = {'global_size':1024}
            def get_observation(self,*args):
                return {'window':np.array([0,0,16]),'response':np.ones((16,16))}
            def rerender(self,*args,**kwargs): return np.ones((16,16))
        checks.physical_reconstruction(Dataset(),0,0,[24,24])
        with self.assertRaises(ValueError): checks.physical_reconstruction(Dataset(),0,0,[8,8])


if __name__ == '__main__':
    unittest.main()
