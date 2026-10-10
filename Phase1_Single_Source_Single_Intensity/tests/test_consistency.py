"""Scientific regressions for public support, stable splits and dense diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from rind_phase1.data import PROJECT_ROOT, Phase1Dataset
from rind_phase1.model import SourceEnergyField
from rind_phase1.predict import predict
from rind_phase1.protocol import fixed_selection, validate_research_config
from rind_phase1.train import run_epoch, TeacherRecords
from rind_phase1.interfaces import training_record
from test_interfaces import fixture


class ConsistencyTests(unittest.TestCase):
    def dataset(self):
        try:
            return Phase1Dataset()
        except FileNotFoundError:
            self.skipTest('Install RIND data to run this integration check')

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_public_invalid_mass_and_explicit_geometry_ablation(self):
        from part_e.evaluate import evaluate_record
        from rind_phase1.evaluate import validate_common_support
        teacher, observation = fixture()
        model = SourceEnergyField(hidden=8, fourier_frequencies=2, use_obstacle=False,
                                  pixel_coordinates=True, normalization_support='world')
        p = predict(model, teacher, observation, chunk=2)
        self.assertGreater(p['student_prob'][~teacher['valid']].sum(), 0)
        validate_common_support(teacher, p)
        row = evaluate_record(teacher, p, threshold=.01)
        self.assertTrue(row['student_expected_physical_cost_infinite'])
        self.assertIsNone(row['student_expected_physical_cost'])
        self.assertGreater(row['student_invalid_mass'], 0)
        self.assertTrue(np.isfinite(row['student_expected_penalized_cost']))
        masked = predict(model, teacher, observation, normalization_support='geometry')
        self.assertEqual(masked['student_prob'][~teacher['valid']].sum(), 0)
        self.assertAlmostEqual(masked['student_prob'].sum(), 1)
        from rind_phase1.evaluate import physical_compatibility_metrics
        invalid_only=np.array([0.,0.,1.,0.])
        metrics=physical_compatibility_metrics(invalid_only,teacher['physical_cost'],teacher['valid'])
        self.assertIsNone(metrics['expected_valid_cost'])
        self.assertEqual(metrics['invalid_probability_mass'],1)

    def test_chunked_training_matches_full_gradient_and_ignores_mask(self):
        teacher, observation = fixture()
        record = training_record(teacher, observation)
        record['support'] = np.ones(4, bool)
        model = SourceEnergyField(hidden=8, fourier_frequencies=2, use_obstacle=False,
                                  pixel_coordinates=True, normalization_support='world')
        other = copy.deepcopy(model)
        a = run_epoch(model, [record], optimizer=torch.optim.SGD(model.parameters(), lr=.01))
        b = run_epoch(other, [record], optimizer=torch.optim.SGD(other.parameters(), lr=.01), candidate_chunk=2)
        self.assertAlmostEqual(a, b, places=6)
        for x, y in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
        changed = {**observation, 'obstacle':np.ones_like(observation['obstacle'])}
        np.testing.assert_allclose(predict(model, teacher, observation)['energy'], predict(model, teacher, changed)['energy'])

    def test_fixed_pools_and_window_selection_survive_training_growth(self):
        c = json.loads((PROJECT_ROOT/'configs/experiment_consistent.json').read_text())
        c['scene_counts'] = dict(train=2, validation=2, test=2)
        dataset = self.dataset()
        before, views, _ = fixed_selection(c, dataset)
        c['scene_counts']['train'] = 10
        after, new_views, _ = fixed_selection(c, dataset)
        self.assertEqual(before['test'], after['test'])
        self.assertEqual(views['validation'], new_views['validation'])
        self.assertEqual(views['train'], new_views['train'][:len(views['train'])])
        self.assertFalse(set(after['train']) & set(after['test']))

    def test_alias_guard(self):
        c = json.loads((PROJECT_ROOT/'configs/experiment_consistent.json').read_text())
        validate_research_config(c)
        c['training']['fourier_frequencies'] = 6
        with self.assertRaisesRegex(ValueError, 'Fourier/sampling mismatch'):
            validate_research_config(c)
        c['teacher']['spacing'] = 16
        validate_research_config(c)

    def test_teacher_records_decode_only_requested_observation(self):
        dataset = self.dataset()
        root = PROJECT_ROOT/'outputs/experiments/train5000/response'
        if not (root/'records.json').exists():
            self.skipTest('No existing experiment')
        paths = json.loads((root/'records.json').read_text())['test'][:3]
        with patch.object(dataset, 'get_observation', wraps=dataset.get_observation) as reader:
            records = TeacherRecords([root/p for p in paths], dataset, 'world')
            self.assertEqual(reader.call_count, 0)
            self.assertEqual(len(records), 3)
            self.assertTrue(records[1]['support'].all())
            self.assertEqual(reader.call_count, 1)

    def test_vector_geometry_matches_reference_including_boundaries(self):
        from rind_phase1.results_browser import geometry_support
        from rind_dataset.geometry import is_driver_valid_mixed
        dataset = self.dataset()
        rng = np.random.default_rng(9)
        for scene_id in (0, 1, 8, 6622):
            scene = dataset.get_scene(scene_id)
            xy = np.vstack([rng.uniform(-1, 1025, (400, 2)), [[0, 0], [1024, 1024]], scene['drivers'][:, :2]])
            types, params, count = scene['obstacle_types'], scene['obstacle_params'], int(scene['obstacle_count'])
            expected = [is_driver_valid_mixed(p, types, params, count, domain_size=1024) for p in xy]
            np.testing.assert_array_equal(geometry_support(xy, types, params, count, 1024), expected)
        # Exact polygon edges/vertices, including a concave interior.
        p = np.zeros((4, 25)); p[0,:5]=[10,10,2,3,0];p[1,:5]=[20,20,2,3,0]
        p[2,:6]=[30,30,35,30,30,35];p[3,:11]=[5,40,40,45,40,42,42,45,45,40,45]
        xy=np.array([[8,10],[12,13],[22,20],[30,30],[32,33],[40,40],[42,42],[43,41],[44,43]],float)
        expected=[is_driver_valid_mixed(x,np.arange(4),p,4,domain_size=100) for x in xy]
        np.testing.assert_array_equal(geometry_support(xy,np.arange(4),p,4,100),expected)

    def test_browser_real_test_only_dense_and_matched(self):
        from rind_phase1.results_browser import create_app
        root = PROJECT_ROOT/'outputs/experiments'
        if not (root/'train5000/response/seed_0/best.pt').exists():
            self.skipTest('No existing checkpoint')
        app = create_app(root)
        explorer = app.extensions['explorer']
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/api/catalog?run=../').status_code, 400)
        args=dict(run='train5000',variant='response',seed=0,index=3,spacing=32,mode='dense',support='world')
        catalog=explorer.catalog(args)
        protocol=json.loads((root/'train5000/protocol.json').read_text())
        self.assertTrue({e['scene_id'] for e in catalog['entries']} <= set(protocol['scene_splits']['test']))
        r, arrays=explorer.compute(args)
        self.assertEqual(arrays['probability'].shape,(32,32))
        self.assertAlmostEqual(arrays['probability'].sum(),1)
        self.assertEqual(len(arrays['teacher_probability']),r['metadata']['teacher_candidates'])
        args['mode']='matched'
        r,arrays=explorer.compute(args)
        self.assertAlmostEqual(arrays['probability'].sum(),1)
        self.assertEqual(arrays['probability'].shape,(16,16))
        self.assertEqual(r['metadata']['spacing'],64)
        self.assertEqual(client.post('/api/jobs',json={**args,'index':-1}).status_code,400)
        self.assertEqual(client.post('/api/jobs',json=args,headers={'Origin':'https://example.com'}).status_code,403)
        with patch.object(explorer,'compute',return_value=(r,arrays)):
            response=client.post('/api/jobs',json=args)
            self.assertEqual(response.status_code,202)
            job_id=response.json['id']
            explorer.executor.shutdown(wait=True)
        self.assertEqual(client.get('/api/jobs/'+job_id).json['status'],'done')
        import io
        with np.load(io.BytesIO(client.get('/api/jobs/'+job_id+'/download').data)) as z:
            np.testing.assert_array_equal(z['probability'],arrays['probability'])

    def test_v2_end_to_end_training_evaluation_and_probe(self):
        from rind_phase1.experiments import plan, prepare, train, evaluation, summarize, sampling_probe
        c=json.loads((PROJECT_ROOT/'configs/experiment_consistent.json').read_text())
        c['scene_counts']=dict(train=1,validation=1,test=1)
        c['window_sizes']=[16]
        c['training'].update(hidden=8,epochs=1,patience=2,num_threads=1,candidate_chunk=64)
        c['evaluation'].update(plot_count=0,probe_observations=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c['output_root']=directory
            dataset=self.dataset();protocol=plan(c,root,dataset)
            prepare(c,root,dataset,protocol,'response')
            train(c,root,dataset,protocol,'response',0)
            report=evaluation(c,root,dataset,protocol,'response',0)
            self.assertEqual(report['normalization_support'],'world')
            self.assertFalse(report['use_obstacle'])
            self.assertIn('student_invalid_mass',report['aggregate']['metrics'])
            summarize(c,root,'response')
            sampling_probe(c,root,dataset,protocol,'response',0)
            probe=json.loads((root/'response/seed_0/sampling_probe.json').read_text())
            self.assertEqual(probe['split'],'validation')
            self.assertLess(probe['probe_spacing'],probe['training_spacing'])
            import importlib.util
            if importlib.util.find_spec('matplotlib'):
                from scripts.report_experiment import main as report_main
                with patch('sys.argv', ['report_experiment.py', '--root', str(root)]):
                    report_main()
                self.assertTrue((root/'learning_curves.png').is_file())
                self.assertIn('有限惩罚代价', (root/'RESULTS.zh-CN.md').read_text())


if __name__ == '__main__':
    unittest.main()
