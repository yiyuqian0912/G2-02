"""Layer sequencing, disconnected equal levels and no truth-label shortcuts."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from rind_phase2.layered_teacher import generate_layered_teacher, NoLayeredSolutionError
from rind_phase2.teacher import save_teacher_record, load_teacher_record
from rind_phase2.search import EvaluationBudgetExceeded
from rind_phase2.regions import response_regions


class ToyObservation:
    def __init__(self, root, response, masks):
        self.root = Path(root)
        self.response = np.asarray(response, dtype=float)
        self.masks = masks
        self.window = np.array([2, 2, len(response)])
        self.manifest = dict(global_size=8, schema_version=3)
    def get_observation(self, *_):
        return dict(response=self.response.astype(np.float32),window=self.window,
                    obstacle=np.zeros_like(self.response,dtype=bool))
    def get_region(self, *_):
        return dict(response=self.response.copy(),obstacle=np.zeros_like(self.response,dtype=bool))
    def get_scene(self, *_):
        return dict(obstacle_types=np.zeros(0,np.uint8),obstacle_params=np.zeros((0,17)),obstacle_count=0)
    def rerender(self, scene, window, sources, **kwargs):
        assert len(sources)==1 and sources[0][2]==1
        return self.masks.get(tuple(sources[0][:2]),np.zeros_like(self.response))
    def get_source_params(self, *_):raise AssertionError('Forbidden truth label API')
    get_channels=get_candidates=get_source_params


class LayeredTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
    def generate(self,ds,**kwargs):
        args=dict(spacing=2,temperature=.01,seed=9,max_final_mae=1e-12,backend='reference')
        args.update(kwargs)
        return generate_layered_teacher(ds,0,0,**args)
    def test_user_example_minimum_neighbors_subtract_then_merge(self):
        a=np.array([[1,1],[1,0]],bool);b=np.array([[0,1],[0,1]],bool)
        ds=ToyObservation(self.tmp.name,.3*a+.5*b,{(1.,1.):a,(7.,7.):b})
        rec=self.generate(ds)
        self.assertEqual(rec['source_counts'].tolist(),[2])
        np.testing.assert_allclose(rec['physical_cost'],0,atol=1e-14)
        trace=rec['metadata']['layer_traces'][0]
        self.assertAlmostEqual(trace[0]['anchor_intensity'],.3)
        self.assertEqual(len(trace[0]['selected_neighbor_ids']),1)
        self.assertAlmostEqual(trace[1]['anchor_intensity'],.5)
        self.assertEqual(sum(n for _,n in trace[1]['anchor_runs']),2)
        path=Path(self.tmp.name)/'teacher.npz';save_teacher_record(rec,path)
        reread=load_teacher_record(path)
        self.assertEqual(reread['metadata']['layer_traces'],rec['metadata']['layer_traces'])
    def test_equal_disconnected_regions_not_merged_and_remote_light_rejected(self):
        a=np.zeros((4,4),bool);a[1:3,0]=1
        b=np.zeros((4,4),bool);b[1:3,3]=1
        ds=ToyObservation(self.tmp.name,.4*(a+b),{(1.,1.):a,(7.,7.):b,(1.,7.):a|b})
        rec=self.generate(ds,samples_per_pattern=8)
        self.assertTrue((rec['source_counts']==2).all())
        self.assertGreater(rec['metadata']['layered_search']['rejected_final_reconstruction'],0)
        for trace in rec['metadata']['layer_traces']:
            self.assertEqual(len(trace),2)
            self.assertTrue(trace[0]['same_intensity_other_region_ids'])
            self.assertEqual(sum(n for _,n in trace[0]['removed_runs']),2)
            self.assertEqual(sum(n for _,n in trace[1]['removed_runs']),2)
        self.assertEqual(rec['metadata']['num_source_renders'],rec['metadata']['layered_search']['num_source_renders'])
    def test_over_one_is_explicit_failure_not_source_splitting(self):
        ds=ToyObservation(self.tmp.name,np.full((2,2),1.2),{})
        with self.assertRaises(NoLayeredSolutionError) as ctx:self.generate(ds)
        self.assertEqual(ctx.exception.report['rejected_intensity_above_one'],1)
    def test_dark_input_has_no_fabricated_source(self):
        ds=ToyObservation(self.tmp.name,np.zeros((2,2)),{})
        with self.assertRaises(NoLayeredSolutionError):self.generate(ds)
    def test_no_local_candidate_and_source_limit_and_budget(self):
        a=np.array([[1,1],[1,0]],bool);b=np.array([[0,1],[0,1]],bool)
        ds=ToyObservation(self.tmp.name,.3*a+.5*b,{(1.,1.):a,(7.,7.):b})
        with self.assertRaises(NoLayeredSolutionError) as ctx:self.generate(ds,max_sources=1)
        self.assertGreater(ctx.exception.report['incomplete_at_source_limit'],0)
        with self.assertRaises(EvaluationBudgetExceeded):self.generate(ds,max_source_renders=0)
        ds.masks={}
        with self.assertRaises(NoLayeredSolutionError) as ctx:self.generate(ds)
        self.assertGreater(ctx.exception.report['rejected_no_local_candidates'],0)
    def test_determinism_and_validation(self):
        a=np.ones((2,2),bool);ds=ToyObservation(self.tmp.name,.3*a,{(1.,1.):a,(7.,7.):a})
        r1=self.generate(ds);r2=self.generate(ds)
        np.testing.assert_array_equal(r1['sources'],r2['sources'])
        np.testing.assert_array_equal(r1['teacher_prob'],r2['teacher_prob'])
        for kw in ({'beam_width':0},{'temperature':0},{'local_mismatch_tolerance':.5}):
            with self.assertRaises(ValueError):self.generate(ds,**kw)
    def test_thin_obstacle_does_not_join_same_response_regions(self):
        t=np.array([0],np.uint8);p=np.zeros((1,17));p[0,:5]=[1,.5,.1,.4,0]
        r=response_regions(np.ones((1,2)),np.zeros((1,2),bool),(0,0),t,p)
        self.assertEqual(len(r.values),2);self.assertEqual(r.neighbors,[[],[]])
        self.assertEqual(r.excluded_edges,1)

if __name__=='__main__':unittest.main()
