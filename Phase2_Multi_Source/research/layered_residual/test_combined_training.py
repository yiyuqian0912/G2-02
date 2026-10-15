import numpy as np
import unittest
import json
import tempfile
from pathlib import Path

from types import SimpleNamespace

from layered_residual import (combine_layer_probabilities, _phase1_layer_teacher,
                              _phase1_layers_from_one_uniform_pass,
                              screen_candidates_by_observed_shadow)
from train_combined import _window_class,read_config
from visualize_combined import layer_peeling_arrays


class ToyDataset:
    def __init__(self,response,obstacle=None):
        self.response=np.asarray(response,dtype=np.float64)
        self.obstacle=np.zeros_like(self.response,dtype=bool) if obstacle is None else np.asarray(obstacle,dtype=bool)

    def get_observation(self,scene,view):
        return {'window':np.array([0,0,len(self.response)]),'obstacle':self.obstacle}

    def get_region(self,scene,x,y,size):
        return {'response':self.response}


class CombinedTargetTests(unittest.TestCase):
    def test_full_record_count_mode_is_explicit(self):
        config={'data_root':'data','output_root':'out','split_seed':1,
                'scene_counts':{'train':2,'validation':1,'test':1},
                'record_counts':{'train':'all','validation':'all','test':'all'},
                'window_sizes':[16],
                'teacher':{'spacing':16,'temperature':.05,'boundary_sigma':1.,
                           'alpha':1.,'beta':0.,'boundary_lambda':0.,
                           'shadow_screening':True,'save_plots':False,
                           'combined_rule':'normalized_elementwise_max'},
                'training':{'epochs':1,'patience':1,'batch_size':1,'hidden':8,
                            'num_threads':1,'candidate_chunk':16,'progress_every_batches':1,
                            'fourier_frequencies':1,'learning_rate':1e-3,'grad_clip':1.,
                            'weight_decay':0.,'device':'cpu','use_obstacle':False,
                            'pixel_coordinates':True,'seeds':[0]}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(config))
            self.assertEqual(read_config(path)['record_counts']['train'],'all')
            config['record_counts']['train']=0;path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'positive integers or "all"'):
                read_config(path)

    def test_normalized_elementwise_max_preserves_each_layers_strong_locations(self):
        first=np.array([.7,.2,.1,0.])
        second=np.array([.1,.3,.1,.5])
        actual=combine_layer_probabilities([first,second])
        expected=np.maximum(first,second)
        expected=expected/expected.sum()
        np.testing.assert_allclose(actual,expected)
        self.assertTrue(np.isclose(actual.sum(),1.))
        np.testing.assert_allclose(actual,[.4375,.1875,.0625,.3125])

    def test_window_class_uses_free_pixels_only_and_skips_uniform_views(self):
        self.assertEqual(_window_class(ToyDataset([[1.,1.],[1.,1.]]),0,0,1e-12),'all_lit')
        self.assertEqual(_window_class(ToyDataset([[0.,0.],[0.,0.]]),0,0,1e-12),'all_dark')
        self.assertEqual(_window_class(ToyDataset([[0.,1.],[0.,0.]]),0,0,1e-12),'eligible')
        self.assertEqual(_window_class(ToyDataset([[9.,1.],[1.,1.]],[[1,0],[0,0]]),0,0,1e-12),'all_lit')

    def test_normalized_max_rejects_misaligned_empty_or_unnormalized_inputs(self):
        for values in ([],[np.ones(2),np.ones(3)]):
            with self.assertRaises(ValueError):
                combine_layer_probabilities(values)
        with self.assertRaisesRegex(ValueError,'sum to one'):
            combine_layer_probabilities([np.array([.2,.2]),np.array([.5,.5])])

    def test_phase1_layer_teacher_does_not_require_an_exact_mask(self):
        class Evaluator:
            def positions_valid(self, positions):
                return np.ones(len(positions), dtype=bool)

            def visibility_at(self, position):
                # Neither valid candidate exactly matches [1, 0].
                return (np.array([[1., 1.]]) if position[0] == 0
                        else np.array([[0., 0.]]))

        grid=SimpleNamespace(candidate_xy=np.array([[0.,0.],[1.,0.]]))
        valid,cost,probability=_phase1_layer_teacher(
            Evaluator(),grid,np.array([[1.,0.]]),temperature=.05,
            alpha=1.,beta=0.,boundary_lambda=0.,boundary_sigma=1.)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(cost,[.5,.5])
        np.testing.assert_allclose(probability,[.5,.5])

    def test_shadow_screen_rejects_only_candidates_lighting_free_zero_pixels(self):
        class Evaluator:
            masks=(np.array([[0,1],[0,0]]),  # lights only an observed-bright pixel
                   np.array([[1,1],[0,0]]),  # incorrectly lights a free shadow pixel
                   np.array([[0,0],[1,0]]),  # light is inside an obstacle, so allowed
                   np.array([[0,0],[0,0]]))
            def positions_valid(self,positions):
                return np.array([1,1,1,0],dtype=bool)
            def visibility_at(self,position):
                return self.masks[int(position[0])]
        grid=SimpleNamespace(candidate_xy=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]]))
        observed=np.array([[0.,.7],[0.,0.]])
        obstacle=np.array([[0,0],[1,0]],dtype=bool)
        retained,report=screen_candidates_by_observed_shadow(
            Evaluator(),grid,observed,obstacle,tolerance=1e-12)
        np.testing.assert_array_equal(retained,[1,0,1,0])
        self.assertEqual(report['shadow_pixels'],2)
        self.assertEqual(report['geometry_valid_candidates'],3)
        self.assertEqual(report['rejected_candidates'],1)
        self.assertEqual(report['retained_candidates'],2)

    def test_phase1_layer_teacher_uses_shared_shadow_screen(self):
        class Evaluator:
            def positions_valid(self,positions):
                raise AssertionError('precomputed candidate mask should be reused')
            def visibility_at(self,position):
                return np.array([[position[0]==1]],dtype=float)
        grid=SimpleNamespace(candidate_xy=np.array([[0.,0.],[1.,0.]]))
        valid,cost,probability=_phase1_layer_teacher(
            Evaluator(),grid,np.array([[1.]]),temperature=.05,alpha=1.,beta=0.,
            boundary_lambda=0.,boundary_sigma=1.,candidate_valid=np.array([0,1],dtype=bool))
        np.testing.assert_array_equal(valid,[0,1])
        self.assertTrue(np.isinf(cost[0]));self.assertEqual(cost[1],0.)
        np.testing.assert_allclose(probability,[0.,1.])

    def test_all_layers_share_one_uniform_candidate_pass(self):
        class Evaluator:
            masks=(np.array([[1,0]]),np.array([[1,1]]),np.array([[0,1]]))
            def __init__(self):self.calls=np.zeros(3,dtype=int)
            def positions_valid(self,positions):return np.ones(len(positions),dtype=bool)
            def visibility_at(self,position):
                index=int(position[0]);self.calls[index]+=1
                return self.masks[index]
        evaluator=Evaluator()
        grid=SimpleNamespace(candidate_xy=np.array([[0.,0.],[1.,0.],[2.,0.]]))
        valid,costs,probabilities,screening=_phase1_layers_from_one_uniform_pass(
            evaluator,grid,[np.array([[1.,0.]]),np.array([[0.,1.]])],
            np.array([[1.,1.]]),np.zeros((1,2),dtype=bool),temperature=.05,
            alpha=1.,beta=0.,boundary_lambda=0.,boundary_sigma=1.,
            shadow_screening=False)
        np.testing.assert_array_equal(evaluator.calls,[1,1,1])
        np.testing.assert_array_equal(valid,[1,1,1])
        np.testing.assert_allclose(costs,[[0.,.5,1.],[1.,.5,0.]])
        self.assertEqual(len(probabilities),2)
        self.assertTrue(all(np.isclose(p.sum(),1.) for p in probabilities))
        self.assertTrue(screening['performed'])
        self.assertEqual(screening['geometry_valid_candidates'],3)

    def test_layer_peeling_visualization_uses_the_saved_subtraction(self):
        layer={'residual_before':np.array([[.3,.8,.5]]),
               'layer_response':np.array([[1,1,0]],dtype=bool),
               'residual_after':np.array([[0.,.5,.5]])}
        before,mask,contribution,after=layer_peeling_arrays(layer,.3)
        np.testing.assert_allclose(contribution,[[.3,.3,0.]])
        np.testing.assert_allclose(after,before-contribution)
        self.assertEqual(mask.tolist(),[[True,True,False]])

    def test_layer_peeling_rejects_nonbinary_response(self):
        layer={'residual_before':np.array([[.3]]),'layer_response':np.array([[.5]]),
               'residual_after':np.array([[.15]])}
        with self.assertRaisesRegex(ValueError,'binary 0/1'):
            layer_peeling_arrays(layer,.3)


if __name__=='__main__':
    unittest.main()
