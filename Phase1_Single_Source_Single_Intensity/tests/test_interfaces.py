import tempfile
import unittest
from pathlib import Path
import numpy as np
from rind_phase1.interfaces import training_record, validate_observation
from rind_phase1.teacher import teacher_probabilities


def fixture(size=16, scene=0):
    observation = dict(scene_id=scene,view_id=0,window=np.array([0,0,size]),
                       response=np.zeros((size,size),dtype=np.float32),obstacle=np.zeros((size,size),dtype=bool))
    cost = np.array([0.,1.,np.inf,0.])
    valid = np.array([True,True,False,True])
    teacher = dict(scene_id=scene,view_id=0,window=observation['window'],
        candidate_xy=np.array([[64.,64.],[192.,192.],[320.,320.],[448.,448.]]),valid=valid,
        evaluated=np.ones(4,dtype=bool),physical_cost=cost,
        teacher_prob=teacher_probabilities(cost,valid,temperature=.1),temperature=.1,
        config_id='fixture',metadata={'complete':True})
    return teacher, observation


class ContractTests(unittest.TestCase):
    def test_join_and_reject_misalignment(self):
        t,o=fixture()
        r=training_record(t,o)
        np.testing.assert_array_equal(r['candidate_xy'],t['candidate_xy'])
        self.assertIn('obstacle',r)
        with self.assertRaises(ValueError): training_record(t,{**o,'view_id':1})
        with self.assertRaises(ValueError): training_record({**t,'area_weight':np.ones(4)},o)
        with self.assertRaises(ValueError): training_record({**t,'evaluated':np.zeros(4,dtype=bool)},o)
        with self.assertRaises(ValueError): training_record({**t,'teacher_prob':np.ones(4)/4},o)

    def test_obstacle_required(self):
        _,o=fixture()
        del o['obstacle']
        with self.assertRaises(ValueError): validate_observation(o)

    def test_main_inspection_accepts_local_mask(self):
        from rind_phase1.evaluate import inspect_observation
        _,o=fixture()
        inspect_observation([o])


class TorchContractTests(unittest.TestCase):
    def test_training_prediction_and_checkpoint(self):
        import torch
        from rind_phase1.model import SourceEnergyField
        from rind_phase1.train import batches,run_epoch,load_model
        from rind_phase1.predict import predict
        from rind_phase1.evaluate import validate_student_result
        from part_e.evaluate import evaluate_record
        torch.set_num_threads(1)
        torch.manual_seed(1)
        t,o=fixture()
        records=[training_record(t,o),training_record(*fixture(32,1))]
        self.assertEqual([b['response'].shape[-1] for b in batches(records,4)],[16,32])
        model=SourceEnergyField(hidden=8,fourier_frequencies=2)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        self.assertTrue(np.isfinite(run_epoch(model,records,4,optimizer)))
        whole=predict(model,t,o,chunk=4)
        chunked=predict(model,t,o,chunk=1)
        np.testing.assert_allclose(whole['student_prob'],chunked['student_prob'],atol=1e-7)
        self.assertEqual(whole['student_prob'][2],0)
        self.assertAlmostEqual(whole['student_prob'].sum(),1)
        validate_student_result(whole)
        evaluate_record(t,whole,threshold=.01,spacing=128)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(interface_version='phase1-v1',model_config=dict(hidden=8,fourier_frequencies=2),model=model.state_dict()),path)
            restored,_=load_model(path)
            np.testing.assert_allclose(predict(restored,t,o)['student_prob'],whole['student_prob'],atol=1e-7)


if __name__=='__main__': unittest.main()
