import unittest
import numpy as np
from rind_phase2.interfaces import validate_observation,validate_joint_candidates
from rind_phase2.search import canonical_sources

class ContractTests(unittest.TestCase):
    def test_additive_response_not_clipped(self):
        sample=dict(scene_id=0,view_id=0,window=np.array([0,0,16]),response=np.full((16,16),2.),obstacle=np.zeros((16,16),bool))
        self.assertIs(validate_observation(sample),sample)
    def test_strengths_and_variable_counts(self):
        v=np.zeros((2,2,3));v[0,0]=[20,20,.2];v[1]=canonical_sources([[30,30,.3],[20,20,.4]])
        a,k=validate_joint_candidates(v,np.array([1,2]));self.assertEqual(k.tolist(),[1,2])
        self.assertEqual(a[0,0,2],.2)
    def test_padding_not_an_extra_zero_strength_source(self):
        v=np.array([[[20,20,.2],[30,30,0.]]])
        with self.assertRaises(ValueError):validate_joint_candidates(v,np.array([1]))
    def test_reject_duplicate_hypotheses(self):
        v=np.array([[[20,20,.2]],[[20,20,.2]]])
        with self.assertRaises(ValueError):validate_joint_candidates(v,np.array([1,1]))
