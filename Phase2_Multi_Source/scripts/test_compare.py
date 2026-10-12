import unittest
import numpy as np
from compare_teachers import fit_strengths, similarity, marginal
class ComparisonTests(unittest.TestCase):
    def test_exact_addition(self):
        v=np.array([[1,0],[1,1],[0,1],[0,0]])
        a,c=fit_strengths(v,np.array([.3,.8,.5,0]))
        np.testing.assert_allclose(a,[.3,.5],atol=1e-8)
        self.assertLess(c,1e-10)
    def test_mae_not_mse(self):
        a,c=fit_strengths(np.ones((3,1)),np.array([.1,.1,.9]))
        self.assertAlmostEqual(a[0],.1)
        self.assertAlmostEqual(c,.8/3)
    def test_bounds(self):
        a,c=fit_strengths(np.ones((1,1)),np.array([2.]))
        self.assertAlmostEqual(a[0],1.)
        self.assertAlmostEqual(c,1.)
    def test_metrics_and_projection(self):
        self.assertEqual(similarity(np.array([1.,0]),np.array([0.,1]))['JS_bits'],1)
        m=marginal([[0,1],[1,2]],[.25,.75],3)
        np.testing.assert_allclose(m,[.125,.5,.375])
if __name__=='__main__': unittest.main()
