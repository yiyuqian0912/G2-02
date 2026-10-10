import unittest
import numpy as np
import torch
from scripts.dense_probability import score_pixels,pixel_probability
from rind_phase1.model import SourceEnergyField


class DenseTests(unittest.TestCase):
    def test_cached_encoding_matches_forward_and_chunking(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model=SourceEnergyField(hidden=8,fourier_frequencies=2,coordinate_scale=32)
        sample=dict(scene_id=0,view_id=0,response=np.zeros((16,16),np.float32),obstacle=np.zeros((16,16),bool),window=np.array([0,0,16]))
        a=score_pixels(model,sample,32,chunk=83)
        b=score_pixels(model,sample,32,chunk=256)
        yy,xx=np.mgrid[:32,:32]
        xy=np.column_stack((xx.ravel()+.5,yy.ravel()+.5))
        with torch.no_grad():
            direct=model(torch.tensor(sample['response'][None]),torch.tensor(sample['obstacle'][None]),torch.tensor(sample['window'][None],dtype=torch.float32),torch.tensor(xy[None],dtype=torch.float32))[0].numpy().reshape(32,32)
        np.testing.assert_allclose(a,direct,atol=1e-6)
        np.testing.assert_allclose(a,b,atol=1e-6)
        valid=np.ones((32,32),bool);valid[:16,:16]=False
        p=pixel_probability(a,valid)
        self.assertAlmostEqual(p.sum(),1)
        self.assertTrue((p[~valid]==0).all())
        self.assertEqual(p.shape,(32,32))


if __name__=='__main__':unittest.main()
