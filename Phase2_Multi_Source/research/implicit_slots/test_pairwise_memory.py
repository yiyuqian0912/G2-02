import unittest
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from pairwise_ops import masked_pairwise_dot
from gpu_runtime import calibration_profiles,_measure
from train_decomposition import make_model
from test_sequential import configuration
from test_boundary_refined import example


class Shapes(TorchDispatchMode):
    def __init__(self):super().__init__();self.shapes=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        def visit(value):
            if isinstance(value,torch.Tensor):self.shapes.append(tuple(value.shape))
            elif isinstance(value,(tuple,list)):
                for item in value:visit(item)
        visit(result)
        return result


class PairwiseMemoryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(19)

    def test_values_and_gradients_equal_the_explicit_weighted_sum(self):
        left=torch.randn(2,5,23,dtype=torch.float64,requires_grad=True)
        right=torch.randn(2,6,23,dtype=torch.float64,requires_grad=True)
        weight=torch.rand(2,23,dtype=torch.float64,requires_grad=True)
        expected=(left[:,:,None]*right[:,None]*weight[:,None,None]).sum(-1)
        actual=masked_pairwise_dot(left,right,weight)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        upstream=torch.randn_like(actual)
        a=torch.autograd.grad((actual*upstream).sum(),(left,right,weight),retain_graph=True)
        b=torch.autograd.grad((expected*upstream).sum(),(left,right,weight))
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)

    def test_reproduces_unoptimized_einsum_broadcast_and_removes_it(self):
        left=torch.rand(2,32,256);right=torch.rand(2,32,256);weight=torch.rand(2,256)
        import torch.backends.opt_einsum as backend
        with patch.object(backend,'enabled',False),Shapes() as old:
            reference=torch.einsum('btp,bsp,bp->bts',left,right,weight)
        self.assertTrue(any(len(s)==4 and s[-1]==256 for s in old.shapes),old.shapes)
        with Shapes() as new:actual=masked_pairwise_dot(left,right,weight)
        self.assertFalse(any(len(s)>3 for s in new.shapes),new.shapes)
        self.assertLessEqual(max(torch.tensor(s).prod().item() for s in new.shapes),left.numel())
        torch.testing.assert_close(actual,reference,rtol=2e-6,atol=1e-4)

    def test_probe_covers_checkpointed_and_uncheckpointed_peak_without_updates(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False)
        m.adaptive_checkpoint_min_layers=3
        self.assertEqual(calibration_profiles(m,True),[
            {'name':'full_horizon','horizon':4},
            {'name':'largest_uncheckpointed_horizon','horizon':2}])
        before={key:value.clone() for key,value in m.state_dict().items()}
        batch=example();observed=[];step=m._step
        def capture(*args):observed.append(m._checkpoint_this_rollout);return step(*args)
        for horizon,checkpointed in ((4,True),(2,False)):
            observed.clear()
            with patch('torch.cuda.synchronize'),patch.object(m,'_step',capture):
                self.assertGreater(_measure(m,batch,c['loss'],torch.device('cpu'),False,1,horizon=horizon),0)
            self.assertEqual(observed,[checkpointed]*(horizon*2))
            self.assertFalse(hasattr(m,'calibration_horizon'))
        for key,value in before.items():torch.testing.assert_close(m.state_dict()[key],value,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        m.activation_checkpoint_decomposer=True
        self.assertEqual(len(calibration_profiles(m,True)),1)

    def test_probe_state_restored_when_a_profile_ooms(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False);batch=example()
        with patch('torch.cuda.synchronize'),patch.object(m,'forward',side_effect=torch.cuda.OutOfMemoryError('probe')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                _measure(m,batch,c['loss'],torch.device('cpu'),False,0,horizon=2)
        self.assertFalse(hasattr(m,'calibration_horizon'))
        self.assertFalse(m.calibration_full_horizon)


if __name__=='__main__':unittest.main()
