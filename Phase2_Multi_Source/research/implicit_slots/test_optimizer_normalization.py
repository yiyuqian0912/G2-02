import copy,json,os,socket,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from sequential_count_shape import peel
from sequential_objective import loss
from optimization_batches import optimizer_batches


OPTIONS={'count_class_weights':[1.,.5,1.3,2.,1.],
         'shape_class_weights':[1.,3.,1.2,.7,1.]}


def batch(counts):
    n=len(counts);m=torch.zeros(n,4,4,4)
    for i,r in enumerate(counts):
        for j in range(r):m[i,j,:,j]=1
    return {'teacher_mask':m,'layer_mask':torch.arange(4)[None]<torch.tensor(counts)[:,None],
            'obstacle':torch.zeros(n,4,4,dtype=torch.bool)}


def output(theta,b):
    n=len(b['layer_mask'])
    z=theta[:4][None,:,None,None].expand(n,4,4,4)
    active=b['layer_mask'].clone()
    return {'mask_logits':z,'soft_mask_probability':z.sigmoid(),
            'mask_probability':(z>=0).float()*active[:,:,None,None],
            'active_slots':active,'predicted_count':active.sum(1),
            'count_logits':theta[4:][None].expand(n,5),
            'global_count_logits':(theta[4:]*.8)[None].expand(n,5),
            'slot_count_logits':(theta[4:]*1.2)[None].expand(n,5)}


def distributed_worker(rank,port,destination):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        theta=torch.linspace(-.4,.4,9,requires_grad=True)
        local=[batch([1]),batch([2])] if rank==0 else [batch([3]),batch([1])]
        for _,b,n in optimizer_batches(local,2,OPTIONS,torch.device('cpu'),2):
            loss(output(theta,b),b,**OPTIONS,optimizer_normalizers=n)['optimization_loss'].backward()
        dist.all_reduce(theta.grad);theta.grad/=2  # exact DDP averaging rule
        Path(destination,f'rank{rank}.json').write_text(json.dumps(theta.grad.tolist()))
    finally:dist.destroy_process_group()


class OptimizerNormalizationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_hard_forward_identical_but_future_residual_has_proxy_gradient(self):
        r=torch.tensor([[[.3,.8,.5]]],dtype=torch.float64)
        p=torch.tensor([[[.8,.8,.2]]],requires_grad=True);on=torch.tensor([True])
        hard,a,old=peel(r,p,on)
        mask,aa,new=peel(r,p,on,straight_through=True)
        self.assertTrue(torch.equal(hard,mask));self.assertTrue(torch.equal(old,new))
        self.assertTrue(torch.equal(a,aa));self.assertTrue((new>=0).all())
        self.assertFalse(old.requires_grad);self.assertTrue(new.requires_grad)
        gradient=torch.autograd.grad(new.square().sum(),p)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(gradient.abs().sum(),0)

    def test_real_model_later_residual_depends_on_earlier_mask_probability(self):
        from test_sequential import configuration
        from test_boundary_refined import example
        from train_decomposition import make_model
        torch.manual_seed(10);c=configuration();m=make_model(c,1024,torch.device('cpu'),False)
        b=example();step=m._step;count=m._count;seen={}
        def count2(*args):o=count(*args);o['predicted_count']=torch.tensor([2]);return o
        def capture(context,r,window,hidden,index):
            z,p,h=step(context,r,window,hidden,index)
            if index==0:
                desired=b['teacher_mask'][:,0]
                p=p+(desired-p).detach();seen['p']=p
            if index==1:seen['r']=r
            return z,p,h
        with patch.object(m,'_count',count2),patch.object(m,'_step',capture):m(b['response'],b['obstacle'],b['window'])
        g=torch.autograd.grad(seen['r'].sum(),seen['p'])[0]
        self.assertGreater(g.abs().sum(),0);self.assertTrue(torch.isfinite(g).all())

    def test_one_big_batch_and_accumulated_microbatches_have_identical_gradients(self):
        theta=torch.linspace(-.4,.4,9,requires_grad=True);full=batch([1,3,2])
        reference=loss(output(theta,full),full,**OPTIONS)['loss']
        expected=torch.autograd.grad(reference,theta)[0]
        actual=torch.linspace(-.4,.4,9,requires_grad=True)
        for _,b,n in optimizer_batches([batch([1]),batch([3]),batch([2])],3,OPTIONS,torch.device('cpu')):
            loss(output(actual,b),b,**OPTIONS,optimizer_normalizers=n)['optimization_loss'].backward()
        torch.testing.assert_close(actual.grad,expected,atol=2e-7,rtol=2e-6)

    def test_last_short_accumulation_group_keeps_every_record(self):
        rows=list(optimizer_batches([batch([1]),batch([3]),batch([2])],2,OPTIONS,torch.device('cpu')))
        self.assertEqual([r[0] for r in rows],[1,2,3])
        self.assertAlmostEqual(rows[0][2]['count'].item(),2.5)
        self.assertAlmostEqual(rows[0][2]['shape'].item(),5.1,places=6)
        self.assertAlmostEqual(rows[2][2]['count'].item(),1.3,places=6)
        self.assertAlmostEqual(rows[2][2]['shape'].item(),2.4,places=6)

    def test_two_rank_unequal_layer_counts_match_a_single_global_batch(self):
        try:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        except PermissionError:self.skipTest('loopback unavailable')
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker,args=(port,tmp),nprocs=2,join=True)
            theta=torch.linspace(-.4,.4,9,requires_grad=True);b=batch([1,2,3,1])
            expected=torch.autograd.grad(loss(output(theta,b),b,**OPTIONS)['loss'],theta)[0]
            for rank in range(2):
                actual=torch.tensor(json.loads(Path(tmp,f'rank{rank}.json').read_text()))
                torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)


if __name__=='__main__':unittest.main()
