import copy
import tempfile
import unittest
import socket
from pathlib import Path
from unittest.mock import patch
import torch
from test_sequential import configuration,fixture,example
from train_decomposition import make_model,load_config,semantic_config
from decomposition_model import decomposition_training_loss
from sequential_count_shape import accepted_layers
from sequential_objective import primary_counts,activation_assignment


def force_count(model,number):
    old=model._count
    def fixed(*args):
        result=old(*args)
        result['predicted_count']=torch.full((len(args[1]),),number,device=args[1].device,dtype=torch.long)
        return result
    return fixed


def asymmetric_ddp_worker(rank,port):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);torch.manual_seed(123)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        c=configuration();model=make_model(c,1024,torch.device('cpu'),False)
        model.adaptive_checkpoint_min_layers=2
        model._count=force_count(model,0 if rank==0 else 2)
        wrapped=DistributedDataParallel(model,find_unused_parameters=False)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
        batch=example()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            out=wrapped(batch['response'],batch['obstacle'],batch['window'],verify_constraints=False)
            decomposition_training_loss(out,batch,**c['loss'])['loss'].backward()
            for name,p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(),name
            optimizer.step()
        digest=torch.stack([p.detach().double().sum() for p in model.parameters()])
        other=digest.clone();dist.broadcast(other,src=0)
        torch.testing.assert_close(digest,other,rtol=0,atol=0)
    finally:dist.destroy_process_group()


class FastSequentialTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(123)

    def test_deferred_gate_preserves_masks_losses_and_gradients_but_cannot_accept(self):
        c=configuration();original=make_model(c,1024,torch.device('cpu'),False);fast=copy.deepcopy(original)
        b=example();outputs=[];losses=[]
        for m,verify in ((original,True),(fast,False)):
            with patch.object(m,'_count',force_count(m,2)):
                out=m(b['response'],b['obstacle'],b['window'],verify_constraints=verify)
            loss=decomposition_training_loss(out,b,**c['loss'])
            loss['loss'].backward();outputs.append(out);losses.append(loss)
        for key in ('mask_logits','soft_mask_probability','mask_probability','count_logits','internal_residual'):
            torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=0,atol=0)
        torch.testing.assert_close(losses[0]['loss'],losses[1]['loss'],rtol=0,atol=0)
        for (name,a),(_,bparam) in zip(original.named_parameters(),fast.named_parameters()):
            if a.requires_grad:torch.testing.assert_close(a.grad,bparam.grad,rtol=0,atol=0,msg=name)
        self.assertTrue(outputs[0]['_constraint_verified']);self.assertFalse(outputs[1]['_constraint_verified'])
        self.assertFalse(outputs[1]['constraint_accepted'].any());self.assertIsNone(accepted_layers(outputs[1],0))
        assignment=activation_assignment(outputs[0],b)
        torch.testing.assert_close(primary_counts(outputs[0],b),primary_counts(outputs[0],b,assignment=assignment))

    def test_adaptive_checkpointing_preserves_outputs_and_finite_gradients(self):
        c=configuration();a=make_model(c,1024,torch.device('cpu'),False);b=copy.deepcopy(a)
        b.adaptive_checkpoint_min_layers=2
        batch=example();outputs=[]
        for m in (a,b):
            with patch.object(m,'_count',force_count(m,2)):
                output=m(batch['response'],batch['obstacle'],batch['window'],verify_constraints=False)
            decomposition_training_loss(output,batch,**c['loss'])['loss'].backward();outputs.append(output)
        self.assertFalse(a._checkpoint_this_rollout);self.assertTrue(b._checkpoint_this_rollout)
        torch.testing.assert_close(outputs[0]['mask_logits'],outputs[1]['mask_logits'])
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7,msg=name)

    def test_zero_predicted_count_keeps_ddp_parameters_in_graph_without_teacher_steps(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False);b=example()
        with patch.object(m,'_count',force_count(m,0)),patch.object(m,'_step',side_effect=AssertionError('no teacher-forced step')):
            output=m(b['response'],b['obstacle'],b['window'],verify_constraints=False)
        decomposition_training_loss(output,b,**c['loss'])['loss'].backward()
        for name,p in m.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),'Gloo unavailable')
    def test_two_rank_zero_vs_two_layer_rollouts_without_unused_parameter_errors(self):
        try:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        except PermissionError:self.skipTest('loopback unavailable')
        torch.multiprocessing.spawn(asymmetric_ddp_worker,args=(port,),nprocs=2,join=True)

    def test_expected_warm_start_epoch_prevents_reusing_an_old_frozen_run(self):
        import configure_sequential as launch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,source,index,_=fixture(root)
            phase=root/'phase';(phase/'configs').mkdir(parents=True)
            template=Path(__file__).resolve().parents[2]/'configs/masks_sequential.json'
            (phase/'configs/masks_sequential.json').write_text(template.read_text())
            with patch.object(launch,'PHASE2',phase):
                with self.assertRaisesRegex(ValueError,'expected 16'):
                    launch.configure(source_root=str(source),run_name='best16',expected_source_epoch=16)
                path=launch.configure(source_root=str(source),run_name='best14',expected_source_epoch=14)
                config=load_config(path)
                self.assertEqual(config['source_training_protocol']['warm_start']['epoch'],14)
                self.assertEqual(config['label_preparation']['record_counts'],index['counts'])
                model=make_model(config,1024,torch.device('cpu'),True)
                report=model.warm_start_load_report
                self.assertEqual(report['source_epoch'],14)
                self.assertFalse(report['optimizer_inherited'])
                self.assertEqual(report['inherited_parameters']+report['new_parameters'],
                                 sum(p.numel() for p in model.parameters()))
                original=torch.load(config['training']['initial_decomposition_checkpoint'],weights_only=True)['model']
                for name,value in original.items():
                    torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
                with self.assertRaisesRegex(ValueError,'existing run froze epoch 14'):
                    launch.configure(source_root=str(source),run_name='best14',expected_source_epoch=16)
                changed=copy.deepcopy(config);changed['training'].update(gpu_inputs='stream',defer_constraint_verification=False,adaptive_checkpoint_min_layers=0)
                self.assertEqual(semantic_config(config),semantic_config(changed))
                snapshot=Path(config['training']['initial_decomposition_checkpoint'])
                tampered=torch.load(snapshot,weights_only=True);tampered['epoch']=16
                torch.save(tampered,snapshot)
                with self.assertRaisesRegex(ValueError,'frozen warm-start checkpoint changed'):
                    make_model(config,1024,torch.device('cpu'),True)


if __name__=='__main__':unittest.main()
