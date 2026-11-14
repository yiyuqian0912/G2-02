import copy,json,os,socket
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch
from decomposition_model import decomposition_training_loss,slot_activation
from disentangled_objective import evaluation_assignment
from train_decomposition import make_model,load_config,train
from evaluate_masks import metric_rows,pad_missing_predictions,evaluate_with_reference
from test_boundary_refined import example,config_for_test
from test_count_first import prepare_fixture
from reuse_mask_cache import prepare_cached_labels

ROOT=Path(__file__).resolve().parents[3]

def config():
    c=load_config(ROOT/'Phase2_Multi_Source/configs/masks_disentangled.json')
    tiny=config_for_test();c['model'].update(tiny['model']);c['model']['decomposition_architecture']='disentangled_slots_v5'
    c['training'].update(tiny['training']);c['training'].pop('initial_decomposition_checkpoint',None)
    c['training'].pop('model_selection',None)
    c['training'].update(evaluate_after_training=False,compare_initial_checkpoint=False,epochs=2)
    return c


def _distributed_worker(rank,port,c,index):
    import torch.distributed as dist
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    try:train(c,Path(c['output_root']),index)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


class DisentangledTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(31)

    def test_forward_count_selects_slots_without_brightness_gate_or_teacher(self):
        c=config();model=make_model(c,1024,torch.device('cpu'),False).eval();b=example()
        with torch.inference_mode():o=model(b['response'],b['obstacle'],b['window'])
        self.assertEqual(slot_activation(o).sum().item(),o['count_logits'].argmax(1).item())
        expected=(o['mask_probability']*(o['active_slots']*o['intensity'])[...,None,None]).sum(1)
        torch.testing.assert_close(expected,o['reconstruction'])
        attenuated=(o['mask_probability']*(o['presence_probability']*o['intensity'])[...,None,None]).sum(1)
        self.assertFalse(torch.allclose(attenuated,expected))
        self.assertIsNone(model.locator)
        for side in (8,32,64):
            b=example(side)
            with torch.inference_mode():
                a=model(b['response'],b['obstacle'],b['window']);d=model(.1*b['response'],~b['obstacle'],b['window'])
            torch.testing.assert_close(a['mask_probability'],d['mask_probability'],atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(a['intensity']*.1,d['intensity'],atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(a['count_logits'],d['count_logits'],atol=3e-5,rtol=3e-5)
        with torch.inference_mode():z=model(torch.zeros_like(b['response']),b['obstacle'],b['window'])
        self.assertFalse(z['active_slots'].any());self.assertEqual(z['reconstruction'].sum(),0)

    def test_losses_separate_presence_and_amplitude_and_finite_amp_gradients(self):
        for amp,checkpointed in ((False,False),(True,False),(False,True)):
            c=config();c['training']['activation_checkpoint_decomposer']=checkpointed
            model=make_model(c,1024,torch.device('cpu'),False);b=example()
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):o=model(b['response'],b['obstacle'],b['window'])
            loss=decomposition_training_loss(o,b,**c['loss'])
            torch.testing.assert_close(loss['assignment'],evaluation_assignment(o,b))
            # The shape, strength and teacher-conditioned reconstruction losses
            # have no direct dependence on the existence logits tensor.
            for key in ('mask','intensity','reconstruction_supervised_normalized'):
                grad=torch.autograd.grad(loss[key],o['presence_logits'],retain_graph=True,allow_unused=True)[0]
                self.assertTrue(grad is None or (grad==0).all(),key)
            loss['loss'].backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            b.update(scene_id=torch.tensor([0]),view_id=torch.tensor([0]))
            row=metric_rows(o,b,loss['assignment'])[0]
            self.assertEqual(row['predicted_count'],o['predicted_count'].item())
            padded=pad_missing_predictions(o,6)
            self.assertEqual(padded['active_slots'].shape,(1,6))
            self.assertFalse(padded['active_slots'][:,4:].any())
            self.assertEqual(padded['global_count_logits'].shape,(1,7))

    def test_perfect_layers_are_separately_correct_and_presence_does_not_dim_response(self):
        from decomposition_model import stable_count_logits
        b=example();m=b['teacher_mask'];real=b['layer_mask'];logits=torch.where(m.bool(),30.,-30.)
        presence=torch.where(real,torch.tensor(20.),torch.tensor(-20.))
        count=torch.full((1,5),-30.);count[:,2]=30.
        o={'_disentangled':True,'mask_probability':m,'mask_logits':logits,'intensity':b['teacher_intensity'],
           'presence_logits':presence,'presence_probability':presence.sigmoid(),'active_slots':real,
           'count_logits':count,'global_count_logits':count,'slot_count_logits':stable_count_logits(presence),
           'reconstruction':b['response']}
        losses=decomposition_training_loss(o,b,**config()['loss'])
        for key in ('mask','intensity','reconstruction_supervised_normalized','reconstruction_deployment_normalized'):
            self.assertLess(abs(losses[key].item()),1e-5,key)
        self.assertEqual(losses['strict_decomposition_success'].item(),1.)
        changed={**o,'presence_logits':torch.zeros_like(presence),'presence_probability':torch.full_like(presence,.5)}
        changed['slot_count_logits']=stable_count_logits(changed['presence_logits'])
        after=decomposition_training_loss(changed,b,**config()['loss'])
        for key in ('mask','intensity','reconstruction_supervised_normalized','reconstruction_deployment_normalized'):
            torch.testing.assert_close(losses[key],after[key])
        self.assertGreater(after['presence'].item(),losses['presence'].item())

    def test_two_rank_shared_refinement_updates_match_serial_and_empty_eval_rank(self):
        try:
            probe=socket.socket();probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close()
        except PermissionError:self.skipTest('loopback ports unavailable in sandbox')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,new,index=prepare_fixture(root)
            c=config();c['training'].update(epochs=1,evaluate_initial_checkpoint=False,evaluate_test_after_training=False,
                                           target_global_batch_size=4)
            c.pop('cache_source_root',None)
            c['output_root']=str(root/'serial');Path(c['output_root']).mkdir()
            import shutil
            shutil.copytree(Path(base['output_root'])/'mask_cache',Path(c['output_root'])/'mask_cache')
            train(c,Path(c['output_root']),index)
            parallel=copy.deepcopy(c);parallel['output_root']=str(root/'parallel');Path(parallel['output_root']).mkdir()
            shutil.copytree(Path(base['output_root'])/'mask_cache',Path(parallel['output_root'])/'mask_cache')
            torch.multiprocessing.spawn(_distributed_worker,args=(port,parallel,index),nprocs=2,join=True)
            a=torch.load(Path(c['output_root'])/'models/seed_0/last.pt',weights_only=True)
            b=torch.load(Path(parallel['output_root'])/'models/seed_0/last.pt',weights_only=True)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=3e-6,rtol=2e-5)
            self.assertEqual(b['history'][0]['validation']['windows_processed'],1)

    def test_only_explicit_migration_loads_every_old_weight_and_new_head_zero(self):
        c=config();old=copy.deepcopy(c);old['model']['decomposition_architecture']='boundary_refined_slots';old['loss']={}
        m=make_model(old,1024,torch.device('cpu'),False)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'v3.pt'
            torch.save({'schema':'phase2-local-mask-decomposition-v1','config':old,'model':m.state_dict()},checkpoint)
            c['training']['initial_decomposition_checkpoint']=str(checkpoint)
            loaded=make_model(c,1024,torch.device('cpu'),True)
            for k,v in m.state_dict().items():torch.testing.assert_close(loaded.state_dict()[k],v)
            self.assertEqual(loaded.global_count_head[-1].weight.abs().sum(),0)
            c['training']['allow_v3_warm_start']=False
            with self.assertRaisesRegex(ValueError,'migration'):make_model(c,1024,torch.device('cpu'),True)

    def test_cache_warm_start_train_resume_and_old_reference_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,exp,index=prepare_fixture(root)
            old=config();old['model']['decomposition_architecture']='boundary_refined_slots'
            old['loss']=load_config(ROOT/'Phase2_Multi_Source/configs/masks_boundary_refined.json')['loss']
            old['output_root']=base['output_root'];old['cache_source_root']=str(root/'unused')
            old['training'].update(epochs=1,evaluate_initial_checkpoint=False)
            train(old,Path(old['output_root']),index)
            c=config();c.update(output_root=str(root/'new'),cache_source_root=old['output_root'])
            c['training'].update(initial_decomposition_checkpoint=str(Path(old['output_root'])/'models/seed_0/best.pt'),
                evaluate_initial_checkpoint=True,compare_initial_checkpoint=True)
            c['training']['model_selection']={'metric':'count_group_macro','max_single_layer_drop':.005,'max_overall_drop':.005,
                'quality_max_drop':{'mask_dice':.01,'strength_within_10pct':.01,'strict_decomposition_success':.005}}
            output=Path(c['output_root']);output.mkdir()
            reused=prepare_cached_labels(c)
            train(c,output,reused,epoch_limit=1)
            train(c,output,reused)
            saved=torch.load(output/'models/seed_0/last.pt',weights_only=True)
            self.assertEqual(saved['epoch'],2)
            self.assertEqual(len(saved['history']),2)
            report=evaluate_with_reference(c,output,reused,repetitions=2)
            self.assertEqual(report['activation_rule'],'predicted_count_argmax_then_top_slots')
            self.assertEqual(report['evaluation_matching'],'shape_strength_v5')
            self.assertTrue((output/'evaluation/initial_reference/decomposition_metrics.json').exists())


if __name__=='__main__':unittest.main()
