import copy,json,os,socket,subprocess,sys,tempfile,unittest,shlex
from pathlib import Path
from unittest.mock import patch
import torch
from decomposition_model import decomposition_training_loss
from train_decomposition import make_model,load_config,train,digest,semantic_config
from test_disentangled import config as old_config
from test_boundary_refined import example
from sequential_count_shape import peel,accepted_layers
from sequential_objective import primary_counts,summarize_primary
from reuse_mask_cache import prepare_cached_labels

ROOT=Path(__file__).resolve().parents[3]


def configuration():
    c=old_config();c['model']['decomposition_architecture']='sequential_count_shape_v6';c['model']['residual_gradient']='straight_through'
    c['training'].update(allow_v5_warm_start=True,compare_initial_checkpoint=False,loss_normalization='optimizer_step')
    c['loss']={'objective':'sequential_count_shape_v6','count_weight':.5,'mask_weight':.5}
    return c


def fixture(root):
    from test_count_first import prepare_fixture
    from expand_complex_masks import prepare_complex
    data,base,expanded,old=prepare_fixture(root)
    # Preserve the appended complex windows, not just the two base windows.
    index=prepare_complex(expanded,data)
    source=Path(expanded['output_root']);c=old_config();c['output_root']=str(source)
    c['model'].update(base['model']);c['model']['decomposition_architecture']='disentangled_slots_v5'
    c['model'].update(head_roles=[4,2,2],mask_feature_dim=16,refinement_rounds=2)
    c['training'].update(base['training']);c['training'].pop('model_selection',None)
    m=make_model(c,1024,torch.device('cpu'),False)
    path=source/'models/seed_0/best.pt';path.parent.mkdir(parents=True)
    signed={k:v for k,v in index.items() if k!='statistics'}
    torch.save({'schema':'phase2-local-mask-decomposition-v1','config':c,'model':m.state_dict(),
                'epoch':14,'signature':digest({'config':semantic_config(c),'index':signed})},path)
    data.calls.clear()
    return data,source,index,c


def ddp_worker(rank,port,c,index):
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK=str(rank),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    import torch.distributed as dist
    try:train(c,Path(c['output_root']),index)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


class SequentialTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(123)

    def test_minimum_examples_binary_overlap_conservation_and_strong_effective_layer(self):
        for values,masks in [((.3,.8,.5),[[1,1,0],[0,1,1]]),
                              ((.3,.8,.3),[[1,1,1],[0,1,0]]),
                              ((.3,.8,.3),[[0,1,0],[1,1,1]]),
                              ((1.4,2.1,.7),[[0,1,1],[1,1,0]])]:
            c=configuration();m=make_model(c,1024,torch.device('cpu'),False).eval()
            y=torch.zeros(1,8,8)
            for i,v in enumerate(values):y[:,:7,[0,2,4][i]:[2,4,8][i]]=v
            expected=[]
            for bits in masks:
                a=torch.zeros_like(y)
                for i,v in enumerate(bits):a[:,:7,[0,2,4][i]:[2,4,8][i]]=v
                expected.append(a)
            original_step=m._step;call=[0]
            def oracle(context,residual,window,hidden,index):
                z,p,updated_hidden=original_step(context,residual,window,hidden,index)
                p=expected[min(call[0],1)];call[0]+=1
                return torch.where(p.bool(),30.,-30.),p,updated_hidden
            count=m._count
            def count2(*args):
                o=count(*args);o['predicted_count']=torch.tensor([2]);return o
            with patch.object(m,'_count',count2),patch.object(m,'_step',oracle),torch.inference_mode():
                o=m(y,torch.zeros_like(y,dtype=torch.bool),torch.tensor([[0.,0.,8.]]))
            self.assertTrue(o['constraint_accepted'].item());self.assertEqual(o['executed_layers'].item(),2)
            self.assertTrue((o['residual']>=0).all());self.assertIsNotNone(accepted_layers(o,0))
            reconstructed=(o['mask_probability'].double()*o['intensity'][...,None,None]).sum(1)
            self.assertLessEqual(float((reconstructed-y).abs().max()),float(o['constraint_tolerance'][0]))
            self.assertTrue(((o['mask_probability']==0)|(o['mask_probability']==1)).all())
            if values[0]>1:self.assertGreater(o['intensity'].max(),1)

    def test_wrong_mask_cannot_overpeel_or_fabricate_last_layer(self):
        r=torch.tensor([[[.3,.8,0.]]],dtype=torch.float64)
        mask,a,after=peel(r,torch.tensor([[[1.,1.,0.]]]),torch.tensor([True]))
        self.assertTrue(mask[0,0,0]);self.assertFalse(mask[0,0,2]);self.assertTrue((after>=0).all())
        self.assertAlmostEqual(a.item(),.3);self.assertAlmostEqual(after[0,0,1].item(),.5)
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False).eval();b=example()
        count=m._count
        def count1(*args):o=count(*args);o['predicted_count']=torch.tensor([1]);return o
        with patch.object(m,'_count',count1),torch.inference_mode():o=m(b['response'],b['obstacle'],b['window'])
        if not o['constraint_accepted'].item():self.assertIsNone(accepted_layers(o,0))
        self.assertTrue((o['residual']>=0).all());self.assertLessEqual(o['executed_layers'].item(),1)

    def test_subtraction_exact_in_binary_fraction_arithmetic_and_early_stop_is_rejected(self):
        c=configuration();c['model']['roundoff_ulps']=0
        m=make_model(c,1024,torch.device('cpu'),False).eval()
        y=torch.zeros(1,8,8);y[:,:7,:4]=.25;y[:,:7,4:]=.75
        # Removing all positive support by its minimum twice exactly reconstructs
        # this two-level response. A predicted third layer must be rejected.
        step=m._step;count=m._count
        def all_support(context,r,window,hidden,index):
            z,p,next_hidden=step(context,r,window,hidden,index)
            return z,(r>0).float(),next_hidden
        def three(*args):o=count(*args);o['predicted_count']=torch.tensor([3]);return o
        with patch.object(m,'_count',three),patch.object(m,'_step',all_support),torch.inference_mode():
            o=m(y,torch.zeros_like(y,dtype=torch.bool),torch.tensor([[0.,0.,8.]]))
        self.assertEqual(o['residual'].sum(),0);self.assertEqual(o['executed_layers'].item(),2)
        self.assertFalse(o['constraint_accepted']);self.assertIsNone(accepted_layers(o,0))
        self.assertTrue(torch.equal((o['mask_probability'].double()*o['intensity'][...,None,None]).sum(1),y.double()))

    def test_only_original_observation_enters_forward_and_no_geometry_leak(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False).eval();b=example()
        with torch.inference_mode():
            a=m(b['response'],b['obstacle'],b['window']);d=m(b['response'],~b['obstacle'],b['window'])
        torch.testing.assert_close(a['mask_probability'],d['mask_probability'])
        torch.testing.assert_close(a['count_logits'],d['count_logits'])
        for key in ('teacher_mask','teacher_intensity','layer_mask','residual_before','seed_region'):
            with self.assertRaises(TypeError):m(b['response'],b['obstacle'],b['window'],**{key:b['teacher_mask']})

    def test_equal_two_losses_and_finite_gradients_including_amp_and_checkpointing(self):
        for amp,ck in ((False,False),(True,False),(False,True)):
            c=configuration();c['training']['activation_checkpoint_decomposer']=ck
            m=make_model(c,1024,torch.device('cpu'),False);b=example()
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):o=m(b['response'],b['obstacle'],b['window'])
            l=decomposition_training_loss(o,b,**c['loss'])
            torch.testing.assert_close(l['loss'],.5*l['count_task']+.5*l['shape_task'])
            self.assertIsNone(torch.autograd.grad(l['shape_task'],o['count_logits'],retain_graph=True,allow_unused=True)[0])
            l['loss'].backward()
            for name,p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                else:self.assertTrue(name.startswith('intensity_head.'))
            self.assertNotIn('reconstruction',l);self.assertNotIn('intensity',l)

    def test_metrics_match_actual_enabled_masks_missing_layers_fail_and_permutation_is_free(self):
        b=example();r=b['layer_mask'].sum(1)
        o={'mask_probability':b['teacher_mask'].roll(1,1),'active_slots':b['layer_mask'].roll(1,1),'predicted_count':r}
        pc=primary_counts(o,b);self.assertEqual(pc[2].tolist(),[1,1,2,2])
        o['active_slots'].zero_();o['active_slots'][0,1]=True;o['predicted_count']=torch.tensor([1])
        pc=primary_counts(o,b);self.assertEqual(pc[2].tolist(),[1,0,2,1])
        self.assertEqual(summarize_primary(pc)['layer_accuracy'],.5)

    def test_teacher_order_and_strength_values_cannot_change_the_training_objective(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False);b=example()
        o=m(b['response'],b['obstacle'],b['window'])
        a=decomposition_training_loss(o,b,**c['loss'])
        perm=torch.tensor([1,3,0,2]);d={**b,
            'teacher_mask':b['teacher_mask'][:,perm],
            'layer_mask':b['layer_mask'][:,perm],
            'teacher_intensity':torch.rand_like(b['teacher_intensity'])*100}
        e=decomposition_training_loss(o,d,**c['loss'])
        torch.testing.assert_close(a['loss'],e['loss'])
        torch.testing.assert_close(a['shape_task'],e['shape_task'])

    def test_large_cache_launcher_keeps_all_added_windows_and_freezes_warm_start(self):
        from configure_sequential import configure
        import configure_sequential as launch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,source,index,old=fixture(root)
            phase=root/'phase';(phase/'configs').mkdir(parents=True)
            (phase/'configs/masks_sequential.json').write_text((ROOT/'Phase2_Multi_Source/configs/masks_sequential.json').read_text())
            with patch.object(launch,'PHASE2',phase):path=configure(source_root=str(source),run_name='new')
            c=load_config(path);self.assertEqual(c['label_preparation']['record_counts'],index['counts'])
            self.assertEqual(index['counts']['train'],4)
            frozen=Path(c['training']['initial_decomposition_checkpoint']);before=frozen.read_bytes()
            c['training'].update(device='cpu',amp_dtype='none',num_threads=1,epochs=2,patience=5,
                 batch_size=1,target_global_batch_size=1,prefetch_batches=0,evaluate_after_training=False,
                 evaluation_bootstrap_repetitions=2)
            c['training'].pop('model_selection',None)
            bound=prepare_cached_labels(c);self.assertEqual(bound['records'],index['records']);self.assertFalse(data.calls)
            prepare_cached_labels(c)  # repeat cache binding, including a source cache view
            m=make_model(c,1024,torch.device('cpu'),True)
            original=make_model(old,1024,torch.device('cpu'),False)
            original.load_state_dict(torch.load(frozen,weights_only=True)['model'])
            b=example()
            with torch.no_grad():
                ctx=m._context(b['response'],b['obstacle'],b['window'])
                a=m._count(ctx,b['response'],b['window'])['count_logits']
                d=original(b['response'],b['obstacle'],b['window'])['count_logits']
            torch.testing.assert_close(a,d)
            train(c,Path(c['output_root']),bound,epoch_limit=1)
            train(c,Path(c['output_root']),bound,epoch_limit=2)
            saved=torch.load(Path(c['output_root'])/'models/seed_0/last.pt',weights_only=True)
            self.assertEqual(saved['epoch'],2);self.assertEqual(frozen.read_bytes(),before)
            from evaluate_sequential import evaluate_with_reference
            report=evaluate_with_reference(c,Path(c['output_root']),bound,repetitions=2)
            self.assertEqual(report['windows'],index['counts']['test'])
            self.assertEqual(set(report['metrics']),{'count_accuracy','layer_accuracy'})
            self.assertFalse(report['inference_teacher_inputs']);self.assertEqual(report['constraint_audit']['negative_residuals'],0)

    def test_equal_selection_does_not_use_strength_or_reconstruction(self):
        from count_objective import selection_key
        opt={'metric':'count_shape_equal','max_single_layer_drop':.005,'max_overall_drop':.005}
        a={'count_group_macro':.8,'shape_group_macro':.6,'count_accuracy':.9,'layer_accuracy':.7,'loss':1}
        b={**a,'count_group_macro':.7,'shape_group_macro':.7}
        self.assertEqual(selection_key(a,opt),selection_key(b,opt))

    def test_training_schedule_and_class_weights_have_explicit_scales(self):
        from sequential_objective import learning_rate_scale,class_weight
        self.assertEqual(learning_rate_scale(1,60),.5)
        self.assertEqual(learning_rate_scale(2,60),1)
        self.assertAlmostEqual(learning_rate_scale(60,60),.1)
        x=class_weight([1.,2.,3.,4.,1.],torch.tensor([1,3]))
        torch.testing.assert_close(x,torch.tensor([2.,4.]))

    def test_completed_windows_are_not_redecoded_and_full_horizon_probe_is_equivalent(self):
        c=configuration();m=make_model(c,1024,torch.device('cpu'),False).eval();b=example()
        y=b['response'].repeat(2,1,1);ob=b['obstacle'].repeat(2,1,1);window=b['window'].repeat(2,1)
        count=m._count;step=m._step;sizes=[]
        def mixed(*args):o=count(*args);o['predicted_count']=torch.tensor([1,3]);return o
        def observed(context,r,*args):sizes.append(len(r));return step(context,r,*args)
        with patch.object(m,'_count',mixed),patch.object(m,'_step',observed),torch.inference_mode():
            a=m(y,ob,window)
            self.assertEqual(sizes,[2,1,1]);sizes.clear();m.calibration_full_horizon=True
            d=m(y,ob,window)
            self.assertEqual(sizes,[2,2,2,2])
        torch.testing.assert_close(a['mask_probability'],d['mask_probability'])
        torch.testing.assert_close(a['residual'],d['residual'])

    def test_config_rejects_process_supervision_and_reconstruction_loss(self):
        c=configuration();c['training']['teacher_forcing_epochs']=2
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.json';p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'teacher forcing'):load_config(p)
            c['training'].pop('teacher_forcing_epochs');c['loss']['reconstruction_weight']=1
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'two equal tasks'):load_config(p)

    def test_tmux_frontend_runs_offline_and_preserves_config_on_failed_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';config=root/'config.json';pointer=root/'pointer'
            config.write_text(json.dumps({'output_root':str(out)}))
            mock=root/'tmux';mock.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit "${MOCK_SESSION_RUNNING:-1}";fi\nexit 0\n');mock.chmod(0o755)
            frontend=root/'frontend.sh'
            text=(ROOT/'run_sequential_masks_tmux.sh').read_text()
            text=text.replace('ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',f'ROOT={shlex.quote(str(ROOT))}')
            text=text.replace('pointer="$ROOT/Phase2_Multi_Source/outputs/sequential_v6_launcher/$session.txt"',f'pointer={shlex.quote(str(pointer))}')
            frontend.write_text(text)
            env={**os.environ,'PATH':str(root)+os.pathsep+os.environ['PATH'],'RIND_PYTHON':sys.executable,
                 'RIND_MASKS_CONFIG':str(config),'RIND_TMUX_SESSION':'rind_seq_test'}
            command=['bash',str(frontend),'start','all','--gpus','auto','--auto-batch']
            p=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
            self.assertIn('Started background training',p.stdout)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            saved=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_MASKS_CONFIG='+str(config.resolve()),saved)
            self.assertEqual(saved[-3:],['--gpus','auto','--auto-batch'])
            self.assertIn('RIND_TMUX_TASK=all',saved)
            self.assertEqual(pointer.read_text().strip(),str(config))
            pointer.write_text('preserve');env['MOCK_SESSION_RUNNING']='0'
            p=subprocess.run(command,env=env,text=True,capture_output=True)
            self.assertNotEqual(p.returncode,0);self.assertEqual(pointer.read_text(),'preserve')

    def test_two_rank_training_and_unequal_validation_partitions(self):
        try:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        except PermissionError:self.skipTest('loopback unavailable')
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,source,index,old=fixture(root)
            c=configuration();c['model'].update(old['model']);c['model']['decomposition_architecture']='sequential_count_shape_v6'
            c['output_root']=str(root/'ddp');c['cache_source_root']=str(source)
            c['training'].update(epochs=2,batch_size=1,target_global_batch_size=2,prefetch_batches=0,
                 evaluate_after_training=False,evaluate_initial_checkpoint=False,evaluate_test_after_training=False)
            c['training'].pop('model_selection',None);c['training'].pop('initial_decomposition_checkpoint',None)
            bound=prepare_cached_labels(c)
            mp.spawn(ddp_worker,args=(port,c,bound),nprocs=2,join=True)
            self.assertTrue((Path(c['output_root'])/'models/seed_0/best.pt').is_file())


if __name__=='__main__':unittest.main()
