import copy
import json
from pathlib import Path
import tempfile
import unittest
import socket
import os
import shlex
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from count_objective import summarize_confusion, selection_key, eligible
from decomposition_model import stable_count_logits,poisson_binomial_logits
from expand_complex_masks import prepare_complex
from mask_only_data import prepare_mask_dataset, MaskRecords
from train_decomposition import train
from evaluate_masks import evaluate_with_reference
import test_masks_only as fixture


def _count_eval_worker(rank, port, config, index, output):
    import torch.distributed as dist
    from train_decomposition import make_model, run_epoch
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        torch.manual_seed(55);torch.set_num_threads(1)
        model=make_model(config,1024,torch.device('cpu'),initialize=False)
        records=MaskRecords(index['records']['validation'],Path(config['output_root'])/'mask_cache','validation',4)
        metrics=run_epoch(model,records,config,torch.device('cpu'),0,rank,2)
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(metrics))
    finally:dist.destroy_process_group()


class Dataset(fixture.LocalDataset):
    num_scenes=5
    manifest={'global_size':1024,'num_scenes':5}
    scene_splits={'train':[0,3,4],'validation':[1],'test':[2]}
    def __init__(self):self.calls=[]
    def geometry_group(self,scene):return f'geometry-{scene}'
    def get_source_params(self,scene):return np.array([[800.+scene,800.,.5]])
    def iter_scene_views(self,scene):return iter([(i,8) for i in range(6)])
    def get_scene(self,scene):
        value=super().get_scene(scene)
        value['local_views']=np.array([[i*8,0,8] for i in range(6)])
        # Two view IDs with the same physical ROI: never count as new information.
        value['local_views'][3]=value['local_views'][2]
        return value
    def get_region(self,scene,x,y,side):
        self.calls.append((scene,x))
        if x==8:response=np.ones((8,8))
        elif x==32:
            response=np.zeros((8,8));response[:2,:2]=.3
        elif x in (16,40):
            response=np.zeros((8,8));response[:2,:2]=.3;response[:2,4:6]=.5;response[4:6,:2]=.7
        else:response=fixture.example()
        return {'response':response,'obstacle':np.zeros((8,8),bool)}


def prepare_fixture(root):
    data=Dataset();base=fixture.MasksOnlyTests()._config(root)
    base['label_preparation'].update(record_counts={'train':2,'validation':1,'test':1},max_views_per_scene=1)
    base['model']['max_slots']=4
    base['training'].update(epochs=1,evaluate_after_training=False)
    old=prepare_mask_dataset(base,data)
    new=copy.deepcopy(base)
    new['output_root']=str(root/'expanded')
    new['cache_source_root']=base['output_root']
    new['label_preparation']['record_counts']['train']=4
    new['loss']['stable_count_loss']=True
    new['training_expansion']={'label_protocol_root':base['output_root'],'seed':123,
        'additional_complex_windows':2,'min_layers':3,'window_sizes':[8],
        'max_views_per_scene':6,'max_added_per_configuration':4,'max_added_per_geometry':8,
        'workers':1,'chunk_size':1}
    data.calls.clear()
    return data,base,new,old


class CountFirstTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_log_count_matches_reference_and_has_extreme_gradients(self):
        logits=torch.linspace(-3,3,12).reshape(3,4).requires_grad_()
        actual=stable_count_logits(logits)
        expected=poisson_binomial_logits(logits.sigmoid())
        torch.testing.assert_close(actual.exp(),expected.exp(),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(actual.exp().sum(1),torch.ones(3),rtol=1e-5,atol=1e-7)
        saturated=torch.full((1,4),30.,requires_grad=True)
        old_loss=torch.nn.functional.cross_entropy(poisson_binomial_logits(saturated.sigmoid()),torch.tensor([2]))
        old_grad=torch.autograd.grad(old_loss,saturated)[0]
        new_loss=torch.nn.functional.cross_entropy(stable_count_logits(saturated),torch.tensor([2]))
        new_grad=torch.autograd.grad(new_loss,saturated)[0]
        self.assertTrue(torch.isfinite(new_loss))
        self.assertTrue(torch.isfinite(new_grad).all())
        self.assertTrue((new_grad.abs()>.1).all())
        self.assertTrue((old_grad==0).all())
        self.assertAlmostEqual(new_loss.item(),60-np.log(6),places=4)

    def test_count_selection_is_not_loss_selection_and_has_guards(self):
        matrix=np.zeros((33,33),np.int64)
        matrix[1,1]=90;matrix[1,2]=10;matrix[3,3]=5;matrix[3,1]=5
        summary=summarize_confusion(matrix)
        self.assertEqual(summary['count_by_group']['R1']['accuracy'],.9)
        self.assertAlmostEqual(summary['count_group_macro'],.7)
        self.assertIsNone(summary['count_by_group']['R4']['accuracy'])
        old={**summary,'count_accuracy':.86,'count_mae':.2,'loss':.2}
        new=copy.deepcopy(old);new.update(count_group_macro=.8,count_accuracy=.88,count_mae=.15,loss=.3)
        opts={'metric':'count_group_macro','max_single_layer_drop':.005,'max_overall_drop':.005}
        self.assertGreater(selection_key(new,opts),selection_key(old,opts))
        self.assertLess(selection_key(new),selection_key(old))
        self.assertTrue(eligible(new,old,opts))
        new['count_by_group']['R1']['accuracy']=.89
        self.assertFalse(eligible(new,old,opts))
        with self.assertRaises(ValueError):selection_key(old,{'metric':'loss'})

    def test_append_real_complex_labels_and_preserve_source_eval_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            source=Path(base['output_root']);out=Path(new['output_root'])
            original={p.name:p.read_bytes() for p in (source/'mask_cache').iterdir()}
            with self.assertRaises(InterruptedError):prepare_complex(new,data,stop_after_chunks=1)
            self.assertFalse((out/'mask_records.json').exists())
            result=prepare_complex(new,data)
            self.assertEqual(result['counts'],{'train':4,'validation':1,'test':1})
            self.assertEqual(result['records']['train'][:2],old['records']['train'])
            for split in ('validation','test'):
                self.assertEqual(result['records'][split],old['records'][split])
            added=result['records']['train'][2:]
            self.assertTrue(all(len(r['intensities'])>=3 for r in added))
            self.assertEqual(len({r['configuration_window_fingerprint'] for r in added}),len(added))
            for name,value in original.items():
                self.assertEqual((source/'mask_cache'/name).read_bytes(),value)
                self.assertTrue((out/'mask_cache'/name).read_bytes().startswith(value))
            records=MaskRecords(result['records']['train'],out/'mask_cache','train',4)
            for i in range(len(records)):
                item=records[i]
                np.testing.assert_allclose((item['teacher_mask']*item['teacher_intensity'][:,None,None]).sum(0),item['response'],atol=1e-7)
            data.calls.clear()
            self.assertEqual(prepare_complex(new),result)  # completed cache needs no raw data
            self.assertEqual(data.calls,[])

    def test_shortage_is_not_filled_with_duplicates_or_single_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            new['training_expansion']['additional_complex_windows']=100
            new['label_preparation']['record_counts']['train']=102
            with self.assertRaisesRegex(RuntimeError,'never fill by replacement'):prepare_complex(new,data)
            out=Path(new['output_root'])
            self.assertFalse((out/'mask_records.json').exists())
            report=json.loads((out/'complex_availability.json').read_text())
            self.assertLess(report['accepted_new'],100)
            self.assertGreater(report['statistics'].get('duplicate_configuration_window',0),0)
            self.assertGreater(report['statistics'].get('too_few_layers',0),0)
            self.assertEqual(report['candidate_renders'],0)

    def test_protocol_change_corrupt_chunk_and_geometry_leak_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            with self.assertRaises(InterruptedError):prepare_complex(new,data,stop_after_chunks=1)
            changed=copy.deepcopy(new);changed['training_expansion']['seed']+=1
            with self.assertRaisesRegex(ValueError,'differs'):prepare_complex(changed,data)
            path=next((Path(new['output_root'])/'complex_chunks').glob('*.npz'))
            with np.load(path) as z:content={k:np.array(z[k]) for k in z.files}
            content['response'][0]+=1
            np.savez_compressed(path,**content)
            with self.assertRaisesRegex(ValueError,'corrupt'):prepare_complex(new,data)
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            data.geometry_group=lambda scene:'same-geometry'
            with self.assertRaisesRegex(ValueError,'held-out'):prepare_complex(new,data)

    def test_count_first_warm_start_train_and_reference_full_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            train(base,Path(base['output_root']),old)
            new['training'].update(initial_decomposition_checkpoint=str(Path(base['output_root'])/'models/seed_0/last.pt'),
                evaluate_initial_checkpoint=True,compare_initial_checkpoint=True,track_count_metrics=True,
                model_selection={'metric':'count_group_macro','max_single_layer_drop':.005,'max_overall_drop':.005})
            index=prepare_complex(new,data)
            train(new,Path(new['output_root']),index)
            report=json.loads((Path(new['output_root'])/'models/seed_0/training.json').read_text())
            self.assertIn('count_group_macro',report['initial_validation'])
            self.assertIn('count_by_R',report['history'][0]['validation'])
            self.assertEqual(sum(map(sum,report['history'][0]['validation']['count_confusion'])),1)
            self.assertEqual(report['selection_options']['metric'],'count_group_macro')
            result=evaluate_with_reference(new,Path(new['output_root']),index,repetitions=2)
            self.assertEqual(result['windows'],1)
            # Exact command reuses the last optimizer/RNG state, not baseline.
            train(new,Path(new['output_root']),index)
            again=json.loads((Path(new['output_root'])/'models/seed_0/training.json').read_text())
            self.assertEqual(again['trained_through_epoch'],1)

    def test_count_best_can_accept_higher_loss_and_guard_single_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,base,new,old=prepare_fixture(Path(tmp))
            train(base,Path(base['output_root']),old)
            new['training'].update(epochs=2,initial_decomposition_checkpoint=str(Path(base['output_root'])/'models/seed_0/last.pt'),
                evaluate_initial_checkpoint=True,track_count_metrics=True,
                model_selection={'metric':'count_group_macro','max_single_layer_drop':.005,'max_overall_drop':.005})
            index=prepare_complex(new,data)
            def fake_epoch(model,records,config,device,epoch,rank,world,optimizer=None,*args,**kwargs):
                # Test the checkpoint decision independently from optimization.
                v={'loss':.2,'count_accuracy':.9,'count_mae':.1,'mask_dice':.9,
                   'count_group_macro':.6,'count_by_group':{'R1':{'accuracy':.95,'windows':1}}}
                if epoch==1:v.update(loss=.3,count_group_macro=.8,count_accuracy=.91)
                if epoch==2:
                    v.update(loss=.1,count_group_macro=.9,count_accuracy=.92)
                    v['count_by_group']['R1']['accuracy']=.9
                return v
            with patch('train_decomposition.run_epoch',side_effect=fake_epoch):
                train(new,Path(new['output_root']),index)
            out=Path(new['output_root'])
            result=json.loads((out/'models/seed_0/training.json').read_text())
            self.assertEqual(result['best_epoch'],1)
            self.assertEqual(result['selected_validation_loss'],.3)
            self.assertEqual(result['best_validation_loss'],.1)
            self.assertTrue(result['history'][0]['selected_as_best'])
            self.assertFalse(result['history'][1]['selection_eligible'])
            alternate=torch.load(out/'models/seed_0/loss_best.pt',weights_only=True)
            self.assertEqual(alternate['epoch'],2)

    def test_distributed_count_metrics_include_empty_validation_rank(self):
        try:
            sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        except PermissionError:self.skipTest('sandbox blocks local Gloo socket')
        import torch.multiprocessing as mp
        from train_decomposition import make_model,run_epoch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,new,index=prepare_fixture(root)
            base['training']['track_count_metrics']=True
            base['loss']['stable_count_loss']=True
            mp.spawn(_count_eval_worker,args=(port,base,index,str(root)),nprocs=2,join=True)
            torch.manual_seed(55)
            model=make_model(base,1024,torch.device('cpu'),initialize=False)
            records=MaskRecords(index['records']['validation'],Path(base['output_root'])/'mask_cache','validation',4)
            serial=run_epoch(model,records,base,torch.device('cpu'),0,0,1)
            for rank in (0,1):
                parallel=json.loads((root/f'rank{rank}.json').read_text())
                self.assertEqual(parallel['count_confusion'],serial['count_confusion'])
                self.assertEqual(parallel['count_by_R'],serial['count_by_R'])
                self.assertAlmostEqual(parallel['count_group_macro'],serial['count_group_macro'])
                self.assertAlmostEqual(parallel['loss'],serial['loss'],places=5)

    def test_count_tmux_selects_new_config_output_and_gpu_arguments(self):
        root=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);out=directory/'output';cfg=directory/'custom.json'
            cfg.write_text(json.dumps({'output_root':str(out)}))
            tmux=directory/'tmux'
            tmux.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n')
            tmux.chmod(0o755)
            env={**os.environ,'PATH':str(directory)+os.pathsep+os.environ['PATH'],
                 'RIND_PYTHON':sys.executable,'RIND_MASKS_CONFIG':str(cfg)}
            env.pop('RIND_TMUX_SESSION',None)
            result=subprocess.run(['bash',str(root/'run_count_masks_tmux.sh'),'start','all','--gpus','auto'],
                                  env=env,capture_output=True,text=True,check=True)
            self.assertIn('rind_masks_count_first',result.stdout)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_MASKS_CONFIG='+str(cfg.resolve()),command)
            self.assertIn('RIND_TMUX_TASK=all',command)
            self.assertEqual(command[-2:],['--gpus','auto'])


if __name__=='__main__':unittest.main()
