import copy
import json
import os
import shlex
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from balanced_sampling import plan, draw_by_bucket
from decomposition_data import _batch_groups, batch_count
from mask_only_data import prepare_mask_dataset, MaskRecords, build_mask_cache
from extend_evaluation import prepare_extended_evaluation
from train_decomposition import train, load_config
from evaluate_masks import evaluate_with_reference
from evaluate_slots import _bootstrap
from search_cache import bind_cache
import test_masks_only as fixture

ROOT = Path(__file__).resolve().parents[3]


class Pool:
    def __init__(self):
        self.entries = [{'intensities': [1.] * r, 'scene_id': i, 'view_id': 0}
                        for i, r in enumerate([1]*60 + [2]*24 + [3]*10 + [4]*4 + [5]*2 + [32])]
        self.window_sizes = [16 if i % 3 else 32 for i in range(len(self.entries))]
        self.candidate_counts = [0]*len(self.entries)
    def __len__(self):return len(self.entries)


class HeldoutDataset(fixture.LocalDataset):
    num_scenes = 9
    manifest = {'global_size': 1024, 'num_scenes': 9}
    scene_splits = {'train': [0,1,2], 'validation': [3,4,5], 'test': [6,7,8]}
    def __init__(self):self.calls = []
    def geometry_group(self, scene):return f'geometry-{scene}'
    def iter_scene_views(self, scene):return iter([(i,8) for i in range(4)])
    def get_scene(self, scene):
        row = super().get_scene(scene)
        row['local_views'] = np.array([[i*8,0,8] for i in range(4)])
        return row
    def get_region(self, scene,x,y,side):
        self.calls.append((scene,x//8))
        response = np.zeros((side,side)) if x==16 else np.ones((side,side)) if x==24 else fixture.example()
        return {'response':response,'obstacle':np.zeros((side,side),bool)}


def make_fixture(root):
    config = load_config(ROOT/'Phase2_Multi_Source/configs/masks_balanced_finetune.json')
    config.pop('evaluation_expansion');config.pop('cache_source_root')
    config['output_root'] = str(root/'source')
    config['data_root'] = str(root/'raw-not-required-for-fixture')
    config['model'].update(max_slots=4,hidden=64,fourier_frequencies=1,decomposition_hidden=32,slot_decoder_layers=1,mask_feature_dim=16)
    config['label_preparation'].update(record_counts={'train':2,'validation':1,'test':1},window_sizes=[8],max_views_per_scene=4)
    for k in ('sampling','initial_decomposition_checkpoint','evaluate_initial_checkpoint','compare_initial_checkpoint'):
        config['training'].pop(k,None)
    config['training'].update(device='cpu',amp_dtype='none',epochs=1,patience=3,batch_size=1,target_global_batch_size=2,
                              prefetch_batches=0,num_threads=1,learning_rate=1e-4,evaluation_bootstrap_repetitions=2)
    data = HeldoutDataset()
    index = prepare_mask_dataset(config,data)
    fine = copy.deepcopy(config)
    fine.update(output_root=str(root/'fine'),cache_source_root=config['output_root'],
                evaluation_expansion={'label_protocol_root':config['output_root'],'seed':20261101,'workers':1,'chunk_size':1,
                    'max_views_per_scene':4,'window_sizes':[8],'record_counts':{'validation':3,'test':3}})
    fine['training'].update(initial_decomposition_checkpoint=str(root/'source/models/seed_0/best.pt'),
                            evaluate_initial_checkpoint=True,compare_initial_checkpoint=True,
                            sampling={'epoch_windows':8,'groups':[{'name':'all','min_layers':1,'max_layers':32,'probability':1.}]})
    data.calls.clear()
    return data,config,fine,index


class BalancedFinetuneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_sampling_quotas_seed_capacity_and_native_size(self):
        pool = Pool()
        options = json.loads((ROOT/'Phase2_Multi_Source/configs/masks_balanced_finetune.json').read_text())['training']['sampling']
        options['epoch_windows'] = 1000
        _, report = plan(pool, options)
        self.assertEqual([g['draws_before_ddp_padding'] for g in report['groups']], [300,300,300,100])
        first = draw_by_bucket(pool,options,np.random.default_rng(42))
        self.assertEqual(first,draw_by_bucket(pool,options,np.random.default_rng(42)))
        counts=[0]*4
        for key,indices in first.items():
            for i in indices:
                self.assertEqual(pool.window_sizes[i],key[0])
                r=len(pool.entries[i]['intensities']);counts[0 if r==1 else 1 if r==2 else 2 if r<=4 else 3]+=1
        self.assertEqual(counts,[300,300,300,100])
        self.assertEqual(len(pool.entries[-1]['intensities']),32)
        both=[_batch_groups(pool,2,np.random.default_rng(42),rank,2,True,options) for rank in (0,1)]
        serial=_batch_groups(pool,4,np.random.default_rng(42),0,1,True,options)
        self.assertEqual(len(serial),batch_count(pool,2,2,0,True,options))
        for a,b,full in zip(*both,serial):
            self.assertEqual([i for pair in zip(a,b) for i in pair],full)
        evaluation=_batch_groups(pool,7,None,0,1,False,options)
        self.assertEqual(sorted(i for batch in evaluation for i in batch),list(range(len(pool))))

    def test_invalid_or_empty_sampling_groups_rejected(self):
        pool=Pool();base={'epoch_windows':100,'groups':[{'name':'all','min_layers':1,'max_layers':32,'probability':1.}]}
        bad=copy.deepcopy(base);bad['groups'][0]['probability']=.9
        with self.assertRaisesRegex(ValueError,'sum to one'):plan(pool,bad)
        bad=copy.deepcopy(base);bad['groups'].append({'name':'overlap','min_layers':3,'max_layers':4,'probability':0.})
        with self.assertRaisesRegex(ValueError,'cover every'):plan(pool,bad)
        bad={'epoch_windows':100,'groups':[{'name':'none','min_layers':30,'max_layers':31,'probability':1.},
                                          {'name':'rest','min_layers':1,'max_layers':29,'probability':0.},
                                          {'name':'cap','min_layers':32,'max_layers':32,'probability':0.}]}
        with self.assertRaisesRegex(ValueError,'no training windows'):plan(pool,bad)

    def test_tmux_entry_selects_new_output_config_and_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=root/'custom.json';out=root/'out'
            cfg.write_text(json.dumps({'output_root':str(out)}))
            tmux=root/'tmux';tmux.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n');tmux.chmod(0o755)
            env={**os.environ,'PATH':str(root)+os.pathsep+os.environ['PATH'],'RIND_PYTHON':sys.executable,'RIND_MASKS_CONFIG':str(cfg)}
            env.pop('RIND_TMUX_SESSION',None)
            result=subprocess.run(['bash',str(ROOT/'run_balanced_masks_tmux.sh'),'start','all','--gpus','8'],env=env,text=True,capture_output=True,check=True)
            self.assertIn('rind_masks_balanced',result.stdout)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_MASKS_CONFIG='+str(cfg.resolve()),command)
            self.assertIn('RIND_TMUX_TASK=all',command)
            self.assertEqual(command[-2:],['--gpus','8'])

    def test_natural_extension_resume_keeps_train_bytes_and_legacy_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data,base,config,old=make_fixture(root)
            source=Path(base['output_root']);output=Path(config['output_root'])
            original={p.name:p.read_bytes() for p in (source/'mask_cache').iterdir()}
            with self.assertRaises(InterruptedError):prepare_extended_evaluation(config,data,stop_after_chunks=1)
            self.assertFalse((output/'mask_records.json').exists())
            index=prepare_extended_evaluation(config,data)
            self.assertEqual(index['counts'],{'train':2,'validation':3,'test':3})
            self.assertEqual(index['records']['train'],old['records']['train'])
            for name,contents in original.items():
                self.assertEqual((source/'mask_cache'/name).read_bytes(),contents)
                target=output/'mask_cache'/name
                if name.startswith('train_'):self.assertTrue(target.is_symlink())
                self.assertEqual(target.read_bytes()[:len(contents)],contents)
            for split in ('validation','test'):
                self.assertEqual(index['records'][split][:1],old['records'][split])
                old_groups={data.geometry_group(r['scene_id']) for r in old['records'][split]}
                self.assertTrue(all(data.geometry_group(r['scene_id']) not in old_groups for r in index['records'][split][1:]))
                records=MaskRecords(index['records'][split],output/'mask_cache',split,4)
                for row in records:
                    np.testing.assert_allclose((row['teacher_mask']*row['teacher_intensity'][:,None,None]).sum(0),row['response'],atol=1e-7)
            data.calls.clear()
            with patch('dataset_update.open_dataset',side_effect=AssertionError('completed cache needs no raw data')):
                self.assertEqual(prepare_extended_evaluation(config),index)
            self.assertEqual(data.calls,[])

    def test_extension_checksum_and_changed_targets_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data,_,config,_=make_fixture(root)
            with self.assertRaises(InterruptedError):prepare_extended_evaluation(config,data,stop_after_chunks=1)
            path=next((root/'fine/evaluation_append_chunks/validation').glob('*.npz'))
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['response'][0]+=1
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'checksum'):prepare_extended_evaluation(config,data)
            changed=copy.deepcopy(config);changed['evaluation_expansion']['record_counts']['test']=4
            with self.assertRaisesRegex(ValueError,'differs'):prepare_extended_evaluation(changed,data)

    def test_weights_only_finetune_baseline_and_same_expanded_test_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data,base,config,source_index=make_fixture(root)
            train(base,root/'source',source_index)
            original=torch.load(root/'source/models/seed_0/best.pt',weights_only=True)
            index=prepare_extended_evaluation(config,data)
            config['training'].update(epochs=2,learning_rate=1e-30)
            train(config,root/'fine',index,epoch_limit=1)
            first=torch.load(root/'fine/models/seed_0/last.pt',weights_only=True)
            train(config,root/'fine',index,epoch_limit=2)
            last=torch.load(root/'fine/models/seed_0/last.pt',weights_only=True)
            self.assertEqual(first['history'],last['history'][:1])
            best=torch.load(root/'fine/models/seed_0/best.pt',weights_only=True)
            self.assertEqual(best['epoch'],0)
            for k,v in original['model'].items():torch.testing.assert_close(best['model'][k],v)
            self.assertEqual(last['history'][0]['train']['draws_R_2'],8)
            report=evaluate_with_reference(config,root/'fine',index,repetitions=2)
            self.assertEqual(report['windows'],3)
            self.assertEqual(report['by_cohort']['legacy']['windows'],1)
            self.assertEqual(report['by_cohort']['added']['windows'],2)
            comparison=json.loads((root/'fine/evaluation/finetuning_comparison.json').read_text())
            self.assertTrue(all(abs(v['change'])<1e-12 for v in comparison['metrics'].values()))

    def test_cluster_bootstrap_vectorization_preserves_original_definition(self):
        rows=[{'scene_id':scene,'metrics':{'x':float(scene+view),'y':float(scene-view)}}
              for scene in range(5) for view in range(scene+1)]
        rng=np.random.default_rng(4);expected={}
        for key in ('x','y'):
            values=[]
            for _ in range(75):
                sample=rng.integers(0,5,size=5)
                values.append(np.mean([np.mean([r['metrics'][key] for r in rows if r['scene_id']==s]) for s in sample]))
            expected[key]=np.quantile(values,[.025,.975])
        got=_bootstrap(rows,['x','y'],75,4)
        for key in expected:np.testing.assert_allclose([got[key]['lower'],got[key]['upper']],expected[key],atol=1e-14)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'Gloo unavailable')
    def test_balanced_ddp_updates_match_serial_with_varied_layer_counts(self):
        try:
            with socket.socket() as probe:probe.bind(('127.0.0.1',0))
        except PermissionError:self.skipTest('sandbox disallows local rendezvous sockets')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=load_config(ROOT/'Phase2_Multi_Source/configs/masks_balanced_finetune.json')
            config.pop('evaluation_expansion');config.pop('cache_source_root')
            config['model'].update(max_slots=8,hidden=64,fourier_frequencies=1,decomposition_hidden=32,slot_decoder_layers=1,mask_feature_dim=16)
            for k in ('initial_decomposition_checkpoint','compare_initial_checkpoint','evaluate_initial_checkpoint'):
                config['training'].pop(k,None)
            config['training'].update(device='cpu',amp_dtype='none',epochs=1,patience=3,batch_size=1,target_global_batch_size=4,
                prefetch_batches=0,num_threads=1,evaluate_test_after_training=False)
            config['training']['sampling']['epoch_windows']=16
            for g in config['training']['sampling']['groups']:g['probability']=.25
            labels={s:[] for s in ('train','validation','test')}
            for split,counts in [('train',[1,1,2,2,3,4,5,7]),('validation',[2]),('test',[2])]:
                for view,r in enumerate(counts):
                    masks=np.zeros((r,8,8),bool)
                    for i in range(r):masks[i,:7,i]=True
                    strengths=np.linspace(.1,.3,r)
                    path=root/f'{split}-{view}.npz'
                    np.savez(path,response=(masks*strengths[:,None,None]).sum(0),obstacle=np.zeros((8,8),bool),
                             masks=masks,intensities=strengths,window=[0,0,8])
                    labels[split].append({'file':str(path),'scene_id':view if split=='train' else 100 if split=='validation' else 101,'view_id':0,'size':8,'layer_count':r})
            source=root/'source';index=build_mask_cache(config,labels,source)
            output=root/'ddp';view=bind_cache(output,source,index,8)
            config['output_root']=str(output);path=root/'config.json';path.write_text(json.dumps(config))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
            for k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_PORT','MASTER_ADDR'):env.pop(k,None)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                     str(ROOT/'Phase2_Multi_Source/research/implicit_slots/train_decomposition.py'),'train','--config',str(path)]
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            ddp=torch.load(output/'models/seed_0/last.pt',weights_only=True)
            config['training']['batch_size']=2;config['output_root']=str(root/'serial')
            serial_index=bind_cache(root/'serial',source,index,8)
            train(config,root/'serial',serial_index)
            serial=torch.load(root/'serial/models/seed_0/last.pt',weights_only=True)
            for k,v in ddp['model'].items():torch.testing.assert_close(v,serial['model'][k],atol=1e-6,rtol=1e-5)
            for name in ('1','2','3_4','5plus'):
                self.assertEqual(ddp['history'][0]['train'][f'draws_R_{name}'],4)


if __name__=='__main__':unittest.main()
