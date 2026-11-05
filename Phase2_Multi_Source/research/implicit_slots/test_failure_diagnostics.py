import copy,gzip,hashlib,json,os,shlex,socket,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from failure_diagnostics import diagnostic_rows,aggregate,write_artifacts,portable_bundle,rate_summary
from evaluation_snapshot import create_snapshot,load_snapshot
from evaluate_masks import metric_rows,evaluate
from train_decomposition import make_model,train
from test_disentangled import config as config_for_test
from test_count_first import prepare_fixture

ROOT=Path(__file__).resolve().parents[3]


def case(kind):
    target=torch.zeros(1,3,8,8);target[0,0,:6,:4]=1;target[0,1,:6,3:7]=1
    truth=torch.tensor([[.3,.5,0.]])
    masks=target.clone();a=torch.tensor([[.3,.5,.2]]);on=torch.tensor([[True,True,False]])
    if kind=='count':on[:]=True  # extra zero-support slot: only count fails
    if kind=='selection':on[:]=torch.tensor([[True,False,True]])
    if kind=='matching':
        masks[:,2]=target[:,1];a[:,2]=.5;on[:]=torch.tensor([[True,False,True]])
    if kind=='shape':
        truth[:,1]=.05;a[:,1]=.05;masks[0,1,0,3:6]=0
    if kind=='strength':a[:,0]=.339  # 13% error, but total response error <5%
    if kind=='reconstruction':a[:,:2]=truth[:,:2]*1.07
    if kind=='multiple':masks[:,1]=0;a[:,0]=.6;on[:,1]=False
    y=(target*truth[:,:,None,None]).sum(1)
    reconstruction=(masks*(a*on)[:,:,None,None]).sum(1)
    b={'response':y,'obstacle':torch.zeros_like(y,dtype=torch.bool),'teacher_mask':target.bool(),
       'teacher_intensity':truth,'layer_mask':truth>0,'scene_id':torch.tensor([0]),'view_id':torch.tensor([0]),
       'window':torch.tensor([[0.,0.,8.]])}
    logits=torch.where(masks>0,torch.tensor(20.),torch.tensor(-20.))
    r={'mask_probability':masks,'mask_logits':logits,'intensity':a,'active_slots':on,
       'presence_probability':on.float(),'presence_logits':torch.where(on,20.,-20.),
       'count_logits':torch.nn.functional.one_hot(on.sum(1),4).float()*20,
       'reconstruction':reconstruction}
    return r,b,torch.tensor([[0,1,2]])


def _snapshot_worker(rank,port,source,output):
    import torch.distributed as dist
    from evaluation_snapshot import create_snapshot as original
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    def mutate_after_freeze(*args):
        meta=original(*args)
        new={'schema':'phase2-local-mask-decomposition-v1','config':{},'model':{'x':torch.tensor([2.])},
             'epoch':11,'signature':'signed'}
        torch.save(new,source)
        return meta
    try:
        with patch('evaluation_snapshot.create_snapshot',side_effect=mutate_after_freeze):
            state,meta=load_snapshot(source,output,rank,2)
        (Path(output)/f'rank{rank}.json').write_text(json.dumps({'epoch':state['epoch'],
             'x':float(state['model']['x'][0]),'sha':meta['source_checkpoint_sha256']}))
    finally:dist.destroy_process_group()


class FailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_each_factor_and_relaxed_thresholds_match_original_strict(self):
        expected={'perfect':[],'count':['count'],'selection':['slot_selection','reconstruction'],
                  'matching':['slot_selection'],'shape':['shape'],'strength':['strength'],
                  'reconstruction':['reconstruction'],'multiple':['count','shape','strength','reconstruction']}
        rows=[]
        for i,(kind,bad) in enumerate(expected.items()):
            r,b,a=case(kind);row=diagnostic_rows(r,b,a)[0];row['scene_id']=i//2;row['view_id']=i
            reported=row['views']['reported']['protocols']['strict']
            self.assertEqual(reported['failed_factors'],bad,kind)
            original=metric_rows(r,b,a)[0]['metrics']['strict_decomposition_success']
            self.assertEqual(reported['success'],bool(original),kind)
            if kind in ('shape','strength','reconstruction'):
                self.assertTrue(row['views']['deployed']['protocols']['relaxed']['success'],kind)
            if kind=='count':self.assertIsNone(reported['slot_selection_given_correct_count'])
            rows.append(row)
        report=aggregate(rows,repetitions=20,seed=3)
        patterns=report['overall']['reported']['strict']['exact_failure_patterns']
        self.assertEqual(sum(patterns.values()),len(rows))
        self.assertEqual(patterns['pass'],1)
        self.assertGreater(report['matching_sensitivity']['strict']['reported_fail_deployed_pass'],0)

    def test_active_only_matching_avoids_inactive_duplicate_penalty(self):
        r,b,a=case('matching');row=diagnostic_rows(r,b,a)[0]
        self.assertFalse(row['views']['reported']['protocols']['strict']['success'])
        self.assertTrue(row['views']['deployed']['protocols']['strict']['success'])
        selected=[x['slot'] for x in row['layers'] if x['matching']=='deployed']
        self.assertEqual(set(selected),{0,2})
        self.assertEqual(row['teacher_strength_relative_l1'],0)

    def test_counterfactuals_distinguish_shape_and_strength_and_are_not_deployed(self):
        for kind,key in (('shape','teacher_mask_relative_l1'),('strength','teacher_strength_relative_l1')):
            r,b,a=case(kind);before=r['reconstruction'].clone();row=diagnostic_rows(r,b,a)[0]
            self.assertLess(row[key],1e-6,kind)
            self.assertGreater(row['deployment_relative_l1'],.001,kind)
            torch.testing.assert_close(before,r['reconstruction'])
        r,b,a=case('multiple');row=diagnostic_rows(r,b,a)[0]
        self.assertIsNone(row['teacher_strength_relative_l1'])

    def test_scene_cluster_bootstrap_denominators_and_na(self):
        rows=[{'scene_id':0}]*100+[{'scene_id':1}]
        x={'rate':np.array([1]*100+[0]),'conditional':np.zeros(101)}
        d={'rate':np.ones(101),'conditional':np.zeros(101)}
        stats=rate_summary(rows,x,d,50,4)
        self.assertAlmostEqual(stats['rate']['window_weighted'],100/101)
        self.assertEqual(stats['rate']['scene_balanced'],.5)
        self.assertEqual(stats['rate']['window_cluster_95ci']['scene_clusters'],2)
        self.assertIsNone(stats['conditional']['window_weighted'])
        one=rate_summary(rows[:100],{'r':np.ones(100)},{'r':np.ones(100)},50,4)
        self.assertIsNone(one['r']['window_cluster_95ci'])

    def test_snapshot_stays_fixed_after_last_checkpoint_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'last.pt'
            state={'schema':'phase2-local-mask-decomposition-v1','config':{},'model':{'x':torch.tensor([1.])},
                   'epoch':10,'signature':'signed','optimizer':{'large':'excluded'}}
            torch.save(state,source);meta=create_snapshot(source,root)
            state['epoch']=11;state['model']['x']=torch.tensor([2.]);torch.save(state,source)
            frozen=torch.load(meta['snapshot_path'],weights_only=True)
            self.assertEqual(frozen['epoch'],10);self.assertEqual(frozen['model']['x'].item(),1)
            self.assertNotIn('optimizer',frozen)
            current,newmeta=load_snapshot(source,root)
            self.assertEqual(current['epoch'],11)
            self.assertNotEqual(meta['source_checkpoint_sha256'],newmeta['source_checkpoint_sha256'])
            Path(newmeta['snapshot_path']).write_bytes(b'corrupt')
            with patch('evaluation_snapshot.create_snapshot',return_value=newmeta):
                with self.assertRaisesRegex(ValueError,'checksum'):load_snapshot(source,root)

    def test_softmask_strength_compensation_is_measured_separately(self):
        r,b,a=case('perfect');r['mask_probability']*=.8;r['intensity'][:,:2]/=.8
        r['mask_logits']=torch.logit(r['mask_probability'].clamp(1e-6,1-1e-6))
        r['reconstruction']=(r['mask_probability']*(r['intensity']*r['active_slots'])[:,:,None,None]).sum(1)
        row=diagnostic_rows(r,b,a)[0]
        report=aggregate([row],repetitions=0)
        stat=report['overall']['deployed']['strict']['softmask_strength_compensation']
        self.assertEqual(stat['shape_pass_strength_fail_layers'],2)
        self.assertEqual(stat['effective_inside_error_pass_layers'],2)
        self.assertLess(row['deployment_relative_l1'],1e-6)
        self.assertGreater(row['hard_mask_relative_l1'],.2)

    def test_two_rank_snapshot_keeps_same_epoch_if_last_changes_during_loading(self):
        try:
            probe=socket.socket();probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close()
        except PermissionError:self.skipTest('loopback ports unavailable in sandbox')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'last.pt'
            torch.save({'schema':'phase2-local-mask-decomposition-v1','config':{},'model':{'x':torch.tensor([1.])},
                        'epoch':10,'signature':'signed'},source)
            torch.multiprocessing.spawn(_snapshot_worker,args=(port,str(source),str(root)),nprocs=2,join=True)
            records=[json.loads((root/f'rank{r}.json').read_text()) for r in range(2)]
            self.assertEqual(records[0],records[1]);self.assertEqual(records[0]['epoch'],10)
            self.assertEqual(torch.load(source,weights_only=True)['epoch'],11)

    def test_artifacts_bundle_has_counts_layers_and_no_weights(self):
        rows=[]
        for i,kind in enumerate(('perfect','strength','matching','shape')):
            r,b,a=case(kind);row=diagnostic_rows(r,b,a)[0];row.update(scene_id=i,view_id=i);rows.append(row)
        report=aggregate(rows,repetitions=5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'point';write_artifacts(out,rows,report)
            (out/'checkpoint_metadata.json').write_text('{}')
            (out/'not_to_copy.pt').write_bytes(b'weights')
            bundle=portable_bundle(out,root,{})
            with tarfile.open(bundle) as archive:
                names=archive.getnames();self.assertTrue(any(n.endswith('windows.csv.gz') for n in names))
                self.assertFalse(any(n.endswith('.pt') for n in names))
                manifest=json.load(archive.extractfile('decomposition_diagnostics/BUNDLE_MANIFEST.json'))
                for name,sha in manifest['files_sha256'].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile('decomposition_diagnostics/'+name).read()).hexdigest(),sha)
            with gzip.open(out/'layers.csv.gz','rt') as stream:self.assertIn('relative_strength_error',stream.readline())

    def test_current_last_snapshot_evaluation_does_not_overwrite_formal_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,new,index=prepare_fixture(root)
            c=config_for_test();c.update(output_root=base['output_root'])
            c['training'].update(epochs=1,evaluate_initial_checkpoint=False,evaluate_test_after_training=False)
            train(c,Path(c['output_root']),index)
            output=Path(c['output_root']);formal=output/'evaluation';formal.mkdir()
            old=formal/'decomposition_metrics.json';old.write_text('keep-original')
            report=evaluate(c,output,index,repetitions=3,split='validation',checkpoint_kind='last',
                            diagnostics_only=True,diagnostics_root=root/'diagnostics')
            self.assertEqual(report['checkpoint']['checkpoint_kind'],'last')
            self.assertEqual(report['checkpoint']['epoch'],1)
            self.assertEqual(old.read_text(),'keep-original')
            self.assertTrue((root/'diagnostics/latest_diagnostics.tar.gz').is_file())

    def test_diagnostic_tmux_uses_source_config_and_separate_job_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'bin';binary.mkdir();config=root/'expanded.json'
            config.write_text(json.dumps({'output_root':'outputs/expanded'}))
            fake=binary/'tmux';fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]];then exit 1;fi\nexit 0\n');fake.chmod(0o755)
            frontend=root/'frontend.sh';text=(ROOT/'run_failure_diagnostics_tmux.sh').read_text()
            text=text.replace('ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',f'ROOT={shlex.quote(str(ROOT))}')
            text=text.replace('base="$ROOT/Phase2_Multi_Source/outputs/failure_diagnostics"',f'base={shlex.quote(str(root/"diagnostics"))}')
            frontend.write_text(text)
            env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'RIND_PYTHON':sys.executable,'RIND_MASKS_CONFIG':str(config)}
            env.pop('RIND_TMUX_SESSION',None)
            result=subprocess.run(['bash',str(frontend),'start'],env=env,capture_output=True,text=True,check=True)
            self.assertIn('Started background diagnostics',result.stdout)
            job=root/'diagnostics/jobs/rind_failure_diagnostics';run=Path((job/'tmux_runs/latest_run.txt').read_text().strip())
            args=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_TMUX_TASK=evaluate',args)
            self.assertIn('--diagnostics-only',args);self.assertIn('--checkpoint',args);self.assertIn('last',args)
            self.assertIn('validation',args);self.assertNotIn('test',args)
            self.assertEqual((job/'source_config.txt').read_text().strip(),str(config))

if __name__=='__main__':unittest.main()
