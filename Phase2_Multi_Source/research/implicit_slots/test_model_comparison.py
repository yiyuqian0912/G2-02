import copy,gzip,hashlib,json,tempfile,unittest,socket
from pathlib import Path
from unittest.mock import patch
import torch
from test_sequential import fixture,configuration
from train_decomposition import make_model,train,write_json
from reuse_mask_cache import prepare_cached_labels
from evaluate_sequential import evaluate_with_reference
from sequential_objective import primary_counts
from model_comparison import compare_reports


def comparison_worker(rank,port,c,index):
    import os,torch.distributed as dist
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK=str(rank),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    try:evaluate_with_reference(c,Path(c['output_root']),index,repetitions=2)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(99)

    def prepared(self,root):
        data,source,index,old=fixture(root)
        c=configuration();c['model'].update(old['model'])
        c['model'].update(decomposition_architecture='sequential_count_shape_v6',residual_gradient='straight_through')
        c['output_root']=str(root/'new');c['cache_source_root']=str(source)
        c['training'].update(epochs=1,batch_size=1,target_global_batch_size=2,
                            evaluate_after_training=False,evaluate_initial_checkpoint=False,
                            evaluate_test_after_training=False,prefetch_batches=0)
        c['training'].pop('initial_decomposition_checkpoint',None);c['training'].pop('model_selection',None)
        bound=prepare_cached_labels(c);train(c,Path(c['output_root']),bound)
        return c,source,bound

    def test_soft_legacy_probabilities_use_threshold_not_boolean_nonzero(self):
        target=torch.zeros(1,4,4,4);target[0,0,:,:2]=1
        b={'teacher_mask':target,'layer_mask':torch.tensor([[True,False,False,False]]),
           'obstacle':torch.zeros(1,4,4,dtype=torch.bool)}
        soft=torch.full_like(target,.1);soft[0,0,:,:2]=.9
        out={'mask_probability':soft,'active_slots':b['layer_mask'],'predicted_count':torch.tensor([1])}
        self.assertEqual(primary_counts(out,b)[1].tolist(),[1,1,1,1])

    def test_in_progress_reference_is_pending_and_never_uses_warm_start_as_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,source,index=self.prepared(root)
            report=evaluate_with_reference(c,Path(c['output_root']),index,repetitions=2)
            directory=Path(c['output_root'])/'evaluation/seed_0'
            pending=json.loads((directory/'reference_comparison_pending.json').read_text())
            self.assertEqual(pending['status'],'pending')
            self.assertFalse((directory/'reference_comparison.json').exists())
            self.assertTrue((directory/'primary_metrics.json').exists())

    def test_final_old_best_and_last_preserved_and_both_models_use_identical_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,source,index=self.prepared(root)
            run=source/'models/seed_0';best=torch.load(run/'best.pt',weights_only=True)
            # The source fixture originally had epoch14. Final source best is
            # now epoch16, proving the comparison does not use an early snapshot.
            best['epoch']=16;torch.save(best,run/'best.pt')
            last={**best,'epoch':20};torch.save(last,run/'last.pt')
            write_json(run/'training.json',{'best_epoch':16,'trained_through_epoch':20})
            before={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('best.pt','last.pt')}
            report=evaluate_with_reference(c,Path(c['output_root']),index,repetitions=3)
            directory=Path(c['output_root'])/'evaluation/seed_0'
            comparison=json.loads((directory/'reference_comparison.json').read_text())
            self.assertEqual(comparison['previous_checkpoint']['epoch'],16)
            self.assertEqual(comparison['reference_retention']['last']['epoch'],20)
            self.assertEqual(comparison['windows'],index['counts']['test'])
            self.assertEqual(set(comparison['metrics']),{'count_accuracy','layer_accuracy'})
            self.assertTrue(comparison['no_constraint_or_confidence_filtering'])
            self.assertTrue((directory/'reference_final/primary_metrics.json').is_file())
            for name,sha in before.items():self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),sha)
            for kind in ('best','last'):
                meta=comparison['reference_retention'][kind]
                self.assertTrue(Path(meta['snapshot_path']).is_file())
                self.assertEqual(meta['source_checkpoint_sha256'],before[kind+'.pt'])
            self.assertEqual(report['metrics']['count_accuracy']['denominator'],comparison['windows'])

    def test_paired_change_uses_layer_denominator_and_rejects_different_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            new=[{'scene_id':0,'view_id':0,'size':16,'layer_count':1,'layers':1,'count_correct':True,'shape_correct':1},
                 {'scene_id':1,'view_id':0,'size':16,'layer_count':3,'layers':3,'count_correct':True,'shape_correct':2}]
            old=[{**new[0],'count_correct':False,'shape_correct':0},{**new[1],'shape_correct':1}]
            metric=lambda c,l:{'count_accuracy':{'window_weighted':c},'layer_accuracy':{'window_weighted':l}}
            groups=lambda c,l:{'R1':{'windows':2,'layers':4,'count_accuracy':c,'layer_accuracy':l}}
            a={'split':'test','epoch':2,'checkpoint':{},'metrics':metric(1,.75),'by_group':groups(1,.75)}
            b={'split':'test','epoch':1,'checkpoint':{},'metrics':metric(.5,.25),'by_group':groups(.5,.25)}
            r=compare_reports(a,b,new,old,{'corpus_sha256':'same'},dest,'cmp',20,0)
            self.assertEqual(r['metrics']['count_accuracy']['paired_change']['window_weighted'],.5)
            self.assertEqual(r['metrics']['layer_accuracy']['paired_change']['window_weighted'],.5)
            self.assertEqual(r['metrics']['layer_accuracy']['paired_change']['denominator'],4)
            wrong=copy.deepcopy(old);wrong[0]['view_id']=10
            with self.assertRaisesRegex(ValueError,'same unique windows'):
                compare_reports(a,b,new,wrong,{'corpus_sha256':'same'},dest,'bad',0,0)

    def test_two_rank_comparison_with_an_empty_rank_partition(self):
        try:
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        except PermissionError:self.skipTest('loopback unavailable')
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            c,source,index=self.prepared(Path(tmp));run=source/'models/seed_0'
            saved=torch.load(run/'best.pt',weights_only=True);torch.save(saved,run/'last.pt')
            write_json(run/'training.json',{'best_epoch':14,'trained_through_epoch':14})
            mp.spawn(comparison_worker,args=(port,c,index),nprocs=2,join=True)
            report=json.loads((Path(c['output_root'])/'evaluation/seed_0/reference_comparison.json').read_text())
            self.assertEqual(report['windows'],1)


if __name__=='__main__':unittest.main()
