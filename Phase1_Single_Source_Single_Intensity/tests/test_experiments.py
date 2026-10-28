"""Protocol freezing, target integrity, resume and test-lock regression checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from rind_phase1.data import PROJECT_ROOT,Phase1Dataset
from rind_phase1.experiments import (plan,prepare,prepared,train,evaluation,summarize,
                                     _cache_sources,_import_compatible_teacher)
from rind_phase1.teacher import generate_uniform_teacher,save_teacher_record


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.dataset=Phase1Dataset()
        except FileNotFoundError as exc:
            raise unittest.SkipTest('Install RIND data to run experiment integration tests') from exc

    def config(self,root):
        c=json.loads((PROJECT_ROOT/'configs/experiment_round1.json').read_text())
        c.update(output_root=str(root),scene_counts=dict(train=2,validation=1,test=1),window_sizes=[16,32])
        c['teacher']['spacing']=256
        c['training'].update(seeds=[0],epochs=2,patience=3,hidden=8,num_threads=1)
        c['evaluation']['plot_count']=0
        return c

    def test_protocol_targets_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=self.config(root)
            protocol=plan(c,root,self.dataset)
            self.assertEqual(protocol,plan(c,root,self.dataset))
            sets=[set(ids) for ids in protocol['scene_splits'].values()]
            self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
            with self.assertRaises(ValueError):
                plan({**c,'split_seed':0},root,self.dataset)
            prepare(c,root,self.dataset,protocol,'response')
            run=train(c,root,self.dataset,protocol,'response',0)
            import torch
            cp=torch.load(run/'last.pt',weights_only=True)
            # Rewind to epoch 1 is not supported by editing metadata; the test lock
            # must reject a nominally unfinished run before it scores test records.
            cp['epoch']=1
            torch.save(cp,run/'last.pt')
            with self.assertRaises(ValueError): evaluation(c,root,self.dataset,protocol,'response',0)
            cp['epoch']=2;torch.save(cp,run/'last.pt')
            before=(run/'best.pt').read_bytes()
            train(c,root,self.dataset,protocol,'response',0)
            self.assertEqual(before,(run/'best.pt').read_bytes())
            report=evaluation(c,root,self.dataset,protocol,'response',0)
            self.assertTrue(report['unseen_scene']['verified'])
            self.assertEqual(report['aggregate']['observations'],2)
            summarize(c,root,'response')
            entries=json.loads((root/'response/records.json').read_text())
            entries['train'].reverse()
            (root/'response/records.json').write_text(json.dumps(entries))
            with self.assertRaises(ValueError): prepared(c,root,protocol,'response')

    def test_interrupted_training_resumes_exactly(self):
        import torch
        from unittest.mock import patch
        from rind_phase1.experiments import atomic_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            models=[]
            for name in ('continuous','resumed'):
                root=Path(directory)/name
                c=self.config(root)
                protocol=plan(c,root,self.dataset)
                prepare(c,root,self.dataset,protocol,'response')
                if name=='resumed':
                    def interrupt_after_epoch(path, checkpoint):
                        atomic_checkpoint(path,checkpoint)
                        if path.name=='last.pt' and checkpoint['epoch']==1:
                            raise RuntimeError('simulated interruption')
                    with patch('rind_phase1.experiments.atomic_checkpoint',side_effect=interrupt_after_epoch):
                        with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                            train(c,root,self.dataset,protocol,'response',0)
                run=train(c,root,self.dataset,protocol,'response',0)
                models.append(torch.load(run/'last.pt',weights_only=True)['model'])
            for key in models[0]:
                self.assertTrue(torch.equal(models[0][key],models[1][key]),key)

    def test_prior_teacher_cache_is_imported_only_when_grid_and_physics_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);response_root=root/'old'/'response'
            (response_root/'teachers').mkdir(parents=True)
            scene=0;view=0
            record=generate_uniform_teacher(
                self.dataset,scene,view,spacing=256,temperature=.05,
                boundary_lambda=2,boundary_sigma=1)
            save_teacher_record(record,response_root/'teachers'/f's{scene}_v{view}.npz')
            old_data=str(self.dataset.root)
            (response_root.parent/'protocol.json').write_text(json.dumps(
                {'config':{'data_root':old_data}}))
            sources=_cache_sources([str(response_root)])
            settings={'spacing':256,'temperature':.05,'alpha':1.,'beta':0.,
                      'boundary_lambda':2.,'boundary_sigma':1.}
            imported,rejected=_import_compatible_teacher(
                sources,self.dataset,scene,view,settings,'new-protocol')
            self.assertIsNotNone(imported);self.assertFalse(rejected)
            self.assertEqual(imported['metadata']['experiment_protocol'],'new-protocol')
            incompatible={**settings,'spacing':128}
            imported,rejected=_import_compatible_teacher(
                sources,self.dataset,scene,view,incompatible,'new-protocol')
            self.assertIsNone(imported)
            self.assertEqual(rejected[0]['reason'],'candidate_grid_differs')

    def test_memmap_teacher_storage_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=self.config(root)
            c['scene_counts']=dict(train=1,validation=1,test=1)
            c['window_sizes']=[16]
            c['teacher']['record_storage']='memmap'
            protocol=plan(c,root,self.dataset)
            prepare(c,root,self.dataset,protocol,'response')
            directory_path,index=prepared(c,root,protocol,'response')
            self.assertEqual(index['record_storage'],'memmap')
            entries=json.loads((directory_path/'records.json').read_text())
            self.assertTrue(entries['train'][0].split('/')[-2].endswith('.rindshard'))
            from rind_phase1.train import load_splits
            splits=load_splits(directory_path/'records.json',self.dataset,'world')
            self.assertAlmostEqual(float(splits['train'][0]['teacher_prob'].sum()),1.0,places=6)
            first=index['records_sha256']
            prepare(c,root,self.dataset,protocol,'response')
            _,resumed=prepared(c,root,protocol,'response')
            self.assertEqual(first,resumed['records_sha256'])
            self.assertEqual(resumed['append_cache_counts']['reused'],3)


if __name__=='__main__': unittest.main()
