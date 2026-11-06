import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from rind_phase1.active_rl.scene_schedule import SceneSchedule, make_schedule, coverage
from rind_phase1.active_rl.prepare import select_scenes
from rind_phase1.active_rl.frontier import make_policy
from rind_phase1.active_rl.experiment import train


class SceneCoverageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(33)

    def test_all_means_every_available_scene_without_replacement(self):
        available=[1,4,7,9]
        selected=select_scenes(available,'all',np.random.default_rng(7))
        self.assertEqual(set(selected),set(available))
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual(len(select_scenes(available,2,np.random.default_rng(7))),2)
        for request in (0,'bad',5):
            with self.assertRaises(ValueError):select_scenes(available,request,np.random.default_rng(7))

    def test_asynchronous_episode_completion_covers_every_scene(self):
        schedule=SceneSchedule(range(9),2,13)
        rng=np.random.default_rng(29)
        active=[schedule.next_scene() for _ in range(3)]
        for _ in range(100):
            self.assertEqual(len(set(active)),3)
            slot=int(rng.integers(3));schedule.finish_scene(active[slot])
            active[slot]=schedule.next_scene()
            progress=coverage(schedule,torch.device('cpu'),1)
            if progress['coverage_complete']:break
        self.assertEqual(progress['training_scenes'],9)
        self.assertEqual(progress['completed_unique_scenes'],9)
        self.assertGreaterEqual(progress['minimum_completed_visits'],2)
        self.assertEqual(progress['scenes_reaching_target'],9)

    def test_checkpoint_retries_partial_scenes_without_false_completions(self):
        first=SceneSchedule(range(7),1,3)
        a,b=first.next_scene(),first.next_scene();first.finish_scene(a)
        before=first.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'state.pt';torch.save(before,path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
        resumed=make_schedule(range(7),1,99,[saved])
        self.assertEqual(resumed.visits,first.visits)
        self.assertEqual(resumed.visits[b],0)
        self.assertEqual(resumed.next_scene(),b)
        self.assertEqual(resumed.visits[b],0)
        resumed.finish_scene(b);self.assertEqual(resumed.visits[b],1)

    def test_gpu_repartition_preserves_completed_counts(self):
        saved=[]
        for ids in ([0,2,4,6],[1,3,5,7]):
            sampler=SceneSchedule(ids,2,19)
            scene=sampler.next_scene();sampler.finish_scene(scene)
            sampler.next_scene();saved.append(sampler.state_dict())
        merged=make_schedule(range(8),2,31,saved)
        expected={s:n for state in saved for s,n in zip(state['scenes'],state['visits'])}
        self.assertEqual(merged.visits,expected)
        self.assertEqual(sum(merged.visits.values()),2)
        narrow=make_schedule([0,4],2,31,saved)
        self.assertEqual(narrow.visits,{s:expected[s] for s in (0,4)})
        with self.assertRaises(ValueError):make_schedule(range(8),2,31,[saved[0],saved[0]])

    def test_training_extends_past_min_updates_and_best_is_fully_covered(self):
        # Algorithm fixture, not a physical-scene performance experiment.
        axis=(torch.arange(8)+.5)*16;yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        loc=SimpleNamespace(grid=8,world=128,tile=16,spacing=16,device=torch.device('cpu'),
            xy=torch.stack((xx,yy),-1).reshape(-1,2),prior=torch.full((64,),-np.log(64)),
            dataset=SimpleNamespace(_base=SimpleNamespace(drivers=np.full((15,1,2),32.,dtype=np.float32))),
            queries=0,hits=0,uniform=0)
        def query(requests):
            loc.queries+=len(requests)
            return {(scene,tile):(torch.sin(torch.arange(64).float()+tile).log_softmax(0),
                                 torch.full((1,1),.5),True) for scene,tile in requests}
        loc.query=query
        config=dict(environment='frontier',fusion_method='minimum',tile_size=16,spacing=16,
            fusion_discount=1.,max_steps=1,budget=512,train_budgets=[512],seed=17,hidden=32,
            num_envs=2,rollout_steps=2,updates=1,minimum_scene_visits=1,learning_rate=.001,
            ppo_epochs=1,minibatch_size=2,clip_ratio=.2,value_weight=.5,action_entropy_bonus=.01,
            grad_clip=.5,target_kl=.1,gae_lambda=.95,eval_every=1,evaluation_envs=4,
            validation_probe_scenes=2,evaluation_curve_stride=1,randomize_train_start=False)
        protocol=dict(scene_splits={'train':list(range(12)),'validation':[12,13],'test':[14]},
            initial_windows={str(s):[32,32,16] for s in range(15)},
            geometry_groups={str(s):str(s) for s in range(15)},snapshot_sha256='algorithm-fixture')
        model=make_policy(config)
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);(output/'protocol.json').write_text(json.dumps(protocol))
            train(model,protocol,config,loc,output,0,1)
            last=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
            self.assertGreater(last['update'],config['updates'])
            self.assertTrue(best['history'][-1]['training_scene_coverage']['coverage_complete'])
            self.assertGreaterEqual(min(last['scene_schedule_states'][0]['visits']),1)
            old_update=last['update']
            config['minimum_scene_visits']=2
            train(model,protocol,config,loc,output,0,1)
            last=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            best=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
            self.assertGreater(last['update'],old_update)
            self.assertGreaterEqual(best['history'][-1]['training_scene_coverage']['minimum_completed_visits'],2)
            self.assertGreaterEqual(min(last['scene_schedule_states'][0]['visits']),2)


if __name__=='__main__':unittest.main()
