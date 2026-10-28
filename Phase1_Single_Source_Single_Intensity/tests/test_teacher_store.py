import tempfile
from pathlib import Path
import unittest
import warnings

import numpy as np
import torch

from rind_phase1.search import CandidateDomain, uniform_grid
from rind_phase1.teacher import load_teacher_record, teacher_probabilities
from rind_phase1.teacher_store import TeacherShard
from rind_phase1.train import batches, collate_records, run_epoch


class TeacherShardTests(unittest.TestCase):
    def test_gradient_accumulation_matches_one_larger_batch(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.25))
            def forward(self,response,obstacle,window,candidate_xy):
                return candidate_xy[...,0]*self.weight+response.mean((1,2))[:,None]
        records=[]
        for index in range(4):
            records.append(dict(
                response=np.full((2,2),index/10,dtype=np.float32),
                obstacle=np.zeros((2,2),bool),window=np.array([0,0,2]),
                candidate_xy=np.array([[0.,0.],[1.,1.],[2.,2.]]),valid=np.ones(3,bool),
                teacher_prob=np.array([.2,.3,.5])))
        large,micro=Tiny(),Tiny();micro.load_state_dict(large.state_dict())
        run_epoch(large,records,batch_size=4,optimizer=torch.optim.SGD(large.parameters(),lr=.1))
        run_epoch(micro,records,batch_size=2,optimizer=torch.optim.SGD(micro.parameters(),lr=.1),
                  accumulation_steps=2)
        torch.testing.assert_close(large.weight,micro.weight)

    def test_distributed_batches_have_equal_steps_and_cover_records(self):
        records=[]
        for index in range(10):
            records.append(dict(
                response=np.full((2,2),index,dtype=np.float32),
                obstacle=np.zeros((2,2),bool),window=np.array([0,0,2]),
                candidate_xy=np.array([[0.,0.],[1.,1.]]),valid=np.ones(2,bool),
                teacher_prob=np.array([.5,.5])))
        by_rank=[list(batches(records,5,np.random.default_rng(7),rank=rank,world_size=3))
                 for rank in range(3)]
        self.assertEqual([len(value) for value in by_rank],[2,2,2])
        for step in range(2):
            self.assertEqual(len({len(by_rank[rank][step]['response']) for rank in range(3)}),1)
        observed={int(batch['response'][row,0,0])
                  for rank in by_rank for batch in rank for row in range(len(batch['response']))}
        self.assertEqual(observed,set(range(10)))

    def test_collate_accepts_read_only_shard_coordinates_without_warning(self):
        candidate=np.array([[1.,2.],[3.,4.]])
        candidate.flags.writeable=False
        record=dict(response=np.zeros((2,2),np.float32),obstacle=np.zeros((2,2),bool),
                    window=np.array([0,0,2]),candidate_xy=candidate,
                    valid=np.ones(2,bool),teacher_prob=np.array([.5,.5]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            batch=collate_records([record])
        self.assertEqual(tuple(batch['candidate_xy'].shape),(1,2,2))
        self.assertFalse(any('not writable' in str(item.message) for item in caught))

    def test_memmap_roundtrip_and_resume(self):
        window = np.array([16, 16, 16], dtype=np.int64)
        grid = uniform_grid(CandidateDomain(tuple(window), 64, False), 16)
        valid = np.ones(len(grid.candidate_xy), dtype=bool)
        cost = np.linspace(0, 1, len(valid), dtype=np.float64)
        probability = teacher_probabilities(cost, valid, temperature=.1)
        record = dict(
            scene_id=7, view_id=3, window=window,
            candidate_xy=grid.candidate_xy, valid=valid,
            evaluated=np.ones(len(valid), dtype=bool), physical_cost=cost,
            teacher_prob=probability, temperature=.1,
            config_id='a' * 64, run_id='test-run',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_size16.rindshard'
            settings = dict(world_size=64, spacing=16, temperature=.1,
                            quadtree_prior=False, physics={'alpha': 1.0})
            shard = TeacherShard(path, capacity=2, candidate_count=len(valid),
                                 settings=settings, protocol_digest='protocol')
            shard.write(0, record)
            self.assertFalse(shard.complete(0))
            shard.flush()
            self.assertTrue(shard.complete(0))
            loaded = load_teacher_record(shard.reference(0))
            np.testing.assert_array_equal(loaded['candidate_xy'], record['candidate_xy'])
            np.testing.assert_array_equal(loaded['valid'], valid)
            np.testing.assert_allclose(loaded['physical_cost'], cost, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(loaded['teacher_prob'], probability, rtol=1e-6, atol=1e-7)
            expected = teacher_probabilities(loaded['physical_cost'], loaded['valid'], temperature=.1)
            np.testing.assert_allclose(loaded['teacher_prob'], expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(loaded['metadata']['experiment_protocol'], 'protocol')
            reopened = TeacherShard(path, capacity=2, candidate_count=len(valid),
                                    settings=settings, protocol_digest='protocol')
            self.assertTrue(reopened.complete(0))
            self.assertFalse(reopened.complete(1))


if __name__ == '__main__':
    unittest.main()
