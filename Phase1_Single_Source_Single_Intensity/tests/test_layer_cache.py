import tempfile
from pathlib import Path
import unittest

import numpy as np

from rind_phase1.layer_cache import _load_layer, _compatible_report
from rind_phase1.train import CompositeRecords, batch_count, batches


class _Records:
    def __init__(self, rows):
        self.rows=rows
        self.window_sizes=[int(row['window'][2]) for row in rows]
        self.candidate_counts=[len(row['candidate_xy']) for row in rows]
        self.scene_ids=set()
        self.config_ids={'test'}
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def _record(side,count):
    candidate=np.column_stack((np.arange(count),np.zeros(count)))
    return dict(response=np.zeros((side,side),np.float32),obstacle=np.zeros((side,side),bool),
                window=np.array([0,0,side]),candidate_xy=candidate,
                valid=np.ones(count,bool),support=np.ones(count,bool),
                teacher_prob=np.full(count,1/count))


class LayerCacheTests(unittest.TestCase):
    def test_sparse_layer_expands_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'layer.npz'
            x=np.array([0.,16.]);y=np.array([0.,16.]);grid=np.array([0,2,3])
            candidate=np.array([[0.,0.],[0.,16.],[16.,16.]])
            np.savez_compressed(path,layer_response=np.array([[True,False],[False,False]]),
                candidate_xy=candidate,grid_index=grid,x_axis=x,y_axis=y,
                valid=np.array([True,False,True]),probability=np.array([.4,0.,.6]),
                physical_cost=np.array([0.,np.inf,.1]))
            layer=_load_layer(path,include_cost=True)
            self.assertEqual(layer['candidate'].shape,(4,2))
            self.assertEqual(layer['candidate_count'],4)
            self.assertAlmostEqual(layer['probability'].sum(),1.)
            self.assertEqual(layer['minimum_cost'],0.)

    def test_batches_separate_candidate_lattices(self):
        a=_Records([_record(16,4),_record(16,4)])
        b=_Records([_record(16,9)])
        combined=CompositeRecords([a,b])
        result=list(batches(combined,3))
        self.assertEqual(batch_count(combined,3),2)
        self.assertEqual(sorted(batch['candidate_xy'].shape[1] for batch in result),[4,9])

    def test_report_requires_same_teacher_definition(self):
        report=dict(status='complete',partial_combined_distribution=False,temperature=.05,
                    phase1_cost=dict(alpha=1.,beta=0.,boundary_lambda=2.,boundary_sigma=1.))
        expected=dict(temperature=.05,alpha=1.,beta=0.,boundary_lambda=2.,boundary_sigma=1.)
        self.assertEqual(_compatible_report(report,expected),(True,None))
        self.assertEqual(_compatible_report(report,{**expected,'beta':.1})[1],'beta_differs')


if __name__=='__main__':unittest.main()
