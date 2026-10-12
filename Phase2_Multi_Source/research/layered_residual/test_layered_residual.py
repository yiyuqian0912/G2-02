import unittest
import numpy as np
from layered_residual import minimum_positive_regions,trace_contribution


class TraceTests(unittest.TestCase):
    def test_zero_response_regions_are_never_selected_as_a_layer(self):
        minimum,seeds=minimum_positive_regions([0.,0.,.3,.8,.3],tolerance=1e-12)
        self.assertEqual(minimum,.3)
        self.assertEqual(seeds,[2,4])
        self.assertEqual(minimum_positive_regions([0.,1e-13],tolerance=1e-12),(None,[]))

    def test_exit_target(self):
        out=trace_contribution([.3,.8,.5],[[1],[0,2],[1]],0)
        self.assertEqual(out['positive'],[0,1])
        self.assertEqual(out['negative'],[2])
        self.assertEqual(out['conflict'],[])

    def test_other_exits_target_continues(self):
        out=trace_contribution([.3,.8,.3],[[1],[0,2],[1]],0)
        self.assertEqual(out['positive'],[0,1,2])
        self.assertEqual(out['negative'],[])

    def test_branch_continues_after_one_exit(self):
        # One branch removes target (.8 -> .5), another removes the added .5 (.8 -> .3).
        out=trace_contribution([.3,.8,.5,.3],[[1],[0,2,3],[1],[1]],0)
        self.assertEqual(out['positive'],[0,1,3])
        self.assertEqual(out['negative'],[2])

    def test_drop_equal_to_layer_stops_target(self):
        # The intended rule only asks whether this layer exited.
        out=trace_contribution([.3,.6,.3],[[1],[0,2],[1]],0)
        self.assertIn(2,out['negative'])
        self.assertTrue(any(x['reason']=='drop_equals_current_layer'
                            for x in out['transitions']))

    def test_other_drop_continues_without_label_decomposition(self):
        out=trace_contribution([.3,.9,.4],[[1],[0,2],[1]],0)
        self.assertIn(2,out['positive'])
        self.assertTrue(any(x['reason']=='other_contribution_exits'
                            for x in out['transitions']))

    def test_destination_below_layer_forces_exit(self):
        out=trace_contribution([.3,.8,.1],[[1],[0,2],[1]],0)
        self.assertIn(2,out['negative'])

    def test_ring_path_conflict(self):
        # One path has a .3 drop (exit), the other a .2 drop (continue).
        values=[.3,.8,.7,.5]
        neighbors=[[1,2],[0,3],[0,3],[1,2]]
        out=trace_contribution(values,neighbors,0)
        self.assertIn(3,out['conflict'])

if __name__=='__main__': unittest.main()
