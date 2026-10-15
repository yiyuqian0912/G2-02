import unittest
import numpy as np

from rind_phase1.multiview import (
    LayerField,associate_layers,fuse_associated_layers,fuse_track,
    layer_association_score)


class MultiViewTests(unittest.TestCase):
    def field(self,view,layer,intensity,probability):
        return LayerField(view,layer,intensity,np.asarray(probability,dtype=np.float64))

    def test_association_links_agreeing_layers_across_views(self):
        fields=[self.field(0,0,.3,[.8,.15,.05]),
                self.field(1,0,.31,[.75,.2,.05]),
                self.field(0,1,.7,[.05,.1,.85]),
                self.field(1,1,.68,[.05,.15,.8])]
        tracks=associate_layers(fields,minimum_score=.6,intensity_log_scale=.5)
        self.assertEqual(tracks,[[0,1],[2,3]])

    def test_association_never_puts_two_layers_from_one_view_in_a_track(self):
        fields=[self.field(0,0,.5,[.8,.1,.1]),
                self.field(0,1,.5,[.78,.12,.1]),
                self.field(1,0,.5,[.79,.11,.1])]
        tracks=associate_layers(fields,minimum_score=.5,intensity_log_scale=.5)
        self.assertEqual(sorted(map(len,tracks)),[1,2])
        for track in tracks:
            self.assertEqual(len({fields[index].view_id for index in track}),len(track))

    def test_fusion_sharpens_consistent_candidate(self):
        fields=[self.field(0,0,.4,[.6,.3,.1]),
                self.field(1,0,.4,[.7,.2,.1])]
        probability=fuse_track(fields,[0,1])
        self.assertGreater(probability[0],.7)
        self.assertAlmostEqual(float(probability.sum()),1.0)

    def test_unmatched_layer_is_preserved_as_own_track(self):
        fields=[self.field(0,0,.3,[.9,.05,.05]),
                self.field(1,0,.8,[.05,.05,.9])]
        fused=fuse_associated_layers(fields,minimum_score=.8)
        self.assertEqual(len(fused),2)
        np.testing.assert_allclose(fused[0]['probability'],fields[0].probability)
        np.testing.assert_allclose(fused[1]['probability'],fields[1].probability)

    def test_uniform_fields_do_not_create_false_association(self):
        uniform=np.full(8,1/8)
        fields=[self.field(0,0,.4,uniform),self.field(1,0,.4,uniform)]
        self.assertEqual(associate_layers(fields,minimum_score=.01),[[0],[1]])

    def test_aligned_near_uniform_noise_is_discounted(self):
        ripple=np.asarray([1,-1,1,-1,1,-1,1,-1],dtype=np.float64)*1e-5
        fields=[self.field(0,0,.4,np.full(8,1/8)+ripple),
                self.field(1,0,.4,np.full(8,1/8)+ripple)]
        self.assertLess(layer_association_score(fields[0],fields[1]),.01)
        self.assertEqual(associate_layers(fields,minimum_score=.01),[[0],[1]])

    def test_track_rejects_two_layers_from_same_view(self):
        fields=[self.field(0,0,.4,[.7,.2,.1]),
                self.field(0,1,.4,[.6,.3,.1])]
        with self.assertRaisesRegex(ValueError,'one layer per view'):
            fuse_track(fields,[0,1])

    def test_redundancy_discount_limits_overconfidence(self):
        fields=[self.field(0,0,.4,[.6,.3,.1]),
                self.field(1,0,.4,[.6,.3,.1])]
        independent=fuse_track(fields,[0,1],redundancy_discount=1)
        discounted=fuse_track(fields,[0,1],redundancy_discount=.5)
        self.assertGreater(independent[0],discounted[0])
        self.assertGreater(discounted[0],fields[0].probability[0])

    def test_exact_continuous_intensity_allows_conservative_lower_threshold(self):
        fields=[self.field(0,0,.314159,[.55,.3,.1,.05]),
                self.field(1,0,.314159,[.3,.25,.4,.05])]
        score=layer_association_score(fields[0],fields[1])
        self.assertLess(score,.55)
        self.assertGreaterEqual(score,.15)
        self.assertEqual(associate_layers(fields),[[0,1]])


if __name__=='__main__':
    unittest.main()
