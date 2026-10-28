import unittest
import torch

from rind_phase1.model import (SourceEnergyField,fuse_view_log_probabilities,
                               MultiScaleBoundaryFeatures,RoleAttention,
                               stable_scaled_dot_product_attention)
from rind_phase1.train import batches


def inputs(size=16,candidates=11,batch=2):
    response=torch.rand(batch,size,size)
    obstacle=torch.zeros(batch,size,size,dtype=torch.bool)
    window=torch.tensor([[0.,0.,float(size)]]*batch)
    xy=torch.rand(batch,candidates,2)*1024
    return response,obstacle,window,xy


class LargeModelTests(unittest.TestCase):
    def test_stable_attention_matches_reference_and_backpropagates(self):
        torch.manual_seed(41)
        query=torch.randn(2,3,11,8,requires_grad=True)
        key=torch.randn(2,3,7,8,requires_grad=True)
        value=torch.randn(2,3,7,8,requires_grad=True)
        actual=stable_scaled_dot_product_attention(query,key,value,query_chunk=4)
        expected=torch.nn.functional.scaled_dot_product_attention(query,key,value)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        actual.square().mean().backward()
        for tensor in (query,key,value):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_large_model_scores_and_normalizes_candidates(self):
        model=SourceEnergyField(hidden=64,fourier_frequencies=3,architecture='large',
                                attention_heads=8,pixel_coordinates=True,
                                normalization_support='world')
        response,obstacle,window,xy=inputs()
        result=model.predict(response,obstacle,window,xy)
        self.assertEqual(result['energy'].shape,(2,11))
        self.assertTrue(torch.isfinite(result['energy']).all())
        torch.testing.assert_close(result['student_prob'].sum(-1),torch.ones(2))

    def test_large_encoded_chunk_scores_equal_full_forward(self):
        torch.manual_seed(3)
        model=SourceEnergyField(hidden=64,fourier_frequencies=2,architecture='large',attention_heads=4).eval()
        response,obstacle,window,xy=inputs(candidates=9,batch=1)
        with torch.no_grad():
            full=model(response,obstacle,window,xy)
            encoded=model.encode_observation(response,obstacle,window)
            chunked=torch.cat([model.score_encoded(encoded,xy[:,start:start+4])
                               for start in range(0,xy.shape[1],4)],dim=1)
        torch.testing.assert_close(chunked,full)

    def test_multiscale_boundary_model_scores_and_chunks(self):
        torch.manual_seed(17)
        model=SourceEnergyField(
            hidden=64,fourier_frequencies=3,architecture='multiscale_boundary',
            attention_heads=8,pixel_coordinates=True,normalization_support='world').eval()
        response,obstacle,window,xy=inputs(size=32,candidates=13,batch=2)
        response[:,8:20,12:]=1
        with torch.no_grad():
            full=model(response,obstacle,window,xy)
            encoded=model.encode_observation(response,obstacle,window)
            chunked=torch.cat([model.score_encoded(encoded,xy[:,start:start+5])
                               for start in range(0,xy.shape[1],5)],dim=1)
        self.assertEqual(full.shape,(2,13))
        self.assertTrue(torch.isfinite(full).all())
        torch.testing.assert_close(chunked,full)

    def test_multiscale_boundary_uses_window_relative_geometry(self):
        torch.manual_seed(23)
        model=SourceEnergyField(hidden=64,fourier_frequencies=2,
                                architecture='multiscale_boundary',attention_heads=8).eval()
        response,obstacle,window,xy=inputs(size=16,candidates=7,batch=1)
        response[:,4:12,8:]=1
        shifted_window=window.clone();shifted_window[:,:2]+=128
        shifted_xy=xy+128
        with torch.no_grad():
            original=model(response,obstacle,window,xy)
            shifted=model(response,obstacle,shifted_window,shifted_xy)
        # Relative geometry is preserved but absolute world geometry changes.
        self.assertFalse(torch.equal(original,shifted))

    def test_multiscale_attention_roles_are_separate(self):
        model=SourceEnergyField(hidden=64,architecture='multiscale_boundary',attention_heads=8)
        self.assertEqual(model.boundary_attention.heads,4)
        self.assertEqual(model.region_attention.heads,2)
        self.assertEqual(model.global_attention.heads,2)
        names=set(dict(model.named_parameters()))
        self.assertIn('boundary_attention.query.weight',names)
        self.assertIn('region_attention.query.weight',names)
        self.assertIn('global_attention.query.weight',names)
        self.assertNotIn('attention.keys.weight',names)

    def test_multiscale_requires_declared_eight_heads(self):
        with self.assertRaisesRegex(ValueError,'exactly eight'):
            SourceEnergyField(hidden=64,architecture='multiscale_boundary',attention_heads=4)

    def test_boundary_features_do_not_invent_crop_frame_edges(self):
        encoder=MultiScaleBoundaryFeatures()
        constant=torch.ones(2,1,32,32)[:,0]
        feature=encoder(constant)
        torch.testing.assert_close(feature,torch.zeros_like(feature))

    def test_role_attention_uses_candidate_to_token_relative_geometry(self):
        torch.manual_seed(29)
        attention=RoleAttention(64,4).eval()
        query=torch.randn(1,3,64);tokens=torch.randn(1,5,64)
        keys,values=attention.encode(tokens)
        candidate=torch.tensor([[[.1,.2],[.4,.6],[.8,.3]]])
        token=torch.tensor([[[.05,.1],[.2,.7],[.5,.5],[.7,.2],[.9,.9]]])
        with torch.no_grad():
            original=attention.attend(query,keys,values,candidate,token)
            changed=attention.attend(query,keys,values,candidate+.07,token)
            translated=attention.attend(query,keys,values,candidate+.07,token+.07)
        self.assertFalse(torch.equal(original,changed))
        torch.testing.assert_close(original,translated,rtol=1e-4,atol=5e-6)

    def test_every_multiscale_branch_receives_gradient(self):
        torch.manual_seed(31)
        model=SourceEnergyField(hidden=64,fourier_frequencies=2,
                                architecture='multiscale_boundary',attention_heads=8)
        response,obstacle,window,xy=inputs(size=16,candidates=7,batch=1)
        response[:,3:11,7:]=1
        model(response,obstacle,window,xy).square().mean().backward()
        prefixes=('image_encoder.boundary.','image_encoder.region.','image_encoder.global_branch.',
                  'boundary_attention.','region_attention.','global_attention.',
                  'relative_query.','token_position.','role_gates.')
        for prefix in prefixes:
            gradients=[parameter.grad for name,parameter in model.named_parameters()
                       if name.startswith(prefix)]
            self.assertTrue(gradients,prefix)
            self.assertTrue(any(gradient is not None and torch.isfinite(gradient).all() and
                                bool(torch.any(gradient!=0)) for gradient in gradients),prefix)

    def test_default_architecture_keeps_legacy_state_dictionary(self):
        implicit=SourceEnergyField(hidden=8,fourier_frequencies=2)
        explicit=SourceEnergyField(hidden=8,fourier_frequencies=2,architecture='baseline',attention_heads=1)
        self.assertEqual(list(implicit.state_dict()),list(explicit.state_dict()))
        explicit.load_state_dict(implicit.state_dict(),strict=True)

    def test_attention_heads_must_divide_hidden(self):
        with self.assertRaisesRegex(ValueError,'positive divisor'):
            SourceEnergyField(hidden=65,architecture='large',attention_heads=8)

    def test_threaded_batch_prefetch_preserves_order_and_values(self):
        records=[]
        for index in range(5):
            records.append({'response':torch.full((16,16),float(index)).numpy(),
                            'obstacle':torch.zeros(16,16,dtype=torch.bool).numpy(),
                            'window':torch.tensor([0,0,16]).numpy(),
                            'candidate_xy':torch.tensor([[1.,2.],[3.,4.]]).numpy(),
                            'valid':torch.tensor([True,True]).numpy(),
                            'support':torch.tensor([True,True]).numpy(),
                            'teacher_prob':torch.tensor([.25,.75]).numpy()})
        direct=list(batches(records,2,prefetch=0))
        threaded=list(batches(records,2,prefetch=3))
        self.assertEqual(len(direct),len(threaded))
        for expected,actual in zip(direct,threaded):
            self.assertEqual(set(expected),set(actual))
            for key in expected:torch.testing.assert_close(expected[key],actual[key])

    def test_residual_geometry_starts_as_exact_blind_model(self):
        torch.manual_seed(11)
        blind=SourceEnergyField(hidden=16,fourier_frequencies=2)
        geometry=SourceEnergyField(
            hidden=16,fourier_frequencies=2,use_scene_geometry=True,
            geometry_parameter_width=8,geometry_type_count=4,
            geometry_attention_heads=4,geometry_layers=1,
            geometry_dropout=.1,geometry_fusion='residual',
            geometry_cross_attention_heads=4)
        missing,unexpected=geometry.load_state_dict(blind.state_dict(),strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith(('geometry_encoder.','geometry_attention.',
                                             'geometry_correction_head.'))
                            for name in missing))
        response,obstacle,window,xy=inputs(candidates=7,batch=2)
        types=torch.tensor([[0,2],[1,3]])
        params=torch.rand(2,2,8)*100
        mask=torch.ones(2,2,dtype=torch.bool)
        blind.eval();geometry.eval()
        with torch.no_grad():
            expected=blind(response,obstacle,window,xy)
            actual=geometry(response,obstacle,window,xy,types,params,mask)
        torch.testing.assert_close(actual,expected)

    def test_geometry_correction_can_change_energy_after_training(self):
        model=SourceEnergyField(
            hidden=16,fourier_frequencies=1,use_scene_geometry=True,
            geometry_parameter_width=8,geometry_attention_heads=4,
            geometry_layers=0,geometry_fusion='residual',
            geometry_cross_attention_heads=4)
        response,obstacle,window,xy=inputs(candidates=5,batch=1)
        types=torch.tensor([[0,2]])
        params=torch.rand(1,2,8)*100
        mask=torch.ones(1,2,dtype=torch.bool)
        before=model(response,obstacle,window,xy,types,params,mask)
        with torch.no_grad():
            model.geometry_correction_head[-1].weight.fill_(.1)
        after=model(response,obstacle,window,xy,types,params,mask)
        self.assertFalse(torch.equal(before,after))

    def test_multiview_fusion_multiplies_agreeing_evidence(self):
        probability=torch.tensor([[[.7,.2,.1],[.6,.3,.1]]])
        fused_log,fused=fuse_view_log_probabilities(probability.log())
        expected=(probability[0,0]*probability[0,1])
        expected=expected/expected.sum()
        torch.testing.assert_close(fused[0],expected)
        torch.testing.assert_close(fused_log.exp(),fused)
        self.assertGreater(float(fused[0,0]),.7)

    def test_multiview_fusion_respects_mask_and_prior(self):
        probability=torch.tensor([[[.5,.3,.2],[.2,.3,.5],[.1,.1,.8]]])
        mask=torch.tensor([[True,True,False]])
        prior=torch.tensor([.2,.3,.5])
        _,fused=fuse_view_log_probabilities(
            probability.log(),mask,log_prior=prior.log())
        expected=probability[0,0]*probability[0,1]/prior
        expected=expected/expected.sum()
        torch.testing.assert_close(fused[0],expected)


if __name__=='__main__':unittest.main()
