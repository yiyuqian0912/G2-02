import unittest
from unittest.mock import patch
import torch
from decomposition_model import exact_batched_assignment
from sequential_objective import effective_assignment,activation_assignment,loss,primary_counts,verify_effective_assignment


def inputs(counts,predicted,slots=32):
    b=len(counts);h=4
    target=torch.rand(b,slots,h,h)>.6
    real=torch.arange(slots)[None]<torch.tensor(counts)[:,None]
    target &= real[:,:,None,None]
    on=torch.zeros(b,slots,dtype=torch.bool)
    for i,r in enumerate(predicted):on[i,torch.randperm(slots)[:r]]=True
    z=torch.randn(b,slots,h,h,requires_grad=True)
    probability=z.sigmoid()*on[:,:,None,None]
    logits=torch.randn(b,slots+1,requires_grad=True)
    out={'mask_logits':z,'soft_mask_probability':probability,
         'mask_probability':(probability>=.5).float(), 'active_slots':on,
         'predicted_count':on.sum(1),'count_logits':logits,
         'global_count_logits':logits*.7,'slot_count_logits':logits*1.1}
    batch={'teacher_mask':target,'layer_mask':real,'obstacle':torch.zeros(b,h,h,dtype=torch.bool)}
    return out,batch


class EffectiveAssignmentTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)

    def test_compaction_keeps_optimal_cost_for_unequal_counts_and_arbitrary_active_ids(self):
        slots=32;counts=[1,2,4,3,0];predicted=[4,1,3,3,0]
        out,b=inputs(counts,predicted,slots);real=b['layer_mask'];on=out['active_slots']
        shape=torch.rand(len(real),slots,slots)
        shape=torch.where(on[:,None],shape,torch.ones_like(shape))
        shape=torch.where(real[:,:,None],shape,torch.zeros_like(shape))
        cost=shape+(slots+1)*(real[:,:,None]!=on[:,None,:])
        reference=exact_batched_assignment(cost)
        with patch('sequential_objective.exact_batched_assignment',wraps=exact_batched_assignment) as called:
            reduced=effective_assignment(cost,real,on)
            self.assertEqual(called.call_args.args[0].shape,(5,4,4))
        torch.testing.assert_close(cost.gather(2,reference[:,:,None]).sum(1),
                                   cost.gather(2,reduced[:,:,None]).sum(1))
        for row in reduced:self.assertEqual(sorted(row.tolist()),list(range(slots)))
        torch.testing.assert_close((on.gather(1,reduced)&real).sum(1),torch.minimum(real.sum(1),on.sum(1)))

    def test_loss_and_all_gradients_match_full_padded_matching(self):
        out,b=inputs([1,3,4],[3,2,4])
        with patch('sequential_objective.effective_assignment',side_effect=lambda cost,*args:exact_batched_assignment(cost)):
            baseline=loss(out,b);base_counts=primary_counts(out,b)
            grad=torch.autograd.grad(baseline['loss'],(out['mask_logits'],out['count_logits']),retain_graph=True)
        actual=loss(out,b);actual_counts=primary_counts(out,b)
        actual_grad=torch.autograd.grad(actual['loss'],(out['mask_logits'],out['count_logits']))
        for key in ('loss','count_task','shape_task','mask_dice'):
            torch.testing.assert_close(actual[key],baseline[key],rtol=0,atol=0)
        for a,e in zip(actual_grad,grad):torch.testing.assert_close(a,e,rtol=0,atol=0)
        torch.testing.assert_close(actual_counts,base_counts,rtol=0,atol=0)

    def test_no_real_or_predicted_layers_and_full_capacity_are_handled(self):
        zero=torch.zeros(2,32,dtype=torch.bool);cost=torch.zeros(2,32,32)
        actual=effective_assignment(cost,zero,zero)
        self.assertTrue(torch.equal(actual,torch.arange(32)[None].expand(2,-1)))
        real=torch.ones(2,32,dtype=torch.bool);cost=torch.rand(2,32,32)
        with patch('sequential_objective.exact_batched_assignment',wraps=exact_batched_assignment) as called:
            result=effective_assignment(cost,real,real)
        self.assertEqual(called.call_args.args[0].shape,(2,32,32))
        torch.testing.assert_close(result,exact_batched_assignment(cost))

    def test_teacher_permutation_and_unused_logits_do_not_change_objective(self):
        out,b=inputs([1,2,3],[1,2,3]);baseline=loss(out,b)
        perm=torch.randperm(32)
        shuffled={**b,'teacher_mask':b['teacher_mask'][:,perm],'layer_mask':b['layer_mask'][:,perm]}
        result=loss(out,shuffled)
        torch.testing.assert_close(baseline['loss'],result['loss'])
        torch.testing.assert_close(primary_counts(out,b),primary_counts(out,shuffled))

    def test_disabled_reference_masks_cannot_steer_active_layer_assignment(self):
        out,b=inputs([3],[2]);baseline=activation_assignment(out,b);counts=primary_counts(out,b)
        changed={**out,'mask_probability':out['mask_probability'].clone()}
        inactive=~out['active_slots']
        changed['mask_probability'][inactive]=torch.rand_like(changed['mask_probability'][inactive])
        torch.testing.assert_close(activation_assignment(changed,b),baseline,rtol=0,atol=0)
        torch.testing.assert_close(primary_counts(changed,b),counts,rtol=0,atol=0)

    def test_kernel_preflight_runs_once_and_does_not_consume_rng(self):
        import sequential_objective as module
        rng=torch.get_rng_state().clone()
        with patch.dict(module._VERIFIED_MATCHING,{},clear=True):
            report=verify_effective_assignment(torch.device('cpu'))
            self.assertEqual(report['status'],'passed')
            with patch('sequential_objective.exact_batched_assignment',side_effect=AssertionError('already verified')):
                self.assertEqual(verify_effective_assignment(torch.device('cpu')),report)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
