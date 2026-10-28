import unittest
import torch

from rind_phase1.train import (DEFAULT_SHARPENED_OBJECTIVE,
                               sharpened_objective_loss,
                               validate_student_objective)


class SharpenedObjectiveTests(unittest.TestCase):
    def test_validation_and_finite_gradient(self):
        objective=validate_student_objective(DEFAULT_SHARPENED_OBJECTIVE)
        energy=torch.tensor([[0.0,1.0,2.0,3.0]],requires_grad=True)
        support=torch.ones_like(energy,dtype=torch.bool)
        log_p=torch.log_softmax(-energy,dim=-1)
        probability=log_p.exp()
        teacher=torch.tensor([[.7,.2,.08,.02]])
        total,ce,entropy,ranking=sharpened_objective_loss(
            energy,log_p,probability,teacher,support,objective)
        self.assertTrue(torch.isfinite(total).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(ranking).all())
        self.assertAlmostEqual(float(ce.detach()),float((-(teacher*log_p).sum()).detach()),places=6)
        total.mean().backward()
        self.assertTrue(torch.isfinite(energy.grad).all())

    def test_ranking_rewards_low_positive_energy(self):
        q=torch.tensor([[.8,.15,.04,.01]])
        support=torch.ones_like(q,dtype=torch.bool)
        def loss(values):
            energy=torch.tensor([values],dtype=torch.float32)
            log_p=torch.log_softmax(-energy,dim=-1)
            return float(sharpened_objective_loss(
                energy,log_p,log_p.exp(),q,support,DEFAULT_SHARPENED_OBJECTIVE)[3])
        good=loss([0.,1.,3.,4.])
        bad=loss([3.,1.,0.,0.])
        self.assertLess(good,bad)

    def test_zero_weights_preserve_cross_entropy_value(self):
        objective={**DEFAULT_SHARPENED_OBJECTIVE,
                   'entropy_match_weight':0.0,'ranking_loss_weight':0.0}
        energy=torch.tensor([[.2,-.1,.4]])
        log_p=torch.log_softmax(-energy,dim=-1);q=torch.tensor([[.2,.5,.3]])
        total,ce,_,_=sharpened_objective_loss(
            energy,log_p,log_p.exp(),q,torch.ones_like(q,dtype=torch.bool),objective)
        self.assertTrue(torch.allclose(total,ce))

    def test_uniform_teacher_has_finite_zero_ranking_gradient(self):
        energy=torch.zeros((1,4),requires_grad=True)
        q=torch.full((1,4),.25);support=torch.ones_like(q,dtype=torch.bool)
        log_p=torch.log_softmax(-energy,dim=-1)
        total,_,_,ranking=sharpened_objective_loss(
            energy,log_p,log_p.exp(),q,support,DEFAULT_SHARPENED_OBJECTIVE)
        self.assertEqual(float(ranking.detach()),0.0)
        total.mean().backward()
        self.assertTrue(torch.isfinite(energy.grad).all())


if __name__=='__main__':unittest.main()
