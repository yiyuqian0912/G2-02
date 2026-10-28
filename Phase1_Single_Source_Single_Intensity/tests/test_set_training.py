import unittest

import torch

from rind_phase1.set_training import (
    discounted_view_weights, multiview_distillation_loss)


class SetTrainingTests(unittest.TestCase):
    def test_discounted_weights_have_declared_total_evidence(self):
        mask = torch.tensor([[True, True, True], [True, False, False]])
        weights = discounted_view_weights(mask, 0.5)
        torch.testing.assert_close(weights.sum(-1), torch.tensor([2.0, 1.0]))
        self.assertEqual(float(weights[1, 1]), 0.0)

    def test_matching_student_has_finite_gradient(self):
        teacher = torch.tensor([[[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]])
        logits = teacher.log().clone().requires_grad_(True)
        result = multiview_distillation_loss(
            logits.log_softmax(-1), teacher, torch.ones(1, 2, dtype=torch.bool))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(
            float(result["student_fused_probability"].sum().detach()), 1.0)

    def test_padding_is_ignored(self):
        student = torch.tensor([
            [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]],
            [[0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]],
        ]).log()
        teacher = torch.tensor([
            [[0.7, 0.2, 0.1], [0.55, 0.35, 0.1]],
            [[0.1, 0.3, 0.6], [0.0, 0.0, 0.0]],
        ])
        mask = torch.tensor([[True, True], [True, False]])
        result = multiview_distillation_loss(student, teacher, mask)
        torch.testing.assert_close(
            result["student_fused_probability"][1], student[1, 0].exp())

    def test_loss_rejects_unnormalized_teacher(self):
        student = torch.full((1, 2, 3), 1 / 3).log()
        teacher = torch.ones(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            multiview_distillation_loss(
                student, teacher, torch.ones(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
