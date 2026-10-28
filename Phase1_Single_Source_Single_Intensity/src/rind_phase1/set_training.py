"""Differentiable supervision for sets of aligned local-view posteriors.

The single-view locator remains the only learned predictor.  This module adds
an auxiliary objective after residual layers from different views have been
associated upstream.  It never receives source IDs or obstacle geometry.
"""
from __future__ import annotations

import math

import torch

from .model import fuse_view_log_probabilities


def discounted_view_weights(view_mask, redundancy_discount, *, dtype=None):
    """Return weights whose total evidence is ``1 + rho * (m - 1)``.

    ``rho=1`` treats the views as conditionally independent.  ``rho=0`` uses
    a normalized geometric mean and therefore does not sharpen repeated
    evidence.  Intermediate values conservatively account for overlapping
    windows and their shared unknown scene geometry.
    """
    if (view_mask.ndim != 2 or view_mask.dtype != torch.bool or
            not view_mask.any(-1).all()):
        raise ValueError("view_mask must be bool [B,V] with at least one view per track")
    if (not math.isfinite(redundancy_discount) or redundancy_discount < 0 or
            redundancy_discount > 1):
        raise ValueError("redundancy_discount must be in [0,1]")
    dtype = dtype or torch.float32
    count = view_mask.sum(-1, keepdim=True).to(dtype)
    total = 1.0 + float(redundancy_discount) * (count - 1.0)
    return view_mask.to(dtype) * (total / count)


def multiview_distillation_loss(student_log_probability, teacher_probability,
                                view_mask, *, layer_loss_weight=1.0,
                                set_loss_weight=0.5,
                                redundancy_discount=0.5,
                                log_prior=None, epsilon=1e-12):
    """Combine per-layer Phase-I loss with an aligned view-set loss.

    Args:
        student_log_probability: normalized ``[B,V,N]`` student log fields.
        teacher_probability: normalized ``[B,V,N]`` teacher probability fields.
        view_mask: active views in each contribution track.

    The teacher and student fields are fused by the same discounted log-
    evidence rule.  The returned objective is averaged over tracks, so scenes
    can be balanced by selecting an equal number of tracks per scene upstream.
    """
    if (student_log_probability.ndim != 3 or
            teacher_probability.shape != student_log_probability.shape or
            view_mask.shape != student_log_probability.shape[:2] or
            view_mask.dtype != torch.bool):
        raise ValueError("student, teacher and view mask shapes are inconsistent")
    if not view_mask.any(-1).all():
        raise ValueError("every track must contain at least one active view")
    if (not math.isfinite(layer_loss_weight) or layer_loss_weight < 0 or
            not math.isfinite(set_loss_weight) or set_loss_weight < 0 or
            layer_loss_weight + set_loss_weight <= 0):
        raise ValueError("loss weights must be finite, nonnegative and not both zero")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    active = view_mask[..., None]
    selected_student = student_log_probability[active.expand_as(student_log_probability)]
    selected_teacher = teacher_probability[active.expand_as(teacher_probability)]
    if (not torch.isfinite(selected_student).all() or
            not torch.isfinite(selected_teacher).all() or
            torch.any(selected_teacher < 0)):
        raise ValueError("active probability fields must be finite and nonnegative")
    teacher_sums = teacher_probability.sum(-1)
    if not torch.allclose(
            teacher_sums[view_mask],
            torch.ones_like(teacher_sums[view_mask]), rtol=0, atol=1e-5):
        raise ValueError("active teacher probability fields must sum to one")
    student_log_sums = torch.logsumexp(student_log_probability, dim=-1)
    if not torch.allclose(
            student_log_sums[view_mask],
            torch.zeros_like(student_log_sums[view_mask]), rtol=0, atol=1e-5):
        raise ValueError("active student log probability fields must be normalized")

    per_view_cross_entropy = -(
        teacher_probability.detach() * student_log_probability).sum(-1)
    per_track_layer_cross_entropy = (
        per_view_cross_entropy.masked_fill(~view_mask, 0).sum(-1) /
        view_mask.sum(-1).to(per_view_cross_entropy.dtype))

    weights = discounted_view_weights(
        view_mask, redundancy_discount, dtype=student_log_probability.dtype)
    teacher_log = torch.log(teacher_probability.clamp_min(epsilon))
    teacher_fused_log, teacher_fused = fuse_view_log_probabilities(
        teacher_log, view_mask=view_mask, weights=weights, log_prior=log_prior)
    student_fused_log, student_fused = fuse_view_log_probabilities(
        student_log_probability, view_mask=view_mask, weights=weights,
        log_prior=log_prior)
    per_track_set_cross_entropy = -(
        teacher_fused.detach() * student_fused_log).sum(-1)
    per_track_objective = (
        float(layer_loss_weight) * per_track_layer_cross_entropy +
        float(set_loss_weight) * per_track_set_cross_entropy)
    return {
        "loss": per_track_objective.mean(),
        "per_track_objective": per_track_objective,
        "per_track_layer_cross_entropy": per_track_layer_cross_entropy,
        "per_track_set_cross_entropy": per_track_set_cross_entropy,
        "teacher_fused_log_probability": teacher_fused_log,
        "teacher_fused_probability": teacher_fused,
        "student_fused_log_probability": student_fused_log,
        "student_fused_probability": student_fused,
        "weights": weights,
    }
