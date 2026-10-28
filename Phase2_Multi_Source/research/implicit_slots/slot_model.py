"""One-pass implicit contribution-slot student for Phase II.

The residual decomposition is teacher-only.  The student encodes the original
multi-source response once, lets unordered DETR-style contribution queries
read the same observation tokens, and reuses the Phase-I candidate scorer for
each query.  Cross-attention is independent across slots: a response pixel may
support several contributions, as required by additive visibility physics.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rind_phase1.model import SourceEnergyField


def masked_log_softmax(energy: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[B,R,N]`` energies to slot-wise probabilities."""
    if energy.ndim != 3 or support.ndim != 2 or energy.shape[0] != support.shape[0] or energy.shape[2] != support.shape[1]:
        raise ValueError("energy [B,R,N] and support [B,N] are misaligned")
    if support.dtype != torch.bool or not bool(support.any(-1).all()):
        raise ValueError("every observation requires at least one supported candidate")
    logits = (-energy.float()).masked_fill(~support[:, None, :], -torch.inf)
    log_probability = torch.log_softmax(logits, dim=-1)
    probability = log_probability.exp().masked_fill(~support[:, None, :], 0.0)
    return log_probability, probability


class ImplicitContributionSlotField(nn.Module):
    """Predict unordered Phase-I-style fields from one multi-source response."""

    def __init__(
        self,
        *,
        max_slots: int = 8,
        slot_decoder_layers: int = 2,
        slot_attention_heads: int = 8,
        slot_dropout: float = 0.0,
        activation_checkpoint_candidates: bool = True,
        backbone_config: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden = int(backbone_config["hidden"])
        if max_slots < 1:
            raise ValueError("max_slots must be positive")
        if slot_decoder_layers < 1:
            raise ValueError("slot_decoder_layers must be positive")
        if slot_attention_heads < 1 or hidden % slot_attention_heads:
            raise ValueError("slot_attention_heads must divide hidden")
        if not 0 <= slot_dropout < 1:
            raise ValueError("slot_dropout must lie in [0,1)")
        self.max_slots = int(max_slots)
        self.hidden = hidden
        self.activation_checkpoint_candidates = bool(activation_checkpoint_candidates)
        self.backbone_config = dict(backbone_config)
        self.backbone = SourceEnergyField(**backbone_config)
        if self.backbone.architecture != "multiscale_boundary":
            raise ValueError("implicit slots require the Phase-I multiscale_boundary backbone")

        self.slot_queries = nn.Parameter(torch.empty(max_slots, hidden))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.role_embeddings = nn.Parameter(torch.empty(3, hidden))
        nn.init.normal_(self.role_embeddings, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=slot_attention_heads,
            dim_feedforward=4 * hidden,
            dropout=slot_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(decoder_layer, slot_decoder_layers)
        self.slot_norm = nn.LayerNorm(hidden)
        self.slot_to_query = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        )
        nn.init.normal_(self.slot_to_query[-1].weight, std=0.01)
        nn.init.zeros_(self.slot_to_query[-1].bias)
        self.presence_head = nn.Linear(hidden, 1)
        self.intensity_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.count_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, max_slots + 1),
        )

    def load_phase1_backbone(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("Phase-I checkpoint does not contain a model state")
        expected = checkpoint.get("model_config")
        if expected is not None:
            relevant = {key: expected.get(key) for key in self.backbone_config}
            if relevant != self.backbone_config:
                raise ValueError("Phase-I checkpoint model_config does not match the slot backbone")
        self.backbone.load_state_dict(state, strict=True)

    @staticmethod
    def _memory(encoded: tuple[torch.Tensor, ...], role_embeddings: torch.Tensor) -> torch.Tensor:
        if len(encoded) not in (11, 14):
            raise ValueError("implicit slots require multiscale Phase-I encoded observations")
        boundary_values = encoded[1] + role_embeddings[0][None, None]
        region_values = encoded[4] + role_embeddings[1][None, None]
        global_values = encoded[7] + role_embeddings[2][None, None]
        return torch.cat((boundary_values, region_values, global_values), dim=1)

    def encode_slots(
        self,
        response: torch.Tensor,
        obstacle: torch.Tensor,
        window: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode_observation(response, obstacle, window)
        memory = self._memory(encoded, self.role_embeddings)
        window_feature = encoded[9]
        queries = self.slot_queries[None].expand(len(response), -1, -1) + window_feature
        slots = self.slot_norm(self.slot_decoder(queries, memory))
        scene_summary = torch.cat((window_feature.squeeze(1), memory.mean(1)), dim=-1)
        return encoded, slots, scene_summary

    def forward(
        self,
        response: torch.Tensor,
        obstacle: torch.Tensor,
        window: torch.Tensor,
        candidate_xy: torch.Tensor,
        support: torch.Tensor,
        candidate_chunk: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_chunk is not None and candidate_chunk < 1:
            raise ValueError("candidate_chunk must be positive")
        encoded, slots, scene_summary = self.encode_slots(response, obstacle, window)
        slot_bias = self.slot_to_query(slots)
        chunk = candidate_xy.shape[1] if candidate_chunk is None else candidate_chunk
        energy_chunks = []
        # Flatten the slot axis into the batch axis so every slot/candidate
        # pair is evaluated by large GPU kernels.  This preserves independent
        # cross-attention while avoiding an eight-call Python loop per chunk.
        repeated_encoded = tuple(value.repeat_interleave(self.max_slots, dim=0) for value in encoded)
        flat_bias = slot_bias.reshape(len(response) * self.max_slots, self.hidden)
        use_checkpoint = (
            self.activation_checkpoint_candidates and self.training and torch.is_grad_enabled()
        )

        def score_chunk(current_candidate: torch.Tensor, current_bias: torch.Tensor,
                        *current_encoded: torch.Tensor) -> torch.Tensor:
            return self.backbone.score_encoded(
                tuple(current_encoded), current_candidate, query_bias=current_bias
            )

        for start in range(0, candidate_xy.shape[1], chunk):
            current = candidate_xy[:, start : start + chunk]
            flat_candidate = current[:, None].expand(-1, self.max_slots, -1, -1).reshape(
                len(response) * self.max_slots, current.shape[1], 2
            )
            if use_checkpoint:
                # Chunking alone only limits temporary tensors: all chunk
                # graphs otherwise remain live until global normalization and
                # backward. Recompute one scorer chunk at a time in backward.
                flat_energy = checkpoint(
                    score_chunk, flat_candidate, flat_bias, *repeated_encoded,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                flat_energy = score_chunk(flat_candidate, flat_bias, *repeated_encoded)
            energy_chunks.append(flat_energy.reshape(len(response), self.max_slots, -1))
        energy = torch.cat(energy_chunks, dim=-1)
        log_probability, probability = masked_log_softmax(energy, support)
        presence_logits = self.presence_head(slots).squeeze(-1).float()
        presence = torch.sigmoid(presence_logits)
        intensity = torch.nn.functional.softplus(self.intensity_head(slots).squeeze(-1).float())
        count_logits = self.count_head(scene_summary).float()
        weighted = presence[:, :, None] * probability
        occupancy = 1.0 - torch.prod((1.0 - weighted).clamp(0.0, 1.0), dim=1)
        contribution_density = weighted.sum(dim=1)
        return {
            "energy": energy,
            "slot_log_probability": log_probability,
            "slot_probability": probability,
            "presence_logits": presence_logits,
            "presence_probability": presence,
            "intensity": intensity,
            "count_logits": count_logits,
            "occupancy": occupancy,
            "contribution_density": contribution_density,
        }


def sinkhorn_assignment(cost: torch.Tensor, temperature: float = 0.05, iterations: int = 20) -> torch.Tensor:
    """GPU-native soft one-to-one assignment for a square batched cost."""
    if cost.ndim != 3 or cost.shape[1] != cost.shape[2]:
        raise ValueError("cost must be [B,R,R]")
    if not math.isfinite(temperature) or temperature <= 0 or iterations < 1:
        raise ValueError("invalid Sinkhorn settings")
    log_assignment = -cost.float().detach() / temperature
    for _ in range(iterations):
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=2, keepdim=True)
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=1, keepdim=True)
    return log_assignment.exp()


def slot_training_loss(
    output: dict[str, torch.Tensor],
    teacher_probability: torch.Tensor,
    teacher_core: torch.Tensor,
    teacher_intensity: torch.Tensor,
    layer_mask: torch.Tensor,
    slot_supervised: torch.Tensor,
    teacher_occupancy: torch.Tensor,
    count_target: torch.Tensor,
    *,
    core_weight: float = 0.25,
    intensity_weight: float = 0.05,
    presence_weight: float = 0.20,
    aggregate_weight: float = 0.50,
    count_weight: float = 0.20,
    consistency_weight: float = 0.10,
    sinkhorn_temperature: float = 0.05,
    sinkhorn_iterations: int = 20,
) -> dict[str, torch.Tensor]:
    """Permutation-invariant teacher matching plus aggregate-field losses."""
    log_p = output["slot_log_probability"]
    p = output["slot_probability"]
    presence_logits = output["presence_logits"]
    intensity = output["intensity"]
    batch, slots, candidates = log_p.shape
    if teacher_probability.shape != (batch, slots, candidates):
        raise ValueError("teacher_probability must be [B,R,N]")
    if teacher_core.shape != teacher_probability.shape or teacher_core.dtype != torch.bool:
        raise ValueError("teacher_core must be boolean [B,R,N]")
    if teacher_intensity.shape != (batch, slots) or layer_mask.shape != (batch, slots):
        raise ValueError("teacher layer metadata must be [B,R]")

    tiny = torch.finfo(torch.float32).tiny
    cross_entropy = -torch.einsum("btn,brn->btr", teacher_probability.float(), log_p.float())
    core_mass = torch.einsum("btn,brn->btr", teacher_core.float(), p.float()).clamp_min(tiny)
    core_loss = -core_mass.log()
    intensity_cost = (teacher_intensity[:, :, None] - intensity[:, None, :]).abs()
    positive_presence = torch.nn.functional.softplus(-presence_logits)[:, None, :]
    real_cost = cross_entropy + core_weight * core_loss + intensity_weight * intensity_cost
    real_cost = real_cost + presence_weight * positive_presence
    negative_presence = presence_weight * torch.nn.functional.softplus(presence_logits)[:, None, :]
    cost = torch.where(layer_mask[:, :, None], real_cost, negative_presence.expand(-1, slots, -1))
    assignment = sinkhorn_assignment(cost, sinkhorn_temperature, sinkhorn_iterations)
    per_record_slot = (assignment * cost).sum((1, 2)) / slots
    supervised_weight = slot_supervised.float()
    slot_loss = (per_record_slot * supervised_weight).sum() / supervised_weight.sum().clamp_min(1.0)

    teacher_occ = teacher_occupancy.float()
    student_occ = output["occupancy"].float()
    intersection = (teacher_occ * student_occ).sum(-1)
    # Squared-norm soft Dice is exactly zero for identical continuous
    # occupancy fields.  The binary-set denominator would penalize a perfect
    # soft target merely because its values are below one.
    aggregate_dice = 1.0 - (2.0 * intersection + 1e-8) / (
        teacher_occ.square().sum(-1) + student_occ.square().sum(-1) + 1e-8
    )
    aggregate_l1 = (teacher_occ - student_occ).abs().mean(-1)
    aggregate_loss = (aggregate_dice + aggregate_l1).mean()
    count_loss = torch.nn.functional.cross_entropy(output["count_logits"], count_target)
    expected_count = output["presence_probability"].sum(-1)
    consistency_loss = torch.nn.functional.smooth_l1_loss(expected_count, count_target.float())
    total = (
        slot_loss
        + aggregate_weight * aggregate_loss
        + count_weight * count_loss
        + consistency_weight * consistency_loss
    )
    return {
        "loss": total,
        "slot": slot_loss.detach(),
        "aggregate": aggregate_loss.detach(),
        "count": count_loss.detach(),
        "consistency": consistency_loss.detach(),
        "count_accuracy": (output["count_logits"].argmax(-1) == count_target).float().mean().detach(),
        "assignment": assignment.detach(),
    }
