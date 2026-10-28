"""Complete local-response supervision followed by shared Phase-I localization.

The only localization input is the student's predicted per-slot response mask.
Teacher masks, residuals and positions are loss targets and never model inputs.
Masks have independent sigmoid activations because additive contributions may
overlap. All assignment and loss tensors stay on their current device.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rind_phase1.model import SourceEnergyField
from slot_model import masked_log_softmax


class _ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(math.gcd(16, outgoing), outgoing), nn.GELU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(16, outgoing), outgoing), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


def poisson_binomial_logits(presence: torch.Tensor) -> torch.Tensor:
    """Normalized count logits implied by independent Bernoulli slot existence.

    This provides a count objective whose gradient trains the same presence
    variables used by reconstruction. It does not assert physical-source
    independence or identify effective-layer count with generating-source count.
    """
    if presence.ndim != 2:
        raise ValueError("presence must be [B,S]")
    batch, slots = presence.shape
    probability = torch.cat((presence.new_ones(batch, 1, dtype=torch.float32),
                             presence.new_zeros(batch, slots, dtype=torch.float32)), dim=-1)
    for slot in range(slots):
        p = presence[:, slot:slot + 1].float()
        probability = probability * (1 - p) + F.pad(probability[:, :-1], (1, 0)) * p
    return probability.clamp_min(torch.finfo(torch.float32).tiny).log()


class ResponseDecompositionField(nn.Module):
    """Original observation -> local contribution masks -> Phase-I fields.

    ``forward`` reads response/obstacle [B,H,H], window [B,3], candidates
    [B,N,2], support [B,N]. ``obstacle`` only enters the image encoder when
    use_local_obstacle=True; it follows the Phase-I locator's input protocol
    there. No global geometry is supplied.
    """
    def __init__(
        self, *, max_slots: int, backbone_config: dict[str, Any],
        decomposition_hidden: int = 128, slot_decoder_layers: int = 2,
        slot_attention_heads: int = 8, slot_dropout: float = 0.0,
        activation_checkpoint_candidates: bool = True,
        activation_checkpoint_locator_encoder: bool = True,
        activation_checkpoint_decomposer: bool = True,
        use_local_obstacle: bool = False,
        enable_localization: bool = True,
    ) -> None:
        super().__init__()
        if type(max_slots) is not int or not 1 <= max_slots <= 32:
            raise ValueError("max_slots must be in [1,32]")
        if decomposition_hidden < 16 or decomposition_hidden % slot_attention_heads:
            raise ValueError("decomposition_hidden must be >=16 and divisible by slot heads")
        self.max_slots = max_slots
        self.hidden = decomposition_hidden
        self.use_local_obstacle = bool(use_local_obstacle)
        self.backbone_config = dict(backbone_config)
        self.coordinate_scale = float(backbone_config["coordinate_scale"])
        self.enable_localization = bool(enable_localization)
        self.locator = SourceEnergyField(**backbone_config) if self.enable_localization else None
        self.activation_checkpoint_candidates = bool(activation_checkpoint_candidates)
        self.activation_checkpoint_locator_encoder = bool(activation_checkpoint_locator_encoder)
        self.activation_checkpoint_decomposer = bool(activation_checkpoint_decomposer)
        stem_width = max(16, decomposition_hidden // 2)
        self.stem = _ConvBlock(4 if use_local_obstacle else 3, stem_width)
        self.down_one = _ConvBlock(stem_width, decomposition_hidden, 2)
        self.down_two = _ConvBlock(decomposition_hidden, decomposition_hidden, 2)
        self.decode_one = _ConvBlock(2 * decomposition_hidden, decomposition_hidden)
        self.decode_two = _ConvBlock(decomposition_hidden + stem_width, decomposition_hidden)
        self.pixel_keys = nn.Conv2d(decomposition_hidden, decomposition_hidden, 1)
        self.window_encoder = nn.Sequential(
            nn.Linear(3, decomposition_hidden), nn.GELU(), nn.LayerNorm(decomposition_hidden),
            nn.Linear(decomposition_hidden, decomposition_hidden),
        )
        self.slot_queries = nn.Parameter(torch.empty(max_slots, decomposition_hidden))
        nn.init.normal_(self.slot_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            decomposition_hidden, slot_attention_heads, 4 * decomposition_hidden,
            dropout=slot_dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(layer, slot_decoder_layers)
        self.slot_norm = nn.LayerNorm(decomposition_hidden)
        self.mask_embedding = nn.Linear(decomposition_hidden, decomposition_hidden)
        self.mask_bias = nn.Linear(decomposition_hidden, 1)
        self.presence_head = nn.Linear(decomposition_hidden, 1)
        # With 32 available slots, rho=0.5 would initially reconstruct 16
        # contributions. Start with a modest expected count while preserving
        # independently learnable, nonzero presence for every slot.
        initial_presence = min(0.5, 2.0 / max_slots)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.constant_(self.presence_head.bias, math.log(initial_presence / (1 - initial_presence)))
        self.intensity_head = nn.Sequential(
            nn.Linear(decomposition_hidden, decomposition_hidden), nn.GELU(),
            nn.Linear(decomposition_hidden, 1),
        )

    def load_phase1_backbone(self, saved: dict[str, Any]) -> None:
        if self.locator is None:
            raise RuntimeError("Phase1 localization is disabled for this mask-only model")
        state = saved.get("model")
        if not isinstance(state, dict):
            raise ValueError("Phase-I checkpoint does not contain model weights")
        expected = saved.get("model_config")
        if expected is not None and any(expected.get(key) != value
                                        for key, value in self.backbone_config.items()):
            raise ValueError("Phase-I checkpoint configuration differs from locator")
        self.locator.load_state_dict(state, strict=True)

    def set_locator_trainable(self, trainable: bool) -> None:
        if self.locator is not None:
            for parameter in self.locator.parameters():
                parameter.requires_grad_(bool(trainable))

    def _spatial_features(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fine = self.stem(channels)
        medium = self.down_one(fine)
        coarse = self.down_two(medium)
        decoded = self.decode_one(torch.cat((
            F.interpolate(coarse, size=medium.shape[-2:], mode="bilinear", align_corners=False),
            medium,
        ), dim=1))
        decoded = self.decode_two(torch.cat((
            F.interpolate(decoded, size=fine.shape[-2:], mode="bilinear", align_corners=False),
            fine,
        ), dim=1))
        return self.pixel_keys(decoded), F.adaptive_avg_pool2d(
            coarse, (min(16, coarse.shape[-2]), min(16, coarse.shape[-1])))

    def decompose(self, response: torch.Tensor, obstacle: torch.Tensor,
                  window: torch.Tensor) -> dict[str, torch.Tensor]:
        if response.ndim != 3 or obstacle.shape != response.shape or window.shape != (len(response), 3):
            raise ValueError("misaligned response, obstacle, or window")
        height, width = response.shape[-2:]
        if height != width:
            raise ValueError("local observations must be square")
        axis = (torch.arange(height, device=response.device, dtype=response.dtype) + 0.5) / height
        yy, xx = torch.meshgrid(2 * axis - 1, 2 * axis - 1, indexing="ij")
        channels = [response, xx.expand_as(response), yy.expand_as(response)]
        if self.use_local_obstacle:
            channels.append(obstacle.to(response.dtype))
        image = torch.stack(channels, dim=1).contiguous(memory_format=torch.channels_last)
        if self.activation_checkpoint_decomposer and self.training and torch.is_grad_enabled():
            pixels, feature = checkpoint(self._spatial_features, image,
                                         use_reentrant=False, preserve_rng_state=False)
        else:
            pixels, feature = self._spatial_features(image)
        memory = feature.flatten(2).transpose(1, 2)
        position = self.window_encoder(window.float() / self.coordinate_scale)[:, None]
        slots = self.slot_norm(self.slot_decoder(
            self.slot_queries[None].expand(len(response), -1, -1) + position, memory))
        mask_logits = torch.einsum("bsc,bchw->bshw", self.mask_embedding(slots), pixels)
        mask_logits = mask_logits / math.sqrt(self.hidden) + self.mask_bias(slots)[..., None]
        # In the noiseless positive-additive model a zero observed pixel cannot
        # contain any active contribution. This exact gate needs no geometry.
        positive = (response > 1e-12)[:, None]
        masks = torch.sigmoid(mask_logits.float()) * positive
        presence_logits = self.presence_head(slots).squeeze(-1).float()
        presence = torch.sigmoid(presence_logits)
        intensity = F.softplus(self.intensity_head(slots).squeeze(-1).float())
        contribution = presence[..., None, None] * intensity[..., None, None] * masks
        return {
            "mask_logits": mask_logits.float(), "mask_probability": masks,
            "presence_logits": presence_logits, "presence_probability": presence,
            "intensity": intensity, "reconstruction": contribution.sum(dim=1),
            "count_logits": poisson_binomial_logits(presence),
        }

    def locate_masks(self, masks: torch.Tensor, obstacle: torch.Tensor, window: torch.Tensor,
                     candidate_xy: torch.Tensor, support: torch.Tensor,
                     candidate_chunk: int | None = None) -> dict[str, torch.Tensor]:
        if self.locator is None:
            raise RuntimeError("mask-only model cannot compute location fields; enable localization explicitly")
        batch, slots, height, width = masks.shape
        flat_masks = masks.reshape(batch * slots, height, width)
        flat_obstacle = obstacle[:, None].expand(-1, slots, -1, -1).reshape(batch * slots, height, width)
        flat_window = window[:, None].expand(-1, slots, -1).reshape(batch * slots, 3)
        grad_enabled = self.training and torch.is_grad_enabled()
        if self.activation_checkpoint_locator_encoder and grad_enabled:
            encoded = checkpoint(
                self.locator.encode_observation, flat_masks, flat_obstacle, flat_window,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            encoded = self.locator.encode_observation(flat_masks, flat_obstacle, flat_window)
        chunk = candidate_xy.shape[1] if candidate_chunk is None else candidate_chunk
        if chunk < 1:
            raise ValueError("candidate_chunk must be positive")
        energies = []

        def score(points: torch.Tensor, *features: torch.Tensor) -> torch.Tensor:
            return self.locator.score_encoded(tuple(features), points)

        for start in range(0, candidate_xy.shape[1], chunk):
            points = candidate_xy[:, start:start + chunk]
            flat_points = points[:, None].expand(-1, slots, -1, -1).reshape(batch * slots, len(points[0]), 2)
            if self.activation_checkpoint_candidates and grad_enabled:
                current = checkpoint(score, flat_points, *encoded,
                                     use_reentrant=False, preserve_rng_state=False)
            else:
                current = score(flat_points, *encoded)
            energies.append(current.reshape(batch, slots, -1))
        energy = torch.cat(energies, dim=-1)
        log_probability, probability = masked_log_softmax(energy, support)
        return {"energy": energy, "slot_log_probability": log_probability,
                "slot_probability": probability}

    def forward(self, response: torch.Tensor, obstacle: torch.Tensor, window: torch.Tensor,
                candidate_xy: torch.Tensor, support: torch.Tensor,
                candidate_chunk: int | None = None,
                compute_location: bool = True) -> dict[str, torch.Tensor]:
        output = self.decompose(response, obstacle, window)
        if compute_location:
            output.update(self.locate_masks(output["mask_probability"], obstacle, window,
                                            candidate_xy, support, candidate_chunk))
            weighted = output["presence_probability"][:, :, None] * output["slot_probability"]
            output["occupancy"] = 1 - torch.prod((1 - weighted).clamp(0, 1), dim=1)
            output["contribution_density"] = weighted.sum(dim=1)
        return output


@torch.no_grad()
def exact_batched_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Exact square Hungarian assignment on GPU or CPU, no host tensor copies.

    Returns [B,S] teacher-row -> predicted-slot indices. All S rows include
    explicit empty targets, so positive and negative presence costs both take
    part in the global assignment. Fixed O(S^3) loops support up to 32 slots.
    """
    if cost.ndim != 3 or cost.shape[1] != cost.shape[2] or cost.shape[1] < 1:
        raise ValueError("cost must be nonempty square [B,S,S]")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("matching cost must be finite")
    cost = cost.detach().float()
    batch, slots, _ = cost.shape
    rows = torch.arange(batch, device=cost.device)
    u = cost.new_zeros(batch, slots + 1)
    v = cost.new_zeros(batch, slots + 1)
    p = torch.zeros(batch, slots + 1, dtype=torch.long, device=cost.device)
    way = torch.zeros_like(p)
    for teacher in range(1, slots + 1):
        p[:, 0] = teacher
        minimum = cost.new_full((batch, slots + 1), torch.inf)
        used = torch.zeros(batch, slots + 1, dtype=torch.bool, device=cost.device)
        column = torch.zeros(batch, dtype=torch.long, device=cost.device)
        active = torch.ones(batch, dtype=torch.bool, device=cost.device)
        for _ in range(slots):
            used[rows, column] |= active
            row = p[rows, column]
            reduced = cost[rows, (row - 1).clamp_min(0)] - u[rows, row, None] - v[:, 1:]
            available = (~used[:, 1:]) & active[:, None]
            improve = available & (reduced < minimum[:, 1:])
            minimum[:, 1:] = torch.where(improve, reduced, minimum[:, 1:])
            way[:, 1:] = torch.where(improve, column[:, None], way[:, 1:])
            selection = minimum[:, 1:].masked_fill(~available, torch.inf)
            delta, next_column = selection.min(dim=1)
            delta = torch.where(active, delta, torch.zeros_like(delta))
            next_column = next_column + 1
            u.scatter_add_(1, p, used.to(cost.dtype) * delta[:, None])
            v -= used.to(cost.dtype) * delta[:, None]
            minimum[:, 1:] -= available.to(cost.dtype) * delta[:, None]
            column = torch.where(active, next_column, column)
            active = active & (p[rows, column] != 0)
        active = torch.ones(batch, dtype=torch.bool, device=cost.device)
        for _ in range(slots):
            previous = way[rows, column]
            p[rows, column] = torch.where(active, p[rows, previous], p[rows, column])
            column = torch.where(active, previous, column)
            active = active & (column != 0)
    result = torch.empty(batch, slots, dtype=torch.long, device=cost.device)
    result.scatter_(1, p[:, 1:] - 1, torch.arange(slots, device=cost.device)[None].expand(batch, -1))
    return result


def decomposition_training_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *,
    mask_weight: float = 1.0, dice_weight: float = 1.0,
    intensity_weight: float = 0.25, presence_weight: float = 1.0,
    empty_presence_weight: float = 0.25, location_weight: float = 1.0,
    reconstruction_weight: float = 1.0, residual_weight: float = 0.25,
    negative_residual_weight: float = 0.1, count_weight: float = 0.25,
    match_location_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Complete decomposition supervision, normalized by actual layer counts.

    batch: teacher_mask [B,S,H,W], teacher_intensity/layer_mask/
    localization_valid [B,S], teacher_probability [B,S,N], response/obstacle
    [B,H,W]. Real teacher rows must be stored in residual-peeling order and
    all arithmetic layers must be present. Padded rows have layer_mask=False.
    """
    masks = output["mask_probability"].float()
    logits = output["mask_logits"].float()
    target = batch["teacher_mask"].float()
    real = batch["layer_mask"].bool()
    if target.shape != masks.shape or logits.shape != masks.shape or real.shape != masks.shape[:2]:
        raise ValueError("misaligned complete layer targets")
    b, s, _, _ = masks.shape
    free = (~batch["obstacle"]).float().flatten(1)
    free_count = free.sum(-1).clamp_min(1)
    target_flat = target.flatten(2)
    mask_flat = masks.flatten(2)
    logit_flat = logits.flatten(2)
    teacher_count = real.sum(-1)
    positive_count = teacher_count.clamp_min(1)
    empty_count = (~real).sum(-1).clamp_min(1)
    bce_pair = (
        (F.softplus(logit_flat) * free[:, None]).sum(-1)[:, None, :]
        - torch.einsum("btp,bsp,bp->bts", target_flat, logit_flat, free)
    ) / free_count[:, None, None]
    intersection = torch.einsum("btp,bsp,bp->bts", target_flat, mask_flat, free)
    area_teacher = (target_flat * free[:, None]).sum(-1)[:, :, None]
    area_student = (mask_flat * free[:, None]).sum(-1)[:, None, :]
    dice_pair = 1 - (2 * intersection + 1e-6) / (area_teacher + area_student + 1e-6)
    mask_pair = bce_pair + dice_weight * dice_pair
    teacher_a = batch["teacher_intensity"].float()
    predicted_a = output["intensity"].float()
    intensity_pair = F.smooth_l1_loss(
        torch.log(teacher_a.clamp_min(1e-6))[:, :, None].expand(-1, -1, s),
        torch.log(predicted_a.clamp_min(1e-6))[:, None, :].expand(-1, s, -1),
        reduction="none",
    )
    presence_logits = output["presence_logits"].float()
    positive_presence = F.softplus(-presence_logits)[:, None, :]
    negative_presence = F.softplus(presence_logits)[:, None, :]
    location_pair = masks.new_zeros((b, s, s))
    location_valid = batch.get("localization_valid", torch.zeros_like(real)).bool() & real
    if "slot_log_probability" in output:
        q = batch["teacher_probability"].float()
        q = q / q.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        log_p = output["slot_log_probability"].float()
        # q has zero mass outside shared support; replacing -inf avoids 0*-inf.
        safe_log_p = torch.where(torch.isfinite(log_p), log_p, torch.zeros_like(log_p))
        entropy_term = (q * q.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
        location_pair = -torch.einsum("btn,bsn->bts", q, safe_log_p) + entropy_term[:, :, None]
    matching_cost = (mask_weight * mask_pair.detach()
                     + intensity_weight * intensity_pair.detach()
                     + presence_weight * positive_presence.detach()) / positive_count[:, None, None]
    matching_cost = (matching_cost + match_location_weight * location_pair.detach()
                     * location_valid[:, :, None] / location_valid.sum(-1).clamp_min(1)[:, None, None])
    dummy_cost = (presence_weight * empty_presence_weight * negative_presence.detach()
                  / empty_count[:, None, None])
    matching_cost = torch.where(real[:, :, None], matching_cost, dummy_cost.expand(-1, s, -1))
    assignment = exact_batched_assignment(matching_cost)
    gather_pair = assignment[:, :, None]
    matched_mask = torch.gather(mask_pair, 2, gather_pair).squeeze(-1)
    matched_dice = torch.gather(dice_pair, 2, gather_pair).squeeze(-1)
    matched_a = torch.gather(intensity_pair, 2, gather_pair).squeeze(-1)
    matched_loc = torch.gather(location_pair, 2, gather_pair).squeeze(-1)
    mask_loss = ((matched_mask * real).sum(-1) / positive_count).mean()
    intensity_loss = ((matched_a * real).sum(-1) / positive_count).mean()
    local_count = location_valid.sum(-1).clamp_min(1)
    location_loss = ((matched_loc * location_valid).sum(-1) / local_count).mean()
    aligned_logits = presence_logits.gather(1, assignment)
    presence_loss = (
        (F.softplus(-aligned_logits) * real).sum(-1) / positive_count
        + empty_presence_weight * (F.softplus(aligned_logits) * ~real).sum(-1) / empty_count
    ).mean()
    reconstruction_error = (output["reconstruction"].float() - batch["response"].float()).abs().flatten(1)
    reconstruction_loss = ((reconstruction_error * free).sum(-1) / free_count).mean()
    aligned_masks = masks.gather(1, assignment[..., None, None].expand_as(masks))
    aligned_strength = predicted_a.gather(1, assignment)
    aligned_presence = output["presence_probability"].float().gather(1, assignment)
    predicted_contribution = aligned_masks * (aligned_strength * aligned_presence * real)[..., None, None]
    target_contribution = target * (teacher_a * real)[..., None, None]
    predicted_residual = batch["response"].float()[:, None] - predicted_contribution.cumsum(1)
    target_residual = batch["response"].float()[:, None] - target_contribution.cumsum(1)
    residual_error = (predicted_residual - target_residual).abs().flatten(2)
    residual_loss = (((residual_error * free[:, None]).sum(-1) / free_count[:, None]
                      * real).sum(-1) / positive_count).mean()
    negative_error = F.relu(-predicted_residual).flatten(2)
    negative_loss = (((negative_error * free[:, None]).sum(-1) / free_count[:, None]
                      * real).sum(-1) / positive_count).mean()
    count_loss = F.cross_entropy(output["count_logits"].float(), teacher_count)
    loss = (mask_weight * mask_loss + intensity_weight * intensity_loss
            + presence_weight * presence_loss + location_weight * location_loss
            + reconstruction_weight * reconstruction_loss + residual_weight * residual_loss
            + negative_residual_weight * negative_loss + count_weight * count_loss)
    predicted_count = (output["presence_probability"] >= 0.5).sum(-1)
    metrics = {
        "mask": mask_loss, "intensity": intensity_loss, "presence": presence_loss,
        "location": location_loss, "reconstruction": reconstruction_loss,
        "residual": residual_loss, "negative_residual": negative_loss, "count": count_loss,
        "mask_dice": ((1 - matched_dice) * real).sum() / real.sum().clamp_min(1),
        "count_accuracy": (predicted_count == teacher_count).float().mean(),
        "count_mae": (predicted_count - teacher_count).abs().float().mean(),
        "matched_presence": (aligned_presence * real).sum() / real.sum().clamp_min(1),
    }
    return {"loss": loss, "assignment": assignment.detach(),
            **{name: value.detach() for name, value in metrics.items()}}
