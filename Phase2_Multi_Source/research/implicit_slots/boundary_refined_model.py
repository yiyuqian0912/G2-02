"""Full observation -> overlapping contribution slots, with no teacher inputs.

Three shared-weight synchronous refinements read ONLY their own reconstruction
errors. Boundary/region/global heads use different memories. Independent mask
sigmoids preserve additive overlap; competition is limited to feature reading.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from decomposition_model import ResponseDecompositionField, _ConvBlock, poisson_binomial_logits, stable_count_logits
from rind_phase1.model import SourceEnergyField


def edge_channels(value):
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-12)
    return torch.stack((dx, dy, magnitude), 1)


def pooled_tokens(value, limits):
    tokens = []
    for limit in limits:
        side = min(limit, value.shape[-1], value.shape[-2])
        tokens.append(F.adaptive_avg_pool2d(value, (side, side)).flatten(2).transpose(1, 2))
    return torch.cat(tokens, 1)


class RoleAttention(nn.Module):
    def __init__(self, hidden, heads, role_heads, competitive):
        super().__init__()
        self.heads, self.width = role_heads, hidden // heads
        width = role_heads * self.width
        self.query = nn.Linear(hidden, width)
        self.key = nn.Linear(hidden, width)
        self.value = nn.Linear(hidden, width)
        self.competitive = competitive

    def forward(self, slots, memory, gate):
        b, s, _ = slots.shape
        q = self.query(slots).reshape(b, s, self.heads, self.width).transpose(1, 2)
        k = self.key(memory).reshape(b, -1, self.heads, self.width).transpose(1, 2)
        v = self.value(memory).reshape(b, -1, self.heads, self.width).transpose(1, 2)
        if self.competitive:
            # A token distributes evidence across slots, then each slot pools
            # its assigned evidence spatially. This is NOT a mask softmax.
            # .float() alone does not override the surrounding autocast:
            # matmul would still run in bf16. Keep competition/normalization
            # explicitly in fp32; value aggregation remains AMP accelerated.
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.width)
                weights = scores.softmax(dim=2) * gate[:, None].float()
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            result = torch.matmul(weights.to(v.dtype), v)
        else:
            result = F.scaled_dot_product_attention(q, k, v)
        return result.transpose(1, 2).reshape(b, s, -1)


class RoleDecoderBlock(nn.Module):
    def __init__(self, hidden, heads, role_heads):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.memory_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.roles = nn.ModuleList([RoleAttention(hidden, heads, h, i < 2)
                                    for i, h in enumerate(role_heads)])
        self.cross_output = nn.Linear(hidden, hidden)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, slots, memories, gates):
        q = self.self_norm(slots)
        slots = slots + self.self_attention(q, q, q, need_weights=False)[0]
        q = self.cross_norm(slots)
        values = [role(q, norm(memory), gate) for role, norm, memory, gate in
                  zip(self.roles, self.memory_norms, memories, gates)]
        slots = slots + self.cross_output(torch.cat(values, -1))
        return slots + self.ff(self.ff_norm(slots))


class BoundaryRefinedDecompositionField(ResponseDecompositionField):
    """Same public forward/output protocol as the legacy decomposer.

    The inherited Phase1 methods are optional and disabled for the delivered
    mask-only configuration. No legacy decomposer parameters are instantiated.
    """
    def __init__(self, *, max_slots, backbone_config, decomposition_hidden=256,
                 slot_decoder_layers=4, slot_attention_heads=8, refinement_rounds=3,
                 head_roles=(4, 2, 2), mask_feature_dim=128, use_local_obstacle=False,
                 activation_checkpoint_decomposer=False, enable_localization=False,
                 activation_checkpoint_candidates=True,
                 activation_checkpoint_locator_encoder=True):
        nn.Module.__init__(self)
        d, heads = decomposition_hidden, slot_attention_heads
        if not 1 <= max_slots <= 32 or d < 32 or d % heads or d % 8:
            raise ValueError("refined slots require S in [1,32], hidden divisible by heads and 8")
        if len(head_roles) != 3 or min(head_roles) < 1 or sum(head_roles) != heads:
            raise ValueError("head_roles must be positive boundary/region/global counts summing to heads")
        if refinement_rounds < 1 or slot_decoder_layers < 1 or mask_feature_dim < 8:
            raise ValueError("refinement rounds, decoder depth and mask width must be positive")
        self.max_slots, self.hidden, self.refinement_rounds = max_slots, d, refinement_rounds
        self.use_local_obstacle = bool(use_local_obstacle)
        self.backbone_config = dict(backbone_config)
        self.coordinate_scale = float(backbone_config["coordinate_scale"])
        self.enable_localization = bool(enable_localization)
        self.locator = SourceEnergyField(**backbone_config) if enable_localization else None
        self.activation_checkpoint_decomposer = activation_checkpoint_decomposer
        self.activation_checkpoint_candidates = activation_checkpoint_candidates
        self.activation_checkpoint_locator_encoder = activation_checkpoint_locator_encoder
        r0, r1, r2, r3 = max(16, d // 4), max(16, d // 2), d, 3 * d // 2
        b0, b1, b2 = max(8, d // 8), max(16, d // 4), max(16, d // 2)
        self.response_stem = _ConvBlock(5 if use_local_obstacle else 4, r0)
        self.response_down = nn.ModuleList([_ConvBlock(r0, r1, 2), _ConvBlock(r1, r2, 2),
                                           _ConvBlock(r2, r3, 2)])
        self.boundary_stem = _ConvBlock(3, b0)
        self.boundary_down = nn.ModuleList([_ConvBlock(b0, b1, 2), _ConvBlock(b1, b2, 2)])
        self.response_projects = nn.ModuleList([nn.Conv2d(c, d, 1) for c in (r1, r2, r3)])
        self.boundary_projects = nn.ModuleList([nn.Conv2d(c, d, 1) for c in (b1, b2, b2)])
        self.decode_medium = _ConvBlock(2 * d, d)
        self.decode_fine = _ConvBlock(2 * d, d // 2)
        self.decode_native = _ConvBlock(d // 2 + r0 + b0, mask_feature_dim)
        self.pixel_keys = nn.Conv2d(mask_feature_dim, mask_feature_dim, 1)
        self.error_encoder = nn.Sequential(_ConvBlock(2, d // 2, 2), _ConvBlock(d // 2, d, 2))
        self.slot_queries = nn.Parameter(torch.randn(max_slots, d) * .02)
        self.round_embedding = nn.Parameter(torch.randn(refinement_rounds, d) * .02)
        self.memory_roles = nn.Parameter(torch.randn(3, d) * .02)
        self.window_encoder = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))
        self.decoder = nn.ModuleList([RoleDecoderBlock(d, heads, head_roles) for _ in range(slot_decoder_layers)])
        self.slot_norm = nn.LayerNorm(d)
        self.mask_embedding = nn.Linear(d, mask_feature_dim)
        self.mask_bias = nn.Linear(d, 1)
        self.presence_head = nn.Linear(d + mask_feature_dim, 1)
        initial = min(.5, 2 / max_slots)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.constant_(self.presence_head.bias, math.log(initial / (1 - initial)))
        # Five statistics: observed mean inside/outside, edge strength inside,
        # previous-round explanation inside, and fractional mask support.
        self.intensity_head = nn.Sequential(nn.Linear(d + mask_feature_dim + 5, d), nn.GELU(), nn.Linear(d, 1))

    def _features(self, image, edges):
        fine = self.response_stem(image)
        responses, current = [], fine
        for block, project in zip(self.response_down, self.response_projects):
            current = block(current)
            responses.append(project(current))
        boundary_fine = self.boundary_stem(edges)
        one = self.boundary_down[0](boundary_fine)
        two = self.boundary_down[1](one)
        three = F.adaptive_avg_pool2d(two, responses[2].shape[-2:])
        boundaries = [project(value) for project, value in zip(self.boundary_projects, (one, two, three))]
        fused = [r + b for r, b in zip(responses, boundaries)]
        up = lambda value, target: F.interpolate(value, size=target.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_medium(torch.cat((up(fused[2], fused[1]), fused[1]), 1))
        decoded = self.decode_fine(torch.cat((up(decoded, fused[0]), fused[0]), 1))
        pixels = self.pixel_keys(self.decode_native(torch.cat((up(decoded, fine), fine, boundary_fine), 1)))
        # Each role retains multiscale evidence. No top-edge selection deletes
        # weak regions or disconnected components.
        boundary_memory = torch.cat([pooled_tokens(value, (limit,)) for value, limit in
                                     zip(boundaries, (16, 12, 8))], 1)
        region_memory = torch.cat([pooled_tokens(value, (limit,)) for value, limit in
                                   zip(responses, (16, 12, 8))], 1)
        global_memory = pooled_tokens(responses[-1], (8, 4, 1))
        return pixels, boundary_memory, region_memory, global_memory

    def _gate(self, masks, side):
        return .1 + .9 * torch.cat([pooled_tokens(masks, (limit,)).transpose(1, 2)
                                    for limit in side], -1)

    def _readout(self, slots, pixels, normalized, edges, scale, previous, positive):
        slots = self.slot_norm(slots)
        logits = torch.einsum("bsc,bchw->bshw", self.mask_embedding(slots), pixels)
        logits = logits / math.sqrt(pixels.shape[1]) + self.mask_bias(slots)[..., None]
        masks = logits.float().sigmoid() * positive[:, None]
        area = masks.sum((-1, -2)).clamp_min(1e-6)
        feature = torch.einsum("bshw,bchw->bsc", masks.to(pixels.dtype), pixels).float() / area[..., None]
        inside = lambda value: (masks * value[:, None]).sum((-1, -2)) / area
        outside = ((1 - masks) * normalized[:, None]).sum((-1, -2)) / (1 - masks).sum((-1, -2)).clamp_min(1e-6)
        stats = torch.stack((inside(normalized), outside, inside(edges[:, 2]), inside(previous),
                             masks.mean((-1, -2))), -1)
        presence_logits = self.presence_head(torch.cat((slots, feature.to(slots.dtype)), -1)).squeeze(-1).float()
        presence = presence_logits.sigmoid()
        relative = self.intensity_head(torch.cat((slots, feature.to(slots.dtype), stats.to(slots.dtype)), -1)).squeeze(-1).float().sigmoid()
        intensity = relative * scale[:, None]
        reconstruction = (masks * (presence * intensity)[..., None, None]).sum(1)
        return {"mask_logits": logits.float(), "mask_probability": masks,
                "presence_logits": presence_logits, "presence_probability": presence,
                "intensity": intensity, "reconstruction": reconstruction,
                "count_logits": (stable_count_logits(presence_logits) if getattr(self, 'stable_count_output', False)
                                 else poisson_binomial_logits(presence)),
                **({'_count_log_space': True} if getattr(self, 'stable_count_output', False) else {})}

    def decompose(self, response, obstacle, window):
        if response.ndim != 3 or obstacle.shape != response.shape or window.shape != (len(response), 3):
            raise ValueError("misaligned response, obstacle, or window")
        h, w = response.shape[-2:]
        if h != w or min(h, w) < 8:
            raise ValueError("refined decomposer requires square windows of side >=8")
        scale = response.float().amax((-1, -2)).clamp_min(1e-12)
        u = response.float() / scale[:, None, None]
        axis = (torch.arange(h, device=response.device, dtype=torch.float32) + .5) / h
        yy, xx = torch.meshgrid(2 * axis - 1, 2 * axis - 1, indexing="ij")
        channels = [u, (response > 1e-12).float(), xx.expand_as(u), yy.expand_as(u)]
        if self.use_local_obstacle:
            channels.append(obstacle.float())
        image = torch.stack(channels, 1).contiguous(memory_format=torch.channels_last)
        edges = edge_channels(u)
        if self.training and self.activation_checkpoint_decomposer:
            pixels, *base_memory = checkpoint(self._features, image, edges, use_reentrant=False)
        else:
            pixels, *base_memory = self._features(image, edges)
        slots = self.slot_queries[None] + self.window_encoder(window.float() / self.coordinate_scale)[:, None]
        previous = torch.zeros_like(u)
        masks = torch.zeros(len(response), self.max_slots, h, w, device=response.device)
        outputs = []
        # Token grids match _features exactly, including small native windows.
        grids = [tuple(min(limit, (h + 2**level - 1) // 2**level) for limit, level in
                       ((16, 1), (12, 2), (8, 3)))] * 2
        global_grids = tuple(min(limit, (h + 7) // 8) for limit in (8, 4, 1))
        for iteration in range(self.refinement_rounds):
            error = u - previous
            error_feature = self.error_encoder(torch.stack((error.clamp_min(0), (-error).clamp_min(0)), 1))
            error_memories = [torch.cat([F.interpolate(error_feature, size=(side, side), mode="bilinear",
                                                                       align_corners=False).flatten(2).transpose(1, 2)
                                             for side in sizes], 1) for sizes in (*grids, global_grids)]
            memories = [memory + delta + self.memory_roles[i] for i, (memory, delta) in
                        enumerate(zip(base_memory, error_memories))]
            gates = [self._gate(masks, sizes) for sizes in (*grids, global_grids)]
            slots = slots + self.round_embedding[iteration]
            for block in self.decoder:
                if self.training and self.activation_checkpoint_decomposer:
                    slots = checkpoint(block, slots, memories, gates, use_reentrant=False)
                else:
                    slots = block(slots, memories, gates)
            current = self._readout(slots, pixels, u, edges, scale, previous, response > 1e-12)
            outputs.append(current)
            masks = current["mask_probability"]
            previous = current["reconstruction"] / scale[:, None, None]
        return {**outputs[-1], "refinement_outputs": outputs[:-1]}

    def forward(self, response, obstacle, window, candidate_xy=None, support=None,
                candidate_chunk=None, compute_location=None):
        # Mask-only inference needs no candidate pool or Phase1 locator. Keep
        # the inherited full pipeline available only when explicitly enabled.
        if compute_location is None:
            compute_location = self.enable_localization
        if compute_location and (candidate_xy is None or support is None):
            raise ValueError("localization requires candidate_xy and support")
        return super().forward(response, obstacle, window, candidate_xy, support,
                               candidate_chunk, compute_location=compute_location)
