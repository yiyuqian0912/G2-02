"""Count-guided slots; existence is a decision, never a brightness multiplier."""
import torch
from torch import nn

from boundary_refined_model import BoundaryRefinedDecompositionField
from decomposition_model import stable_count_logits


class DisentangledDecompositionField(BoundaryRefinedDecompositionField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.mask_embedding.out_features
        self.global_count_head = nn.Sequential(
            nn.LayerNorm(self.hidden + 2 * width + 4),
            nn.Linear(self.hidden + 2 * width + 4, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.max_slots + 1))
        # Uniform global scores initially preserve the pretrained slot count
        # distribution. Only these new parameters may be absent in warm-start.
        nn.init.zeros_(self.global_count_head[-1].weight)
        nn.init.zeros_(self.global_count_head[-1].bias)

    def _readout(self, slots, pixels, normalized, edges, scale, previous, positive):
        output = super()._readout(slots, pixels, normalized, edges, scale, previous, positive)
        stats = torch.stack((normalized.mean((-1,-2)), normalized.std((-1,-2), correction=0),
                             positive.float().mean((-1,-2)), edges[:,2].mean((-1,-2))), -1)
        context = torch.cat((self.slot_norm(slots).mean(1), pixels.mean((-1,-2)),
                             pixels.amax((-1,-2)), stats.to(pixels.dtype)), -1)
        global_logits = self.global_count_head(context).float()
        slot_logits = stable_count_logits(output['presence_logits'])
        # These are learned classifier scores, NOT a product of physical posteriors.
        fused = slot_logits + global_logits
        classes = torch.arange(self.max_slots + 1, device=slots.device)[None]
        nonempty = positive.flatten(1).any(1, keepdim=True)
        possible = torch.where(nonempty, classes > 0, classes == 0)
        fused = fused.masked_fill(~possible, -1e4)
        global_logits = global_logits.masked_fill(~possible, -1e4)
        predicted_count = fused.argmax(1)
        # Hard cardinality is deliberately nondifferentiable: supervised count
        # and existence losses train this decision, not amplitude compensation.
        order = output['presence_logits'].argsort(dim=1, descending=True, stable=True)
        ranks = order.argsort(dim=1)
        active = ranks < predicted_count[:,None]
        output.update(global_count_logits=global_logits, slot_count_logits=slot_logits,
                      count_logits=fused, active_slots=active, predicted_count=predicted_count,
                      reconstruction=(output['mask_probability'] *
                          (active * output['intensity'])[...,None,None]).sum(1),
                      _disentangled=True, _count_log_space=True)
        return output
