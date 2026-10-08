"""Student-only conditional energy field. Coordinates must already be normalized."""
import math
import torch
from torch import nn


class FourierXY(nn.Module):
    def __init__(self, frequencies=6):
        super().__init__()
        if frequencies < 0:
            raise ValueError('frequencies must be nonnegative')
        self.frequencies = frequencies

    def forward(self, xy):
        out = [xy]
        for k in range(self.frequencies):
            w = 2.0 ** k * math.pi
            out.extend((torch.sin(w * xy), torch.cos(w * xy)))
        return torch.cat(out, dim=-1)


class ObservationEncoder(nn.Sequential):
    def __init__(self, hidden=128):
        super().__init__(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
                         nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                         nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.GELU())


class CandidateAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.keys = nn.Linear(hidden, hidden)
        self.values = nn.Linear(hidden, hidden)

    def forward(self, query, tokens):
        logits = query @ self.keys(tokens).transpose(1, 2) / math.sqrt(query.shape[-1])
        return logits.softmax(dim=-1) @ self.values(tokens)


def student_distribution(energy, valid_mask=None):
    if energy.ndim != 2 or energy.shape[1] == 0:
        raise ValueError('energy must have shape [B,K], K > 0')
    mask = torch.ones_like(energy, dtype=torch.bool) if valid_mask is None else valid_mask
    if mask.shape != energy.shape or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('mask must be bool [B,K] with at least one valid candidate per row')
    if not torch.isfinite(energy[mask]).all():
        raise ValueError('valid energies must be finite')
    log_prob = (-energy).masked_fill(~mask, -torch.inf).log_softmax(-1)
    return log_prob, log_prob.exp()


class SourceEnergyField(nn.Module):
    def __init__(self, hidden=128, fourier_frequencies=6):
        super().__init__()
        if hidden < 2:
            raise ValueError('hidden must be at least 2')
        self.image_encoder = ObservationEncoder(hidden)
        self.xy = FourierXY(fourier_frequencies)
        self.source_query = nn.Sequential(nn.Linear(2 * (1 + 2 * fourier_frequencies), hidden),
                                          nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = CandidateAttention(hidden)
        self.energy_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden // 2), nn.GELU(),
                                         nn.Linear(hidden // 2, 1))

    def forward(self, observation, candidates):
        if observation.ndim != 4 or observation.shape[1] != 1:
            raise ValueError('observation must be [B,1,H,W]')
        if candidates.ndim != 3 or candidates.shape[0] != observation.shape[0] or candidates.shape[-1] != 2 or candidates.shape[1] == 0:
            raise ValueError('candidates must be [B,K,2], K > 0')
        tokens = self.image_encoder(observation).flatten(2).transpose(1, 2)
        query = self.source_query(self.xy(candidates))
        context = self.attention(query, tokens)
        return self.energy_head(torch.cat((query, context), -1)).squeeze(-1)

    def predict(self, response, candidate_xy, valid_mask=None):
        energy = self(response, candidate_xy)
        log_prob, prob = student_distribution(energy, valid_mask)
        return {'energy': energy, 'student_probability': prob, 'student_log_probability': log_prob}
