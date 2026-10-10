"""Canonical Phase I energy field; inputs and saved candidates use world coordinates."""
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
    def __init__(self, hidden=128, channels=2):
        super().__init__(nn.Conv2d(channels, 32, 3, padding=1), nn.GELU(),
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
    def __init__(self, hidden=128, fourier_frequencies=6, coordinate_scale=1024.0,
                 use_obstacle=True, pixel_coordinates=False, normalization_support="geometry"):
        super().__init__()
        if hidden < 2:
            raise ValueError('hidden must be at least 2')
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError('coordinate_scale must be positive')
        if normalization_support not in ('world', 'geometry'):
            raise ValueError('normalization_support must be world or geometry')
        self.use_obstacle = bool(use_obstacle)
        self.pixel_coordinates = bool(pixel_coordinates)
        self.normalization_support = normalization_support
        self.coordinate_scale = float(coordinate_scale)
        self.interface_version = 'phase1-v1'
        self.window_encoder = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.image_encoder = ObservationEncoder(hidden, 4 if self.pixel_coordinates else 2)
        self.xy = FourierXY(fourier_frequencies)
        self.source_query = nn.Sequential(nn.Linear(2 * (1 + 2 * fourier_frequencies), hidden),
                                          nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = CandidateAttention(hidden)
        self.energy_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden // 2), nn.GELU(),
                                         nn.Linear(hidden // 2, 1))

    def encode_observation(self, response, obstacle, window):
        """Encode once for any number of candidates; legacy weights remain loadable."""
        if response.ndim != 3 or obstacle.shape != response.shape or obstacle.dtype != torch.bool:
            raise ValueError('response/obstacle must be [B,size,size], obstacle boolean')
        if window.shape != (len(response),3) or response.shape[1] != response.shape[2] or not (window[:,2] == response.shape[1]).all():
            raise ValueError('window must match the native response size')
        if not torch.isfinite(response).all() or not torch.isfinite(window).all():
            raise ValueError('inputs must be finite')
        mask = obstacle.to(response.dtype) if self.use_obstacle else torch.zeros_like(response)
        channels = [response,mask]
        if self.pixel_coordinates:
            size=response.shape[-1]
            axis=(torch.arange(size,device=response.device,dtype=response.dtype)+.5)/size*2-1
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            channels += [xx.expand_as(response), yy.expand_as(response)]
        tokens=self.image_encoder(torch.stack(channels,dim=1)).flatten(2).transpose(1,2)
        return (self.attention.keys(tokens),self.attention.values(tokens),
                self.window_encoder(window.to(response.dtype)/self.coordinate_scale)[:,None])

    def score_encoded(self, encoded, candidate_xy):
        keys,values,window_feature=encoded
        if candidate_xy.ndim != 3 or candidate_xy.shape[0] != len(keys) or candidate_xy.shape[-1] != 2 or candidate_xy.shape[1] == 0:
            raise ValueError('candidate_xy must be [B,N,2], N>0')
        if not torch.isfinite(candidate_xy).all():
            raise ValueError('candidate coordinates must be finite')
        query=self.source_query(self.xy(candidate_xy/self.coordinate_scale))+window_feature
        context=((query @ keys.transpose(1,2))/math.sqrt(query.shape[-1])).softmax(-1) @ values
        return self.energy_head(torch.cat((query,context),-1)).squeeze(-1)

    def forward(self, response, obstacle, window, candidate_xy):
        return self.score_encoded(self.encode_observation(response,obstacle,window),candidate_xy)

    def predict(self, response, obstacle, window, candidate_xy, valid=None):
        energy = self(response, obstacle, window, candidate_xy)
        support = valid if self.normalization_support == 'geometry' else None
        log_prob, prob = student_distribution(energy, support)
        return {'energy': energy, 'student_prob': prob, 'student_log_prob': log_prob}
