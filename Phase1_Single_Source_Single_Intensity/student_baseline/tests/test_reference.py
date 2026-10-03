"""Independent transcription of PDF Section 5.4, used only for parity testing."""
import math
import pytest
import torch
from torch import nn
from model import SourceEnergyField
from train import teacher_student_loss


class ReferenceFourierXY(nn.Module):
    def forward(self, xy):
        out = [xy]
        for k in range(6):
            w = (2.0 ** k) * math.pi
            out += [torch.sin(w * xy), torch.cos(w * xy)]
        return torch.cat(out, dim=-1)


class ReferenceSourceEnergyField(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.GELU())
        self.xy = ReferenceFourierXY()
        self.source_query = nn.Sequential(nn.Linear(26, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.keys = nn.Linear(hidden, hidden)
        self.values = nn.Linear(hidden, hidden)
        self.energy_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, observation, candidates):
        feat = self.image_encoder(observation)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        k = self.keys(tokens)
        v = self.values(tokens)
        q = self.source_query(self.xy(candidates))
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(c)
        attn = torch.softmax(logits, dim=-1)
        context = torch.matmul(attn, v)
        return self.energy_head(torch.cat([q, context], -1)).squeeze(-1)


@pytest.mark.parametrize('hidden', [16, 128])
def test_pdf_reference_outputs_loss_and_gradients(hidden):
    torch.manual_seed(27)
    model = SourceEnergyField(hidden=hidden).double()
    reference = ReferenceSourceEnergyField(hidden=hidden).double()
    reference.load_state_dict({key.removeprefix('attention.'): value for key, value in model.state_dict().items()})
    observation = torch.randn(2, 1, 13, 17, dtype=torch.float64, requires_grad=True)
    candidates = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    ref_observation = observation.detach().clone().requires_grad_()
    ref_candidates = candidates.detach().clone().requires_grad_()
    energy = model(observation, candidates)
    ref_energy = reference(ref_observation, ref_candidates)
    torch.testing.assert_close(energy, ref_energy, rtol=1e-12, atol=1e-12)
    target = torch.rand(2, 7, dtype=torch.float64).softmax(-1)
    loss = teacher_student_loss(energy, target)
    ref_loss = -(target * torch.log_softmax(-ref_energy, dim=-1)).sum(-1).mean()
    torch.testing.assert_close(loss, ref_loss, rtol=1e-12, atol=1e-12)
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(observation.grad, ref_observation.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(candidates.grad, ref_candidates.grad, rtol=1e-10, atol=1e-12)
    ref_parameters = dict(reference.named_parameters())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, ref_parameters[name.removeprefix('attention.')].grad,
                                   rtol=1e-10, atol=1e-12, msg=name)
