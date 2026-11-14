"""Lab Student models. Every model maps X = (pooled window features, origin, side[, X-measurable facets]) to flat logits
in the (4096, 512) layout of slab.density; the normaliser, loss and evaluator are shared.

generic   the pilot encoder/decoder (RIND3D_StudentResearch/pilot/student.py: 3D CNN -> 8^3 tokens, 4096 coarse-cell
          queries, cross-attention, coarse + fine heads), read out as one flat density. With blind=True the window
          content is zeroed: a window-aware spatial prior with identical capacity (origin and side only).
facet     generic + a projective factor: for each X-measurable interface facet k (centroid mu_k, normal n_k; slab/facets),
          log(eps_k + (1 - eps_k) exp(-d_k(s)^2 / 2 sigma_k(s)^2)), d_k = n_k . (s - mu_k), sigma_k^2 = cell^2 + s0_k^2 +
          (|s - mu_k| sigma_theta_k)^2. eps_k, s0_k, sigma_theta_k are predicted per facet from its descriptors, the other
          facets (one self-attention layer) and the encoder context. The plane itself is never learned: it is computed
          from X. Facet slots with fewer than min_inliers points are masked.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import PILOT
import sys
sys.path.insert(0, str(PILOT))
from student import Student as PilotStudent, fourier, coarse_centers, RES   # noqa: E402

from .density import legal_logv, plane_factor, NC, NF

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True          # convolutions only


class Generic(PilotStudent):
    """Pilot architecture, flat readout. forward -> (logits (B, 4096, 512), context dict)."""

    def __init__(self, d=128, layers=3, heads=4, ckpt=True, blind=False):
        super().__init__(d=d, layers=layers, heads=heads, ckpt=ckpt)
        self.blind = blind

    def encode(self, feats, origin, side):
        B = feats.shape[0]
        x = feats.float() / 255.0
        if self.blind:
            x = torch.zeros_like(x)
        sidx = torch.bucketize(side, torch.tensor([96, 192], device=side.device))
        h = self.down2(self.down1(self.stem(x)))
        tok = h.flatten(2).transpose(1, 2)
        g = (torch.arange(8, device=feats.device, dtype=torch.float32) + 0.5) / 8
        gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
        rel = torch.stack([gx, gy, gz], -1).reshape(1, 512, 3)
        wpos = (origin.float()[:, None] + rel * side.float()[:, None, None]) / RES
        tok = tok + self.tok_pos(torch.cat([fourier(wpos * 2 - 1, [1, 2, 4, 8]), wpos * 2 - 1], -1)) + self.side_emb(sidx)[:, None]
        return tok, h, sidx

    def decode(self, tok, origin, side, sidx):
        B = tok.shape[0]
        src = self.centers[None].expand(B, -1, -1)
        half = (side.float() / 2 / RES)[:, None, None]
        centre = (origin.float() + side.float()[:, None] / 2) / RES
        qrel = (src - centre[:, None]) / half
        r = qrel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        u = qrel / r
        inbox = qrel.abs().amax(-1, keepdim=True).clamp_min(1e-2)
        qf = torch.cat([fourier(u, [1, 2, 4, 8]), u, torch.log(r), torch.log(inbox), (1 / r).clamp_max(10),
                        torch.tanh(qrel / 4), fourier(src * 2 - 1, [1, 2, 4]), src * 2 - 1,
                        F.one_hot(sidx, 3).float()[:, None].expand(-1, src.shape[1], -1)], -1)
        q = self.qmlp(qf)
        for b in self.blocks:
            if self.training and self.ckpt:
                q = torch.utils.checkpoint.checkpoint(b, q, tok, use_reentrant=False)
            else:
                q = b(q, tok)
        return q

    def readout(self, q):
        with torch.autocast("cuda", enabled=False):
            q = q.float()
            cl = self.coarse_head(q).squeeze(-1)
            fl = self.fine_head(self.fine_norm(q))
        return cl[:, :, None] + fl                                   # (B, 4096, 512)

    def forward(self, B):
        tok, h, sidx = self.encode(B["feats"], B["origin"], B["side"])
        q = self.decode(tok, B["origin"], B["side"], sidx)
        return self.readout(q), dict(tok=tok, h=h, sidx=sidx)


class FacetHead(nn.Module):
    """Per-facet reliability and width from X-measurable descriptors + context."""

    def __init__(self, d=128, dh=96, init_eps=1e-3, init_s0=0.5, init_kappa=1.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(16 + d, dh), nn.GELU(), nn.Linear(dh, dh))
        self.att = nn.MultiheadAttention(dh, 4, batch_first=True)
        self.n1 = nn.LayerNorm(dh); self.n2 = nn.LayerNorm(dh)
        self.mlp = nn.Sequential(nn.Linear(dh, 2 * dh), nn.GELU(), nn.Linear(2 * dh, dh))
        self.out = nn.Linear(dh, 3)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([math.log(init_eps / (1 - init_eps)), math.log(init_s0), math.log(init_kappa)]))

    @staticmethod
    def descriptors(Fc, origin, side):
        """Fc: (B, K, 12) [mu(3), n(3), n_inl, rms, sd1, sd2, remaining, 0]."""
        mu, n = Fc[..., :3], Fc[..., 3:6]
        ninl, rms, sd1, sd2, rem = Fc[..., 6], Fc[..., 7], Fc[..., 8], Fc[..., 9], Fc[..., 10]
        L = side.float()[:, None]
        rel = (mu - origin.float()[:, None]) / L[..., None] - 0.5
        sth_fit = rms.clamp_min(0.2) / (sd1.clamp_min(1.0) * ninl.clamp_min(1).sqrt())
        nn_ = n * torch.sign(n[..., 2:3] + 1e-9)                     # sign-invariant normal
        return torch.cat([torch.log1p(ninl)[..., None], rms[..., None], torch.log1p(sd1)[..., None], torch.log1p(sd2)[..., None],
                          torch.log1p(rem)[..., None], nn_.abs(), nn_ * nn_.abs(), rel, torch.log(L / 64.0)[..., None].expand_as(rms[..., None]),
                          torch.log(sth_fit)[..., None]], -1), sth_fit

    def forward(self, Fc, mask, origin, side, ctx):
        desc, sth_fit = self.descriptors(Fc, origin, side)
        x = self.inp(torch.cat([desc, ctx[:, None].expand(-1, desc.shape[1], -1)], -1))
        a, _ = self.att(self.n1(x), self.n1(x), self.n1(x), key_padding_mask=~mask)
        x = x + torch.nan_to_num(a)
        x = x + self.mlp(self.n2(x))
        o = self.out(x)
        return o[..., 0], torch.exp(o[..., 1]), torch.exp(o[..., 2]) * sth_fit


class Facet(Generic):
    def __init__(self, min_inliers=24, facet_only=False, fixed=False, **kw):
        super().__init__(**kw)
        self.min_inliers = min_inliers
        self.facet_only = facet_only       # ablation: uniform-legal x facet factor (no generic field)
        self.fixed = fixed                 # ablation: hand-set eps / s0 / kappa (no learned facet head)
        self.fhead = FacetHead(d=self.coarse_head[1].in_features)

    def facet_params(self, B, ctx):
        Fc = B["facets"].float()
        mask = (Fc[..., 6] >= self.min_inliers) & (torch.arange(Fc.shape[1], device=Fc.device)[None] < B["nf"][:, None])
        if self.fixed:
            sth_fit = Fc[..., 7].clamp_min(0.2) / (Fc[..., 8].clamp_min(1.0) * Fc[..., 6].clamp_min(1).sqrt())
            e = torch.full_like(Fc[..., 6], math.log(1e-3 / (1 - 1e-3)))
            return Fc, mask, e, torch.full_like(e, 0.5), sth_fit
        with torch.autocast("cuda", enabled=False):
            e, s0, sth = self.fhead(Fc, mask, B["origin"], B["side"], ctx["tok"].float().mean(1))
        return Fc, mask, e, s0, sth

    def forward(self, B):
        logits, ctx = super().forward(B)
        if self.facet_only:
            logits = logits * 0
        Fc, mask, e, s0, sth = self.facet_params(B, ctx)
        pf = plane_factor(Fc[..., :3], Fc[..., 3:6], s0, sth, e, mask)
        ctx.update(eps_logit=e, s0=s0, sth=sth, fmask=mask)
        return logits + pf, ctx


def build(cfg):
    kind = cfg.get("model", "generic")
    kw = dict(d=cfg.get("d", 128), layers=cfg.get("layers", 3), heads=cfg.get("heads", 4), ckpt=cfg.get("ckpt", True))
    if kind == "generic":
        return Generic(blind=cfg.get("blind", False), **kw)
    if kind == "facet":
        return Facet(min_inliers=cfg.get("min_inliers", 24), facet_only=cfg.get("facet_only", False),
                     fixed=cfg.get("fixed", False), **kw)
    raise ValueError(kind)
