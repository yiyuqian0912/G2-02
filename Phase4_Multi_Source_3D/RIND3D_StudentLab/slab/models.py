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
from .plane_triton import plane_factor_fused, plane_factor2_fused, plane_factor_qs_fused

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True          # convolutions only


class Generic(PilotStudent):
    """Pilot architecture, flat readout. forward -> (logits (B, 4096, 512), context dict).
    facet_tokens=True (arm G1): the X-measurable facets enter as K extra key/value tokens of every cross-attention
    block (embedding of centroid, normal, size and fit quality), but the decoder has no geometric operator: the same
    information as the facet model, generic decoding.
    side_bucket_max (evaluation-time diagnostic for the side-holdout arms, §8 of B_RESULTS): clamp the window-size
    bucket (0: L64, 1: L128, 2: L256) so a side never seen in training reuses the largest trained bucket's embedding."""

    def __init__(self, d=128, layers=3, heads=4, ckpt=True, blind=False, facet_tokens=False, min_inliers=24,
                 side_bucket_max=None):
        super().__init__(d=d, layers=layers, heads=heads, ckpt=ckpt)
        self.blind = blind
        self.side_bucket_max = side_bucket_max
        self.facet_tokens = facet_tokens; self.min_inliers = min_inliers
        if facet_tokens:
            self.ftok = nn.Sequential(nn.Linear(16 + 3 * 2 * 4 + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
            self.fnull = nn.Parameter(torch.zeros(1, 1, d))

    def encode(self, feats, origin, side):
        B = feats.shape[0]
        x = feats.float() / 255.0
        if self.blind:
            x = torch.zeros_like(x)
        sidx = torch.bucketize(side, torch.tensor([96, 192], device=side.device))
        if self.side_bucket_max is not None:
            sidx = sidx.clamp_max(self.side_bucket_max)
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

    def facet_token_set(self, B, tok):
        Fc = B["facets"].float()
        mask = (Fc[..., 6] >= self.min_inliers) & (torch.arange(Fc.shape[1], device=Fc.device)[None] < B["nf"][:, None])
        desc, _ = FacetHead.descriptors(Fc, B["origin"], B["side"])
        wpos = Fc[..., :3] / RES
        nrm = Fc[..., 3:6] * torch.sign(Fc[..., 5:6] + 1e-9)
        # plane offset n.mu / RES: with the normal, the plane in world coordinates
        e = self.ftok(torch.cat([desc, fourier(wpos * 2 - 1, [1, 2, 4, 8]), nrm, ((nrm * wpos).sum(-1, keepdim=True))], -1))
        e = torch.where(mask[..., None], e, self.fnull.expand_as(e))   # absent facets: a learned null token
        return torch.cat([tok, e.to(tok.dtype)], 1)

    def forward(self, B):
        tok, h, sidx = self.encode(B["feats"], B["origin"], B["side"])
        if "fflag" in B and getattr(self, "flag_emb", None) is not None:
            tok = tok + self.flag_emb(B["fflag"].long())[:, None].to(tok.dtype)   # X-measurable: facets present?
        kv = self.facet_token_set(B, tok) if self.facet_tokens else tok
        q = self.decode(kv, B["origin"], B["side"], sidx)
        return self.readout(q), dict(tok=tok, h=h, sidx=sidx)


def consensus(Fc, mask, origin, side, rho=1e-6):
    """Leave-one-out consistency of each facet with the others (X only). For facet k, x_k = argmin_x
    sum_{j != k} w_j (n_j . x - b_j)^2 / sig_j^2 + rho |x - c|^2 (c = window centre, w_j = valid), the residual
    r_k = |n_k . x_k - b_k| (voxels) and the variance v_k = n_k^T M_k^{-1} n_k of n_k . x under the others' constraints.
    Returns (B, K, 3): log1p(r), log(v + 1), z = r / sqrt(v + 4) clipped to 50. With fewer than 3 other facets v is huge
    and z ~ 0 (no evidence)."""
    with torch.autocast("cuda", enabled=False):
        n = Fc[..., 3:6].double(); b = (n * Fc[..., :3].double()).sum(-1)
        c = (origin.double() + side.double()[:, None] / 2)
        sig2 = (1.0 + (Fc[..., 7].double().clamp_min(0.2) / Fc[..., 8].double().clamp_min(1.0)) ** 2 * 700.0 ** 2)
        w = mask.double() / sig2                                                     # (B, K)
        nn_ = n[..., :, None] * n[..., None, :]                                     # (B, K, 3, 3)
        Mall = (w[..., None, None] * nn_).sum(1); rall = (w[..., None] * n * b[..., None]).sum(1)
        I = torch.eye(3, device=Fc.device, dtype=torch.float64)
        M = Mall[:, None] - w[..., None, None] * nn_ + rho * I                       # leave one out
        r = rall[:, None] - w[..., None] * n * b[..., None] + rho * c[:, None]
        x = torch.linalg.solve(M, r[..., None])[..., 0]                              # (B, K, 3)
        res = ((n * x).sum(-1) - b).abs()
        v = (n[..., None, :] @ torch.linalg.solve(M, n[..., :, None]))[..., 0, 0]
        z = (res / (v + 4.0).sqrt()).clamp(max=50.0)
        out = torch.stack([torch.log1p(res.clamp(max=1e6)), torch.log1p(v.clamp(max=1e12)), z], -1)
        return torch.where(mask[..., None], out, torch.zeros_like(out)).float()


class FacetHead(nn.Module):
    """Per-facet reliability and width from X-measurable descriptors + context."""

    def __init__(self, d=128, dh=96, init_eps=1e-3, init_s0=0.5, init_kappa=1.0, two_hyp=False, cons=False):
        super().__init__()
        self.two_hyp, self.cons = two_hyp, cons
        self.inp = nn.Sequential(nn.Linear(16 + int(two_hyp) + 3 * int(cons) + d, dh), nn.GELU(), nn.Linear(dh, dh))
        self.att = nn.MultiheadAttention(dh, 4, batch_first=True)
        self.ctx_tok = nn.Linear(d, dh)
        self.n1 = nn.LayerNorm(dh); self.n2 = nn.LayerNorm(dh)
        self.mlp = nn.Sequential(nn.Linear(dh, 2 * dh), nn.GELU(), nn.Linear(2 * dh, dh))
        self.out = nn.Linear(dh, 5 if two_hyp else 3)
        nn.init.zeros_(self.out.weight)
        b0 = [math.log(init_eps / (1 - init_eps)), math.log(init_s0), math.log(init_kappa)]
        if two_hyp:
            b0 += [3.0, math.log(0.02)]              # P(shadow facet) = 0.95; occluder half-space density ratio 0.02
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor(b0))

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
        if self.two_hyp:
            desc = torch.cat([desc, Fc[..., 11:12].abs()], -1)          # consistency of the lit side
        if self.cons:
            desc = torch.cat([desc, consensus(Fc, mask, origin, side)], -1)  # agreement with the other facets
        x = self.inp(torch.cat([desc, ctx[:, None].expand(-1, desc.shape[1], -1)], -1))
        # keys: one always-present context token + the facets (no fully masked rows: no NaN in softmax or its gradient)
        kv = torch.cat([self.ctx_tok(ctx)[:, None], self.n1(x)], 1)
        kpm = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        a, _ = self.att(self.n1(x), kv, kv, key_padding_mask=kpm)
        x = x + a
        x = x + self.mlp(self.n2(x))
        o = self.out(x)
        if self.two_hyp:
            return o[..., 0], torch.exp(o[..., 1]), torch.exp(o[..., 2]) * sth_fit, o[..., 3], o[..., 4]
        return o[..., 0], torch.exp(o[..., 1]), torch.exp(o[..., 2]) * sth_fit


class Facet(Generic):
    """two_hyp (F3): each facet is a mixture of a shadow plane through the source and an occluder face (the source in
    the lit half-space; facets table v2 gives the lit side), plus the outlier floor. drop: probability that all facets of
    a training example are hidden (facet dropout); flag: the generic field is told whether facets are present."""

    def __init__(self, min_inliers=24, facet_only=False, fixed=False, two_hyp=False, drop=0.0, flag=False, cons=False, quad=1,
                 amp=True, **kw):
        super().__init__(**kw)
        self.quad = quad                   # F5: M^3 sparse sub-cell quadrature of the facet product (1 = cell-centre factor)
        self.amp = amp                     # F5: amplitude-consistent cell average (slab.plane_triton, sparse section)
        self.min_inliers = min_inliers
        self.facet_only = facet_only       # ablation: uniform-legal x facet factor (no generic field)
        self.fixed = fixed                 # ablation: hand-set eps / s0 / kappa (no learned facet head)
        self.two_hyp, self.drop = two_hyp, drop
        self.flag_emb = nn.Embedding(2, self.coarse_head[1].in_features) if flag else None
        if flag:
            nn.init.zeros_(self.flag_emb.weight)
        self.fhead = FacetHead(d=self.coarse_head[1].in_features, two_hyp=two_hyp, cons=cons)

    def facet_mask(self, B):
        Fc = B["facets"].float()
        mask = (Fc[..., 6] >= self.min_inliers) & (torch.arange(Fc.shape[1], device=Fc.device)[None] < B["nf"][:, None])
        if self.training and self.drop > 0:
            mask = mask & (torch.rand(mask.shape[0], 1, device=mask.device) >= self.drop)
        return Fc, mask

    def facet_params(self, B, ctx):
        Fc, mask = B["_fmask"]
        if self.fixed:
            sth_fit = Fc[..., 7].clamp_min(0.2) / (Fc[..., 8].clamp_min(1.0) * Fc[..., 6].clamp_min(1).sqrt())
            e = torch.full_like(Fc[..., 6], math.log(1e-3 / (1 - 1e-3)))
            return Fc, mask, e, torch.full_like(e, 0.5), sth_fit
        with torch.autocast("cuda", enabled=False):
            return (Fc, mask) + tuple(self.fhead(Fc, mask, B["origin"], B["side"], ctx["tok"].float().mean(1)))

    def forward(self, B):
        B = dict(B); B["_fmask"] = self.facet_mask(B)
        if self.flag_emb is not None:
            B["fflag"] = B["_fmask"][1].any(-1)
        logits, ctx = super().forward(B)
        if self.facet_only:
            logits = logits * 0
        Fc, mask, e, s0, sth, *extra = self.facet_params(B, ctx)
        if self.two_hyp:
            pi, lrho = extra
            sgn = torch.where(Fc[..., 11:12] < 0, -1.0, 1.0)
            pf = plane_factor2_fused(Fc[..., :3], Fc[..., 3:6] * sgn, s0, sth, e, pi, lrho, mask)
            ctx.update(pi=pi)
        elif self.quad > 1:
            pf = plane_factor_qs_fused(Fc[..., :3], Fc[..., 3:6], s0, sth, e, mask, M=self.quad, amp=self.amp)
        else:
            pf = (plane_factor_fused if Fc.is_cuda else plane_factor)(Fc[..., :3], Fc[..., 3:6], s0, sth, e, mask)
        ctx.update(eps_logit=e, s0=s0, sth=sth, fmask=mask)
        return logits + pf, ctx


def build(cfg):
    kind = cfg.get("model", "generic")
    kw = dict(d=cfg.get("d", 128), layers=cfg.get("layers", 3), heads=cfg.get("heads", 4), ckpt=cfg.get("ckpt", True),
              side_bucket_max=cfg.get("side_bucket_max"))
    if kind == "generic":
        return Generic(blind=cfg.get("blind", False), facet_tokens=cfg.get("facet_tokens", False),
                       min_inliers=cfg.get("min_inliers", 24), **kw)
    if kind == "facet":
        return Facet(min_inliers=cfg.get("min_inliers", 24), facet_only=cfg.get("facet_only", False),
                     fixed=cfg.get("fixed", False), two_hyp=cfg.get("two_hyp", False), drop=cfg.get("drop", 0.0),
                     flag=cfg.get("flag", False), cons=cfg.get("cons", False), quad=cfg.get("quad", 1),
                     amp=cfg.get("amp", True), **kw)
    if kind == "learned_facet":
        return LearnedFacet(K=cfg.get("slots", 8), min_mass=cfg.get("min_mass", 2.0), planar_w=cfg.get("planar_w", 0.1),
                            min_inliers=0, **kw)
    raise ValueError(kind)


class SlotPlanes(nn.Module):
    """Learned facet extraction from the POOLED input (arm F2): slot attention over the pooled boundary cells assigns
    cells to K planar facets; each facet plane is the weighted PCA of its cells' centres (smallest eigenvector, refined by
    differentiable shifted power iterations). Self-supervised planarity loss (X only): the assigned-mass-weighted
    thickness of the facets. Returns facets in the 12-float format of the facet table (no RANSAC, no full-resolution
    crop), so the same FacetHead and plane factor are used as in F1."""

    def __init__(self, K=8, dc=48, iters=3, ncell=3072):
        super().__init__()
        self.K, self.dc, self.iters, self.ncell = K, dc, iters, ncell
        self.conv = nn.Sequential(nn.Conv3d(4, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, dc, 3, padding=1))
        self.pos = nn.Linear(3 * 2 * 4, dc)
        self.slots_mu = nn.Parameter(torch.randn(1, K, dc) * dc ** -0.5)
        self.slots_ls = nn.Parameter(torch.full((1, 1, dc), -1.0))
        self.n_in = nn.LayerNorm(dc); self.n_s = nn.LayerNorm(dc); self.n_m = nn.LayerNorm(dc)
        self.q = nn.Linear(dc, dc, bias=False); self.k = nn.Linear(dc, dc, bias=False); self.v = nn.Linear(dc, dc, bias=False)
        self.gru = nn.GRUCell(dc, dc); self.mlp = nn.Sequential(nn.Linear(dc, 2 * dc), nn.GELU(), nn.Linear(2 * dc, dc))

    def forward(self, feats, origin, side):
        with torch.autocast("cuda", enabled=False):
            return self._forward(feats, origin, side)

    def _forward(self, feats, origin, side):
        B = feats.shape[0]; K, dc = self.K, self.dc
        x = feats.float() / 255.0
        bf = x[:, 3].flatten(1)                                         # boundary fraction, (B, 32768), XYZ order
        val, idx = bf.topk(self.ncell, dim=1)
        valid = val > 0
        e = self.conv(x).flatten(2).transpose(1, 2)                     # (B, 32768, dc)
        E = e.gather(1, idx[..., None].expand(-1, -1, dc))
        ix, iy, iz = idx // 1024, (idx // 32) % 32, idx % 32
        rel = (torch.stack([ix, iy, iz], -1).float() + 0.5) / 32
        X = origin[:, None].float() + rel * side[:, None, None].float()  # cell centres, world voxels
        E = E + self.pos(fourier(rel * 2 - 1, [1, 2, 4, 8]))
        slots = self.slots_mu.expand(B, -1, -1)
        if self.training:
            slots = slots + self.slots_ls.exp() * torch.randn_like(slots)
        inp = self.n_in(E); k = self.k(inp); v = self.v(inp)
        for _ in range(self.iters):
            q = self.q(self.n_s(slots))
            attn = torch.softmax(k @ q.transpose(1, 2) / dc ** 0.5, -1) * valid[..., None]   # (B, N, K), slots compete
            w = attn / (attn.sum(1, keepdim=True) + 1e-8)
            slots = self.gru((w.transpose(1, 2) @ v).reshape(-1, dc), slots.reshape(-1, dc)).view(B, K, dc)
            slots = slots + self.mlp(self.n_m(slots))
        W = (attn * val[..., None]).transpose(1, 2)                      # (B, K, N) mass weights
        mass = W.sum(-1).clamp_min(1e-6)
        mu = (W @ X) / mass[..., None]
        D = X[:, None] - mu[:, :, None]                                  # (B, K, N, 3)
        C = torch.einsum("bkn,bkni,bknj->bkij", W, D, D) / mass[..., None, None]
        Cd = C.double()
        ev, V = torch.linalg.eigh(Cd.detach() + 1e-9 * torch.eye(3, device=C.device, dtype=torch.float64))
        nv = V[..., 0]
        M = ev[..., 2, None, None] * torch.eye(3, device=C.device, dtype=torch.float64) - Cd
        for _ in range(3):                                               # differentiable refinement of the eigenvector
            nv = F.normalize((M @ nv[..., None])[..., 0], dim=-1)
        n = nv.float()
        lam0 = (n[..., None, :] @ C @ n[..., :, None])[..., 0, 0].clamp_min(0)
        cell = (side.float() / 32)[:, None]
        Fc = torch.zeros(B, K, 12, device=feats.device)
        Fc[..., :3] = mu; Fc[..., 3:6] = n
        Fc[..., 6] = mass * cell ** 2                                    # ~ boundary voxels on the facet
        Fc[..., 7] = lam0.sqrt(); Fc[..., 8] = ev[..., 1].float().clamp_min(0).sqrt(); Fc[..., 9] = ev[..., 2].float().clamp_min(0).sqrt()
        planar = (mass * lam0 / cell ** 2).sum(-1) / mass.sum(-1)        # thickness in pooled cells^2, mass-weighted
        return Fc, mass, planar


class LearnedFacet(Facet):
    """F2: the facet model with SlotPlanes in place of the X-measured RANSAC facets (pooled input only)."""

    def __init__(self, K=8, min_mass=2.0, planar_w=0.1, **kw):
        super().__init__(**kw)
        self.slot = SlotPlanes(K=K); self.min_mass = min_mass; self.planar_w = planar_w
        with torch.no_grad():                                            # learned planes start less trusted
            self.fhead.out.bias[0] = math.log(0.05 / 0.95)

    def forward(self, B):
        Fc, mass, planar = self.slot(B["feats"], B["origin"], B["side"])
        B = dict(B); B["facets"] = Fc; B["nf"] = torch.full_like(B["side"], Fc.shape[1])
        logits, ctx = Generic.forward(self, B)
        mask = mass >= self.min_mass
        with torch.autocast("cuda", enabled=False):
            e, s0, sth = self.fhead(Fc, mask, B["origin"], B["side"], ctx["tok"].float().mean(1))
        pf = plane_factor_fused(Fc[..., :3], Fc[..., 3:6], s0, sth, e, mask)
        ctx.update(eps_logit=e, s0=s0, sth=sth, fmask=mask, aux_loss=self.planar_w * planar)
        return logits + pf, ctx
