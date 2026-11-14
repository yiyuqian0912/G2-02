"""Pilot Student: local window features -> exactly normalised global source density on the 128^3 label grid.

Density (adapted from the 2026-11-10 study's HierDensity; same output family, smaller encoder for a 6 GB laptop GPU):
  P(c)   = V_c exp(l_c) / sum_c' V_c' exp(l_c')     16^3 coarse cells, V_c = legal volume fraction (cube minus window box)
  P(f|c) = v_cf exp(m_cf) / sum_f' v_cf' exp(m_cf')  8^3 fine cells per coarse cell, v_cf legal fraction
Cell masses P(c) P(f|c) on the 128^3 label cells sum to exactly 1 over the cube minus the closed window box; this is the
Student's support (W is observed; hidden solids are never used to restrict it).
Indexing: label cell = (ix * 128 + iy) * 128 + iz; coarse c = (ix//8)*256 + (iy//8)*16 + iz//8; fine f = (ix%8)*64 +
(iy%8)*8 + iz%8.
Inputs: uint8 [B, 4, 32, 32, 32] (XYZ; lit-FREE, dark-FREE, SOLID, boundary fractions), origin (voxels), side (voxels)."""
import math
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

RES, NC, NF, NFINE = 1024, 16, 8, 128
LOG_NCELL = 3 * math.log(NFINE)


def fourier(x, freqs):
    out = [torch.sin(math.pi * f * x) for f in freqs] + [torch.cos(math.pi * f * x) for f in freqs]
    return torch.cat(out, -1)


def coarse_centers(device):
    c = (torch.arange(NC, device=device, dtype=torch.float32) + 0.5) / NC
    x, y, z = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([x, y, z], -1).reshape(-1, 3)


def axis_overlap(origin, side, n):
    """(B, 3, n): fraction of each cell (per axis) covered by the window box."""
    w = RES // n
    lo = torch.arange(n, device=origin.device, dtype=torch.float32) * w
    o = origin.float()[:, :, None]
    hi = o + side.float()[:, None, None]
    return (torch.minimum(lo + w, hi) - torch.maximum(lo, o)).clamp(0, w) / w


def coarse_legal(origin, side):
    ov = axis_overlap(origin, side, NC)
    win = ov[:, 0, :, None, None] * ov[:, 1, None, :, None] * ov[:, 2, None, None, :]
    return (1 - win).reshape(origin.shape[0], -1).clamp_min(0)


def fine_legal(origin, side, b, c):
    """legal fractions of the 512 fine cells of coarse cells c of batch rows b: (U, 512)."""
    ov = axis_overlap(origin, side, NFINE)                       # (B, 3, 128)
    cx, cy, cz = c // 256, (c // 16) % 16, c % 16
    ar = torch.arange(NF, device=c.device)
    ox = torch.gather(ov[b, 0], 1, cx[:, None] * NF + ar)
    oy = torch.gather(ov[b, 1], 1, cy[:, None] * NF + ar)
    oz = torch.gather(ov[b, 2], 1, cz[:, None] * NF + ar)
    win = ox[:, :, None, None] * oy[:, None, :, None] * oz[:, None, None, :]
    v = (1 - win).reshape(len(b), NF ** 3).clamp_min(0)
    # coarse cells entirely inside the window box: P(c) = 0 already; keep the conditional finite (no NaN)
    return torch.where((v.sum(-1) > 0)[:, None], v, torch.ones_like(v))


def _log(x):
    return torch.log(x.clamp_min(1e-30)).masked_fill(x <= 0, float("-inf"))


class Res3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        return x + self.c2(F.gelu(self.n2(h)))


class Cross(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.o = nn.Linear(d, d)
        self.nm = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, t):
        B, Nq, d = x.shape
        q = self.q(self.nq(x)).view(B, Nq, self.h, -1).transpose(1, 2)
        k, v = self.kv(self.nk(t)).view(B, t.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, d)
        x = x + self.o(a)
        return x + self.mlp(self.nm(x))


class Student(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.stem = nn.Sequential(nn.Conv3d(4, 32, 4, stride=2, padding=1), nn.GELU(), Res3(32))   # 16^3
        self.down1 = nn.Sequential(nn.Conv3d(32, 64, 4, stride=2, padding=1), Res3(64))             # 8^3
        self.down2 = nn.Sequential(nn.Conv3d(64, d, 1), Res3(d))                                    # 8^3 tokens
        self.side_emb = nn.Embedding(3, d)
        self.tok_pos = nn.Linear(3 * 2 * 4 + 3, d)
        self.qmlp = nn.Sequential(nn.Linear(3 * 2 * 4 + 3 + 3 + 3 + 3 * 2 * 3 + 3 + 3, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Cross(d, heads) for _ in range(layers)])
        self.coarse_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.fine_norm = nn.LayerNorm(d); self.fine_head = nn.Linear(d, NF ** 3)
        for lin in (self.coarse_head[1], self.fine_head):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.register_buffer("centers", coarse_centers("cpu"), persistent=False)

    def forward(self, feats, origin, side):
        B = feats.shape[0]
        sidx = torch.bucketize(side, torch.tensor([96, 192], device=side.device))
        h = self.down2(self.down1(self.stem(feats.float() / 255.0)))              # (B, d, 8, 8, 8) XYZ
        tok = h.flatten(2).transpose(1, 2)                                         # (B, 512, d), x-major
        g = (torch.arange(8, device=feats.device, dtype=torch.float32) + 0.5) / 8
        gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
        rel = torch.stack([gx, gy, gz], -1).reshape(1, 512, 3)
        wpos = (origin.float()[:, None] + rel * side.float()[:, None, None]) / RES        # token centres, world coords
        tok = tok + self.tok_pos(torch.cat([fourier(wpos * 2 - 1, [1, 2, 4, 8]), wpos * 2 - 1], -1)) + self.side_emb(sidx)[:, None]
        # queries: coarse-cell centres relative to the window
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
        return self.coarse_head(q).squeeze(-1).float(), q

    def fine_logits(self, emb):
        return self.fine_head(self.fine_norm(emb.float())).float()     # fp32 head (called outside autocast)


def coarse_logp(cl, origin, side):
    return torch.log_softmax(cl + _log(coarse_legal(origin, side)), -1)


def target_ce(model, out, origin, side, tb, tcell, tq):
    """Per-example cross-entropy -sum_cells T(cell) log p(cell) for sparse targets (tb: batch row, tcell: label cell,
    tq: target mass; each row's tq sums to 1). Exact on the 128^3 grid. Returns (B,)."""
    cl, emb = out
    lpc = coarse_logp(cl, origin, side)
    ix, iy, iz = tcell // (NFINE * NFINE), (tcell // NFINE) % NFINE, tcell % NFINE
    c = (ix // NF) * 256 + (iy // NF) * 16 + iz // NF
    f = (ix % NF) * 64 + (iy % NF) * 8 + iz % NF
    key = tb * (NC ** 3) + c
    uk, inv = torch.unique(key, return_inverse=True)
    ub, uc = uk // (NC ** 3), uk % (NC ** 3)
    fl = model.fine_logits(emb[ub, uc])
    lpf = torch.log_softmax(fl + _log(fine_legal(origin, side, ub, uc)), -1)
    lp = lpc[tb, c] + lpf[inv, f]
    contrib = -tq * lp
    contrib = torch.where(tq > 0, contrib, torch.zeros_like(contrib))
    return torch.zeros(cl.shape[0], device=cl.device).index_add_(0, tb, contrib)


@torch.no_grad()
def full_logmass(model, out, origin, side):
    """log P(cell) on all 128^3 cells in label indexing order: (B, 128^3); -inf on cells fully inside the window box."""
    cl, emb = out
    B = cl.shape[0]
    lpc = coarse_logp(cl, origin, side)                          # (B, 4096)
    res = torch.empty(B, NC ** 3, NF ** 3, device=cl.device)
    allc = torch.arange(NC ** 3, device=cl.device)
    for b in range(B):
        bb = torch.full_like(allc, b)
        fl = model.fine_logits(emb[b])
        lpf = torch.log_softmax(fl + _log(fine_legal(origin, side, bb, allc)), -1)
        res[b] = torch.where(torch.isfinite(lpc[b])[:, None], lpc[b][:, None] + lpf, torch.full_like(lpf, float("-inf")))
    # (c, f) -> label cell order
    res = res.view(B, NC, NC, NC, NF, NF, NF).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, NFINE ** 3)
    return res
