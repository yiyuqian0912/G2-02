"""Flat, exactly normalised densities on the 128^3 label grid, held in the 16^3 (coarse c) x 8^3 (fine f) layout.

  log p(c, f) = logit(c, f) + log v(c, f) - logsumexp_{c', f'} [logit + log v]
v = legal volume fraction of the fine cell (the cube minus the closed window box: the window is observed). Every
model of the lab returns logits in this layout, so all models share one normaliser, one loss and one evaluator.
Cell centres are in world voxel units (N = 1024; label cell i spans [8 i, 8 i + 8))."""
import math
import torch
import torch.utils.checkpoint

RES, NC, NF, G = 1024, 16, 8, 128
SIGMA_CELL2 = 8.0 ** 2 / 12.0        # variance of a uniform 8-voxel cell projected on a unit normal


def axis_overlap(origin, side, n):
    w = RES // n
    lo = torch.arange(n, device=origin.device, dtype=torch.float32) * w
    o = origin.float()[:, :, None]; hi = o + side.float()[:, None, None]
    return (torch.minimum(lo + w, hi) - torch.maximum(lo, o)).clamp(0, w) / w


def to_cf(x):
    """(B, 128, 128, 128) [ix, iy, iz] -> (B, 4096, 512)."""
    B = x.shape[0]
    return x.view(B, NC, NF, NC, NF, NC, NF).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, NC ** 3, NF ** 3)


def legal_logv(origin, side):
    ov = axis_overlap(origin, side, G)
    win = ov[:, 0, :, None, None] * ov[:, 1, None, :, None] * ov[:, 2, None, None, :]
    v = (1 - win).clamp_min(0)
    return to_cf(torch.log(v.clamp_min(1e-30)).masked_fill(v <= 0, float("-inf")))


_CENTRES = {}


def cell_centres(device):
    """(4096, 512, 3) centres of the 128^3 cells in world voxels, (c, f) layout."""
    if device not in _CENTRES:
        g = (torch.arange(G, device=device, dtype=torch.float32) + 0.5) * (RES / G)
        x, y, z = torch.meshgrid(g, g, g, indexing="ij")
        C = torch.stack([x, y, z], -1)[None]                       # (1, 128, 128, 128, 3)
        _CENTRES[device] = torch.stack([to_cf(C[..., i]) [0] for i in range(3)], -1).contiguous()
    return _CENTRES[device]


def log_normalise(logits, logv):
    z = logits + logv
    return z - torch.logsumexp(z.flatten(1), 1)[:, None, None]


def sparse_ce(logp, tb, tc, tf, tq):
    """-sum T log p per example for sparse targets; (B,)."""
    lp = logp[tb, tc, tf]
    contrib = torch.where(tq > 0, -tq * lp, torch.zeros_like(lp))
    return torch.zeros(logp.shape[0], device=logp.device).index_add_(0, tb, contrib)


def _plane_term(C, mu, n, s02, sth2, leps, l1meps):
    """log(eps + (1 - eps) exp(-d^2 / (2 sigma^2))) for one facet slot of every batch row: (B, 4096, 512).
    mu, n: (B, 3); s02, sth2, leps, l1meps: (B,). sigma^2 = cell + s0^2 + (r sigma_theta)^2, r = |centre - mu|."""
    Cf = C.view(-1, 3)                                              # no (B, 4096, 512, 3) temporaries
    d = (torch.mm(n, Cf.t()) - (mu * n).sum(-1, keepdim=True)).view(-1, NC ** 3, NF ** 3)
    r2 = ((Cf * Cf).sum(-1)[None] - 2 * torch.mm(mu, Cf.t()) + (mu * mu).sum(-1, keepdim=True)).view(-1, NC ** 3, NF ** 3)
    s2 = SIGMA_CELL2 + s02[:, None, None] + r2.clamp_min(0) * sth2[:, None, None]
    return torch.logaddexp(leps[:, None, None], l1meps[:, None, None] - d * d / (2 * s2))


def plane_factor(mu, n, s0, sth, eps_logit, mask, checkpoint=True):
    """Sum over facet slots k of the robust plane factors. mu, n: (B, K, 3); s0 (voxels), sth (radians), eps_logit:
    (B, K); mask: (B, K) bool. Returns (B, 4096, 512); masked slots contribute 0."""
    with torch.autocast("cuda", enabled=False):
        return _plane_factor(mu.float(), n.float(), s0.float(), sth.float(), eps_logit.float(), mask, checkpoint)


def _plane_factor(mu, n, s0, sth, eps_logit, mask, checkpoint):
    C = cell_centres(mu.device)
    B, K, _ = mu.shape
    out = None
    leps = torch.nn.functional.logsigmoid(eps_logit); l1meps = torch.nn.functional.logsigmoid(-eps_logit)
    for k in range(K):
        if not bool(mask[:, k].any()):
            continue
        m = mask[:, k]
        # masked rows: eps -> 1 (log 1 = 0, log(1 - eps) -> -inf) gives exactly 0
        le = torch.where(m, leps[:, k], torch.zeros_like(leps[:, k]))
        l1 = torch.where(m, l1meps[:, k], torch.full_like(l1meps[:, k], -1e4))
        args = (C, mu[:, k].float(), n[:, k].float(), (s0[:, k] ** 2).float(), (sth[:, k] ** 2).float(), le.float(), l1.float())
        t = torch.utils.checkpoint.checkpoint(_plane_term, *args, use_reentrant=False) if checkpoint else _plane_term(*args)
        out = t if out is None else out + t
    if out is None:
        out = torch.zeros(B, NC ** 3, NF ** 3, device=mu.device)
    return out
