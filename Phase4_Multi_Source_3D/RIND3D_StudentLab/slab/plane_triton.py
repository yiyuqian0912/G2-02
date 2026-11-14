"""Fused Triton kernels for the robust plane factor of slab.density (forward and analytic backward).

phi(s) = sum_k logaddexp(LE_k, L1_k - u_k(s)),  u = d^2 / (2 sigma^2),  d = n.(C - mu),  r2 = |C - mu|^2,
sigma^2 = c0 + S0 + r2 * STH.  Parameters per (b, k): P = [mu(3), n(3), S0, STH, LE, L1] (float32).
Cell centres C are generated from the (c, f) index (no memory reads). The backward reduces, per block of cells, the
gradients of the 10 parameters; partial sums (B, n_blocks, K, 10) are summed in torch (deterministic)."""
import torch
import triton
import triton.language as tl

from .density import SIGMA_CELL2, NC, NF

BLOCK = 1024
NCELL = NC ** 3 * NF ** 3


@triton.jit
def _centre(j):
    c = j // 512; f = j % 512
    cx = c // 256; cy = (c // 16) % 16; cz = c % 16
    fx = f // 64; fy = (f // 8) % 8; fz = f % 8
    x = ((cx * 8 + fx).to(tl.float32) + 0.5) * 8.0
    y = ((cy * 8 + fy).to(tl.float32) + 0.5) * 8.0
    z = ((cz * 8 + fz).to(tl.float32) + 0.5) * 8.0
    return x, y, z


@triton.jit
def _fwd(P, OUT, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 10
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        a = l1 - d * d / (2.0 * s2)
        m = tl.maximum(le, a)
        acc += m + tl.log(1.0 + tl.exp(-tl.abs(le - a)))
    tl.store(OUT + b * NCELL + j, acc)


@triton.jit
def _bwd(P, G, PART, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr, NBLK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    g = tl.load(G + b * NCELL + j)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 10
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        u = d * d / (2.0 * s2)
        a = l1 - u
        w = 1.0 / (1.0 + tl.exp(le - a))                  # d phi / d a
        gw = g * w
        ds = d / s2; us = u / s2
        o = PART + ((b * NBLK + blk) * K + k) * 10
        # mu_j: w (d/s2) n_j - 2 w (u/s2) STH D_j
        tl.store(o + 0, tl.sum(gw * (ds * nx - 2.0 * us * sth * dx), 0))
        tl.store(o + 1, tl.sum(gw * (ds * ny - 2.0 * us * sth * dy), 0))
        tl.store(o + 2, tl.sum(gw * (ds * nz - 2.0 * us * sth * dz), 0))
        # n_j: -w (d/s2) D_j
        tl.store(o + 3, tl.sum(-gw * ds * dx, 0))
        tl.store(o + 4, tl.sum(-gw * ds * dy, 0))
        tl.store(o + 5, tl.sum(-gw * ds * dz, 0))
        tl.store(o + 6, tl.sum(gw * us, 0))               # S0
        tl.store(o + 7, tl.sum(gw * us * r2, 0))          # STH
        tl.store(o + 8, tl.sum(g * (1.0 - w), 0))         # LE
        tl.store(o + 9, tl.sum(gw, 0))                    # L1


class PlaneFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        _fwd[(B, NCELL // BLOCK)](P, out, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK)
        ctx.save_for_backward(P)
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        (P,) = ctx.saved_tensors; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.empty(B, nblk, K, 10, device=P.device, dtype=torch.float32)
        _bwd[(B, nblk)](P, g.contiguous().float(), part, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK, NBLK=nblk)
        return part.sum(1)


def plane_factor_fused(mu, n, s0, sth, eps_logit, mask):
    """Same contract as slab.density.plane_factor (fp32, any autocast state)."""
    with torch.autocast("cuda", enabled=False):
        le = torch.nn.functional.logsigmoid(eps_logit.float()); l1 = torch.nn.functional.logsigmoid(-eps_logit.float())
        le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
        P = torch.cat([mu.float(), n.float(), (s0.float() ** 2)[..., None], (sth.float() ** 2)[..., None], le[..., None], l1[..., None]], -1)
        return PlaneFactorFn.apply(P)


# ---------------------------------------------------------------- two-hypothesis facets (shadow plane | occluder face)
# phi(s) = logaddexp(LE, L1 + logaddexp(LPI - u, LQ + logsigmoid(d * TI))), d = n.(C - mu) with n oriented to the lit side.
# Parameters per (b, k): [mu(3), n(3), S0, STH, LE, L1, LPI, LQ, TI] (13); gradients for the first 12 (TI fixed).
@triton.jit
def _logsig(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _lae(a, b):
    m = tl.maximum(a, b)
    return m + tl.log(1.0 + tl.exp(-tl.abs(a - b)))


@triton.jit
def _fwd2(P, OUT, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 13
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        lpi = tl.load(base + 10); lq = tl.load(base + 11); ti = tl.load(base + 12)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        inner = _lae(lpi - d * d / (2.0 * s2), lq + _logsig(d * ti))
        acc += _lae(le, l1 + inner)
    tl.store(OUT + b * NCELL + j, acc)


@triton.jit
def _bwd2(P, G, PART, K: tl.constexpr, NCELL: tl.constexpr, C0: tl.constexpr, BLOCK: tl.constexpr, NBLK: tl.constexpr):
    b = tl.program_id(0); blk = tl.program_id(1)
    j = blk * BLOCK + tl.arange(0, BLOCK)
    x, y, z = _centre(j)
    g = tl.load(G + b * NCELL + j)
    for k in tl.static_range(K):
        base = P + (b * K + k) * 13
        mx = tl.load(base + 0); my = tl.load(base + 1); mz = tl.load(base + 2)
        nx = tl.load(base + 3); ny = tl.load(base + 4); nz = tl.load(base + 5)
        s0 = tl.load(base + 6); sth = tl.load(base + 7); le = tl.load(base + 8); l1 = tl.load(base + 9)
        lpi = tl.load(base + 10); lq = tl.load(base + 11); ti = tl.load(base + 12)
        dx = x - mx; dy = y - my; dz = z - mz
        d = nx * dx + ny * dy + nz * dz
        r2 = dx * dx + dy * dy + dz * dz
        s2 = C0 + s0 + r2 * sth
        u = d * d / (2.0 * s2)
        a1 = lpi - u; a2 = lq + _logsig(d * ti)
        inner = _lae(a1, a2)
        w = 1.0 / (1.0 + tl.exp(le - l1 - inner))            # non-outlier responsibility
        r1 = 1.0 / (1.0 + tl.exp(a2 - a1))                   # shadow-plane share
        sm = 1.0 / (1.0 + tl.exp(d * ti))                    # sigmoid(-d ti)
        gd = g * w * (-r1 * d / s2 + (1.0 - r1) * ti * sm)   # d phi / d d
        gs = g * w * r1 * u / s2                             # d phi / d s2
        o = PART + ((b * NBLK + blk) * K + k) * 12
        tl.store(o + 0, tl.sum(-gd * nx - 2.0 * gs * sth * dx, 0))
        tl.store(o + 1, tl.sum(-gd * ny - 2.0 * gs * sth * dy, 0))
        tl.store(o + 2, tl.sum(-gd * nz - 2.0 * gs * sth * dz, 0))
        tl.store(o + 3, tl.sum(gd * dx, 0))
        tl.store(o + 4, tl.sum(gd * dy, 0))
        tl.store(o + 5, tl.sum(gd * dz, 0))
        tl.store(o + 6, tl.sum(gs, 0))
        tl.store(o + 7, tl.sum(gs * r2, 0))
        tl.store(o + 8, tl.sum(g * (1.0 - w), 0))
        tl.store(o + 9, tl.sum(g * w, 0))
        tl.store(o + 10, tl.sum(g * w * r1, 0))
        tl.store(o + 11, tl.sum(g * w * (1.0 - r1), 0))


class PlaneFactor2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P):
        P = P.contiguous().float(); B, K, _ = P.shape
        out = torch.empty(B, NCELL, device=P.device, dtype=torch.float32)
        _fwd2[(B, NCELL // BLOCK)](P, out, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK)
        ctx.save_for_backward(P)
        return out.view(B, NC ** 3, NF ** 3)

    @staticmethod
    def backward(ctx, g):
        (P,) = ctx.saved_tensors; B, K, _ = P.shape; nblk = NCELL // BLOCK
        part = torch.empty(B, nblk, K, 12, device=P.device, dtype=torch.float32)
        _bwd2[(B, nblk)](P, g.contiguous().float(), part, K=K, NCELL=NCELL, C0=SIGMA_CELL2, BLOCK=BLOCK, NBLK=nblk)
        gp = part.sum(1)
        return torch.cat([gp, torch.zeros_like(gp[..., :1])], -1)


def two_hyp_params(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau=4.0):
    """Parameter rows for PlaneFactor2Fn. n_lit: unit normals oriented to the lit side. Masked slots contribute 0."""
    le = torch.nn.functional.logsigmoid(eps_logit); l1 = torch.nn.functional.logsigmoid(-eps_logit)
    lpi = torch.nn.functional.logsigmoid(pi_logit); lq = torch.nn.functional.logsigmoid(-pi_logit) + log_rho
    le = torch.where(mask, le, torch.zeros_like(le)); l1 = torch.where(mask, l1, torch.full_like(l1, -1e4))
    ti = torch.full_like(le, 1.0 / tau)
    return torch.stack([mu[..., 0], mu[..., 1], mu[..., 2], n_lit[..., 0], n_lit[..., 1], n_lit[..., 2], s0 ** 2, sth ** 2,
                        le, l1, lpi, lq, ti], -1).float()


def plane_factor2_fused(*a, **kw):
    with torch.autocast("cuda", enabled=False):
        return PlaneFactor2Fn.apply(two_hyp_params(*[t.float() if torch.is_tensor(t) and t.dtype != torch.bool else t for t in a], **kw))


def plane_factor2_torch(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau=4.0):
    """Reference implementation (tests)."""
    from .density import cell_centres
    P = two_hyp_params(mu, n_lit, s0, sth, eps_logit, pi_logit, log_rho, mask, tau)
    C = cell_centres(P.device).view(-1, 3)
    out = 0
    for k in range(P.shape[1]):
        p = P[:, k]
        D = C[None] - p[:, None, :3]; d = (D * p[:, None, 3:6]).sum(-1); r2 = (D * D).sum(-1)
        s2 = SIGMA_CELL2 + p[:, 6:7] + r2 * p[:, 7:8]
        inner = torch.logaddexp(p[:, 10:11] - d * d / (2 * s2), p[:, 11:12] + torch.nn.functional.logsigmoid(d * p[:, 12:13]))
        out = out + torch.logaddexp(p[:, 8:9], p[:, 9:10] + inner)
    return out.view(-1, NC ** 3, NF ** 3)
