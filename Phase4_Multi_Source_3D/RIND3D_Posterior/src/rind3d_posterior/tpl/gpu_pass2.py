"""Exact GPU pass, version 2: the queue lives on the device during a pass; survivors and children are compacted on
the device with an order-preserving scan (survivor and expansion ranks packed in one int64 cumsum), so FIFO order,
survivor order, the first refuter and Unew/Lnew are identical to the CPU pass (see gpu_pass.py).  If a segment could
reach the CPU's queue-capacity guard or survivor overflow, the pass is handed back to the CPU from scratch
(`fallback`), which is exact because the GPU path has not modified any host state that the CPU pass reads."""
import time
import numpy as np
from numba import njit
import cupy as cp
from .gpu_pass import _okey_host, _okey_decode

KERNEL2 = r'''
#define INF __longlong_as_double(0x7ff0000000000000LL)
#define AMARG 1e-12
#define MARG 1e-11
__device__ __forceinline__ unsigned long long okey(double x) {
    long long b = __double_as_longlong(x);
    return b >= 0 ? ((unsigned long long)b ^ 0x8000000000000000ULL) : ~((unsigned long long)b);
}
__device__ __forceinline__ void pbr(const double* pl, const double* lo, const double* hi, double* mn, double* mx) {
    double a0 = pl[0], a1 = pl[1], a2 = pl[2], b = pl[3];
    double m = b, M = b;
    if (a0 >= 0) { m += a0 * lo[0]; M += a0 * hi[0]; } else { m += a0 * hi[0]; M += a0 * lo[0]; }
    if (a1 >= 0) { m += a1 * lo[1]; M += a1 * hi[1]; } else { m += a1 * hi[1]; M += a1 * lo[1]; }
    if (a2 >= 0) { m += a2 * lo[2]; M += a2 * hi[2]; } else { m += a2 * hi[2]; M += a2 * lo[2]; }
    *mn = m; *mx = M;
}
// rpl.predicates.hull_misses_component(Rlo, Rhi, qlo, qhi, P, K, c, clo, chi)
__device__ bool hull_misses(const double* R, const double* q, const double* P, int kmax, const int* K, int c,
                            const double* clo, const double* chi) {
    double ulo[3], uhi[3];
    for (int k = 0; k < 3; k++) { ulo[k] = fmin(R[k], q[k]); uhi[k] = fmax(R[3 + k], q[3 + k]); }
    bool disj = false;
    for (int k = 0; k < 3; k++) if (uhi[k] < clo[3 * c + k] || chi[3 * c + k] < ulo[k]) { disj = true; break; }
    if (disj) return true;
    for (int i = 0; i < K[c]; i++) {
        const double* pl = P + ((long long)c * kmax + i) * 4;
        double mn1, mx1, mn2, mx2;
        pbr(pl, R, R + 3, &mn1, &mx1);
        if (mn1 <= MARG) continue;
        pbr(pl, q, q + 3, &mn2, &mx2);
        if (mn2 > MARG) return true;
    }
    return false;
}
// face value at corner cr, computed exactly as eval_comp2: base (+a) (+b) (+g)
__device__ __forceinline__ double fcorner(const double* pl, const double* q, double d0, double d1, double d2, int cr) {
    double v = pl[0] * q[0] + pl[1] * q[1] + pl[2] * q[2] + pl[3];
    if (cr & 1) v += pl[0] * d0;
    if (cr & 2) v += pl[1] * d1;
    if (cr & 4) v += pl[2] * d2;
    return v;
}
// engine_p2.eval_comp2: 0 COVER, 1 NOCOVER, 2 PARTIAL, 3 AMBG
__device__ int eval_comp(const double* q, bool ispt, const double* P, int kmax, int c, int nuf, int nfr,
                         const int* uf, int nsil, const int* sfi, const int* sbi, const double* tlo, const double* thi,
                         double m, double* tmn, double* tmx, int* nund_o, int* kund_o) {
    double d0 = q[3] - q[0], d1 = q[4] - q[1], d2 = q[5] - q[2];
    int nc = ispt ? 1 : 8;
    bool amb = false;
    for (int u = 0; u < nfr; u++) {
        const double* pl = P + ((long long)c * kmax + uf[u]) * 4;
        double mn = INF, mx = -INF;
        for (int cr = 0; cr < nc; cr++) {
            double v = fcorner(pl, q, d0, d1, d2, cr);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (mn > AMARG) return 1;
        if (mx >= -AMARG) amb = true;
    }
    if (amb) return 3;
    int nund = 0, kund = -1;
    for (int k = 0; k < nsil; k++) {
        const double* pf = P + ((long long)c * kmax + uf[sfi[k]]) * 4;
        const double* pb = P + ((long long)c * kmax + uf[sbi[k]]) * 4;
        double tm = INF, tM = -INF;
        for (int cr = 0; cr < nc; cr++) {
            double ff = fcorner(pf, q, d0, d1, d2, cr);
            double fb = fcorner(pb, q, d0, d1, d2, cr);
            double t = m * fb / ff;
            if (t < tm) tm = t;
            if (t > tM) tM = t;
        }
        tmn[k] = tm; tmx[k] = tM;
        if (tlo[k] >= tM) return 1;
        if (thi[k] <= tm) continue;
        nund++; kund = k;
    }
    *nund_o = nund; *kund_o = kund;
    return nund == 0 ? 0 : 2;
}
__device__ bool receiver_limited(int nsil, const double* tlo, const double* thi, const double* tmn, const double* tmx,
                                 double rho) {
    for (int k = 0; k < nsil; k++) {
        if (thi[k] <= tmn[k] || tlo[k] >= tmx[k]) continue;
        double wr = thi[k] - tlo[k];
        if (!isfinite(wr) || (tmx[k] - tmn[k]) >= rho * wr) return true;
    }
    return false;
}

// ---- FP32 filter (USE_F32).  Decides only results that need no tau values (NOCOVER, COVER, AMBG) and only when
// certain; otherwise returns -1 and the caller evaluates in FP64.  Bounds: |f32 - f| <= E (per face, from the host:
// 8u(|n0|+|n1|+|n2|+|d|)(1 + 2^-20), rounded up; the 2^-20 term covers the FP64 engine's own error), directed
// rounding for every bound operation, so [lo, hi] below contain the FP64 engine's values.
__device__ __forceinline__ float f32c(const float* pl, const float* x) {
    return pl[0] * x[0] + pl[1] * x[1] + pl[2] * x[2] + pl[3];
}
__device__ int eval_f32(const float* x6, bool ispt, const float* P32, const float* E32, int kmax, int c, int nfr,
                        const int* uf, int nsil, const int* sfi, const int* sbi, const double* tlo, const double* thi,
                        float m) {
    const float u = 5.9604645e-08f;
    int nc = ispt ? 1 : 8;
    bool unc = false, amb = false, ambu = false;
    for (int t = 0; t < nfr; t++) {
        const float* pl = P32 + ((long long)c * kmax + uf[t]) * 4;
        float E = E32[c * kmax + uf[t]];
        float lmin = 3.4e38f, hmin = 3.4e38f, lmax = -3.4e38f, hmax = -3.4e38f;
        for (int cr = 0; cr < nc; cr++) {
            float x[3] = {(cr & 1) ? x6[3] : x6[0], (cr & 2) ? x6[4] : x6[1], (cr & 4) ? x6[5] : x6[2]};
            float v = f32c(pl, x);
            float lo = __fsub_rd(v, E), hi = __fadd_ru(v, E);
            lmin = fminf(lmin, lo); hmin = fminf(hmin, hi); lmax = fmaxf(lmax, lo); hmax = fmaxf(hmax, hi);
        }
        if ((double)lmin > AMARG) return 1;                 // NOCOVER certain (FP64 returns NOCOVER at some face)
        if (!((double)hmin <= AMARG)) unc = true;
        if ((double)lmax >= -AMARG) amb = true;
        else if (!((double)hmax < -AMARG)) ambu = true;
    }
    if (unc) return -1;
    if (amb) return 3;
    if (ambu) return -1;
    bool allcert = true;
    for (int k = 0; k < nsil; k++) {
        const float* pf = P32 + ((long long)c * kmax + uf[sfi[k]]) * 4;
        const float* pb = P32 + ((long long)c * kmax + uf[sbi[k]]) * 4;
        float Ef = E32[c * kmax + uf[sfi[k]]], Eb = E32[c * kmax + uf[sbi[k]]];
        float mnl = 3.4e38f, mnh = 3.4e38f, mxl = -3.4e38f, mxh = -3.4e38f;
        bool bad = false;
        for (int cr = 0; cr < nc; cr++) {
            float x[3] = {(cr & 1) ? x6[3] : x6[0], (cr & 2) ? x6[4] : x6[1], (cr & 4) ? x6[5] : x6[2]};
            float vf = f32c(pf, x), vb = f32c(pb, x);
            float af = fabsf(vf);
            if (!(af > 2.0f * Ef)) { bad = true; break; }
            float t = m * vb / vf;
            float at = fabsf(t);
            float num = __fadd_ru(Eb, __fmul_ru(at, Ef));
            float den = __fsub_rd(af, Ef);
            float err = __fadd_ru(__fdiv_ru(num, den), __fmul_ru(4.0f * u, at));
            err = __fadd_ru(__fmul_ru(err, 1.0f + 16.0f * u), 1e-30f);
            float tl = __fsub_rd(t, err), th = __fadd_ru(t, err);
            mnl = fminf(mnl, tl); mnh = fminf(mnh, th); mxl = fmaxf(mxl, tl); mxh = fmaxf(mxh, th);
        }
        if (bad) { allcert = false; continue; }
        if (tlo[k] >= (double)mxh) return 1;                 // NOCOVER certain via edge k
        if (!(tlo[k] < (double)mxl)) { allcert = false; continue; }
        if (!(thi[k] <= (double)mnl)) allcert = false;       // not certainly lit (PARTIAL or uncertain)
    }
    return allcert ? 0 : -1;
}
// act: 0 drop, 1 survivor, 2 expand, 3 refutes the region
extern "C" __global__ void pass_seg2(
    const long long* __restrict__ Wc, const signed char* __restrict__ Wk, const long long* __restrict__ Wm, int n,
    long long relmask, long long Xmask, int dual, double rho,
    const double* __restrict__ P, int kmax, const int* __restrict__ K, const double* __restrict__ clo,
    const double* __restrict__ chi, const double* __restrict__ Rbox, long long o0, long long o1, long long o2,
    double inv, int C, int emax, const int* __restrict__ ci, const double* __restrict__ cd,
    const float* __restrict__ P32, const float* __restrict__ E32,
    signed char* __restrict__ act, long long* __restrict__ newm_o, long long* __restrict__ flag, unsigned long long* __restrict__ ref, long long seg_lo,
    unsigned long long* __restrict__ Ub, unsigned long long* __restrict__ Lb, unsigned long long* __restrict__ cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    // ci layout: nuf[C] nfr[C] nsil[C] uf[C*kmax] silfi[C*emax] silbi[C*emax]; cd: mult[C] tlo[C*emax] thi[C*emax]
    const int* nuf = ci; const int* nfr = ci + C; const int* nsil = ci + 2 * C;
    const int* uf = ci + 3 * C; const int* sfi = uf + C * kmax; const int* sbi = sfi + C * emax;
    const double* mult = cd; const double* tlo = cd + C; const double* thi = tlo + C * emax;
    long long code = Wc[tid];
    int kls = Wk[tid];
    long long mask = Wm[tid] & relmask;
    int l = (int)((code >> 48) & 0xFFFF);
    long long bx = (code >> 32) & 0xFFFF, by = (code >> 16) & 0xFFFF, bz = code & 0xFFFF;
    long long s = 1LL << l;
    double q[6];
    q[0] = (o0 + bx * s + 0.5) * inv; q[1] = (o1 + by * s + 0.5) * inv; q[2] = (o2 + bz * s + 0.5) * inv;
    q[3] = (o0 + (bx + 1) * s - 0.5) * inv; q[4] = (o1 + (by + 1) * s - 0.5) * inv; q[5] = (o2 + (bz + 1) * s - 0.5) * inv;
    bool ispt = (l == 0);
    float x6[6];
    for (int k = 0; k < 6; k++) x6[k] = (float)q[k];
    unsigned long long n32 = 0, n64 = 0;
    long long newmask = 0;
    bool xunk = false, rexp = !dual;
    double tmn[EMAXC], tmx[EMAXC], tmu[EMAXC];
    unsigned long long ncomp = 0, namb = 0;
    signed char a = 1;
    if (kls == 0) {                                   // DARK
        bool cover = false, amb = false;
        int ncand = 0, uc = -1;
        for (int c = 0; c < C; c++) {
            if (!((mask >> c) & 1)) continue;
            ncomp++;
            if ((Xmask >> c) & 1) {
                if (hull_misses(Rbox, q, P, kmax, K, c, clo, chi)) continue;
                newmask |= 1LL << c; ncand++; amb = true; xunk = true; continue;
            }
            int nund = 0, kund = -1;
            int st = -1;
#if USE_F32
            st = eval_f32(x6, ispt, P32, E32, kmax, c, nfr[c], uf + c * kmax, nsil[c], sfi + c * emax, sbi + c * emax,
                          tlo + c * emax, thi + c * emax, (float)mult[c]);
#endif
            if (st < 0) {
                n64++;
                st = eval_comp(q, ispt, P, kmax, c, nuf[c], nfr[c], uf + c * kmax, nsil[c], sfi + c * emax,
                               sbi + c * emax, tlo + c * emax, thi + c * emax, mult[c], tmn, tmx, &nund, &kund);
            } else n32++;
            if (st == 0) { cover = true; break; }
            if (st == 1) continue;
            if (st == 3) { amb = true; rexp = true; namb++; }
            else if (!ispt && !rexp) rexp = receiver_limited(nsil[c], tlo + c * emax, thi + c * emax, tmn, tmx, rho);
            newmask |= 1LL << c; ncand++; uc = c;
            if (st == 2) for (int k = 0; k < nsil[c]; k++) tmu[k] = tmn[k];
        }
        if (cover) a = 0;
        else if (ncand == 0) a = 3;
        else if (ncand == 1 && !amb) {
            for (int k = 0; k < nsil[uc]; k++) {
                if (tmu[k] < thi[uc * emax + k]) atomicMin(&Ub[uc * emax + k], okey(tmu[k]));
            }
            a = 0;
        } else a = 1;
    } else {                                          // LIT
        bool out = false;
        for (int c = 0; c < C; c++) {
            if (!((mask >> c) & 1)) continue;
            ncomp++;
            if ((Xmask >> c) & 1) {
                if (hull_misses(Rbox, q, P, kmax, K, c, clo, chi)) continue;
                newmask |= 1LL << c; xunk = true; continue;
            }
            int nund = 0, kund = -1;
            int st = -1;
#if USE_F32
            st = eval_f32(x6, ispt, P32, E32, kmax, c, nfr[c], uf + c * kmax, nsil[c], sfi + c * emax, sbi + c * emax,
                          tlo + c * emax, thi + c * emax, (float)mult[c]);
#endif
            if (st < 0) {
                n64++;
                st = eval_comp(q, ispt, P, kmax, c, nuf[c], nfr[c], uf + c * kmax, nsil[c], sfi + c * emax,
                               sbi + c * emax, tlo + c * emax, thi + c * emax, mult[c], tmn, tmx, &nund, &kund);
            } else n32++;
            if (st == 0) { out = true; break; }
            if (st == 1) continue;
            if (st == 2 && nund == 1) { atomicMax(&Lb[c * emax + kund], okey(tmx[kund])); continue; }
            if (st == 3) { rexp = true; namb++; }
            else if (!ispt && !rexp) rexp = receiver_limited(nsil[c], tlo + c * emax, thi + c * emax, tmn, tmx, rho);
            newmask |= 1LL << c;
        }
        if (out) a = 3;
        else if (newmask == 0) a = 0;
        else a = 1;
    }
    if (a == 1 && !ispt && !xunk && rexp) a = 2;
    act[tid] = a; newm_o[tid] = newmask;
    flag[tid] = (a == 1 ? 1LL : 0LL) | (a == 2 ? (1LL << 32) : 0LL);
    if (a == 1) atomicAdd(&cnt[2], 1ULL);
    if (a == 2) atomicAdd(&cnt[3], 1ULL);
    if (a == 3) atomicMin(ref, (unsigned long long)(seg_lo + tid));
    atomicAdd(&cnt[0], ncomp);
    if (namb) atomicAdd(&cnt[1], namb);
    if (n32) atomicAdd(&cnt[5], n32);
    if (n64) atomicAdd(&cnt[6], n64);
}

// exact capacity guard: expansion candidates after the first A (FIFO order) are kept as survivors
extern "C" __global__ void guardfix(signed char* act, long long* flag, const long long* __restrict__ scan, int n,
                                    long long A) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    signed char a = act[tid];
    if (a == 2 && (scan[tid] >> 32) > A) { a = 1; act[tid] = 1; }
    flag[tid] = (a == 1 ? 1LL : 0LL) | (a == 2 ? (1LL << 32) : 0LL);
}
extern "C" __global__ void scatter(const signed char* __restrict__ act, const long long* __restrict__ newm,
    const long long* __restrict__ scan, int n, long long seg_lo,
    long long* Wc, signed char* Wk, long long* Wm, long long* Wa, long long nq_dev,
    long long* Qc, signed char* Qk, long long* Qm, long long* Qa, long long ns_dev)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    signed char a = act[tid];
    if (a != 1 && a != 2) return;
    long long idx = seg_lo + tid;
    long long sc = scan[tid];
    long long code = Wc[idx]; signed char kls = Wk[idx]; long long anc = Wa[idx]; long long m = newm[tid];
    if (a == 1) {
        long long r = ns_dev + (sc & 0xFFFFFFFFLL) - 1;
        Qc[r] = code; Qk[r] = kls; Qm[r] = m; Qa[r] = anc;
    } else {
        long long r = nq_dev + 8 * ((sc >> 32) - 1);
        long long l1 = ((code >> 48) & 0xFFFF) - 1;
        long long bx = (code >> 32) & 0xFFFF, by = (code >> 16) & 0xFFFF, bz = code & 0xFFFF;
        for (int c8 = 0; c8 < 8; c8++) {
            Wc[r + c8] = (l1 << 48) | ((2 * bx + (c8 & 1)) << 32) | ((2 * by + ((c8 >> 1) & 1)) << 16) | (2 * bz + ((c8 >> 2) & 1));
            Wk[r + c8] = kls; Wm[r + c8] = m; Wa[r + c8] = anc;
        }
    }
}
'''

_K2 = {}


def _kernels(emax, f32):
    e = (max(int(emax), 1), bool(f32))
    if e not in _K2:
        mod = cp.RawModule(code=KERNEL2, options=('--fmad=false', '-DEMAXC=%d' % e[0], '-DUSE_F32=%d' % int(e[1])))
        _K2[e] = (mod.get_function('pass_seg2'), mod.get_function('scatter'), mod.get_function('guardfix'))
    return _K2[e]


def _f32_up(x):
    y = np.asarray(x, np.float64).astype(np.float32)
    return np.where(y.astype(np.float64) < x, np.nextafter(y, np.float32(np.inf)), y).astype(np.float32)


@njit(cache=True)
def _pack(nuf, nfrc, nsil, uf, silfi, silbi, mult, tlo, thi, Rlo, Rhi, ci, cd):
    C = nuf.shape[0]; kmax = uf.shape[1]; emax = silfi.shape[1]
    j = 0
    for c in range(C):
        ci[j] = nuf[c]; j += 1
    for c in range(C):
        ci[j] = nfrc[c]; j += 1
    for c in range(C):
        ci[j] = nsil[c]; j += 1
    for c in range(C):
        for t in range(kmax):
            ci[j] = uf[c, t]; j += 1
    for c in range(C):
        for t in range(emax):
            ci[j] = silfi[c, t]; j += 1
    for c in range(C):
        for t in range(emax):
            ci[j] = silbi[c, t]; j += 1
    j = 0
    for c in range(C):
        cd[j] = mult[c]; j += 1
    for c in range(C):
        for t in range(emax):
            cd[j] = tlo[c, t]; j += 1
    for c in range(C):
        for t in range(emax):
            cd[j] = thi[c, t]; j += 1
    for k in range(3):
        cd[j] = Rlo[k]; j += 1
    for k in range(3):
        cd[j] = Rhi[k]; j += 1


class GpuPass2:
    def __init__(self, P, K, clo, chi, o, N, NQ, block=128, f32=False):
        self.f32 = f32
        u = 2.0 ** -24
        Pd = np.ascontiguousarray(P, np.float64)
        Eh = 8 * u * np.abs(Pd).sum(axis=2) * (1 + 2.0 ** -20)
        Eh = Eh * (1 + 2.0 ** -40)
        self.dP32 = cp.asarray(Pd.astype(np.float32)); self.dE32 = cp.asarray(_f32_up(Eh))
        self.C, self.kmax = P.shape[0], P.shape[1]
        self.dP = cp.asarray(np.ascontiguousarray(P, np.float64)); self.dK = cp.asarray(np.ascontiguousarray(K, np.int32))
        self.dclo = cp.asarray(np.ascontiguousarray(clo, np.float64)); self.dchi = cp.asarray(np.ascontiguousarray(chi, np.float64))
        self.o = [np.int64(v) for v in o]; self.inv = np.float64(1.0 / N); self.NQ = NQ; self.block = block
        self.wcap = 0; self.qcap = 0; self.scap = 0
        self._wgrow(1 << 18); self._qgrow(1 << 18); self._sgrow(1 << 18)
        self.dcnt = cp.zeros(8, cp.uint64); self.dref = self.dcnt[4:5]
        self.n32 = 0; self.n64 = 0
        self.dsm = cp.empty(5, cp.uint64)          # [cnt0..3, ref] staged for one small download
        import cupyx
        self._pin = cupyx.empty_pinned
        self.hsm = cupyx.empty_pinned(8, np.uint64)
        self.hanc = cupyx.empty_pinned(1, np.int64)
        self.ev = cp.cuda.Event(block=True, disable_timing=True)   # blocking sync: waiting does not spin a core
        self.pcap = 0
        self._pgrow(1 << 18)
        self.segments = 0; self.gpu_entries = 0; self.fallbacks = 0; self.chunk = 1 << 21; self.guards = 0
        self.tt = np.zeros(6); self.profile = False
        self.ci = None; self.cd = None

    def _pgrow(self, n):
        n = 1 << int(np.ceil(np.log2(max(n, 1))))
        if n <= self.pcap:
            return
        P = self._pin
        self.hWc = P(n, np.int64); self.hWk = P(n, np.int8); self.hWm = P(n, np.int64); self.hWa = P(n, np.int64)
        self.pcap = n

    def _wgrow(self, n):
        n = 1 << int(np.ceil(np.log2(max(n, 1))))
        if n <= self.wcap:
            return
        old = (self.dWc, self.dWk, self.dWm, self.dWa) if self.wcap else None
        self.dWc = cp.empty(n, cp.int64); self.dWk = cp.empty(n, cp.int8); self.dWm = cp.empty(n, cp.int64); self.dWa = cp.empty(n, cp.int64)
        if old is not None:
            m = self.wcap
            self.dWc[:m] = old[0]; self.dWk[:m] = old[1]; self.dWm[:m] = old[2]; self.dWa[:m] = old[3]
        self.wcap = n

    def _qgrow(self, n):
        n = 1 << int(np.ceil(np.log2(max(n, 1))))
        if n <= self.qcap:
            return
        old = (self.dQc, self.dQk, self.dQm, self.dQa) if self.qcap else None
        self.dQc = cp.empty(n, cp.int64); self.dQk = cp.empty(n, cp.int8); self.dQm = cp.empty(n, cp.int64); self.dQa = cp.empty(n, cp.int64)
        if old is not None:
            m = self.qcap
            self.dQc[:m] = old[0]; self.dQk[:m] = old[1]; self.dQm[:m] = old[2]; self.dQa[:m] = old[3]
        self.qcap = n

    def _sgrow(self, n):
        n = 1 << int(np.ceil(np.log2(max(n, 1))))
        if n <= self.scap:
            return
        self.dact = cp.empty(n, cp.int8); self.dnewm = cp.empty(n, cp.int64); self.dflag = cp.empty(n, cp.int64)
        self.dscan = cp.empty(n, cp.int64)
        self.scap = n

    def run(self, i, nq, ns, Wc, Wk, Wm, Wa, Qc, Qk, Qm, Qa, relmask, Xmask, dual, rho, Rlo, Rhi, nuf, nfrc, uf,
            nsil, silfi, silbi, tlo, thi, mult, Unew, Lnew):
        """Returns (i, nq, ns, out, anc, overflow, processed, comp_evals, amb_evals, expansions, guards, fallback)."""
        pr = self.profile
        T0 = time.perf_counter()
        C, kmax = self.C, self.kmax
        emax = tlo.shape[1]
        if self.ci is None or self.ci.shape[0] != 3 * C + C * kmax + 2 * C * emax:
            import cupyx
            self.ci = cupyx.empty_pinned(3 * C + C * kmax + 2 * C * emax, np.int32)
            self.cd = cupyx.empty_pinned(C + 2 * C * emax + 6, np.float64)
            self.hUL = cupyx.empty_pinned(2 * C * emax, np.uint64)
            self.dci = cp.empty(self.ci.shape[0], cp.int32); self.dcd = cp.empty(self.cd.shape[0], cp.float64)
            self.dUL = cp.empty(2 * C * emax, cp.uint64)
        _pack(nuf, nfrc, nsil, uf, silfi, silbi, mult, tlo, thi, Rlo, Rhi, self.ci, self.cd)
        self.hUL[:C * emax] = _okey_host(Unew.ravel()); self.hUL[C * emax:] = _okey_host(Lnew.ravel())
        st = cp.cuda.get_current_stream()
        dci = self.dci; dcd = self.dcd; dUL = self.dUL
        dci.set(self.ci, stream=st); dcd.set(self.cd, stream=st); dUL.set(self.hUL, stream=st)
        dU = dUL[:C * emax]; dL = dUL[C * emax:]
        dR = dcd[C + 2 * C * emax:]
        kern, scat, gfix = _kernels(emax, self.f32)
        CH = self.chunk
        self.dcnt.fill(0); self.dref.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
        # upload the FIFO segment W[i:nq] (device index d <-> host FIFO index i + d)
        n0 = nq - i
        self._wgrow(n0); self._pgrow(n0)
        self.hWc[:n0] = Wc[i:nq]; self.hWk[:n0] = Wk[i:nq]; self.hWm[:n0] = Wm[i:nq]; self.hWa[:n0] = Wa[i:nq]
        self.dWc[:n0].set(self.hWc[:n0], stream=st); self.dWk[:n0].set(self.hWk[:n0], stream=st)
        self.dWm[:n0].set(self.hWm[:n0], stream=st); self.dWa[:n0].set(self.hWa[:n0], stream=st)
        if pr:
            T1 = time.perf_counter(); self.tt[0] += T1 - T0
        lo_d = 0; nq_d = n0; ns_d = 0
        nq_h = nq; ns_h = ns
        out = False; anc = -1; processed = 0; nexp_t = 0; nguard_t = 0
        S_prev = 0; E_prev = 0
        while lo_d < nq_d:
            n = min(nq_d - lo_d, CH)
            self._sgrow(n)
            a1 = time.perf_counter() if pr else 0.0
            grid = ((n + self.block - 1) // self.block,)
            kern(grid, (self.block,), (self.dWc[lo_d:], self.dWk[lo_d:], self.dWm[lo_d:], np.int32(n), np.int64(relmask),
                                       np.int64(Xmask), np.int32(1 if dual else 0), np.float64(rho), self.dP, np.int32(kmax),
                                       self.dK, self.dclo, self.dchi, dR, self.o[0], self.o[1], self.o[2], self.inv,
                                       np.int32(C), np.int32(emax), dci, dcd, self.dP32, self.dE32, self.dact, self.dnewm, self.dflag, self.dref,
                                       np.int64(lo_d), dU, dL, self.dcnt))
            scan = cp.cumsum(self.dflag[:n], out=self.dscan[:n])
            self.dcnt.get(stream=st, out=self.hsm, blocking=False)
            self.ev.record(st); self.ev.synchronize()
            Scum = int(self.hsm[2]); Ecum = int(self.hsm[3]); ref = int(self.hsm[4])
            if pr:
                cp.cuda.Stream.null.synchronize(); a2 = time.perf_counter(); self.tt[2] += a2 - a1
            self.segments += 1; self.gpu_entries += n
            S = Scum - S_prev; E = Ecum - E_prev; S_prev = Scum; E_prev = Ecum
            if ref != 0xFFFFFFFFFFFFFFFF:
                out = True; processed += ref - lo_d + 1
                self.dWa[ref:ref + 1].get(stream=st, out=self.hanc, blocking=False)
                self.ev.record(st); self.ev.synchronize()
                anc = int(self.hanc[0])
                i = i + ref + 1
                break
            # exact capacity guard (CPU: an expansion with nq + 8 > NQ - 1024 is kept as a survivor instead)
            A = max(0, (self.NQ - 1024 - nq_h) // 8)
            if E > A:
                gfix(grid, (self.block,), (self.dact, self.dflag, scan, np.int32(n), np.int64(A)))
                scan = cp.cumsum(self.dflag[:n], out=self.dscan[:n])
                g = E - A; S += g; E = A; nguard_t += g; self.guards += g
            # survivor overflow (CPU: ns >= NQ at a survivor) -> redo the pass on the CPU (exact)
            if ns_h + S > self.NQ:
                self.fallbacks += 1
                return 0, 0, 0, False, -1, False, 0, 0, 0, 0, 0, True
            self._wgrow(nq_d + 8 * E); self._qgrow(ns_d + S)
            scat(grid, (self.block,), (self.dact, self.dnewm, scan, np.int32(n), np.int64(lo_d), self.dWc, self.dWk,
                                       self.dWm, self.dWa, np.int64(nq_d), self.dQc, self.dQk, self.dQm, self.dQa, np.int64(ns_d)))
            processed += n; nexp_t += E
            ns_d += S; ns_h += S
            lo_d += n; nq_d += 8 * E; nq_h += 8 * E
            if pr:
                cp.cuda.Stream.null.synchronize(); self.tt[3] += time.perf_counter() - a2
        a4 = time.perf_counter() if pr else 0.0
        if not out and ns_d:
            self._pgrow(ns_d)
            self.dQc[:ns_d].get(stream=st, out=self.hWc[:ns_d], blocking=False)
            self.dQk[:ns_d].get(stream=st, out=self.hWk[:ns_d], blocking=False)
            self.dQm[:ns_d].get(stream=st, out=self.hWm[:ns_d], blocking=False)
            self.dQa[:ns_d].get(stream=st, out=self.hWa[:ns_d], blocking=False)
        dUL.get(stream=st, out=self.hUL, blocking=False)
        self.dcnt.get(stream=st, out=self.hsm, blocking=False)
        self.ev.record(st); self.ev.synchronize()
        if not out and ns_d:
            Qc[ns:ns + ns_d] = self.hWc[:ns_d]; Qk[ns:ns + ns_d] = self.hWk[:ns_d]
            Qm[ns:ns + ns_d] = self.hWm[:ns_d]; Qa[ns:ns + ns_d] = self.hWa[:ns_d]
        Unew.ravel()[:] = _okey_decode(self.hUL[:C * emax]); Lnew.ravel()[:] = _okey_decode(self.hUL[C * emax:])
        cnt = self.hsm[:4]
        self.n32 += int(self.hsm[5]); self.n64 += int(self.hsm[6])
        if pr:
            self.tt[4] += time.perf_counter() - a4
        return i if out else nq_h, nq_h, ns_h, out, anc, False, processed, int(cnt[0]), int(cnt[1]), nexp_t, nguard_t, False
