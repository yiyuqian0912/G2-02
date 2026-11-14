"""Exact GPU evaluation of one engine pass segment (Phase 2, step 3).

The CPU pass loop (engine_p2.stage2, `while i < nq`) processes a FIFO queue of entries.  Each entry's outcome
(drop / survivor / expand / refute) and its contribution to the collapsed bounds Unew/Lnew depend only on the entry
and on per-pass region data, never on other entries.  The only order-dependent parts are:
  * the first refuting entry (it ends the pass and its top-level ancestor goes to the learned ring);
  * the queue-capacity guard on expansions (`nq + 8 > NQ - 1024`);
  * survivor overflow (`ns >= NQ`);
  * the order of survivors and children.

This module evaluates a FIFO segment [i, nq) on the GPU in FP64, with the same operations in the same order as
the numba code (compiled with --fmad=false), then replays the order-dependent parts sequentially on the host
(`compact`).  Processing segment after segment (each one being the children appended by the previous one) is
exactly the CPU's FIFO order.  The resulting queue, survivor list, Unew/Lnew and refuter are therefore identical
to the CPU pass.  Statistics differ only when a pass ends early: the GPU has also evaluated the entries after the
refuter."""
import time
import numpy as np
from numba import njit

try:
    import cupy as cp
except Exception:          # pragma: no cover
    cp = None

KERNEL = r'''
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
// act: 0 drop, 1 survivor, 2 expand, 3 refutes the region
extern "C" __global__ void pass_seg(
    const long long* __restrict__ Wc, const signed char* __restrict__ Wk, const long long* __restrict__ Wm, int n,
    long long relmask, long long Xmask, int dual, double rho,
    const double* __restrict__ P, int kmax, const int* __restrict__ K, const double* __restrict__ clo,
    const double* __restrict__ chi, const double* __restrict__ Rbox, long long o0, long long o1, long long o2,
    double inv, int C, int emax, const int* __restrict__ ci, const double* __restrict__ cd,
    signed char* __restrict__ act, long long* __restrict__ newm_o,
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
            int st = eval_comp(q, ispt, P, kmax, c, nuf[c], nfr[c], uf + c * kmax, nsil[c], sfi + c * emax,
                               sbi + c * emax, tlo + c * emax, thi + c * emax, mult[c], tmn, tmx, &nund, &kund);
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
            int st = eval_comp(q, ispt, P, kmax, c, nuf[c], nfr[c], uf + c * kmax, nsil[c], sfi + c * emax,
                               sbi + c * emax, tlo + c * emax, thi + c * emax, mult[c], tmn, tmx, &nund, &kund);
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
    atomicAdd(&cnt[0], ncomp);
    if (namb) atomicAdd(&cnt[1], namb);
}
'''


@njit(cache=True)
def compact(act, newm, n, i0, Wc, Wk, Wm, Wa, nq, Qc, Qk, Qm, Qa, ns, NQ):
    """Sequential replay of the order-dependent part of the CPU pass for entries W[i0 : i0+n].
    Returns (nq, ns, refuter index or -1, overflow, processed, expansions, capacity guards)."""
    nexp = 0
    nguard = 0
    for j in range(n):
        a = act[j]
        i = i0 + j
        if a == 3:
            return nq, ns, i, False, j + 1, nexp, nguard
        if a == 0:
            continue
        if a == 2 and nq + 8 > NQ - 1024:
            a = 1
            nguard += 1
        if a == 2:
            nexp += 1
            code = Wc[i]
            l1 = ((code >> 48) & 0xFFFF) - 1
            bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
            for c8 in range(8):
                Wc[nq] = (np.int64(l1) << 48) | (np.int64(2 * bx + (c8 & 1)) << 32) | \
                         (np.int64(2 * by + ((c8 >> 1) & 1)) << 16) | np.int64(2 * bz + ((c8 >> 2) & 1))
                Wk[nq] = Wk[i]
                Wm[nq] = newm[j]
                Wa[nq] = Wa[i]
                nq += 1
        else:
            if ns >= NQ:
                return nq, ns, -1, True, j + 1, nexp, nguard
            Qc[ns] = Wc[i]; Qk[ns] = Wk[i]; Qm[ns] = newm[j]; Qa[ns] = Wa[i]
            ns += 1
    return nq, ns, -1, False, n, nexp, nguard


def _okey_host(x):
    b = np.asarray(x, np.float64).view(np.int64)
    u = b.view(np.uint64)
    return np.where(b >= 0, u ^ np.uint64(0x8000000000000000), ~u)


def _okey_decode(k):
    k = np.asarray(k, np.uint64)
    pos = (k & np.uint64(0x8000000000000000)) != 0
    b = np.where(pos, k ^ np.uint64(0x8000000000000000), ~k)
    return b.view(np.float64)


_KCACHE = {}


class GpuPass:
    """Per-window GPU context: static geometry on the device and work buffers of the engine's queue size."""

    def __init__(self, P, K, clo, chi, o, N, NQ, block=128):
        self.C, self.kmax = P.shape[0], P.shape[1]
        self.dP = cp.asarray(np.ascontiguousarray(P, np.float64))
        self.dK = cp.asarray(np.ascontiguousarray(K, np.int32))
        self.dclo = cp.asarray(np.ascontiguousarray(clo, np.float64))
        self.dchi = cp.asarray(np.ascontiguousarray(chi, np.float64))
        self.o = [np.int64(v) for v in o]
        self.inv = np.float64(1.0 / N)
        self.NQ = NQ
        self.block = block
        self.cap = 1 << 20
        self._alloc(self.cap)
        self.dUL = cp.empty(0, cp.uint64)
        self.dcnt = cp.zeros(2, cp.uint64)
        self.hact = np.empty(0, np.int8)
        self.hnewm = np.empty(0, np.int64)
        self.kern = {}
        self.t_kernel_calls = 0
        self.segments = 0
        self.gpu_entries = 0
        self.tt = np.zeros(6)   # pack, h2d, kernel, d2h, compact, final
        self.profile = False

    def _alloc(self, cap):
        self.cap = cap
        self.dWc = cp.empty(cap, cp.int64); self.dWk = cp.empty(cap, cp.int8); self.dWm = cp.empty(cap, cp.int64)
        self.dact = cp.empty(cap, cp.int8); self.dnewm = cp.empty(cap, cp.int64)

    def _kernel(self, emax):
        e = max(int(emax), 1)
        k = _KCACHE.get(e)
        if k is None:
            k = cp.RawKernel(KERNEL, 'pass_seg', options=('--fmad=false', '-DEMAXC=%d' % e))
            _KCACHE[e] = k
        return k

    def run(self, i, nq, ns, Wc, Wk, Wm, Wa, Qc, Qk, Qm, Qa, relmask, Xmask, dual, rho, Rlo, Rhi, nuf, nfrc, uf,
            nsil, silfi, silbi, tlo, thi, mult, Unew, Lnew):
        C, kmax = self.C, self.kmax
        emax = tlo.shape[1]
        pr = self.profile
        T0 = time.perf_counter()
        ci = np.concatenate([nuf, nfrc, nsil, uf.ravel(), silfi.ravel(), silbi.ravel()]).astype(np.int32)
        cd = np.concatenate([mult, tlo.ravel(), thi.ravel()])
        dci = cp.asarray(ci); dcd = cp.asarray(cd)
        dR = cp.asarray(np.concatenate([Rlo, Rhi]))
        UL = np.concatenate([_okey_host(Unew.ravel()), _okey_host(Lnew.ravel())])
        dUL = cp.asarray(UL)
        dU = dUL[:C * emax]; dL = dUL[C * emax:]
        self.dcnt.fill(0)
        kern = self._kernel(emax)
        if pr:
            T1 = time.perf_counter(); self.tt[0] += T1 - T0
        out = False; anc = -1; overflow = False
        processed = 0; nexp_t = 0; nguard_t = 0
        while i < nq:
            n = nq - i
            if n > self.cap:
                self._alloc(1 << int(np.ceil(np.log2(n))))
            if pr:
                a0 = time.perf_counter()
            self.dWc[:n].set(Wc[i:nq]); self.dWk[:n].set(Wk[i:nq]); self.dWm[:n].set(Wm[i:nq])
            if pr:
                cp.cuda.Stream.null.synchronize(); a1 = time.perf_counter(); self.tt[1] += a1 - a0
            grid = ((n + self.block - 1) // self.block,)
            kern(grid, (self.block,), (self.dWc, self.dWk, self.dWm, np.int32(n), np.int64(relmask), np.int64(Xmask),
                                       np.int32(1 if dual else 0), np.float64(rho), self.dP, np.int32(kmax), self.dK,
                                       self.dclo, self.dchi, dR, self.o[0], self.o[1], self.o[2], self.inv,
                                       np.int32(C), np.int32(emax), dci, dcd, self.dact, self.dnewm, dU, dL, self.dcnt))
            if pr:
                cp.cuda.Stream.null.synchronize(); a2 = time.perf_counter(); self.tt[2] += a2 - a1
            if self.hact.shape[0] < n:
                self.hact = np.empty(self.cap, np.int8); self.hnewm = np.empty(self.cap, np.int64)
            self.dact[:n].get(out=self.hact[:n]); self.dnewm[:n].get(out=self.hnewm[:n])
            self.segments += 1; self.gpu_entries += n
            if pr:
                a3 = time.perf_counter(); self.tt[3] += a3 - a2
            nq2, ns, ref, overflow, proc, nexp, nguard = compact(self.hact, self.hnewm, n, i, Wc, Wk, Wm, Wa, nq, Qc,
                                                                 Qk, Qm, Qa, ns, self.NQ)
            processed += proc; nexp_t += nexp; nguard_t += nguard
            if pr:
                self.tt[4] += time.perf_counter() - a3
            if ref >= 0:
                out = True; anc = int(Wa[ref]); i = ref + 1; nq = nq2
                break
            if overflow:
                i = i + proc; nq = nq2
                break
            i = nq; nq = nq2
        if pr:
            a4 = time.perf_counter()
        ULh = dUL.get()
        Unew.ravel()[:] = _okey_decode(ULh[:C * emax])
        Lnew.ravel()[:] = _okey_decode(ULh[C * emax:])
        cnt = self.dcnt.get()
        if pr:
            self.tt[5] += time.perf_counter() - a4
        return i, nq, ns, out, anc, overflow, processed, int(cnt[0]), int(cnt[1]), nexp_t, nguard_t, False
