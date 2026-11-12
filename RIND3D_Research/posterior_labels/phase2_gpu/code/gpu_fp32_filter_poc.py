"""Phase 2: FP32 filtered pencil predicate on the GPU, with a rigorous forward-error bound and FP64 fallback.

For each (region, item, component) the kernel computes, in float32, every corner value f(x) = n.x + d directly
(4-term dot product) and tau = m f_b / f_f, with the error bounds below. Each edge is classified as:
* GT  (NOCOVER via this edge), when lo >= tmax_true is certain;
* LE, when hi <= tmin_true is certain;
* UND, when both are certainly false;
* AMB otherwise.

The (region, item, component) result is then:
* NOCOVER if any edge is certainly GT;
* otherwise AMB if any edge is AMB, and the triple is recomputed exactly in float64 (same kernel arithmetic as
  the CPU engine, --fmad=false);
* otherwise COVER (all edges LE) or PARTIAL (number of UND edges).

Why this equals the float64 engine. The float64 engine decides with float64 values whose error is ~1e-16
relative. A certain float32 decision means the true value is farther from the threshold than the float32 error
bound (>= 1e-6 relative), so the float64 values fall on the same side.

Error bounds (u = 2^-24; window corner coordinates are dyadic, hence exact in float32; n, d rounded to float32):
* |f32(x) - f(x)| <= E = 8u * (|n0| + |n1| + |n2| + |d|)
  (input rounding + 4-term dot product, generous; x in [0,1]);
* |tau32 - tau| <= (E_b + |tau32| E_f) / (|f_f32| - E_f) + 4u |tau32|, valid when |f_f32| > 2 E_f
  (otherwise AMB).
All bound arithmetic is inflated by (1 + 16u) plus 1e-30 absolute.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gpu_eval_poc import build_batch, cpu_eval, KERNEL as K64
from tpl.geom import Geom
from tpl.window import Window

K32 = r'''
extern "C" __global__ void eval_f32(
    const float* __restrict__ P32, const float* __restrict__ E32, int kmax,
    const long long* __restrict__ codes, int n_items,
    const int* __restrict__ pairs_rc, int n_rc,
    const float* __restrict__ lo_dn, const float* __restrict__ lo_up,
    const float* __restrict__ hi_dn, const float* __restrict__ hi_up, int emax,
    const int* __restrict__ nsil, const int* __restrict__ silf, const int* __restrict__ silb,
    const long long* o, float invN, float m,
    signed char* __restrict__ status, signed char* __restrict__ nund_out)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_rc * n_items;
    if (tid >= total) return;
    const float u = 5.9604645e-08f;          // 2^-24
    const float infl = 1.0f + 16.0f * u;
    int q = (int)(tid / n_items);
    int it = (int)(tid % n_items);
    int c = pairs_rc[2 * q + 1];
    long long code = codes[it];
    int l = (int)((code >> 48) & 0xFFFF);
    long long bx = (code >> 32) & 0xFFFF, by = (code >> 16) & 0xFFFF, bz = code & 0xFFFF;
    long long s = 1LL << l;
    // dyadic corner coordinates: exact in float32 (integers < 2^24 times a power of two)
    float x0a = (float)(2 * (o[0] + bx * s) + 1) * (0.5f * invN), x0b = (float)(2 * (o[0] + (bx + 1) * s) - 1) * (0.5f * invN);
    float x1a = (float)(2 * (o[1] + by * s) + 1) * (0.5f * invN), x1b = (float)(2 * (o[1] + (by + 1) * s) - 1) * (0.5f * invN);
    float x2a = (float)(2 * (o[2] + bz * s) + 1) * (0.5f * invN), x2b = (float)(2 * (o[2] + (bz + 1) * s) - 1) * (0.5f * invN);
    int ncorner = (l == 0) ? 1 : 8;
    int nund = 0;
    bool amb = false;
    int ns = nsil[c];
    for (int k = 0; k < ns; k++) {
        int f = silf[c * emax + k], b = silb[c * emax + k];
        const float* pf = P32 + ((long long)c * kmax + f) * 4;
        const float* pb = P32 + ((long long)c * kmax + b) * 4;
        float Ef = E32[c * kmax + f], Eb = E32[c * kmax + b];
        float tmin_lo = 3.0e38f, tmax_hi = -3.0e38f, tmin_hi = 3.0e38f, tmax_lo = -3.0e38f;
        bool bad = false;
        for (int cr = 0; cr < ncorner; cr++) {
            float x0 = (cr & 1) ? x0b : x0a, x1 = (cr & 2) ? x1b : x1a, x2 = (cr & 4) ? x2b : x2a;
            float vf = pf[0] * x0 + pf[1] * x1 + pf[2] * x2 + pf[3];
            float vb = pb[0] * x0 + pb[1] * x1 + pb[2] * x2 + pb[3];
            if (fabsf(vf) <= 2.0f * Ef) { bad = true; break; }
            float t = m * vb / vf;
            float err = ((Eb + fabsf(t) * Ef) / (fabsf(vf) - Ef) + 4.0f * u * fabsf(t)) * infl + 1e-30f;
            float tl = t - err, th = t + err;
            if (tl < tmin_lo) tmin_lo = tl;      // lower bound of the true min
            if (th < tmin_hi) tmin_hi = th;      // upper bound of the true min
            if (th > tmax_hi) tmax_hi = th;      // upper bound of the true max
            if (tl > tmax_lo) tmax_lo = tl;      // lower bound of the true max
        }
        if (bad) { amb = true; continue; }
        long long e = (long long)q * emax + k;
        // GT (NOCOVER via edge k): lo >= tmax_true.  Certain if lo_dn >= upper bound of tmax_true.
        if (lo_dn[e] >= tmax_hi) { status[tid] = 1; nund_out[tid] = 0; return; }
        // certainly not GT if lo_up < lower bound of tmax_true
        if (!(lo_up[e] < tmax_lo)) { amb = true; continue; }
        // LE: hi <= tmin_true.  Certain if hi_up <= lower bound of tmin_true; certainly not if hi_dn > upper bound.
        if (hi_up[e] <= tmin_lo) continue;
        if (hi_dn[e] > tmin_hi) { nund++; continue; }
        amb = true;
    }
    if (amb) { status[tid] = 3; nund_out[tid] = 0; return; }                  // fallback to FP64
    status[tid] = (nund == 0) ? 0 : 2;
    nund_out[tid] = (signed char)nund;
}
'''


def f32_down(x):
    y = np.asarray(x, np.float64).astype(np.float32)
    y = np.where(y.astype(np.float64) > x, np.nextafter(y, np.float32(-np.inf)), y)
    return np.where(np.isfinite(x), y, np.where(x > 0, np.float32(np.inf), np.float32(-np.inf))).astype(np.float32)


def f32_up(x):
    y = np.asarray(x, np.float64).astype(np.float32)
    y = np.where(y.astype(np.float64) < x, np.nextafter(y, np.float32(np.inf)), y)
    return np.where(np.isfinite(x), y, np.where(x > 0, np.float32(np.inf), np.float32(-np.inf))).astype(np.float32)


def run(bundle, name, reps=5):
    import cupy as cp
    z = np.load(os.path.join(bundle, name + '.npz'))
    geom = Geom(z['P'], z['K']); win = Window(z['cls'], z['origin'], int(z['N']))
    P, codes, pairs, TL, TH, nsil, silf, silb, emax = build_batch(geom, win)
    n = len(pairs) * len(codes)
    if n == 0:
        return dict(name=name, pairs=0)
    o = win.origin.astype(np.int64)
    # CPU float64 reference (the engine's predicate)
    st_c = np.zeros(n, np.int8); nu_c = np.zeros(n, np.int8); fbuf = np.zeros((P.shape[1], 8))
    cpu_eval(P, codes[:1], pairs[:1], TL[:1], TH[:1], nsil, silf, silb, o, float(win.N), st_c, nu_c, fbuf)
    t = time.process_time(); cpu_eval(P, codes, pairs, TL, TH, nsil, silf, silb, o, float(win.N), st_c, nu_c, fbuf)
    cpu_s = time.process_time() - t
    # FP32 kernel inputs (bounds rounded outward)
    E64 = 8 * 2.0 ** -24 * (np.abs(P[..., :3]).sum(-1) + np.abs(P[..., 3]))
    P32 = cp.asarray(P.astype(np.float32)); E32 = cp.asarray(f32_up(E64 * (1 + 1e-6)))
    lo_dn = cp.asarray(f32_down(TL)); lo_up = cp.asarray(f32_up(TL)); hi_dn = cp.asarray(f32_down(TH)); hi_up = cp.asarray(f32_up(TH))
    k32 = cp.RawKernel(K32, 'eval_f32', options=('--fmad=false',))
    dcodes = cp.asarray(codes); dpairs = cp.asarray(pairs.astype(np.int32).ravel())
    dns = cp.asarray(nsil.astype(np.int32)); dsf = cp.asarray(silf.astype(np.int32).ravel()); dsb = cp.asarray(silb.astype(np.int32).ravel())
    do = cp.asarray(o); dst = cp.zeros(n, cp.int8); dnu = cp.zeros(n, cp.int8)
    args = (P32, E32, np.int32(P.shape[1]), dcodes, np.int32(len(codes)), dpairs, np.int32(len(pairs)), lo_dn, lo_up, hi_dn, hi_up,
            np.int32(emax), dns, dsf, dsb, do, np.float32(1.0 / win.N), np.float32(-1.0), dst, dnu)
    grid = ((n + 255) // 256,)
    k32(grid, (256,), args); cp.cuda.Device().synchronize()
    ev = cp.cuda.Event(); ev2 = cp.cuda.Event(); ts = []
    for _ in range(reps):
        ev.record(); k32(grid, (256,), args); ev2.record(); ev2.synchronize(); ts.append(cp.cuda.get_elapsed_time(ev, ev2) / 1000)
    gpu32_s = min(ts)
    st_g = cp.asnumpy(dst); nu_g = cp.asnumpy(dnu)
    amb = st_g == 3
    # FP64 fallback for the ambiguous triples (CPU reference values used as the fallback result here)
    st_f = st_g.copy(); nu_f = nu_g.copy(); st_f[amb] = st_c[amb]; nu_f[amb] = nu_c[amb]
    mism = int((st_f != st_c).sum()); mism_n = int((nu_f != nu_c).sum())
    # FP64 GPU kernel throughput on the same batch for comparison
    k64 = cp.RawKernel(K64, 'eval_w', options=('--fmad=false',))
    dP = cp.asarray(P); dTL = cp.asarray(np.ascontiguousarray(TL)); dTH = cp.asarray(np.ascontiguousarray(TH))
    d64st = cp.zeros(n, cp.int8); d64nu = cp.zeros(n, cp.int8)
    a64 = (dP, np.int32(P.shape[1]), dcodes, np.int32(len(codes)), dpairs, np.int32(len(pairs)), dTL, dTH, np.int32(emax),
           dns, dsf, dsb, do, np.float64(1.0 / win.N), d64st, d64nu)
    k64(grid, (256,), a64); cp.cuda.Device().synchronize(); ts = []
    for _ in range(reps):
        ev.record(); k64(grid, (256,), a64); ev2.record(); ev2.synchronize(); ts.append(cp.cuda.get_elapsed_time(ev, ev2) / 1000)
    gpu64_s = min(ts)
    return dict(name=name, side=int(win.L), evaluations=int(n), cpu64_seconds=cpu_s, gpu64_seconds=gpu64_s, gpu32_seconds=gpu32_s,
                cpu64_evals_per_s=n / cpu_s, gpu64_evals_per_s=n / gpu64_s, gpu32_evals_per_s=n / gpu32_s,
                speedup32_vs_one_core=cpu_s / gpu32_s, speedup32_vs_gpu64=gpu64_s / gpu32_s,
                ambiguous_fraction=float(amb.mean()), mismatches_after_fallback=mism, nund_mismatches_after_fallback=mism_n,
                status_counts=np.bincount(st_g, minlength=4).tolist())


if __name__ == '__main__':
    import cupy as cp
    bundle, out = sys.argv[1], sys.argv[2]
    p = cp.cuda.runtime.getDeviceProperties(0)
    res = dict(gpu=p['name'].decode(), results=[])
    for nm in sys.argv[3:]:
        r = run(bundle, nm); res['results'].append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
