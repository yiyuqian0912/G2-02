"""Phase 2 GPU PoC: the dominant operation of TPE is the per-(region, item, component) pencil predicate
(engine eval_comp2).  Here the same predicate, same float64 arithmetic order and same margins, runs as a CuPy
RawKernel over a batch of (region, item) pairs, and is compared decision-by-decision with the CPU (numba) version.

Batch: for a real window, regions = the stage-1 root boxes of the frozen engine (depth 4); items = the window's
maximal pure blocks; components = the window-side (W-mode) components whose gating region G_c contains the root
(other roots/components are skipped in this PoC).  Outputs per (region, item, comp): status (COVER/NOCOVER/PARTIAL)
and the number of undecided edges.  Reports CPU and GPU evaluations per second and the mismatch count."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from numba import njit
from tpl.geom import Geom
from tpl.window import Window
from tpl import engine_b4 as E
from tpl.run2 import window_aspects

KERNEL = r'''
extern "C" __global__ void eval_w(
    const double* __restrict__ P, int kmax,                 // P[(c*kmax + i)*4 + j]
    const long long* __restrict__ codes, int n_items,       // item codes
    const int* __restrict__ pairs_rc, int n_rc,             // (region, comp) pairs, length 2*n_rc
    const double* __restrict__ tlo, const double* __restrict__ thi, int emax,   // per pair, per edge
    const int* __restrict__ nsil, const int* __restrict__ silf, const int* __restrict__ silb,  // per comp
    const long long* o, double invN,
    signed char* __restrict__ status, signed char* __restrict__ nund_out)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_rc * n_items;
    if (tid >= total) return;
    int q = (int)(tid / n_items);
    int it = (int)(tid % n_items);
    int c = pairs_rc[2 * q + 1];
    long long code = codes[it];
    int l = (int)((code >> 48) & 0xFFFF);
    long long bx = (code >> 32) & 0xFFFF, by = (code >> 16) & 0xFFFF, bz = code & 0xFFFF;
    long long s = 1LL << l;
    double qlo0 = (o[0] + bx * s + 0.5) * invN, qlo1 = (o[1] + by * s + 0.5) * invN, qlo2 = (o[2] + bz * s + 0.5) * invN;
    double qhi0 = (o[0] + (bx + 1) * s - 0.5) * invN, qhi1 = (o[1] + (by + 1) * s - 0.5) * invN, qhi2 = (o[2] + (bz + 1) * s - 0.5) * invN;
    double d0 = qhi0 - qlo0, d1 = qhi1 - qlo1, d2 = qhi2 - qlo2;
    int ncorner = (l == 0) ? 1 : 8;
    int nund = 0;
    signed char st = 0;   // COVER unless changed
    int ns = nsil[c];
    for (int k = 0; k < ns; k++) {
        int f = silf[c * emax + k], b = silb[c * emax + k];
        const double* pf = P + ((long long)c * kmax + f) * 4;
        const double* pb = P + ((long long)c * kmax + b) * 4;
        double basef = pf[0] * qlo0 + pf[1] * qlo1 + pf[2] * qlo2 + pf[3];
        double baseb = pb[0] * qlo0 + pb[1] * qlo1 + pb[2] * qlo2 + pb[3];
        double af = pf[0] * d0, bf = pf[1] * d1, gf = pf[2] * d2;
        double ab = pb[0] * d0, bb = pb[1] * d1, gb = pb[2] * d2;
        double tm = 1e308, tM = -1e308;
        for (int cr = 0; cr < ncorner; cr++) {
            double vf = basef, vb = baseb;
            if (cr & 1) { vf += af; vb += ab; }
            if (cr & 2) { vf += bf; vb += bb; }
            if (cr & 4) { vf += gf; vb += gb; }
            double t = -1.0 * vb / vf;
            if (t < tm) tm = t;
            if (t > tM) tM = t;
        }
        double lo = tlo[(long long)q * emax + k], hi = thi[(long long)q * emax + k];
        if (lo >= tM) { st = 1; nund = 0; break; }       // NOCOVER
        if (hi <= tm) continue;
        nund++;
    }
    if (st == 0 && nund > 0) st = 2;                     // PARTIAL
    status[tid] = st; nund_out[tid] = (signed char)nund;
}
'''


@njit(cache=True)
def cpu_eval(P, codes, pairs, tlo, thi, nsil, silf, silb, o, N, status, nund_out, fbuf):
    n_items = codes.shape[0]
    qlo = np.empty(3); qhi = np.empty(3)
    kmax = P.shape[1]
    uf = np.zeros(kmax, np.int64); silfi = np.zeros(silf.shape[1], np.int64); silbi = np.zeros(silf.shape[1], np.int64)
    tmn = np.zeros(silf.shape[1]); tmx = np.zeros(silf.shape[1])
    for q in range(pairs.shape[0]):
        c = pairs[q, 1]
        nuf, nfr = E.prep_comp_w(nsil[c], silf[c], silb[c], uf, silfi, silbi)
        for it in range(n_items):
            lvl = E.item_box(codes[it], o, N, qlo, qhi)
            st, nund, kund = E.eval_comp2(qlo, qhi, lvl == 0, P, c, nuf, 0, uf, nsil[c], silfi, silbi, tlo[q], thi[q],
                                          tmn, tmx, fbuf, -1.0)
            status[q * n_items + it] = 0 if st == E.COVER else (1 if st == E.NOCOVER else 2)
            nund_out[q * n_items + it] = nund


def build_batch(geom, win, d1=4):
    P, K, clo, chi, adj, aoff = geom.tup()
    wlo, whi = win.box()
    SX, SL = win.sentinels(512, 64, 0)
    cand = np.empty((1 << (3 * d1), 3), np.int64); s1 = np.zeros(4, np.int64)
    nc = E.stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, 32, 640, cand, s1)
    wmode, wfront = window_aspects(geom, win)
    C, kmax = P.shape[:2]
    emax = max(int(np.max(np.diff(aoff))), 1)
    nsil = np.zeros(C, np.int64); silf = np.zeros((C, emax), np.int64); silb = np.zeros((C, emax), np.int64)
    silt = np.zeros((C, emax), np.int64)
    for c in range(C):
        if wmode[c]:
            nsil[c] = E.sil_edges(adj, aoff, c, wfront[c], silf[c], silb[c], silt[c])
    pairs = []; TL = []; TH = []
    h = 1.0 / (1 << d1)
    for r in range(nc):
        lo = cand[r] * h; hi = lo + h
        V = np.array([[hi[0] if a else lo[0], hi[1] if b else lo[1], hi[2] if cc else lo[2]]
                      for a in (0, 1) for b in (0, 1) for cc in (0, 1)])
        for c in range(C):
            if not wmode[c] or nsil[c] == 0:
                continue
            f = V @ P[c, :K[c], :3].T + P[c, :K[c], 3]
            fr = wfront[c, :K[c]]
            if np.any(f[:, fr].min(0) > 1e-11) or np.any(f[:, fr].max(0) >= -1e-11):
                continue          # outside or crossing G_c: not part of this PoC batch
            tl = np.full(emax, np.inf); th = np.full(emax, -np.inf)
            for k in range(nsil[c]):
                tl[k], th[k] = E.tau_range(V, 8, P, c, silf[c, k], silb[c, k], -np.inf, np.inf, -1.0)
            pairs.append((r, c)); TL.append(tl); TH.append(th)
    codes, _ = win.top_items()
    return (P, codes, np.array(pairs, np.int64).reshape(-1, 2), np.array(TL).reshape(-1, emax), np.array(TH).reshape(-1, emax),
            nsil, silf, silb, emax)


def run(bundle, name, reps=3):
    import cupy as cp
    z = np.load(os.path.join(bundle, name + '.npz'))
    geom = Geom(z['P'], z['K']); win = Window(z['cls'], z['origin'], int(z['N']))
    P, codes, pairs, TL, TH, nsil, silf, silb, emax = build_batch(geom, win)
    n = len(pairs) * len(codes)
    if n == 0:
        return dict(name=name, pairs=0)
    o = win.origin.astype(np.int64)
    st_c = np.zeros(n, np.int8); nu_c = np.zeros(n, np.int8); fbuf = np.zeros((P.shape[1], 8))
    cpu_eval(P, codes[:1], pairs[:1], TL[:1], TH[:1], nsil, silf, silb, o, float(win.N), st_c, nu_c, fbuf)   # JIT
    t = time.process_time()
    cpu_eval(P, codes, pairs, TL, TH, nsil, silf, silb, o, float(win.N), st_c, nu_c, fbuf)
    cpu_s = time.process_time() - t
    k = cp.RawKernel(KERNEL, 'eval_w', options=('--fmad=false',))
    dP = cp.asarray(P); dcodes = cp.asarray(codes); dpairs = cp.asarray(pairs.astype(np.int32).ravel())
    dTL = cp.asarray(np.ascontiguousarray(TL)); dTH = cp.asarray(np.ascontiguousarray(TH))
    dns = cp.asarray(nsil.astype(np.int32)); dsf = cp.asarray(silf.astype(np.int32).ravel()); dsb = cp.asarray(silb.astype(np.int32).ravel())
    do = cp.asarray(o); dst = cp.zeros(n, cp.int8); dnu = cp.zeros(n, cp.int8)
    args = (dP, np.int32(P.shape[1]), dcodes, np.int32(len(codes)), dpairs, np.int32(len(pairs)), dTL, dTH, np.int32(emax),
            dns, dsf, dsb, do, np.float64(1.0 / win.N), dst, dnu)
    grid = ((n + 255) // 256,)
    k(grid, (256,), args); cp.cuda.Device().synchronize()       # compile + warm
    ev = cp.cuda.Event(); ev2 = cp.cuda.Event()
    ts = []
    for _ in range(reps):
        ev.record(); k(grid, (256,), args); ev2.record(); ev2.synchronize()
        ts.append(cp.cuda.get_elapsed_time(ev, ev2) / 1000)
    gpu_s = min(ts)
    st_g = cp.asnumpy(dst); nu_g = cp.asnumpy(dnu)
    mism = int((st_g != st_c).sum()); mism_n = int((nu_g != nu_c).sum())
    return dict(name=name, side=int(win.L), regions=int(len(set(pairs[:, 0].tolist()))), region_comp_pairs=int(len(pairs)),
                items=int(len(codes)), evaluations=int(n), cpu_seconds=cpu_s, gpu_seconds=gpu_s,
                cpu_evals_per_s=n / cpu_s, gpu_evals_per_s=n / gpu_s, speedup_vs_one_core=cpu_s / gpu_s,
                status_mismatches=mism, nund_mismatches=mism_n,
                status_counts_cpu=np.bincount(st_c, minlength=3).tolist())


if __name__ == '__main__':
    import cupy as cp
    bundle, out = sys.argv[1], sys.argv[2]
    p = cp.cuda.runtime.getDeviceProperties(0)
    res = dict(gpu=p['name'].decode(), sms=p['multiProcessorCount'], results=[])
    for name in sys.argv[3:]:
        r = run(bundle, name)
        res['results'].append(r)
        print(json.dumps(r), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
