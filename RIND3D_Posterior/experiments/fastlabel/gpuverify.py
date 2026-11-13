"""Batched certified point-source verification on the GPU (host side). Kernel: verify.cu.

`WindowVerifier(fac, ctx, S, w, s, cls)` holds the device-resident observation of one label (window w, source
channel s): stored packed lit and SOLID crops and the 8^3 leaf codes. `test_batch(points)` decides, for every
candidate source, EXACTLY whether it is in F_W^j:
1. legality (certified clearance, rind3d_1k.physical.legal_source) and exclusion of the closed window box (host);
2. shadow rows on the GPU (directed intervals), exact host rows for flagged sources;
3. one verify launch over (source, 16^3 brick) work items: FP32 filter -> FP64 intervals -> UNKNOWN;
4. exact host decision (native GMP predicate) of UNKNOWN voxels of otherwise passing sources.
Returns per source (ok, violated receivers (n, 3) window xyz or None, info).

The module is loaded into the CUDA context of the factory's Runtime (one context per process)."""
import ctypes as C
import contextlib
import time
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
_MOD = {}


def _module(rt):
    import os
    defs = os.environ.get("FASTLABEL_VERIFY_DEFINES", "")       # validation builds only, e.g. "E32=1e-3f;UNK_BAND=1e-4f"
    m = _MOD.get((id(rt), defs))
    if m is None:
        code = (HERE / "verify.cu").read_bytes()
        if defs:
            code = "".join(f"#define {d.replace('=', ' ', 1)}\n" for d in defs.split(";") if d).encode() + code
        ptx, log = rt.compile(code)                       # NVRTC with --fmad=false --ftz=false (Runtime options)
        mod = C.c_void_p(); rt.ck("cuModuleLoadData", C.byref(mod), C.create_string_buffer(ptx))
        fns = {}
        for name in ("shadow_rows", "verify"):
            f = C.c_void_p(); rt.ck("cuModuleGetFunction", C.byref(f), mod, name.encode()); fns[name] = f
        d = rt.driver
        d.cuMemsetD32_v2.argtypes = [C.c_uint64, C.c_uint, C.c_size_t]
        d.cuMemsetD8_v2.argtypes = [C.c_uint64, C.c_ubyte, C.c_size_t]
        d.cuCtxSynchronize.argtypes = []
        m = _MOD[(id(rt), defs)] = dict(mod=mod, fns=fns, rt=rt, defines=defs)
    return m


def _launch(rt, fn, grid, block, args):
    vals = []
    for x in args:
        if isinstance(x, (C.c_uint64, C.c_int, C.c_float, C.c_double)):
            vals.append(x)
        elif isinstance(x, float):
            vals.append(C.c_float(x))
        else:
            vals.append(C.c_int(int(x)))
    params = (C.c_void_p * len(vals))(*(C.addressof(v) for v in vals))
    rt.ck("cuLaunchKernel", fn, *grid, *block, 0, None, params, None)


class DevBuf:
    """Growable device buffer."""
    def __init__(self, rt):
        self.rt = rt; self.p = None; self.n = 0

    def ensure(self, nbytes):
        nbytes = max(int(nbytes), 16)
        if self.n < nbytes:
            if self.p is not None:
                self.rt.free(self.p)
            self.p = self.rt.alloc(nbytes); self.n = nbytes
        return self.p

    def put(self, a):
        a = np.ascontiguousarray(a); p = self.ensure(a.nbytes)
        if a.nbytes:
            self.rt.ck("cuMemcpyHtoD_v2", p, a.ctypes.data, a.nbytes)
        return p

    def zero(self, nbytes):
        p = self.ensure(nbytes); self.rt.driver.cuMemsetD32_v2(p, 0, max(int(nbytes) // 4, 1)); return p

    def get(self, shape, dtype, count=None):
        a = np.empty(shape, dtype)
        if a.nbytes:
            self.rt.ck("cuMemcpyDtoH_v2", a.ctypes.data, self.p, a.nbytes)
        return a

    def close(self):
        if self.p is not None:
            self.rt.free(self.p); self.p = None; self.n = 0


@njit(cache=True)
def leaf_codes(cls, solid_mode):
    """cls[x, y, z] in {0 dark FREE, 1 lit FREE, 2 SOLID}. Code per 8^3 leaf (index (lz * nl + ly) * nl + lx):
    0 every care voxel expects 'outside', 1 every care voxel expects 'inside', 2 mixed, 3 no care voxel."""
    L = cls.shape[0]; nl = L // 8
    out = np.empty(nl * nl * nl, np.int8)
    for lz in range(nl):
        for ly in range(nl):
            for lx in range(nl):
                n0 = 0; n1 = 0
                for z in range(lz * 8, lz * 8 + 8):
                    for y in range(ly * 8, ly * 8 + 8):
                        for x in range(lx * 8, lx * 8 + 8):
                            c = cls[x, y, z]
                            if solid_mode:
                                if c == 2:
                                    n1 += 1
                                else:
                                    n0 += 1
                            elif c == 0:
                                n1 += 1
                            elif c == 1:
                                n0 += 1
                out[(lz * nl + ly) * nl + lx] = 3 if n0 + n1 == 0 else (0 if n1 == 0 else (1 if n0 == 0 else 2))
    return out


def _bricks(lcode, L):
    """16^3 bricks (index (bz * nb + by) * nb + bx) that contain a care voxel."""
    nl = L // 8; nbr = L // 16
    c = lcode.reshape(nl, nl, nl)                                    # [lz, ly, lx]
    has = (c != 3).reshape(nbr, 2, nbr, 2, nbr, 2).any(axis=(1, 3, 5))  # [bz, by, bx]
    return np.flatnonzero(has.ravel()).astype(np.int32)


def _host_rows(ctx, s, rows_per_src, rstart):
    """Exact rational shadow rows (fastplanes, identical to polyhedra.shadow_planes) in the kernel layout."""
    import fastplanes
    planes = [fastplanes.shadow_planes_fast(b, np.asarray(s, "<f8")) for b in ctx.bodies]
    R64 = np.zeros((rows_per_src, 8)); R32 = np.full((rows_per_src, 4), np.nan, np.float32)
    cnt = np.zeros(len(planes), np.int32)
    for b, pl in enumerate(planes):
        rows, _ = fastplanes.encode_fast([pl])
        n = len(rows); cnt[b] = n
        R64[rstart[b]:rstart[b] + n] = rows
        mid = 0.5 * rows[:, 0::2] + 0.5 * rows[:, 1::2]
        w = rows[:, 1::2] - rows[:, 0::2]
        f = mid.astype(np.float32)
        f[~np.all(w <= 2.0 ** -30, axis=1)] = np.nan
        R32[rstart[b]:rstart[b] + n] = f
    return R64, R32, cnt


class GeometryRows:
    """Per-context device copy of the exact body-row enclosures and the per-source row layout."""
    def __init__(self, rt, ctx):
        import fastplanes
        rows, offs = fastplanes.encode_fast(ctx.bodies)
        self.rows, self.offs = rows, offs.astype(np.int32)
        K = np.diff(self.offs)
        if np.any(K > 64) or len(K) > 64:
            raise ValueError("verify kernel: at most 64 bodies and 64 faces per body")
        mx = K + (K // 2) * (K - K // 2)                       # n_neg + n_zero + n_neg * n_pos <= K + floor(K^2 / 4)
        self.rstart = np.concatenate([[0], np.cumsum(mx)[:-1]]).astype(np.int32)
        self.rows_per_src = int(mx.sum())
        self.nb = len(K)
        self.d_rows = rt.upload(np.ascontiguousarray(rows, "<f8")); self.d_offs = rt.upload(self.offs)
        self.d_rstart = rt.upload(self.rstart)
        # geometry bodies as one "source" (solid mode): rows are already normalised exact enclosures
        R64 = np.zeros((self.rows_per_src, 8)); R32 = np.full((self.rows_per_src, 4), np.nan, np.float32)
        for b in range(self.nb):
            r = rows[self.offs[b]:self.offs[b + 1]]; n = len(r)
            R64[self.rstart[b]:self.rstart[b] + n] = r
            f = (0.5 * r[:, 0::2] + 0.5 * r[:, 1::2]).astype(np.float32)
            f[~np.all(r[:, 1::2] - r[:, 0::2] <= 2.0 ** -30, axis=1)] = np.nan
            R32[self.rstart[b]:self.rstart[b] + n] = f
        self.geo64, self.geo32, self.geocnt = R64, R32, np.diff(self.offs).astype(np.int32)


class WindowVerifier:
    def __init__(self, fac, ctx, S, w, s, cls, kmax=1024, ucap=4096, lock=None, check_solid=True, mem_budget=96 << 20):
        from fractions import Fraction
        from rind3d.lite_acceptance import DRIVER_RADIUS
        self.fac = fac; self.rt = rt = fac._cuda(); self.ctx = ctx; self.m = _module(rt); self.lock = lock
        self.o = np.asarray(S.origin(w), np.int64); self.L = int(S.W[w]["L"]); self.N = 1024
        r2 = float(DRIVER_RADIUS) ** 2
        if Fraction(r2) < Fraction(DRIVER_RADIUS) ** 2:
            r2 = float(np.nextafter(r2, np.inf))
        self.r2 = r2                                             # upper bound of r^2 (exact for r = 2^-26)
        key = ("georows", ctx.geometry_id)
        g = _MOD.get(key)
        if g is None or g[0] is not rt:
            g = (rt, GeometryRows(rt, ctx)); _MOD[key] = g
        self.G = g[1]
        def raw(j):
            o, n = S._pos[(w, j)]
            return np.frombuffer(S.dz.decompress(S.blob[o:o + n]), np.uint8)
        self.cls = cls
        self.d_lit = rt.upload(raw(s)); self.d_sol = rt.upload(raw(-1))
        self.lcode = leaf_codes(cls, False); self.d_lcode = rt.upload(self.lcode)
        self.bricks = _bricks(self.lcode, self.L)
        self.ucap = int(ucap)
        self.rcap = 2 * (self.L // 16) ** 3
        per_src = self.G.rows_per_src * 80 + len(self.bricks) * 8 + self.rcap * 4 + self.ucap * 4
        self.kmax = int(min(kmax, max(16, mem_budget // per_src)))           # device memory guard per worker
        self.buf = {k: DevBuf(rt) for k in ("src", "R64", "R32", "rcnt", "status", "work", "nmis", "recn", "rec", "nunk", "unk")}
        self.stats = dict(batches=0, points=0, legal_fail=0, box_fail=0, host_legal=0, host_rows=0, unknown_voxels=0,
                          unknown_sources=0, work_items=0, gpu_s=0.0, legal_s=0.0, host_s=0.0, exact_s=0.0)
        import json
        sj = json.loads((S.dir / "privileged" / "sources.json").read_text())
        self.true_source = np.array([float.fromhex(v) for v in sj["sources_hex"][s]])
        if check_solid:
            self.solid_check()

    def close(self):
        for b in self.buf.values():
            b.close()
        for p in (self.d_lit, self.d_sol, self.d_lcode):
            self.rt.free(p)

    # ---------------------------------------------------------------- core launch
    def _run(self, K, work, rows=None, solid_mode=False, dump=None, lcode_ptr=None):
        """rows: None -> rows (and legality flags) built on the GPU from buf['src']; else host (R64, R32, rcnt)."""
        rt = self.rt; G = self.G; b = self.buf; fns = self.m["fns"]
        rps = G.rows_per_src
        if rows is None:
            b["R64"].ensure(K * rps * 64); b["R32"].ensure(K * rps * 16); b["rcnt"].ensure(K * G.nb * 4)
            b["status"].zero(K * 4)
            _launch(rt, fns["shadow_rows"], (K, (G.nb + 31) // 32, 1), (32, 1, 1),
                    [G.d_rows, G.d_offs, G.nb, b["src"].p, K, G.d_rstart, rps, b["R64"].p, b["R32"].p, b["rcnt"].p,
                     b["status"].p, C.c_double(self.r2)])
        else:
            R64, R32, rcnt = rows
            b["R64"].put(R64); b["R32"].put(R32); b["rcnt"].put(rcnt)
        b["work"].put(work)
        b["nmis"].zero(K * 4); b["recn"].zero(K * 4); b["nunk"].zero(K * 4)
        b["rec"].ensure(K * self.rcap * 4); b["unk"].ensure(K * self.ucap * 4)
        nwork = len(work) // 2
        if nwork:
            _launch(rt, fns["verify"], (nwork, 1, 1), (64, 1, 1),
                    [b["R32"].p, b["R64"].p, G.d_rstart, b["rcnt"].p, rps, G.nb, b["work"].p, nwork, self.d_lit, self.d_sol,
                     int(solid_mode), lcode_ptr if lcode_ptr is not None else self.d_lcode, int(self.o[0]), int(self.o[1]),
                     int(self.o[2]), self.L, self.N, b["nmis"].p, b["recn"].p, b["rec"].p, self.rcap, b["nunk"].p,
                     b["unk"].p, self.ucap, dump if dump is not None else C.c_uint64(0)])
        status = b["status"].get(K, np.int32) if rows is None else np.zeros(K, np.int32)
        return status, b["nmis"].get(K, np.int32), b["recn"].get(K, np.int32), b["nunk"].get(K, np.int32)

    def _lists(self, K, need_rec, need_unk):
        b = self.buf
        rec = b["rec"].get((K, self.rcap), np.int32) if np.any(need_rec) else None
        unk = b["unk"].get((K, self.ucap), np.int32) if np.any(need_unk) else None
        return rec, unk

    def _work(self, K):
        nbk = len(self.bricks)
        work = np.empty((K * nbk, 2), np.int32)
        work[:, 0] = np.repeat(np.arange(K, dtype=np.int32), nbk); work[:, 1] = np.tile(self.bricks, K)
        return work.ravel()

    def solid_check(self):
        """Stored SOLID mask == certified membership of the canonical geometry at every voxel centre of the window."""
        G = self.G
        lc = leaf_codes(self.cls, True); d_lc = self.rt.upload(lc)
        nbr = self.L // 16
        work = np.stack([np.zeros(nbr ** 3, np.int32), np.arange(nbr ** 3, dtype=np.int32)], 1).ravel()
        with (self.lock if self.lock is not None else contextlib.nullcontext()):
            _, nmis, recn, nunk = self._run(1, work, rows=(G.geo64, G.geo32, G.geocnt), solid_mode=True, lcode_ptr=d_lc)
            _, unk = self._lists(1, [False], nunk > 0)
        self.rt.free(d_lc)
        if nmis[0]:
            raise AssertionError("stored SOLID crop disagrees with the canonical geometry")
        if nunk[0]:
            if nunk[0] > self.ucap:
                raise RuntimeError("solid check: UNKNOWN overflow")
            idx = unk[0, :nunk[0]]
            cpu = self.ctx.bind(self.true_source)           # membership does not depend on the source
            _, mm, _ = cpu.native.points(np.ascontiguousarray(self._centres(idx), "<f8"), False)
            x, y, z = self._vox(idx).T
            if np.any(mm < 0) or np.any((mm == 1) != (self.cls[x, y, z] == 2)):
                raise AssertionError("stored SOLID crop disagrees with exact membership")
        self.stats["solid_unknown"] = int(nunk[0])

    def _centres(self, idx):
        return (self._vox(idx) + self.o + 0.5) / self.N

    def _vox(self, idx):
        L = self.L
        z, rem = np.divmod(np.asarray(idx, np.int64), L * L); y, x = np.divmod(rem, L)
        return np.stack([x, y, z], 1)

    # ---------------------------------------------------------------- public API
    def test_batch(self, pts, dump=False):
        """Exact membership in F of every candidate. Returns [(ok, violated window xyz (n, 3) or None, reason)]."""
        pts = np.ascontiguousarray(pts, "<f8").reshape(-1, 3); n = len(pts)
        out = [None] * n
        lo = self.o / self.N; hi = (self.o + self.L) / self.N
        inbox = np.all((pts >= lo) & (pts <= hi), axis=1); cube = np.all((pts > 0) & (pts < 1), axis=1)
        ids = []
        for i in range(n):
            if inbox[i]:
                out[i] = (False, None, "box"); self.stats["box_fail"] += 1
            elif not cube[i]:
                out[i] = (False, None, "illegal"); self.stats["legal_fail"] += 1
            else:
                ids.append(i)
        self.stats["points"] += n
        for c0 in range(0, len(ids), self.kmax):
            self._batch(pts, ids[c0:c0 + self.kmax], out, dump)
        return out

    def _batch(self, pts, ids, out, dump):
        from rind3d_1k.physical import legal_source
        K = len(ids)
        if K == 0:
            return
        P = pts[ids]; G = self.G; nbk = len(self.bricks)
        self.stats["batches"] += 1; self.stats["work_items"] += K * nbk
        d_dump = None
        if dump:
            self._dump = DevBuf(self.rt); d_dump = self._dump.ensure(self.L ** 3)
            self.rt.ck("cuMemsetD8_v2", d_dump, 3, self.L ** 3)
        t = time.perf_counter()
        with (self.lock if self.lock is not None else contextlib.nullcontext()):
            self.buf["src"].put(P)
            status, nmis, recn, nunk = self._run(K, self._work(K), dump=d_dump)
            plain = status == 0
            rec, unk = self._lists(K, plain & (nmis > 0), plain & (nmis == 0) & (nunk > 0))
            if dump:
                self.dump = self._dump.get(self.L ** 3, np.uint8)
        self.stats["gpu_s"] += time.perf_counter() - t
        res = {}
        for kk in np.flatnonzero(plain):
            res[kk] = (int(nmis[kk]), rec[kk, :min(recn[kk], self.rcap)] if nmis[kk] else None, int(nunk[kk]),
                       unk[kk] if (nmis[kk] == 0 and nunk[kk]) else None)
        # flagged sources: host legality (exact legal_source) and, if legal and needed, exact host rows
        redo = []
        for kk in np.flatnonzero(~plain):
            t1 = time.perf_counter(); ok = legal_source(self.ctx, P[kk]) is not None; self.stats["legal_s"] += time.perf_counter() - t1
            self.stats["host_legal"] += 1
            if not ok:
                out[ids[kk]] = (False, None, "illegal"); self.stats["legal_fail"] += 1; continue
            if status[kk] & 5:
                if status[kk] & 4:
                    raise AssertionError("legal source flagged as inside a closed body")
                redo.append(kk)
            else:
                res[kk] = (int(nmis[kk]), None, int(nunk[kk]), None)
                if nmis[kk] or nunk[kk]:
                    redo.append(kk)                      # lists were not downloaded: recompute (rare)
        if redo:
            t1 = time.perf_counter()
            rps = G.rows_per_src; m = len(redo)
            R64 = np.zeros((m * rps, 8)); R32 = np.zeros((m * rps, 4), np.float32); rc = np.zeros(m * G.nb, np.int32)
            for j, kk in enumerate(redo):
                if status[kk] & 1:
                    a, b_, c = _host_rows(self.ctx, P[kk], rps, G.rstart); self.stats["host_rows"] += 1
                else:
                    a, b_, c = self._device_rows(P[kk])
                R64[j * rps:(j + 1) * rps] = a; R32[j * rps:(j + 1) * rps] = b_; rc[j * G.nb:(j + 1) * G.nb] = c
            self.stats["host_s"] += time.perf_counter() - t1
            with (self.lock if self.lock is not None else contextlib.nullcontext()):
                if dump:
                    self.rt.ck("cuMemsetD8_v2", d_dump, 3, self.L ** 3)
                _, m2, r2, u2 = self._run(m, self._work(m), rows=(R64, R32, rc), dump=d_dump)
                rec2, unk2 = self._lists(m, m2 > 0, (m2 == 0) & (u2 > 0))
                if dump and redo[0] == 0:
                    self.dump = self._dump.get(self.L ** 3, np.uint8)
            for j, kk in enumerate(redo):
                res[kk] = (int(m2[j]), rec2[j, :min(r2[j], self.rcap)] if m2[j] else None, int(u2[j]),
                           unk2[j] if (m2[j] == 0 and u2[j]) else None)
        if dump:
            self._dump.close()
        for kk, (nm, rl, nu, ul) in res.items():
            i = ids[kk]
            if nm > 0:
                out[i] = (False, self._vox(np.unique(rl)), "obs"); continue
            if nu > 0:
                self.stats["unknown_sources"] += 1; self.stats["unknown_voxels"] += nu
                if nu > self.ucap:
                    raise RuntimeError("UNKNOWN list overflow")
                t1 = time.perf_counter()
                idx = ul[:nu]
                cpu = self.ctx.bind(pts[i])
                yy, _, _ = cpu.native.points(np.ascontiguousarray(self._centres(idx), "<f8"), False)
                self.stats["exact_s"] += time.perf_counter() - t1
                if np.any(yy < 0):
                    raise RuntimeError("native exact predicate did not certify")
                x, y, z = self._vox(idx).T
                bad = idx[yy.astype(np.int64) != self.cls[x, y, z]]   # care voxels are FREE: cls 0 dark, 1 lit
                if len(bad):
                    out[i] = (False, self._vox(bad), "obs"); continue
            out[i] = (True, None, "pass")

    def _device_rows(self, s):
        """Download the GPU rows of one source (used only when its lists must be recomputed)."""
        G = self.G; rps = G.rows_per_src
        self.buf["src"].put(np.asarray(s, "<f8")[None])
        with (self.lock if self.lock is not None else contextlib.nullcontext()):
            self._run(1, np.zeros(0, np.int32))
            R64 = self.buf["R64"].get((rps, 8), np.float64); R32 = self.buf["R32"].get((rps, 4), np.float32)
            rc = self.buf["rcnt"].get(G.nb, np.int32)
        return R64, R32, rc
