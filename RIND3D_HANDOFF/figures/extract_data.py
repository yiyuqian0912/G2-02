"""Figure data for the RIND3D design document (labeller venv, CUDA). Writes compact arrays to figures/data/.

Every quantity shown in the figures comes from real data or real solver runs:
* scene 000007 of the 1K release (train split, barriers family, K = 4): polytopes (float vertices from the exact
  half-space rows, display only), sources, windows, stored crops (re-rendered with the certified CUDA renderer and
  compared bit for bit), rays of one source into the main window (float segment clipping, checked against the stored
  lit / dark bits);
* the main task (window 12, source 0): stored FastLabel label, exact tau-pencil label, and an instrumented FastLabel
  run (production code; stage-2 outer bounds per round, sampled candidates and their exact verification results);
* the response-type windows (window 6: all-dark, solid-separated, boundary-mixed; window 12 source 2: all-lit);
* the 100K local-only renderer: slot 0 of contract v1 rendered into a scratch container (proposal logs, work
  counters), plus one full-world z-plane of a source rendered with the certified renderer for comparison.

  $RIND3D_LABEL_PY RIND3D_HANDOFF/figures/extract_data.py --container <slot_000000.npz> [--out RIND3D_HANDOFF/figures/data]"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())


def polytopes(P, K):
    """Float vertices and hull triangles of each convex body {x : a.x + d <= 0} (display only)."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    out = []
    for b in range(len(K)):
        H = np.asarray(P[b, :int(K[b])], float)
        A, d = H[:, :3], H[:, 3]
        nrm = np.linalg.norm(A, axis=1)
        r = linprog(np.r_[0, 0, 0, -1.0], A_ub=np.c_[A, nrm], b_ub=-d, bounds=[(None, None)] * 3 + [(0, None)])
        V = HalfspaceIntersection(H, r.x[:3]).intersections
        hull = ConvexHull(V)
        out.append((V, hull.simplices))
    return out


def segment_hit(P, K, s, x):
    """Smallest t in [0, 1] where the segment s + t (x - s) enters a body (float Cyrus-Beck), else None."""
    best = None
    dvec = x - s
    for b in range(len(K)):
        H = np.asarray(P[b, :int(K[b])], float)
        num = -(H[:, :3] @ s + H[:, 3]); den = H[:, :3] @ dvec
        t0, t1 = 0.0, 1.0; ok = True
        for n_, d_ in zip(num, den):
            if abs(d_) < 1e-300:
                if n_ < 0:
                    ok = False; break
                continue
            t = n_ / d_
            if d_ < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                ok = False; break
        if ok and (best is None or t0 < best):
            best = t0
    return best


def render_tiles(fac, bundle, policy, src, origins, N=1024):
    out = []
    S = fac.source(bundle, np.asarray(src, "<f8"), "cuda_certified", policy)
    try:
        for i in range(0, len(origins), 128):
            out += S.packed_batch(np.asarray(origins[i:i + 128], np.int32), 64, N)
    finally:
        S.close()
    return out


def window_render(fac, bundle, policy, src, origin, L, N=1024):
    """Certified lit and SOLID fields of a window, bool [z, y, x]."""
    o = np.asarray(origin, np.int64); n = L // 64
    offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
    res = render_tiles(fac, bundle, policy, src, [o + np.array(t) * 64 for t in offs], N)
    lit = np.zeros((L, L, L), bool); sol = np.zeros((L, L, L), bool)
    for (a, b, c), (pl, ps) in zip(offs, res):
        sl = np.s_[c * 64:(c + 1) * 64, b * 64:(b + 1) * 64, a * 64:(a + 1) * 64]
        lit[sl] = np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
        sol[sl] = np.unpackbits(np.frombuffer(ps, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
    return lit, sol


def response_type(lit, free):
    if not np.any(lit & free):
        return "all_dark"
    if np.all(lit[free]):
        return "all_lit"
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3; a[ax] = slice(0, -1); b[ax] = slice(1, None)
        if np.any(free[tuple(a)] & free[tuple(b)] & (lit[tuple(a)] != lit[tuple(b)])):
            return "boundary_mixed"
    return "solid_separated"


def cells_to_q(cell, w):
    u, inv = np.unique(np.asarray(cell, np.int64), return_inverse=True)
    m = np.bincount(inv, weights=np.asarray(w, float))
    return u, m


def instrumented_fastlabel(task):
    """Run the production label_window_fast and record stage-2 outer bounds, draws and verification results."""
    from rind3d_posterior.fastlabel import engine as E, gpuverify as GV, pipeline as PL
    ev = []
    s2, BS, WV = E.stage2, PL.BSampler, GV.WindowVerifier

    def stage2(*a, **k):
        r = s2(*a, **k)
        u, hi = cells_to_q(r["cell"], r["hi"]); _, lo = cells_to_q(r["cell"], r["lo"])
        ev.append(("stage2", dict(cells=u, hi=hi, lo=lo, records=len(r["cell"]), undet=int(r["stats"]["undet"]))))
        return r

    class Sampler(BS):
        def draw(self, n):
            p = BS.draw(self, n); ev.append(("draw", np.asarray(p))); return p

    class Verifier(WV):
        def test_batch(self, pts, dump=False):
            r = WV.test_batch(self, pts, dump); ev.append(("test", np.array([bool(x[0]) for x in r]))); return r

    E.stage2, PL.BSampler, GV.WindowVerifier = stage2, Sampler, Verifier
    try:
        lab = PL.label_window_fast(task)
    finally:
        E.stage2, PL.BSampler, GV.WindowVerifier = s2, BS, WV
    return lab, ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--container", required=True)
    ap.add_argument("--out", default=str(HERE / "data")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rel = Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "release_1k_v1"
    from rind3d.geometry import load_geometry
    from rind3d_posterior.fastlabel import window_from_1k, window_from_100k, POLICY
    from rind3d_posterior.fastlabel.pipeline import process_factory
    from rind3d_posterior.fastlabel.window import window_classes
    import rind3d_posterior as RP
    fac = process_factory(); log = {}
    sc = CFG["scene"]; d = rel / "scenes" / f"{sc:06d}"
    W = np.load(d / "public" / "windows.npy"); g = np.load(d / "privileged" / "geometry_rows.npz"); P, K = g["P"], g["K"]
    src = np.array([[float.fromhex(v) for v in q] for q in json.loads((d / "privileged" / "sources.json").read_text())["sources_hex"]])
    bundle = load_geometry(d / "privileged" / "geometry.npz")
    scene_json = json.loads((d / "scene.json").read_text())

    # ---- scene geometry
    polys = polytopes(P, K)
    V = np.concatenate([p[0] for p in polys]); voff = np.cumsum([0] + [len(p[0]) for p in polys])
    F = np.concatenate([p[1] for p in polys]); foff = np.cumsum([0] + [len(p[1]) for p in polys])
    np.savez_compressed(out / "scene.npz", V=V, voff=voff, F=F, foff=foff, sources=src,
                        windows=np.array([[r["window"], r["x"], r["y"], r["z"], r["L"], r["mask"], r["k"]] for r in W], np.int64))
    log["scene"] = dict(slot=sc, family=scene_json.get("family"), k=int(W[0]["k"]), bodies=len(K), windows=len(W))

    # ---- crops of the figure windows, re-rendered and compared
    tasks = {}
    crops = {}
    for key, (w, s) in CFG["channels"].items():
        t = window_from_1k(d, w, s); L = t.L
        lit = np.unpackbits(t.lit_bits, count=L ** 3).astype(bool).reshape(L, L, L)
        sol = np.unpackbits(t.solid_bits, count=L ** 3).astype(bool).reshape(L, L, L)
        rl, rs = window_render(fac, bundle, POLICY, src[s], t.origin, L)
        crops[key] = (w, s, lit, sol)
        tasks[key] = dict(window=w, source=s, L=L, origin=list(t.origin), type=response_type(lit, ~sol),
                          lit_frac=float(lit[~sol].mean()), solid_frac=float(sol.mean()),
                          rerender_mismatch=int(np.sum(rl != lit) + np.sum(rs != sol)))
        lab = json.loads(str(np.load(d / "labels" / "tasks" / f"w{w:03d}_s{s}.npz")["meta"])) \
            if (d / "labels" / "tasks" / f"w{w:03d}_s{s}.npz").exists() else None
        tasks[key]["label"] = None if lab is None else {k: lab.get(k) for k in ("status", "path", "n_cells", "wall", "tv_bound")}
    np.savez_compressed(out / "crops.npz", **{f"{k}_lit": np.packbits(v[2].ravel()) for k, v in crops.items()},
                        **{f"{k}_solid": np.packbits(v[3].ravel()) for k, v in crops.items()},
                        **{f"{k}_meta": np.array([v[0], v[1], v[2].shape[0]]) for k, v in crops.items()})
    log["channels"] = tasks

    # ---- rays of the main source into the main window (checked against the stored bits)
    mk = CFG["main"]; w, s = CFG["channels"][mk]; t = window_from_1k(d, w, s); L = t.L
    lit = crops[mk][2]; sol = crops[mk][3]; rng = np.random.default_rng(7)
    rays = []
    for want, n in ((True, CFG["rays_lit"]), (False, CFG["rays_dark"])):
        idx = np.argwhere((lit == want) & ~sol)                                          # [z, y, x]
        for z_, y_, x_ in idx[rng.choice(len(idx), min(4 * n, len(idx)), replace=False)]:
            xr = (np.array([x_, y_, z_]) + np.asarray(t.origin) + 0.5) / t.N
            th = segment_hit(P, K, src[s], xr)
            if (th is None) == want and (th is None or th < 0.999):
                rays.append(np.r_[xr, -1.0 if th is None else th, float(want)])
            if sum(1 for r in rays if r[4] == float(want)) >= n:
                break
    np.save(out / "rays.npy", np.array(rays))

    # ---- main task: stored FastLabel label, exact tau-pencil label, instrumented FastLabel run
    with np.load(d / "labels" / "tasks" / f"w{w:03d}_s{s}.npz") as z:
        fl_cell, fl_lo = z["cell"], z["lo"]; fl_meta = json.loads(str(z["meta"]))
    t0 = time.perf_counter()
    ex = RP.label_window(np.asarray(t.P, float), np.asarray(t.K), window_classes(t), np.asarray(t.origin), t.N, engine="cpu")
    tpe_s = time.perf_counter() - t0
    u_ex, lo_ex = cells_to_q(ex["cell"], ex["lo"]); _, hi_ex = cells_to_q(ex["cell"], ex["hi"])
    lab, ev = instrumented_fastlabel(window_from_1k(d, w, s))
    same_as_release = bool(np.asarray(lab["cell"]).astype(fl_cell.dtype).tobytes() == fl_cell.tobytes()
                           and np.asarray(lab["lo"]).tobytes() == fl_lo.tobytes())
    rounds = []; cur = None
    for kind, v in ev:
        if kind == "stage2":
            cur = dict(B=v, draws=[], tests=[]); rounds.append(cur)
        elif kind == "draw":
            cur["draws"].append(v)
        elif kind == "test":
            cur["tests"].append(v)
    arr = dict(ex_cell=u_ex, ex_lo=lo_ex, ex_hi=hi_ex, fl_cell=fl_cell, fl_lo=fl_lo, true_source=src[s])
    for i, r in enumerate(rounds):
        arr[f"r{i}_cells"] = r["B"]["cells"]; arr[f"r{i}_hi"] = r["B"]["hi"]
        if r["draws"]:
            arr[f"r{i}_pts"] = np.concatenate(r["draws"]); arr[f"r{i}_ok"] = np.concatenate(r["tests"])
    np.savez_compressed(out / "main_task.npz", **arr)
    qa = dict(zip(u_ex.tolist(), (0.5 * (lo_ex + hi_ex) / np.sum(0.5 * (lo_ex + hi_ex))).tolist()))
    qb = dict(zip(*[x.tolist() for x in cells_to_q(fl_cell, fl_lo)])); sb = sum(qb.values())
    tv = 0.5 * sum(abs(qa.get(k, 0) - qb.get(k, 0) / sb) for k in set(qa) | set(qb))
    log["main"] = dict(window=w, source=s, rounds=[dict(records=r["B"]["records"], cells=int(len(r["B"]["cells"])), undet=r["B"]["undet"],
                                                      tested=int(sum(len(x) for x in r["tests"])),
                                                      K=int(sum(int((~x).sum()) for x in r["tests"]))) for r in rounds],
                       meta={k: lab["meta"].get(k) for k in ("path", "round", "n", "K", "stage", "a1", "a2", "n1", "n_max", "delta_r", "n_cells")},
                       status=lab["status"], same_as_release=same_as_release, release_path=fl_meta.get("path"),
                       exact=dict(tv_bound=float(ex["tv_bound"]), cells=int(len(u_ex)), seconds=round(tpe_s, 2),
                                  vol_lo=float(lo_ex.sum()), vol_hi=float(hi_ex.sum())),
                       tv_fastlabel_vs_exact=round(tv, 5))

    # ---- solid-separated reference: broad B (round sizes only)
    ws, ss = CFG["channels"]["solid_separated"]
    t0 = time.perf_counter(); lab2, ev2 = instrumented_fastlabel(window_from_1k(d, ws, ss))
    log["solid_separated_run"] = dict(seconds=round(time.perf_counter() - t0, 2), status=lab2["status"],
                                      rounds=[dict(records=v["records"], cells=int(len(v["cells"]))) for k, v in ev2 if k == "stage2"])

    # ---- 100K local-only renderer: container logs, work counters, a full-world plane for comparison
    z = np.load(a.container); meta = json.loads(z["meta"].tobytes())
    t100 = window_from_100k(a.container, 0, 0)
    W100 = z["windows"]; srcs100 = np.array([[float.fromhex(str(v)) for v in q] for q in z["sources_hex"]])
    j = int(CFG["e_source"]); wz = int(CFG["e_window"]); r = W100[wz]
    zc = int(r["z"]) + int(r["L"]) // 2; z0 = (zc // 64) * 64
    origins = [np.array([x0, y0, z0]) for y0 in range(0, 1024, 64) for x0 in range(0, 1024, 64)]
    t0 = time.perf_counter(); res = render_tiles(fac, t100.bundle, POLICY, srcs100[j], origins); slab_s = time.perf_counter() - t0
    plane_lit = np.zeros((1024, 1024), bool); plane_sol = np.zeros((1024, 1024), bool)
    for o, (pl, ps) in zip(origins, res):
        a3 = np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").reshape(64, 64, 64)
        b3 = np.unpackbits(np.frombuffer(ps, np.uint8), bitorder="little").reshape(64, 64, 64)
        plane_lit[o[1]:o[1] + 64, o[0]:o[0] + 64] = a3[zc - z0]; plane_sol[o[1]:o[1] + 64, o[0]:o[0] + 64] = b3[zc - z0]
    polys100 = polytopes(z["P"], z["K"])
    V1 = np.concatenate([p[0] for p in polys100]); v1off = np.cumsum([0] + [len(p[0]) for p in polys100])
    np.savez_compressed(out / "local100k.npz", plane_lit=np.packbits(plane_lit), plane_sol=np.packbits(plane_sol), plane_z=zc,
                        windows=np.array([[x["window"], x["x"], x["y"], x["z"], x["L"], x["mask"], x["m"]] for x in W100], np.int64),
                        sources=srcs100, V=V1, voff=v1off, **{k: z[k] for k in z.files if k.startswith("log_")})
    log["local100k"] = dict(slot=meta["slot"], family=meta["family"], k=meta["k"], work=meta["work"], timing=meta["timing"],
                            sides=[{kk: s_.get(kk) for kk in ("Lp", "proposals", "reasons", "Lambda_hat", "work_quota", "work_rate")} for s_ in meta["sides"]],
                            plane=dict(z=zc, source=j, window=wz, tiles=len(origins), seconds=round(slab_s, 2)))
    (out / "LOG.json").write_text(json.dumps(log, indent=1, default=float))
    print(json.dumps({k: (v if k != "channels" else {kk: {x: vv[x] for x in ("type", "rerender_mismatch")} for kk, vv in v.items()})
                      for k, v in log.items() if k != "local100k"}, default=float, indent=1)[:3000])


if __name__ == "__main__":
    main()
