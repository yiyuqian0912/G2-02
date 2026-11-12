"""Prototype: two-phase exact labelling.
Phase 1: exact TPE engine with a receiver SUBSET X' (any subset gives B = F(X') >= F).
Phase 2: exact TPE engine with ALL receivers, restricted to disjoint octree boxes covering B.
Result = F exactly (F <= B <= union of phase-2 roots)."""
import sys, json, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "RIND3D_Dataset1K"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from pathlib import Path
import numpy as np
from rind3d_1k.labels import SceneObs
from rind3d_posterior.tpl import engine_p2 as E
from rind3d_posterior.tpl.run_p2 import window_aspects, summarize
from rind3d_posterior.tpl.geom import Geom
from rind3d_posterior.tpl.window import Window, enc

R = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")


def stage2(geom, win, codes, kls, roots, max_records=1 << 22, ne_cap=1 << 26, nq_cap=1 << 25, max_vertices=1 << 23, bf=None):
    P, K, clo, chi, adj, aoff = geom.tup()
    wmode, wfront = window_aspects(geom, win)
    rec_cell = np.empty(max_records, np.int64); rec_lo = np.empty(max_records); rec_hi = np.empty(max_records)
    rec_kind = np.empty(max_records, np.int8); rec_voff = np.empty(max_records, np.int64); rec_vn = np.empty(max_records, np.int64)
    vpool = np.empty((max_vertices, 3)); vtop = np.zeros(1, np.int64)
    stats = np.zeros(len(E.STAT_NAMES), np.int64)
    n = len(roots)
    cell_nodes = np.zeros(max(n, 1), np.int64); cell_evals = np.zeros(max(n, 1), np.int64)
    dep_evals = np.zeros(64, np.int64); dep_nodes = np.zeros(64, np.int64)
    fbc = np.zeros(geom.C, np.int64); fs = np.zeros(geom.C, np.int64)
    t0 = time.perf_counter()
    args = (P, K, clo, chi, adj, aoff, win.origin, float(win.N), win.L, codes, kls, np.ascontiguousarray(roots, np.int64), n,
            2_000_000, 5_000_000, 400, 24, True, True, 2.0, wmode, wfront, 2.0 ** -26,
            rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
            fbc, fs, dep_evals, dep_nodes, ne_cap, nq_cap)
    if bf is None:
        nrec = E.stage2(*args)
    else:
        from rind3d_posterior.tpl import engine_p2x as EX
        nrec = EX.stage2(*args, *bf, True)
    assert nrec <= max_records
    res = dict(cell=rec_cell[:nrec].copy(), lo=rec_lo[:nrec].copy(), hi=rec_hi[:nrec].copy(), kind=rec_kind[:nrec].copy(),
               voff=rec_voff[:nrec].copy(), vn=rec_vn[:nrec].copy(), vpool=vpool[:vtop[0]].copy(),
               stats=dict(zip(E.STAT_NAMES, stats.tolist())), seconds=time.perf_counter() - t0)
    res.update(summarize(res))
    return res


def stage1(geom, win, d1=4, n_boundary=512, n_class=64, n_learn=32, max_tests=640):
    P, K, clo, chi, adj, aoff = geom.tup()
    wlo, whi = win.box()
    SX, SL = win.sentinels(n_boundary, n_class, 0)
    cand = np.empty((1 << (3 * d1), 3), np.int64); s1 = np.zeros(4, np.int64)
    t0 = time.perf_counter()
    nc = E.stage1(P, K, clo, chi, adj, aoff, wlo, whi, SX, SL, d1, n_learn, max_tests, cand, s1)
    roots = np.zeros((nc, 4), np.int64); roots[:, 0] = d1; roots[:, 1:] = cand[:nc]
    return roots, time.perf_counter() - t0


def subset_items(codes, kls, min_level, frac, seed=0):
    lvl = codes >> 48
    keep = lvl >= min_level
    rng = np.random.default_rng(seed)
    small = np.flatnonzero(~keep)
    pick = small[rng.random(len(small)) < frac]
    keep[pick] = True
    return codes[keep], kls[keep]


def point_items(win, n_bd, n_in, seed=0):
    """Level-0 items only: n_bd random boundary receivers + n_in random interior receivers (no block expansion)."""
    rng = np.random.default_rng(seed)
    bd = win.boundary_mask().reshape(-1)
    c = win.cls.reshape(-1)
    free = c != 2
    ib = np.flatnonzero(bd); ii = np.flatnonzero(free & ~bd)
    ib = rng.choice(ib, min(n_bd, len(ib)), replace=False) if len(ib) else ib
    ii = rng.choice(ii, min(n_in, len(ii)), replace=False) if len(ii) else ii
    idx = np.concatenate([ib, ii])
    v = np.stack(np.unravel_index(idx, win.cls.shape), 1)
    codes = np.array([enc(0, a, b, d) for a, b, d in v], np.int64)
    return codes, c[idx].astype(np.int8)


def cover_roots(res, coarsen=True):
    """Disjoint octree boxes covering every phase-1 record.
    A record is a piece of its 128^3 cell (emit_region splits exactly on cell planes). Large pieces -> the cell box;
    small pieces -> the finest octree boxes (inside the cell) covering the piece's vertex bbox with a 1e-13 margin.
    Complete sibling sets are merged upward (fewer roots for fat regions)."""
    G = 128
    boxes = set()
    for i in range(len(res["cell"])):
        c = int(res["cell"][i]); cx, cy, cz = c // (G * G), (c // G) % G, c % G
        if res["voff"][i] >= 0 and res["vn"][i] > 0:
            V = res["vpool"][res["voff"][i]:res["voff"][i] + res["vn"][i]]
            lo = V.min(0); hi = V.max(0)
            ext = float(np.max(hi - lo))
        else:
            ext = 1.0
        if ext >= 0.25 / G:
            boxes.add((7, cx, cy, cz)); continue
        d = int(min(40, max(7, np.floor(-np.log2(ext * (1 + 1e-6))))))
        s = d - 7
        a_ = np.floor((lo - 1e-13) * (1 << d)).astype(np.int64); b_ = np.floor((hi + 1e-13) * (1 << d)).astype(np.int64)
        cmin = np.array([cx, cy, cz], np.int64) << s; cmax = cmin + (1 << s) - 1
        a_ = np.maximum(a_, cmin); b_ = np.minimum(b_, cmax)
        for ix in range(a_[0], b_[0] + 1):
            for iy in range(a_[1], b_[1] + 1):
                for iz in range(a_[2], b_[2] + 1):
                    boxes.add((d, int(ix), int(iy), int(iz)))
    # drop boxes nested in a coarser box
    def nested(bx, have):
        d, x, y, z = bx
        for dd in range(0, d):
            s = d - dd
            if (dd, x >> s, y >> s, z >> s) in have:
                return True
        return False
    have = set()
    for bx in sorted(boxes):
        pass
    for bx in sorted(boxes, key=lambda t: t[0]):
        if not nested(bx, have):
            have.add(bx)
    if coarsen:
        changed = True
        while changed:
            changed = False
            par = {}
            for (d, x, y, z) in have:
                if d > 4:
                    par.setdefault((d - 1, x >> 1, y >> 1, z >> 1), []).append((d, x, y, z))
            for p, ch in par.items():
                if len(ch) == 8:
                    for c in ch:
                        have.discard(c)
                    have.add(p); changed = True
    return np.array(sorted(have), np.int64).reshape(-1, 4)


def build_filter(res):
    """Outer-bound filter arrays for engine_p2x from phase-1 records (cells: kind 2 = whole cell, 1 = sub-boxes)."""
    G = 128; cs = 1.0 / G
    full = set(); fine = {}
    for i in range(len(res["cell"])):
        c = int(res["cell"][i])
        if res["voff"][i] >= 0 and res["vn"][i] > 0:
            V = res["vpool"][res["voff"][i]:res["voff"][i] + res["vn"][i]]
            lo = V.min(0); hi = V.max(0)
            if float(np.max(hi - lo)) < 0.25 * cs:
                fine.setdefault(c, []).append(np.concatenate([lo - 1e-13, hi + 1e-13]))
                continue
        full.add(c)
    cells = sorted(full | set(fine))
    kind = np.array([2 if c in full else 1 for c in cells], np.int8)
    start = np.zeros(len(cells), np.int64); ln = np.zeros(len(cells), np.int64); boxes = []
    for j, c in enumerate(cells):
        start[j] = len(boxes)
        if c not in full:
            boxes += fine[c]
        ln[j] = len(boxes) - start[j]
    bb = np.array(boxes, float).reshape(-1, 6) if boxes else np.zeros((1, 6))
    return (np.array(cells, np.int64), kind, start, ln, np.ascontiguousarray(bb))


def cell_masses(cell, lo, hi):
    u, inv = np.unique(cell, return_inverse=True)
    return dict(zip(u.tolist(), np.bincount(inv, weights=0.5 * (lo + hi)).tolist()))


def tv(a, b):
    sa, sb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in set(a) | set(b))


def main():
    sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    min_level, frac = int(sys.argv[4]), float(sys.argv[5])
    S = SceneObs(R / sc); cls = S.cls(w, s)
    geom = Geom(np.asarray(S.P, float), np.asarray(S.K)); win = Window(np.asarray(cls, np.uint8), np.asarray(S.origin(w), np.int64), 1024)
    ref = np.load(R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz"); refm = cell_masses(ref["cell"], ref["lo"], ref["hi"])
    t0 = time.perf_counter(); c0 = time.process_time()
    roots1, ts1 = stage1(geom, win)
    codes, kls = win.top_items()
    if min_level >= 0:
        sc_, sk_ = subset_items(codes, kls, min_level, frac)
    else:
        sc_, sk_ = point_items(win, int(frac), int(-min_level))
    r1 = stage2(geom, win, sc_, sk_, roots1)
    mode = sys.argv[7] if len(sys.argv) > 7 else "cover"
    if mode == "bf":
        roots2 = roots1
        r2 = stage2(geom, win, codes, kls, roots1, bf=build_filter(r1))
    else:
        roots2 = cover_roots(r1)
        r2 = stage2(geom, win, codes, kls, roots2)
    wall = time.perf_counter() - t0
    out = dict(sc=sc, w=w, s=s, min_level=min_level, frac=frac, mode=mode, items_full=len(codes), items_sub=len(sc_), stage1=ts1,
               p1=r1["seconds"], p1_evals=r1["stats"]["entry_evals"], p1_nodes=r1["stats"]["nodes"], p1_records=len(r1["cell"]),
               p1_U=r1["U_total"], roots2=len(roots2), roots2_depths=sorted(set(roots2[:, 0].tolist())),
               p2=r2["seconds"], p2_evals=r2["stats"]["entry_evals"], p2_nodes=r2["stats"]["nodes"],
               wall=wall, tv_bound=r2["tv_bound"], tv_vs_ref=tv(cell_masses(r2["cell"], r2["lo"], r2["hi"]), refm),
               L_total=r2["L_total"], ref_L=float(ref["lo"].sum()), L=int(win.L), p1_U_total=r1["U_total"])
    out["cpu_twophase"] = time.process_time() - c0
    if len(sys.argv) > 6 and sys.argv[6] == "ref":
        from rind3d_posterior import label_window
        c1 = time.process_time(); t1 = time.perf_counter()
        rr = label_window(S.P, S.K, cls, S.origin(w), 1024, engine="cpu", nq_cap=1 << 25, ne_cap=1 << 26)
        out.update(cpu_ref=time.process_time() - c1, wall_ref=time.perf_counter() - t1, ref_records=len(rr["cell"]),
                   ref_evals=rr["stats"]["entry_evals"])
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
