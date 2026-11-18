"""Validators for RIND3D-100K scene containers, labels and the campaign root.

**Per scene (`validate_scene`):**
* container sha256 equals the manifest's;
* contract id;
* exactly 4/4/4 windows; L = Lp at N = 1024; origins inside the world; no repeated origin per side;
* no source inside any closed window box;
* sources legal: certified FREE plus clearance (deep mode);
* one SOLID crop per window, one lit crop per (window, source);
* no lit SOLID receiver;
* the counts in windows_full equal the crops;
* m and mask recomputed from the crops equal the stored ones; every window has m >= 1. The rule is the contract's:
  v1/v2 "mixed" (an active and an inactive FREE receiver), v3 "direct_interface" (two 6-adjacent FREE receivers with
  different responses; the stored v2 mask `mixed_mask` is checked too);
* **physics spot check:** random receivers against the certified CPU point predicate and the independent predicate
  (`ctx.bind(s, independent=True)`), as in 1K.

**Labels:** consolidated slots align with (window, source); every informative slot has a status; every certified slot
has tv_bound <= 0.05; every uninformative slot is `not_computed`.
"""
import json
from pathlib import Path

import numpy as np

from .storage import SceneFile, scene_path, labels_path, sha256_file, task_key
from .interface import interface_axes

QUOTA = {64: 4, 128: 4, 256: 4}


def validate_scene(root, slot, rec=None, physics=0, deep=False, seed=0):
    errs = []
    p = scene_path(root, slot)
    if not p.exists():
        return dict(slot=slot, ok=False, errors=["missing container"])
    if rec is not None and rec.get("sha256") and sha256_file(p) != rec["sha256"]:
        errs.append("sha256 mismatch")
    S = SceneFile(p)
    meta = S.meta; N = int(meta["resolution"]); k = S.k
    if rec is not None and meta["contract"] != rec.get("contract", meta["contract"]):
        errs.append("contract id")
    W = S.W
    by = {L: int(np.sum(W["Lp"] == L)) for L in QUOTA}
    if by != QUOTA:
        errs.append(f"quota {by}")
    src = S.sources()
    if src.shape != (k, 3):
        errs.append("sources shape")
    seen = set()
    m_hist = np.zeros(4, np.int64)
    rule = ((meta.get("sampling") or {}).get("eligibility") or {}).get("rule", "mixed")
    if rule not in ("mixed", "direct_interface"):
        errs.append(f"unknown eligibility rule {rule}"); rule = "mixed"
    if rule == "direct_interface" and "mixed_mask" not in W.dtype.names:
        errs.append("v3 windows without mixed_mask")
    for w, r in enumerate(W):
        L = int(r["L"]); o = np.array([r["x"], r["y"], r["z"]], np.int64)
        if L != int(r["Lp"]) * N // 1024:
            errs.append(f"w{w} side")
        if np.any(o < 0) or np.any(o + L > N):
            errs.append(f"w{w} bounds")
        key = (int(r["Lp"]), *o.tolist())
        if key in seen:
            errs.append(f"w{w} duplicate")
        seen.add(key)
        lo = o / N; hi = (o + L) / N
        if np.any(np.all((src >= lo) & (src <= hi), axis=1)):
            errs.append(f"w{w} contains a source")
        sol = S.crop(w, -1); free = ~sol
        inf = []; mixed = []
        for j in range(k):
            lit = S.crop(w, j)
            if np.any(lit & sol):
                errs.append(f"w{w} s{j} lit solid")
            nl = int(lit.sum()); nd = int((free & ~lit).sum())
            cnt = meta["windows_full"][w]["counts"][j]
            if [nl, nd, int(sol.sum())] != list(cnt):
                errs.append(f"w{w} s{j} counts")
            mixed.append(nl > 0 and nd > 0)
            if rule == "direct_interface":
                e = interface_axes(lit, sol)
                if list(e) != list(meta["windows_full"][w].get("interface_edges", [[None]] * k)[j]):
                    errs.append(f"w{w} s{j} interface edges")
                inf.append(sum(e) > 0)
            else:
                inf.append(mixed[-1])
        if rule == "direct_interface" and "mixed_mask" in W.dtype.names and \
                int(r["mixed_mask"]) != int(sum(1 << j for j in range(k) if mixed[j])):
            errs.append(f"w{w} mixed_mask")
        m = int(sum(inf)); mask = int(sum(1 << j for j in range(k) if inf[j]))
        if m != int(r["m"]) or mask != int(r["mask"]) or m < 1:
            errs.append(f"w{w} m/mask")
        if 1 <= m <= 4:
            m_hist[m - 1] += 1
    if "sampling" in meta:
        check_sampling_v2(meta, W, k, errs)
    phys = None
    if physics or deep:
        phys = physics_spot_check(S, n_points=int(physics or 512), seed=seed, deep=deep)
        if phys["mismatches"] or phys["independent_mismatches"] or phys.get("illegal_sources"):
            errs.append(f"physics {phys}")
    return dict(slot=int(slot), ok=not errs, errors=errs[:20], m_hist=m_hist.tolist(), k=k, physics=phys)


def check_sampling_v2(meta, W, k, errs):
    """Contract v2 (declared law): k from the kappa draw; streams per side consistent with the windows and the
    declared coverage rule; coverage acceptances consistent with U < c(m); complete exact-m rate histograms."""
    smp = meta["sampling"]; cov = (smp.get("selection") or {}).get("coverage") or {}
    if int(smp["k"]) != k or int(smp.get("k_from_draw", k)) != k:
        errs.append("k differs from the kappa draw")
    if "stream" not in W.dtype.names:
        errs.append("windows without stream field"); return
    full = meta["windows_full"]
    for sd in meta["sides"]:
        Lp = int(sd["Lp"]); idx = [w for w in range(len(W)) if int(W[w]["Lp"]) == Lp]
        st = {x["name"]: x for x in sd["streams"]}
        n_nat = sum(int(W[w]["stream"]) in (0, 2) for w in idx); n_cov = sum(int(W[w]["stream"]) == 1 for w in idx)
        if st["natural"]["accepted"] != n_nat or st.get("coverage", {}).get("accepted", 0) != n_cov:
            errs.append(f"L{Lp} stream counts")
        allowed = cov.get("enabled") and Lp in [int(x) for x in cov.get("sides", [])] and k in [int(x) for x in cov.get("k", [])]
        if ("coverage" in st) != bool(allowed):
            errs.append(f"L{Lp} coverage stream not as declared")
        if any(int(W[w]["stream"]) == 2 for w in idx) and st["natural"].get("fallback", 0) == 0:
            errs.append(f"L{Lp} unrecorded fallback")
        for w in idx:
            f = full[w]; c = f["c_row"]; m = int(W[w]["m"])
            if f["stream"] == "natural" and any(float(x) != 1.0 for x in c):
                errs.append(f"w{w} natural window with thinning")
            if not (m >= 1 and float(f["U"]) < float(c[m - 1])):
                errs.append(f"w{w} acceptance inconsistent with U < c(m)")
        if int(sd.get("rate_exact", 0)) != int(sd["rate_n"]) or int(sum(sd["rate_m"])) != int(sd["rate_A"]):
            errs.append(f"L{Lp} rate stream histogram incomplete")


def physics_spot_check(S, n_points=512, n_windows=3, independent_points=32, seed=0, deep=False):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY, legal_source
    fac = BackendFactory(cuda_allowed=False)
    ctx = fac.context(S.geometry(), POLICY)
    src = S.sources(); N = int(S.meta["resolution"]); W = S.W
    rng = np.random.default_rng(seed)
    mism = checked = imism = ichecked = 0
    for w in rng.choice(len(W), size=min(n_windows, len(W)), replace=False):
        r = W[w]; L = int(r["L"]); o = np.array([r["x"], r["y"], r["z"]])
        zyx = rng.integers(0, L, size=(n_points, 3))
        X = (o[None, :] + zyx[:, ::-1] + 0.5) / N
        for j in range(S.k):
            stored = S.crop(w, j)[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
            y, _, _ = ctx.bind(src[j]).native.points(np.ascontiguousarray(X))
            mism += int(np.sum((y == 1) != stored)); checked += len(X)
            yi, _, _ = ctx.bind(src[j], independent=True).native.points(np.ascontiguousarray(X[:independent_points]))
            imism += int(np.sum((yi == 1) != stored[:independent_points])); ichecked += independent_points
    out = dict(checked=checked, mismatches=mism, independent_checked=ichecked, independent_mismatches=imism)
    if deep:
        out["illegal_sources"] = int(sum(legal_source(ctx, s) is None for s in src))
    return out


def validate_labels(root, slot, scene_rec, cons_rec=None):
    """cons_rec: the slot's last labels/consolidated.jsonl record; its sha256 must match the file."""
    p = labels_path(root, slot)
    if not p.exists():
        return dict(slot=int(slot), ok=None, note="labels not consolidated")
    errs = []
    if cons_rec is None:
        errs.append("no consolidation record (sha256 unknown)")
    elif sha256_file(p) != cons_rec["sha256"]:
        errs.append("consolidated labels sha256 mismatch")
    z = np.load(p)
    status = json.loads(z["status"].tobytes()); metas = json.loads(z["metas"].tobytes()); offs = z["offsets"]
    W = scene_rec["windows"]; k = int(scene_rec["k"])
    want = {task_key(slot, w, j) for w in range(len(W)) for j in range(k)}
    if set(status) != want:
        errs.append("slot set")
    for w, (L, mask, _m) in enumerate(W):
        for j in range(k):
            st = status.get(task_key(slot, w, j))
            if not (int(mask) >> j) & 1 and st != "not_computed":
                errs.append(f"w{w} s{j} uninformative status {st}")
    cp = Path(root) / "CONTRACT.json"
    fl = (json.loads(cp.read_text()).get("labels", {}).get("fastlabel") or {}) if cp.exists() else {}
    src = SceneFile(scene_path(root, slot)).sources()
    cells_all = z["cell"]
    for i, m in enumerate(metas):
        a_, b_ = int(offs[i]), int(offs[i + 1])
        j = int(m["source"]); s_ = np.minimum((src[j] * 128).astype(np.int64), 127)
        if (s_[0] * 128 + s_[1]) * 128 + s_[2] not in set(cells_all[a_:b_].astype(np.int64).tolist()):
            errs.append(f"{m['key']} true source outside the label support")
        if m["status"] == "certified_stat" and fl.get("n_tasks") and (m.get("certificate") or {}).get("n_tasks") != fl["n_tasks"]:
            errs.append(f"{m['key']} certificate n_tasks differs from the contract")
        if m["status"] == "certified" and m["tv_bound"] > 0.05:
            errs.append(f"{m['key']} certified with tv {m['tv_bound']}")
        if m["status"] == "certified_stat":
            c = m.get("certificate") or {}
            if c.get("kind") != "statistical" or not (c.get("eps", 1) <= 0.05 and m["tv_bound"] == c.get("eps")) or \
                    not (0 < c.get("delta_label", 1) < 1) or c.get("K", 1) > c.get("a2", -1):
                errs.append(f"{m['key']} certified_stat without a valid statistical certificate")
        nseg = offs[i + 1] - offs[i]
        if ("n_cells" in m and nseg != m["n_cells"]) or ("n_cells" not in m and not 0 < nseg <= m["n_records"]):
            errs.append(f"{m['key']} cell count")       # legacy piece-format records: at most one cell per piece
    from collections import Counter
    return dict(slot=int(slot), ok=not errs, errors=errs[:20], coverage=dict(Counter(status.values())))
