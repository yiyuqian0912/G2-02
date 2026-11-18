"""Certified deterministic labels: produce (labeller + verifier) and independently re-verify a stored label.

Released label artifact (`*_label.npz`, every array binary64/int64, no pickles):
  cells (n,) int64     128^3 cell ids (ix*128 + iy)*128 + iz, ascending, every cell with hi > 0
  q (n,) float64       released distribution: binary64 rounding of lo_c / sum(lo) (0 where lo_c = 0)
  lo, hi (n,) float64  certified per-cell volume enclosures lo_c <= Vol(F n c) <= hi_c (+ pooled mass)
  pool () float64      certified upper volume of undetermined mass whose cell is not resolved (treated as w = 1)
  tv_bound () float64  certified upper bound of TV(q, q_exact) (exact rational evaluation, rounded up)
  meta (json bytes)    status, verifier version, digests of geometry bytes, observation bits and certificate file
Certificate (`*_cert.npz`): the proof tree written by the (untrusted) labeller: s1, ev, evf, root_order.

A label is valid only if `verify_label` re-proves it: it decodes the canonical geometry and the stored observation,
re-runs the verifier on the certificate, recomputes the TV bound of the STORED q against its own enclosures and
checks it does not exceed the stored tv_bound. Nothing computed by the labeller is trusted."""
import hashlib
import io
import json
import time

import numpy as np

from . import VERIFIER_VERSION
from . import scene as SC, verify as V, tvb

LABEL_FORMAT = "rind3d-certified-label-1"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _digests(container, cls):
    with np.load(container, allow_pickle=False) as z:
        geo = z["geometry"].tobytes()
    return dict(geometry_sha256=_sha(geo), observation_sha256=_sha(np.ascontiguousarray(cls, np.uint8).tobytes()))


def produce(container, w, s, d1=4, engine_kw=None, local_first=False, lf_kw=None):
    """Label one (scene, window, channel): untrusted labeller -> certificate -> verifier -> artifact dicts.
    local_first: Proposition A' root drops and wedge clips (planned by localfirst.certclip, re-proved by the verifier).
    Returns (label dict, certificate dict, report)."""
    from ..tpl.geom import Geom
    from ..tpl.window import Window
    from ..tpl import run_cert
    t0 = time.perf_counter()
    cls, o, N = SC.window_from_container(container, w, s)
    scene = SC.scene_from_container(container)
    with np.load(container, allow_pickle=False) as z:
        P = z["P"].astype(float); K = z["K"]
    geom = Geom(P, K); win = Window(cls, o, N)
    roots, s1, _ = run_cert.stage1_cert(geom, win, d1=d1)
    lf = {}
    if local_first and len(roots):
        from ..localfirst import certclip
        pl = certclip.plan(scene, cls, o, N, roots, key=f"{container}:{w}:{s}", **(lf_kw or {}))
        keep = pl["keep"]
        kept = roots[keep]
        rows, recs, off = [], [], [0]
        for q in np.flatnonzero(keep):
            rows += pl["rows"].get(int(q), []); recs += pl["clips"].get(int(q), []); off.append(len(recs))
        drop = [(*roots[q].tolist(), *pl["drop"][q]) for q in np.flatnonzero(~keep)]
        lab = run_cert.stage2_cert(geom, win, kept, clip_off=np.array(off, np.int64),
                                   clip_rows=np.array(rows, float).reshape(-1, 4), **(engine_kw or {}))
        lf = dict(lf_clip=np.array(recs, np.int64).reshape(-1, 6), lf_clip_off=np.array(off, np.int64),
                  lf_drop=np.array(drop, np.int64).reshape(-1, 6))
        roots = kept
        lf_seconds = pl["seconds"]
    else:
        lab = run_cert.stage2_cert(geom, win, roots, **(engine_kw or {}))
        lf_seconds = 0.0
    t1 = time.perf_counter()
    cert = dict(s1=s1, ev=lab["ev"], evf=lab["evf"], root_order=roots, **lf)
    rep = V.verify_certificate(scene, cls, o, N, cert)
    t2 = time.perf_counter()
    label = assemble(rep, container, cls, cert, w, s)
    label["meta"].update(label_seconds=t1 - t0, verify_seconds=t2 - t1, engine_tv_bound_untrusted=lab["res"]["tv_bound"],
                         local_first=bool(local_first), lf_plan_seconds=lf_seconds, local_first_verified=rep.get("local_first"))
    return label, cert, rep


def _cert_bytes(cert):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in cert.items()})
    return buf.getvalue()


def assemble(rep, container, cls, cert, w, s):
    cells, lo, hi = V.cell_enclosures(rep)
    pool = float(rep["pool"])
    meta = dict(format=LABEL_FORMAT, verifier=VERIFIER_VERSION, window=int(w), channel=int(s), stats=rep["stats"],
                stage1=rep["stage1"], **_digests(container, cls), certificate_sha256=_sha(_cert_bytes(cert)))
    keep = hi > 0
    cells, lo, hi = cells[keep], lo[keep], hi[keep]
    b, Lt = tvb.tv_bound_exact(cells, lo, hi, pool)
    if b is None or rep["stage1"]["partition"] is not True or rep.get("roots_without_tree"):
        meta["status"] = "undefined" if b is None else "invalid"
        return dict(cells=cells, q=np.zeros(len(cells)), lo=lo, hi=hi, pool=np.float64(pool), tv_bound=np.float64(1.0), meta=meta)
    qc, qv = tvb.label_from_bounds(cells, lo)
    q = np.zeros(len(cells)); q[np.searchsorted(cells, qc)] = qv
    t = tvb.tv_of_stored(cells, q, cells, lo, hi, pool)
    meta["status"] = "certified"
    return dict(cells=cells, q=q, lo=lo, hi=hi, pool=np.float64(pool), tv_bound=np.float64(t["tv"]), meta=meta)


def save(path_label, path_cert, label, cert):
    """Write both artifacts (numpy npz, no pickles). The certificate digest in the label refers to the cert arrays."""
    meta = json.dumps(label["meta"], sort_keys=True, default=float).encode()
    np.savez_compressed(path_label, cells=label["cells"], q=label["q"], lo=label["lo"], hi=label["hi"], pool=label["pool"],
                        tv_bound=label["tv_bound"], meta=np.frombuffer(meta, np.uint8))
    with open(path_cert, "wb") as f:
        f.write(_cert_bytes(cert))


def load_label(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in ("cells", "q", "lo", "hi", "pool", "tv_bound")}
        d["meta"] = json.loads(z["meta"].tobytes().decode())
    return d


def load_cert(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify_label(container, w, s, path_label, path_cert):
    """Independent re-verification of a stored label. Returns dict(ok, reasons, tv_recomputed, ...)."""
    label = load_label(path_label); cert = load_cert(path_cert)
    cls, o, N = SC.window_from_container(container, w, s)
    reasons = []
    dg = _digests(container, cls)
    for k in ("geometry_sha256", "observation_sha256"):
        if label["meta"].get(k) != dg[k]:
            reasons.append(f"{k} mismatch")
    if label["meta"].get("certificate_sha256") != _sha(_cert_bytes(cert)):
        reasons.append("certificate digest mismatch")
    if label["meta"].get("status") != "certified":
        reasons.append(f"label status {label['meta'].get('status')}")
    scene = SC.scene_from_container(container)
    rep = V.verify_certificate(scene, cls, o, N, cert)
    cells, lo, hi = V.cell_enclosures(rep)
    keep = hi > 0
    t = tvb.tv_of_stored(label["cells"], label["q"], cells[keep], lo[keep], hi[keep], rep["pool"])
    if t["status"] != "certified":
        reasons.append("no certified positive feasible volume")
    elif not (t["tv"] <= float(label["tv_bound"])):
        reasons.append(f"recomputed TV bound {t['tv']} exceeds the stored {float(label['tv_bound'])}")
    if abs(float(np.sum(label["q"])) - 1.0) > 1e-9:
        reasons.append("stored q does not sum to 1")
    return dict(ok=not reasons, reasons=reasons, tv_recomputed=t.get("tv"), tv_stored=float(label["tv_bound"]),
                verifier=VERIFIER_VERSION, stats=rep["stats"])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Produce or independently verify a certified RIND3D posterior label.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("produce", "verify"):
        p = sp.add_parser(name)
        p.add_argument("--container", required=True); p.add_argument("--window", type=int, required=True)
        p.add_argument("--channel", type=int, required=True)
        p.add_argument("--label", required=True); p.add_argument("--cert", required=True)
        p.add_argument("--local-first", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "produce":
        label, cert, rep = produce(a.container, a.window, a.channel, local_first=a.local_first)
        save(a.label, a.cert, label, cert)
        print(json.dumps(dict(status=label["meta"]["status"], tv_bound=float(label["tv_bound"]), cells=int(len(label["cells"])),
                              label_seconds=label["meta"]["label_seconds"], verify_seconds=label["meta"]["verify_seconds"])))
    else:
        r = verify_label(a.container, a.window, a.channel, a.label, a.cert)
        print(json.dumps(r, default=float))
        raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
