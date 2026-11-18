"""Certified deterministic labelling benchmark: untrusted tau-pencil labeller (engine_cert) + independent verifier.

usage:
  bench.py run --tasks tasks.json --out OUT [--workers 3] [--timeout 3600] [--oracle 2] [--keep-certs]
  bench.py one --task JSON [--oracle K] [--cert-out FILE]

Per task (fresh process, nothing read from the generated source):
* labeller: stage1_cert (sentinel octree, depth 4) + stage2_cert (proof tree);
* verifier: rind3d_posterior.certify.verify_certificate on the canonical geometry and the stored observation bits;
* certified TV of the reference estimate lo/sum(lo) (certify.tvb), engine's own (untrusted) tv_bound for comparison;
* optional oracle audit (--oracle K): K random points in each of up to 256 certified IN pieces are rendered by the
  certified physical renderer (cpu_native_certified); every FREE receiver must reproduce the observation.
Output: one JSON line per task in OUT/bench.jsonl."""
import argparse, io, json, os, resource, subprocess, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()


def cmd_one(a):
    t = json.loads(a.task)
    from rind3d_posterior.certify import scene as SC, verify as V, tvb, VERIFIER_VERSION
    from rind3d_posterior.tpl.geom import Geom
    from rind3d_posterior.tpl.window import Window
    from rind3d_posterior.tpl import run_cert
    out = dict(key=t["key"], family=t.get("family"), L=t.get("L"), verifier=VERIFIER_VERSION)
    cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"])
    scene = SC.scene_from_container(t["container"])
    with np.load(t["container"], allow_pickle=False) as z:
        P = z["P"].astype(float); K = z["K"]
    geom = Geom(P, K); win = Window(cls, o, N)
    t0 = time.perf_counter()
    roots, s1, sec1 = run_cert.stage1_cert(geom, win, d1=4)
    lab = run_cert.stage2_cert(geom, win, roots)
    out["label_s"] = time.perf_counter() - t0
    out["engine"] = dict(L_total=lab["res"]["L_total"], tv_bound=lab["res"]["tv_bound"], records=int(len(lab["res"]["cell"])),
                         nodes=lab["res"]["stats"]["nodes"], seconds_stage2=lab["seconds"] if "seconds" in lab else lab["res"]["seconds"])
    cert = dict(s1=s1, ev=lab["ev"], evf=lab["evf"], root_order=roots)
    blob = zlib.compress(np.ascontiguousarray(lab["ev"]).tobytes() + np.ascontiguousarray(lab["evf"]).tobytes() + s1.tobytes(), 6)
    out["certificate"] = dict(events=int(len(lab["ev"])), stage1_boxes=int(len(s1)), raw_bytes=int(lab["ev"].nbytes + lab["evf"].nbytes + s1.nbytes),
                              zlib_bytes=len(blob))
    if a.cert_out:
        np.savez_compressed(a.cert_out, **cert)
    t1 = time.perf_counter()
    rep = V.verify_certificate(scene, cls, o, N, cert)
    out["verify_s"] = time.perf_counter() - t1
    cells, lo, hi = V.cell_enclosures(rep)
    pool = rep["pool"]
    b, Lt = tvb.tv_bound_exact(cells, lo, hi, pool)
    out["certified"] = dict(tv=None if b is None else float(b), Lt=float(Lt), Ht=float(np.sum(hi)) + pool, cells=int(np.sum(lo > 0)),
                            cells_possible=int(len(cells)), pool=pool)
    out["stats"] = rep["stats"]; out["stage1"] = rep["stage1"]
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if a.oracle > 0:
        from rind3d.geometry import load_geometry
        from rind3d.rendering.backends import BackendFactory
        from rind3d_1k.physical import POLICY
        with np.load(t["container"], allow_pickle=False) as z:
            bundle = load_geometry(io.BytesIO(z["geometry"].tobytes()))
        fac = BackendFactory(cuda_allowed=False); L = cls.shape[0]; free = cls != 2
        rng = np.random.default_rng(zlib.crc32(t["key"].encode()))
        tested = bad = 0
        for Vp in rep.get("debug_in_pieces", []):
            ctr = Vp.mean(0)
            for _ in range(a.oracle):
                p = 0.9 * (rng.dirichlet(np.ones(len(Vp))) @ Vp) + 0.1 * ctr
                y, _, _ = fac.source(bundle, p, backend="cpu_native_certified", policy=POLICY).render(tuple(int(v) for v in o), L, N)
                mism = int(np.sum((np.asarray(y, bool).transpose(2, 1, 0) != (cls == 1)) & free))   # renderer: [z, y, x]
                tested += 1; bad += int(mism > 0)
        out["oracle"] = dict(points=tested, points_with_mismatch=bad)
    print(json.dumps(out, default=float))


def cmd_run(a):
    tasks = json.loads(Path(a.tasks).read_text()); outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    fp = outd / "bench.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = [t for t in tasks if t["key"] not in done]
    if a.only:
        todo = [t for t in todo if t["key"] in a.only.split(",")]

    def one(t):
        t0 = time.perf_counter()
        args = [sys.executable, str(HERE), "one", "--task", json.dumps(t), "--oracle", str(a.oracle)]
        if a.keep_certs:
            args += ["--cert-out", str(outd / f"{t['key']}_cert.npz")]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else dict(key=t["key"], status="error", stderr=p.stderr[-3000:])
        except subprocess.TimeoutExpired:
            rec = dict(key=t["key"], status="timeout")
        rec["wall"] = time.perf_counter() - t0
        with fp.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(t["key"], rec.get("status", "ok"), round(rec["wall"], 1), rec.get("certified", {}).get("tv"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("one"); o.add_argument("--task", required=True); o.add_argument("--oracle", type=int, default=0)
    o.add_argument("--cert-out", default=None)
    r = sp.add_parser("run"); r.add_argument("--tasks", required=True); r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=3); r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--oracle", type=int, default=0); r.add_argument("--only", default=None); r.add_argument("--keep-certs", action="store_true")
    a = ap.parse_args()
    (cmd_one if a.cmd == "one" else cmd_run)(a)


if __name__ == "__main__":
    main()
