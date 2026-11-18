"""Certified deterministic labelling benchmark: untrusted tau-pencil labeller (engine_cert) + independent verifier.

usage:
  bench.py run --tasks tasks.json --out OUT [--workers 3] [--timeout 3600] [--local-first] [--reverify] [--oracle K]
  bench.py one --task JSON --out-dir DIR [--local-first] [--reverify] [--oracle K]

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
    """Produce a certified label (labeller + verifier), save label and certificate, optionally re-verify from disk
    in this process (independent path: decode, verify, recompute the TV of the stored q) and run the oracle audit."""
    t = json.loads(a.task)
    from rind3d_posterior.certify import label as LB, VERIFIER_VERSION
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    tag = "lf" if a.local_first else "tp"
    pl, pc = outd / f"{t['key']}_{tag}_label.npz", outd / f"{t['key']}_{tag}_cert.npz"
    out = dict(key=t["key"], family=t.get("family"), L=t.get("L"), mode=tag, verifier=VERIFIER_VERSION)
    t0 = time.perf_counter()
    label, cert, rep = LB.produce(t["container"], t["w"], t["s"], local_first=a.local_first)
    out["produce_s"] = time.perf_counter() - t0
    LB.save(pl, pc, label, cert)
    m = label["meta"]
    out.update(status=m["status"], tv_bound=float(label["tv_bound"]), label_s=m["label_seconds"], verify_s=m["verify_seconds"],
               lf_plan_s=m.get("lf_plan_seconds"), local_first_verified=m.get("local_first_verified"),
               engine_tv_untrusted=m.get("engine_tv_bound_untrusted"), cells=int(np.sum(label["lo"] > 0)),
               cells_possible=int(len(label["cells"])), Lt=float(np.sum(label["lo"])), Ht=float(np.sum(label["hi"])) + float(label["pool"]),
               pool=float(label["pool"]), stats=rep["stats"], stage1=rep["stage1"],
               label_bytes=pl.stat().st_size, cert_bytes=pc.stat().st_size, events=int(len(cert["ev"])))
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if a.reverify:
        t1 = time.perf_counter()
        r = LB.verify_label(t["container"], t["w"], t["s"], pl, pc)
        out["reverify"] = dict(ok=r["ok"], reasons=r["reasons"], tv=r["tv_recomputed"], seconds=time.perf_counter() - t1)
    if a.oracle > 0:
        from rind3d_posterior.certify import scene as SC
        from rind3d.geometry import load_geometry
        from rind3d.rendering.backends import BackendFactory
        from rind3d_1k.physical import POLICY
        cls, o, N = SC.window_from_container(t["container"], t["w"], t["s"])
        with np.load(t["container"], allow_pickle=False) as z:
            bundle = load_geometry(io.BytesIO(z["geometry"].tobytes()))
        fac = BackendFactory(cuda_allowed=False); L = cls.shape[0]; free = cls != 2
        rng = np.random.default_rng(zlib.crc32(t["key"].encode()))
        tested = bad = 0
        for Vp in rep.get("debug_in_pieces", [])[:a.oracle_pieces]:
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
    fp = outd / ("bench_lf.jsonl" if a.local_first else "bench_tp.jsonl")
    done = {json.loads(l)["key"] for l in fp.read_text().splitlines() if l.strip()} if fp.exists() else set()
    todo = [t for t in tasks if t["key"] not in done]
    if a.only:
        todo = [t for t in todo if t["key"] in a.only.split(",")]

    def one(t):
        t0 = time.perf_counter()
        args = [sys.executable, str(HERE), "one", "--task", json.dumps(t), "--oracle", str(a.oracle), "--out-dir", str(outd / "labels"),
                "--oracle-pieces", str(a.oracle_pieces)]
        if a.local_first:
            args.append("--local-first")
        if a.reverify:
            args.append("--reverify")
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else dict(key=t["key"], status="error", stderr=p.stderr[-3000:])
        except subprocess.TimeoutExpired:
            rec = dict(key=t["key"], status="timeout")
        rec["wall"] = time.perf_counter() - t0
        with fp.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(t["key"], rec.get("status", "ok"), round(rec["wall"], 1), rec.get("tv_bound"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("one"); o.add_argument("--task", required=True); o.add_argument("--oracle", type=int, default=0)
    o.add_argument("--out-dir", required=True); o.add_argument("--local-first", action="store_true")
    o.add_argument("--reverify", action="store_true"); o.add_argument("--oracle-pieces", type=int, default=32)
    r = sp.add_parser("run"); r.add_argument("--tasks", required=True); r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=3); r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--oracle", type=int, default=0); r.add_argument("--only", default=None)
    r.add_argument("--local-first", action="store_true"); r.add_argument("--reverify", action="store_true")
    r.add_argument("--oracle-pieces", type=int, default=32)
    a = ap.parse_args()
    (cmd_one if a.cmd == "one" else cmd_run)(a)


if __name__ == "__main__":
    main()
