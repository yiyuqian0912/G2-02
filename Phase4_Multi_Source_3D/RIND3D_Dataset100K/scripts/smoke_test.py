#!/usr/bin/env python3
"""RIND3D-100K server smoke test. Every step must PASS before any benchmark or production run.

  smoke_test.py --work /scratch/r3d_smoke [--gpu 0] [--skip-pytest] [--full-fixtures]

Steps:
1. environment: imports, rind3d and CGAL natives, NVRTC, cupy kernel compile, GPU visible;
2. engine regression: RIND3D_Posterior/tests (bit-exact CPU and GPU digests, record-buffer rerun);
3. label fixtures: the smoke set (or all, with --full-fixtures) on CPU and GPU; every digest must equal the 1K
   reference;
4. physical fixtures: contract-v1 slots 0..3 regenerated from scratch; windows, every crop's packed bits and
   Lambda_hat must equal the reference (cross-machine determinism of the certified renderer);
4b. contract-v2 physical fixtures (regression): slots 0..3 regenerated; windows with streams, crop bits,
   rate-stream histograms and P_A must equal the reference; the containers are kept in <work>/fixtures_v2;
4c. FastLabel fixtures: the v2 fixture labels recomputed on those containers; every digest (cells, masses, status,
   tv_bound) must equal the reference (cross-device bit-exactness of the statistical labeller);
4d/4e. the same for contract v3 (PRODUCTION: direct response-interface eligibility, thin slabs T_min = 2);
4f. Gate A smoke: the GPU verifier against the certified renderer on two windows of the v3 fixture scenes;
5. mini node run: run_node on 2 slots with the default contract (v3) and FastLabel labels, then the validator with
   physics spot checks and label validation (sha256, certificates, true source in support);
6. journal: append, torn tail, recovery.
Steps 4b-5 run in subprocesses: one rind3d CUDA runtime per process.
Writes <work>/smoke_report.json and exits non-zero on any failure."""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src"), str(Path(__file__).resolve().parent)]


def step(name, fn, report):
    t = time.perf_counter()
    try:
        out = fn()
        ok = bool(out.get("ok", True)) if isinstance(out, dict) else bool(out)
        report[name] = dict(ok=ok, seconds=round(time.perf_counter() - t, 2), detail=out)
    except Exception as e:
        report[name] = dict(ok=False, seconds=round(time.perf_counter() - t, 2), error=repr(e)[:500], trace=traceback.format_exc()[-2000:])
    print(f"[{'PASS' if report[name]['ok'] else 'FAIL'}] {name} ({report[name]['seconds']} s)", flush=True)
    return report[name]["ok"]


def s_env():
    import numpy, numba, cupy, zstandard  # noqa
    import rind._exact2d, rind._exact3d, rind3d._native, rind3d._convex, rind3d._research  # noqa
    import cupy as cp
    k = cp.RawModule(code='extern "C" __global__ void f(double* x){x[threadIdx.x]*=2.0;}')
    a = cp.ones(8); k.get_function("f")((1,), (8,), (a,))
    from rind3d.rendering.cuda_runtime import compiler_path
    return dict(ok=bool(float(a.sum().get()) == 16.0), numpy=numpy.__version__, numba=numba.__version__, cupy=cupy.__version__,
                devices=cp.cuda.runtime.getDeviceCount(), nvrtc=str(compiler_path()))


def s_pytest():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "RIND3D_Posterior" / "src"), str(REPO / "RIND3D_Dataset1K"), str(HERE)]))
    out = {}
    for name, d in (("posterior", REPO / "RIND3D_Posterior" / "tests"), ("dataset100k", HERE / "tests")):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(d)], capture_output=True, text=True, env=env, timeout=3600)
        out[name] = dict(ok=r.returncode == 0, tail=r.stdout.strip().splitlines()[-3:])
    return dict(ok=all(v["ok"] for v in out.values()), **out)


def _sub(fn_name, *args, timeout=7200):
    """Run a function of this script in a fresh interpreter (isolates the CUDA runtime) and return its JSON result."""
    code = (f"import sys, json; sys.path[:0] = {[str(Path(__file__).resolve().parent)]!r}; import smoke_test as T; "
            f"print('@@RESULT@@' + json.dumps(T.{fn_name}(*{list(args)!r}), default=str))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    line = [l for l in r.stdout.splitlines() if l.startswith("@@RESULT@@")]
    if r.returncode != 0 or not line:
        return dict(ok=False, error=r.stderr[-1500:])
    return json.loads(line[-1][len("@@RESULT@@"):])


def physical_v2(work):
    return physical_ver(work, "v2")


def physical_ver(work, ver):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k.controller import plan_epoch
    from rind3d_100k.physical import run_scene
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_fixtures import v2_scene_fixture
    cfg = json.loads((HERE / "contract" / f"RIND3D_100K_CONTRACT_{ver}.json").read_text())
    plans = plan_epoch(cfg, 0, []); fac = BackendFactory(); res = []
    out = Path(work) / f"fixtures_{ver}"; out.mkdir(parents=True, exist_ok=True)
    for f in sorted((HERE / "handoff" / "fixtures" / f"physical_{ver}").glob("slot_*.json")):
        ref = json.loads(f.read_text()); slot = ref["slot"]; pl = plans[slot]
        rec, payload = run_scene(cfg, slot, pl["k"], pl["c_table"], fac, 0)
        d = dict(slot=slot, k=pl["k"] == ref["k"], status=rec["status"] == ref["status"])
        if payload is not None:
            (out / f"slot_{slot:06d}.npz").write_bytes(payload)
            got = json.loads(json.dumps(v2_scene_fixture(rec, payload), default=str))
            for kk in ("windows", "crops", "sides", "scene_id"):
                d[kk] = got[kk] == ref[kk]
            d["seconds"] = rec["timing"]["total"]
        d["ok"] = all(v for kk, v in d.items() if kk not in ("slot", "seconds"))
        res.append(d)
    return dict(ok=bool(res) and all(r["ok"] for r in res), results=res)


def fastlabel_v2(work):
    return fastlabel_ver(work, "v2")


def fastlabel_ver(work, ver):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_fixtures import fastlabel_digest
    from rind3d_posterior.fastlabel import label_window_fast, window_from_100k
    ref = json.loads((HERE / "handoff" / "fixtures" / f"fastlabel_{ver}" / "INDEX.json").read_text())
    res = []
    for r in ref["labels"]:
        c = Path(work) / f"fixtures_{ver}" / f"slot_{r['slot']:06d}.npz"
        t0 = time.perf_counter()
        lab = label_window_fast(window_from_100k(c, r["window"], r["source"]), **ref["config"])
        res.append(dict(slot=r["slot"], window=r["window"], source=r["source"], ok=fastlabel_digest(lab) == r["digest"],
                        status=lab["status"], wall=round(time.perf_counter() - t0, 2)))
    return dict(ok=bool(res) and all(x["ok"] for x in res), results=res)


def gate_a_v2(work):
    return gate_a_ver(work, "v2")


def gate_a_ver(work, ver):
    from rind3d_posterior.fastlabel import window_from_100k
    from rind3d_posterior.fastlabel.gate_a import check_window
    ref = json.loads((HERE / "handoff" / "fixtures" / f"fastlabel_{ver}" / "INDEX.json").read_text())
    res = []
    for r in ref["labels"][:2]:
        rec = check_window(window_from_100k(Path(work) / f"fixtures_{ver}" / f"slot_{r['slot']:06d}.npz", r["window"], r["source"]),
                           seed=r["window"])
        res.append({kk: rec[kk] for kk in ("key", "L", "legal", "field_diff", "batch_disagree", "bad_violation", "ok")})
    return dict(ok=bool(res) and all(x["ok"] for x in res), results=res)


def s_fixtures(full):
    from rind3d_100k.fixtures import index, SMOKE, fixture_task
    idx = {d["name"]: d for d in index()}
    names = list(idx) if full else SMOKE
    kw = dict(tier1=dict(nq_cap=1 << 25, ne_cap=1 << 26), gpu=dict(gpu_cap=1 << 22))
    out = []
    for n in names:
        for eng in ("cpu", "gpu"):
            r = fixture_task(n, eng, kw)
            r["ok"] = r.get("digest") == idx[n]["digest"]
            out.append({k: r.get(k) for k in ("name", "engine", "ok", "digest", "wall", "error")})
    return dict(ok=all(r["ok"] for r in out), results=out)


def s_physical():
    import numpy as np, zstandard
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k.controller import plan_epoch
    from rind3d_100k.physical import run_scene
    cfg = json.loads((HERE / "contract" / "RIND3D_100K_CONTRACT_v1.json").read_text())
    plans = plan_epoch(cfg, 0, [], str(HERE / "handoff" / "prior" / "release_1k_rate_stats.jsonl"))
    fac = BackendFactory(); res = []
    for f in sorted((HERE / "handoff" / "fixtures" / "physical").glob("slot_*.json")):
        ref = json.loads(f.read_text()); slot = ref["slot"]; pl = plans[slot]
        same_plan = pl["k"] == ref["k"] and pl["c_table"] == ref["c_table"]
        rec, payload = run_scene(cfg, slot, pl["k"], pl["c_table"], fac, 0)
        d = dict(slot=slot, plan_equal=same_plan, status=rec["status"] == ref["status"])
        if payload is not None:
            z = np.load(io.BytesIO(payload)); blob = z["obs"].tobytes(); dz = zstandard.ZstdDecompressor()
            crops = {f"{int(a)}:{int(b)}": hashlib.sha256(dz.decompress(blob[int(c):int(c) + int(n)])).hexdigest()[:16]
                     for a, b, c, n in z["obs_index"]}
            d.update(windows=[list(x) for x in z["windows"].tolist()] == [list(x) for x in ref["windows"]], crops=crops == ref["crops"],
                     lam=rec["Lambda_hat"] == ref["Lambda_hat"], seconds=rec["timing"]["total"])
        d["ok"] = all(v for k, v in d.items() if k not in ("slot", "seconds"))
        res.append(d)
    return dict(ok=bool(res) and all(r["ok"] for r in res), results=res)


def s_node(work, gpu):
    root = Path(work) / "node"
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    cmd = [sys.executable, str(HERE / "scripts" / "run_node.py"), "--root", str(root), "--slots", "0:2",
           "--gpus", str(gpu), "--progress-s", "10"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    from rind3d_100k.storage import read_jsonl
    from rind3d_100k.validate import validate_scene, validate_labels
    recs = read_jsonl(root / "manifest" / "physical.jsonl")
    cons = {int(x["slot"]): x for x in read_jsonl(root / "labels" / "consolidated.jsonl")}
    acc = [x for x in recs if x["status"] == "accepted"]
    vals = [validate_scene(root, x["slot"], x, physics=256, deep=True) for x in acc]
    labs = [validate_labels(root, x["slot"], x, cons.get(int(x["slot"]))) for x in acc]
    return dict(ok=r.returncode == 0 and len(vals) == 2 and all(v["ok"] for v in vals) and all(v["ok"] for v in labs),
                validation=[{k: v[k] for k in ("slot", "ok", "errors", "physics")} for v in vals],
                labels=[{k: v.get(k) for k in ("slot", "ok", "errors", "coverage")} for v in labs], stderr_tail=r.stderr[-500:])


def s_journal(work):
    import numpy as np
    from rind3d_100k.storage import Journal, journal_index, read_record, iter_journal, decode_payload
    root = Path(work) / "journal"
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    j = Journal(root, "t")
    j.append("000001:00:0", dict(cell=np.arange(3, dtype=np.int32), lo=np.ones(3), hi=np.ones(3), kind=np.zeros(3, np.int8)), dict(a=1))
    j.append("000001:01:0", dict(cell=np.arange(2, dtype=np.int32), lo=np.ones(2), hi=np.ones(2), kind=np.zeros(2, np.int8)), dict(a=2))
    j.close()
    with open(j.path, "ab") as f:
        f.write(b"R3LJ\x10\x00\x00\x00{\"key\": \"torn")          # simulated crash mid-append
    keys = [k for k, _ in iter_journal(j.path)]
    idx = journal_index(root)
    d = decode_payload(read_record(idx["000001:01:0"])[1])
    return dict(ok=keys == ["000001:00:0", "000001:01:0"] and d["meta"]["a"] == 2 and len(d["cell"]) == 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-pytest", action="store_true"); ap.add_argument("--full-fixtures", action="store_true")
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from run_node import default_env
    default_env(a.work)
    rep = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=os.uname().nodename,
               git=subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    ok = step("1_environment", s_env, rep)
    if not a.skip_pytest:
        ok &= step("2_engine_regression", s_pytest, rep)
    ok &= step("3_label_fixtures", lambda: s_fixtures(a.full_fixtures), rep)
    ok &= step("4_physical_fixtures", s_physical, rep)
    ok &= step("4b_physical_fixtures_v2", lambda: _sub("physical_ver", a.work, "v2"), rep)
    ok &= step("4c_fastlabel_fixtures_v2", lambda: _sub("fastlabel_ver", a.work, "v2"), rep)
    ok &= step("4d_physical_fixtures_v3", lambda: _sub("physical_ver", a.work, "v3"), rep)
    ok &= step("4e_fastlabel_fixtures_v3", lambda: _sub("fastlabel_ver", a.work, "v3"), rep)
    ok &= step("4f_gate_a_smoke_v3", lambda: _sub("gate_a_ver", a.work, "v3"), rep)
    ok &= step("5_node_run_and_validation", lambda: _sub("s_node", a.work, 0), rep)
    ok &= step("6_journal_recovery", lambda: s_journal(a.work), rep)
    rep["ok"] = bool(ok)
    (Path(a.work) / "smoke_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
