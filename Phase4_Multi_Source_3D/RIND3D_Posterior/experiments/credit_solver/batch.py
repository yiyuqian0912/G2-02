"""Run the credit solver on a frozen task list, one process per task (no shared state).

Writes one result envelope per task (run.py format) and one JSON line per task to
OUT/runs.jsonl with the printed run summary, exit code and wall time. The task list
is a tasks.json-style manifest; only slot/w/s identify the input. No reference label
or source position is read here.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3")


def tasks_from(manifests):
    out = []
    for spec in manifests:
        path, _, split = spec.partition(":")
        d = json.loads(Path(path).read_bytes())
        for split_name in (split.split(",") if split else sorted(d)):
            for t in d[split_name]:
                out.append(dict(t, split=f"{Path(path).stem}:{split_name}"))
    seen, uniq = set(), []
    for t in out:
        if t["key"] not in seen:
            seen.add(t["key"])
            uniq.append(t)
    return uniq


def run_one(t, args, env):
    container = args.root / "scenes" / f"{t['slot']//1000:04d}" / f"{t['slot']:06d}.npz"
    result = args.out / "results" / f"{t['key']}.json"
    cmd = [args.python, str(HERE / args.runner), str(container), str(t["w"]), str(t["s"]), str(result),
           "--seconds", str(args.seconds)] + args.extra
    start = perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.seconds * 4 + 120)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        code, out, err = -9, exc.stdout or "", "hard timeout"
    wall = perf_counter() - start
    summary = None
    for line in reversed((out or "").splitlines()):
        try:
            summary = json.loads(line)
            break
        except ValueError:
            continue
    rec = dict(task={k: t[k] for k in ("slot", "w", "s", "L", "family", "key", "split")},
               exit=code, process_wall=wall, summary=summary,
               stderr_tail=None if code == 0 else (err or "")[-2000:])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("manifests", nargs="+", help="path[:split1,split2]")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--runner", default="run.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", nargs="*", default=None, help="task keys")
    ap.add_argument("--extra", default="", help="extra runner arguments, one string")
    args = ap.parse_args()
    args.extra = args.extra.split()
    tasks = tasks_from(args.manifests)
    if args.only:
        tasks = [t for t in tasks if t["key"] in set(args.only)]
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    log = (args.out / "runs.jsonl").open("a")
    with ThreadPoolExecutor(args.jobs) as pool:
        for rec in pool.map(lambda t: run_one(t, args, env), tasks):
            log.write(json.dumps(rec) + "\n")
            log.flush()
            s = rec["summary"] or {}
            print(rec["task"]["key"], rec["exit"], s.get("status"), s.get("reason"),
                  None if s.get("tv") is None else float(eval_fraction(s["tv"])),
                  round(rec["process_wall"], 2), flush=True)


def eval_fraction(s):
    from fractions import Fraction
    return Fraction(s)


if __name__ == "__main__":
    main()
