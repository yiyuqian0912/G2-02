"""CLI for the CBC kernel v2 + search. Same envelope format and atomic writer as run.py."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import inputs
from cbc_search import solve
from run import write_result


def warm_jit():
    """Load the cached Numba screens on a dummy 64^3 input (no physics, no result)."""
    import numpy as np
    from screen2 import scan_collect, scan_ids
    t = perf_counter()
    cls = np.zeros((64, 64, 64), np.uint8)
    rows = np.zeros((1, 4), np.int64)
    off = np.array([0, 1], np.int32)
    o = np.zeros(3, np.int64)
    scan_collect(cls, o, 2048, rows, off, np.zeros(0, np.int64), 1)
    scan_ids(cls, np.zeros(0, np.int64), o, 2048, rows, off)
    return perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("container", type=Path)
    ap.add_argument("window", type=int)
    ap.add_argument("channel", type=int)
    ap.add_argument("output", type=Path)
    ap.add_argument("--tolerance", type=Fraction, default=Fraction(1, 20))
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--split-above", type=int, default=1500)
    ap.add_argument("--no-single-dark", dest="prefer_single_dark", action="store_false")
    ap.add_argument("--vertex-sum-anchor", dest="centroid_anchor", action="store_false")
    args = ap.parse_args()
    t, c = perf_counter(), process_time()
    warm = warm_jit()  # one-time per process; a long-lived worker pays it once
    obs, identity = inputs.load(args.container, args.window, args.channel)
    key = f"{identity['slot']:06d}_w{args.window:02d}_s{args.channel}"
    payload, details = solve(obs, key, args.tolerance, args.seconds, cap=args.cap, pairs=args.pairs,
                             batch=args.batch, split_above=args.split_above,
                             prefer_single_dark=args.prefer_single_dark, centroid_anchor=args.centroid_anchor)
    details["solver"] = "claude-cbc-v2"
    details["jit_warm_seconds"] = warm
    write_result(args.output, identity, args.tolerance, payload, details)
    print(json.dumps(dict(key=key, wall=perf_counter() - t, cpu=process_time() - c,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details)))


if __name__ == "__main__":
    main()
