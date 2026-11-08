"""Measure continuous A/B certificates and original neighboring scalar costs."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_miss.bench import read
from ..b2_precision.build import extension
from ..b2_primary_miss.bench import make_session
from ..study import atomic_json
from .freeze import OUT

ORDER = (-2, -3, -1, 2, 3, 1)


def neighbors(xyz, n, steps):
    for d in ORDER:
        axis = abs(d)-1
        start = int(xyz[axis])
        end = start + (steps if d > 0 else -steps)
        if not 0 <= end < n:
            continue
        points, future = [], []
        for distance in range(1, steps+1):
            p = xyz.copy()
            dest = start + (distance if d > 0 else -distance)
            p[axis] = dest
            index = int((p[2]*n + p[1])*n + p[0])
            points.append(index)
            # Reverse Morton order of 2^3 cells and ascending xyz within a cell.
            if (dest > start if start//2 == dest//2 else dest < start):
                future.append(index)
        yield (d, steps), (points, future)


def quantiles(x):
    if not len(x):
        return {}
    return {f"p{p}": float(np.percentile(x,p)) for p in (0, 10, 50, 90, 95, 100)}


def summarize(rows, ns):
    out = {"rays": len(rows), "strict": sum(r["margin"] > 0 for r in rows),
           "kinds": dict(Counter(r["kind"] for r in rows)),
           "endpoints": sum(r["t"] in (0, 1) for r in rows),
           "margin_relative": quantiles([r["margin"]/r["tau_upper"] for r in rows]),
           "t": quantiles([r["t"] for r in rows if r["t"] is not None]),
           "capture_minus_control_ns": quantiles([(r["capture_seconds"]-r["scalar_seconds"])*1e9
                                                   for r in rows]),
           "capture_seconds": sum(r["capture_seconds"] for r in rows),
           "control_seconds": sum(r["scalar_seconds"] for r in rows)}
    all_trials = [r for row in rows for r in row["trials"]]
    out["region_controls"] = {}
    for step in (1, 4):
        trials = [r for r in all_trials if r["steps"] == step]
        controls = [r for r in trials if "cover" in r]
        out["region_controls"][str(step)] = {
            "attempts": len(controls),
            **{k: sum(r[k] for r in controls) for k in ("cover", "proxy", "section", "direct", "gradient")},
            **{k+"_seconds": sum(r[k+"_seconds"] for r in controls)
               for k in ("cover", "proxy", "section", "direct", "gradient")},
            "direct_beyond_all_controls": sum(r["direct"] and not (r["cover"] or r["proxy"] or r["section"])
                                                for r in controls),
        }
        picked = [max(eligible, key=lambda r: len(r["future"]))
                  for row in rows
                  if (eligible := [r for r in row["trials"] if r["steps"] == step and
                                   len(r["future"]) >= (1 if step == 1 else 2)])]
        for method in ("gradient", "direct"):
            for label, population in (("all", trials), ("one_per_seed", picked)):
                accepted = [r for r in population if r[method]]
                covered = {i for r in accepted for i in r["future_residual"]}
                out[f"{step}_{method}_{label}"] = {
                    "attempts": len(population), "certified": len(accepted),
                    "centers_on_certified_axial_intervals": sum(len(r["centers"]) for r in accepted),
                    "unique_future_scalar_HIT": len(covered),
                    "original_potential_seconds": sum(int(ns[i]) for i in covered)/1e9,
                    "construction_seconds": sum(r[method+"_seconds"] for r in population),
                    "failure_seconds": sum(r[method+"_seconds"] for r in population if not r[method]),
                    "exp_endpoints": sum(r[method+"_exp"] for r in population),
                    "construction_us": quantiles([r[method+"_seconds"]*1e6 for r in population]),
                }
    return out


def run(args):
    module = extension(args.native)
    inputs = read(args.plan)["strata"]
    metadata, rows = [], []
    sessions = {}
    start = time.perf_counter()
    for stratum in inputs:
        tile = stratum["tile"]
        case = tile["case"]
        if case not in sessions:
            session, setup, cpu = make_session(module, tile, "P_J", True)
            sessions[case] = session
            metadata.append({"case": case, "setup_seconds": setup, "CPU_seconds": cpu,
                             "info": session.info()})
        archive = Path(stratum["archive"])
        if sha(archive) != stratum["archive_SHA256"]:
            raise ValueError("Frozen P_J trace differs")
        with np.load(archive) as a:
            path, ns = (a[k].ravel() for k in ("evaluation_path", "ray_nanoseconds"))
            response, valid = (a[k].ravel() for k in ("response", "valid"))
        items = []
        n = tile["side"]
        for order, idx in enumerate(stratum["indices"]):
            if path[idx] != 7:
                raise ValueError("Audit seed did not use original scalar HIT")
            xyz = np.array(np.unravel_index(idx, (n,)*3))[::-1]
            receiver = (xyz + tile["origin"] + .5) / tile["grid"]
            proposed = dict(pair for s in (1,4) for pair in neighbors(xyz, n, s))
            row = sessions[case].hit_audit(receiver, list(proposed),
                                          controls=order % 16 == 0, capture_first=order % 2 == 0)
            if row["status"] != 1:
                raise AssertionError(("P_J scalar label changed", tile["id"], idx, row))
            for trial in row["trials"]:
                centers, future = proposed[trial["direction"], trial["steps"]]
                if any(trial.get(k, False) for k in ("gradient", "direct", "cover", "proxy", "section")):
                    if not all(valid[i] and not response[i] for i in centers):
                        raise AssertionError(("A certified HIT conflicts with the reference", trial, centers))
                trial.update(centers=centers, future=future,
                             future_residual=[i for i in future if path[i] == 7])
            items.append({"index": idx, "original_ns": int(ns[idx]), **row})
        metrics = summarize(items, ns)
        rows.append({"tile": tile["id"], "stratum": stratum["stratum"],
                     "population": stratum["population"], "summary": metrics, "samples": items})
        print(tile["id"], stratum["stratum"], len(items), metrics["strict"],
              *(metrics[f"{n}_direct_one_per_seed"]["unique_future_scalar_HIT"] for n in (1,4)),
              flush=True)
    atomic_json(args.output, {"plan_SHA256": sha(args.plan), "build": module.study_identity,
                              "setup": metadata, "wall_seconds": time.perf_counter()-start, "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=OUT / "audit-plan.json")
    p.add_argument("--native", type=Path, default=OUT / "audit-native")
    p.add_argument("--output", type=Path, default=OUT / "witness-audit.json")
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Keep the measured witness audit")
    run(a)
