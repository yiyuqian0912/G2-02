"""Summarize exact matched supports and separate measured from estimated work."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ..b2_angular.preservation import sha
from ..study import atomic_json
from .audit import PREVIOUS
from .build import OUT


def read(path):
    return json.loads(path.read_text())


def historic_path(previous, case, n, method):
    ordinal = int(case.rsplit("-", 1)[-1])
    if n == 128:
        group, name = f"fields128-case{ordinal}", method
    elif ordinal >= 60:
        group, name = "heldout-fields", method
    elif method == "D":
        group, name = "frozen-fields", method
    else:
        group, name = "baseline", {"A": "A_independent", "P": "A_primary"}[method]
    return previous / group / f"{case}-{n}-{name}.json"


def checked_field(path, row):
    archive = path.with_suffix(".npz")
    if row["archive_SHA256"] != sha(archive):
        raise ValueError(f"Scientific array digest differs at {archive}")
    return {**row, "record_SHA256": sha(path), "record": str(path)}


def field_groups(output, previous):
    matched = defaultdict(dict)
    stages = {}
    repeats = {}
    repeat = output / "repeats/frozen64-case21/summary.json"
    summaries = sorted(output.glob("*/summary.json"))
    if repeat.exists():
        repeat_rows = read(repeat)["results"]
        if len(repeat_rows) == 3 and all(r["complete"] for r in repeat_rows):
            summaries.append(repeat)
    for summary in summaries:
        rows = read(summary).get("results")
        if not rows or "resolution" not in rows[0]:
            continue
        stages[summary.parent.name] = {
            "identity_SHA256": sha(summary.with_name("identity.json")),
            "summary_SHA256": sha(summary),
        }
        for row in rows:
            c, n, method = row["case"], row["resolution"], row["method"]
            path = summary.parent / f"{c}-{n}-{method}.json"
            if method in matched[c, n]:
                if summary != repeat:
                    raise ValueError(f"Duplicate matched field record for {c}, {n}, {method}")
                original = matched[c, n][method]
                if any(original[k] != row[k] for k in ("geometry_id", "source_SHA256", "physical_bits_SHA256")):
                    raise AssertionError("Overlapped run and repeat differ scientifically")
                repeats[method] = original
            matched[c, n][method] = checked_field(path, row) | {
                "timing": "current uncontended repeat" if summary == repeat else "current"
            }
    for (c, n), records in sorted(matched.items()):
        for method in ("A", "P", "D"):
            if method in records:
                continue
            path = historic_path(previous, c, n, method)
            old = read(path)
            records[method] = checked_field(path, old) | {"timing": "historical same host and support"}
            records[method]["cold_total_seconds"] = old.get("cold_including_clearance_seconds", old["cold_total_seconds"])
        if len({(r["geometry_id"], r["source_SHA256"]) for r in records.values()}) != 1:
            raise AssertionError((c, n, "geometry or source mismatch"))
        if any(not r["complete"] or r["valid_voxels"] != n**3 for r in records.values()):
            raise AssertionError((c, n, "incomplete field is not a matched result"))
        if len({r["physical_bits_SHA256"] for r in records.values()}) != 1:
            raise AssertionError((c, n, "physical bits disagree"))
    return {"stages": stages, "compiler_overlap_repeat_originals": repeats, "supports": [
        {"case": c, "resolution": n, "records": records}
        for (c, n), records in sorted(matched.items())
    ]}


def diagnostics(output):
    audits = {}
    for name in ("audit-timing", "audit-probe64"):
        path = output / (name + ".json")
        audit = read(path)
        groups = {}
        for n in (64, 128):
            rows = [r for r in audit["rows"] if r["grid"] == n]
            weighted = Counter()
            raw = Counter()
            actual = Counter()
            for row in rows:
                work = row["full_D_work"]
                for key in ("study_low_nodes", "study_high_nodes", "study_precision_escalations",
                            "reference_line_builds", "reference_exp_endpoints"):
                    actual[key] += work[key]
                actual["full_D_cold_seconds"] += row["full_D_cold_seconds"]
                actual["full_D_scalar_seconds"] += row["full_D_scalar_seconds"]
                actual["repeated_256_line_kernel_preparations"] += work["study_precision_escalations"] * row["input"]["kernels"]
                for group in row["groups"]:
                    weight = group["population"] / group["sample_size"]
                    raw.update(group["sample_work"])
                    for k, v in group["sample_work"].items():
                        weighted[k] += weight * v
            groups[str(n)] = {"frozen_full_field_totals": dict(actual),
                              "sample_work": dict(raw), "population_weighted_estimates": dict(weighted)}
        audits[name] = {"SHA256": sha(path), "build": audit["instrumented_build"], "groups": groups}
    rays = {}
    for path in sorted((output / "ray-stage").glob("rays-*.json")):
        trial = read(path)
        rays[path.stem] = {k: v for k, v in trial.items() if k != "trials"}
        rays[path.stem]["SHA256"] = sha(path)
        rays[path.stem]["trial_count"] = len(trial["trials"])
    adversary = output / "ray-stage/adversarial.json"
    return {"D_transition_audits": audits, "ray_trials": rays,
            "adversaries": {"SHA256": sha(adversary), **read(adversary)}}


def tile_diagnostics(output):
    result = {}
    for name in ("tile-reuse/reuse.json", "tile-reuse/monolithic.json",
                 "scaling256/scale-identity.json", "scaling256/budget-gate.json",
                 "scaling256/coverage.json"):
        path = output / name
        if path.exists():
            result[name] = {"record_SHA256": sha(path), **read(path)}
    scale = output / "scaling256"
    if (scale / "coverage.json").exists():
        from .tiles import scaling_summary
        if scaling_summary(scale) != read(scale / "coverage.json"):
            raise AssertionError("Recorded and reconstructed scale coverage differ")
        for backend in ("P", "F"):
            for phase in ("probe", "remaining"):
                path = scale / backend / f"{phase}.json"
                if path.exists():
                    result[f"scaling256/{backend}/{phase}.json"] = {
                        "record_SHA256": sha(path), **read(path),
                    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT)
    p.add_argument("--previous", type=Path, default=PREVIOUS)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    atomic_json(args.output, {"schema": "b2-precision-measured-v1",
                              "fields": field_groups(args.root, args.previous),
                              "tile_experiments": tile_diagnostics(args.root),
                              **diagnostics(args.root)})
