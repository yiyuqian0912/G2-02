"""Summarize paired measurements without dropping abstentions from the timing totals."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import median


def read(path):
    with path.open() as stream:
        return list(map(json.loads, stream))


def describe(rows):
    times = sorted(row["total_s"] for row in rows)
    p95 = times[(95 * len(times) + 99) // 100 - 1]
    return (len(rows), sum(row["status"] == "CERTIFIED" for row in rows),
            sum(times), median(times), p95, max(times), sum(row["cpu_s"] for row in rows),
            max(row["peak_process_kib"] for row in rows) / 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    folder = parser.parse_args().results
    frozen = {r["key"]: r for r in read(folder/"frozen.jsonl")}
    print("arm depth attempts certified wall_s median_s p95_s max_s cpu_s process_peak_MiB")
    for name in ("frozen", "pre_layout", "pre_homogeneous", "final", "tail"):
        path = folder/(name+".jsonl")
        if name in ("pre_layout","pre_homogeneous") and not path.exists():
            continue
        rows = read(path)
        for depth in (None,1,2):
            group = [r for r in rows if r["arm"].get("patch_depth") == depth]
            if not group:
                continue
            for row in group:
                old = frozen[row["key"]]
                assert (old["geometry"],old["crop"]) == (row["geometry"],row["crop"])
            print(name, depth, *(round(v,4) for v in describe(group)))

    primary = [r for r in read(folder/"final.jsonl") if r["arm"]["patch_depth"] is None]
    matched = [r for r in primary if r.get("historical_fastlabel_seconds") is not None]
    for group in (matched,[r for r in matched if r["status"]=="CERTIFIED"]):
        fast = sum(r["historical_fastlabel_seconds"] for r in group)
        ours = sum(r["total_s"] for r in group)
        print("FL matched channels, FastLabel seconds, deterministic seconds, ratio",
              len(group),fast,ours,ours/fast)
    print("case, side, frozen wall, new status, new wall, cells, best TV diagnostic")
    for r in primary:
        bound = None if r["bound"] is None else float(Fraction(r["bound"]))
        print(r["key"],r["side"],round(frozen[r["key"]]["total_s"],3),
              r["status"],round(r["total_s"],3),r["cells"],bound)
    print("Endpoint controls")
    for r in read(folder/"controls.jsonl"):
        print(r["key"],r["arm"],r["seed_volume"],r["status"],round(r["total_s"],4))


if __name__=="__main__":
    main()
