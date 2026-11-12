#!/usr/bin/env python3
"""100K production projection from measurements (never from hardware ratios alone).

  estimate.py --report REPORT.json [--label-bench bench_labels.json] [--scaling SCALING.json] \
              [--n-scenes 100000] [--labels-per-scene auto] [--machine-labels-per-s X] [--machine-scenes-per-h Y]

**Inputs:**
* `--report`: a canary root's report.py output. It gives the per-side label costs (CPU-s; the cost distribution of
  the production mix, heavy tails included), labels per scene, scene bytes and label bytes.
* `--label-bench`: bench_labels.py on the TARGET machine with the production worker layout. Its labels/s on the
  fixture mix converts to the production mix by the ratio of mean reference costs (both in the same reference
  CPU-seconds):
      throughput_prod = throughput_bench x mean_ref(fixture mix) / mean_ref(production mix).
* `--scaling`: bench_scaling.py; scenes/h at the largest GPU count.
* or explicit `--machine-labels-per-s` / `--machine-scenes-per-h`.

**Uncertainty:** a bootstrap over the canary's label-cost sample gives the mean production cost; a 90% interval is
reported. Heavy tails enter through the sample. The timeout fraction is reported separately.

**Output:** labelling hours, physical hours, end-to-end hours (concurrent stages: max, and serial: sum), GPU-hours,
CPU-core-hours, storage."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True); ap.add_argument("--label-bench"); ap.add_argument("--scaling")
    ap.add_argument("--n-scenes", type=int, default=100000); ap.add_argument("--labels-per-scene", default="auto")
    ap.add_argument("--machine-labels-per-s", type=float); ap.add_argument("--machine-scenes-per-h", type=float)
    ap.add_argument("--cost-samples", help="jsonl of label metas (L, cpu) to use instead of the report's summary")
    ap.add_argument("--gpus", type=int, default=16); ap.add_argument("--cpus", type=int)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--out")
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    acc = rep["accepted"]
    lps = rep["informative_channels"] / acc if a.labels_per_scene == "auto" else float(a.labels_per_scene)
    n_labels = lps * a.n_scenes
    # production-mix cost sample
    samples = {}
    if a.cost_samples:
        for line in Path(a.cost_samples).read_text().splitlines():
            m = json.loads(line); samples.setdefault(str(m["L"]), []).append(m["cpu"])
    else:
        raise SystemExit("--cost-samples is required (jsonl of label metas from the canary journal: L, cpu)")
    by_side = {L: np.asarray(v, float) for L, v in samples.items()}
    # side shares of informative labels from the canary windows
    share = {L: 1.0 / 3 for L in ("64", "128", "256")}
    if "m_by_side" in rep:
        tot = {L: sum((i + 1) * c for i, c in enumerate(v)) for L, v in rep["m_by_side"].items()}
        s = sum(tot.values()); share = {L: tot[L] / s for L in tot}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        boots.append(sum(share[L] * rng.choice(by_side[L], len(by_side[L])).mean() for L in by_side))
    mean_cost = sum(share[L] * by_side[L].mean() for L in by_side)
    lo, hi = np.percentile(boots, [5, 95])
    out = dict(n_scenes=a.n_scenes, labels_per_scene=lps, n_labels=n_labels, side_share=share,
               mean_label_cost_ref_s=mean_cost, mean_label_cost_ref_s_90ci=[float(lo), float(hi)],
               per_side={L: dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                                 p99=float(np.percentile(v, 99)), max=float(v.max())) for L, v in by_side.items()},
               total_label_cost_ref_core_h=n_labels * mean_cost / 3600)
    thr = a.machine_labels_per_s
    if a.label_bench:
        b = json.loads(Path(a.label_bench).read_text())
        fx = {d["name"]: d for d in json.loads((HERE / "handoff" / "fixtures" / "labels" / "INDEX.json").read_text())}
        ref = [fx[k.split("|")[0]]["ref_cpu_s"] for k, v in b["per_fixture"].items() for _ in range(v["n"])]
        thr = b["labels_per_s"] * float(np.mean(ref)) / mean_cost
        out["label_bench"] = dict(labels_per_s_fixture_mix=b["labels_per_s"], fixture_mean_ref_s=float(np.mean(ref)),
                                  cpu_workers=b["cpu_workers"], gpu_workers=b["gpu_workers"], digests_ok=b["all_digests_ok"])
    if thr:
        hrs = n_labels / thr / 3600
        out.update(machine_labels_per_s=thr, labelling_hours=hrs,
                   labelling_hours_90ci=[n_labels * lo / (mean_cost * thr) / 3600, n_labels * hi / (mean_cost * thr) / 3600],
                   required_labels_per_s_24h=n_labels / 86400, required_labels_per_s_72h=n_labels / (3 * 86400))
    sph = a.machine_scenes_per_h
    if a.scaling:
        rows = json.loads(Path(a.scaling).read_text()); r = max((x for x in rows if "scenes_per_hour" in x), key=lambda x: x["gpus"])
        sph = r["scenes_per_hour"] * a.gpus / r["gpus"]
        out["scaling_used"] = r
    if sph:
        out.update(machine_scenes_per_h=sph, physical_hours=a.n_scenes / sph)
    if thr and sph:
        out.update(end_to_end_hours_concurrent=max(out["labelling_hours"], out["physical_hours"]),
                   end_to_end_hours_serial=out["labelling_hours"] + out["physical_hours"])
    sb = rep.get("scene_bytes", {}).get("mean", 0); lb = rep.get("labels", {}).get("bytes_per_label", 0)
    out.update(storage_scenes_GB=a.n_scenes * sb / 1e9, storage_labels_GB=n_labels * lb / 1e9)
    s = json.dumps(out, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
