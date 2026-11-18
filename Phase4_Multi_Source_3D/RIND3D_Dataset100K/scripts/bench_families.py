"""Per-family physical scene-generation benchmark (docs/GENERATOR_BENCHMARK_V3.md). Posterior labelling is NOT included.

usage: bench_families.py run --out DIR [--n 8] [--families a,b] [--contract v3|curved]
       bench_families.py report --out DIR
`run` starts one fresh process per family (`one`), so peak CPU memory (VmHWM) belongs to that family. Each process
generates one warm-up scene (not recorded: NVRTC/numba compilation and caches), then n scenes of its family at fixed
seeds (slots f, f + F, f + 2F, ... of the contract's round robin), sequentially, one GPU. Peak GPU memory is sampled
with nvidia-smi for the process after every scene. Recorded per scene: every timing stage of the physical stage, the
attempts and rejection reasons, proposals per side and the work counters."""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
CONTRACTS = {"v3": "RIND3D_100K_CONTRACT_v3.json", "curved": "RIND3D_100K_CURVED_RESEARCH_v0.json"}


def gpu_mem_mb(pid):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    for l in r.stdout.splitlines():
        p, m = [x.strip() for x in l.split(",")]
        if int(p) == pid:
            return float(m)
    return 0.0


def vmhwm():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1]) / 1024


def cmd_one(a):
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k import sampling as SM
    cfg = json.loads((HERE / "contract" / CONTRACTS[a.contract]).read_text())
    if a.contract == "curved":
        from rind3d_100k.physical_curved import run_scene_curved as run
        runf = lambda slot, k, fac, att: run(cfg, slot, k, fac, att)
    else:
        from rind3d_100k.physical import run_scene
        runf = lambda slot, k, fac, att: run_scene(cfg, slot, k, None, fac, att)
    fams = cfg["families"]; F = len(fams); f = fams.index(a.family)
    fac = BackendFactory()
    out = Path(a.out); pid = os.getpid(); gpu_peak = 0.0
    slots = [f + F * i for i in range(a.n + 1)]
    for i, slot in enumerate(slots):
        k, _ = SM.kappa_draw(cfg, slot)
        attempts = []; t0 = time.perf_counter()
        for att in range(int(cfg["max_attempts"])):
            ta = time.perf_counter(); rec, payload = runf(slot, k, fac, att)
            attempts.append(dict(attempt=att, status=rec["status"], reason=rec.get("reason"), wall=time.perf_counter() - ta))
            if payload is not None:
                break
        wall = time.perf_counter() - t0
        gpu_peak = max(gpu_peak, gpu_mem_mb(pid))
        if i == 0:
            continue                                       # warm-up
        r = dict(family=a.family, slot=slot, k=k, wall=wall, status=rec["status"], attempts=attempts, timing=rec.get("timing", {}),
                 work={kk: v for kk, v in (rec.get("work") or {}).items() if isinstance(v, (int, float))},
                 sides=[dict(Lp=s["Lp"], proposals=s["proposals"], rate_n=s["rate_n"], P_A=s["P_A_hat"]) for s in rec.get("sides", [])],
                 rss_peak_mb=vmhwm(), gpu_peak_mb=gpu_peak)
        with open(out / f"{a.contract}_{a.family}.jsonl", "a") as fh:
            fh.write(json.dumps(r, default=float) + "\n")
        print(a.family, slot, rec["status"], round(wall, 1), flush=True)


def cmd_run(a):
    cfg = json.loads((HERE / "contract" / CONTRACTS[a.contract]).read_text())
    fams = a.families.split(",") if a.families else cfg["families"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    (out / f"env_{a.contract}.json").write_text(json.dumps(dict(gpu=gpu, cpu=cpu, cores=os.cpu_count(), python=platform.python_version(),
                                                                 host=platform.node(), contract=cfg["contract"], n=a.n,
                                                                 OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                                                                 NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS"))))
    for fam in fams:
        subprocess.run([sys.executable, __file__, "one", "--family", fam, "--n", str(a.n), "--out", str(out), "--contract", a.contract], check=False)


def cmd_report(a):
    out = Path(a.out); rows = []
    for p in sorted(out.glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        acc = [r for r in recs if r["status"] == "accepted"]
        q = lambda xs: [round(float(np.percentile(xs, x)), 2) for x in (50, 95, 100)] if len(xs) else None
        stage = lambda key: q([r["timing"].get(key, 0.0) for r in acc])
        sel = [r["timing"].get("selection", 0) - r["work"].get("accept_render_s", 0) - r["work"].get("rate_s", 0) for r in acc]
        rej = sum(len(r["attempts"]) - 1 for r in recs) + sum(r["status"] != "accepted" for r in recs)
        rows.append(dict(contract=p.stem.split("_")[0], family=recs[0]["family"], scenes=len(recs), accepted=len(acc),
                         rejected_attempts=rej, reasons=sorted({x["reason"] for r in recs for x in r["attempts"] if x["reason"]}),
                         total=q([r["wall"] for r in recs]), geometry=stage("geometry"), sources=stage("sources"),
                         proposal_screening=q(sel), accepted_render=q([r["work"].get("accept_render_s", 0) for r in acc]),
                         rate_stream=q([r["work"].get("rate_s", 0) for r in acc]), serialize=stage("serialize"),
                         proposals_per_scene=q([sum(s["proposals"] for s in r["sides"]) for r in acc]),
                         rss_peak_mb=max(r["rss_peak_mb"] for r in recs), gpu_peak_mb=max(r["gpu_peak_mb"] for r in recs)))
    (out / "report.json").write_text(json.dumps(rows, indent=1))
    hdr = "| family | scenes | rejected attempts | total s (p50/p95/max) | geometry | sources | proposal screening | accepted render | rate stream | proposals/scene | peak RSS MB | peak GPU MB |"
    print(hdr); print("|" + "---|" * 12)
    f = lambda v: "/".join(f"{x:g}" for x in v) if v else "—"
    for r in rows:
        print(f"| {r['family']} | {r['accepted']}/{r['scenes']} | {r['rejected_attempts']} {','.join(r['reasons'])} | {f(r['total'])} | {f(r['geometry'])} | "
              f"{f(r['sources'])} | {f(r['proposal_screening'])} | {f(r['accepted_render'])} | {f(r['rate_stream'])} | {f(r['proposals_per_scene'])} | "
              f"{r['rss_peak_mb']:.0f} | {r['gpu_peak_mb']:.0f} |")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#fcfcfb")
        keys = [("geometry", "#2a78d6"), ("sources", "#eb6834"), ("proposal_screening", "#1baf7a"), ("accepted_render", "#eda100"),
                ("rate_stream", "#e87ba4"), ("serialize", "#008300")]
        names = [r["family"] for r in rows]; bottom = np.zeros(len(rows))
        for k_, c in keys:
            v = np.array([r[k_][0] if r[k_] else 0 for r in rows])
            ax.barh(names, v, left=bottom, color=c, label=k_.replace("_", " "), edgecolor="#fcfcfb", linewidth=2); bottom += v
        ax.set_xlabel("median seconds per accepted scene (laptop RTX 3060)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor("#fcfcfb"); fig.tight_layout(); fig.savefig(out / "bench_breakdown.png", dpi=150)
    except Exception as e:                     # noqa: BLE001 (plot is optional)
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=8)
    p.add_argument("--families"); p.add_argument("--contract", default="v3", choices=list(CONTRACTS))
    p = sub.add_parser("one"); p.add_argument("--family", required=True); p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", required=True); p.add_argument("--contract", default="v3", choices=list(CONTRACTS))
    p = sub.add_parser("report"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(run=cmd_run, one=cmd_one, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
