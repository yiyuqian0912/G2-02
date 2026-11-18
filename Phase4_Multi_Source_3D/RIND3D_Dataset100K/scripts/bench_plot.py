"""Figures of docs/GENERATOR_BENCHMARK_V3.md from a bench_families.py run (needs matplotlib).
usage: bench_plot.py BENCH_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = json.loads((B / "report.json").read_text())
order = ["rotated_boxes", "thin_slabs", "prisms", "wedges", "convex_polyhedra", "multiscale_convex", "corridors", "barriers", "occluder_clusters", "spheres", "spheres_polyhedra"]
rows = sorted(rows, key=lambda r: order.index(r["family"]))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
keys = [("geometry", "geometry", "#2a78d6"), ("proposal_screening", "proposal screening", "#eb6834"), ("accepted_render", "accepted-window rendering", "#1baf7a"),
        ("rate_stream", "rate stream (exact m)", "#eda100"), ("serialize", "serialize", "#e87ba4")]
fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURF)
names = [r["family"] for r in rows]; y = np.arange(len(rows))[::-1]; left = np.zeros(len(rows))
for k, lab, c in keys:
    v = np.array([r[k][0] if r[k] else 0 for r in rows])
    ax.barh(y, v, left=left, color=c, label=lab, edgecolor=SURF, linewidth=2, height=0.7); left += v
for yi, r in zip(y, rows):
    ax.text(r["total"][0] + 0.4, yi, f"{r['total'][0]:.1f} s", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(names, color=INK, fontsize=9)
ax.set_xlabel("median seconds per accepted scene (stage medians; total median at the bar end)", color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right", labelcolor=INK)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2)
ax.set_facecolor(SURF); ax.set_title("RIND3D v3 physical scene generation by family (laptop RTX 3060, no labels)", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out / "bench_breakdown.png", dpi=150, facecolor=SURF)
# distribution of total scene time
fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=SURF)
for i, r in enumerate(rows):
    recs = [json.loads(l) for l in (B / f"{'curved' if r['family'].startswith('sphere') else 'v3'}_{r['family']}.jsonl").read_text().splitlines() if l.strip()]
    w = [x["wall"] for x in recs]
    ax.scatter(np.full(len(w), i) + np.linspace(-.15, .15, len(w)), w, s=22, color="#2a78d6", edgecolor=SURF, linewidth=1, zorder=3)
    ax.plot([i - .25, i + .25], [np.median(w)] * 2, color=INK, lw=2, zorder=4)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["family"].replace("_", "\n") for r in rows], fontsize=7.5, color=INK)
ax.set_yscale("log"); from matplotlib.ticker import FixedLocator, FixedFormatter; ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50])); ax.yaxis.set_major_formatter(FixedFormatter(["2", "3", "5", "10", "20", "30", "50"])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.set_ylabel("seconds per scene (log)", color=INK2, fontsize=9)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_facecolor(SURF); ax.set_title("every measured scene (dots) and the median (bar)", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out / "bench_scenes.png", dpi=150, facecolor=SURF)
print("ok")
