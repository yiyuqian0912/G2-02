"""Figures for the lab reports (matplotlib, light surface; palette from the dataviz reference: categorical slots in fixed
order, one-hue sequential blue for magnitudes).

python -m slab.figures posteriors <out.png> <row_idx,...> <name=run_dir> ...   max-projections of label / models
python -m slab.figures curves <out.png> <name=run_dir> ...                     dev learning curves
python -m slab.figures by_nf <out.png> <name=eval.npz> ...                     information gain by facet count
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("seqblue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def curves(out, runs):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE); _style(ax)
    for i, (name, d) in enumerate(runs.items()):
        c = [json.loads(l) for l in open(Path(d) / "curve.jsonl")]
        ax.plot([x["step"] for x in c], [x["dev_nll"] for x in c], color=SERIES[i % 8], lw=2, marker="o", ms=4, label=name)
    ax.set_xlabel("training step", color=INK2); ax.set_ylabel("dev one-hot NLL (nats, 1,024 T* rows)", color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURFACE)


def by_nf(out, evals, s="tstar"):
    from .analyze import load
    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor=SURFACE); _style(ax)
    names = list(evals); E = {n: load(p)[s] for n, p in evals.items()}
    ks = [0, 1, 2, 3, 4]; w = 0.8 / (len(names) + 1)
    ref = E[names[0]]
    for i, n in enumerate(names + ["label (full world)"]):
        d = ref if n.startswith("label") else E[n]
        nf = np.minimum(d["nf"], 4)
        v = [np.nanmean(d["ig_ref"][nf == k]) if n.startswith("label") else np.nanmean(d["ig"][nf == k]) for k in ks]
        col = INK2 if n.startswith("label") else SERIES[i % 8]
        ax.bar(np.array(ks) + (i - len(names) / 2) * w, v, w * 0.92, color=col, label=n, alpha=0.55 if n.startswith("label") else 1)
    ax.set_xticks(ks); ax.set_xticklabels(["0", "1", "2", "3", "≥4"])
    cnt = np.bincount(np.minimum(ref["nf"], 4), minlength=5)
    for k in ks:
        ax.text(k, -0.9, f"n={cnt[k]}", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("X-measured interface facets in the window", color=INK2); ax.set_ylabel("information gain (nats)", color=INK2)
    ax.set_ylim(-1.2, None); ax.legend(frameon=False, fontsize=8, ncol=3, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURFACE)


def posteriors(out, rows, runs):
    """Max-projection of the cell masses (log10) along the axis most parallel to the window's largest facet, so a
    sheet appears as a line. Columns: certified label, then each model. White cross: true source."""
    import torch
    from .data import LabTable, collate, to_dev, cf_of_cell
    from .evaluate import lab_model, pilot_model
    T = LabTable(); dev = "cuda"
    models = {n: (pilot_model(d[6:], dev) if d.startswith("pilot:") else lab_model(d, dev)) for n, d in runs.items()}
    fig, axes = plt.subplots(len(rows), 1 + len(models), figsize=(2.3 * (1 + len(models)), 2.3 * len(rows)), facecolor=SURFACE,
                             squeeze=False)
    for ri, i in enumerate(rows):
        B = to_dev(collate(T, np.array([i]), "onehot"), dev)
        r = T.rows[i]; F = T.F[i]; ax_proj = int(np.argmin(np.abs(F[0, 3:6]))) if T.nf[i] > 0 else 2
        cells, q, _ = T.label(i)
        vols = {}
        lab = np.zeros(128 ** 3); lab[cells] = q; vols["certified label"] = lab.reshape(128, 128, 128)
        for n, f in models.items():
            with torch.no_grad():
                lp = f(B)[0]                                                 # (4096, 512)
            from .density import NC, NF
            v = lp.exp().view(NC, NC, NC, NF, NF, NF).permute(0, 3, 1, 4, 2, 5).reshape(128, 128, 128)
            vols[n] = v.float().cpu().numpy()
        s = np.array([r["sx"], r["sy"], r["sz"]]) * 128
        keep = [a for a in range(3) if a != ax_proj]
        for ci, (n, v) in enumerate(vols.items()):
            ax = axes[ri, ci]
            img = np.log10(np.maximum(v.max(ax_proj), 1e-9))
            ax.imshow(img.T, origin="lower", cmap=BLUES, vmin=-7, vmax=0, extent=(0, 128, 0, 128))
            o = np.array([r["ox"], r["oy"], r["oz"]]) / 8; L = r["Lp"] / 8
            ax.add_patch(plt.Rectangle((o[keep[0]], o[keep[1]]), L, L, fill=False, ec=SERIES[1], lw=1.2))
            ax.plot(s[keep[0]], s[keep[1]], marker="x", color=SERIES[7], ms=7, mew=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if ri == 0:
                ax.set_title(n, fontsize=9, color=INK)
        axes[ri, 0].set_ylabel(f"L{int(r['Lp'])}, {int(T.nf[i])} facets", fontsize=8, color=INK2)
    fig.text(0.5, 0.005, "log10 cell mass, max-projected along the axis closest to the main facet plane; orange: window; red ×: true source",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    if kind == "posteriors":
        rows = [int(x) for x in sys.argv[3].split(",")]
        posteriors(out, rows, dict(a.split("=", 1) for a in sys.argv[4:]))
    elif kind == "curves":
        curves(out, dict(a.split("=", 1) for a in sys.argv[3:]))
    elif kind == "by_nf":
        by_nf(out, dict(a.split("=", 1) for a in sys.argv[3:]))
