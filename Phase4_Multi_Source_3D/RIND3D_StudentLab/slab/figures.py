"""Figures for the lab reports (matplotlib, light surface; palette from the dataviz reference: categorical slots in fixed
order, one-hue sequential blue for magnitudes).

python -m slab.figures posteriors <out.png> <row_idx,...> <name=run_dir> ...   max-projections of label / models
python -m slab.figures curves <out.png> <name=run_dir> ...                     dev learning curves
python -m slab.figures by_nf <out.png> <name=eval.npz> ...                     information gain by facet count
python -m slab.figures scaling <out.png> <name:n_scenes=eval.npz> ...          learning curve over training worlds
python -m slab.figures zoom <out.png> <row_idx,...> <name=run_dir[@override]> ... zoom on the true source (depth in plane)
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


def scaling(out, groups):
    """groups: {model label: {n_scenes: eval.npz}}; IG (side-balanced T*, scene-bootstrap 95% band) vs training worlds."""
    from .analyze import load, boot
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE); _style(ax)
    for i, (name, pts) in enumerate(groups.items()):
        xs = sorted(pts); m, lo, hi = [], [], []
        for x in xs:
            d = load(pts[x])["tstar"]; b = boot(d["ig"], d["scene"], d["Lp"], n=500)
            m.append(b["mean"]); lo.append(b["lo"]); hi.append(b["hi"])
        ax.fill_between(xs, lo, hi, color=SERIES[i], alpha=0.18, lw=0)
        ax.plot(xs, m, color=SERIES[i], lw=2, marker="o", ms=5, label=name)
        ax.annotate(f"{m[-1]:.2f}", (xs[-1], m[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks([86, 171, 342, 684]); ax.set_xticklabels(["86", "171", "342", "684"])
    ax.set_xlabel("independent training worlds (scenes)", color=INK2); ax.set_ylabel("dev T* information gain (nats)", color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURFACE)


def cf_panel(out, cfd_dir, analysis_json, run_dir, which="median"):
    """Depth counterfactual panel: the stored label and the X-identical worlds' labels (casters scaled by λ about the
    source), and the single prediction an X-only Student makes for all of them."""
    import torch
    from .data import LabTable, collate, to_dev
    from .evaluate import lab_model
    from .density import NC, NF
    A = json.load(open(analysis_json))["rows"]
    A = [r for r in A if r["jsd"] > 1e-3]
    A.sort(key=lambda r: r["jsd"]); r = A[len(A) // 2] if which == "median" else A[-1]
    T = LabTable(); i = r["i"]; row = T.rows[i]
    key = (int(row["scene"]), int(row["window"]), int(row["source"]))
    vols = []
    cells, q, _ = T.label(i); v = np.zeros(128 ** 3); v[cells] = q; vols.append(("world G (λ = 1)", v))
    for lam in r["lams"][1:]:
        z = np.load(Path(cfd_dir) / f"cfd_{key[0]:06d}_w{key[1]:03d}_s{key[2]}_l{lam:g}.npz")
        v = np.zeros(128 ** 3); np.add.at(v, z["cell"].astype(np.int64), z["lo"]); vols.append((f"casters × {lam:g} about s", v / v.sum()))
    f = lab_model(run_dir, "cuda"); B = to_dev(collate(T, np.array([i]), "onehot"), "cuda")
    with torch.no_grad():
        lp = f(B)[0]
    pv = lp.exp().view(NC, NC, NC, NF, NF, NF).permute(0, 3, 1, 4, 2, 5).reshape(-1).float().cpu().numpy()
    vols.append(("F1 (one prediction: same X)", pv))
    F = T.F[i]; ax_proj = int(np.argmin(np.abs(F[0, 3:6]))) if T.nf[i] > 0 else 2
    keep = [a for a in range(3) if a != ax_proj]
    fig, axes = plt.subplots(1, len(vols), figsize=(2.4 * len(vols), 2.7), facecolor=SURFACE, squeeze=False)
    s = np.array([row["sx"], row["sy"], row["sz"]]) * 128; o = np.array([row["ox"], row["oy"], row["oz"]]) / 8; L = row["Lp"] / 8
    for ax, (t, v) in zip(axes[0], vols):
        img = np.log10(np.maximum(v.reshape(128, 128, 128).max(ax_proj), 1e-9))
        ax.imshow(img.T, origin="lower", cmap=BLUES, vmin=-7, vmax=0, extent=(0, 128, 0, 128))
        ax.add_patch(plt.Rectangle((o[keep[0]], o[keep[1]]), L, L, fill=False, ec=SERIES[1], lw=1.2))
        ax.plot(s[keep[0]], s[keep[1]], marker="x", color=SERIES[7], ms=7, mew=1.6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8, color=INK)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.text(0.5, 0.01, f"L{int(row['Lp'])} window, {int(T.nf[i])} facets; JSD of the worlds = {r['jsd']:.2f} nats; every panel has the same "
             "observation X. log10 mass, max-projected; orange: window; red ×: true source", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out, dpi=160, facecolor=SURFACE)


def zoom(out, rows, runs, half=12):
    """Zoomed max-projections around the true source (±half cells), projected along the axis least aligned with the
    window-to-source direction, so the ill-conditioned depth direction lies in the image plane."""
    import torch
    from .data import LabTable, collate, to_dev
    from .evaluate import lab_model
    from .density import NC, NF
    T = LabTable(); dev = "cuda"
    models = {n: lab_model(d.split("@")[0], dev, override=json.loads(d.split("@")[1]) if "@" in d else None) for n, d in runs.items()}
    fig, axes = plt.subplots(len(rows), 1 + len(models), figsize=(2.3 * (1 + len(models)), 2.4 * len(rows)), facecolor=SURFACE,
                             squeeze=False)
    for ri, i in enumerate(rows):
        B = to_dev(collate(T, np.array([i]), "onehot"), dev)
        r = T.rows[i]
        s = np.array([r["sx"], r["sy"], r["sz"]]) * 128
        c = (np.array([r["ox"], r["oy"], r["oz"]]) + r["Lp"] / 2) / 8
        t = s - c; ax_proj = int(np.argmin(np.abs(t))); keep = [a for a in range(3) if a != ax_proj]
        cells, q, _ = T.label(i)
        vols = {}
        lab = np.zeros(128 ** 3); lab[cells] = q; vols["certified label"] = lab.reshape(128, 128, 128)
        for n, f in models.items():
            with torch.no_grad():
                lp = f(B)[0]
            vols[n] = lp.exp().view(NC, NC, NC, NF, NF, NF).permute(0, 3, 1, 4, 2, 5).reshape(128, 128, 128).float().cpu().numpy()
        lo = np.clip(np.floor(s[keep]).astype(int) - half, 0, 128 - 2 * half)
        for ci, (n, v) in enumerate(vols.items()):
            ax = axes[ri, ci]
            img = np.log10(np.maximum(v.max(ax_proj), 1e-9))[lo[0]:lo[0] + 2 * half, lo[1]:lo[1] + 2 * half]
            ax.imshow(img.T, origin="lower", cmap=BLUES, vmin=-5, vmax=0, extent=(lo[0], lo[0] + 2 * half, lo[1], lo[1] + 2 * half))
            d2 = t[keep] / np.linalg.norm(t[keep]) * half * 0.8           # depth direction (window -> source), in-plane part
            ax.annotate("", xy=(s[keep[0]] + d2[0] * 0.5, s[keep[1]] + d2[1] * 0.5), xytext=(s[keep[0]] - d2[0] * 0.5, s[keep[1]] - d2[1] * 0.5),
                        arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.0))
            ax.plot(s[keep[0]], s[keep[1]], marker="x", color=SERIES[7], ms=7, mew=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if ri == 0:
                ax.set_title(n, fontsize=9, color=INK)
        axes[ri, 0].set_ylabel(f"L{int(r['Lp'])}, {int(T.nf[i])} facets", fontsize=8, color=INK2)
    fig.text(0.5, 0.005, f"log10 cell mass, max-projected; {2 * half}x{2 * half} cells (8 voxels each) around the true source (red ×); "
             "orange arrow: window-to-source (depth) direction", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    if kind == "posteriors":
        rows = [int(x) for x in sys.argv[3].split(",")]
        posteriors(out, rows, dict(a.split("=", 1) for a in sys.argv[4:]))
    elif kind == "zoom":                          # zoom <out> <rows> name=run_dir[@override_json] ...
        zoom(out, [int(x) for x in sys.argv[3].split(",")], dict(a.split("=", 1) for a in sys.argv[4:]))
    elif kind == "curves":
        curves(out, dict(a.split("=", 1) for a in sys.argv[3:]))
    elif kind == "by_nf":
        by_nf(out, dict(a.split("=", 1) for a in sys.argv[3:]))
    elif kind == "cf":                            # cf <out> <cfd_dir> <analysis.json> <run_dir> [median|max]
        cf_panel(out, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else "median")
    elif kind == "scaling":                       # name:n=eval.npz ...
        g = {}
        for a in sys.argv[3:]:
            k, v = a.split("=", 1); nm, n = k.rsplit(":", 1); g.setdefault(nm, {})[int(n)] = v
        scaling(out, g)
