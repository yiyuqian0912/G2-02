#!/usr/bin/env python3
"""Optional publication-style figure. Requires NumPy and Matplotlib.

The probability checks themselves have no external dependencies.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rind-proof-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

import verify as v


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    colors = ["#e6e8ed", "#f28a64", "#76a9e8", "#f5d479"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    x = np.linspace(-3, 3, 700)
    xx, yy = np.meshgrid(x, x)
    labels = (yy > xx).astype(int) + 2*(yy < 2*xx).astype(int)
    ax = axes[0]
    ax.imshow(labels, origin="lower", extent=(-3, 3, -3, 3), cmap=cmap, norm=norm,
              interpolation="nearest")
    for slope in (1, 2):
        ax.plot(x, slope*x, color="#344054", lw=1.1)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", xlabel="x / h", ylabel="y / h",
           title="A. Continuous source subsets")
    ax.legend(handles=[Patch(color=colors[i], label=label) for i, label in enumerate(
        ("Neither: R=0", "A only: R=1", "B only: R=1", "Both: R=2"))],
        loc="upper left", fontsize=8, framealpha=.95)
    ax.text(.5, -.19, "Singleton regions meet only at one point.\nNo shared boundary segment.",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)

    phase = (.37, .64)
    coords_x = np.array([phase[0]+k for k in range(-3, 3)])
    coords_y = np.array([phase[1]+k for k in range(-3, 3)])
    gx, gy = np.meshgrid(coords_x, coords_y)
    bits = (gy > gx).astype(int) + 2*(gy < 2*gx).astype(int)
    brightness = (bits & 1) + ((bits >> 1) & 1)
    ax = axes[1]
    brightness_cmap = ListedColormap([colors[0], "#86a6c7", colors[3]])
    extent = (coords_x[0]-.5, coords_x[-1]+.5, coords_y[0]-.5, coords_y[-1]+.5)
    ax.imshow(brightness, origin="lower", extent=extent, cmap=brightness_cmap,
              vmin=-.5, vmax=2.5, interpolation="nearest")
    for row, y in enumerate(coords_y):
        for col, x0 in enumerate(coords_x):
            ax.text(x0, y, str(brightness[row, col]), ha="center", va="center", fontsize=12)
    edges = v.bad_edges(v.affine_label, 1., phase,
                        (coords_x[0]-.01, coords_x[-1]+.01, coords_y[0]-.01, coords_y[-1]+.01))
    for edge in edges:
        a, b = edge["origin"], edge["end"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#c12639", lw=5, solid_capstyle="round")
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="x / h", ylabel="y / h",
           title="B. Student sees summed brightness")
    ax.text(.5, -.19, "Red edge joins different singleton subsets.\nBoth endpoint brightness values are 1.",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)

    ax = axes[2]
    spacings = np.linspace(.0005, 2., 700)
    ax.plot(spacings, [v.strip_probability(.25, h) for h in spacings],
            color="#396eab", lw=2.3, label=r"Fixed gap $|d|=L/4$")
    ax.plot(spacings, [v.uniform_gap_probability(h, 1.) for h in spacings],
            color="#b45338", lw=2.3, label=r"Random gap $D\sim U[-L,L]$")
    ax.set(xlim=(0, 2), ylim=(0, 1), xlabel="Pixel spacing h / L",
           ylabel="False-connection probability", title="C. Exact parallel-strip benchmark")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(.5, -.19, "These decaying strip laws do not extend\nto arbitrary crossing-shadow scenes.",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)
    fig.suptitle("Equal brightness: geometric contact and sampled connectivity are different events",
                 fontsize=14, fontweight="bold")
    folder = Path(__file__).with_name("figures")
    folder.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"ambiguity.{suffix}", dpi=200)
    svg = folder / "ambiguity.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    print(folder.resolve())


if __name__ == "__main__":
    main()
