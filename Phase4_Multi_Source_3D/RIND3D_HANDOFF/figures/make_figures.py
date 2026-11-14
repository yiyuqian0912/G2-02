"""Figures of RIND3D_数据集设计与技术说明 (matplotlib venv; reads figures/data/ from extract_data.py).

  $RIND3D_TORCH_PY RIND3D_HANDOFF/figures/make_figures.py [--only 01,04] [--out RIND3D_HANDOFF/figures/out]

Writes <name>.svg, <name>.pdf and <name>.png (300 dpi) per figure, contact_sheet.png, and copies the PNGs to
docs/assets/. Figures are drawn at the document's text width (config width_in = 5.83 in = 14.8 cm), so font sizes
are the printed sizes. No titles or decorative bands: the captions live in the document."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib import colors as mcolors                            # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = json.loads((HERE / "config.json").read_text())
DATA = HERE / "data"
W_IN = float(CFG["width_in"])

# ---------------------------------------------------------------- style (one palette for every figure)
C = dict(lit="#E8B04B", dark="#22344F", solid="#9AA3AE", free="#F4F1EA", teal="#1F7A7A", red="#C2453D",
         B="#C9D3DE", ink="#1E2A36", ink2="#56616D", muted="#8A949E", grid="#E3E7EB", panel="#F6F8F9",
         src_other="#B8C0C8", body="#8E99A6")
CLS = mcolors.ListedColormap([C["dark"], C["lit"], C["solid"]])          # cls 0 dark FREE, 1 lit FREE, 2 SOLID
SEQ = mcolors.LinearSegmentedColormap.from_list("q", ["#EEF4F1", "#9CCFC2", "#2E8C8A", "#1B4F6B", "#14213D"])

plt.rcParams.update({
    "font.family": CFG["font"], "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.8, "axes.edgecolor": C["muted"],
    "axes.linewidth": 0.6, "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"],
    "axes.labelcolor": C["ink"], "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "mathtext.fontset": "dejavusans", "svg.fonttype": "path", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False, "svg.hashsalt": "rind3d",
})


def unpack(bits, L):
    return np.unpackbits(np.asarray(bits, np.uint8), count=L ** 3).astype(bool).reshape(L, L, L)   # [z, y, x]


def crop(name):
    z = np.load(DATA / "crops.npz"); w, s, L = z[f"{name}_meta"]
    return int(w), int(s), unpack(z[f"{name}_lit"], int(L)), unpack(z[f"{name}_solid"], int(L))


def sci(x, d=1):
    e = int(np.floor(np.log10(abs(x)))); m_ = x / 10 ** e
    return rf"${m_:.{d}f}\times10^{{{e}}}$"


def hull2d(P):
    P = np.unique(np.round(P, 9), axis=0); P = P[np.lexsort((P[:, 1], P[:, 0]))]
    if len(P) < 3:
        return P
    def half(pts):
        h = []
        for p_ in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p_[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p_[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p_)
        return h
    lo, up = half(P), half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def cells_xyz(c):
    c = np.asarray(c, np.int64)
    return c // 16384, (c // 128) % 128, c % 128


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    for ext, md in (("svg", {"Date": None}), ("pdf", {"CreationDate": None, "ModDate": None}), ("png", {})):
        fig.savefig(out / f"{name}.{ext}", facecolor="white", metadata=md)      # reproducible bytes (no timestamps)
    plt.close(fig)
    assets = REPO / "docs" / "assets"; assets.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / f"{name}.png", assets / f"{name}.png")


# ---------------------------------------------------------------- 3D helpers
def facets(V, F):
    """Merge coplanar hull triangles into polygons (display)."""
    tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    ctr = V.mean(0); out = []
    keys = {}
    for i, nn in enumerate(n):
        if np.dot(nn, tri[i, 0] - ctr) < 0:
            nn = -nn
        k = tuple(np.round(np.r_[nn, np.dot(nn, tri[i, 0])], 6)); keys.setdefault(k, []).append(i)
    for k, idx in keys.items():
        pts = np.unique(np.round(tri[idx].reshape(-1, 3), 12), axis=0)
        c = pts.mean(0); nn = np.array(k[:3]); u = pts[0] - c; u /= np.linalg.norm(u); v = np.cross(nn, u)
        ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
        out.append((pts[np.argsort(ang)], nn))
    return out


def cls_xyz(lit, sol):
    """bool [z, y, x] crops -> cls[x, y, z]."""
    c = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(c, (2, 1, 0)))


# ---------------------------------------------------------------- ray-cast helpers (2D axes)
import render3d as R3                                               # noqa: E402


def scene_bodies(S):
    out = []
    for b in range(len(S["voff"]) - 1):
        out.append(R3.halfspaces_from_facets(facets(S["V"][S["voff"][b]:S["voff"][b + 1]], S["F"][S["foff"][b]:S["foff"][b + 1]])))
    return out


def img_axes(fig, rect, cam):
    ax = fig.add_axes(rect); ax.set_xlim(0, cam.W); ax.set_ylim(cam.H, 0); ax.set_axis_off(); ax.set_aspect("equal")
    return ax


def draw_segments(ax, cam, segs, depth, color, lw, hidden=None, n=300, zorder=3):
    for P0, P1 in segs:
        vis, uv = R3.visible_polyline(cam, np.asarray(P0, float), np.asarray(P1, float), depth, n=n)
        if hidden is not None:
            ax.plot(uv[:, 0], uv[:, 1], color=color, lw=lw * 0.6, alpha=hidden, ls=(0, (1.5, 1.5)), zorder=zorder - 1,
                    solid_capstyle="round")
        for pl in vis:
            ax.plot(pl[:, 0], pl[:, 1], color=color, lw=lw, zorder=zorder, solid_capstyle="round")


def draw_points(ax, cam, P, depth, **kw):
    uv, z = cam.project(P)
    i = np.clip(uv[:, 1].astype(int), 0, cam.H - 1); j = np.clip(uv[:, 0].astype(int), 0, cam.W - 1)
    vis = z <= depth[i, j] + 1e-3
    ax.scatter(uv[vis, 0], uv[vis, 1], **kw)
    kw2 = dict(kw); kw2["facecolors"] = "none"; kw2.pop("c", None); kw2["edgecolors"] = kw.get("edgecolors", C["ink2"])
    if np.any(~vis):
        ax.scatter(uv[~vis, 0], uv[~vis, 1], **kw2)
    return uv


def fig01(out):
    S = np.load(DATA / "scene.npz"); m = np.load(DATA / "main_task.npz"); rays = np.load(DATA / "rays.npy")
    w, s, lit, sol = crop("main"); Wr = S["windows"][w]; o = Wr[1:4] / 1024; Lw = Wr[4] / 1024
    src = S["sources"]; s0 = src[s]
    fig = plt.figure(figsize=(W_IN, 2.45))
    cam = R3.Camera(elev=18, azim=-62, size=(760, 760), half_width=0.84)
    img, depth = R3.render_bodies(cam, scene_bodies(S), "#A3ADB8")
    # (a) full world
    ax = img_axes(fig, [0.0, 0.0, 0.37, 0.80], cam); ax.imshow(img, interpolation="lanczos", zorder=1)
    draw_segments(ax, cam, R3.cube_segments((0, 0, 0), 1), depth, C["muted"], 0.5, hidden=0.5)
    draw_segments(ax, cam, [(s0, r[:3]) for r in rays if r[4] == 1], depth, C["lit"], 0.55)
    draw_segments(ax, cam, [(s0, s0 + r[3] * (r[:3] - s0)) for r in rays if r[4] == 0], depth, C["dark"], 0.55)
    draw_segments(ax, cam, R3.cube_segments(o, Lw), np.full_like(depth, np.inf), C["teal"], 0.9, zorder=6)
    others = np.array([p for j, p in enumerate(src) if j != s])
    draw_points(ax, cam, others, depth, s=14, c=C["src_other"], edgecolors=C["ink2"], linewidths=0.4, zorder=7)
    uv = draw_points(ax, cam, s0[None], depth, s=34, c=C["lit"], edgecolors=C["ink"], linewidths=0.6, zorder=8)
    ax.text(uv[0, 0] + 14, uv[0, 1] - 8, r"$s_j$", fontsize=7.5)
    uw, _ = cam.project(o + Lw / 2)
    ax.annotate("窗口 W", xy=(uw[0, 0], uw[0, 1]), xytext=(uw[0, 0] - 250, uw[0, 1] - 40), fontsize=6.8, color=C["teal"],
                arrowprops=dict(arrowstyle="-", color=C["teal"], lw=0.5), zorder=9)
    fig.text(0.035, 0.905, "完整世界（仅标注阶段可见）", fontsize=7.5, va="bottom")
    fig.text(0.03, 0.835, "全局几何 G、全部 K 个源；亮 / 暗射线", fontsize=6.4, color=C["ink2"], va="bottom")
    # (b) Student input
    cb_ = R3.Camera(elev=24, azim=-50, size=(420, 420), half_width=0.98)
    axb = img_axes(fig, [0.385, 0.20, 0.215, 0.52], cb_)
    axb.imshow(R3.render_cutaway(cb_, cls_xyz(lit, sol), cut=False), interpolation="lanczos")
    fig.text(0.4925, 0.905, "Student 输入 X", ha="center", fontsize=7.5, va="bottom")
    fig.text(0.4925, 0.835, "只含局部窗口", ha="center", fontsize=6.4, color=C["ink2"], va="bottom")
    fig.text(0.4925, 0.135, r"窗口内 SOLID（本例为空）", ha="center", fontsize=6.3, color=C["ink2"])
    fig.text(0.4925, 0.065, r"单源亮 / 暗 $O_j$、$L=64$、原点", ha="center", fontsize=6.3, color=C["ink2"])
    # (c) posterior
    q = 0.5 * (m["ex_lo"] + m["ex_hi"]); q = q / q.sum()
    norm = mcolors.LogNorm(vmin=q.max() * 1e-3, vmax=q.max())
    pimg, pdepth = R3.render_voxels(cam, dict(zip(m["ex_cell"].tolist(), q.tolist())), SEQ, norm)
    axc = img_axes(fig, [0.615, 0.06, 0.37, 0.80], cam)
    axc.imshow(img * np.array([1, 1, 1, 0.18]), interpolation="lanczos", zorder=1)
    axc.imshow(pimg, interpolation="lanczos", zorder=2)
    dd = np.minimum(pdepth, np.inf)
    draw_segments(axc, cam, R3.cube_segments((0, 0, 0), 1), dd, C["muted"], 0.5, hidden=0.5)
    draw_segments(axc, cam, R3.cube_segments(o, Lw), np.full_like(dd, np.inf), C["teal"], 0.9, zorder=6)
    ut = draw_points(axc, cam, m["true_source"][None], np.full_like(dd, np.inf), s=46, marker="*", c=C["lit"],
                     edgecolors=C["ink"], linewidths=0.5, zorder=8)
    fig.text(0.635, 0.905, r"监督标签：全局后验 $q_F$", fontsize=7.5, va="bottom")
    fig.text(0.635, 0.835, "128³ 网格上 3,862 个非零格；★ 真实源", fontsize=6.4, color=C["ink2"], va="bottom")
    cax = fig.add_axes([0.80, 0.055, 0.16, 0.02])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cbar.outline.set_linewidth(0.4); cbar.ax.tick_params(labelsize=5.6, width=0.4, length=1.8, pad=1)
    cax.text(-0.04, 0.5, "网格质量", fontsize=6, ha="right", va="center", transform=cax.transAxes)
    for x0, x1 in ((0.355, 0.395), (0.585, 0.625)):
        fig.add_artist(FancyArrowPatch((x0, 0.46), (x1, 0.46), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, color=C["teal"], lw=0.9))
    for x, l in ((0.005, "a"), (0.385, "b"), (0.615, "c")):
        fig.text(x, 0.905, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    save(fig, "01_research_task", out)


# ---------------------------------------------------------------- figure 04: the four response types
def boundary_voxels(cls):
    """FREE voxels with a 6-adjacent FREE voxel of the other class (cls[x, y, z])."""
    L = cls.shape[0]; b = np.zeros(cls.shape, bool)
    for ax_ in range(3):
        a = [slice(None)] * 3; c = [slice(None)] * 3; a[ax_] = slice(0, L - 1); c[ax_] = slice(1, L)
        A, Bq = cls[tuple(a)], cls[tuple(c)]
        pair = (A != 2) & (Bq != 2) & (A != Bq)
        b[tuple(a)] |= pair; b[tuple(c)] |= pair
    return b


def fig04(out):
    L_ = json.loads((DATA / "LOG.json").read_text())["channels"]
    cols = [("all_lit", "全亮", "所有 FREE 接收点为亮"), ("all_dark", "全暗", "所有 FREE 接收点为暗"),
            ("solid_separated", "SOLID 分隔的明暗", "有亮有暗，但无 6-邻接亮 / 暗对"),
            ("main", "直接明暗交界", "存在 6-邻接的亮 / 暗 FREE 对")]
    data = {k: crop(k) for k, _, _ in cols}
    # one slice plane for window 6: the plane where lit, dark and SOLID are most balanced in the solid-separated channel
    _, _, lit, sol = data["solid_separated"]; cs = cls_xyz(lit, sol); best = (-1, 2, 32)
    for a_ in range(3):
        for k in range(64):
            sl = np.take(cs, k, axis=a_); sc = min((sl == 0).sum(), (sl == 1).sum(), (sl == 2).sum())
            if sc > best[0]:
                best = (sc, a_, k)
    _, pa, pk = best
    fig = plt.figure(figsize=(W_IN, 3.2))
    cam = R3.Camera(elev=26, azim=-48, size=(420, 420), half_width=0.98)
    xs = [0.005 + 0.25 * i for i in range(4)]
    for i, (key, name, rule) in enumerate(cols):
        w, s, lit, sol = data[key]; cls = cls_xyz(lit, sol)
        ax = img_axes(fig, [xs[i] + 0.025, 0.585, 0.19, 0.30], cam)
        ax.imshow(R3.render_cutaway(cam, cls, cut=True), interpolation="lanczos")
        fig.text(xs[i] + 0.12, 0.935, name, ha="center", fontsize=8)
        fig.text(xs[i] + 0.12, 0.905, rule, ha="center", fontsize=6.1, color=C["ink2"], va="top")
        # slice
        bv = boundary_voxels(cls)
        if w == data["solid_separated"][0]:
            a_, k = pa, pk
        elif bv.any():
            a_, k = max(((a2, k2) for a2 in range(3) for k2 in range(64)),
                        key=lambda t: min((np.take(cls, t[1], axis=t[0]) == 0).sum(), (np.take(cls, t[1], axis=t[0]) == 1).sum()))
        else:
            a_, k = 2, 32
        sl = np.take(cls, k, axis=a_); bd = np.take(bv, k, axis=a_)
        axs = fig.add_axes([xs[i] + 0.05, 0.235, 0.14, 0.14 * W_IN / 3.2])
        axs.imshow(sl.T, cmap=CLS, vmin=0, vmax=2, origin="lower", interpolation="nearest")
        img2 = sl.T                                                   # [row = second axis, col = first axis]
        segs = []
        hz = (img2[:, :-1] != 2) & (img2[:, 1:] != 2) & (img2[:, :-1] != img2[:, 1:])
        for r_, c_ in zip(*np.nonzero(hz)):
            segs.append([(c_ + 0.5, r_ - 0.5), (c_ + 0.5, r_ + 0.5)])
        vt = (img2[:-1, :] != 2) & (img2[1:, :] != 2) & (img2[:-1, :] != img2[1:, :])
        for r_, c_ in zip(*np.nonzero(vt)):
            segs.append([(c_ - 0.5, r_ + 0.5), (c_ + 0.5, r_ + 0.5)])
        if segs:
            from matplotlib.collections import LineCollection
            axs.add_collection(LineCollection(segs, colors=C["red"], linewidths=1.1, capstyle="projecting"))
        axs.set_xticks([]); axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_linewidth(0.4); sp.set_color(C["muted"])
        lab = L_[key]["label"]
        if lab is None:
            note = "保存响应，不生产标签\n独立 OOD 评估"
        else:
            note = f"FastLabel：{lab['n_cells']:,} 个非零格\n{lab['wall']:.1f} s（{'统计认证' if lab['status'] == 'certified_stat' else '确定性'}）"
        fig.text(xs[i] + 0.12, 0.205, note, ha="center", va="top", fontsize=6.1, color=C["ink2"], linespacing=1.35)
        fig.text(xs[i] + 0.12, 0.535, f"二维切片 {'xyz'[a_]} = {k}", ha="center", fontsize=5.8, color=C["muted"])
    # legend row
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(fc=C["lit"], ec="none", label="亮 FREE"), Patch(fc=C["dark"], ec="none", label="暗 FREE"),
         Patch(fc=C["solid"], ec="none", label="SOLID"),
         Line2D([], [], lw=1.2, color=C["red"], label="亮 / 暗 FREE 直接相邻的面（切片内）")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, handlelength=1.0, handletextpad=0.4,
               columnspacing=1.2, fontsize=6.3)
    axi = fig.add_axes([0.735, 0.012, 0.05, 0.05 * W_IN / 3.2]); axi.set_axis_off(); axi.set_xlim(-1.5, 1.5); axi.set_ylim(-1.5, 1.5)
    for (dx, dy), col in [((0, 0), C["lit"]), ((1, 0), C["dark"]), ((-1, 0), C["dark"]), ((0, 1), C["dark"]), ((0, -1), C["dark"])]:
        axi.add_patch(Rectangle((dx - 0.46, dy - 0.46), 0.92, 0.92, fc=col, ec="none"))
    fig.text(0.79, 0.058, "6-邻接：共享一个面\n（图示 ±x, ±y；另有 ±z）", fontsize=5.9, color=C["ink2"], va="center", linespacing=1.25)
    for xx_ in (0.255, 0.505, 0.755):
        fig.add_artist(plt.Line2D([xx_, xx_], [0.14, 0.95], transform=fig.transFigure, color=C["grid"], lw=0.6))
    save(fig, "04_response_types", out)


# ---------------------------------------------------------------- figure 08: FastLabel
def column_map(cells, vals, box):
    ix, iy, iz = cells_xyz(cells); (x0, x1), (y0, y1) = box
    M = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
    np.add.at(M, (iy - y0, ix - x0), np.asarray(vals, float))
    return M


def flow_box(fig, x, y, w, h, text, fc, ec=None, fs=6.0, color=None):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008", transform=fig.transFigure,
                                  fc=fc, ec=ec or "none", lw=0.6))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"], linespacing=1.3)


def flow_arrow(fig, p0, p1, color=None, style="-|>"):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style, mutation_scale=6,
                                   color=color or C["ink2"], lw=0.7, shrinkA=0, shrinkB=0))


def fig08(out):
    m = np.load(DATA / "main_task.npz"); LG = json.loads((DATA / "LOG.json").read_text())["main"]
    ix, iy, iz = cells_xyz(m["r0_cells"]); box = ((ix.min() - 2, ix.max() + 1), (iy.min() - 2, iy.max() + 9))
    vcell = (1 / 128) ** 3
    F = column_map(m["ex_cell"], 0.5 * (m["ex_lo"] + m["ex_hi"]), box)
    vf = float(np.sum(0.5 * (m["ex_lo"] + m["ex_hi"])))
    fig = plt.figure(figsize=(W_IN, 3.35))
    gray = mcolors.LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#DCE3EA", "#9FAFBF", "#5E7286"])
    vmax = max(column_map(m[f"r{r}_cells"], m[f"r{r}_hi"], box).max() for r in range(2)) / vcell
    ext = (box[0][0] / 128, (box[0][1] + 1) / 128, box[1][0] / 128, (box[1][1] + 1) / 128)
    for r in range(2):
        ax = fig.add_axes([0.055 + 0.245 * r, 0.43, 0.205, 0.45])
        B = column_map(m[f"r{r}_cells"], m[f"r{r}_hi"], box) / vcell
        ax.imshow(np.ma.masked_equal(B, 0), cmap=gray, vmin=0, vmax=vmax, origin="lower", extent=ext, interpolation="nearest")
        ax.contour((F > 0).astype(float), levels=[0.5], colors=[C["teal"]], linewidths=0.8, origin="lower", extent=ext)
        P = m[f"r{r}_pts"]; ok = m[f"r{r}_ok"]
        ax.scatter(P[ok, 0], P[ok, 1], s=2.2 if r else 5, c=C["teal"], linewidths=0, alpha=0.9, zorder=3)
        ax.scatter(P[~ok, 0], P[~ok, 1], s=7 if r else 6, c=C["red"], marker="x", linewidths=0.6, zorder=4)
        ax.scatter(*m["true_source"][:2], s=40, marker="*", c=C["lit"], edgecolors=C["ink"], linewidths=0.4, zorder=5)
        vb = float(m[f"r{r}_hi"].sum()); K = int((~ok).sum())
        ax.set_title(f"第 {r + 1} 轮：$B_{r + 1}\\supseteq F$", fontsize=7.5, loc="left", pad=3)
        ax.text(0.97, 0.97, f"Vol(B)/Vol(F) = {vb / vf:.3g}\nK / n = {K} / {len(ok)}", transform=ax.transAxes,
                fontsize=5.9, color=C["ink2"], va="top", ha="right", linespacing=1.3)
        ax.set_xlabel("x", fontsize=6.5, labelpad=1); ax.set_ylabel("y", fontsize=6.5, labelpad=1) if r == 0 else None
        ax.tick_params(labelsize=5.6, pad=1.5)
        ax.set_aspect("auto")
    # certificate test
    axc = fig.add_axes([0.565, 0.43, 0.185, 0.45])
    a1, a2 = LG["meta"]["a1"], LG["meta"]["a2"]; ok1 = m["r1_ok"]; ok0 = m["r0_ok"]
    ns = [128, 512, 1024]; K1 = [int((~ok1[:n]).sum()) for n in ns]
    axc.plot([0, 512], [a1, a1], color=C["ink2"], lw=0.7, ls=(0, (3, 2)))
    axc.plot([0, 1024], [a2, a2], color=C["ink2"], lw=0.7, ls=(0, (3, 2)))
    axc.text(20, a1 * 0.62, f"$a_1={a1}$（$n_1=512$）", fontsize=5.8, color=C["ink2"], va="center")
    axc.text(1080, a2 * 1.45, f"$a_2={a2}$（$n=1024$）", fontsize=5.8, color=C["ink2"], ha="right", va="center")
    axc.plot(ns, K1, "-o", color=C["teal"], ms=3, lw=1.0)
    axc.plot([128], [int((~ok0).sum())], "x", color=C["red"], ms=5, mew=1.0)
    axc.text(175, int((~ok0).sum()), f"第 1 轮：K = {int((~ok0).sum())} > $a_2$\n提前停止，细化 B", fontsize=5.7, color=C["red"],
             va="center", linespacing=1.25)
    axc.text(330, 3.1, "第 2 轮", fontsize=5.9, color=C["teal"])
    axc.set_yscale("log"); axc.set_ylim(1, 400); axc.set_xlim(0, 1100); axc.set_xticks([128, 512, 1024])
    axc.set_xlabel("已检验样本数", fontsize=6.3, labelpad=1); axc.set_ylabel("累计失败 K", fontsize=6.3, labelpad=1)
    axc.tick_params(labelsize=5.6, pad=1.5)
    axc.set_title("两阶段二项检验", fontsize=7.5, loc="left", pad=3)
    axc.text(1060, K1[-1] * 0.6, f"K = {K1[-1]} ≤ $a_2$\n通过", fontsize=5.8, color=C["teal"], ha="right", va="top",
             linespacing=1.2)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    # aggregation
    axq = fig.add_axes([0.80, 0.43, 0.185, 0.45])
    Q = column_map(m["fl_cell"], m["fl_lo"], box); Q = Q / Q.sum()
    nq = mcolors.LogNorm(vmin=Q[Q > 0].max() * 1e-3, vmax=Q.max())
    axq.imshow(np.ma.masked_equal(Q, 0), cmap=SEQ, norm=nq, origin="lower", extent=ext, interpolation="nearest")
    axq.set_title("$q_B$：按 128³ 网格聚合", fontsize=7.5, loc="left", pad=3)
    axq.text(0.97, 0.97, f"{len(m['fl_cell']):,} 格，沿 z 求和\nTV($q_B$, $q_F$) = {LG['tv_fastlabel_vs_exact']:.4f}",
             transform=axq.transAxes, fontsize=5.9, color=C["ink2"], va="top", ha="right", linespacing=1.3)
    axq.set_xlabel("x", fontsize=6.5, labelpad=1); axq.tick_params(labelsize=5.6, pad=1.5); axq.set_aspect("auto")
    for x, l in ((0.005, "a"), (0.25, "b"), (0.515, "c"), (0.765, "d")):
        fig.text(x, 0.91, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    # flow row
    y0, h = 0.095, 0.125
    boxes = [(0.01, 0.14, "局部观测 O", C["panel"]),
             (0.185, 0.18, "第一阶段 tau-pencil\nreceiver 子集 → B ⊇ F", C["panel"]),
             (0.41, 0.13, "在 B 中\n均匀采样", C["panel"]),
             (0.585, 0.17, "GPU 精确核验\nFP32 → FP64 区间 → GMP", C["panel"]),
             (0.80, 0.19, "两阶段二项检验\n通过：$q_B$，TV ≤ 0.05", "#E3F1EE")]
    for x, w_, t, fc in boxes:
        flow_box(fig, x, y0, w_, h, t, fc)
    for i in range(len(boxes) - 1):
        x, w_ = boxes[i][0], boxes[i][1]; flow_arrow(fig, (x + w_ + 0.002, y0 + h / 2), (boxes[i + 1][0] - 0.002, y0 + h / 2))
    # feedback and deterministic exits
    yf = y0 + h + 0.045
    fig.text(0.545, yf + 0.012, "未通过：失败样本违背的 receiver 加入约束，进入下一轮", fontsize=5.8, color=C["red"], ha="center")
    flow_arrow(fig, (0.895, y0 + h + 0.004), (0.895, yf), color=C["red"], style="-")
    flow_arrow(fig, (0.895, yf), (0.275, yf), color=C["red"], style="-")
    flow_arrow(fig, (0.275, yf), (0.275, y0 + h + 0.004), color=C["red"])
    fig.text(0.01, 0.03, "确定性出口：B 落在单个网格 → one-hot，TV = 0；R 轮仍未通过 → 精确第二阶段，tau-pencil 确定性 TV 界。"
             "统计失败预算 δ = 0.01 / N_planned。", fontsize=5.8, color=C["teal"])
    save(fig, "08_fastlabel", out)


# ---------------------------------------------------------------- figure 07: a real posterior field
def fig07(out):
    m = np.load(DATA / "main_task.npz")
    vc = (1 / 128) ** 3
    q = 0.5 * (m["ex_lo"] + m["ex_hi"]); Q = q / q.sum(); f = q / vc
    ix, iy, iz = cells_xyz(m["ex_cell"])
    t = np.minimum((m["true_source"] * 128).astype(int), 127)
    norm = mcolors.LogNorm(vmin=Q.max() * 1e-2, vmax=Q.max())
    fig = plt.figure(figsize=(W_IN, 2.5))
    # (a) face-on view of the feasible sheet, coloured by the mass of the first cell hit
    ctr = np.array([(ix.min() + ix.max() + 1) / 2, (iy.min() + iy.max() + 1) / 2, (iz.min() + iz.max() + 1) / 2]) / 128
    cam = R3.Camera(elev=18.4, azim=90, size=(720, 620), center=ctr, half_width=0.33)
    img, dep = R3.render_voxels(cam, dict(zip(m["ex_cell"].tolist(), Q.tolist())), SEQ, norm, mip=True, amb=1.0)
    ax = img_axes(fig, [0.0, 0.10, 0.36, 0.72], cam); ax.imshow(img, interpolation="lanczos")
    ut = draw_points(ax, cam, m["true_source"][None], np.full_like(dep, np.inf), s=44, marker="*", c=C["lit"],
                     edgecolors=C["ink"], linewidths=0.4, zorder=5)
    fig.text(0.035, 0.05, "颜色为视线上最大的格质量；★ 真实源", fontsize=5.9, color=C["ink2"])
    fig.text(0.035, 0.905, "可行集 F：一张极薄的楔形片（正视）", fontsize=7.2, va="bottom")
    fig.text(0.035, 0.845, f"跨 {len(Q):,} 个网格，总体积仅 {q.sum() / vc:.3f} 个网格体积", fontsize=6.2, color=C["ink2"], va="bottom")
    # (b) zoomed cross-section x = t_x
    axs = fig.add_axes([0.42, 0.24, 0.20, 0.56])
    Z = np.zeros((128, 128)); sel = ix == t[0]; Z[iy[sel], iz[sel]] = Q[sel]
    y0, y1, z0, z1 = t[1] - 9, t[1] + 9, t[2] - 14, t[2] + 14
    axs.imshow(np.ma.masked_equal(Z[y0:y1 + 1, z0:z1 + 1].T, 0), cmap=SEQ, norm=norm, origin="lower", interpolation="nearest",
               extent=(y0 - 0.5, y1 + 0.5, z0 - 0.5, z1 + 0.5))
    for g_ in range(y0, y1 + 2):
        axs.axvline(g_ - 0.5, color=C["grid"], lw=0.25, zorder=0)
    for g_ in range(z0, z1 + 2):
        axs.axhline(g_ - 0.5, color=C["grid"], lw=0.25, zorder=0)
    axs.scatter([t[1]], [t[2]], s=40, marker="*", c=C["lit"], edgecolors=C["ink"], linewidths=0.4, zorder=4)
    axs.set_xlabel("y（网格）", fontsize=6.3, labelpad=1); axs.set_ylabel("z（网格）", fontsize=6.3, labelpad=1)
    axs.tick_params(labelsize=5.6, pad=1.5); axs.set_aspect("equal")
    axs.set_title(f"x = {t[0]} 截面（放大）", fontsize=7, loc="left", pad=3)
    cax = fig.add_axes([0.455, 0.055, 0.165, 0.02])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.4, width=0.4, length=1.8, pad=1)
    cax.text(-0.04, 0.5, "$q_F(i)$", transform=cax.transAxes, fontsize=6.2, ha="right", va="center")
    # (c) fill fraction
    axh = fig.add_axes([0.71, 0.20, 0.275, 0.60])
    ff = f[f > 0]
    bins = np.logspace(np.floor(np.log10(ff.min())), np.ceil(np.log10(ff.max())), 31)
    axh.hist(ff, bins=bins, color=C["teal"], edgecolor="white", linewidth=0.5)
    axh.set_xscale("log")
    axh.set_xlabel(r"$\mathrm{Vol}(F\cap C_i)\,/\,\mathrm{Vol}(C_i)$", fontsize=6.3, labelpad=1)
    axh.set_ylabel("网格数", fontsize=6.3, labelpad=1); axh.tick_params(labelsize=5.6, pad=1.5)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.set_title("各格质量不相等", fontsize=7, loc="left", pad=3)
    p10, p50, p90 = np.percentile(ff, [10, 50, 90])
    axh.text(0.03, 0.97, f"中位数 {sci(p50)}\n10% 分位 {sci(p10)}\n90% 分位 {sci(p90)}\n$q_F(i)\\propto\\mathrm{{Vol}}(F\\cap C_i)$",
             transform=axh.transAxes, fontsize=5.9, color=C["ink2"], va="top", linespacing=1.35)
    for x, l in ((0.0, "a"), (0.375, "b"), (0.665, "c")):
        fig.text(x + 0.003, 0.905, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    save(fig, "07_posterior_field", out)


# ---------------------------------------------------------------- figure 05: full fields vs local-only generation
def fig05(out):
    z = np.load(DATA / "local100k.npz"); LG = json.loads((DATA / "LOG.json").read_text())["local100k"]
    lit = np.unpackbits(z["plane_lit"], count=1024 * 1024).reshape(1024, 1024).astype(bool)
    sol = np.unpackbits(z["plane_sol"], count=1024 * 1024).reshape(1024, 1024).astype(bool)
    zc = int(z["plane_z"]); W = z["windows"]; K = int(LG["k"])
    H_ = 3.0
    fig = plt.figure(figsize=(W_IN, H_))
    sq = 0.255
    # (a) full-world plane of one source
    ax = fig.add_axes([0.065, 0.27, sq, sq * W_IN / H_])
    cls = np.where(sol, 2, np.where(lit, 1, 0))
    ax.imshow(cls, cmap=CLS, vmin=0, vmax=2, origin="lower", interpolation="nearest", extent=(0, 1024, 0, 1024), rasterized=True)
    for r in W:
        if r[3] <= zc < r[3] + r[4]:
            ax.add_patch(Rectangle((r[1], r[2]), r[4], r[4], fill=False, ec=C["teal"], lw=1.0))
    ax.set_xticks([0, 512, 1024]); ax.set_yticks([0, 512, 1024]); ax.tick_params(labelsize=5.6, pad=1.5)
    ax.set_xlabel("x（体素）", fontsize=6.2, labelpad=1); ax.set_ylabel("y（体素）", fontsize=6.2, labelpad=1)
    fig.text(0.035, 0.915, "旧方式：完整响应场", fontsize=7.3, va="bottom")
    fig.text(0.035, 0.86, f"源 {LG['plane']['source'] + 1} 在 z = {zc} 的整面；每源 1024³ 个接收点", fontsize=6.0,
             color=C["ink2"], va="bottom")
    fig.text(0.035, 0.115, "绿框：与该平面相交的接受窗口；\n其余接收点从未被 Student 使用", fontsize=5.8, color=C["ink2"],
             va="top", linespacing=1.35)
    # (b) proposals of the local-only selection
    axb = fig.add_axes([0.41, 0.27, sq, sq * W_IN / H_])
    for b in range(len(z["voff"]) - 1):
        Vb = z["V"][z["voff"][b]:z["voff"][b + 1]] * 1024
        h = hull2d(Vb[:, :2]); axb.fill(h[:, 0], h[:, 1], color=C["solid"], alpha=0.35, lw=0)
    sizes = {64: 8, 128: 14, 256: 22}
    acc_n = 0; rej_n = 0
    for key in ("log_L64", "log_L128", "log_L256"):
        Lp = int(key.split("L")[-1]); Lg = z[key]
        if not len(Lg):
            continue
        c = Lg[:, :3] + Lp / 2; ok = Lg[:, 8] > 0
        axb.scatter(c[~ok, 0], c[~ok, 1], s=sizes[Lp], marker="o", facecolors="none", edgecolors=C["muted"], linewidths=0.5)
        axb.scatter(c[ok, 0], c[ok, 1], s=sizes[Lp] * 1.3, marker="s", c=C["teal"], linewidths=0)
        acc_n += ok.sum(); rej_n += (~ok).sum()
    src = z["sources"] * 1024
    axb.scatter(src[:, 0], src[:, 1], s=26, marker="*", c=C["lit"], edgecolors=C["ink"], linewidths=0.4, zorder=5)
    axb.set_xlim(0, 1024); axb.set_ylim(0, 1024); axb.set_aspect("equal")
    axb.set_xticks([0, 512, 1024]); axb.set_yticks([0, 512, 1024]); axb.tick_params(labelsize=5.6, pad=1.5)
    axb.set_xlabel("x（体素）", fontsize=6.2, labelpad=1)
    fig.text(0.385, 0.915, "100K：局部按需计算", fontsize=7.3, va="bottom")
    fig.text(0.385, 0.86, f"同一场景 {acc_n + rej_n} 个配额候选（俯视）", fontsize=6.0, color=C["ink2"], va="bottom")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([], [], marker="s", ls="none", ms=4, color=C["teal"], label=f"接受（{acc_n}）：渲染并保存完整 crop"),
          Line2D([], [], marker="o", ls="none", ms=4, mfc="none", mec=C["muted"], label=f"拒绝（{rej_n}）：探针、外包证书或早停 tile 判定"),
          Line2D([], [], marker="*", ls="none", ms=6, mfc=C["lit"], mec=C["ink"], mew=0.4, label="源"),
          Patch(fc=C["solid"], alpha=0.35, label="实体（xy 投影）")]
    fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.375, 0.135), ncol=1, fontsize=5.5, handletextpad=0.3,
               labelspacing=0.25, borderaxespad=0)
    # (c) work per scene
    axc = fig.add_axes([0.82, 0.47, 0.165, 0.36])
    wk = LG["work"]; tiles = int(wk["tiles"]); probes = int(wk["probes"])
    crop_vox = int(sum(int(r[4]) ** 3 for r in W)) * (K + 1)
    rows = [("完整场（K 个源）", K * 1024 ** 3, C["solid"]), ("认证 64³ tile", tiles * 64 ** 3, C["teal"]),
            ("保存的 crop", crop_vox, C["teal"]), ("精确探针", probes, C["teal"])]
    yv = np.arange(len(rows))[::-1]
    axc.barh(yv, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    axc.set_xscale("log"); axc.set_xlim(1e5, 1e10)
    axc.set_yticks(yv); axc.set_yticklabels([r[0] for r in rows], fontsize=5.8); axc.tick_params(axis="x", labelsize=5.4, pad=1)
    for y_, r in zip(yv, rows):
        inside = r[1] > 3e8
        axc.text(r[1] / 1.25 if inside else r[1] * 1.25, y_, sci(r[1]), va="center", ha="right" if inside else "left",
                 fontsize=5.4, color="white" if inside else C["ink2"])
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_xlabel("每场景接收点评估数", fontsize=6.0, labelpad=1)
    fig.text(0.715, 0.915, "计算量（同一场景）", fontsize=7.3, va="bottom")
    fig.text(0.715, 0.86, f"K = {K}；tile 含 rate 流候选", fontsize=6.0, color=C["ink2"], va="bottom")
    fig.text(0.715, 0.335, "物理阶段（笔记本）：\n1K 完整场 24.6 s / 场景\n100K 局部 3.7 s / 场景", fontsize=5.9, va="top",
             linespacing=1.35)
    fig.text(0.715, 0.135, "全局几何仍完整保存：\n窗口外实体决定可见性，\n也参与反向求解", fontsize=5.9, color=C["teal"], va="top",
             linespacing=1.35)
    for x, l in ((0.005, "a"), (0.36, "b"), (0.69, "c")):
        fig.text(x, 0.915, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    save(fig, "05_local_rendering", out)


# ---------------------------------------------------------------- figure 02: dataset specification (100K scene, to scale)
def bodies_from_vertices(V, voff):
    from scipy.spatial import ConvexHull
    out = []
    for b in range(len(voff) - 1):
        Vb = V[voff[b]:voff[b + 1]]; h = ConvexHull(Vb)
        out.append(R3.halfspaces_from_facets(facets(Vb, h.simplices)))
    return out


def fig02(out):
    z = np.load(DATA / "local100k.npz"); W = z["windows"]; src = z["sources"]
    fig = plt.figure(figsize=(W_IN, 2.55))
    cam = R3.Camera(elev=20, azim=-60, size=(760, 760), half_width=0.84)
    img, depth = R3.render_bodies(cam, bodies_from_vertices(z["V"], z["voff"]), "#A3ADB8")
    ax = img_axes(fig, [0.0, 0.0, 0.43, 0.86], cam); ax.imshow(img, interpolation="lanczos")
    draw_segments(ax, cam, R3.cube_segments((0, 0, 0), 1), depth, C["muted"], 0.5, hidden=0.5)
    col = {64: "#7FC4B8", 128: "#2E8C8A", 256: "#1B4F6B"}
    for r in W:
        draw_segments(ax, cam, R3.cube_segments(r[1:4] / 1024, r[4] / 1024), depth, col[int(r[4])], 0.8, hidden=0.35, n=120)
    draw_points(ax, cam, src, depth, s=30, marker="*", c=C["lit"], edgecolors=C["ink"], linewidths=0.4, zorder=8)
    fig.text(0.035, 0.915, "一个 100K 场景（合同 v1，slot 0）", fontsize=7.3, va="bottom")
    fig.text(0.035, 0.86, f"1024³ 体素世界；{len(z['voff']) - 1} 个凸多面体；K = {len(src)} 个源；12 个窗口按比例", fontsize=6.0,
             color=C["ink2"], va="bottom")
    # (b) the three window sizes against the world and the label grid
    axb = fig.add_axes([0.50, 0.10, 0.47, 0.70]); axb.set_xlim(-10, 1034); axb.set_ylim(-120, 330); axb.set_axis_off()
    axb.set_aspect("equal", adjustable="box", anchor="C")
    x = 0
    for L in (64, 128, 256):
        axb.add_patch(Rectangle((x, 40), L, L, fc=col[L], ec="none", alpha=0.9))
        axb.text(x + L / 2, 40 + L + 10, f"{L}³", ha="center", fontsize=6.6)
        axb.text(x + L / 2, 22, "4 个", ha="center", va="center", fontsize=5.8, color=C["ink2"])
        x += L + 60
    axb.text(x + 10, 40 + 128, "每场景 12 个物理窗口\n每窗口：1 个 SOLID crop\n+ K 个源的分源响应 crop", fontsize=6.1,
             va="center", linespacing=1.4)
    axb.add_patch(Rectangle((0, -12), 1024, 14, fc=C["panel"], ec=C["muted"], lw=0.5))
    axb.text(512, -30, "世界边长 1024 体素（Ω = (0, 1)³）", ha="center", va="top", fontsize=6.1, color=C["ink2"])
    for i in range(0, 1025, 8):
        axb.plot([i, i], [-82, -72], color=C["teal"] if i % 64 == 0 else C["muted"], lw=0.5 if i % 64 == 0 else 0.25)
    axb.plot([0, 1024], [-77, -77], color=C["muted"], lw=0.4)
    axb.text(512, -92, "标签网格 128³：每格 8³ 体素（刻度每 8 体素）", ha="center", va="top", fontsize=6.1, color=C["ink2"])
    fig.text(0.50, 0.915, "窗口与网格尺度", fontsize=7.3, va="bottom")
    fig.text(0.50, 0.86, "原点均匀取自 {0, …, 1024 − L}³，源不在窗口闭包内", fontsize=6.0, color=C["ink2"], va="bottom")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], color=col[L], lw=1.2, label=f"L = {L}") for L in (64, 128, 256)] + \
         [Line2D([], [], marker="*", ls="none", ms=6, mfc=C["lit"], mec=C["ink"], mew=0.4, label="源")]
    fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=4, fontsize=5.8, handlelength=1.4, columnspacing=1.0)
    for x_, l in ((0.005, "a"), (0.47, "b")):
        fig.text(x_, 0.915, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    save(fig, "02_dataset_spec", out)


# ---------------------------------------------------------------- figure 03: sampling and coverage policy
def fig03(out):
    fig = plt.figure(figsize=(W_IN, 2.45))
    # (a) quota layout
    ax = fig.add_axes([0.0, 0.0, 0.56, 0.90]); ax.set_xlim(-0.2, 10); ax.set_ylim(-0.7, 6.2); ax.set_axis_off()
    rows = [(64, 4.6), (128, 2.85), (256, 1.1)]
    for L, y in rows:
        ax.text(0.6, y + 0.55, f"L = {L}", fontsize=6.8, ha="center", va="center")
        for k in range(4):
            cov = (L == 64 and k == 3)
            x = 1.25 + k * 2.15
            ax.add_patch(FancyBboxPatch((x, y), 1.95, 1.1, boxstyle="round,pad=0.0,rounding_size=0.12",
                                        fc="#F5E6C4" if cov else "#DDEEEA", ec=C["lit"] if cov else "none", lw=0.8 if cov else 0,
                                        ls=(0, (3, 2)) if cov else "-"))
            ax.text(x + 0.975, y + 0.70, "Coverage" if cov else "Natural", fontsize=6.6, ha="center", va="center")
            ax.text(x + 0.975, y + 0.33, "仅 K = 4 场景" if cov else "自然窗口", fontsize=5.8, ha="center", va="center",
                    color=C["ink2"])
    fig.text(0.035, 0.915, "每个窗口尺寸 4 个配额（合同 v2，暂定：3 + 1）", fontsize=7.3, va="bottom")
    ax.text(0.0, 0.75, "Natural：合法候选按自然律接受。Coverage：按 m 以 c(m) = (1/8, 1/4, 1/2, 1) 接受；\n"
                        "记录来源、接受概率、m 与通道掩码，训练时用平衡启发式权重恢复 T*。", fontsize=5.9, color=C["ink2"],
            va="top", linespacing=1.35)
    # (b) natural share of m = 4 windows in K = 4 scenes
    axb = fig.add_axes([0.66, 0.30, 0.31, 0.46])
    labels = ["L = 64\n全部", "L = 64\nmultiscale", "L = 128", "L = 256"]; vals = [4.7, 1.8, 14, 47]
    cs = [C["red"], C["red"], C["teal"], C["teal"]]
    axb.bar(range(4), vals, color=cs, width=0.62)
    for i, v in enumerate(vals):
        axb.text(i, v + 1.5, f"{v:g}%", ha="center", fontsize=6)
    axb.set_xticks(range(4)); axb.set_xticklabels(labels, fontsize=5.7); axb.tick_params(axis="y", labelsize=5.6, pad=1)
    axb.set_ylim(0, 55); axb.set_ylabel("m = 4 占比（%）", fontsize=6.2, labelpad=1)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    fig.text(0.62, 0.915, "自然采样下 m = 4 的稀缺性", fontsize=7.3, va="bottom")
    fig.text(0.62, 0.86, "K = 4 场景中自然窗口的 m = 4 比例（1K 世界模拟）", fontsize=6.0, color=C["ink2"], va="bottom")
    fig.text(0.62, 0.06, "稀缺集中在 64³。若 K = 4 场景 ≥ 50%，纯自然采样有望\n满足覆盖目标；L64 Coverage 配额作为保险。", fontsize=5.9,
             color=C["ink2"], linespacing=1.35)
    for x, l in ((0.005, "a"), (0.595, "b")):
        fig.text(x, 0.915, l, fontsize=8, fontweight="bold", family="DejaVu Sans", va="bottom")
    save(fig, "03_sampling_coverage", out)


# ---------------------------------------------------------------- figure 06: storage layout
def fig06(out):
    fig = plt.figure(figsize=(W_IN, 2.55))
    def box(x, y, w, h, title, lines, fc, title_fs=7.0):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012", transform=fig.transFigure,
                                      fc=fc, ec="none"))
        fig.text(x + 0.012, y + h - 0.035, title, fontsize=title_fs, va="top")
        for i, ln in enumerate(lines):
            fig.text(x + 0.016, y + h - 0.11 - 0.066 * i, ln, fontsize=5.8, va="top", color=C["ink2"])
    box(0.01, 0.10, 0.30, 0.80, "场景容器（每场景一个 .npz）",
        ["meta：分辨率、slot、K、几何 id、采样日志", "geometry：规范多面体（精确有理数）", "P, K：标注器使用的半空间行",
         "sources_hex：K 个源（binary64）", "windows：12 个窗口（原点、L、m、掩码）", "obs：bit-pack + zstd 的局部 crop",
         "obs_index：(窗口, 源 或 SOLID, 偏移, 长度)", "log_L64/128/256：候选与接受记录"], C["panel"])
    # window detail
    x0, y0 = 0.36, 0.16
    fig.text(x0, 0.86, "一个窗口 w（源维度不打平）", fontsize=7.0, va="top")
    fig.add_artist(Rectangle((x0, y0 + 0.50), 0.055, 0.10, transform=fig.transFigure, fc=C["solid"], ec="none"))
    fig.text(x0 + 0.065, y0 + 0.55, "SOLID crop（每窗口一次）", fontsize=6.0, va="center")
    kinds = [("mixed", "（mixed，有标签）"), ("mixed", "（mixed，有标签）"), ("lit", "（全亮：保存，不标注）"), ("dark", "（全暗：保存，不标注）")]
    for j, (kind, txt) in enumerate(kinds):
        yy = y0 + 0.36 - 0.10 * j; a_ = 1.0
        if kind == "mixed":
            fig.add_artist(Rectangle((x0, yy), 0.055, 0.075, transform=fig.transFigure, fc=C["dark"], ec="none"))
            fig.add_artist(plt.Polygon([(x0, yy + 0.075), (x0 + 0.055, yy + 0.075), (x0, yy + 0.02)], closed=True,
                                       transform=fig.transFigure, fc=C["lit"], ec="none"))
        else:
            fig.add_artist(Rectangle((x0, yy), 0.055, 0.075, transform=fig.transFigure, fc=C[kind], ec="none", alpha=a_))
        fig.text(x0 + 0.065, yy + 0.037, f"源 {j + 1} 响应 crop{txt}", fontsize=5.9, va="center",
                 color=C["ink"] if kind == "mixed" else C["ink2"])
    # labels
    box(0.70, 0.42, 0.29, 0.48, "标签（每个 mixed 通道一个）",
        ["R3LJ 追加式日志", "记录级 SHA-256，截断尾可检测", "合并为每场景 labels.npz：", "网格 id、质量、状态、证书"], "#E3F1EE")
    box(0.70, 0.10, 0.29, 0.27, "训练 loader",
        ["按总体（T*、T*_mis）展开单通道样本", "SOLID + 一个源 + L + 原点"], "#F5E6C4")
    for yy in (y0 + 0.36 + 0.037, y0 + 0.26 + 0.037):
        flow_arrow(fig, (0.655, yy), (0.698, 0.64))
    flow_arrow(fig, (0.845, 0.42), (0.845, 0.372))
    fig.text(0.36, 0.06, "RINDPack（64³ brick 随机访问）只用于 1K 的完整响应场；100K 不生成完整场。", fontsize=5.9, color=C["ink2"])
    save(fig, "06_storage", out)


# ---------------------------------------------------------------- figure 09: Student training
def fig09(out):
    w, s, lit, sol = crop("main"); cls = cls_xyz(lit, sol)
    L = cls.shape[0]; f = L // 32
    def pool(a):
        return a.reshape(32, f, 32, f, 32, f).mean(axis=(1, 3, 5))
    feats = [("亮 FREE", pool(cls == 1)), ("暗 FREE", pool(cls == 0)), ("SOLID（空）", pool(cls == 2)),
             ("明暗边界", pool(boundary_voxels(cls)))]
    fig = plt.figure(figsize=(W_IN, 2.15))
    fig.text(0.01, 0.915, "Student 输入（真实窗口，汇聚到 32³；中间 z 切片）", fontsize=7.0, va="bottom")
    for i, (name, F_) in enumerate(feats):
        ax = fig.add_axes([0.01 + 0.093 * i, 0.43, 0.083, 0.083 * W_IN / 2.15])
        ax.imshow(F_[:, :, 16].T, cmap=mcolors.LinearSegmentedColormap.from_list("g", ["#FFFFFF", C["ink"]]), vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color(C["muted"])
        ax.set_title(name, fontsize=6.0, pad=2)
    fig.text(0.01, 0.30, "+ 窗口大小 L 与全局原点（位置编码）", fontsize=6.0, color=C["ink2"])
    fig.text(0.01, 0.20, "不输入：其他源、窗口外几何、真实源坐标", fontsize=6.0, color=C["red"])
    flow_box(fig, 0.42, 0.47, 0.20, 0.33, "参考 Student（约 200 万参数）\n3D 卷积 + 全局位置编码\n+ cross-attention\n分层归一化输出",
             C["panel"], fs=6.0)
    flow_arrow(fig, (0.385, 0.635), (0.418, 0.635))
    flow_box(fig, 0.67, 0.47, 0.13, 0.33, "$p_\\theta(i\\mid X)$\n128³ 网格\n概率质量", "#E3F1EE", fs=6.2)
    flow_arrow(fig, (0.622, 0.635), (0.668, 0.635))
    flow_box(fig, 0.845, 0.47, 0.145, 0.33, "监督 $q_G$\nFastLabel /\ntau-pencil\n认证后验", "#F5E6C4", fs=6.0)
    flow_arrow(fig, (0.843, 0.635), (0.802, 0.635), style="<|-|>")
    fig.text(0.42, 0.29, r"损失：$\mathcal{L}(\theta)=-\mathbb{E}_{T^*}\,\sum_i q_G(i)\log p_\theta(i\mid X)$（交叉熵，等价于 forward KL）",
             fontsize=6.3)
    fig.text(0.42, 0.175, "Natural / Coverage 混合时按记录的接受概率与 rate 统计加权，使目标仍为 T*", fontsize=5.9, color=C["ink2"])
    fig.text(0.42, 0.07, "完整几何只在标注阶段使用；主流程不需要额外的神经网络 Teacher", fontsize=5.9, color=C["teal"])
    save(fig, "09_student_training", out)


FIGS = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05, "06": fig06, "07": fig07, "08": fig08, "09": fig09}


def contact_sheet(out, width_px=620):
    """All figures at document width (14.8 cm ~ 620 px at 100 % zoom), two per row, with file names."""
    from PIL import Image, ImageDraw, ImageFont
    files = sorted(out.glob("[0-9][0-9]_*.png"))
    ims = []
    for f in files:
        im = Image.open(f).convert("RGB"); h = int(im.height * width_px / im.width)
        ims.append((f.stem, im.resize((width_px, h), Image.LANCZOS)))
    pad, lab = 24, 26; cols = 2
    rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
    H = sum(max(im.height for _, im in r) + lab + pad for r in rows) + pad
    sheet = Image.new("RGB", (cols * width_px + (cols + 1) * pad, H), "white"); d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    y = pad
    for r in rows:
        for c, (name, im) in enumerate(r):
            x = pad + c * (width_px + pad)
            d.text((x, y), name, fill=(40, 50, 60), font=font)
            sheet.paste(im, (x, y + lab)); d.rectangle([x - 1, y + lab - 1, x + im.width, y + lab + im.height], outline=(220, 224, 228))
        y += max(im.height for _, im in r) + lab + pad
    sheet.save(out / "contact_sheet.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args(); out = Path(a.out)
    for k, f in FIGS.items():
        if a.only and k not in a.only.split(","):
            continue
        f(out); print("figure", k)
    contact_sheet(out)


if __name__ == "__main__":
    main()
