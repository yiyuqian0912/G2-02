"""Teacher-only sampled inverse-geometry diagnostics and optional static figures."""

from collections import deque
from pathlib import Path

import numpy as np

from rind_phase1.physics import check_reference_solutions
from rind_phase1.search import CandidateDomain, UniformGrid


def grid_from_record(record):
    support = record["metadata"]["settings"]["support"]
    domain = CandidateDomain(tuple(record["window"]), support["world_size"], support["quadtree_prior"])
    return UniformGrid(domain, support["candidate_spacing"], record["candidate_xy"],
                       record["grid_index"], record["x_axis"], record["y_axis"])


def zero_components(mask, grid):
    """Four-neighbor components of sampled cells; not continuous-set topology."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != grid.shape:
        raise ValueError("Component mask must match the candidate raster")
    seen = np.zeros_like(mask)
    components = []
    height, width = mask.shape
    for row, column in zip(*np.nonzero(mask)):
        if seen[row, column]:
            continue
        queue = deque([(int(row), int(column))])
        seen[row, column] = True
        count, r0, r1, c0, c1 = 0, row, row, column, column
        while queue:
            r, c = queue.popleft()
            count += 1
            r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        s = grid.spacing
        components.append({"sample_count": count, "approximate_area": count * s * s,
                           "bounding_box": [float(grid.x_axis[c0] - s / 2),
                                            float(grid.y_axis[r0] - s / 2),
                                            float(grid.x_axis[c1] + s / 2),
                                            float(grid.y_axis[r1] + s / 2)],
                           "bounding_box_fill_fraction": count / ((r1 - r0 + 1) * (c1 - c0 + 1))})
    return sorted(components, key=lambda component: component["sample_count"], reverse=True)


def inspect_teacher(dataset, record, *, low_cost_threshold=None, backend=None):
    """Check ten witnesses independently and quantify this declared uniform support."""
    if not record["metadata"]["complete"] or not record["evaluated"].all():
        raise ValueError("Diagnostics require a completed teacher record")
    if low_cost_threshold is not None and (not np.isfinite(low_cost_threshold) or low_cost_threshold < 0):
        raise ValueError("low_cost_threshold must be finite and nonnegative, or None")
    grid = grid_from_record(record)
    valid, costs, probabilities = record["valid"], record["physical_cost"], record["teacher_prob"]
    zero = valid & (costs == 0)
    components = zero_components(grid.as_map(zero, fill=0).astype(bool), grid)
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    valid_count, total = int(valid.sum()), len(valid)
    backend = backend or record["metadata"]["settings"]["physics"]["requested_backend"]
    witnesses = check_reference_solutions(dataset, record["scene_id"], record["view_id"], backend=backend)
    references = witnesses["reference_xy"]
    true_source = dataset.get_scene(record["scene_id"])["drivers"][0]
    nearest = []
    for xy in references:
        column = int(np.argmin(np.abs(grid.x_axis - xy[0])))
        row = int(np.argmin(np.abs(grid.y_axis - xy[1])))
        flat = row * len(grid.x_axis) + column
        index = int(np.searchsorted(grid.grid_index, flat))
        present = index < total and grid.grid_index[index] == flat
        is_valid = bool(present and valid[index])
        center = [float(grid.x_axis[column]), float(grid.y_axis[row])]
        nearest.append({"grid_xy": center, "distance": float(np.linalg.norm(xy - center)),
                        "valid": is_valid, "cost": float(costs[index]) if is_valid else None,
                        "zero_cost": bool(is_valid and costs[index] == 0)})
    low_count = int(np.count_nonzero(valid & (costs <= low_cost_threshold))) if low_cost_threshold is not None else None
    return {"scene_id": record["scene_id"], "view_id": record["view_id"],
            "window": record["window"].tolist(), "candidate_spacing": grid.spacing,
            "candidate_domain": grid.domain.name, "config_id": record["config_id"], "run_id": record["run_id"],
            "domain_candidates": total, "valid_candidates": valid_count,
            "zero_cost_candidates": int(zero.sum()), "minimum_valid_cost": float(costs[valid].min()),
            "zero_fraction_of_domain_grid": int(zero.sum()) / total,
            "zero_fraction_of_valid_grid": int(zero.sum()) / valid_count,
            "low_cost_threshold": low_cost_threshold, "low_cost_candidates": low_count,
            "low_cost_fraction_of_domain_grid": low_count / total if low_count is not None else None,
            "zero_component_count": len(components), "zero_components": components,
            "component_connectivity": "4-neighbor sampled lattice",
            "teacher_entropy_nats": entropy,
            "normalized_teacher_entropy": entropy / np.log(valid_count) if valid_count > 1 else 0.0,
            "effective_teacher_candidates": float(np.exp(entropy)),
            "num_physics_evaluations": record["metadata"]["num_physics_evaluations"],
            "diagnostic_physics_evaluations": witnesses["diagnostic_physics_evaluations"],
            "elapsed_search_seconds": record["metadata"]["elapsed_seconds"],
            "true_source_xy": true_source.tolist(), "reference_xy": references.tolist(),
            "reference_cost": witnesses["reference_cost"].tolist(),
            "reference_domain_coverage_fraction": float(np.mean(grid.domain.contains(references))),
            "reference_nearest_grid": nearest,
            "reference_nearest_grid_zero_fraction": sum(item["zero_cost"] for item in nearest) / len(nearest),
            "interpretation": "Fractions, component areas, connectivity, and nearest-grid witness coverage describe the sampled support. They do not establish continuous inverse-set size or exhaustive search recall. Witnesses are evaluated separately and never enter target normalization."}


def representative_view_ids(dataset, scene_id):
    """First stored view of each size, ordered small to large; no truth-based selection."""
    views = dataset.get_scene(scene_id)["local_views"]
    return [int(np.flatnonzero(views[:, 2] == size)[0]) for size in np.unique(views[:, 2])]


def plot_teacher(dataset, record, diagnostics, path):
    """Response, local mask, source-space costs/zeros/probabilities, and witness zoom."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("Install plotting dependencies: uv sync --locked --extra diagnostics") from exc
    grid = grid_from_record(record)
    observation = dataset.get_observation(record["scene_id"], record["view_id"])
    x, y, side = record["window"]
    local_extent = [x, x + side, y + side, y]
    extent = [grid.x_axis[0] - grid.spacing / 2, grid.x_axis[-1] + grid.spacing / 2,
              grid.y_axis[-1] + grid.spacing / 2, grid.y_axis[0] - grid.spacing / 2]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), layout="constrained")
    axes[0, 0].imshow(observation["response"], cmap="gray", vmin=0, vmax=1, extent=local_extent,
                       interpolation="nearest")
    axes[0, 0].set_title(f"Local response ({side} x {side})")
    axes[0, 1].imshow(observation["obstacle"], cmap="gray", vmin=0, vmax=1, extent=local_extent,
                       interpolation="nearest")
    axes[0, 1].set_title("Local obstacle mask (white = obstacle)")
    costs = np.where(record["valid"], record["physical_cost"], np.nan)
    zeros = np.where(record["valid"], record["physical_cost"] == 0, np.nan)
    probabilities = np.where(record["valid"], record["teacher_prob"], np.nan)
    references = np.asarray(diagnostics["reference_xy"])
    true = np.asarray(diagnostics["true_source_xy"])
    for ax, values, title, cmap, limits in (
            (axes[0, 2], costs, "Physical cost C(s): mismatch fraction", "viridis", (0, 1)),
            (axes[1, 0], zeros, "Exact matches on this grid (yellow = zero)", "viridis", (0, 1)),
            (axes[1, 1], probabilities, "Teacher probability (discrete mass)", "magma", (0, None))):
        colors = plt.get_cmap(cmap).copy()
        colors.set_bad("#cccccc")
        image = ax.imshow(grid.as_map(values), extent=extent, interpolation="nearest",
                          cmap=colors, vmin=limits[0], vmax=limits[1])
        ax.add_patch(Rectangle((x, y), side, side, fill=False, edgecolor="red", linewidth=1.2))
        ax.scatter(references[:, 0], references[:, 1], marker="x", color="cyan", s=28,
                   label="10 exact witnesses")
        ax.scatter(*true, marker="*", color="white", edgecolors="black", s=90, label="True source")
        ax.set_title(title)
        ax.set_xlabel("Source x")
        ax.set_ylabel("Source y")
        fig.colorbar(image, ax=ax, shrink=0.8)
    axes[0, 2].legend(loc="upper right", fontsize=7)
    zoom = axes[1, 2]
    # Some witnesses differ by tiny continuous offsets. Relative axes avoid
    # unreadable overlapping world-coordinate tick labels in the zoom panel.
    offsets = references - true
    zoom.scatter(offsets[1:, 0], offsets[1:, 1], marker="x", label="9 additional witnesses")
    zoom.scatter(0, 0, marker="*", color="red", s=90, label="True / reference 0")
    span = max(float(np.ptp(offsets, axis=0).max()), 1e-6)
    center = (offsets.min(axis=0) + offsets.max(axis=0)) / 2
    zoom.set_xlim(center[0] - span * 0.7, center[0] + span * 0.7)
    zoom.set_ylim(center[1] + span * 0.7, center[1] - span * 0.7)
    zoom.set_title("Offsets from truth (x = 9 refs; star = truth)", fontsize=10)
    zoom.set_xlabel("Delta x (world units)")
    zoom.set_ylabel("Delta y (world units)")
    zoom.ticklabel_format(useOffset=False, style="sci", scilimits=(-3, 3), useMathText=True)
    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"Scene {record['scene_id']}, view {record['view_id']} | {grid.domain.name} | "
                 f"spacing={grid.spacing:g}, tau={record['temperature']:g}\n"
                 f"zero={diagnostics['zero_cost_candidates']}/{diagnostics['valid_candidates']} valid points, "
                 f"{diagnostics['zero_component_count']} sampled components, "
                 f"entropy={diagnostics['teacher_entropy_nats']:.3f} nats", fontsize=12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
