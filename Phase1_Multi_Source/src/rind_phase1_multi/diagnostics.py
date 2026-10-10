"""Joint-support diagnostics and clearly labelled two-dimensional projections."""

from pathlib import Path

import numpy as np

from rind_phase1_multi.physics import JointResponseEvaluator, REFERENCE_ATOL


def source_space_projections(record):
    """Project complete-set probabilities; these arrays are NOT joint targets.

    presence[y,x] = probability at least one source occupies this grid point.
    expected_count sums to E[K]; expected_intensity sums to E[sum strengths].
    minimum_joint_cost is a sampled conditional minimum, not C(one source).
    Unvisited legal grid points have zero empirical mass and unknown cost.
    """
    xy = np.asarray(record["position_xy"])
    lookup = {tuple(point): i for i, point in enumerate(xy)}
    n = len(xy)
    presence, expected_count, expected_intensity = (np.zeros(n) for _ in range(3))
    minimum = np.full(n, np.nan)
    coverage = np.zeros(n, dtype=np.int64)
    for i in np.flatnonzero(record["valid"]):
        count = int(record["source_counts"][i])
        sources = record["sources"][i, :count]
        try:
            ids = np.array([lookup[tuple(row[:2])] for row in sources], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("Projection requires sources on the recorded position grid") from exc
        unique = np.unique(ids)
        probability, cost = record["teacher_prob"][i], record["physical_cost"][i]
        presence[unique] += probability
        np.add.at(expected_count, ids, probability)
        np.add.at(expected_intensity, ids, probability * sources[:, 2])
        minimum[unique] = np.fmin(minimum[unique], cost)
        coverage[unique] += 1
    shape = (len(record["y_axis"]), len(record["x_axis"]))
    output = {}
    for name, values in (("presence", presence), ("expected_count", expected_count),
                         ("expected_intensity", expected_intensity), ("minimum_joint_cost", minimum)):
        image = np.full(shape, np.nan)
        indices = record["position_grid_index"][record["position_valid"]]
        image[indices[:, 0], indices[:, 1]] = values[record["position_valid"]]
        output[name] = image
    coverage_image = np.full(shape, -1, dtype=np.int64)
    indices = record["position_grid_index"][record["position_valid"]]
    coverage_image[indices[:, 0], indices[:, 1]] = coverage[record["position_valid"]]
    output["hypotheses_per_position"] = coverage_image
    return output


def inspect_teacher(dataset, record, *, low_cost_threshold=None, backend="auto"):
    """References are checked separately after target construction; never injected."""
    if low_cost_threshold is not None and (not np.isfinite(low_cost_threshold) or low_cost_threshold < 0):
        raise ValueError("low_cost_threshold must be finite and nonnegative")
    meta = record["metadata"]
    sid, vid = meta["scene_id"], meta["view_id"]
    observed = dataset.get_observation(sid, vid)
    if not np.array_equal(observed["window"], record["window"]):
        raise ValueError("Teacher record window does not match this dataset observation")
    evaluator = JointResponseEvaluator(dataset, sid, vid, backend=backend)
    references = []
    for sources in dataset.get_candidates(sid, vid):
        result = evaluator.evaluate(sources)
        if not result.valid or result.physical_cost > REFERENCE_ATOL:
            raise AssertionError(f"Reference does not reproduce observation: cost={result.physical_cost}")
        references.append({"sources": sources.tolist(), "physical_cost": result.physical_cost})
    q, costs, valid = record["teacher_prob"], record["physical_cost"], record["valid"]
    positive = q > 0
    entropy = float(-np.sum(q[positive] * np.log(q[positive])))
    total_mass = float(record["base_mass"][valid].sum())
    zero = valid & (costs == 0)
    counts = record["source_counts"]
    by_count = []
    for count in meta["settings"]["support"]["source_counts"]:
        mask = valid & (counts == count)
        prob = float(q[mask].sum())
        conditional = q[mask] / prob if prob > 0 else np.zeros(mask.sum())
        nonzero = conditional > 0
        by_count.append({"source_count": count, "hypotheses": int(mask.sum()),
                         "base_prior_mass": float(record["base_mass"][mask].sum() / total_mass),
                         "teacher_probability": prob,
                         "minimum_physical_cost": float(costs[mask].min()) if mask.any() else None,
                         "conditional_entropy_nats": float(-np.sum(conditional[nonzero] * np.log(conditional[nonzero])))})
    order = np.flatnonzero(valid)[np.argsort(-q[valid], kind="stable")]
    top = [{"index": int(i), "sources": record["sources"][i, :int(counts[i])].tolist(),
            "physical_cost": float(costs[i]), "teacher_probability": float(q[i])} for i in order[:10]]
    maps = source_space_projections(record)
    coverage = maps["hypotheses_per_position"]
    valid_positions = coverage >= 0
    visited = coverage > 0
    report = {"scene_id": sid, "view_id": vid, "window": record["window"].tolist(),
              "run_id": meta["run_id"], "config_id": meta["config_id"],
              "search_mode": meta["settings"]["search_mode"],
              "completion_scope": meta["completion_scope"],
              "joint_hypotheses": len(q), "valid_hypotheses": int(valid.sum()),
              "num_joint_evaluations": meta["num_joint_evaluations"],
              "num_source_renders": meta["num_source_renders"],
              "visibility_cache_bytes": meta["visibility_cache_bytes"],
              "physical_cost_quantiles": np.quantile(costs[valid], [0, .1, .5, .9, 1]).tolist(),
              "zero_cost_hypotheses": int(zero.sum()),
              "zero_cost_sampled_prior_fraction": float(record["base_mass"][zero].sum() / total_mass),
              "teacher_entropy_nats": entropy, "effective_hypotheses": float(1 / np.square(q).sum()),
              "expected_source_count": float(np.dot(q, counts)),
              "count_distribution": by_count, "top_hypotheses": top,
              "valid_grid_positions": int(valid_positions.sum()), "visited_grid_positions": int(visited.sum()),
              "position_coverage_fraction": float(visited.sum() / valid_positions.sum()),
              "position_coverage_is_joint_coverage": False,
              "zero_region_topology": "not estimated: sampled joint space cannot be treated as a 2D zero map",
              "references_checked": len(references), "reference_atol": REFERENCE_ATOL,
              "max_reference_cost": max((r["physical_cost"] for r in references), default=0),
              "references": references, "reference_diagnostic_source_renders": evaluator.num_source_renders,
              "reference_solutions_used_as_targets": False,
              "projection_semantics": {"presence": "P(at least one source at this grid point), not normalized to sum one",
                                       "expected_count": "E[number of sources at point]; sum = E[K]",
                                       "expected_intensity": "E[sum source strengths at point]",
                                       "minimum_joint_cost": "minimum full-set cost among sampled sets containing point"}}
    if low_cost_threshold is not None:
        low = valid & (costs <= low_cost_threshold)
        report.update(low_cost_threshold=float(low_cost_threshold), low_cost_hypotheses=int(low.sum()),
                      low_cost_sampled_prior_fraction=float(record["base_mass"][low].sum() / total_mass))
    return report


def plot_teacher(dataset, record, report, path):
    """Six panels; all 2D maps explicitly labelled as finite-support projections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    meta = record["metadata"]
    sid, vid = meta["scene_id"], meta["view_id"]
    x, y, side = map(int, record["window"])
    raw = dataset.get_region(sid, x, y, side)
    best = report["top_hypotheses"][0]
    best_sources = np.array(best["sources"])
    rendered = dataset.rerender(sid, record["window"], best_sources)
    maps = source_space_projections(record)
    truth = dataset.get_source_params(sid)
    world = meta["settings"]["support"]["grid"]["world_size"]
    spacing = meta["settings"]["support"]["grid"]["candidate_spacing"]
    local_extent = (x, x + side, y + side, y)
    grid_extent = (0, len(record["x_axis"]) * spacing, len(record["y_axis"]) * spacing, 0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(top=.86, bottom=.14, hspace=.38, wspace=.30)
    fig.suptitle(f"Multi-source joint teacher | scene {sid}, view {vid}, L={side}, N={len(record['teacher_prob'])}\n"
                 "Whole source sets carry probabilities; 2D maps project the finite sampled support", fontsize=15)
    maximum = max(float(raw["response"].max()), float(rendered.max()), 1e-12)
    for ax, image, title in ((axes[0, 0], raw["response"], "Observed additive response"),
                             (axes[0, 1], rendered, f"Highest-probability source set | MAE={best['physical_cost']:.4g}")):
        shown = ax.imshow(image, origin="upper", extent=local_extent, vmin=0, vmax=maximum, cmap="viridis")
        ax.set_title(title, fontsize=11)
        fig.colorbar(shown, ax=ax, shrink=.80, label="Intensity (unclipped)")
    axes[0, 2].imshow(raw["obstacle"], origin="upper", extent=local_extent, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 2].set_title("Local obstacle mask | black = obstacle", fontsize=11)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9dce1")
    ax = axes[1, 0]
    shown = ax.imshow(maps["presence"], origin="upper", extent=grid_extent, cmap=cmap, vmin=0,
                      vmax=max(float(np.nanmax(maps["presence"])), 1e-12))
    ax.set_title("Source presence probability | sampled marginal", fontsize=11)
    fig.colorbar(shown, ax=ax, shrink=.80, label="P(at least one source at grid point)")
    ax.scatter(truth[:, 0], truth[:, 1], marker="*", s=95, color="#39dcff", edgecolor="black", label="Generating sources")
    ax.scatter(best_sources[:, 0], best_sources[:, 1], marker="o", s=45, facecolors="none", edgecolors="white", label="Highest-probability set")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, 1]
    shown = ax.imshow(maps["minimum_joint_cost"], origin="upper", extent=grid_extent, cmap="viridis_r", vmin=0)
    ax.set_title("Minimum full-set MAE among sampled sets at point", fontsize=11)
    fig.colorbar(shown, ax=ax, shrink=.80, label="Conditional sampled minimum MAE")
    for i, reference in enumerate(report["references"]):
        points = np.array(reference["sources"])[:, :2]
        ax.scatter(points[:, 0], points[:, 1], marker="+", s=80, color="#ef4770",
                   label="Reference positions (overlap)" if i == 0 else None)
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes[1, :2]:
        ax.add_patch(Rectangle((x, y), side, side, fill=False, edgecolor="black", linewidth=1.8, label="Observation"))
        ax.set(xlim=(0, world), ylim=(world, 0), xlabel="World x", ylabel="World y")
    by_count = report["count_distribution"]
    ax = axes[1, 2]
    locations = np.arange(len(by_count))
    ax.bar(locations - .18, [c["base_prior_mass"] for c in by_count], width=.36, label="Declared prior", color="#a9b3c2")
    ax.bar(locations + .18, [c["teacher_probability"] for c in by_count], width=.36, label="Teacher mass", color="#446dcc")
    ax.set(xticks=locations, xticklabels=[str(c["source_count"]) for c in by_count],
           ylim=(0, 1), xlabel="Candidate source count K", ylabel="Probability")
    ax.set_title(f"Count probabilities | joint entropy={report['teacher_entropy_nats']:.3f} nats", fontsize=11)
    ax.legend(fontsize=9)
    for ax in axes[0]:
        ax.set(xlabel="World x", ylabel="World y")
    parameters = "; ".join(f"({row[0]:g}, {row[1]:g}, a={row[2]:.3f})" for row in best_sources)
    fig.text(.025, .075, f"Highest-probability set: {parameters}\n"
             f"q={best['teacher_probability']:.4g}; {report['num_source_renders']} unique visibility renders; "
             f"{report['references_checked']} separate reference checks passed.", fontsize=10)
    fig.text(.025, .02, "Grey map cells are excluded/unknown. Zero empirical mass does not prove absence. "
             "Finite joint sampling can miss compatible regions; these are not exhaustive inverse maps.", fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path
