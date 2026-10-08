"""Compare L_resp, L_edge, and their combined physical cost on one observation.

Diagnostic only. This does not define the official search or teacher distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rind_phase1.data import Phase1Dataset
from rind_phase1.physics import evaluate_candidate, evaluate_candidates


def make_candidate_grid(world_size, spacing):
    xs = np.arange(
        0.0,
        world_size + 1e-9,
        spacing,
        dtype=np.float64,
    )
    ys = np.arange(
        0.0,
        world_size + 1e-9,
        spacing,
        dtype=np.float64,
    )

    grid_x, grid_y = np.meshgrid(xs, ys)

    candidates = np.column_stack(
        [grid_x.ravel(), grid_y.ravel()]
    )

    return candidates, xs, ys


def cost_map_image(
    values,
    valid,
    xs,
    ys,
    *,
    panel_size=512,
):
    """Make a grayscale map where bright means low cost."""
    value_grid = values.reshape(
        len(ys),
        len(xs),
    )

    valid_grid = valid.reshape(
        len(ys),
        len(xs),
    )

    finite = value_grid[
        valid_grid & np.isfinite(value_grid)
    ]

    if finite.size == 0:
        raise RuntimeError(
            "No valid finite candidates were evaluated."
        )

    minimum = float(np.min(finite))
    upper = float(np.percentile(finite, 95))

    if upper <= minimum:
        upper = minimum + 1.0

    normalized = (
        value_grid - minimum
    ) / (upper - minimum)

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    # Low cost = bright.
    gray = (
        255.0 * (1.0 - normalized)
    ).astype(np.uint8)

    # Invalid candidates = fixed gray.
    gray[~valid_grid] = 96

    image = Image.fromarray(
        gray,
        mode="L",
    )

    return image.resize(
        (panel_size, panel_size),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")


def observation_image(
    response,
    *,
    panel_size=512,
):
    """Upscale the local binary response for inspection."""
    response = np.asarray(
        response,
        dtype=np.float64,
    )

    response = np.clip(
        response,
        0.0,
        1.0,
    )

    gray = (
        response * 255.0
    ).astype(np.uint8)

    image = Image.fromarray(
        gray,
        mode="L",
    )

    return image.resize(
        (panel_size, panel_size),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")


def world_to_pixel(
    x,
    y,
    world_size,
    panel_size,
):
    px = int(
        round(
            (float(x) / world_size)
            * (panel_size - 1)
        )
    )

    py = int(
        round(
            (float(y) / world_size)
            * (panel_size - 1)
        )
    )

    return px, py


def annotate_world_map(
    image,
    *,
    world_size,
    window,
    true_source,
    references,
):
    """Overlay observation window, true source, and known references."""
    draw = ImageDraw.Draw(image)

    panel_size = image.width

    x, y, size = map(
        float,
        window,
    )

    x0, y0 = world_to_pixel(
        x,
        y,
        world_size,
        panel_size,
    )
    x1, y1 = world_to_pixel(
        x + size,
        y + size,
        world_size,
        panel_size,
    )

    # Observation window.
    draw.rectangle(
        [x0, y0, x1, y1],
        outline=(255, 215, 0),
        width=3,
    )

    # Known compatible references.
    for reference in references:
        px, py = world_to_pixel(
            reference[0],
            reference[1],
            world_size,
            panel_size,
        )

        radius = 4

        draw.ellipse(
            [
                px - radius,
                py - radius,
                px + radius,
                py + radius,
            ],
            outline=(60, 160, 255),
            width=2,
        )

    # True generating source.
    px, py = world_to_pixel(
        true_source[0],
        true_source[1],
        world_size,
        panel_size,
    )

    marker = 7

    draw.line(
        [px - marker, py, px + marker, py],
        fill=(255, 40, 40),
        width=3,
    )

    draw.line(
        [px, py - marker, px, py + marker],
        fill=(255, 40, 40),
        width=3,
    )


def labeled_panel(image, title):
    """Add a small title area above an image."""
    header = 28

    panel = Image.new(
        "RGB",
        (
            image.width,
            image.height + header,
        ),
        "white",
    )

    panel.paste(
        image,
        (0, header),
    )

    draw = ImageDraw.Draw(panel)

    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    return panel


def build_comparison(
    response,
    l_resp_image,
    l_edge_image,
    combined_image,
):
    response_panel = labeled_panel(
        observation_image(response),
        "Observed local response",
    )

    resp_panel = labeled_panel(
        l_resp_image,
        "L_resp: response disagreement",
    )

    edge_panel = labeled_panel(
        l_edge_image,
        "L_edge: boundary disagreement",
    )

    combined_panel = labeled_panel(
        combined_image,
        "Combined physical cost",
    )

    width = (
        response_panel.width
        + resp_panel.width
    )

    height = (
        response_panel.height
        + edge_panel.height
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    canvas.paste(
        response_panel,
        (0, 0),
    )

    canvas.paste(
        resp_panel,
        (response_panel.width, 0),
    )

    canvas.paste(
        edge_panel,
        (0, response_panel.height),
    )

    canvas.paste(
        combined_panel,
        (
            edge_panel.width,
            resp_panel.height,
        ),
    )

    return canvas


def print_best_candidates(
    candidates,
    valid,
    l_resp,
    l_edge,
    combined,
    count=15,
):
    valid_indices = np.flatnonzero(
        valid
        & np.isfinite(combined)
    )

    ordered = valid_indices[
        np.argsort(
            combined[valid_indices]
        )
    ]

    print()
    print(
        "Lowest combined-cost candidates:"
    )
    print(
        "---------------------------------------------------------------"
    )
    print(
        " rank      x       y       L_resp       L_edge       combined"
    )
    print(
        "---------------------------------------------------------------"
    )

    for rank, index in enumerate(
        ordered[:count],
        start=1,
    ):
        x, y = candidates[index]

        print(
            f"{rank:5d} "
            f"{x:7.1f} "
            f"{y:7.1f} "
            f"{l_resp[index]:12.6f} "
            f"{l_edge[index]:12.6f} "
            f"{combined[index]:12.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--scene-id",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--view-id",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=32.0,
    )

    parser.add_argument(
        "--boundary-lambda",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--boundary-sigma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Diagnostic L_edge weight.",
    )

    args = parser.parse_args()

    ds = Phase1Dataset()

    sample = ds.get_observation(
        args.scene_id,
        args.view_id,
    )

    response = sample["response"]
    window = sample["window"]

    world_size = int(
        ds.manifest["global_size"]
    )

    reference_records = ds.get_candidates(
        args.scene_id,
        args.view_id,
    )

    references = np.stack(
        [
            reference[0, :2]
            for reference in reference_records
        ]
    )

    true_source = references[0]

    candidates, xs, ys = make_candidate_grid(
        world_size,
        args.spacing,
    )

    print(
        f"Scene {args.scene_id}, view {args.view_id}"
    )
    print(
        f"Observation window: {window.tolist()}"
    )
    print(
        f"True source: {true_source.tolist()}"
    )
    print(
        f"Candidate spacing: {args.spacing:g}"
    )
    print(
        f"beta: {args.beta:g}"
    )
    print(
        f"Evaluating {len(candidates)} candidates..."
    )

    # Pass 1 baseline.
    response_only = evaluate_candidates(
        ds,
        args.scene_id,
        response,
        window,
        candidates,
        boundary_lambda=args.boundary_lambda,
        boundary_sigma=args.boundary_sigma,
        alpha=1.0,
        beta=0.0,
    )

    # Pass 2.
    response_plus_edge = evaluate_candidates(
        ds,
        args.scene_id,
        response,
        window,
        candidates,
        boundary_lambda=args.boundary_lambda,
        boundary_sigma=args.boundary_sigma,
        alpha=1.0,
        beta=args.beta,
    )

    if not np.array_equal(
        response_only["valid"],
        response_plus_edge["valid"],
    ):
        raise RuntimeError(
            "Pass 1 and Pass 2 disagree on candidate validity."
        )

    valid = response_plus_edge["valid"]
    l_resp = response_plus_edge["L_resp"]
    l_edge = response_plus_edge["L_edge"]
    combined = response_plus_edge["physical_cost"]

    print(
        f"Valid candidates: {int(valid.sum())}/{len(valid)}"
    )

    # Exact generating source should remain zero-cost.
    exact = evaluate_candidate(
        ds,
        args.scene_id,
        response,
        window,
        true_source,
        boundary_lambda=args.boundary_lambda,
        boundary_sigma=args.boundary_sigma,
        alpha=1.0,
        beta=args.beta,
    )

    print()
    print("Exact generating source:")
    print(
        f"  L_resp        = {exact['L_resp']:.12f}"
    )
    print(
        f"  L_edge        = {exact['L_edge']:.12f}"
    )
    print(
        f"  physical_cost = {exact['physical_cost']:.12f}"
    )

    print_best_candidates(
        candidates,
        valid,
        l_resp,
        l_edge,
        combined,
    )

    resp_image = cost_map_image(
        l_resp,
        valid,
        xs,
        ys,
    )

    edge_image = cost_map_image(
        l_edge,
        valid,
        xs,
        ys,
    )

    combined_image = cost_map_image(
        combined,
        valid,
        xs,
        ys,
    )

    for image in (
        resp_image,
        edge_image,
        combined_image,
    ):
        annotate_world_map(
            image,
            world_size=world_size,
            window=window,
            true_source=true_source,
            references=references,
        )

    comparison = build_comparison(
        response,
        resp_image,
        edge_image,
        combined_image,
    )

    output_dir = Path(
        "outputs/figures"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    beta_label = f"{args.beta:g}".replace(
        ".",
        "p",
    )

    output_path = output_dir / (
        f"physics_ablation_"
        f"scene{args.scene_id}_"
        f"view{args.view_id}_"
        f"spacing{args.spacing:g}_"
        f"beta{beta_label}.png"
    )

    comparison.save(
        output_path
    )

    print()
    print(
        f"Saved comparison to: {output_path}"
    )
    print()
    print("Interpretation:")
    print("  white/bright = low cost / more compatible")
    print("  dark         = high cost / less compatible")
    print("  gray         = invalid candidate")
    print("  red +        = true source")
    print("  blue o       = known compatible reference")
    print("  yellow box   = observation window")
    print()
    print(
        "Each cost panel is contrast-normalized independently, "
        "so compare spatial patterns/rankings, not raw brightness "
        "between different panels."
    )


if __name__ == "__main__":
    main()