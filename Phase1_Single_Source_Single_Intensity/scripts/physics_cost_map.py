"""Visual sanity check for the Phase I physical response cost.

Samples a uniform grid of hypothetical source locations for one real
observation, evaluates them with physics.py, and saves an annotated
physical-cost map.

This is a diagnostic script only. It does not define the official search
strategy or teacher distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rind_phase1.data import Phase1Dataset
from rind_phase1.physics import evaluate_candidates


def find_interesting_observation(
    dataset,
    *,
    requested_window_size=64,
):
    """Find an observation containing both response states 0 and 1."""
    for index in range(len(dataset)):
        sample = dataset[index]

        if int(sample["window"][2]) != requested_window_size:
            continue

        response = sample["response"]

        if np.any(response == 0) and np.any(response == 1):
            return sample

    raise RuntimeError(
        f"No observation with window size {requested_window_size} "
        "and both response values 0/1 was found."
    )


def make_candidate_grid(world_size, spacing):
    """Return candidate coordinates and their x/y grid coordinates."""
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
        [
            grid_x.ravel(),
            grid_y.ravel(),
        ]
    )

    return candidates, xs, ys


def cost_image(costs, valid, xs, ys, world_size):
    """Convert grid costs into a grayscale world-space image.

    Low physical cost appears bright.
    High physical cost appears dark.
    Invalid candidates appear gray.
    """
    cost_grid = costs.reshape(
        len(ys),
        len(xs),
    )

    valid_grid = valid.reshape(
        len(ys),
        len(xs),
    )

    finite_costs = cost_grid[
        valid_grid & np.isfinite(cost_grid)
    ]

    if finite_costs.size == 0:
        raise RuntimeError(
            "No valid candidate locations were evaluated."
        )

    minimum = float(np.min(finite_costs))
    upper = float(np.percentile(finite_costs, 95))

    if upper <= minimum:
        upper = minimum + 1.0

    normalized = (
        cost_grid - minimum
    ) / (upper - minimum)

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    # Low cost -> brighter.
    grayscale = (
        255.0 * (1.0 - normalized)
    ).astype(np.uint8)

    # Gray marks invalid candidates.
    grayscale[~valid_grid] = 96

    small_image = Image.fromarray(
        grayscale,
        mode="L",
    )

    image = small_image.resize(
        (world_size, world_size),
        resample=Image.Resampling.NEAREST,
    )

    return image.convert("RGB")


def world_to_pixel(x, y, world_size):
    """Convert world coordinates to image coordinates."""
    px = int(
        round(
            (float(x) / world_size)
            * (world_size - 1)
        )
    )

    py = int(
        round(
            (float(y) / world_size)
            * (world_size - 1)
        )
    )

    return px, py


def annotate_image(
    image,
    *,
    world_size,
    window,
    true_source,
    references,
):
    """Overlay the observation window and known compatible sources."""
    draw = ImageDraw.Draw(image)

    x, y, size = map(
        float,
        window,
    )

    x0, y0 = world_to_pixel(
        x,
        y,
        world_size,
    )

    x1, y1 = world_to_pixel(
        x + size,
        y + size,
        world_size,
    )

    # Observation window.
    draw.rectangle(
        [x0, y0, x1, y1],
        outline=(255, 215, 0),
        width=4,
    )

    # Compatible reference sources.
    reference_radius = 5

    for reference in references:
        px, py = world_to_pixel(
            reference[0],
            reference[1],
            world_size,
        )

        draw.ellipse(
            [
                px - reference_radius,
                py - reference_radius,
                px + reference_radius,
                py + reference_radius,
            ],
            outline=(60, 160, 255),
            width=3,
        )

    # True generating source: reference 0.
    px, py = world_to_pixel(
        true_source[0],
        true_source[1],
        world_size,
    )

    marker = 9

    draw.line(
        [
            px - marker,
            py,
            px + marker,
            py,
        ],
        fill=(255, 40, 40),
        width=4,
    )

    draw.line(
        [
            px,
            py - marker,
            px,
            py + marker,
        ],
        fill=(255, 40, 40),
        width=4,
    )


def print_best_candidates(
    candidates,
    valid,
    costs,
    *,
    count=10,
):
    """Print the lowest-cost valid grid candidates."""
    valid_indices = np.flatnonzero(
        valid & np.isfinite(costs)
    )

    ordered = valid_indices[
        np.argsort(costs[valid_indices])
    ]

    print()
    print("Lowest-cost grid candidates:")
    print("----------------------------")

    for rank, index in enumerate(
        ordered[:count],
        start=1,
    ):
        x, y = candidates[index]

        print(
            f"{rank:2d}. "
            f"({x:7.1f}, {y:7.1f}) "
            f"cost={costs[index]:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Observation size to visualize.",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=64.0,
        help="Uniform candidate-grid spacing in world units.",
    )

    parser.add_argument(
        "--boundary-lambda",
        type=float,
        default=1.0,
        help="Diagnostic boundary weight strength.",
    )

    parser.add_argument(
        "--boundary-sigma",
        type=float,
        default=1.0,
        help="Diagnostic boundary distance scale in pixels.",
    )

    args = parser.parse_args()

    ds = Phase1Dataset()

    world_size = int(
        ds.manifest["global_size"]
    )

    sample = find_interesting_observation(
        ds,
        requested_window_size=args.window_size,
    )

    scene_id = int(
        sample["scene_id"]
    )

    view_id = int(
        sample["view_id"]
    )

    response = sample["response"]
    window = sample["window"]

    references_raw = ds.get_candidates(
        scene_id,
        view_id,
    )

    references = np.stack(
        [
            reference[0, :2]
            for reference in references_raw
        ]
    )

    true_source = references[0]

    candidates, xs, ys = make_candidate_grid(
        world_size,
        args.spacing,
    )

    print(
        f"Scene {scene_id}, view {view_id}"
    )

    print(
        f"Observation window: {window.tolist()}"
    )

    print(
        f"True source: {true_source.tolist()}"
    )

    print(
        f"Evaluating {len(candidates)} candidate locations "
        f"with spacing {args.spacing:g}..."
    )

    results = evaluate_candidates(
        ds,
        scene_id,
        response,
        window,
        candidates,
        boundary_lambda=args.boundary_lambda,
        boundary_sigma=args.boundary_sigma,
        alpha=1.0,
        beta=0.0,
    )

    costs = results["physical_cost"]
    valid = results["valid"]

    print(
        f"Valid candidates: "
        f"{int(np.sum(valid))}/{len(valid)}"
    )

    print_best_candidates(
        candidates,
        valid,
        costs,
    )

    image = cost_image(
        costs,
        valid,
        xs,
        ys,
        world_size,
    )

    annotate_image(
        image,
        world_size=world_size,
        window=window,
        true_source=true_source,
        references=references,
    )

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
    f"physics_pass1_cost_map_"
    f"scene{scene_id}_view{view_id}_"
    f"spacing{args.spacing:g}.png"
    )

    image.save(
        output_path
    )

    print()
    print(
        f"Saved cost map to: {output_path}"
    )

    print()
    print("Map interpretation:")
    print("  bright = lower physical cost / more compatible")
    print("  dark   = higher physical cost")
    print("  gray   = invalid source candidate")
    print("  red +  = true generating source")
    print("  blue o = provided compatible reference source")
    print("  yellow = observation window")
    print()
    print(
        "boundary_lambda and boundary_sigma here are diagnostic "
        "values only, not finalized experiment hyperparameters."
    )


if __name__ == "__main__":
    main()