"""
RIND Phase I - Evaluation

Purpose
-------
Evaluate whether the Phase I system recovers source-space compatibility
structure and uncertainty from a local response observation.

This file belongs to Part E of the Phase I pipeline.

README-required evaluation goals:
1. Compare teacher and student distributions on common support.
2. Report physical compatibility.
3. Inspect ambiguous / multi-solution observations.
4. Evaluate held-out scenes and unseen obstacle combinations.
5. Separate results by observation window size.
6. Compare Lresp vs Lresp + Ledge.
7. Compare uniform vs adaptive search quality / computational cost.
8. Produce figures, metrics, failure cases, and concise reports.

Important
---------
Parts B-D may not yet be implemented.

Therefore this module is deliberately split into:

A. Functions that work now:
   - dataset / observation inspection
   - reference-candidate inspection
   - handoff validation
   - evaluation checklist
   - completed-stage summary

B. Functions that activate when upstream outputs exist:
   - teacher/student comparison
   - physical compatibility
   - ambiguity metrics
   - window-size aggregation
   - edge-loss ablation
   - search comparison
   - figure generation

This module does NOT construct:
    physical costs,
    search candidates,
    teacher distributions,
    student energies,
    trained models.

Those belong to upstream modules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rind_phase1.data import Phase1Dataset


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "reports"


WORLD_MIN = 0.0
WORLD_MAX = 1024.0

VALID_WINDOW_SIZES = {
    16,
    32,
    64,
    128,
    256,
    512,
}


# ============================================================
# Output helpers
# ============================================================

def ensure_output_dirs() -> None:
    """Create evaluation output directories if necessary."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _json_compatible(value: Any) -> Any:
    """Convert common NumPy objects to JSON-compatible Python objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_compatible(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item)
            for item in value
        ]

    return value


def save_json_report(
    data: Mapping[str, Any],
    filename: str,
) -> Path:
    """Save a JSON report under outputs/reports/."""

    ensure_output_dirs()

    path = REPORT_DIR / filename

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_compatible(dict(data)),
            file,
            indent=2,
        )

    return path


def save_text_report(
    text: str,
    filename: str,
) -> Path:
    """Save a text / Markdown report under outputs/reports/."""

    ensure_output_dirs()

    path = REPORT_DIR / filename

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


# ============================================================
# Optional plotting dependency
# ============================================================

def _get_pyplot():
    """
    Import matplotlib only when a figure is actually requested.

    Keeping plotting as a lazy dependency allows the core evaluation
    utilities and smoke checks to run even if matplotlib is not part
    of the current basic environment.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure generation requires matplotlib, but matplotlib "
            "is not available in the current environment."
        ) from exc

    return plt


# ============================================================
# Basic numeric utilities
# ============================================================

def as_1d_float(
    values: Any,
    name: str,
) -> np.ndarray:
    """Convert an input to a finite 1D float array."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be 1D; "
            f"got shape {array.shape}."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def normalize_probability(
    prob: Any,
    name: str = "probability",
) -> np.ndarray:
    """
    Normalize non-negative masses into a discrete probability distribution.
    """

    prob = as_1d_float(
        prob,
        name,
    )

    if np.any(prob < 0):
        raise ValueError(
            f"{name} contains negative values."
        )

    total = float(
        np.sum(prob)
    )

    if total <= 0:
        raise ValueError(
            f"{name} has zero total mass."
        )

    return prob / total


def validate_probability(
    prob: Any,
    name: str = "probability",
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate a discrete probability distribution."""

    prob = as_1d_float(
        prob,
        name,
    )

    if np.any(prob < 0):
        raise ValueError(
            f"{name} contains negative values."
        )

    total = float(
        np.sum(prob)
    )

    if not np.isclose(
        total,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            f"{name} must sum to 1; "
            f"got {total:.10f}."
        )

    return prob


def entropy(
    prob: Any,
) -> float:
    """
    Shannon entropy of a discrete probability distribution.

    Natural logarithm is used.
    """

    p = normalize_probability(
        prob,
    )

    positive = p > 0

    return float(
        -np.sum(
            p[positive]
            * np.log(p[positive])
        )
    )


def effective_support_size(
    prob: Any,
) -> float:
    """
    Effective number of supported candidates.

    Defined as exp(entropy).
    """

    return float(
        math.exp(
            entropy(prob)
        )
    )


# ============================================================
# Current data-layer inspection
# ============================================================

def inspect_observation(
    dataset: Phase1Dataset,
    index: int = 0,
) -> dict[str, Any]:
    """
    Inspect one currently implemented Phase I observation.

    This does not depend on Parts B-D.
    """

    sample = dataset[index]

    required = {
        "scene_id",
        "view_id",
        "response",
        "window",
    }

    if not required.issubset(sample):
        raise ValueError(
            "Unexpected observation fields: "
            f"{sorted(sample.keys())}"
        )

    from .interfaces import validate_observation
    validate_observation(sample)

    response = np.asarray(
        sample["response"]
    )

    window = np.asarray(
        sample["window"]
    )

    if response.ndim != 2:
        raise ValueError(
            "response must have shape [L, L]."
        )

    if window.shape != (3,):
        raise ValueError(
            "window must have shape [3]."
        )

    x, y, size = map(
        int,
        window,
    )

    if size not in VALID_WINDOW_SIZES:
        raise ValueError(
            f"Unexpected window size: {size}."
        )

    if response.shape != (
        size,
        size,
    ):
        raise ValueError(
            "Response shape does not match "
            "window size."
        )

    unique_values = np.unique(
        response
    )

    if not np.all(
        np.isin(
            unique_values,
            [0.0, 1.0],
        )
    ):
        raise ValueError(
            "Response is expected to be binary."
        )

    return {
        "scene_id": int(sample["scene_id"]),
        "view_id": int(sample["view_id"]),
        "window": window.copy(),
        "response": response.copy(),
        "x": x,
        "y": y,
        "window_size": size,
    }


def inspect_reference_candidates(
    dataset: Phase1Dataset,
    scene_id: int,
    view_id: int,
) -> dict[str, Any]:
    """
    Inspect dataset-provided reference candidates.

    These references are compatibility examples only.
    They are NOT a teacher distribution and NOT exhaustive
    source-space coverage.
    """

    references = dataset.get_candidates(
        scene_id,
        view_id,
    )

    array = np.asarray(
        references,
        dtype=np.float64,
    )

    if array.ndim != 3:
        raise ValueError(
            "Reference candidates are expected "
            f"to have 3 dimensions; got {array.shape}."
        )

    if array.shape[1:] != (1, 3):
        raise ValueError(
            "Expected reference shape [N, 1, 3]; "
            f"got {array.shape}."
        )

    candidate_xy = array[
        :,
        0,
        :2,
    ]

    strength = array[
        :,
        0,
        2,
    ]

    return {
        "candidate_xy": candidate_xy,
        "strength": strength,
        "generating_source_xy": (
            candidate_xy[0].copy()
        ),
        "num_references": len(candidate_xy),
    }


def verify_reference_rerenders(
    dataset: Phase1Dataset,
    scene_id: int,
    window: Sequence[int],
    response: np.ndarray,
    candidate_xy: np.ndarray,
) -> dict[str, Any]:
    """
    Verify that dataset reference candidates reproduce
    the current sampled observation.
    """

    matches: list[bool] = []

    for candidate in candidate_xy:

        fresh = dataset.rerender(
            scene_id,
            window,
            candidate,
        )

        matches.append(
            bool(
                np.array_equal(
                    fresh,
                    response,
                )
            )
        )

    return {
        "num_candidates": len(matches),
        "num_exact_matches": int(
            sum(matches)
        ),
        "all_exact": bool(
            all(matches)
        ),
        "matches": matches,
    }


# ============================================================
# Observation / reference visualization
# ============================================================

def plot_observation(
    response: np.ndarray,
    scene_id: int,
    view_id: int,
    save: bool = False,
) -> None:
    """
    Plot the observed binary response.

    This is a diagnostic observation figure, not a teacher figure.
    """

    plt = _get_pyplot()

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    image = ax.imshow(
        response,
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Response",
    )

    ax.set_xlabel(
        "Local x / column"
    )

    ax.set_ylabel(
        "Local y / row"
    )

    ax.set_title(
        f"Observed Response - "
        f"Scene {scene_id}, View {view_id}"
    )

    fig.tight_layout()

    if save:
        ensure_output_dirs()

        fig.savefig(
            FIGURE_DIR
            / (
                f"scene_{scene_id}_"
                f"view_{view_id}_observation.png"
            ),
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()


def plot_reference_candidates(
    candidate_xy: np.ndarray,
    scene_id: int,
    view_id: int,
    local_zoom: bool = False,
    save: bool = False,
) -> None:
    """
    Plot dataset reference candidates.

    Important:
    This is NOT a teacher posterior figure.
    """

    plt = _get_pyplot()

    xy = np.asarray(
        candidate_xy,
        dtype=np.float64,
    )

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            "candidate_xy must have shape [N, 2]."
        )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    if len(xy) > 1:
        ax.scatter(
            xy[1:, 0],
            xy[1:, 1],
            label="Compatible references",
        )

    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        marker="*",
        s=180,
        label="Generating source",
    )

    if local_zoom:

        ax.ticklabel_format(
            style="plain",
            axis="both",
            useOffset=False,
        )

        title = (
            "Reference Candidates - Local Zoom"
        )

    else:

        ax.set_xlim(
            WORLD_MIN,
            WORLD_MAX,
        )

        # README coordinates use y downward.
        ax.set_ylim(
            WORLD_MAX,
            WORLD_MIN,
        )

        title = (
            "Reference Candidates - Global View"
        )

    ax.set_xlabel(
        "Source x"
    )

    ax.set_ylabel(
        "Source y"
    )

    ax.set_title(
        f"{title}\n"
        f"Scene {scene_id}, View {view_id}"
    )

    ax.legend()

    fig.tight_layout()

    if save:

        ensure_output_dirs()

        suffix = (
            "local"
            if local_zoom
            else "global"
        )

        fig.savefig(
            FIGURE_DIR
            / (
                f"scene_{scene_id}_"
                f"view_{view_id}_"
                f"reference_{suffix}.png"
            ),
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# Handoff validation
# ============================================================

def validate_candidate_xy(
    candidate_xy: Any,
) -> np.ndarray:
    """Validate candidate coordinates [N, 2]."""

    xy = np.asarray(
        candidate_xy,
        dtype=np.float64,
    )

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            "candidate_xy must have shape [N, 2]."
        )

    if len(xy) == 0:
        raise ValueError(
            "candidate_xy cannot be empty."
        )

    if not np.all(
        np.isfinite(xy)
    ):
        raise ValueError(
            "candidate_xy contains non-finite values."
        )

    if np.any(
        xy < WORLD_MIN
    ) or np.any(
        xy > WORLD_MAX
    ):
        raise ValueError(
            "candidate_xy contains coordinates "
            "outside [0, 1024]."
        )

    return xy


def validate_teacher_record(
    record: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """
    Validate the minimum README teacher handoff.

    Required:
        scene_id
        view_id
        candidate_xy [N,2]
        valid [N]
        physical_cost [N]
        teacher_prob [N]
        temperature
        config_id
    """

    required = {
        "scene_id",
        "view_id",
        "candidate_xy",
        "valid",
        "physical_cost",
        "teacher_prob",
        "temperature",
        "config_id",
    }

    missing = (
        required
        - set(record.keys())
    )

    if missing:
        raise KeyError(
            "Teacher record is missing: "
            f"{sorted(missing)}"
        )

    xy = validate_candidate_xy(
        record["candidate_xy"]
    )

    n = len(xy)

    valid = np.asarray(
        record["valid"],
        dtype=bool,
    )

    cost = np.asarray(
        record["physical_cost"],
        dtype=np.float64,
    )

    prob = np.asarray(
        record["teacher_prob"],
        dtype=np.float64,
    )

    if valid.shape != (n,):
        raise ValueError(
            "valid must have shape [N]."
        )

    if cost.shape != (n,):
        raise ValueError(
            "physical_cost must have shape [N]."
        )

    if prob.shape != (n,):
        raise ValueError(
            "teacher_prob must have shape [N]."
        )

    if not np.all(
        np.isfinite(cost[valid])
    ):
        raise ValueError(
            "Valid physical costs contain "
            "non-finite values."
        )

    if np.sum(valid) == 0:
        raise ValueError(
            "Teacher record has no valid candidates."
        )

    temperature = float(
        record["temperature"]
    )

    if not np.isfinite(
        temperature
    ) or temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    validate_probability(
        prob,
        "teacher_prob",
    )

    # README: invalid candidates receive zero mass.
    if np.any(
        prob[~valid] > 1e-12
    ):
        raise ValueError(
            "Invalid teacher candidates "
            "must have zero probability."
        )

    return {
        "candidate_xy": xy,
        "valid": valid,
        "physical_cost": cost,
        "teacher_prob": prob,
    }


def validate_student_result(
    result: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """
    Validate minimum student result fields.

    Required:
        candidate_xy
        energy
        student_prob

    Model/config identifiers are expected in experiment records
    but their naming is not yet fixed, so this function does not
    invent field names for them.
    """

    required = {
        "candidate_xy",
        "energy",
        "student_prob",
    }

    missing = (
        required
        - set(result.keys())
    )

    if missing:
        raise KeyError(
            "Student result is missing: "
            f"{sorted(missing)}"
        )

    xy = validate_candidate_xy(
        result["candidate_xy"]
    )

    n = len(xy)

    energy = as_1d_float(
        result["energy"],
        "energy",
    )

    prob = validate_probability(
        result["student_prob"],
        "student probability",
    )

    if energy.shape != (n,):
        raise ValueError(
            "energy must have shape [N]."
        )

    if prob.shape != (n,):
        raise ValueError(
            "student probability must have shape [N]."
        )

    return {
        "candidate_xy": xy,
        "energy": energy,
        "student_prob": prob,
    }


def validate_common_support(
    teacher_record: Mapping[str, Any],
    student_result: Mapping[str, Any],
    atol: float = 1e-10,
) -> None:
    """
    Verify that teacher and student use the same candidates
    in the same order.

    Candidate ordering is part of the handoff contract.
    """

    for key in ("scene_id", "view_id", "window", "valid", "config_id"):
        if key not in student_result or not np.array_equal(teacher_record[key], student_result[key]):
            raise ValueError(f"Teacher/student mismatch: {key}")
    from part_e.checks import prediction_support
    mask = prediction_support(student_result, np.asarray(teacher_record["valid"], dtype=bool))
    if np.any(np.asarray(student_result["student_prob"])[~mask] != 0):
        raise ValueError("Invalid candidates must have zero student probability")

    teacher_xy = np.asarray(
        teacher_record["candidate_xy"],
        dtype=np.float64,
    )

    student_xy = np.asarray(
        student_result["candidate_xy"],
        dtype=np.float64,
    )

    if teacher_xy.shape != student_xy.shape:
        raise ValueError(
            "Teacher/student candidate sets "
            "have different shapes."
        )

    if not np.allclose(
        teacher_xy,
        student_xy,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Teacher/student candidates are not "
            "aligned in the same order."
        )


# ============================================================
# Teacher / student agreement metrics
# ============================================================

def teacher_student_metrics(
    teacher_prob: Any,
    student_prob: Any,
) -> dict[str, float]:
    """
    Compare two discrete probability distributions
    on common candidate support.

    These are evaluation metrics, not training objectives.
    """

    q = normalize_probability(
        teacher_prob,
        "teacher_prob",
    )

    p = normalize_probability(
        student_prob,
        "student_prob",
    )

    if q.shape != p.shape:
        raise ValueError(
            "Teacher and student distributions "
            "must have the same shape."
        )

    eps = 1e-12

    q_safe = np.clip(
        q,
        eps,
        1.0,
    )

    p_safe = np.clip(
        p,
        eps,
        1.0,
    )

    midpoint = 0.5 * (
        q_safe + p_safe
    )

    l1 = float(
        np.sum(
            np.abs(q - p)
        )
    )

    total_variation = (
        0.5 * l1
    )

    kl_teacher_student = float(
        np.sum(
            q_safe
            * np.log(
                q_safe / p_safe
            )
        )
    )

    js = float(
        0.5
        * np.sum(
            q_safe
            * np.log(
                q_safe / midpoint
            )
        )
        +
        0.5
        * np.sum(
            p_safe
            * np.log(
                p_safe / midpoint
            )
        )
    )

    return {
        "l1_distance": l1,
        "total_variation": total_variation,
        "kl_teacher_to_student": (
            kl_teacher_student
        ),
        "js_divergence": js,
        "teacher_entropy": entropy(q),
        "student_entropy": entropy(p),
        "teacher_effective_support": (
            effective_support_size(q)
        ),
        "student_effective_support": (
            effective_support_size(p)
        ),
    }


# ============================================================
# Physical compatibility
# ============================================================

def physical_compatibility_metrics(
    probability: Any,
    physical_cost: Any,
    valid: Any | None = None,
) -> dict[str, float]:
    """
    Measure whether predicted probability mass lies on
    physically compatible low-cost candidates.

    Lower expected physical cost is better.

    This does NOT construct the physical cost.
    """

    prob = normalize_probability(
        probability,
        "student_prob",
    )

    cost = np.asarray(
        physical_cost,
        dtype=np.float64,
    )

    if prob.shape != cost.shape:
        raise ValueError(
            "probability and physical_cost "
            "must have the same shape."
        )

    if valid is None:

        mask = np.ones(
            len(prob),
            dtype=bool,
        )

    else:

        mask = np.asarray(
            valid,
            dtype=bool,
        )

        if mask.shape != prob.shape:
            raise ValueError(
                "valid must match probability shape."
            )

    if np.sum(mask) == 0:
        raise ValueError(
            "No valid candidates available."
        )

    valid_prob_mass = float(
        np.sum(
            prob[mask]
        )
    )

    valid_cost = cost[mask]
    expected_cost = (float(np.sum(prob[mask] * valid_cost) / valid_prob_mass)
                     if valid_prob_mass > 0 else None)

    most_probable_index = int(
        np.argmax(prob)
    )

    return {
        "expected_physical_cost": expected_cost if not np.any(prob[~mask] > 0) else None,
        "expected_physical_cost_infinite": bool(np.any(prob[~mask] > 0)),
        "expected_valid_cost": expected_cost,
        "invalid_probability_mass": float(prob[~mask].sum()),
        "minimum_physical_cost": float(
            np.min(valid_cost)
        ),
        "most_probable_candidate_cost": float(cost[most_probable_index]) if mask[most_probable_index] else None,
        "valid_probability_mass": (
            valid_prob_mass
        ),
    }


# ============================================================
# Ambiguity / multi-solution summaries
# ============================================================

def ambiguity_metrics(
    probability: Any,
    top_k: int = 5,
) -> dict[str, float]:
    """
    Basic discrete uncertainty summary.

    This does not by itself prove geometric multimodality.
    Representative source-space figures are still needed.
    """

    p = normalize_probability(
        probability,
    )

    sorted_prob = np.sort(
        p
    )[::-1]

    k = min(
        int(top_k),
        len(sorted_prob),
    )

    return {
        "entropy": entropy(p),
        "effective_support_size": (
            effective_support_size(p)
        ),
        "max_probability": float(
            sorted_prob[0]
        ),
        f"top_{k}_probability_mass": float(
            np.sum(
                sorted_prob[:k]
            )
        ),
    }


# ============================================================
# Teacher / student source-space visualization
# ============================================================

def plot_candidate_distribution(
    candidate_xy: Any,
    probability: Any,
    title: str,
    label: str,
    save_name: str | None = None,
) -> None:
    """
    Scatter candidate locations using color to represent
    discrete probability mass.

    Important:
    The plot represents discrete candidate masses.
    It must not be described as a calibrated continuous density.
    """

    plt = _get_pyplot()

    xy = validate_candidate_xy(
        candidate_xy
    )

    prob = normalize_probability(
        probability,
    )

    if len(xy) != len(prob):
        raise ValueError(
            "candidate_xy and probability "
            "must have equal length."
        )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=prob,
        s=35,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label=label,
    )

    ax.set_xlim(
        WORLD_MIN,
        WORLD_MAX,
    )

    # World y points downward.
    ax.set_ylim(
        WORLD_MAX,
        WORLD_MIN,
    )

    ax.set_xlabel(
        "Source x"
    )

    ax.set_ylabel(
        "Source y"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    if save_name is not None:

        ensure_output_dirs()

        fig.savefig(
            FIGURE_DIR
            / save_name,
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()


def plot_teacher_student_common_support(
    candidate_xy: Any,
    teacher_prob: Any,
    student_prob: Any,
    scene_id: int,
    view_id: int,
    save: bool = False,
) -> None:
    """
    Generate separate teacher and student figures
    on the same candidate support.
    """

    teacher_name = None
    student_name = None

    if save:

        teacher_name = (
            f"scene_{scene_id}_"
            f"view_{view_id}_teacher.png"
        )

        student_name = (
            f"scene_{scene_id}_"
            f"view_{view_id}_student.png"
        )

    plot_candidate_distribution(
        candidate_xy,
        teacher_prob,
        title=(
            "Teacher Candidate Probability Mass\n"
            f"Scene {scene_id}, View {view_id}"
        ),
        label="Teacher probability mass",
        save_name=teacher_name,
    )

    plot_candidate_distribution(
        candidate_xy,
        student_prob,
        title=(
            "Student Candidate Probability Mass\n"
            f"Scene {scene_id}, View {view_id}"
        ),
        label="Student probability mass",
        save_name=student_name,
    )


# ============================================================
# Complete single-record evaluation
# ============================================================

def evaluate_teacher_student_record(
    teacher_record: Mapping[str, Any],
    student_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Evaluate one observation once both teacher and student
    results exist.
    """

    teacher = validate_teacher_record(
        teacher_record
    )

    student = validate_student_result(
        student_result
    )

    validate_common_support(
        teacher_record,
        student_result,
    )

    agreement = teacher_student_metrics(
        teacher["teacher_prob"],
        student["student_prob"],
    )

    physical = (
        physical_compatibility_metrics(
            student["student_prob"],
            teacher["physical_cost"],
            teacher["valid"],
        )
    )

    teacher_ambiguity = (
        ambiguity_metrics(
            teacher["teacher_prob"]
        )
    )

    student_ambiguity = (
        ambiguity_metrics(
            student["student_prob"]
        )
    )

    return {
        "scene_id": int(
            teacher_record["scene_id"]
        ),
        "view_id": int(
            teacher_record["view_id"]
        ),
        "agreement": agreement,
        "student_physical_compatibility": (
            physical
        ),
        "teacher_uncertainty": (
            teacher_ambiguity
        ),
        "student_uncertainty": (
            student_ambiguity
        ),
    }


# ============================================================
# Window-size aggregation
# ============================================================

def group_metrics_by_window_size(
    records: Iterable[Mapping[str, Any]],
    metric_key: str,
) -> dict[int, dict[str, float]]:
    """
    Aggregate one scalar metric by observation window size.

    Expected record example:
        {
            "window_size": 64,
            "js_divergence": 0.12
        }
    """

    grouped: dict[
        int,
        list[float],
    ] = {}

    for record in records:

        size = int(
            record["window_size"]
        )

        value = float(
            record[metric_key]
        )

        grouped.setdefault(
            size,
            [],
        ).append(
            value
        )

    summary: dict[
        int,
        dict[str, float],
    ] = {}

    for size, values in grouped.items():

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        summary[size] = {
            "count": int(
                len(array)
            ),
            "mean": float(
                np.mean(array)
            ),
            "median": float(
                np.median(array)
            ),
            "std": float(
                np.std(array)
            ),
        }

    return dict(
        sorted(
            summary.items()
        )
    )


# ============================================================
# Lresp vs Lresp + Ledge ablation
# ============================================================

def compare_edge_ablation(
    response_only_metrics: Mapping[str, float],
    response_edge_metrics: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """
    Compare matched scalar evaluation results from:

        Lresp
        Lresp + Ledge

    This function does not compute Lresp or Ledge.
    """

    common = (
        set(
            response_only_metrics
        )
        &
        set(
            response_edge_metrics
        )
    )

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for metric in sorted(
        common
    ):

        baseline = float(
            response_only_metrics[
                metric
            ]
        )

        edge = float(
            response_edge_metrics[
                metric
            ]
        )

        result[metric] = {
            "Lresp": baseline,
            "Lresp_plus_Ledge": edge,
            "difference": (
                edge - baseline
            ),
        }

    return result


# ============================================================
# Uniform vs adaptive search comparison
# ============================================================

def search_quality_cost_summary(
    *,
    uniform_evaluations: int,
    adaptive_evaluations: int,
    uniform_recall: float,
    adaptive_recall: float,
    uniform_final_spacing: float,
    adaptive_final_spacing: float,
) -> dict[str, float]:
    """
    Summarize uniform vs adaptive search.

    Search recall must be measured on a common uniform
    reference evaluation grid.

    Pruned reference locations should count as uncovered.
    """

    if uniform_evaluations <= 0:
        raise ValueError(
            "uniform_evaluations must be positive."
        )

    if adaptive_evaluations <= 0:
        raise ValueError(
            "adaptive_evaluations must be positive."
        )

    for name, value in {
        "uniform_recall": uniform_recall,
        "adaptive_recall": adaptive_recall,
    }.items():

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must lie in [0,1]."
            )

    reduction = (
        1.0
        -
        adaptive_evaluations
        / uniform_evaluations
    )

    return {
        "uniform_physical_evaluations": int(
            uniform_evaluations
        ),
        "adaptive_physical_evaluations": int(
            adaptive_evaluations
        ),
        "evaluation_reduction_fraction": float(
            reduction
        ),
        "uniform_search_recall": float(
            uniform_recall
        ),
        "adaptive_search_recall": float(
            adaptive_recall
        ),
        "uniform_final_spacing": float(
            uniform_final_spacing
        ),
        "adaptive_final_spacing": float(
            adaptive_final_spacing
        ),
    }


# ============================================================
# Evaluation checklist
# ============================================================

def evaluation_checklist() -> dict[str, Any]:
    """
    README-aligned checklist for Part E.

    This deliberately distinguishes:
        available now,
        blocked by upstream modules,
        final research obligations.
    """

    return {
        "available_now": [
            (
                "Load and inspect held-out-compatible "
                "observation records."
            ),
            (
                "Inspect dataset reference candidates "
                "without treating them as teacher targets."
            ),
            (
                "Validate teacher/student handoff "
                "record structure."
            ),
            (
                "Prepare observation and "
                "candidate-space visualization utilities."
            ),
            (
                "Generate completed-stage summary."
            ),
        ],
        "requires_physics_teacher_search": [
            (
                "Plot physical-cost and "
                "teacher-distribution results."
            ),
            (
                "Inspect ambiguous alternatives."
            ),
            (
                "Find representative one-boundary and "
                "two-independent-boundary cases."
            ),
            (
                "Run Lresp vs Lresp + Ledge ablation."
            ),
            (
                "Compare uniform vs adaptive search "
                "quality and physical-evaluation cost."
            ),
        ],
        "requires_trained_student": [
            (
                "Teacher/student distribution agreement "
                "on common support."
            ),
            (
                "Student physical compatibility."
            ),
            (
                "Held-out unseen-scene evaluation."
            ),
            (
                "Unseen obstacle-combination evaluation."
            ),
            (
                "Window-size-stratified student results."
            ),
            (
                "Observation-teacher-student "
                "comparison figures."
            ),
        ],
        "final_questions": [
            (
                "Are source-space constraints and "
                "uncertainty recovered?"
            ),
            (
                "Does Ledge add independent value "
                "beyond Lresp?"
            ),
            (
                "Does adaptive search save computation "
                "while retaining plausible regions?"
            ),
        ],
    }


# ============================================================
# Completed-stage summary
# ============================================================

def completed_stage_summary() -> str:
    """
    Produce a concise README-style status report.

    Written under the current assumption that Parts B-D
    are not yet implemented.
    """

    return """# RIND Phase I Evaluation Status

## Implemented and verified

- Dataset installation and access.
- Observation interface (`scene_id`, `view_id`, `response`, `window`).
- Scene and view lookup.
- Dataset reference candidates.
- Reference-candidate rerendering.
- Evaluation handoff validators.
- Teacher/student distribution comparison utilities.
- Physical-compatibility metric utilities.
- Ambiguity / uncertainty summaries.
- Window-size aggregation utilities.
- Edge-ablation comparison utilities.
- Uniform/adaptive search comparison utilities.
- Figure and report helpers.

## Not yet executable end-to-end

The following evaluation stages require upstream research outputs:

- candidate physical costs;
- uniform/adaptive search records;
- teacher distributions;
- trained student energies and probabilities;
- agreed held-out scene splits / unseen obstacle-combination protocol.

## Important interpretation

Dataset reference candidates are compatibility examples only.
They are not exhaustive candidate coverage and are not a teacher
probability distribution.

Teacher and student comparison must use aligned candidate support.

Adaptive probabilities are discrete candidate probability masses,
not continuous spatial densities.

Coordinate error alone is not sufficient evidence of recovering
source-space constraints or uncertainty.

## Next integration milestone

One observation
-> candidate physical costs
-> teacher distribution
-> student scores
-> visualization
"""


# ============================================================
# Current runnable smoke evaluation
# ============================================================

def run_current_smoke_evaluation(
    sample_index: int = 0,
    save_reports: bool = True,
    save_figures: bool = False,
) -> dict[str, Any]:
    """
    Run everything that is legitimately available before
    Parts B-D are completed.

    This is NOT a full Phase I evaluation.
    """

    print(
        "=== RIND Phase I "
        "Evaluation Smoke Check ==="
    )

    dataset = Phase1Dataset()

    observation = inspect_observation(
        dataset,
        index=sample_index,
    )

    references = (
        inspect_reference_candidates(
            dataset,
            observation["scene_id"],
            observation["view_id"],
        )
    )

    rerender_check = (
        verify_reference_rerenders(
            dataset,
            observation["scene_id"],
            observation["window"],
            observation["response"],
            references["candidate_xy"],
        )
    )

    print()
    print(
        "scene_id:",
        observation["scene_id"],
    )

    print(
        "view_id:",
        observation["view_id"],
    )

    print(
        "window:",
        observation["window"],
    )

    print(
        "response shape:",
        observation["response"].shape,
    )

    print(
        "reference candidates:",
        references["num_references"],
    )

    print(
        "reference rerenders:",
        f"{rerender_check['num_exact_matches']}"
        f"/{rerender_check['num_candidates']}",
    )

    print(
        "all reference rerenders exact:",
        rerender_check["all_exact"],
    )

    if not rerender_check[
        "all_exact"
    ]:
        raise RuntimeError(
            "Dataset reference rerender "
            "verification failed."
        )

    result = {
        "status": (
            "current implemented "
            "evaluation prerequisites pass"
        ),
        "scene_id": (
            observation["scene_id"]
        ),
        "view_id": (
            observation["view_id"]
        ),
        "window_size": (
            observation["window_size"]
        ),
        "num_reference_candidates": (
            references[
                "num_references"
            ]
        ),
        "reference_rerender_check": (
            rerender_check
        ),
        "upstream_research_available": False,
        "full_evaluation_complete": False,
    }

    if save_figures:

        plot_observation(
            observation["response"],
            observation["scene_id"],
            observation["view_id"],
            save=True,
        )

        plot_reference_candidates(
            references["candidate_xy"],
            observation["scene_id"],
            observation["view_id"],
            local_zoom=False,
            save=True,
        )

        plot_reference_candidates(
            references["candidate_xy"],
            observation["scene_id"],
            observation["view_id"],
            local_zoom=True,
            save=True,
        )

    if save_reports:

        save_json_report(
            result,
            "current_stage_metrics.json",
        )

        save_json_report(
            evaluation_checklist(),
            "evaluation_checklist.json",
        )

        save_text_report(
            completed_stage_summary(),
            "current_stage_summary.md",
        )

    print()
    print(
        "Current evaluation prerequisites: PASS"
    )

    print(
        "Full research evaluation: "
        "NOT YET AVAILABLE"
    )

    print(
        "Reason: Parts B-D outputs "
        "do not yet exist."
    )

    return result


# ============================================================
# Command-line entry point
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RIND Phase I evaluation utilities."
        )
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run evaluation checks that are "
            "currently possible before Parts B-D "
            "are implemented."
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help=(
            "Dataset observation index used "
            "by --smoke."
        ),
    )

    parser.add_argument(
        "--save-figures",
        action="store_true",
        help=(
            "Save currently available "
            "diagnostic figures."
        ),
    )

    parser.add_argument(
        "--no-save-reports",
        action="store_true",
        help=(
            "Do not write current-stage reports."
        ),
    )

    parser.add_argument(
        "--checklist",
        action="store_true",
        help=(
            "Print the README-aligned "
            "evaluation checklist."
        ),
    )

    return parser


def main() -> None:
    parser = build_arg_parser()

    args = parser.parse_args()

    if args.checklist:

        print(
            json.dumps(
                evaluation_checklist(),
                indent=2,
            )
        )

    if args.smoke:

        run_current_smoke_evaluation(
            sample_index=args.sample_index,
            save_reports=(
                not args.no_save_reports
            ),
            save_figures=(
                args.save_figures
            ),
        )

    if (
        not args.smoke
        and not args.checklist
    ):

        parser.print_help()


if __name__ == "__main__":
    main()