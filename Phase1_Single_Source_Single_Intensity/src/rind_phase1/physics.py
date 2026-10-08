"""Rerender candidate sources and compute Phase I physical costs.

The physical cost is

    C = alpha * L_resp + beta * L_edge

where L_resp is a boundary-weighted mean absolute response error and
L_edge is an optional symmetric distance between observed and candidate
response boundaries.

Boundary distances are measured in pixel coordinates and normalized by
the observation-window diagonal so the edge cost is comparable across
different window sizes.

Setting beta=0 exactly recovers the response-only Pass 1 baseline.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from rind_dataset.geometry import is_driver_valid_mixed


# ---------------------------------------------------------------------------
# Euclidean distance transform
# ---------------------------------------------------------------------------

@njit(cache=True)
def _edt_1d(f):
    """Compute an exact squared Euclidean distance transform in 1D."""
    n = f.shape[0]

    d = np.empty(n, dtype=np.float64)
    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        while True:
            p = v[k]

            s = (
                (f[q] + q * q)
                - (f[p] + p * p)
            ) / (2.0 * (q - p))

            if k == 0 or s > z[k]:
                break

            k -= 1

        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0

    for q in range(n):
        while z[k + 1] < q:
            k += 1

        delta = q - v[k]
        d[q] = delta * delta + f[v[k]]

    return d


@njit(cache=True)
def _squared_distance_to_mask(mask):
    """Return squared Euclidean distance to the nearest True pixel."""
    h, w = mask.shape

    # Larger than any possible squared pixel-to-pixel distance.
    far = float(
        4 * (h * h + w * w) + 1
    )

    row_pass = np.empty(
        (h, w),
        dtype=np.float64,
    )

    row_input = np.empty(
        w,
        dtype=np.float64,
    )

    for r in range(h):
        for c in range(w):
            row_input[c] = (
                0.0 if mask[r, c] else far
            )

        row_pass[r, :] = _edt_1d(
            row_input
        )

    result = np.empty(
        (h, w),
        dtype=np.float64,
    )

    col_input = np.empty(
        h,
        dtype=np.float64,
    )

    for c in range(w):
        for r in range(h):
            col_input[r] = row_pass[r, c]

        result[:, c] = _edt_1d(
            col_input
        )

    return result


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _response_array(
    response,
    name="response",
):
    """Convert and validate a response array."""
    response = np.asarray(
        response,
        dtype=np.float64,
    )

    if (
        response.ndim != 2
        or min(response.shape) == 0
    ):
        raise ValueError(
            f"{name} must be a non-empty 2D array"
        )

    if not np.isfinite(response).all():
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return response


def _window_array(window):
    """Validate window=(x, y, size)."""
    window = np.asarray(window)

    if (
        window.shape != (3,)
        or window.dtype.kind not in "iu"
    ):
        raise ValueError(
            "window must contain integer (x, y, size)"
        )

    if int(window[2]) <= 0:
        raise ValueError(
            "window size must be positive"
        )

    return window.astype(
        np.int64,
        copy=False,
    )


def _candidate_array(candidate_xy):
    """Validate a single source coordinate."""
    candidate = np.asarray(
        candidate_xy,
        dtype=np.float64,
    )

    if (
        candidate.shape != (2,)
        or not np.isfinite(candidate).all()
    ):
        raise ValueError(
            "candidate_xy must contain two finite coordinates"
        )

    return candidate


# ---------------------------------------------------------------------------
# Observed response boundary and weighting
# ---------------------------------------------------------------------------

def response_boundary_mask(response):
    """Return pixels touching an internal response boundary.

    A pixel is considered a boundary pixel when one of its
    four direct neighbors has a different response value.

    The outside edge of the observation crop is not treated as
    a response boundary because the response outside the observed
    window is unknown.
    """
    response = _response_array(
        response
    )

    boundary = np.zeros(
        response.shape,
        dtype=bool,
    )

    # Left/right transitions.
    horizontal = (
        response[:, 1:]
        != response[:, :-1]
    )

    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    # Up/down transitions.
    vertical = (
        response[1:, :]
        != response[:-1, :]
    )

    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical

    return boundary


def boundary_weight_map(
    response,
    *,
    boundary_lambda,
    boundary_sigma,
):
    """Compute observed-boundary weights for L_resp.

    w(u) =
        1 + lambda * exp(
            -d(u)^2 / (2 sigma^2)
        )

    d(u) is Euclidean distance in pixels from pixel u to the
    closest boundary in the observed response.

    If the observation has no internal boundary, uniform weights
    are returned.
    """
    response = _response_array(
        response
    )

    boundary_lambda = float(
        boundary_lambda
    )

    boundary_sigma = float(
        boundary_sigma
    )

    if (
        not np.isfinite(boundary_lambda)
        or boundary_lambda < 0
    ):
        raise ValueError(
            "boundary_lambda must be finite and nonnegative"
        )

    if (
        not np.isfinite(boundary_sigma)
        or boundary_sigma <= 0
    ):
        raise ValueError(
            "boundary_sigma must be finite and positive"
        )

    boundary = response_boundary_mask(
        response
    )

    if (
        boundary_lambda == 0
        or not boundary.any()
    ):
        return np.ones(
            response.shape,
            dtype=np.float64,
        )

    distance_squared = (
        _squared_distance_to_mask(
            boundary
        )
    )

    weights = (
        1.0
        + boundary_lambda
        * np.exp(
            -distance_squared
            / (
                2.0
                * boundary_sigma
                * boundary_sigma
            )
        )
    )

    return weights


# ---------------------------------------------------------------------------
# Response cost
# ---------------------------------------------------------------------------

def response_cost(
    observed_response,
    candidate_response,
    weights,
):
    """Compute boundary-weighted mean absolute response error L_resp."""
    observed = _response_array(
        observed_response,
        "observed_response",
    )

    candidate = _response_array(
        candidate_response,
        "candidate_response",
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if candidate.shape != observed.shape:
        raise ValueError(
            "candidate_response and observed_response "
            "must have the same shape"
        )

    if weights.shape != observed.shape:
        raise ValueError(
            "weights and observed_response "
            "must have the same shape"
        )

    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "weights must be finite and strictly positive"
        )

    weighted_error = np.sum(
        weights
        * np.abs(
            candidate - observed
        ),
        dtype=np.float64,
    )

    total_weight = np.sum(
        weights,
        dtype=np.float64,
    )

    return float(
        weighted_error / total_weight
    )


# ---------------------------------------------------------------------------
# Boundary / edge cost
# ---------------------------------------------------------------------------

def _directed_boundary_distance(
    source_boundary,
    target_boundary,
):
    """Compute directed mean nearest-boundary distance.

    Distance is normalized by the diagonal length of the
    observation window.

    Empty-set convention:

    - empty source boundary:
        returns 0 because there are no source edges to match;

    - nonempty source boundary and empty target boundary:
        returns 1, representing a maximally missing boundary
        in that directed comparison.
    """
    source_boundary = np.asarray(
        source_boundary,
        dtype=bool,
    )

    target_boundary = np.asarray(
        target_boundary,
        dtype=bool,
    )

    if (
        source_boundary.shape
        != target_boundary.shape
    ):
        raise ValueError(
            "boundary masks must have the same shape"
        )

    if source_boundary.ndim != 2:
        raise ValueError(
            "boundary masks must be 2D"
        )

    if not source_boundary.any():
        return 0.0

    h, w = source_boundary.shape

    diagonal = float(
        np.hypot(
            max(h - 1, 0),
            max(w - 1, 0),
        )
    )

    if diagonal == 0.0:
        return 0.0

    if not target_boundary.any():
        return 1.0

    distance_squared = (
        _squared_distance_to_mask(
            target_boundary
        )
    )

    distances = np.sqrt(
        distance_squared
    )

    mean_distance = float(
        np.mean(
            distances[
                source_boundary
            ]
        )
    )

    return (
        mean_distance
        / diagonal
    )


def edge_cost(
    observed_response,
    candidate_response,
):
    """Compute symmetric normalized boundary disagreement L_edge.

    L_edge =
        D(E_observed, E_candidate)
        +
        D(E_candidate, E_observed)

    D is mean nearest-boundary distance normalized by the
    observation-window diagonal.

    The same internal four-neighbor boundary convention used
    for response weighting is used here. Crop borders themselves
    are not treated as response boundaries.
    """
    observed = _response_array(
        observed_response,
        "observed_response",
    )

    candidate = _response_array(
        candidate_response,
        "candidate_response",
    )

    if observed.shape != candidate.shape:
        raise ValueError(
            "candidate_response and observed_response "
            "must have the same shape"
        )

    observed_boundary = (
        response_boundary_mask(
            observed
        )
    )

    candidate_boundary = (
        response_boundary_mask(
            candidate
        )
    )

    if (
        not observed_boundary.any()
        and not candidate_boundary.any()
    ):
        return 0.0

    observed_to_candidate = (
        _directed_boundary_distance(
            observed_boundary,
            candidate_boundary,
        )
    )

    candidate_to_observed = (
        _directed_boundary_distance(
            candidate_boundary,
            observed_boundary,
        )
    )

    return float(
        observed_to_candidate
        + candidate_to_observed
    )


# ---------------------------------------------------------------------------
# Candidate validity
# ---------------------------------------------------------------------------

def _point_in_window(
    candidate_xy,
    window,
    world_size,
):
    """Return whether a source candidate belongs to the observation window."""
    x, y, size = map(
        float,
        window,
    )

    source_x, source_y = map(
        float,
        candidate_xy,
    )

    right = x + size
    bottom = y + size

    # Dataset partition convention:
    # a source exactly on a right/lower partition line belongs
    # to the neighboring child, except that the outer world
    # boundary belongs to the terminal window.
    x_inside = (
        x <= source_x < right
    )

    y_inside = (
        y <= source_y < bottom
    )

    if (
        right == world_size
        and source_x == world_size
    ):
        x_inside = True

    if (
        bottom == world_size
        and source_y == world_size
    ):
        y_inside = True

    return (
        x_inside
        and y_inside
    )


def _candidate_valid(
    candidate_xy,
    window,
    scene,
    world_size,
    exclude_observation_window,
):
    """Check candidate domain and obstacle restrictions."""
    if (
        np.any(candidate_xy < 0)
        or np.any(
            candidate_xy > world_size
        )
    ):
        return False

    if (
        exclude_observation_window
        and _point_in_window(
            candidate_xy,
            window,
            world_size,
        )
    ):
        return False

    return bool(
        is_driver_valid_mixed(
            candidate_xy,
            scene["obstacle_types"],
            scene["obstacle_params"],
            int(
                scene["obstacle_count"]
            ),
            domain_size=world_size,
        )
    )


def _check_settings(
    alpha,
    beta,
    fixed_intensity,
):
    """Validate physical-cost settings."""
    alpha = float(alpha)
    beta = float(beta)

    fixed_intensity = float(
        fixed_intensity
    )

    if (
        not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError(
            "alpha must be finite and positive"
        )

    if (
        not np.isfinite(beta)
        or beta < 0
    ):
        raise ValueError(
            "beta must be finite and nonnegative"
        )

    # The installed Phase I dataset is specifically the
    # single-source fixed-unit-intensity release.
    if not np.isclose(
        fixed_intensity,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "the current Phase I dataset "
            "requires fixed_intensity=1.0"
        )

    return (
        alpha,
        beta,
    )


# ---------------------------------------------------------------------------
# Public candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    dataset,
    scene_id,
    observed_response,
    window,
    candidate_xy,
    *,
    boundary_lambda,
    boundary_sigma,
    alpha=1.0,
    beta=0.0,
    fixed_intensity=1.0,
    exclude_observation_window=True,
    weights=None,
    backend="auto",
):
    """Rerender and score one candidate source.

    Invalid candidates return:

        valid = False
        candidate_response = None
        L_resp = inf
        physical_cost = inf

    If beta > 0, invalid candidates also receive L_edge = inf.

    Setting beta=0 disables L_edge and exactly preserves the
    response-only Pass 1 baseline.
    """
    observed = _response_array(
        observed_response,
        "observed_response",
    )

    window = _window_array(
        window
    )

    candidate = _candidate_array(
        candidate_xy
    )

    expected_shape = (
        int(window[2]),
        int(window[2]),
    )

    if observed.shape != expected_shape:
        raise ValueError(
            "observed_response shape "
            "must match window size"
        )

    alpha, beta = _check_settings(
        alpha,
        beta,
        fixed_intensity,
    )

    if weights is None:
        weights = boundary_weight_map(
            observed,
            boundary_lambda=boundary_lambda,
            boundary_sigma=boundary_sigma,
        )
    else:
        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if weights.shape != observed.shape:
            raise ValueError(
                "weights and observed_response "
                "must have the same shape"
            )

        if (
            not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "weights must be finite "
                "and strictly positive"
            )

    world_size = float(
        dataset.manifest[
            "global_size"
        ]
    )

    scene = dataset.get_scene(
        int(scene_id)
    )

    valid = _candidate_valid(
        candidate,
        window,
        scene,
        world_size,
        exclude_observation_window,
    )

    if not valid:
        return {
            "candidate_xy": (
                candidate.copy()
            ),
            "candidate_response": None,
            "L_resp": float("inf"),
            "L_edge": (
                float("inf")
                if beta > 0
                else None
            ),
            "physical_cost": (
                float("inf")
            ),
            "valid": False,
        }

    candidate_response = (
        dataset.rerender(
            int(scene_id),
            window,
            candidate,
            backend=backend,
        )
    )

    l_resp = response_cost(
        observed,
        candidate_response,
        weights,
    )

    if beta > 0:
        l_edge = edge_cost(
            observed,
            candidate_response,
        )

        physical_cost = (
            alpha * l_resp
            + beta * l_edge
        )
    else:
        l_edge = None

        physical_cost = (
            alpha * l_resp
        )

    return {
        "candidate_xy": (
            candidate.copy()
        ),
        "candidate_response": (
            candidate_response
        ),
        "L_resp": l_resp,
        "L_edge": l_edge,
        "physical_cost": float(
            physical_cost
        ),
        "valid": True,
    }


def evaluate_candidates(
    dataset,
    scene_id,
    observed_response,
    window,
    candidate_xy,
    *,
    boundary_lambda,
    boundary_sigma,
    alpha=1.0,
    beta=0.0,
    fixed_intensity=1.0,
    exclude_observation_window=True,
    backend="auto",
):
    """Evaluate an ordered [N,2] set of candidate source coordinates.

    Observed-boundary response weights are computed once and
    reused for all candidates.

    Candidate ordering is preserved in every returned array.

    Setting beta=0 disables L_edge and exactly preserves the
    response-only Pass 1 baseline.
    """
    observed = _response_array(
        observed_response,
        "observed_response",
    )

    window = _window_array(
        window
    )

    candidates = np.asarray(
        candidate_xy,
        dtype=np.float64,
    )

    if (
        candidates.ndim != 2
        or candidates.shape[1] != 2
    ):
        raise ValueError(
            "candidate_xy must have shape [N, 2]"
        )

    if not np.isfinite(
        candidates
    ).all():
        raise ValueError(
            "candidate_xy must contain "
            "only finite coordinates"
        )

    expected_shape = (
        int(window[2]),
        int(window[2]),
    )

    if observed.shape != expected_shape:
        raise ValueError(
            "observed_response shape "
            "must match window size"
        )

    alpha, beta = _check_settings(
        alpha,
        beta,
        fixed_intensity,
    )

    weights = boundary_weight_map(
        observed,
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
    )

    world_size = float(
        dataset.manifest[
            "global_size"
        ]
    )

    scene = dataset.get_scene(
        int(scene_id)
    )

    valid = np.zeros(
        len(candidates),
        dtype=bool,
    )

    l_resp = np.full(
        len(candidates),
        np.inf,
        dtype=np.float64,
    )

    physical_cost = np.full(
        len(candidates),
        np.inf,
        dtype=np.float64,
    )

    if beta > 0:
        l_edge = np.full(
            len(candidates),
            np.inf,
            dtype=np.float64,
        )
    else:
        l_edge = None

    for i, candidate in enumerate(
        candidates
    ):
        if not _candidate_valid(
            candidate,
            window,
            scene,
            world_size,
            exclude_observation_window,
        ):
            continue

        candidate_response = (
            dataset.rerender(
                int(scene_id),
                window,
                candidate,
                backend=backend,
            )
        )

        l_resp[i] = response_cost(
            observed,
            candidate_response,
            weights,
        )

        if beta > 0:
            l_edge[i] = edge_cost(
                observed,
                candidate_response,
            )

            physical_cost[i] = (
                alpha * l_resp[i]
                + beta * l_edge[i]
            )
        else:
            physical_cost[i] = (
                alpha * l_resp[i]
            )

        valid[i] = True

    return {
        "candidate_xy": (
            candidates.copy()
        ),
        "valid": valid,
        "L_resp": l_resp,
        "L_edge": l_edge,
        "physical_cost": (
            physical_cost
        ),
    }