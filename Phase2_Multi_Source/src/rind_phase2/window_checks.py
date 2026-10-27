"""Exact free-pixel information and source-exclusion checks for stored windows."""
import numpy as np
from numba import njit


def response_integrals(response, obstacle):
    response, obstacle = np.asarray(response), np.asarray(obstacle, dtype=bool)
    if response.ndim != 2 or response.shape != obstacle.shape:
        raise ValueError("Response and obstacle must be equally shaped 2D arrays")
    free = ~obstacle
    values = np.where(free, response, 0).astype(np.int32)
    if not np.array_equal(values, np.where(free, response, 0)):
        raise ValueError("Information test expects integer unit-source response")
    if np.any(values < 0) or np.any(values > 4) or max(values.shape) > 1024:
        raise ValueError("Integral images support the declared 1024-world, 1-4 unit-source response")
    return _integrals(values, free)


@njit(cache=True)
def _integrals(values, free):
    """Same exact integer sums as two cumsums, in one cache-friendly row pass."""
    rows, cols = values.shape
    counts = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    total = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    squared = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for y in range(rows):
        n, s, q = 0, 0, 0
        for x in range(cols):
            value = values[y, x]
            n += int(free[y, x])
            s += value
            q += value * value
            counts[y + 1, x + 1] = counts[y, x + 1] + n
            total[y + 1, x + 1] = total[y, x + 1] + s
            squared[y + 1, x + 1] = squared[y, x + 1] + q
    return counts, total, squared


@njit(cache=True)
def rectangle_sum(integral, x, y, side):
    return np.int64(integral[y + side, x + side]) - np.int64(integral[y, x + side]) - np.int64(integral[y + side, x]) + np.int64(integral[y, x])


@njit(cache=True)
def validate_windows(free, response, squared, sources, views):
    for view in views:
        x, y, side = int(view[0]), int(view[1]), int(view[2])
        for source in sources:
            if x <= source[0] < x + side and y <= source[1] < y + side:
                return False
        n = rectangle_sum(free, x, y, side)
        total = rectangle_sum(response, x, y, side)
        if n < 2 or n * rectangle_sum(squared, x, y, side) <= total * total:
            return False
    return True
