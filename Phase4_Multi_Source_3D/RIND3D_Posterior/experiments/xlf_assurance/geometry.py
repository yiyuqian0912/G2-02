"""Small exact geometric reference used by the assurance tests (axis-box scenes)."""
from fractions import Fraction as Q
from itertools import product

from ledger import Box


def rows_of(box):
    rows = []
    for k in range(3):
        for sign, offset in ((-1, box.lo[k]), (1, -box.hi[k])):
            row = [0, 0, 0, offset]
            row[k] = sign
            rows.append(tuple(row))
    return rows


def affine(row, point):
    return row[3] + sum(a * p for a, p in zip(row[:3], point))


def inverse_rows(rows, receiver):
    """Full Fourier elimination at the receiver, including zero face evaluations."""
    a = [affine(row, receiver) for row in rows]
    front = [i for i, v in enumerate(a) if v > 0]
    if not front:
        return ()  # receiver inside the closed solid
    result = [r for r, v in zip(rows, a) if v >= 0]
    for i in front:
        for j, v in enumerate(a):
            if v < 0:
                result.append(tuple(a[i]*y - a[j]*x for x, y in zip(rows[i], rows[j])))
    return tuple(result)


def segment_hit(rows, source, receiver):
    """Independent parameter-interval clipping, including endpoints."""
    lo, hi = Q(0), Q(1)
    for row in rows:
        fs, fr = affine(row, source), affine(row, receiver)
        delta = fr - fs
        if delta == 0:
            if fs > 0:
                return False
        elif delta > 0:
            hi = min(hi, -Q(fs) / delta)
        else:
            lo = max(lo, -Q(fs) / delta)
    return lo <= hi


def physical_classifier(blockers, origin, n, side, observations):
    """One complete tiny sampled window, a closed exclusion box and strict clearance.

    Each observation is 0 (blocked), 1 (visible) or 2 (solid). Prove IN only on
    free source boxes with every receiver resolved. All predicates use rationals.
    """
    if len(origin) != 3 or n < 1 or side < 1 or any(not 0 <= v <= n-side for v in origin):
        raise ValueError("invalid window")
    window = Box(tuple(Q(v, n) for v in origin), tuple(Q(v+side, n) for v in origin))
    if len(observations) != side**3:
        raise ValueError("incomplete observation")
    bodies = [rows_of(b) for b in blockers]
    constraints = []
    for ijk, y in zip(product(range(side), repeat=3), observations):
        r = tuple(Q(2*(a+i)+1, 2*n) for a, i in zip(origin, ijk))
        solid = any(all(affine(row,r) <= 0 for row in rows) for rows in bodies)
        if y not in (0, 1, 2) or (y == 2) != solid:
            raise ValueError("observation and geometry disagree on solid")
        if not solid:
            constraints.append((y, [inverse_rows(rows, r) for rows in bodies]))

    def classify(box):
        if window.contains(box) or any(b.contains(box) for b in blockers):
            return "OUT"
        legal = box.overlap(window) == 0  # excluded window faces have zero source volume
        for b in blockers:
            gaps = [max(Q(0), a-d, c-h) for a,h,c,d in zip(box.lo, box.hi, b.lo, b.hi)]
            legal &= sum(g*g for g in gaps) > Q(1, 2**52)
        vertices = tuple(box.vertices())
        all_proved = legal
        for y, shadows in constraints:
            blocked, visible = False, True
            for rows in shadows:
                values = [tuple(affine(row, v) for v in vertices) for row in rows]
                blocked |= all(max(col) <= 0 for col in values)
                visible &= any(min(col) > 0 for col in values)
            if (y == 1 and blocked) or (y == 0 and visible):
                return "OUT"
            all_proved &= visible if y == 1 else blocked
        return "IN" if all_proved else "R"
    return classify
