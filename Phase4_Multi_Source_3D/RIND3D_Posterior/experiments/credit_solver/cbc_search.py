"""Untrusted search for the CBC kernel v2. It only proposes; the kernel decides.

Policy (all choices affect speed or success, never soundness):
  * condition on spread interface pairs (Astra's maximin order), exclude the window;
  * always work on the largest open piece (it dominates the unknown mass);
  * screen it (inherited obligations); if ambiguous receivers remain:
      - many ambiguous receivers and the piece touches an obstacle, or is very
        elongated: split it geometrically by a dyadic axis plane (apex isolation /
        long-axis bisection);
      - otherwise enforce a batch of ambiguous receivers exactly, wrong-at-anchor and
        interface receivers first;
  * stop when the exact bound on the decoded payload meets the tolerance, when no
    open piece remains, or at the time budget.
"""
from fractions import Fraction
from hashlib import sha256
import heapq
from time import perf_counter

import numpy as np

from cbc_kernel import certify
from solver import Limit, choose_pairs


def dyadic_between(lo, hi, target):
    """A dyadic rational strictly inside (lo, hi), close to target (all Fractions)."""
    for m in range(4, 200, 4):
        s = 1 << m
        c = Fraction(int(target * s), s)
        if lo < c < hi:
            return c
    return None


def axis_plane(k, c):
    row = [0, 0, 0, -c.numerator]
    row[k] = c.denominator
    return tuple(row)


def extents(poly):
    vs = [tuple(Fraction(v[i], v[3]) for i in range(3)) for v in poly.vertices]
    lo = [min(v[i] for v in vs) for i in range(3)]
    hi = [max(v[i] for v in vs) for i in range(3)]
    return vs, lo, hi


def facet_contact_plane(poly, bodies):
    """If the piece lies outside one obstacle facet and touches its plane, a parallel plane at
    half the piece's height above it (dyadic offset). Proposal only."""
    best = None
    for b in bodies:
        for r in b:
            vals = [Fraction(r[0] * v[0] + r[1] * v[1] + r[2] * v[2] + r[3] * v[3], v[3]) for v in poly.vertices]
            lo_v, hi_v = min(vals), max(vals)
            if lo_v == 0 and hi_v > 0 and (best is None or hi_v > best[1]):
                best = (r, hi_v)
    if best is None:
        return None
    r, hmax = best
    tau = dyadic_between(Fraction(0), hmax, hmax / 2)
    if tau is None:
        return None
    return (r[0] * tau.denominator, r[1] * tau.denominator, r[2] * tau.denominator,
            r[3] * tau.denominator - tau.numerator)


def propose_split(kernel, h, bodies):
    """Facet-parallel split for face contact, apex isolation for vertex contact, else long-axis bisection."""
    poly = kernel.poly(h)
    plane = facet_contact_plane(poly, bodies)
    if plane is not None:
        return plane
    vs, lo, hi = extents(poly)
    contact = [v for v, hv in zip(vs, poly.vertices)
               if any(all(r[0] * hv[0] + r[1] * hv[1] + r[2] * hv[2] + r[3] * hv[3] <= 0 for r in b) for b in bodies)]
    k = max(range(3), key=lambda i: hi[i] - lo[i])
    if contact:
        v = contact[0]
        # move half way from the contact vertex toward the far end on the longest axis
        far = hi[k] if v[k] - lo[k] < hi[k] - v[k] else lo[k]
        target = v[k] + (far - v[k]) / 2
    else:
        target = (lo[k] + hi[k]) / 2
    c = dyadic_between(lo[k], hi[k], target)
    return None if c is None else axis_plane(k, c)


def hull_receivers(kernel, h, obs, r, minimum):
    """Convex-hull extreme receivers of the single-blocker dark ambiguous sets (proposal)."""
    from cbc_kernel import encoded_subset
    from hull_proposals import from_list, from_scan, scan_extremes
    L = obs.side
    if r["ids"] is not None and r.get("masks") is not None:
        dark = obs.scancls.ravel()[r["ids"]] == 0
        per_body = from_list(L, r["ids"], r["masks"], dark, minimum)
    else:
        rows, offs = encoded_subset(obs, kernel.poly(h), range(len(obs.bodies)))
        lo, hi, count = scan_extremes(obs.scancls, np.asarray(obs.origin, np.int64), obs.denom, rows, offs,
                                      np.zeros(0, np.int64))
        per_body = from_scan(L, lo, hi, count, minimum)
    if not per_body:
        return []
    return [int(i) for i in np.unique(np.concatenate(list(per_body.values())))]


def elongation(poly):
    _, lo, hi = extents(poly)
    ext = sorted(float(hi[i] - lo[i]) for i in range(3))
    vol = float(poly.volume)
    return ext[2] ** 3 / max(vol, 1e-300)


def solve(obs, key, tolerance, seconds, cap=4096, **policy):
    """Label one channel: the trusted driver cbc_kernel.certify runs this search policy."""
    return certify(obs, tolerance, seconds, cap, lambda kernel, budget: search(kernel, budget, obs, key, tolerance,
                                                                                 **policy))


def search(kernel, budget, obs, key, tolerance, pairs=24, strategy="interface",
           batch=96, split_above=1500, elongated=1e4, max_splits=4000, always_split=True, negligible=Fraction(1, 2000),
           shrink=0.8, prefer_single_dark=True, centroid_anchor=True, park_share=0.25, hull=True, hull_min=512, frag_cap=16):
    """Untrusted policy. Returns diagnostics only; the status comes from kernel.release()."""
    if pairs < 0:
        raise ValueError("invalid pair count")
    obs.centroid_anchor = centroid_anchor  # anchor hint for the screens' `wrong` flags
    timings = {}
    parked = set()
    hull_spent = set()  # handles where the hull step found nothing or changed nothing
    hull_from = {}  # child handle -> ambiguous count of the parent's hull step
    reason = "unresolved_frontier"
    trace = []

    def checkpoint():
        t = perf_counter()
        res = kernel.snapshot()
        timings["bound_seconds"] = timings.get("bound_seconds", 0) + perf_counter() - t
        return res

    try:
        t0 = perf_counter()
        seed = int.from_bytes(sha256(key.encode()).digest()[:4], "big")
        for dark, lit in (choose_pairs(obs.cls, pairs, seed, strategy) if pairs else []):
            kernel.condition_pair(dark, lit)
        timings["discovery_seconds"] = perf_counter() - t0
        kernel.exclude_window()
        splits = 0
        heap = []  # (-volume, handle); stale entries are skipped on pop
        parked_volume = [0.0]

        def push(handles):
            for x in handles:
                if kernel.is_open(x):
                    heapq.heappush(heap, (-float(kernel.volume(x)), x))
        push(kernel.open_handles())
        while True:
            budget.check()
            h = None
            while heap:
                _, x = heapq.heappop(heap)
                if kernel.is_open(x) and x not in parked:
                    h = x
                    break
            if h is None:
                break
            vol_h = float(kernel.volume(h))
            if kernel.positive and vol_h < float(negligible) * kernel.lower_total_float() and \
                    parked_volume[0] + vol_h <= park_share * float(tolerance) * kernel.lower_total_float():
                parked.add(h)  # charged as unknown; small enough, within the parking budget
                parked_volume[0] += vol_h
                continue
            inherited = kernel.pending_count(h)
            r = kernel.screen(h)
            if r["outcome"] in ("out", "in", "shell"):
                if r["outcome"] == "in" and kernel.positive:
                    lo_v, unk_v = kernel.lower_total_float(), kernel.unknown_volume()
                    if unk_v > 2 * float(tolerance) * (lo_v + unk_v):
                        continue  # hint: the bound cannot be near the target yet; skip the snapshot
                    payload, bound = checkpoint()
                    trace.append([round(perf_counter() - t0, 3), float(bound)])
                    if bound <= tolerance:
                        reason = "tv_target"
                        break
                continue
            h, n = r["handle"], r["ambiguous"]
            if hull and h in hull_from and n > shrink * hull_from[h]:
                hull_spent.add(h)  # the last hull step did not shrink the obligations enough
            if hull and h not in hull_spent and n > hull_min:
                # receiver convexity: the extreme points of each single-blocker dark set settle
                # the whole set by exact (non-fragmenting) intersections
                chosen = hull_receivers(kernel, h, obs, r, hull_min)
                if chosen:
                    kids, same = kernel.enforce(h, chosen)
                    timings["hull_enforcements"] = timings.get("hull_enforcements", 0) + 1
                    timings["hull_receivers"] = timings.get("hull_receivers", 0) + len(chosen)
                    if same:
                        hull_spent.update(kids)
                    for x in kids:
                        hull_from[x] = n
                    push(kids)
                    continue
                hull_spent.add(h)
            if r["ids"] is None:
                # obligation list too large to store: split geometrically
                plane = propose_split(kernel, h, obs.bodies)
                kids = kernel.split(h, plane) if plane is not None and splits < max_splits else [h]
                if kids == [h]:
                    parked.add(h)
                else:
                    push(kids)
                splits += 1
                continue
            poly = kernel.poly(h)
            # Split geometrically only while splitting still shrinks the obligation set;
            # receivers whose class genuinely varies over the piece need exact cuts.
            progress = inherited is None or n < shrink * inherited
            contact = kernel.contact(h)
            if n > split_above and splits < max_splits and (contact or (progress and (
                    always_split or elongation(poly) > elongated))):
                plane = propose_split(kernel, h, obs.bodies)
                if plane is not None:
                    kids = kernel.split(h, plane)
                    if kids != [h]:
                        splits += 1
                        push(kids)
                        continue
            ids, wrong, bnd = r["ids"], r["wrong"], r["boundary"]
            score = wrong.astype(np.int64) * 2 + bnd.astype(np.int64)
            pure = None
            if prefer_single_dark and r.get("masks") is not None:
                m = r["masks"]
                single = (m != 0) & ((m & (m - np.uint64(1))) == 0)
                dark = obs.scancls.ravel()[ids] == 0
                # a dark receiver with one possible blocker is a pure intersection (no fragmentation)
                pure = single & dark
                score = score * 2 + pure.astype(np.int64)
            order = np.argsort(-score, kind="stable")
            k = n if n <= 2 * batch else batch
            if pure is not None and frag_cap is not None:
                # every other receiver can split each current part (unions of cones, or
                # subtractions): cap them per batch so fragmentation stays bounded
                frag = ~pure[order]
                keep = ~frag | (np.cumsum(frag) <= frag_cap)
                order = order[keep]
            chosen = [int(ids[i]) for i in order[:k]]
            kids, _ = kernel.enforce(h, chosen)
            push(kids)
    except Limit as exc:
        reason = str(exc)
    unk, lo = kernel.unknown_volume(), kernel.lower_total_float()
    return dict(search_reason=reason, unknown_volume=float(unk), lower_volume=float(lo), trace=trace[-20:], **timings)
