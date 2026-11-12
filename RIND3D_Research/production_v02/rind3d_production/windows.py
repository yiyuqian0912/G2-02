"""Observation-window policy (contract `obs-policy-v0.2`): natural base distribution + declared subsets.

Every window is an index record (origin, side) into the stored complete 1024^3 fields; responses are never
duplicated. Subsets, all with recorded proposal counts so inclusion probabilities can be reconstructed:
- natural:      origins i.i.d. uniform on {0..1024-L}^3, no acceptance filter of any kind (all-lit, all-shadow,
                SOLID-only and source-containing windows are kept and flagged);
- informative:  uniform proposals accepted iff the response is non-constant over FREE receivers and no source
                lies in the closed window box (the v0.1 informativeness rule without its lit-ratio targets);
- extreme:      informative proposals with minority fraction min(lit,shadow)/FREE <= EXTREME (per source for
                single-source; for multi-source the minority of the source with the smallest minority);
- rare:         informative proposals whose minority class has <= RARE receivers (thin / rare boundaries);
- nonlocal:     informative proposals where >= NONLOCAL of up to NL_SAMPLES sampled shadow-FREE receivers have
                their first blocker (from the receiver towards the source) outside the window box.
Subset proposal streams are independent and seeded per (seed, slot, L, subset). Counts are exact (CountIndex
over the certified fields). The difficult subsets are bounded (proposal caps); shortfalls are recorded, never
filled with non-qualifying windows.
"""
import numpy as np

from rind3d.geometry import keyed_rng

SIDES = (64, 128, 256)
QUOTA = {"natural": 16, "informative": 8, "extreme": 4, "rare": 2, "nonlocal": 4}
CAPS = {"informative": 2048, "extreme": 4096, "rare": 4096, "nonlocal": 96}
EXTREME, RARE, NONLOCAL, NL_SAMPLES = 0.01, 64, 0.9, 32
STREAM = {"natural": 3001, "informative": 3011, "extreme": 3021, "rare": 3031, "nonlocal": 3041}
DTYPE = np.dtype([("scene", "<u4"), ("window", "<u4"), ("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("L", "<u2"),
                  ("subset", "<u1"), ("free", "<u4"), ("solid", "<u4"), ("lit", "<u4", (4,)),
                  ("contains_source", "?"), ("minority_fraction", "<f4"), ("nonlocal_fraction", "<f4"),
                  ("proposal_index", "<u4")])
SUBSET_CODES = {"natural": 0, "informative": 1, "extreme": 2, "rare": 3, "nonlocal": 4}


def contains_source(origin, L, sources, n=1024):
    o = np.asarray(origin, np.float64)/n
    return bool(np.any(np.all((sources >= o) & (sources <= o + L/n), axis=1)))


def window_counts(counts, origin, L):
    """Exact per-source lit counts; FREE and SOLID from the first index (SOLID is source independent)."""
    c0 = counts[0].query(origin, L)
    lit = [c0.lit] + [c.query(origin, L).lit for c in counts[1:]]
    return c0.free, c0.solid, lit


def minority(free, lit):
    if not free:
        return 0.0, 0
    m = [min(l, free-l) for l in lit]
    mixed = [x for x in m if x > 0]
    if not mixed:
        return 0.0, 0
    k = int(np.argmin(mixed))
    return float(mixed[k]/free), int(mixed[k])


def nonlocal_fraction(context, sources, lit_window_fn, origin, L, rng, n=1024):
    """Fraction of sampled shadow-FREE receivers (source 0) whose first blocker towards the source is outside."""
    resp, solid = lit_window_fn(origin, L)
    sh = np.flatnonzero((~resp & ~solid).ravel())
    if not len(sh):
        return None
    take = rng.choice(sh, min(NL_SAMPLES, len(sh)), replace=False)
    z, y, x = np.unravel_index(take, (L, L, L))
    pts = np.column_stack([(origin[0]+x+.5)/n, (origin[1]+y+.5)/n, (origin[2]+z+.5)/n])
    lo, hi = np.asarray(origin)/n, (np.asarray(origin)+L)/n
    s = sources[0]
    outside = 0
    for p in pts:
        hits, entry = context.bind(p).native.first_hits(np.ascontiguousarray(s[None]))
        if hits[0] < 0:
            continue                                   # not blocked: cannot be a shadow receiver of this source
        q = p + float(entry[0])*(s - p)
        outside += int(np.any(q < lo) or np.any(q > hi))
    return outside/len(pts)


def select(seed, slot, sources, counts, context, lit_window_fn):
    """All window records of one scene plus a sampling report (proposals, acceptances, shortfalls)."""
    recs, report = [], {}
    K = len(sources)
    for L in SIDES:
        for subset in ("natural", "informative", "extreme", "rare", "nonlocal"):
            rng = keyed_rng(seed, slot, STREAM[subset]*8 + SIDES.index(L))
            quota = QUOTA[subset]
            cap = CAPS.get(subset, quota)
            got, proposals = 0, 0
            nl_rng = np.random.default_rng(np.random.SeedSequence([seed, slot, L, 77]))
            while got < quota and proposals < cap:
                proposals += 1
                o = tuple(int(v) for v in rng.integers(0, 1024-L+1, 3))
                free, solid, lit = window_counts(counts, o, L)
                cs = contains_source(o, L, sources)
                mf, mc = minority(free, lit)
                nl = np.nan
                ok = True
                if subset != "natural":
                    ok = mf > 0 and not cs
                    if ok and subset == "extreme":
                        ok = mf <= EXTREME
                    elif ok and subset == "rare":
                        ok = mc <= RARE
                    elif ok and subset == "nonlocal":
                        v = nonlocal_fraction(context, sources, lit_window_fn, o, L, nl_rng)
                        nl = np.nan if v is None else v
                        ok = v is not None and v >= NONLOCAL
                if not ok:
                    continue
                r = np.zeros((), DTYPE)
                r["x"], r["y"], r["z"], r["L"] = o[0], o[1], o[2], L
                r["subset"], r["free"], r["solid"] = SUBSET_CODES[subset], free, solid
                lit4 = np.zeros(4, np.uint32)
                lit4[:K] = lit
                r["lit"] = lit4
                r["contains_source"], r["minority_fraction"], r["nonlocal_fraction"] = cs, mf, nl
                r["proposal_index"] = proposals
                recs.append(r)
                got += 1
            report[f"{subset}/{L}"] = {"quota": quota, "accepted": got, "proposals": proposals,
                                       "shortfall": quota-got, "cap": cap}
    arr = np.array(recs, dtype=DTYPE)
    arr["window"] = np.arange(len(arr))
    return arr, report


def flags(rec):
    """Derived natural-distribution categories for reporting (single-source semantics on source 0)."""
    free, lit = int(rec["free"]), int(rec["lit"][0])
    if free == 0:
        return "no_free"
    if lit == free:
        return "all_lit"
    if lit == 0:
        return "all_shadow"
    return "mixed"
