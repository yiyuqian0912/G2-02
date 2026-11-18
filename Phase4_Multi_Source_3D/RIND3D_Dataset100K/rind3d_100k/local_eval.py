"""Exact local-only window evaluation (no global response fields).

Semantics: identical to the 1K physical stage (`rind3d_1k.physical.evaluate`). It uses rind3d closed-solid /
closed-segment visibility (POLICY), receivers at voxel centres (i + 1/2)/N, and SOLID receivers unlit. A proposal W
of side L with origin o is accepted iff
    A(W) = [no source in the closed box] and [FREE(W) > 0] and [m(W) >= 1]   and   U < c(m(W), L),
with m(W) = #{j : channel j has a lit FREE and a dark FREE receiver in W}.

How the exact facts are obtained (this changes cost, never a decision):
* **Exact probes.** `CPUSource.native.points` evaluates a receiver with the certified CPU predicate. A lit FREE probe
  plus a dark FREE probe of channel j PROVE I_j = 1; any FREE probe proves FREE(W) > 0.
* **Lazy decision.** With m_lo channels proven informative and u channels unresolved, m lies in [m_lo, m_lo + u].
  - If U < c(m) for every m in that range (and m_lo >= 1, FREE > 0), the proposal is accepted.
  - If U >= c(m) for every m >= 1 in the range, it is rejected.
  Both outcomes are the exact rule's outcome.
* **Exact rendering.** Otherwise the unresolved channels (and SOLID, if FREE > 0 is not yet proven) are rendered
  exactly with the certified CUDA tile renderer (64^3 tiles; directed-rounding intervals plus exact CPU recomputation
  of every UNKNOWN voxel). This gives exact I_j and exact m.
* **Accepted windows** are always rendered completely for every source: these are the stored responses. Their exact m
  is asserted to reproduce the decision.

**Contract v3 (`rule="direct_interface"`, module `interface`).** A channel is informative iff the window has two
6-adjacent FREE receivers with different responses (a direct response interface). The same lazy decision is used
with exact facts of that rule: a constructive interface proof by bisection along a lattice path between an active and
an inactive FREE probe (exact point queries), the all-active hull certificate (I_j = 0), and an exact tile scan that
includes the edges crossing tile faces (`interface.TileScan`). Probes that only show both responses no longer prove
I_j = 1. The v1/v2 path (`rule="mixed"`) is unchanged.

Array layout: bool [Z, Y, X] crops, the same as `RINDPack.read_window` and the 1K public store."""
import time

import numpy as np

from .interface import TileScan, interface_axes, lattice_path, bisect_interface

TILE = 64


class LocalScene:
    """GPU-resident geometry plus one bound source per physical source. Owns no global field."""

    def __init__(self, factory, bundle, sources, policy, N=1024, backend="cuda_certified", batch=32):
        self.N = int(N); self.k = len(sources); self.batch = int(batch)
        self.src = [factory.source(bundle, s, backend, policy) for s in sources]
        self.cpu = [getattr(s, "cpu", s) for s in self.src]          # certified CPU binding (exact point queries)
        self.stats = dict(probes=0, probe_s=0.0, tiles=0, render_s=0.0, rendered_channels=0, decided_by_probes=0, decided_by_render=0, contains_source=0)
        self._solid_cache = {}
        self.hull = None

    def enable_hull(self, context):
        self.hull = HullCert(context, self.N)

    def close(self):
        for s in self.src:
            s.close()

    # ---------------------------------------------------------------- exact receiver probes
    def probe(self, j, vox):
        """Exact (lit, solid) at integer voxel indices vox (n, 3) in (x, y, z) order."""
        t = time.perf_counter()
        pts = (np.asarray(vox, np.float64) + 0.5) / self.N
        yy, mm, _ = self.cpu[j].native.points(np.ascontiguousarray(pts, dtype="<f8"), False)
        self.stats["probes"] += len(vox); self.stats["probe_s"] += time.perf_counter() - t
        if np.any(yy < 0) or np.any(mm < 0):
            raise RuntimeError("exact point query did not certify")
        return yy.astype(bool), mm.astype(bool)

    # ---------------------------------------------------------------- exact window rendering
    def tile_origins(self, o, L, order="spread"):
        """64^3 tile offsets of the window; "spread" puts the 8 even-indexed tiles first (they span the window)."""
        o = np.asarray(o, np.int64); n = max(L // TILE, 1)
        offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        if order == "spread" and n > 2:
            offs.sort(key=lambda t: (t[0] % 2) + (t[1] % 2) + (t[2] % 2) > 0)
        return offs, np.array([o + np.array(t) * TILE for t in offs], np.int32)

    def informative(self, j, o, L, need_free=False, first=8):
        """Exact I_j for the window (lit FREE and dark FREE both present), rendering 64^3 tiles progressively and
        stopping as soon as both classes (and a FREE receiver, if need_free) are seen. Returns (I_j, free_seen,
        complete, counts or None); counts = (lit, dark, solid) when every tile was rendered."""
        t = time.perf_counter()
        offs, origins = self.tile_origins(o, L)
        nl = nd = ns = 0; done = 0; nt = len(origins); per = TILE ** 3
        step = min(first, self.batch)
        while done < nt:
            chunk = origins[done:done + step]
            out = self.src[j].packed_batch(chunk, TILE, self.N)
            for pl, ps in out:
                a = np.bitwise_count(np.frombuffer(pl, np.uint8)).sum(dtype=np.int64)
                b = np.bitwise_count(np.frombuffer(ps, np.uint8)).sum(dtype=np.int64)
                nl += int(a); ns += int(b); nd += per - int(a) - int(b)
            done += len(chunk); step = self.batch
            self.stats["tiles"] += len(chunk)
            if nl > 0 and nd > 0:
                break
        self.stats["render_s"] += time.perf_counter() - t; self.stats["rendered_channels"] += 1
        complete = done == nt
        return (nl > 0 and nd > 0), (nl + nd) > 0, complete, ((nl, nd, ns) if complete else None)

    def interface(self, j, o, L, first=8):
        """Exact direct-interface indicator of channel j (contract v3), rendering 64^3 tiles progressively and stopping
        at the first interface edge (inside a tile or across a tile face). Returns (I_j, free_seen, complete, counts or
        None, mixed or None); counts = (lit, dark, solid) and mixed = the v1/v2 rule, both exact when complete."""
        t = time.perf_counter()
        offs, origins = self.tile_origins(o, L)
        scan = TileScan(L); done = 0; nt = len(origins)
        step = min(first, self.batch)
        while done < nt and not scan.found:
            chunk = origins[done:done + step]
            out = self.src[j].packed_batch(chunk, TILE, self.N)
            for off, (pl, ps) in zip(offs[done:done + len(chunk)], out):
                tl = np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").astype(bool).reshape(TILE, TILE, TILE)
                ts = np.unpackbits(np.frombuffer(ps, np.uint8), bitorder="little").astype(bool).reshape(TILE, TILE, TILE)
                scan.add(off, tl, ts)
            done += len(chunk); step = self.batch
            self.stats["tiles"] += len(chunk)
        self.stats["render_s"] += time.perf_counter() - t; self.stats["rendered_channels"] += 1
        complete = scan.complete
        mixed = True if scan.found else scan.mixed
        return scan.found, (scan.nl + scan.nd) > 0, complete, ((scan.nl, scan.nd, scan.ns) if complete else None), mixed

    def render(self, j, o, L):
        """Exact lit and solid crops bool [L, L, L] (Z, Y, X) of channel j for the window with origin o (x, y, z)."""
        t = time.perf_counter()
        o = np.asarray(o, np.int64)
        n = L // TILE if L >= TILE else 1
        side = TILE if L >= TILE else L
        offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        origins = np.array([o + np.array([a, b, c]) * side for (a, b, c) in offs], np.int32)
        out = []
        for i in range(0, len(origins), self.batch):
            out += self.src[j].packed_batch(origins[i:i + self.batch], side, self.N)
        lit = np.empty((L, L, L), bool); sol = np.empty((L, L, L), bool)
        for (a, b, c), (pl, ps) in zip(offs, out):
            tl = np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").astype(bool).reshape(64, 64, 64)
            ts = np.unpackbits(np.frombuffer(ps, np.uint8), bitorder="little").astype(bool).reshape(64, 64, 64)
            sl = (slice(c * side, (c + 1) * side), slice(b * side, (b + 1) * side), slice(a * side, (a + 1) * side))
            lit[sl] = tl[:side, :side, :side]; sol[sl] = ts[:side, :side, :side]
        self.stats["tiles"] += len(origins); self.stats["render_s"] += time.perf_counter() - t
        self.stats["rendered_channels"] += 1
        return lit, sol


def contains_any(sources, o, L, N):
    lo = np.asarray(o, dtype=np.float64) / N
    hi = (np.asarray(o, dtype=np.float64) + L) / N
    return bool(np.any(np.all((sources >= lo) & (sources <= hi), axis=1)))


class HullCert:
    """Exact all-lit certificate. If for every body C some face row (a, d) of C (a.x + d <= 0 inside) satisfies
    a.s + d > 0 and a.r + d > 0 at the 8 corners r of the box of the window's receiver centres, then the convex hull
    of {s} and every receiver lies strictly outside every closed body. Every closed segment [s, x] is then FREE of
    solid: every receiver is lit and FREE. The witness is searched in float and re-verified in exact rational
    arithmetic with the canonical rows; a failed search only means "not certified", which is then decided by rendering."""

    def __init__(self, context, N):
        from fractions import Fraction
        self.F = Fraction; self.N = int(N)
        self.rows = [[tuple(Fraction(v) for v in r) for r in b] for b in context.bodies]
        self.frows = [np.array([[float(v) for v in r] for r in b]) for b in self.rows]
        self.calls = 0; self.certified = 0

    def all_lit(self, s, o, L):
        self.calls += 1
        F = self.F; N = self.N
        lo = [F(2 * int(v) + 1, 2 * N) for v in o]; hi = [F(2 * (int(v) + L) - 1, 2 * N) for v in o]
        flo = np.array([float(v) for v in lo]); fhi = np.array([float(v) for v in hi]); fs = np.asarray(s, float)
        sx = [F(float(v)) for v in s]
        for rows, fr in zip(self.rows, self.frows):
            a = fr[:, :3]; d = fr[:, 3]
            vs = a @ fs + d
            mn = d + np.sum(np.where(a >= 0, a * flo, a * fhi), axis=1)       # min of a.r + d over the receiver box
            cand = np.flatnonzero((vs > 1e-12) & (mn > 1e-12))
            ok = False
            for i in cand[np.argsort(-np.minimum(vs[cand], mn[cand]))][:4]:
                ar = rows[int(i)]
                if ar[0] * sx[0] + ar[1] * sx[1] + ar[2] * sx[2] + ar[3] <= 0:
                    continue
                m = ar[3] + sum(ar[t] * (lo[t] if ar[t] >= 0 else hi[t]) for t in range(3))   # exact box minimum
                if m > 0:
                    ok = True
                    break
            if not ok:
                return False
        self.certified += 1
        return True


class Decision:
    __slots__ = ("A", "accept", "m", "m_lo", "m_hi", "mask", "counts", "why", "exact", "crops", "mixed_mask", "m_mixed")

    def __init__(self):
        self.A = 0; self.accept = False; self.m = None; self.m_lo = 0; self.m_hi = 0; self.mask = None
        self.counts = None; self.why = None; self.exact = False; self.crops = None
        self.mixed_mask = None; self.m_mixed = None        # v3 only: the v1/v2 rule on the same proposal (when exact)


def evaluate_proposal(scene, sources, o, L, U, c_row, rng, n_probe=48, need_exact=False, rule="mixed"):
    """Exact acceptance decision for one proposal (see module docstring). c_row[m-1] = c(m, L, k).
    need_exact=True forces exact m (every channel not proven informative by probes is decided by rendering).
    rule: "mixed" (contracts v1/v2) or "direct_interface" (contract v3).
    Returns a Decision; decisions equal the exact rule by construction."""
    if rule == "direct_interface":
        return evaluate_proposal_interface(scene, sources, o, L, U, c_row, rng, n_probe, need_exact)
    if rule != "mixed":
        raise ValueError(f"unknown eligibility rule {rule!r}")
    N = scene.N; k = scene.k
    d = Decision()
    if contains_any(sources, o, L, N):
        d.why = "contains_source"; d.exact = True; d.m = 0; scene.stats["contains_source"] += 1
        return d
    o = np.asarray(o, np.int64)
    inf = np.zeros(k, bool); seen_lit = np.zeros(k, bool); seen_dark = np.zeros(k, bool)
    free_known = False
    if n_probe > 0:
        vox = o + rng.integers(0, L, (n_probe, 3))
        for j in range(k):
            lit, sol = scene.probe(j, vox)
            fr = ~sol
            if fr.any():
                free_known = True
            seen_lit[j] = np.any(lit & fr); seen_dark[j] = np.any(~lit & fr)
        inf = seen_lit & seen_dark
    hull = getattr(scene, "hull", None)
    unres = [j for j in range(k) if not inf[j]]
    if hull is not None and unres:                     # exact all-lit certificates: I_j = 0 and FREE > 0, no rendering
        keep = []
        for j in unres:
            if hull.all_lit(sources[j], o, L):
                free_known = True
            else:
                keep.append(j)
        n_cert0 = len(unres) - len(keep)
        unres = keep
    d.m_lo = int(inf.sum()); d.m_hi = d.m_lo + len(unres)
    rendered = False
    while True:
        m_lo = int(inf.sum()); m_hi = m_lo + len(unres)
        if not need_exact:
            cs = {m: c_row[m - 1] for m in range(max(m_lo, 1), m_hi + 1)}
            if free_known and m_lo >= 1 and U < min(cs.values()):
                d.A = 1; d.accept = True; d.why = "A"; d.m_lo, d.m_hi = m_lo, m_hi
                scene.stats["decided_by_render" if rendered else "decided_by_probes"] += 1
                return d                   # accepted for every possible m; the caller renders all channels
            if not cs or U >= max(cs.values()):
                d.A = int(free_known and m_lo >= 1); d.m_lo, d.m_hi = m_lo, m_hi
                d.why = "thinned" if d.A else "thinned_or_uninformative"
                scene.stats["decided_by_render" if rendered else "decided_by_probes"] += 1
                return d                   # rejected for every possible m (m = 0 or U >= c(m))
        if not unres:
            break
        j = unres.pop(0)
        I, fseen, complete, cnt = scene.informative(j, o, L)
        rendered = True
        inf[j] = I
        if fseen:
            free_known = True
        elif complete:                     # no FREE receiver at all: SOLID is geometry-only, same for every channel
            d.why = "no_free"; d.exact = True; d.m = 0; scene.stats["decided_by_render"] += 1
            return d
    if not free_known:                     # every channel certified/proven without a FREE witness: render SOLID
        I, fseen, complete, cnt = scene.informative(0, o, L)
        if not fseen:
            d.why = "no_free"; d.exact = True; d.m = 0; scene.stats["decided_by_render"] += 1
            return d
    m = int(inf.sum())
    scene.stats["decided_by_render"] += 1
    d.m = m; d.exact = True; d.m_lo = d.m_hi = m
    d.mask = int(sum(1 << j for j in range(k) if inf[j]))
    if m == 0:
        d.why = "uninformative"
        return d
    d.A = 1
    if U < c_row[m - 1]:
        d.accept = True; d.why = "A"
    else:
        d.why = "thinned"
    return d


def evaluate_proposal_interface(scene, sources, o, L, U, c_row, rng, n_probe=48, need_exact=False, n_paths=3):
    """Contract v3 decision: A(W) = [no source in the closed box] and [FREE(W) > 0] and [m(W) >= 1] with
    m(W) = #channels with a direct FREE-FREE response interface; accepted iff A(W) and U < c(m). Same lazy scheme as
    the v2 rule, with v3 facts: bisection proofs (I_j = 1), hull all-active certificates (I_j = 0), exact tile scans.
    With need_exact the v1/v2 'mixed' rule of the same proposal is also exact (Decision.mixed_mask / m_mixed)."""
    N = scene.N; k = scene.k
    d = Decision()
    if contains_any(sources, o, L, N):
        d.why = "contains_source"; d.exact = True; d.m = 0; scene.stats["contains_source"] += 1
        return d
    o = np.asarray(o, np.int64)
    inf = np.zeros(k, bool); mixed = [None] * k
    free_known = False
    st = scene.stats
    st.setdefault("bisect_q", 0); st.setdefault("bisect_ok", 0); st.setdefault("bisect_fail", 0)
    if n_probe > 0:
        vox = o + rng.integers(0, L, (n_probe, 3))
        for j in range(k):
            lit, sol = scene.probe(j, vox)
            fr = ~sol
            if fr.any():
                free_known = True
            la = np.flatnonzero(lit & fr); da = np.flatnonzero(~lit & fr)
            if len(la) and len(da):
                mixed[j] = True
                q = lambda v, j=j: scene.probe(j, v)
                for t_ in range(min(n_paths, len(la))):
                    a = vox[la[t_]]; dist = np.abs(vox[da] - a).sum(1); b = vox[da[int(np.argmin(dist))]]
                    order = ((0, 1, 2), (2, 1, 0), (1, 2, 0))[t_ % 3]
                    pair, nq = bisect_interface(q, lattice_path(a, b, order), True)
                    st["bisect_q"] += nq
                    if pair is not None:
                        inf[j] = True; st["bisect_ok"] += 1
                        break
                else:
                    st["bisect_fail"] += 1
    hull = getattr(scene, "hull", None)
    unres = [j for j in range(k) if not inf[j]]
    if hull is not None and unres:
        keep = []
        for j in unres:
            if hull.all_lit(sources[j], o, L):
                free_known = True; mixed[j] = False
            else:
                keep.append(j)
        unres = keep
    d.m_lo = int(inf.sum()); d.m_hi = d.m_lo + len(unres)
    rendered = False
    while True:
        m_lo = int(inf.sum()); m_hi = m_lo + len(unres)
        if not need_exact:
            cs = {m: c_row[m - 1] for m in range(max(m_lo, 1), m_hi + 1)}
            if free_known and m_lo >= 1 and U < min(cs.values()):
                d.A = 1; d.accept = True; d.why = "A"; d.m_lo, d.m_hi = m_lo, m_hi
                scene.stats["decided_by_render" if rendered else "decided_by_probes"] += 1
                return d
            if not cs or U >= max(cs.values()):
                d.A = int(free_known and m_lo >= 1); d.m_lo, d.m_hi = m_lo, m_hi
                d.why = "thinned" if d.A else "thinned_or_uninformative"
                scene.stats["decided_by_render" if rendered else "decided_by_probes"] += 1
                return d
        if not unres:
            break
        j = unres.pop(0)
        I, fseen, complete, cnt, mx = scene.interface(j, o, L)
        rendered = True
        inf[j] = I
        if mx is not None:
            mixed[j] = mx
        if fseen:
            free_known = True
        elif complete:
            d.why = "no_free"; d.exact = True; d.m = 0; scene.stats["decided_by_render"] += 1
            return d
    if not free_known:
        I, fseen, complete, cnt, mx = scene.interface(0, o, L)
        if not fseen:
            d.why = "no_free"; d.exact = True; d.m = 0; scene.stats["decided_by_render"] += 1
            return d
    m = int(inf.sum())
    scene.stats["decided_by_render"] += 1
    d.m = m; d.exact = True; d.m_lo = d.m_hi = m
    d.mask = int(sum(1 << j for j in range(k) if inf[j]))
    for j in range(k):                     # an interface implies mixed; resolved channels are otherwise exact
        if inf[j]:
            mixed[j] = True
    if all(x is not None for x in mixed):
        d.mixed_mask = int(sum(1 << j for j in range(k) if mixed[j])); d.m_mixed = int(sum(bool(x) for x in mixed))
    if m == 0:
        d.why = "no_interface" if any(bool(x) for x in mixed) else "uninformative"
        return d
    d.A = 1
    if U < c_row[m - 1]:
        d.accept = True; d.why = "A"
    else:
        d.why = "thinned"
    return d


def render_accepted(scene, o, L, rule="mixed"):
    """All channels of an accepted window, exactly: (lit crops list, solid crop, counts (k,3), m, mask)."""
    k = scene.k
    lits = []; sol_ref = None; counts = np.zeros((k, 3), np.int64)
    for j in range(k):
        lit, sol = scene.render(j, o, L)
        if sol_ref is None:
            sol_ref = sol
        elif not np.array_equal(sol, sol_ref):
            raise AssertionError("SOLID differs between channels")
        if np.any(lit & sol):
            raise AssertionError("solid receiver lit")
        fr = ~sol
        counts[j] = (int(lit.sum()), int((fr & ~lit).sum()), int(sol.sum()))
        lits.append(lit)
    inf = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    if rule == "mixed":
        return lits, sol_ref, counts, int(inf.sum()), int(sum(1 << j for j in range(k) if inf[j]))
    if rule != "direct_interface":
        raise ValueError(f"unknown eligibility rule {rule!r}")
    edges = [interface_axes(lit, sol_ref) for lit in lits]          # per channel: interface edges along Z, Y, X
    itf = np.array([sum(e) > 0 for e in edges])
    extra = dict(mixed_mask=int(sum(1 << j for j in range(k) if inf[j])), interface_edges=edges)
    return lits, sol_ref, counts, int(itf.sum()), int(sum(1 << j for j in range(k) if itf[j])), extra
