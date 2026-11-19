"""Exact structural certificate for poly.Poly (a trusted check; the producer is Poly.split).

Poly.volume (fan volume), the facet rows of the cell octree and Poly.split (edge walk) all
assume the face-structure invariant FS: the face cycles of p tile the boundary of
conv(vertices) exactly once, each as a convex polygon. check(p) verifies, with integer
arithmetic only:

  (a) every vertex is primitive homogeneous (x, y, z, w) with w > 0; each face has >= 3
      distinct vertices;
  (b) each face spans a plane; its primitive outward row is <= 0 at every vertex of p and
      = 0 at every vertex of the face (a supporting plane of conv(vertices));
  (c) each face cycle, oriented by that outward normal, is strictly convex: every other face
      vertex lies strictly left of every directed edge;
  (d) the oriented cycles use every directed edge exactly once and its reverse exactly once;
  (e) distinct faces lie on distinct planes;
  (f) the faces are connected through shared edges, and the faces around each vertex form a
      single cycle;
  (g) V - E + F = 2.

Lemma FS (XLF_CBC_IMPLEMENTATION.md, section 5): (a)-(f) imply that the union S of the faces
is a closed surface mapped homeomorphically onto the boundary of H = intersection of the
face rows, and H = conv(vertices). Hence the fan volume equals vol(conv V), the face rows cut
out exactly conv V, and every face edge is an edge of conv V. (g) is a redundant cross-check.

check() raises ArithmeticError on the first violated condition and returns the outward rows.
"""
from math import gcd

import numpy as np

from poly import form


def _sign_matrix(rows, p):
    """Exact signs of form(row, v) for every row and vertex of p.

    Binary64 filter with the same operations, order and margin as poly._fsign_terms
    (a*x + b*y + c*z, then + d; certain iff |v| > 1e-12 * sum|terms| with
    1e-280 <= sum|terms| < 1e300); rows that do not convert to finite floats and vertices
    whose coordinates are not normal binary64 numbers (poly.filter_vertices) are exact.
    Uncertain entries are evaluated exactly.
    """
    verts = p.vertices
    fv = p.filter_vertices
    F, V = len(rows), len(verts)
    R = np.zeros((F, 4))
    rok = np.ones(F, bool)
    for i, r in enumerate(rows):
        try:
            R[i] = [float(x) for x in r]
        except OverflowError:
            rok[i] = False
    X = np.zeros((V, 3))
    vok = np.ones(V, bool)
    for j, f in enumerate(fv):
        if f is None:
            vok[j] = False
        else:
            X[j] = f
    with np.errstate(over="ignore", invalid="ignore"):
        t0 = R[:, 0:1] * X[None, :, 0]
        t1 = R[:, 1:2] * X[None, :, 1]
        t2 = R[:, 2:3] * X[None, :, 2]
        v = ((t0 + t1) + t2) + R[:, 3:4]
        m = ((np.abs(t0) + np.abs(t1)) + np.abs(t2)) + np.abs(R[:, 3:4])
        ok = rok[:, None] & vok[None, :] & (m < 1e300) & (m >= 1e-280)
        sg = np.where(ok & (v > 1e-12 * m), 1, np.where(ok & (v < -1e-12 * m), -1, 0)).astype(np.int8)
        sure = ok & (np.abs(v) > 1e-12 * m)
    out = sg.tolist()
    for i, j in zip(*np.nonzero(~sure)):
        x = form(rows[i], verts[j])
        out[i][j] = (x > 0) - (x < 0)
    return out


VERIFIED = 0  # splits certified by verify_split in this process (test and audit instrument)


def _diff(b, a):
    """(b/wb - a/wa) scaled by wa*wb > 0 (exact, sign-preserving)."""
    return (b[0] * a[3] - a[0] * b[3], b[1] * a[3] - a[1] * b[3], b[2] * a[3] - a[2] * b[3])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _dot3(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _fail(msg):
    raise ArithmeticError("face structure: " + msg)


def check_reference(p):
    """Reference implementation (O(k^2) exact convexity tests); differential oracle for check()."""
    verts = p.vertices
    faces = p.faces
    if len(faces) < 4 or len(verts) < 4:
        _fail("fewer than four faces or vertices")
    for v in verts:
        if len(v) != 4 or v[3] <= 0 or gcd(*v) != 1:
            _fail("vertex not primitive with w > 0")
    rows = []
    oriented = []
    for face in faces:
        k = len(face)
        if k < 3 or len(set(face)) != k:
            _fail("degenerate face cycle")
        a = face[0]
        n = _cross(_diff(face[1], a), _diff(face[2], a))
        if not any(n):
            _fail("collinear leading triple (not strictly convex)")
        # plane through a with normal n, homogeneous row (n * wa, -n . a_xyz)
        row = (n[0] * a[3], n[1] * a[3], n[2] * a[3], -_dot3(n, a))
        g = gcd(*row)
        row = tuple(x // g for x in row)
        vals = [form(row, q) for q in verts]
        if max(vals) > 0:
            row = tuple(-x for x in row)
            vals = [-x for x in vals]
        if max(vals) > 0:
            _fail("face plane separates vertices (not a supporting plane)")
        if any(form(row, q) != 0 for q in face):
            _fail("face not planar")
        if all(x == 0 for x in vals):
            _fail("all vertices on one plane")
        normal = row[:3]  # outward: the row increases away from the polytope
        # orient counterclockwise seen from outside: interior to the left of each edge
        cyc = list(face)
        e0 = _diff(cyc[1], cyc[0])
        w0 = _diff(cyc[2], cyc[0])
        if _dot3(_cross(e0, w0), normal) < 0:
            cyc.reverse()
        for i in range(k):
            s, t = cyc[i], cyc[(i + 1) % k]
            e = _diff(t, s)
            for j in range(k):
                if j == i or j == (i + 1) % k:
                    continue
                if _dot3(_cross(e, _diff(cyc[j], s)), normal) <= 0:
                    _fail("face cycle not strictly convex")
        rows.append(row)
        oriented.append(cyc)
    if len(set(rows)) != len(rows):
        _fail("two faces on one plane")
    directed = {}
    for fi, cyc in enumerate(oriented):
        for i in range(len(cyc)):
            e = (cyc[i], cyc[(i + 1) % len(cyc)])
            if e in directed:
                _fail("directed edge used twice")
            directed[e] = fi
    for (s, t) in directed:
        if (t, s) not in directed:
            _fail("edge without its reverse (open surface)")
    # connectivity of faces through shared edges
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (s, t), fi in directed.items():
        a, b = find(fi), find(directed[(t, s)])
        if a != b:
            parent[a] = b
    if len({find(i) for i in range(len(faces))}) != 1:
        _fail("faces not connected")
    # single cycle of faces around each vertex (no pinch point)
    around = {}
    for (s, t), fi in directed.items():
        around.setdefault(s, []).append((t, fi))
    for v in verts:
        out = around.get(v)
        if not out:
            _fail("vertex on no edge")
        # oriented: face fi uses s->t; the next face around s (sharing edge s-t) uses t->s.
        # Walk fi -> face of the outgoing edge (s, t') that follows (t'', s) in the same face.
        nxt = {}
        for fi, cyc in enumerate(oriented):
            k = len(cyc)
            for i in range(k):
                if cyc[i] == v:
                    incoming = cyc[i - 1]  # edge (incoming, v) in face fi
                    nxt[fi] = directed[(v, incoming)]  # the face across that edge
        start = next(iter(nxt))
        seen, f = 1, nxt[start]
        while f != start:
            f = nxt[f]
            seen += 1
            if seen > len(nxt):
                _fail("vertex link is not a cycle")
        if seen != len(nxt):
            _fail("pinch point (several face cycles at one vertex)")
    E = len(directed) // 2
    if len(verts) - E + len(faces) != 2:
        _fail("Euler characteristic is not 2")
    return rows


def _topology(faces, verts, oriented, rows):
    """(d)-(g) on outward-oriented cycles: edge pairing, distinct planes, connectivity, links, Euler."""
    if len(set(rows)) != len(rows):
        _fail("two faces on one plane")
    directed = {}
    for fi, cyc in enumerate(oriented):
        k = len(cyc)
        for i in range(k):
            e = (cyc[i], cyc[(i + 1) % k])
            if e in directed:
                _fail("directed edge used twice")
            directed[e] = fi
    for (s_, t) in directed:
        if (t, s_) not in directed:
            _fail("edge without its reverse (open surface)")
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (s_, t), fi in directed.items():
        a, b = find(fi), find(directed[(t, s_)])
        if a != b:
            parent[a] = b
    if len({find(i) for i in range(len(faces))}) != 1:
        _fail("faces not connected")
    nxt = {}  # (vertex, face) -> next face around the vertex
    for fi, cyc in enumerate(oriented):
        k = len(cyc)
        for i in range(k):
            v = cyc[i]
            nxt[(v, fi)] = directed[(v, cyc[i - 1])]
    count = {}
    for (v, fi) in nxt:
        count[v] = count.get(v, 0) + 1
    if set(count) != set(verts):
        _fail("vertex on no face")
    seen = set()
    for (v, fi) in nxt:
        if v in seen:
            continue
        seen.add(v)
        n, f = 1, nxt[(v, fi)]
        while f != fi:
            f = nxt[(v, f)]
            n += 1
            if n > count[v]:
                _fail("vertex link is not a cycle")
        if n != count[v]:
            _fail("pinch point (several face cycles at one vertex)")
    E = len(directed) // 2
    if len(verts) - E + len(faces) != 2:
        _fail("Euler characteristic is not 2")


def _lex(a, b, i, j):
    """Exact lexicographic comparison of the (i, j) projections of homogeneous a, b: -1, 0, 1."""
    l, r = a[i] * b[3], b[i] * a[3]
    if l != r:
        return -1 if l < r else 1
    l, r = a[j] * b[3], b[j] * a[3]
    return (l > r) - (l < r)


def check(p):
    """FS certificate, same conditions as check_reference with cheaper exact tests.

    (b) uses Poly.side (proved binary64 filter, exact fallback) for the supporting test and
    exact integers for the on-plane test. (c) uses the injective projection that drops the
    largest normal component: a cycle whose consecutive turns are all strictly of one sign and
    whose lexicographic order changes direction exactly twice is a strictly convex polygon
    traversed once. Topology as in check_reference. Returns the outward rows.
    """
    verts = p.vertices
    faces = p.faces
    if len(faces) < 4 or len(verts) < 4:
        _fail("fewer than four faces or vertices")
    for v in verts:
        if len(v) != 4 or v[3] <= 0 or gcd(*v) != 1:
            _fail("vertex not primitive with w > 0")
    rows, oriented = [], []
    for face in faces:
        k = len(face)
        if k < 3 or len(set(face)) != k:
            _fail("degenerate face cycle")
        a = face[0]
        n = _cross(_diff(face[1], a), _diff(face[2], a))
        if not any(n):
            _fail("collinear leading triple (not strictly convex)")
        row = (n[0] * a[3], n[1] * a[3], n[2] * a[3], -_dot3(n, a))
        g = gcd(*row)
        rows.append(tuple(x // g for x in row))
    signs = _sign_matrix(rows, p)
    index = {v: j for j, v in enumerate(verts)}
    for fi, face in enumerate(faces):
        k = len(face)
        row = rows[fi]
        sg = signs[fi]
        if max(sg) > 0:
            if min(sg) < 0:
                _fail("face plane separates vertices (not a supporting plane)")
            row = tuple(-x for x in row)
            rows[fi] = row
        if any(sg[index[q]] != 0 for q in face):
            _fail("face not planar")
        if all(x == 0 for x in sg):
            _fail("all vertices on one plane")
        drop = max(range(3), key=lambda t: abs(row[t]))
        i, j = (drop + 1) % 3, (drop + 2) % 3
        turn = 0
        for t in range(k):
            a, b, c = face[t - 2], face[t - 1], face[t]
            cr = ((b[i] * a[3] - a[i] * b[3]) * (c[j] * b[3] - b[j] * c[3])
                  - (b[j] * a[3] - a[j] * b[3]) * (c[i] * b[3] - b[i] * c[3]))
            sg = (cr > 0) - (cr < 0)
            if sg == 0 or (turn and sg != turn):
                _fail("face cycle not strictly convex")
            turn = sg
        changes = 0
        prev = _lex(face[-1], face[0], i, j)
        for t in range(k):
            cur = _lex(face[t], face[(t + 1) % k], i, j)
            if cur == 0:
                _fail("repeated projected vertex")
            if cur != prev:
                changes += 1
            prev = cur
        if changes != 2:
            _fail("face cycle winds more than once")
        outward_ccw = turn * (1 if row[drop] > 0 else -1) > 0
        oriented.append(list(face) if outward_ccw else list(reversed(face)))
    _topology(faces, verts, oriented, rows)
    return rows


def verify_split(parent, row, neg, pos):
    """Per-split certificate of the cover invariant (no feasible volume is lost by a split).

    Lemma V: for a polytope P and a plane meeting its interior, the vertices of
    P & {row <= 0} are exactly the vertices of P with row <= 0 together with the points where
    an edge of P with strictly opposite end signs crosses the plane (and symmetrically for
    {row >= 0}). If the face structure of P satisfies FS, its face edges are the edges of P.

    Inductively every polytope satisfies FS (the cube does; each split output is checked
    here), so the halves are certified exactly when (1) their vertex sets equal the two
    expected sets of Lemma V (exact signs and exact crossing points) and (2) both pass check().
    Then conv(V(neg)) = P & {row <= 0} and conv(V(pos)) = P & {row >= 0}: the halves cover P,
    overlap only on the plane, and their faces describe them exactly. Raises ArithmeticError.
    """
    global VERIFIED
    signs = {v: form(row, v) for v in parent.vertices}
    crossing = set()
    for f in parent.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            u, w = signs[a], signs[b]
            if (u < 0 < w) or (w < 0 < u):
                q = tuple(abs(w) * x + abs(u) * y for x, y in zip(a, b))
                g = gcd(*q)
                crossing.add(tuple(x // g for x in q))
    expect_neg = {v for v, x in signs.items() if x <= 0} | crossing
    expect_pos = {v for v, x in signs.items() if x >= 0} | crossing
    if set(neg.vertices) != expect_neg or set(pos.vertices) != expect_pos:
        _fail("split vertex sets differ from the exact half-space vertices")
    check(neg)
    check(pos)
    VERIFIED += 1


class instrument_splits:
    """Test/audit oracle: certify every Poly.split in the block (not used in production).

    with polycheck.instrument_splits(): ...   # each split must pass verify_split
    """

    def __enter__(self):
        import poly
        self._poly = poly
        self._orig = poly.Poly.split

        def split(p, row, _orig=self._orig):
            a, b = _orig(p, row)
            if a is not None and b is not None:
                verify_split(p, row, a, b)
            return a, b
        poly.Poly.split = split
        return self

    def __exit__(self, *exc):
        self._poly.Poly.split = self._orig
        return False

