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

from poly import form


def _diff(b, a):
    """(b/wb - a/wa) scaled by wa*wb > 0 (exact, sign-preserving)."""
    return (b[0] * a[3] - a[0] * b[3], b[1] * a[3] - a[1] * b[3], b[2] * a[3] - a[2] * b[3])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _dot3(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _fail(msg):
    raise ArithmeticError("face structure: " + msg)


def check(p):
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
