"""Extended obstacle families for the geometry-diversity study (docs/GEOMETRY_DIVERSITY.md).

Every new family is a finite union of bounded convex polyhedra, built with the same canonical leaves as the v0.1
generator (`affine_box_v1`, `convex_halfspaces_v1`). Curved-LOOKING obstacles are therefore faceted polyhedra
that ARE the canonical physical geometry (case 1). The polyhedral posterior engine stays exact for the declared
world, and no smooth object is being approximated.

New families:
* faceted_cylinders: n-gon prisms, n in {12, 16, 24}, elongated pillars and disks.
* faceted_spheres: Fibonacci-sphere polyhedra with 20, 32 or 48 planes.
* rounded_boxes: boxes with chamfered edges and corners (26 planes).
* curved_walls: arcs of 4-8 thin slabs on a random circle (curved partial barriers).
* mixed_irregular: per object a mixture of box, thin, prism, poly, cyl, sphere and rbox kinds.

The v0.1 families are delegated unchanged to rind3d.lite_generators.generate."""
import numpy as np
from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind, make_leaf
from rind.storage import CanonicalBundle, pack_arrays, _decode_bundle
from rind3d.geometry import keyed_rng
from rind3d.lite_generators import FAMILIES as V01_FAMILIES, generate as v01_generate, rotation, log_uniform

VERSION = "rind3d-1k-families-v1"
NEW_FAMILIES = ("faceted_cylinders", "faceted_spheres", "rounded_boxes", "curved_walls", "mixed_irregular")
COUNTS = {"faceted_cylinders": (2, 8), "faceted_spheres": (2, 8), "rounded_boxes": (2, 8), "curved_walls": (1, 3),
          "mixed_irregular": (3, 12)}


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _rbox_normals(ch):
    f = np.r_[np.eye(3), -np.eye(3)]
    e = []
    for a in range(3):
        for b in range(a + 1, 3):
            for sa in (1, -1):
                for sb in (1, -1):
                    v = np.zeros(3); v[a] = sa; v[b] = sb; e.append(v / np.sqrt(2))
    c = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]) / np.sqrt(3)
    N = np.r_[f, np.array(e), c]
    off = np.r_[np.ones(6), np.full(12, (2 - ch) / np.sqrt(2)), np.full(8, (3 - 1.5 * ch) / np.sqrt(3))]
    return N, off


def generate(seed, slot, family, attempt=0):
    if family in V01_FAMILIES:
        return v01_generate(seed, slot, family, attempt)
    if family not in NEW_FAMILIES:
        raise ValueError(family)
    rng = keyed_rng(seed, slot, 18101 + attempt * 97)
    bundle = CanonicalBundle(3); roots = []; objects = []; failures = []
    lo, hi = COUNTS[family]; count = int(rng.integers(lo, hi + 1))

    def leaf(centre_fn, kind, rot, half, normals=None, offsets=None, radius=np.ones(3), index=0):
        matrix = np.asarray(rot @ np.diag(half), dtype="<f8"); extent = np.abs(matrix) @ radius
        if np.any(extent >= .49):
            failures.append({"object": index, "reason": "conservative_world_fitting_interval_empty"}); return
        centre = np.asarray(centre_fn(extent), dtype="<f8")
        fields = {"center": centre, "matrix": matrix}
        if normals is not None:
            fields.update(normal=np.asarray(normals, dtype="<f8"), plane_offset=np.asarray(offsets, dtype="<f8"))
        item = make_leaf("affine_box_v1" if normals is None else "convex_halfspaces_v1", 3, **fields)
        bundle.add_leaf(item); roots.append(item.content_hash)
        objects.append({"kind": kind, "leaf_id": item.content_hash.hex(), "center": centre.tolist(),
                        "matrix": matrix.tolist(), "half_parameters": np.asarray(half).tolist(),
                        "declared_planes": 6 if normals is None else len(normals)})

    uniform_centre = lambda ext: rng.uniform(ext + .01, 1 - ext - .01)

    def add_kind(kind, index):
        rot = rotation(rng)
        if kind == "cyl":
            n = int(rng.choice([12, 16, 24])); a = np.arange(n) * 2 * np.pi / n
            N = np.r_[np.c_[np.cos(a), np.sin(a), np.zeros(n)], [[0, 0, -1], [0, 0, 1]]]
            r = log_uniform(rng, .012, .07); h = log_uniform(rng, .01, .22)
            leaf(uniform_centre, kind, rot, np.array([r, r, h]), N, np.ones(n + 2), np.array([1.1, 1.1, 1.]), index)
        elif kind == "sphere":
            n = int(rng.choice([20, 32, 48])); r = log_uniform(rng, .02, .12)
            leaf(uniform_centre, kind, rot, np.full(3, r), _fib(n), np.ones(n), np.full(3, 1.6), index)
        elif kind == "rbox":
            N, off = _rbox_normals(float(rng.uniform(.2, .8)))
            leaf(uniform_centre, kind, rot, log_uniform(rng, .02, .2, 3), N, off, np.ones(3), index)
        elif kind == "thin":
            leaf(uniform_centre, kind, rot, np.r_[log_uniform(rng, .07, .24, 2), 2. ** rng.uniform(-20, -5)], index=index)
        elif kind == "prism":
            s = int(rng.integers(3, 11)); a = np.arange(s) * 2 * np.pi / s
            N = np.r_[np.c_[np.cos(a), np.sin(a), np.zeros(s)], [[0, 0, -1], [0, 0, 1]]]
            leaf(uniform_centre, kind, rot, log_uniform(rng, .02, .12, 3), N, np.ones(s + 2), np.array([2.1, 2.1, 1.]), index)
        elif kind == "poly":
            p = int(rng.choice([8, 12, 16, 24, 32])); ex = rng.normal(size=(p - 6, 3)); ex /= np.linalg.norm(ex, axis=1)[:, None]
            leaf(uniform_centre, kind, rot, log_uniform(rng, .035, .22, 3), np.r_[np.eye(3), -np.eye(3), ex],
                 np.r_[np.ones(6), rng.uniform(.58, 1.05, p - 6)], np.ones(3), index)
        else:
            leaf(uniform_centre, kind, rot, log_uniform(rng, .025, .22, 3), index=index)

    if family == "curved_walls":
        idx = 0
        for _ in range(count):
            frame = rotation(rng); R = float(rng.uniform(.12, .3)); span = float(rng.uniform(np.pi / 3, np.pi))
            nseg = int(rng.integers(4, 9)); a0 = float(rng.uniform(0, 2 * np.pi))
            c0 = rng.uniform(.35, .65, 3); th = float(rng.uniform(.002, .014)); hh = float(rng.uniform(.05, .14))
            seg = R * span / nseg / 2 * 1.08
            for i in range(nseg):
                a = a0 + span * (i + .5) / nseg
                rot = frame @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
                local = np.array([R * np.cos(a), R * np.sin(a), 0.])
                cen = lambda ext, local=local: np.clip(c0 + frame @ local, ext + .01, 1 - ext - .01)
                leaf(cen, "arcwall", rot, np.array([th, seg, hh]), index=idx); idx += 1
    else:
        for index in range(count):
            if family == "faceted_cylinders": kind = "cyl"
            elif family == "faceted_spheres": kind = "sphere"
            elif family == "rounded_boxes": kind = "rbox"
            else: kind = str(rng.choice(["box", "thin", "prism", "poly", "cyl", "sphere", "rbox"],
                                        p=[.2, .15, .1, .1, .15, .15, .15]))
            add_kind(kind, index)
    if not roots:
        failures.append({"object": -1, "reason": "no objects"})
        return None, {"proposal_failures": failures}
    if len(roots) > 1:
        node = CSGNode(3, CSGOpcode.UNION, tuple(roots)); bundle.add_node(node); root = node.content_hash
    else:
        root = roots[0]
    bundle.add_geometry(Geometry(3, RootKind.REF, root)); bundle = _decode_bundle(pack_arrays(bundle), 3)
    return bundle, {"version": VERSION, "family": family, "seed": int(seed), "slot": int(slot), "attempt": int(attempt),
                    "geometry_id": bundle.geometries[0].content_hash.hex(), "objects": objects,
                    "proposal_failures": failures}


def acceptance(context, family, seed, slot, attempt=0):
    """The v0.1 geometry acceptance gate. Its checks are family-independent: certified geometry proof, world-face
    gaps, membership probes, segment reversal, two independent physical predicates. New families are run under the
    stand-in label 'convex_polyhedra' and the real family is recorded."""
    from rind3d.lite_acceptance import geometry_acceptance
    rep = geometry_acceptance(context, family if family in V01_FAMILIES else "convex_polyhedra", seed, slot, attempt)
    rep["family"] = family
    return rep
