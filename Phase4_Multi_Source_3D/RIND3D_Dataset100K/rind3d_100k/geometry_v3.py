"""Contract v3 geometry: the v0.1 convex recipes with observable thin slabs (docs/THIN_SLABS_V3.md).

**Change from v0.1 (`rind3d.lite_generators.generate`).** Only the half-thickness of the `thin` kind changes. That
kind is every body of `thin_slabs` and a quarter of the bodies of `occluder_clusters`.
* v0.1 draws log2(half) ~ U(-20, -5): full thickness from 2^-19 (0.002 voxel at N = 1024) to 1/16.
* v3 draws log2(half) ~ U(log2(T_min / (2 N)), -5): full thickness >= T_min voxels. This is exactly the v0.1 law
  conditioned on thickness >= T_min.
* It consumes the same single uniform. Every other draw of the scene (counts, rotations, sizes, centres, the other
  bodies) is therefore the v0.1 draw for the same (seed, slot, attempt), which makes paired comparisons possible.
* With T_min = 0 the generator is byte-identical to v0.1 (tests/test_thin_slabs_v3.py).

**Validity (exact).** Every thin slab must have a nonempty, 6-connected voxel-centre sampling (receivers (v + 1/2)/N
inside the closed slab, within the world).
* Checked by `sampled_solid_components` on the slab's exact canonical rows. A column decomposition along one lattice
  axis is used; convexity makes each column a run. Membership near the boundary is re-checked in exact rationals.
* The SOLID observation and the visibility predicate both use the same continuous canonical body. Nothing is repaired
  on the voxel side.
* A scene whose slab fails the check is a failed geometry proposal: it is recorded (`thin_slab_invalid`) and the scene
  attempt is rejected, then retried by the normal bounded-attempt rule. No invalid slab is ever accepted.

**Theory (docs/THIN_SLABS_V3.md).** For an *unbounded* slab of thickness T >= ||n||_1 voxels (n its unit normal) the
sampling is 6-connected (proved there). Since ||n||_1 <= sqrt(3), T >= 1.74 voxels suffices for every orientation.
For a finite slab the rim is not covered by that proof, so the check above is what guarantees validity. T = 1 voxel
fails for some orientations and offsets.
"""
from fractions import Fraction

import numpy as np
from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind, make_leaf
from rind.storage import CanonicalBundle, pack_arrays, _decode_bundle
from rind3d.geometry import keyed_rng
from rind3d.lite_generators import DISTRIBUTION as V01_DISTRIBUTION, FAMILIES, VERSION as V01_VERSION, rotation, log_uniform

VERSION = "lite-convex-recipes-v0.1+thin-min-v3"


def thin_log2_lo(t_min_vox, N=1024):
    """Lower bound of log2(half thickness) for a minimum full thickness of t_min_vox voxels (v0.1: -20)."""
    return -20.0 if t_min_vox <= 0 else max(-20.0, float(np.log2(t_min_vox / (2.0 * N))))


def generate_v3(seed, slot, family, attempt=0, thin_min_vox=2.0, N=1024):
    """v0.1 recipes (lite_generators.generate, same streams) with thin half-thickness ~ 2^U(thin_log2_lo, -5).
    Returns (bundle, recipe); recipe['objects'][i]['kind'] marks the thin slabs."""
    if family not in FAMILIES:
        raise ValueError('Uncertified Lite generator family: ' + str(family))
    lo2 = thin_log2_lo(thin_min_vox, N)
    rng = keyed_rng(seed, slot, 8101 + attempt * 97); bundle = CanonicalBundle(3); roots = []; objects = []; proposal_failures = []
    lo, hi = V01_DISTRIBUTION['object_count_ranges'][family]; count = int(rng.integers(lo, hi + 1)); frame = rotation(rng)
    anchor = rng.uniform(.4, .6, 3); layout_scale = float(rng.uniform(.18, .29))

    def add(kind, index):
        rot = rotation(rng); normals = offsets = None; radius = np.ones(3)
        if kind == 'thin':
            half = np.r_[log_uniform(rng, .07, .24, 2), 2. ** rng.uniform(lo2, -5)]
        elif kind == 'prism':
            sides = int(rng.integers(3, 11)); angles = np.arange(sides) * 2 * np.pi / sides
            normals = np.r_[np.c_[np.cos(angles), np.sin(angles), np.zeros(sides)], [[0, 0, -1], [0, 0, 1]]]
            offsets = np.ones(sides + 2); half = log_uniform(rng, .02, .12, 3); radius = np.array([2.1, 2.1, 1.])
        elif kind == 'wedge':
            normals = np.array([[-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [1, 0, 1]], dtype=float)
            offsets = np.array([1, 1, 1, 1, .4]); half = log_uniform(rng, .02, .16, 3); radius = np.full(3, 1.5)
        elif kind == 'poly':
            planes = int(rng.choice([8, 12, 16, 24, 32])); extra = rng.normal(size=(planes - 6, 3)); extra /= np.linalg.norm(extra, axis=1)[:, None]
            normals = np.r_[np.eye(3), -np.eye(3), extra]; offsets = np.r_[np.ones(6), rng.uniform(.58, 1.05, planes - 6)]
            half = log_uniform(rng, .035, .22, 3)
        elif kind == 'small':
            half = log_uniform(rng, .008, .04, 3)
        elif kind == 'large':
            half = log_uniform(rng, .07, .24, 3)
        else:
            half = log_uniform(rng, .025, .22, 3)
        if family in ('corridors', 'barriers'):
            rot = frame if family == 'corridors' else frame @ rotation(rng)
            half = np.array([rng.uniform(.002, .014), layout_scale * rng.uniform(.55, 1.), layout_scale * rng.uniform(.55, 1.)])
        matrix = np.asarray(rot @ np.diag(half), dtype='<f8'); extent = np.abs(matrix) @ radius
        if np.any(extent >= .49):
            centre = np.full(3, .5); proposal_failures.append({'object': index, 'reason': 'conservative_world_fitting_interval_empty'})
        else:
            centre = rng.uniform(extent + .01, 1 - extent - .01)
            if family == 'occluder_clusters':
                centre = np.clip(anchor + rng.normal(0, .09, 3), extent + .01, 1 - extent - .01)
            elif family in ('corridors', 'barriers'):
                local = np.array([(-1 if index % 2 else 1) * layout_scale * .65, rng.uniform(-.12, .12), rng.uniform(-.1, .1)])
                if family == 'barriers':
                    local[0] = (index / (count - 1) - .5) * .5
                centre = np.clip(anchor + frame @ local, extent + .01, 1 - extent - .01)
        fields = {'center': np.asarray(centre, dtype='<f8'), 'matrix': matrix}
        if normals is not None:
            fields.update(normal=np.asarray(normals, dtype='<f8'), plane_offset=np.asarray(offsets, dtype='<f8'))
        item = make_leaf('affine_box_v1' if normals is None else 'convex_halfspaces_v1', 3, **fields)
        bundle.add_leaf(item); roots.append(item.content_hash)
        objects.append({'kind': kind, 'leaf_id': item.content_hash.hex(), 'center': centre.tolist(), 'matrix': matrix.tolist(),
                        'half_parameters': np.asarray(half).tolist(), 'declared_planes': 6 if normals is None else len(normals),
                        'orientation': rot.tolist(), 'conservative_extent': extent.tolist()})

    for index in range(count):
        if family == 'thin_slabs':
            kind = 'thin'
        elif family == 'prisms':
            kind = 'prism'
        elif family == 'wedges':
            kind = 'wedge'
        elif family == 'convex_polyhedra':
            kind = 'poly'
        elif family == 'multiscale_convex':
            kind = str(rng.choice(['small', 'large', 'prism', 'poly'], p=[.4, .35, .15, .1]))
        elif family == 'occluder_clusters':
            kind = str(rng.choice(['small', 'box', 'thin'], p=[.5, .25, .25]))
        else:
            kind = 'box'
        add(kind, index)
    if len(roots) > 1:
        node = CSGNode(3, CSGOpcode.UNION, tuple(roots)); bundle.add_node(node); root = node.content_hash
    else:
        root = roots[0]
    bundle.add_geometry(Geometry(3, RootKind.REF, root)); bundle = _decode_bundle(pack_arrays(bundle), 3)
    dist = dict(V01_DISTRIBUTION); dist['version'] = VERSION if thin_min_vox > 0 else V01_VERSION
    if thin_min_vox > 0:
        dist['thin_half_thickness_log2'] = [lo2, -5]; dist['thin_min_thickness_vox'] = float(thin_min_vox)
    return bundle, {'version': dist['version'], 'family': family, 'seed': int(seed), 'slot': int(slot), 'attempt': int(attempt),
                    'geometry_id': bundle.geometries[0].content_hash.hex(), 'objects': objects, 'proposal_failures': proposal_failures,
                    'distribution': dist, 'source_selection': 'independent uniform world proposals with exact FREE/clearance rejection'}


# ------------------------------------------------------------------ exact sampled-SOLID connectivity of a convex body
def _exact_in(rows_q, p_q):
    return all(r[0] * p_q[0] + r[1] * p_q[1] + r[2] * p_q[2] + r[3] <= 0 for r in rows_q)


def sampled_solid_components(rows_q, N=1024, margin=1e-9):
    """Exact 6-connectivity of {v in {0..N-1}^3 : (v + 1/2)/N in the closed convex body} for exact rational rows
    (a, b, c, d) with a x + b y + c z + d <= 0 inside.
    Returns dict(n_vox, n_components, axis, n_columns, exact_rechecks).
    Method: columns along the axis of largest extent; each column is one run (convexity); runs of 4-adjacent columns
    are 6-connected iff they overlap. Bounds are computed in float64, and every run endpoint within `margin` (in
    voxels) of a float bound is re-decided exactly with the rational rows."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    R = np.array([[float(v) for v in r] for r in rows_q])
    A, d = R[:, :3] / N, R[:, 3]                              # in voxel units: A.(v + 1/2) + d <= 0
    # bounding box of the body (LP-free: vertices of the polytope by triple intersections)
    verts = []
    import itertools
    for i, j, k in itertools.combinations(range(len(R)), 3):
        M = A[[i, j, k]]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, -d[[i, j, k]])
        if np.all(A @ x + d <= 1e-9 * np.abs(d).max()):
            verts.append(x)
    if not verts:
        return dict(n_vox=0, n_components=0, axis=-1, n_columns=0, exact_rechecks=0)
    V = np.array(verts) - 0.5                                  # vertex positions in lattice-index units
    lo = np.clip(np.floor(V.min(0)) - 1, 0, N - 1).astype(np.int64); hi = np.clip(np.ceil(V.max(0)) + 1, 0, N - 1).astype(np.int64)
    ax = int(np.argmax(hi - lo)); o1, o2 = [a for a in range(3) if a != ax]
    I = np.arange(lo[o1], hi[o1] + 1); J = np.arange(lo[o2], hi[o2] + 1)
    II, JJ = np.meshgrid(I, J, indexing="ij"); II = II.ravel(); JJ = JJ.ravel()
    base = A[:, o1][None] * (II[:, None] + 0.5) + A[:, o2][None] * (JJ[:, None] + 0.5) + d[None]     # (cols, rows)
    az = A[:, ax]
    zlo = np.full(len(II), -np.inf); zhi = np.full(len(II), np.inf); ok = np.ones(len(II), bool); amb = np.zeros(len(II), bool)
    for r in range(len(R)):
        if az[r] > 0:
            zhi = np.minimum(zhi, (-base[:, r]) / az[r] - 0.5)
        elif az[r] < 0:
            zlo = np.maximum(zlo, (-base[:, r]) / az[r] - 0.5)
        else:
            ok &= base[:, r] <= margin * np.abs(A[r]).sum()
            amb |= np.abs(base[:, r]) <= margin * np.abs(A[r]).sum()
    m_lo = np.maximum(np.ceil(zlo - margin), lo[ax]).astype(np.int64)
    m_hi = np.minimum(np.floor(zhi + margin), hi[ax]).astype(np.int64)
    amb |= (np.abs(zlo - np.round(zlo)) <= margin) | (np.abs(zhi - np.round(zhi)) <= margin)
    rechecks = 0
    for c in np.flatnonzero(amb & ok & (m_lo <= m_hi + 1)):      # exact endpoint decisions (rare)
        def inside(m):
            p = [None] * 3; p[ax] = Fraction(2 * int(m) + 1, 2 * N); p[o1] = Fraction(2 * int(II[c]) + 1, 2 * N); p[o2] = Fraction(2 * int(JJ[c]) + 1, 2 * N)
            return _exact_in(rows_q, p)
        a_, b_ = int(m_lo[c]), int(m_hi[c])
        while a_ <= b_ and not inside(a_):
            a_ += 1; rechecks += 1
        if a_ > lo[ax] and inside(a_ - 1):
            a_ -= 1; rechecks += 1
        while b_ >= a_ and not inside(b_):
            b_ -= 1; rechecks += 1
        if b_ < hi[ax] and b_ >= a_ and inside(b_ + 1):
            b_ += 1; rechecks += 1
        m_lo[c], m_hi[c] = a_, b_
    good = ok & (m_lo <= m_hi)
    n_vox = int(np.sum((m_hi - m_lo + 1)[good]))
    if n_vox == 0:
        return dict(n_vox=0, n_components=0, axis=ax, n_columns=0, exact_rechecks=rechecks)
    nI, nJ = len(I), len(J)
    idx = np.full(nI * nJ, -1, np.int64); g = np.flatnonzero(good); idx[g] = np.arange(len(g))
    rows_, cols_ = [], []
    for di, dj in ((1, 0), (0, 1)):
        a = np.arange(nI * nJ).reshape(nI, nJ)
        p = a[:nI - di, :nJ - dj].ravel(); q = a[di:, dj:].ravel()
        e = good[p] & good[q] & (np.maximum(m_lo[p], m_lo[q]) <= np.minimum(m_hi[p], m_hi[q]))
        rows_.append(idx[p[e]]); cols_.append(idx[q[e]])
    r_ = np.concatenate(rows_); c_ = np.concatenate(cols_)
    G = coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(g), len(g)))
    ncomp, _ = connected_components(G, directed=False)
    return dict(n_vox=n_vox, n_components=int(ncomp), axis=ax, n_columns=int(len(g)), exact_rechecks=rechecks)


def thin_slab_report(context, recipe, N=1024, min_thickness_vox=None):
    """Check every thin body of a scene: exact sampled SOLID nonempty and 6-connected, and the declared thickness.
    The canonical bodies of `context` are in recipe order (one leaf per object)."""
    out = []
    by_leaf = {c.leaf_id: c.rows for c in context.components}
    for i, ob in enumerate(recipe["objects"]):
        if ob["kind"] != "thin":
            continue
        rows = [tuple(Fraction(v) for v in r) for r in by_leaf[ob["leaf_id"]]]
        rep = sampled_solid_components(rows, N)
        t_vox = 2 * float(ob["half_parameters"][2]) * N
        rep.update(object=i, thickness_vox=t_vox,
                   ok=bool(rep["n_vox"] > 0 and rep["n_components"] == 1 and (min_thickness_vox is None or t_vox >= min_thickness_vox * (1 - 1e-12))))
        out.append(rep)
    return out
