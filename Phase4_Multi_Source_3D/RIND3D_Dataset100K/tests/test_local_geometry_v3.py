"""Local inverse geometry under the actual v3 geometry distribution (docs/LOCAL_INVERSE_GEOMETRY.md §3b).

* Every v3 body contains a ball of radius >= 1 voxel, so the sub-voxel construction of Theorem 3 is not realizable.
* Theorem 3': an explicit, v3-admissible knife-edge wedge scene whose bodies have nonempty 6-connected samplings, with a
  direct-interface window in which 270+ interface edges cross the sub-voxel knife strip (non-free edges) and no plane
  through the true driver source crosses them all (exact LP of Proposition 4); robust under small perturbations.
CPU only (certified CPU predicates)."""
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linprog

from rind3d.geometry import keyed_rng
from rind3d.lite_generators import rotation
from rind3d_100k.geometry_v3 import generate_v3, sampled_solid_components
from rind3d_posterior.localfirst.local import interface_edges, edge_free_mask

N = 1024
WEDGE_N = np.array([[-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [1, 0, 1]], float)
WEDGE_B = np.array([1, 1, 1, 1, .4])


@pytest.fixture(scope="module")
def fac():
    from rind3d.rendering.backends import BackendFactory
    return BackendFactory(cuda_allowed=False)


def _ctx(fac, leaves):
    from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind
    from rind.storage import CanonicalBundle, pack_arrays, _decode_bundle
    from rind3d_1k.physical import POLICY
    b = CanonicalBundle(3)
    for x in leaves:
        b.add_leaf(x)
    if len(leaves) > 1:
        node = CSGNode(3, CSGOpcode.UNION, tuple(x.content_hash for x in leaves)); b.add_node(node); root = node.content_hash
    else:
        root = leaves[0].content_hash
    b.add_geometry(Geometry(3, RootKind.REF, root))
    return fac.context(_decode_bundle(pack_arrays(b), 3), POLICY)


def _wedge(c, rot, half):
    from rind.contracts import make_leaf
    return make_leaf("convex_halfspaces_v1", 3, center=np.asarray(c, "<f8"), matrix=np.asarray(rot @ np.diag(half), "<f8"),
                     normal=WEDGE_N.astype("<f8"), plane_offset=WEDGE_B.astype("<f8"))


def _inradius(rows):
    R = np.array([[float(v) for v in r] for r in rows]); A, d = R[:, :3], R[:, 3]; nA = np.linalg.norm(A, axis=1)
    res = linprog([0, 0, 0, -1], A_ub=np.c_[A, nA], b_ub=-d, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    return float(res.x[3])


@pytest.mark.parametrize("family", ["thin_slabs", "occluder_clusters", "corridors", "barriers", "wedges", "prisms",
                                    "rotated_boxes", "convex_polyhedra", "multiscale_convex"])
def test_v3_bodies_contain_a_voxel_ball(fac, family):
    """Theorem 3's construction needs bodies smaller than a voxel; under v3 every body has inradius >= h."""
    for slot in range(3):
        b, rec = generate_v3(20261118, slot, family, 0, thin_min_vox=2)
        if rec["proposal_failures"]:
            continue
        from rind3d_1k.physical import POLICY
        ctx = fac.context(b, POLICY)
        for comp in ctx.components:
            assert _inradius(comp.rows) >= (1.0 / N) * (1 - 1e-9)


def _stab_through(li, da, s_vox):
    """Proposition 4 exactly: is there a plane n.x = d through s with n.l >= d >= n.k for every edge (l, k)?"""
    l = li + .5; k = da + .5; g = (l - k).mean(0); g /= np.linalg.norm(g)
    A_ub = np.r_[np.c_[-l, np.ones(len(l))], np.c_[k, -np.ones(len(k))]]
    A_eq = np.array([np.r_[s_vox, -1.0], np.r_[g, 0.0]])
    r = linprog(np.zeros(4), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq, b_eq=[0.0, 1.0], bounds=[(None, None)] * 4, method="highs")
    return r.status == 0


def knife_scene(fac, seed=121, half=(0.159, 0.10, 0.0201), d_rot=None, d_s=None, d_o=None):
    """The Theorem 3' scene: a `wedges`-recipe wedge (half sizes inside LU(.02, .16)) with a 7.2 degree knife edge, plus
    a second wedge; source on the oblique-face side; an L64 window centred 4 voxels inside the knife tip."""
    from rind3d_1k.physical import legal_source
    half = np.asarray(half)
    rot = rotation(keyed_rng(11, seed, 1))
    if d_rot is not None:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_rotvec(d_rot).as_matrix() @ rot
    leaves = [_wedge([.5, .5, .5], rot, half), _wedge([0.2, 0.2, 0.8], rotation(keyed_rng(11, seed, 2)), np.array([0.05, 0.05, 0.05]))]
    ctx = _ctx(fac, leaves); by = {c.leaf_id: c.rows for c in ctx.components}
    M = rot @ np.diag(half); c1 = np.full(3, .5)
    no = np.linalg.inv(M).T @ np.array([1, 0, 1.]); nb = np.linalg.inv(M).T @ np.array([0, 0, -1.])
    tip = c1 + M @ np.array([1.4, 0, -1]); inward = c1 + M @ np.array([1.0, 0, -1]) - tip; inward /= np.linalg.norm(inward)
    s = tip + 0.25 * no / np.linalg.norm(no) + 0.05 * inward + (np.zeros(3) if d_s is None else d_s)
    L = 64; o = np.floor((tip + 4.0 / N * inward) * N - L / 2).astype(int) + (0 if d_o is None else np.asarray(d_o))
    vox = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3); X = (o + vox + .5) / N
    sol = ctx.contains(np.ascontiguousarray(X)); lit, _, _ = ctx.bind(s).native.points(np.ascontiguousarray(X))
    cls = np.where(sol, 2, np.where(lit == 1, 1, 0)).reshape(L, L, L).astype(np.uint8)
    li, da = interface_edges(cls)
    free = edge_free_mask([by[x.content_hash.hex()] for x in leaves], o, N, li, da)
    knife = float(np.degrees(np.pi - np.arccos(nb @ no / np.linalg.norm(nb) / np.linalg.norm(no))))
    return dict(ctx=ctx, by=by, leaves=leaves, s=s, o=o, L=L, cls=cls, li=li, da=da, free=free, legal=legal_source(ctx, s) is not None,
                knife_deg=knife, s_vox=s * N - o)


def test_theorem3prime_knife_edge_counterexample(fac):
    k = knife_scene(fac)
    # v3-admissible and stronger: every body has a nonempty 6-connected voxel-centre sampling
    for x in k["leaves"]:
        rep = sampled_solid_components([tuple(Fraction(v) for v in r) for r in k["by"][x.content_hash.hex()]])
        assert rep["n_vox"] > 0 and rep["n_components"] == 1
    assert 0.02 < 0.0201 and 0.159 < 0.16 and 6.0 < k["knife_deg"] < 8.0          # recipe ranges (strictly inside)
    assert k["legal"]                                                              # legal driver source with clearance
    lo, hi = k["o"] / N, (k["o"] + k["L"]) / N
    assert not np.all((k["s"] >= lo) & (k["s"] <= hi))                             # outside the closed window box
    assert len(k["li"]) > 1000                                                     # v3-eligible: direct interfaces exist
    nf = ~k["free"]
    assert nf.sum() >= 200                                                         # H1 fails on a whole strip
    assert not _stab_through(k["li"][nf], k["da"][nf], k["s_vox"])                 # no plane through s crosses them
    assert _stab_through(k["li"][k["free"]], k["da"][k["free"]], k["s_vox"])       # the free edges: a plane through s does


def test_theorem3prime_is_robust_to_small_perturbations(fac):
    """The conclusion depends on finitely many strict inequalities, so it holds on an open set of generator parameters
    (positive probability). Evidence: small random perturbations of the rotation, source and window keep it."""
    rng = np.random.default_rng(0); held = 0; tried = 0
    for _ in range(8):
        k = knife_scene(fac, d_rot=rng.normal(0, np.radians(0.3), 3), d_s=rng.uniform(-2, 2, 3) / N, d_o=rng.integers(-1, 2, 3))
        if not k["legal"]:
            continue
        tried += 1
        nf = ~k["free"]
        held += int(nf.sum() >= 100 and not _stab_through(k["li"][nf], k["da"][nf], k["s_vox"]))
    assert tried >= 6 and held >= tried - 1
