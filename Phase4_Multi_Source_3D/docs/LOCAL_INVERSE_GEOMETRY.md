# Local inverse geometry of binary response interfaces

**Question.** Can the local response interface of one window (the local SOLID mask and one binary response channel)
define a safe candidate region for the driver source, so that the full world geometry is needed only inside that
region?

**Answer, in brief.**
1. *Theorem 1, the cone law.* For any closed occluder set and a point source, the response interface in FREE space is
   ruled by rays from the source. Every tangent plane of the interface contains the source. For polyhedra its pieces
   are planes through the source and an occluder edge or face (the "facet law"). For a ball it is a right circular
   cone with apex at the source.
2. *Theorem 3 (impossibility) applies to the old distribution.* For finite unions of closed convex polytopes,
   including sub-voxel bodies, local-only certified pruning is impossible: for every observation, almost every source
   position outside the window is feasible for some world that reproduces the observation exactly. This covers v2,
   whose `thin_slabs` are sub-voxel.
3. *Under the actual v3 distribution (§3b) the picture changes, but the obstruction does not disappear.*
   - **What v3 removes.** Every v3 body contains a ball of radius ≥ 1 voxel, and slab samplings are nonempty and
     6-connected. The construction of Theorem 3 is not realizable, and its full-measure conclusion is withdrawn for v3.
   - **What v3 does not remove (Theorem 3′).** Connected sampled solids are not locally observable occluders. Large
     convex bodies have sub-voxel *parts* near sharp edges and vertices, and such a part may be the only portion of
     the body inside the window. An explicit v3-admissible scene (a `wedges` knife edge of 7.2°) has 273 direct
     interface edges crossing the sub-voxel strip, and no plane through the true source crosses them all. The
     conclusion holds on an open set of generator parameters, so it has positive probability.
   - **What would make H1 certifiable (Proposition 11).** H1 is locally certifiable away from SOLID receivers and the
     window boundary, given a declared *local-thickness radius*. v3 does not declare one, and the always-valid bound
     is hundreds of voxels. H2 (one interface plane per component) stays undecidable from the observation.
   - **New structure (Lemma 10).** The interface extends away from the source along rulings, through free space. Its
     creases and the ends of its pieces lie on lines through the source.
4. *Restricted results that hold in both distributions.*
   - *Propositions 4–5.* The local constraint "the source lies on the stabbing set of a component" is safe under two
     hypotheses: H1, every interface edge is free; H2, the component is crossed by a single interface plane. Its
     region's size is governed by the singular values of the stacked plane normals and a boundedness threshold.
   - *Proposition 6.* H1 becomes locally checkable only under an inner-rolling-ball condition that polytopes violate.
   - *Estimate 7.* Curved interfaces carry apex information that is second order in the angular extent of the patch.
5. *Certified pathway (Proposition 8).* A sound local-first algorithm uses the local geometry only to choose which
   receivers' exact constraints to evaluate first. It obtains the certificate from the exact constraints themselves:
   an outer bound from a receiver subset, then the exact tau-pencil computation restricted to that bound. This is
   implemented in `RIND3D_Posterior/src/rind3d_posterior/localfirst/`. The measured comparison is in
   `RIND3D_Posterior/docs/LOCALFIRST_COMPARISON.md`.

Status labels used below: **Theorem/Proposition/Lemma** (proved here), **Estimate** (first-order analysis, not a bound),
**Observation** (empirical), **Heuristic** (an engineering choice with no correctness role).

## 1. Setting

**World and solid.**
* World Ω = (0,1)³.
* Solid S = ∪ᵢ Cᵢ, a finite union of closed convex bodies.
* Receivers are lattice points x = (v + ½)h, with h = 1/N and N = 1024.

**Observation.**
* A window W is a set of L³ receivers with closed box B_W.
* The response of driver source s at a FREE receiver x ∉ S is R_s(x) = 1 (response-active) iff the closed segment
  [s, x] misses S, and 0 (response-inactive) otherwise. This is the generator's closed-solid, closed-segment
  visibility.
* The observation is X = (σ, r): σ(x) = 1[x ∈ S] and r(x) = R_s(x) for x ∈ W.

**Domain and target.**
* Source domain: D(G) = {s ∈ Ω : dist(s, S) > ρ}, with ρ = 2⁻²⁶.
* Feasible set: F(G, X) = {s ∈ D(G) \ B_W : R_s(x) = r(x) for every FREE x ∈ W}.
* The label is the uniform law on F, as cell masses on the 128³ grid.

**Response-inactive sets.**
* Inactive set of a driver source: In(s) = {y : [s, y] ∩ S ≠ ∅} (the "shadow" of classical graphics).
* For a convex body C and a point x ∉ C: Occ(x, C) = {s : [s, x] ∩ C ≠ ∅} (the occlusion set; a shadow volume in graphics terms).
* A receiver x is inactive for s iff s ∈ ∪ᵢ Occ(x, Cᵢ).

**Interfaces.**
* A *direct interface edge* is a pair of 6-adjacent FREE receivers (u, v), |u − v| = h, with r(u) ≠ r(v). This is the
  contract-v3 eligibility primitive.
* The edge is *free* if the closed segment [u, v] misses S.

## 2. The proposed construction (hypothesis under test)

* **Stage 1 (local).** Extract the interface edges, group them into planar (or curved) components, and intersect the
  implied constraints into a candidate region B_loc(X).
* **Stage 2 (global).** Check the candidates against the exact predicate with the full geometry.
* **Stage 3.** Compute the certified cell masses.

The intuition: one plane gives a plane, two give a line, three give a point. §3 to §5 decide when Stage 1 is safe.

## 3. Proven statements

**Lemma 1 (convexity).**
* For a compact convex C and x ∉ C, Occ(x, C) = {x + t(c − x) : c ∈ C, t ≥ 1} is closed and convex.
* By symmetry, for fixed s, the set {x : [s, x] ∩ C ≠ ∅} is convex in x.

*Proof.* Take a = x + t₁(c₁ − x) and b = x + t₂(c₂ − x) with t₁, t₂ ≥ 1 and λ ∈ [0, 1]. Then
λa + (1−λ)b = x + T(c − x) with T = λt₁ + (1−λ)t₂ ≥ 1 and c = [λt₁c₁ + (1−λ)t₂c₂]/T ∈ C. Closedness follows from
compactness. ∎

*Consequences, used in §6.*
* **Inactive constraint.** A region R of sources makes a whole block Q of receivers inactive through one body C
  whenever every vertex pair (sᵢ, xⱼ) of conv R × conv Q satisfies [sᵢ, xⱼ] ∩ C ≠ ∅. This is bi-convexity: fix xⱼ
  and use Lemma 1 in s, then fix s and use it in x.
* **Active constraint.** The union of the segments between two convex sets is their convex hull. Hence a block Q is
  active for every s ∈ R iff conv(R ∪ Q) ∩ S = ∅.

**Theorem 1 (cone law).** Let S be closed and s ∉ S. Let p ∈ ∂In(s) \ S, and let c be the point of [s, p] ∩ S
farthest from s. Then the half-open segment (c, p] lies in ∂In(s). In other words, the interface is ruled by rays
from s. At every point where ∂In(s) has a tangent plane, that plane contains s.

*Proof.*
* In(s) is closed, so [s, p] meets S and c exists.
* Take y ∈ (c, p] and suppose y were interior to In(s), with a neighbourhood U ⊂ In(s).
* For every y′ ∈ U, the ray from s through y′ beyond y′ lies in In(s), because [s, z] ⊇ [s, y′] meets S. The union of
  these rays contains a neighbourhood of p, since p lies beyond y on the ray from s through y. So p would be interior,
  a contradiction.
* The boundary therefore contains a nondegenerate segment through p in direction p − s. A tangent plane at p
  contains that direction, and so contains s. ∎

**Corollary 1a (polyhedra; Crow 1977).** If every Cᵢ is a polytope, ∂In(s) \ S is contained in a finite union of:
* planar pieces of the planes aff(s ∪ e), for edges e of the Cᵢ (silhouette edges);
* planes of faces that contain s;
* rays from s through vertices.

*Proof.* The ray of Theorem 1 touches some Cᵢ only on its boundary, at c. The minimal face of Cᵢ containing the
contact is a facet (the ray lies in its plane, which then contains s), an edge e (the ray through e spans aff(s ∪ e)),
or a vertex (a ray through it). Each case gives finitely many pieces. ∎

This is the facet law measured in the Student lab: large facets lie a median 0.04–0.09 voxel from the source.

**Corollary 1b (ball).** For C = B(z, ρ_C) with |s − z| > ρ_C, ∂Occ(s, C) \ C is the part, beyond the contact circle,
of the right circular cone with apex s, axis s → z and half-angle α = arcsin(ρ_C / |s − z|).

**Proposition 2 (free interface edges cross a ruled sheet).** Let (u, v) be a direct interface edge for s that is
free. Then [u, v] contains a point p ∈ ∂In(s) \ S. In particular (Corollary 1a), for polyhedra p lies on a plane
through s spanned by an occluder edge or face, or on a ray from s through a vertex.

*Proof.* [u, v] ∩ In(s) is closed, nonempty and not all of [u, v], so it has a relative boundary point p. Then
p ∈ ∂In(s), and p ∉ S because the edge is free. ∎

*Without H1 (a free edge) nothing follows.* Counterexample C1 in §4 is a thin slab crossing [u, v]: u lies in front
of it and is active, v lies behind it and is inactive, and the interface is the slab's face, which need not contain s.

**Theorem 3 (no local-only certified pruning).** Let 𝒢 be the class of finite unions of closed convex polytopes. Take
any observation X = (σ, r) on W with at least one FREE receiver, and any s ∈ Ω \ B_W off a finite union of lines.
Then there is G ∈ 𝒢 with these properties:
* G reproduces X exactly;
* every FREE–FREE lattice edge of W is free in G;
* s ∈ F(G, X), clearance included.

Consequently, any B(X) with F(G, X) ⊆ B(X) for all G ∈ 𝒢 has full Lebesgue measure in Ω \ B_W.

*Proof (construction).*
* *Bodies.* Put a tetrahedron T_c of circumradius δ at every SOLID receiver c. For every inactive FREE receiver x,
  let q_x = x + ε(s − x)/|s − x| with ε = min(h/4, |s − x|/2), and put a tetrahedron T_x of circumradius δ at q_x.
  Let G = ∪T.
* *Generic s.* The excluded s are those on a line through two receivers, on an axis line through a receiver, or on a
  ray from an active receiver through a SOLID receiver.
* *Choice of δ.* For such s, choose δ > 0 below each of the finitely many positive quantities:
  - ε (so no T_x contains a receiver);
  - h/2 (so T_c contains only c);
  - the distances from every q_x to the lattice edges through x;
  - the distances from every T centre to every segment [s, y], for active y (these are positive because s is off
    the excluded lines);
  - dist(s, centres) − ρ.
* *Verification.*
  - σ is reproduced: only the SOLID receivers lie in G.
  - Every inactive x is inactive, because q_x ∈ [s, x].
  - Every active y is active.
  - No FREE–FREE edge meets G, because T_c touches only edges at the SOLID receiver c.
  - s has clearance from G. ∎

*Remark.* The construction uses sub-voxel bodies. The v3 families contain such bodies: `thin_slabs` thickness is
2^U(−20, −5) in world units. The Student lab measured that 3.8% of large facets violate the facet law because of
invisible slabs. Theorem 3 therefore applies to the declared population, not only to contrived worlds.

**Proposition 4 (the restricted local constraint).** Let C be a set of direct interface edges (lᵢ, kᵢ), with lᵢ
active and kᵢ inactive. Assume:
* *H1:* every edge of C is free;
* *H2:* one plane Π_s ∋ s crosses all of them, i.e. the edges cross the same interface plane.

Then s ∈ U(C) = ∪{Π : Π ∈ Stab(C)}, where
Stab(C) = {(n, d) : n · lᵢ − d ≥ 0 ≥ n · kᵢ − d for all i}, with the orientation fixed by the active side.

Stab(C) is a convex polyhedral cone in (n, d), and testing s ∈ U(C) is a linear feasibility problem.

*Proof.* By Proposition 2 and H2, Π_s ∩ [lᵢ, kᵢ] ≠ ∅ for every i. Oriented, this gives the two linear inequalities. ∎

*Notes.*
* Stab(C) is the oriented digital-plane preimage of C, in the sense of Coeurjolly et al. 2005: the planes whose
  digitization *contains* the edge set.
* The heuristic band in the code, `local.local_band_cells`, approximates U(C) by the fitted plane ± τ(s), where
  τ(s) = (√3/2) h (1 + 2|s − m| / ℓ) and ℓ is the minor in-plane extent of C. The thickness grows linearly with the
  distance from the window.

**Proposition 5 (rank and conditioning).**
* *Setting.* Let the components c = 1..K have unit normals n_c, offsets d_c and band half-widths
  τ_c(s) = a_c + b_c |s − m_c|, with all m_c within ℓ_W of a reference point m.
* *Normal matrix.* N = [n₁; …; n_K]. Write σ₁ ≥ σ₂ ≥ σ₃ for its singular values, and set b = max b_c and
  A = max |d_c − n_c·m| + a_c + b ℓ_W.
* *Bound.* Every s in the intersection of the bands satisfies
  σ₃ |s − m| ≤ |N(s − m)| ≤ √K (A + b |s − m|).
  So if σ₃ > √K b, the intersection lies in the ball of radius √K A / (σ₃ − √K b).
* *Proof.* Apply the triangle inequality to each row and take the Euclidean norm over K rows. ∎

*Cases.*
* **Rank 3, well conditioned:** a bounded "point-like" region.
* **Rank 3 with σ₃ ≤ √K b (near-pencil):** no boundedness guarantee; the region is a widening tube.
* **Rank 2:** the null direction ν of N is unconstrained except through the widening, so the region is line-like.
* **Rank 1:** a thickening plane.
* **Empty intersection:** proves that H1 or H2 failed for some component. It is evidence of an occluder face or a bent
  component, never of infeasibility.

*Numbers.* With the band of Proposition 4, b_c = √3 / ℓ_c (in voxels). Planes 50 voxels wide give b ≈ 0.035, so three
components need σ₃ > 0.06. The Student lab's gap-to-label tertiles by σ_min (3.12 / 1.54 / 0.83 nats) are the
empirical face of this threshold.

**Proposition 6 (when H1 can be certified locally).**
* *Assumption.* Every point of S lies in a closed ball of radius ρ₀ ≥ (√3/2) h contained in S (an inner rolling
  ball).
* *Claim.* Any edge [u, v] such that no SOLID receiver lies within √3 h + h/2 of its midpoint is free.
* *Proof.* If p ∈ [u, v] ∩ S, take the ball B ⊂ S of radius ρ₀ containing p. The covering radius of the cubic lattice
  is (√3/2) h ≤ ρ₀, so B contains a receiver, which is SOLID, within 2ρ₀ of p. ∎

Balls of radius ≥ 0.87 voxel satisfy the assumption. Polytopes, with their sharp edges and vertices, and thin slabs
do not. So H1 is locally checkable for sphere-only worlds with a minimum radius, but not for the v3 population.

**Estimate 7 (curved interfaces carry second-order apex information).**
* *Setting.* A window at distance D from s sees an arc of a ball's interface cone (Corollary 1b). The arc spans an
  azimuth Δφ ≈ ℓ / (D sin α).
* *Normals.* The unit normals n(φ) = cos α (cos φ e₁ + sin φ e₂) − sin α a lie on a cone around the axis a.
* *Conditioning.* For the stacked normals, σ₂ = O(Δφ) and σ₃ = O(Δφ²): a circular arc departs from a straight line
  quadratically. Every tangent plane contains s (Theorem 1), so apex localization from one curved patch is
  conditioned by σ₃ = O(Δφ²).
* *Visibility of curvature.* The patch's sagitta, ℓ² cos α / (8 D sin α), is below h/2 iff ℓ² < 4 h D tan α. Below
  that size the patch is a plane at the voxel scale, and its curvature carries no information.
* *Example.* With D = 0.5 and tan α = 0.17, ℓ must exceed about 19 voxels for the curvature to be resolvable.

A curved patch is not unique evidence of one cone. A polyhedral occluder with small facets produces a piecewise-planar
interface that approximates it. By Theorem 1, every facet plane still contains s, so the constraint set is the same;
only its conditioning changes.

## 3b. The actual v3 distribution

**v3 world class.** The support of the v3 generator: the nine v0.1 recipes with `thin` bodies at least
T_min = 2 voxels thick, each slab's voxel-centre sampling nonempty and 6-connected (`rind3d_100k/geometry_v3.py`).

**Proposition 9 (no sub-voxel bodies).** Every v3 body contains a closed ball of radius ≥ h.
* *Proof.* Read off the recipe ranges.
  - Thin slabs have half-thickness ≥ T_min h/2 = h.
  - Corridor and barrier walls have half-thickness ≥ 0.002 > 2h.
  - The other kinds have half-sizes ≥ 0.008 (small boxes) and plane offsets ≥ 0.58 on half-sizes ≥ 0.02.
  - Each body is an affine image of a fixed convex shape. The ball's radius follows from the smallest half-size
    times the shape's inradius.
  - The test `test_v3_bodies_contain_a_voxel_ball` checks the Chebyshev inradius by LP on generated scenes of every
    family. ∎
* *Consequences.*
  - The covering radius of the cubic lattice is (√3/2) h < h, so every v3 body contains a receiver: no body is
    invisible.
  - The construction of Theorem 3 (bodies of circumradius δ < h placed at chosen points) is not realizable.
  - Whether ∪_G F(G, X) still has full measure for v3 is open. Theorem 3's proof no longer shows it.

**Connected sampling is not local observability.** v3 guarantees a *global* property of each body: nonempty,
connected sampling. A local-first argument needs a *local* property: every piece of solid that meets a receiver edge
of the window is seen as SOLID in the window. Two mechanisms separate the two.
* **Sub-voxel parts of large bodies.** Near an edge of dihedral angle θ, a convex body is thinner than h within about
  h / (2 tan(θ/2)) of the edge, and contains no receiver there.
  - The `wedges` recipe allows θ down to about 7° (half-sizes 0.16 and 0.02), so the strip is about 8 voxels wide.
  - Random `convex_polyhedra` planes allow arbitrarily small angles.
  - Rims of boxes and slabs (90°) and vertices behave the same way at the voxel scale.
* **Window-boundary entry.** A body whose thick, observable part lies outside the window can enter the window only
  through such a thin part. Its sampling is then connected, but contains no receiver of the window.

**Theorem 3′ (v3 counterexample; explicit and verified, `test_local_geometry_v3.py`).** There is a v3-admissible
scene, of positive probability under the v3 generator, with all of the following properties:
* **Admissibility.**
  - Family `wedges`: two wedges, half-sizes (0.159, 0.10, 0.0201) inside the recipe range LU(0.02, 0.16), a generic
    rotation, and a knife edge of 7.2°.
  - Every body has a nonempty, 6-connected sampling.
  - The source is legal and outside the window.
* **The window.** An L = 64 window with 2,924 direct interface edges. Of these, 273 are not free: [u, v] crosses the
  sub-voxel knife strip, a strip about 3.5 × 40 voxels.
* **Exact failure.** No plane through the true source s crosses all 273 edges with the right orientation (the exact
  LP of Proposition 4 is infeasible). Hence s ∉ U(C) for that component.
* **Contrast.** The free edges are all crossed by one plane through s.

*Positive probability.* The conclusion depends on finitely many strict inequalities: receiver classifications with no
ties, and LP infeasibility with a positive Farkas margin. So it holds on an open neighbourhood of the scene's
parameters (rotation, half-sizes, centres, source, window origin). The generator has a positive density there. The
test perturbs the rotation (0.3°), the source (2 voxels) and the window origin (±1 voxel); the conclusion holds in
every trial.

*Consequence.* Under v3, a local constraint of the form "s lies on a plane stabbing this component" can exclude the
true source. Local-only certified pruning remains impossible for the v3 population, now with an explicit
occluder-face component rather than the old full-measure construction. Interfaces that do not contain the source are
rarer than in v2 (Observation below), but not absent.

**Observation (v3 canary).** See §3c for the measured share of direct interface edges that are not free, v2 thin
slabs versus v3.

**Lemma 10 (ruling extension).** Let p ∈ ∂In(s) \ S and p′ = s + t(p − s) with t > 1. If the closed segment [p, p′]
misses S, then [p, p′] ⊂ ∂In(s).

*Proof.*
* Every y on [p, p′] is inactive, because [s, y] ⊇ [s, p] meets S.
* Take active points qₙ → p, i.e. [s, qₙ] ∩ S = ∅. For y = s + τ(p − s) with τ ∈ [1, t], set yₙ = s + τ(qₙ − s).
* [p, y] has positive distance from the closed S, and [qₙ, yₙ] → [p, y]. So [qₙ, yₙ] misses S for large n.
* Hence [s, yₙ] misses S, yₙ is active, and yₙ → y. ∎

*Consequences, for any occluder shape.*
* Within free space, the interface continues away from the source along every ruling until it meets solid.
* A planar piece that ends in free space ends along a ray through s.
* Two pieces meet along a line through s: two distinct planes through s meet in a line through s.

These are strong *line* constraints on s from a single window. They are usable only where the free space along the
ruling is certified (the observability problem below), and with the digitization tolerance of §5.

**Proposition 11 (local certificate for H1, given a declared local-thickness radius).**
* *Assumption.* Every point of every body lies within distance R of the centre of a ball of radius (√3/2) h contained
  in the body.
* *Claim.* A direct interface edge whose distance to every SOLID receiver of the window and to the window's faces
  exceeds R + (√3/2) h + h is free.
* *Proof.* Suppose p ∈ [u, v] ∩ S. The ball near p contains a receiver q (covering radius), with |q − p| ≤ R + (√3/2) h.
  That q is SOLID. If q lies in the window, the distance condition fails. If not, p is within R + (√3/2) h of the
  window's boundary, and the condition fails again. ∎

*Size of R.*
* **Always valid.** R ≤ diam(C) (√3/2) h / r_in(C): the convex hull of p and the inscribed ball B(c, r_in) contains
  the ball of radius (√3/2) h centred on [p, c] at distance |p − c| (√3/2) h / r_in from p. For v3 slabs (diameter up
  to about 700 voxels, r_in = 1 voxel) this gives R ≈ 600 voxels, so the certificate is vacuous in L ≤ 256 windows.
* **Estimate, not proved here.** For bodies whose vertices all admit an inscribed circular cone of half-angle ≥ α, the
  local bound is R ≈ (√3/2) h / sin α. That would require a declared sharpness bound, which v3 does not have
  (knife edges near 7° give R ≈ 14 voxels).

**What remains undecidable locally (H2).** Even when every edge is free, a component may merge pieces of different
interface planes. Their crease is a line through s, but the merge can make the stabbing set miss s (counterexample C2).
A component may also contain a free interface-plane piece next to a non-free occluder-face piece (Theorem 3′).
Segmentation is a heuristic, and no local test certifies it.

**Answer to the question for v3.** Improving physical observability removed the *sub-voxel body* obstruction, but
not the *sharp-feature* and *window-boundary* obstructions. A deterministic local-first algorithm can be certified
only in one of two ways:
* a declared sharpness bound, so that H1 is certifiable with margin R, plus a certified segmentation (H2), which no
  local test provides;
* the route of Proposition 8, which takes the certificate from exact receiver constraints with the world geometry.

Response-interface planes, creases, ruling ends and tangent planes of curved patches therefore stay what the
prototype uses them for: a guide for which receivers to evaluate. They are not pruning.

## 4. Counterexamples

| | construction | what fails |
|---|---|---|
| C1 occluder face | a slab of thickness < h between u and v: u in front (active), v behind (inactive) | H1. The interface is the slab face; s need not lie on it. 3.8% of large facets (Student lab) |
| C2 bent component | two interface planes through s meeting at a ray through the window (an occluder vertex) | H2 for the merged component. Stab(C) can exclude s: a single fitted plane over both pieces is offset from s by O(D · angle) |
| C3 sub-voxel construction | Theorem 3 | every local-only bound |
| C4 near-pencil | three planes whose normals have σ₃ < √K b | boundedness (Prop. 5): the candidate region is a long tube, and the localization is unstable |
| C5 small curved patch | a ball's interface cone with ℓ² < 4 h D tan α | curvature information (Est. 7): the patch acts as a plane |

## 5. Discretization

The observation gives only the edge [u, v] of length h, not the crossing point p.
* Proposition 4 already uses exactly this: a stabbing condition on segments, not on fitted points.
* A least-squares or RANSAC plane with a small residual is not a certificate. It is one element of Stab(C) at best.
  The certified object is the whole preimage cone, and only under H1–H2.
* Receivers sit at voxel centres, so no constraint is attached to SOLID receivers. They carry no response, which is
  why the target conditions only on FREE receivers.

## 6. Conditions for certification, and the certified local-first pathway

**What can and cannot be certified.**
* By Theorem 3, Stage 1 of §2 can never be certified pruning for the v3 population.
* It could be certified for a declared world class with an inner rolling ball (Prop. 6, for H1) and single-sheet
  components (H2). The second is not decidable from the observation, so a component split would itself need a
  certificate.

The pathway below is sound for every world, because it never trusts a local plane.

**Proposition 8 (soundness of the receiver-subset outer bound and of its exact restriction).**
* Let X′ ⊆ X be any subset of the observed FREE receivers. Then F(G, X) ⊆ F(G, X′), since dropping a constraint
  cannot remove a feasible source.
* Let B ⊇ F(G, X′) be a union of pieces computed by the exact engine. Running the exact engine on all receivers inside
  B computes F(G, X) ∩ B = F(G, X). Its interval records (lo, hi) per cell give the same tv_bound certificate as the
  unrestricted reference.
* *Assumptions.* The certificate's assumptions are those of the reference engine: A1, sound float-margin decisions;
  and of the outer-bound filter, A2, filter boxes contain B's pieces
  (`RIND3D_Posterior/experiments/fastlabel/CERTIFICATE_AUDIT.md`).

*Proof.* The first point is immediate. The second follows because B contains F, so restricting the search space
removes only infeasible sources. ∎

**What the local geometry contributes.** Lemma 1 explains why a few direct-interface receivers carry most of the
constraint. The set of sources that make v inactive and its neighbour u active is a thin shell. It lies within the
wedges {s : ∃ p ∈ [u, v], [s, p] ∩ e ≠ ∅} of occluder edges e. These are the "penumbra" regions of graphics, with the
segment [u, v] in the role of a linear emitter (Teller 1992; Drettakis–Fiume 1994). Its angular thickness is about h / dist(e, W).

**Heuristic.** Choosing those receivers spread over each planar component (farthest-point sampling), across
components with independent normals (Prop. 5), is a cost choice. It is not a correctness choice.

**Implementation.** `localfirst.pipeline.label_window_localfirst`.

## 7. References (bibliographic data verified against the publishers' records)

1. F. C. Crow. Shadow algorithms for computer graphics. *Computer Graphics* (SIGGRAPH '77) 11(2):242–248, 1977.
   doi:10.1145/965141.563901. *Supports Corollary 1a:* the shadow surface is bounded by planes through the light and
   the contour edges.
2. J.-Y. Bouguet, P. Perona. 3D photography using shadows in dual-space geometry. *IJCV* 35(2):129–149, 1999.
   doi:10.1023/A:1008124523456. *Supports:* the shadow of a straight edge lies in a plane through the light (the shadow
   plane). They locate the light by intersecting shadow *lines* (pencil tip to shadow tip), not shadow planes.
3. S. J. Teller. Computing the antipenumbra of an area light source. *Computer Graphics* (SIGGRAPH '92) 26(2):139–148,
   1992. doi:10.1145/142920.134029. *Supports §6:* visibility from area lights is bounded by planes and quadric
   surfaces.
4. Z. Gigus, J. Malik. Computing the aspect graph for line drawings of polyhedral objects. *IEEE TPAMI* 12(2):113–122,
   1990. doi:10.1109/34.44399. D. Lischinski, F. Tampieri, D. P. Greenberg. Discontinuity meshing for accurate
   radiosity. *IEEE CG&A* 12(6):25–39, 1992. doi:10.1109/38.163622. *Supports:* EV (planar) and EEE (quadric) visual
   events. Their area-light setting includes the point-light planes of ref. 1 as the degenerate case.
5. G. Drettakis, E. Fiume. A fast shadow algorithm for area light sources using backprojection. *Proc. SIGGRAPH '94*,
   223–230, 1994. doi:10.1145/192161.192207. *Supports:* complete visibility events for polyhedral scenes.
6. A. Laurentini. The visual hull concept for silhouette-based image understanding. *IEEE TPAMI* 16(2):150–162, 1994.
   doi:10.1109/34.273735. Also: The visual hull of curved objects, *ICCV* 1999, 356–361, doi:10.1109/ICCV.1999.791242.
   *Supports:* silhouettes and shadows from point lights determine only an outer bound (the visual hull). This is the
   shape-side analogue of Theorem 3's non-identifiability.
7. J. J. Koenderink. What does the occluding contour tell us about solid shape? *Perception* 13(3):321–330, 1984.
   doi:10.1068/p130321. R. Cipolla, P. Giblin. *Visual Motion of Curves and Surfaces*. Cambridge University Press,
   2000 (ISBN 0-521-63251-X). *Background for Corollary 1b and Estimate 7:* contour generators and apparent contours.
   The cone statement used here is proved directly in Theorem 1.
8. D. Coeurjolly, I. Sivignon, F. Dupont, F. Feschet, J.-M. Chassery. On digital plane preimage structure. *Discrete
   Applied Mathematics* 151(1–3):78–92, 2005. doi:10.1016/j.dam.2005.02.022. *Supports Proposition 4:* the preimage of
   a digital plane segment (planes whose digitization contains it) is a convex polyhedron.
9. L. Jaulin, E. Walter. Set inversion via interval analysis for nonlinear bounded-error estimation. *Automatica*
   29(4):1053–1064, 1993. doi:10.1016/0005-1098(93)90106-4. *Background for §6:* certified inner/outer
   approximations by box subdivision, as in the tau-pencil interval records.
10. J. R. Shewchuk. Adaptive precision floating-point arithmetic and fast robust geometric predicates. *Discrete
    Comput. Geom.* 18(3):305–363, 1997. doi:10.1007/PL00009321. *Background:* exact predicates. The generator uses
    exact rational fallbacks.
11. H. Santo, M. Waechter, W.-Y. Lin, Y. Sugano, Y. Matsushita. Light structure from pin motion: geometric point light
    source calibration. *IJCV* 128(7):1889–1912, 2020. doi:10.1007/s11263-020-01312-3. *Supports:* point-light
    position recovery from cast shadows, using pinhole/epipolar geometry, with known occluder structure.

The facts that cones are developable (tangent plane constant along rulings) and that the cubic lattice has covering
radius √3h/2 are standard. Proposition 6 and Theorem 1 prove what they need directly, so no page-specific textbook
citation is relied on.
