# B2 fixed-source angular certificates: research plan and proof obligations

Baseline: `9bbd13dc9b46b25940afcba7c36826bc18b131e3` on
`yushu/gaussian-b2-native`. This is a separate experiment, not a replacement
renderer or a new geometry family. Canonical positive Gaussian sums, old B2
sources/results, RIND3D v0.1, Phase 1/2 and teacher WIP remain unchanged.
No push or production campaign is authorized.

## Audit and research mapping

Existing B2 already has continuous full-sum membership bounds, ray-volume FREE
proofs, common SOLID cross-sections, exact/conservative single-convex-body
shadows, and independently built FREE/SOLID/UNKNOWN proof trees. Its residual
leaf work dominates the two completed 128³ comparisons. Repeating those
same-body corner tests under a different name would add no capability.

The following are methodological influences, not imported numerical proofs:

| Primary source | Reusable idea | Existing B2 / proposed difference |
|---|---|---|
| [Schaufler et al., 2000](https://graphics.cs.yale.edu/sites/default/files/siggraph2000_consvol.pdf) | Interior blockers and conservative occluder fusion; hidden space can help aggregate occlusion. | B2 already proves continuous interior boxes. New work combines their directional coverage, without importing sampled occupancy or flood-fill as truth. |
| [Durand et al., 2000](https://people.csail.mit.edu/fredo/PUBLI/Sig2000/occluext.pdf) | Extended projections, depth-aware accumulation, hierarchical queries. | A fixed point source is simpler than their volumetric view cell. New chart tiles carry proved depth envelopes; their hardware rasterization is not our proof. |
| [Boulos, Wald, Shirley, 2006](https://graphics.stanford.edu/~boulos/papers/ia.pdf) | Whole-packet inclusion/exclusion and the distinction between sufficient culling and exact intersection. | Existing B2 already has interval packet tests. New union certificates partition direction space instead of requiring one blocker for a receiver box. |
| [Knoll et al., 2009](https://www.vacet.org/publications/Knoll_CGF-implicits-2009.pdf) | Inclusion-preserving interval/affine arithmetic and iterative subdivision. | Interval inclusion and bounded fallbacks already exist. Affine arithmetic is a possible residual study, not permission to approximate Gaussian truth. |
| [Aydinlilar and Zanni, 2023](https://www.sciencedirect.com/science/article/pii/S009784932300081X) | Asymmetric forward linear/quadratic inclusions and derivative bounds for grazing rays. | Existing B2 already has rigorous ray Taylor tests. A new numerical variant must outperform those, including preparation. Publisher full text was restricted; the indexed publisher abstract/sections were accessible. |
| [Flórez-Díaz and Sbert, 2010](https://recerca.udg.edu/en/publications/reliable-soft-shadows-over-implicit-surfaces/) | Certified beam exclusion can avoid many individual rays. | Reuse existing FREE ray-volume/outer-cover proofs first. A sampled soft-shadow estimate would not meet binary closed-segment semantics. The institutional abstract and indexed repository text were accessible. |

The strongest initial hypothesis is a **depth envelope over a certified union
of projected continuous inner boxes**, cached for a fixed source. This can
reuse evidence across receiver depths and certify cases no one blocker covers.
Its usefulness, construction cost and residual savings must be measured.

## Coordinates and sufficient shadow theorem

Use six closed dominant-axis charts. A direction is `d=(±1,u,v)` with the
appropriate axis permutation and `u,v in [-1,1]`. A point is `s+rho*d`, rho>0.
Here rho is a dominant-axis distance, **not Euclidean distance**. This avoids
normalization/square roots. The first-hit envelope in rho has the same monotone
visibility interpretation as the unit-direction T after multiplying by ||d||.

For an angular rectangle U and a certified continuous inner box Q, prove
that all four corner rays meet Q before depth H. The shadow of one convex Q
from a fixed point is convex; hence the rectangle `s+H*d(U)` is in that shadow.
Every direction in U has first-hit depth at most H. Neither corner samples of
Gaussian visibility nor corners blocked by unrelated bodies prove this.

Primary construction uses inward feasible clipping intervals for each corner
ray, with outward binary64 arithmetic. Reference construction independently
projects its own SOLID boxes into exact rational shadow halfspaces, and tests
the angular rectangle at a conservative far-depth plane. They do not exchange
trees, planes, maps or physical decisions. Closed source contact must be handled
separately; zero direction and uncertain chart ownership fall back.

Independence has an explicit scope: Gaussian bounds and geometric depth proofs
are different, while chart bookkeeping and the finite union-composition checker
are shared tested code. An independent exact rational rectangle arrangement
test checks that composition on a class with a complete continuum reference.
This is not a claim that two whole angular systems were independently coded.
The **unchanged** original primary and independent complete-field renderers
remain acceptance references, so the new shared composition is not its own
sole validation oracle. Old Arb also checks selected physical controls.

Partition an angular tile into four **closed** child rectangles with shared
faces. If every intersected child is proved shadowed before receiver depth,
their union covers the query directions. Different children may use different
Q and depths. A parent upper bound is the maximum of all child upper bounds.
Missing or budget-limited children remain UNKNOWN. No angular pixel sample is
extended into an unproved area. A tiny gap must prevent union certification.

Project arbitrary receiver boxes with outward division only after proving the
whole box lies in one chart. Use a lower bound on every receiver's rho when
checking H. A projected blocker behind any receiver cannot certify that region.
Cross-chart boxes subdivide or use the unchanged canonical fallback.

An inner set never proves visibility. Optional empty-prefix caching must use
the **complete outer cover**, including UNKNOWN and pending boxes. Prefix
claims are open at the first possible contact; equality is never visible.

## Query-driven refinement and ablations

Compare old fixed-depth F6/F7, adaptive spatial-only, single-occluder angular
reuse, multi-occluder angular fusion, and a hybrid. Start from the same depth-6
continuous cover. Refine only unresolved spatial/angle regions touched by
queries, with deterministic priorities and explicit node/work budgets.
Work estimates guide scheduling only; labels always require certificates.

The minimal spatial priority is touched-receiver count times box width, with
stable node-ID ties. It is not a calibrated cost/benefit or success-probability
model. Exp counts and actual residual timings are diagnostics, not assumed
derivatives of that priority. In the first prototype the angular cache keeps
the first useful single-body depth bound, then refines when a receiver is in
front of it. A far blocker can therefore make near-layer reuse unnecessarily
conservative; UNKNOWN still invokes the canonical path. Depth-ordered candidate
selection is a possible later improvement, not a correctness assumption.

Controls separate using all inner boxes, directional caching, genuine union
coverage, and the simpler change of querying the old outer cover at individual
residual receivers. The last is not a new mathematical certificate. A gain from
more boxes or finer query granularity must not be attributed solely to angular
fusion.

Retain every uncertain outer region after a budget stops. Existing proofs stay
valid when more inner boxes are discovered. Record setup, failed work, map and
cover sizes, Gaussian/exp counts, fallback time, output cost and memory. Compare
the actual original residual receiver set and its recorded cost, not just the
new aggregate certificate percentage.

## Acceptance sequence

1. A decisive continuous multi-blocker fixture and tiny-gap/depth/chart/contact
   counterexamples, independently checked with exact rationals.
2. Native isolated prototype and bounded small-field tests.
3. Four frozen 32³ comparisons, then four 64³ dual comparisons.
4. Bounded complete case 14 and case 7 at 128³, only after smaller gates pass.
5. If useful, arbitrary-origin bounded tiles in a 1024-grid world.

All completed response and solid bits must match existing evidence. UNKNOWN,
INVALID, numerical failure and time exhaustion remain distinct from labels.
Keep old Arb checks, exact source-clearance checks, and B2 ready_capable=False.
Report negatives and do not infer asymptotic scaling from these finite fields.

## Optional FREE prefix and numerical follow-up

The optional prefix experiment subdivides only queried chart regions, including
directions with no inner blocker. It caches a lower bound on the first possible
intersection with **any** non-FREE leaf of the complete continuous outer cover.
Primary computes a necessary interval for beam/box intersection. Reference
instead solves exact rational linear inequalities in rho. They inspect their
own cover, not the other implementation's results. At most 512 new prefixes
are constructed by default; exhaustion falls back. Refinement can improve a
cached prefix, but cannot invalidate an older valid bound. Equality is never
classified visible. The cover only spans the unit world: these FREE statements
apply to legal in-world source/receiver segments, which remain in that convex
world; no claim about rays beyond the world boundary is made.

For a Gaussian with A=M^(-T)M^(-1), z=x-c and g=w exp(-z^T A z/2),

`grad(g) = -g A z`,

`Hess(g) = g ((Az)(Az)^T - A)`.

On a ray z=z0+t*d, write q=a*t²+2*b*t+c. Then
`g' = -(a*t+b)*g` and `g'' = ((a*t+b)²-a)*g`. Summing signed derivative
enclosures before taking absolute bounds can retain cancellation between
kernels. A local Hessian/Taylor inclusion would still require rigorous
enclosures for the entire interval, every positive tail and transcendental
threshold. Sampled derivatives are insufficient, and a zero derivative is not
a FREE certificate.

The old primary and reference already use midpoint full-sum derivatives and a
rigorous global second-derivative bound. Before adding another numerical
engine, the bounded secondary study compares those unchanged tests off and at
depths 2/4/8 on identical expensive residual receivers. Preparation, failures
and resolved coverage must be included. A proposed local-curvature inclusion
is a next hypothesis, not an implemented or certified performance result here.

## Boundary and proof audit

| Condition | Required behavior | Implementation / check |
|---|---|---|
| Source on/in a closed inner box | Contact is blocking, including rays directed away. | `upper` handles rho=0 before projection; physical source-contact tests compare every field bit. Such sources are not admissible dataset sources. |
| Legal FREE source in an UNKNOWN outer box | No empty-prefix claim may omit t=0. | `excluded` retains that box; exact and interval prefix clips have lower depth zero. |
| Blocker behind receiver | Its footprint alone cannot prove shadow. | `query` compares outward receiver near-depth against a proved H. |
| Different blockers cover adjacent angular pieces | A complete closed union may prove shadow. | `covered` requires every intersected child; exact arrangement and universal two-blocker fixtures test the continuum. |
| Tiny uncovered angular gap | UNKNOWN, never hole filling. | The 2^-30 gap counterexample and complete exact rectangle arrangement checker. |
| Chart seam or zero source direction | No unsafe normalization or guessed chart. | `chart` proves whole-box dominance; otherwise old subdivision/fallback. |
| Pending spatial node or exhausted refinement budget | It remains in the outer set. | Cover leaves retain UNKNOWN; no proof is inferred from scheduling scores. |
| New inner boxes or refined outer boxes | Old sound evidence remains usable. | Old H still has its original solid witness; old L remains valid, and two valid lower bounds may be combined with max. |
| Changed source on a later call | Rebuild all source-dependent maps. | Map and Cover are local to one field/tile call; geometry alone is reusable. |
| General exact transcendental tangency | Preserve original UNKNOWN when no proof exists. | Unchanged native/Arb fallback; no epsilon or new equality heuristic. |

See [geometry.cpp](../experiments/gaussian_geometry/b2_angular/geometry.cpp),
[field.cpp](../experiments/gaussian_geometry/b2_angular/field.cpp), and the
[regression tests](../experiments/gaussian_geometry/tests/test_b2_angular.py).
The tests support the stated sufficient predicates; they are not a formal
verification of the C++ compiler, interval library or all possible inputs.
