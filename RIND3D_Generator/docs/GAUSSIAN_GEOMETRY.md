# RIND3D Gaussian geometry research

This work lives in `experiments/gaussian_geometry` on the child branch
`yushu/gaussian-geometry`, based on `yushu/3d` commit `59e343a`. The existing
1,000-scene RIND3D v0.1 release, public reader, canonical engine, production
profiles, A/B verification and Phase 1/2/3 implementations are unchanged.
The original worktree and its uncommitted teacher study remain separate.
No research geometry is registered as a production family or published READY.

## What the four representations mean

**A: Gaussian-distributed convex objects.** Large-scale anchors cover the world
using randomized Latin hypercube placement. Each cluster has an anisotropic
covariance factor and a variable number of normal center proposals. Boxes,
thin boxes, prisms, wedges and convex polyhedra are built around the centers
using the existing canonical binary64 codecs. The normal distribution controls
placement; it does not define their surfaces. Center bounds and world fitting
condition the proposal distribution, and all rejected proposals are recorded.

**B1: independent Gaussian levels / ellipsoid union.** Each kernel has center
`c`, invertible binary64 matrix `M`, positive amplitude `w`, and
`q(x)=||M^-1(x-c)||²`. Its independent level is `w exp(-q/2)>=w exp(-1/2)`,
exactly `q<=1`. Amplitudes cancel in these normalized individual levels.
The union preserves overlaps and disconnected components without a mesh.

**B2: one threshold of a positive Gaussian sum.** The solid is
`sum_i w_i exp(-q_i(x)/2)>=tau`. It includes all nonzero tails. Different
kernels can jointly cross the threshold and make bridges where neither
individual level is solid. This is not B1. A controlled equal-amplitude,
equal-threshold two-kernel test demonstrates the difference. The distribution
study separately adjusts density, so its B1/B2 comparisons are not confused
with that controlled same-threshold experiment.

Kernel amplitudes are geometry parameters. They do not alter the light-source
intensity or the existing binary visibility semantics.

The actual `M` entries are exact binary64 inputs. Inversion for physical queries
is rational, so `A=M^-T M^-1` is exactly positive definite. A rounded NumPy
inverse is used only for explicitly approximate displays and density estimates.
General Gaussian thresholds contain transcendental logarithms; GMP rational
arithmetic alone does not decide them. The special one-kernel amplitude-one
threshold `exp(-r²/2)` reduces exactly to a rational quadratic when `r²` is
binary64. Both predicate paths exploit this reduction without rounding a log.

**C: a finite continuous Gaussian spectral field.** For fixed retained
frequencies and covariance parameters, independent normal cosine/sine
coefficients define a finite Gaussian field. Squared-exponential spectral
weights, anisotropic rotated frequencies and an optional second scale vary
correlation structure. A deterministic compact polynomial taper is part of the
model:

`E(x)=product_j max(0,1-((x_j-c_j)/r_j)^2)^2`

`F(x)=E(x) [bias + sum_k a_k cos(2*pi*f_k.x)+b_k sin(2*pi*f_k.x)]`.

Its conditional covariance is exactly
`E(x)E(y) sum_k alpha_k² cos(2*pi*f_k.(x-y))` before PRNG/binary64 realization.
It is a nonstationary finite conditional Gaussian model, not a grid
interpolation or an exact draw from an infinite stationary GRF. Random
hyperparameters form a mixture of such conditional models. Density-dependent
threshold selection further changes the obstacle distribution. The taper is
C1 and identically zero outside its declared box; it is not post-hoc clipping
of a previously generated solid. The old `random_field` is instead a
three-frequency cosine field with fixed amplitudes and random phases; the old
`smooth_blobs` is an ellipsoid union. Neither is silently relabeled Gaussian.

## Hierarchy and declared sampling choices

`configs/study-v1.json` exposes cluster and kernel counts, anchor ranges,
covariance factors, amplitude variation, small scales, world margin, six layout
styles, spectral parameters and proof budgets. Styles cover clustered,
dispersed, curved-chain, branching-anchor, elongated and multiscale proposals;
they are generation recipes, not guarantees of continuous topology.

The initial study uses 24 seeds per representation, six styles at each target
occupancy 0.1%, 1%, 5% and 15%. A 32³ ordinary-float grid chooses parameter
scales/thresholds. These become explicit continuous parameters. A separate 64³
grid estimates geometry quality. Neither grid certifies a solid, ray, boundary
containment or continuous component count. World fitting can limit a scale or
raise a sum threshold, creating a measured density deficit; it is never hidden
by clipping or retrying until an attractive picture appears.

## Continuous semantics and containment proofs

All solids include their threshold boundary. Finite positive Gaussian sums
are continuous, and at positive tau their superlevel sets are closed and
bounded in R³. This does not itself prove they lie inside the unit world.
For a kernel, Cauchy–Schwarz gives the exact lower bound on its quadratic over
an exterior coordinate halfspace:

`q_i(x) >= distance(center_i, halfspace)^2 / ||row_j(M_i)||²`.

An Arb upper enclosure of the positive sum on **each entire exterior
halfspace** must be strictly below tau. The six checks prove containment in an
inset world cube, globally, not just on a sampled face grid. B1 uses the
corresponding exact quadratic comparison. A reuses the original independently
verified convex interior and world-bound dual certificates. C uses the exact
compact taper support and positive threshold.

An additional positive-width SOLID cube proves a nonempty interior for each
retained implicit geometry. It does not prove global regular-closed topology,
manifold boundary, absence of critical-level pinches or full relative Ω-face
acceptance. Such obligations remain blockers for production certification.
“Self-intersection” of a display mesh is not a property of the implicit solid.

## Physical queries and sound acceleration

`queries.py` exposes membership, closed-segment intersection, conservative box
classification, world containment and sufficient strict source-clearance
proofs. Source clearance retains `2^-26`; an exactly evaluated free cube
containing that closed ball is a sufficient GT proof. It is not a complete LE
decision service. Sources are uniformly proposed and require both FREE and
clearance proofs. No teacher candidate minimum separation is introduced.

For B1, rational quadratic minimization decides closed-segment contact exactly.
For B2, each quadratic on `s+t(x-s)` has exact rational extrema on a parameter
interval. Arb encloses the exponentials and their positive sum. A strict upper
bound below tau proves MISS; a certified point or solid interval proves HIT.
Ambiguous intervals subdivide. C uses conservative continuous spectral/taper
bounds on the same closed parameter intervals. Node, depth, precision and time
limits produce UNKNOWN or BUDGET_EXHAUSTED, never an assumed physical label.

Finite support acceleration for B2 uses the theorem that a sum above tau has
at least one term above `tau/N`. Each resulting ellipsoid gets an outward
bounded AABB. These are outer bounds on the solid, **not truncation of the
Gaussian field**. Strictly contained single-kernel ellipsoids can additionally
certify occlusion. A whole receiver box is shadowed only when all eight corners
are shadowed by the same convex interior proxy. This uses convexity of a point
source's shadow of a convex body. Corners blocked by different bodies do not
justify a whole-box certificate. Unknown regions refine or use scalar queries.

Arb ball enclosures and exact inputs follow the [python-flint API](https://python-flint.readthedocs.io/en/stable/arb.html).
`reference.py` independently uses Fraction cofactor inverses, natural coordinate
hulls, directed [MPFR arithmetic through gmpy2](https://gmpy2.readthedocs.io/en/stable/contexts.html)
and a separate discriminant path for ellipsoids. It calls no primary predicate,
support classifier or proxy. Spectral reference bounds use independently
directed pi/trig evaluations plus a rigorous Lipschitz remainder. Extremely
small exponential tails retain a positive `[0,2^-4096]` enclosure, justified by
`exp(x)<2^-4096` for `x<=-4096`; they are not dropped. At much smaller thresholds
this coarse guard may leave queries unresolved. Integer/rational libraries and
canonical input parameters are common dependencies; visibility decisions and
transcendental numerical implementations differ.

These are mathematically justified experimental query paths with independent
regression evidence, not a new RIND3D publication certificate. The existing
production A/B implementations remain unchanged and are used directly for A.

## What is measured

The study records generation, setup, actual completed point/ray work, source
rejections, all unknown/exhausted queries, reference disagreements, CPU/wall
time, geometry bytes and process peak RSS. Component counts, thickness,
orientation, surface complexity and coverage are explicitly finite-grid
estimates. Full small fields save response, solid and validity separately; an
incomplete field is never called completed. Larger fields require a completed,
agreeing smaller independent comparison. 1024³ and production are disabled.

Window diagnostics use actual 1024-grid voxel centers in arbitrary 64/128/256
windows, exclude the source and require separately proved FREE lit/shadow
witnesses. They do not claim full window responses or fulfill 256-window
production quotas. For nonlocal shadow samples, a full segment HIT plus a MISS
on its exact rational window-entry-to-receiver suffix proves the blocker is
outside. A suffix HIT only shows an internal blocker exists; it does not locate
the first blocker. Long ray distance alone supplies no external-occlusion proof.

Run commands, resume rules and output interpretation are in the
[experiment guide](../experiments/gaussian_geometry/README.md). Measured results
and certification blockers are in [the results report](GAUSSIAN_GEOMETRY_RESULTS.md).
