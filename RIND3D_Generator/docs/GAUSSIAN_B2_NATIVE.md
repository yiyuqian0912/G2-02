# B2 native engine: audit and implementation plan

Frozen baseline: `bf4caf1`. Child branch `yushu/gaussian-b2-native`. Previous v1 code and evidence are unchanged.

## Measured before native implementation

Four saved B2 geometries, 32 fixed certified-FREE receiver rays each. cProfile and allocation tracing are separate from normal timings. Cumulative categories overlap; they cannot be summed as independent fractions.

| Case | Kernels | Normal median ms/ray | Profile segment ms | Coordinates cumulative ms | Coefficients cumulative ms | Score sum cumulative ms | Independent 8 rays ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_sum-000 | 21 | 0.263 | 90.27 | 17.63 | 49.42 | 4.22 | 153.23 |
| gaussian_sum-007 | 16 | 0.230 | 293.52 | 74.47 | 76.06 | 34.78 | 171.60 |
| gaussian_sum-014 | 11 | 0.782 | 251.43 | 66.64 | 71.24 | 29.47 | 415.59 |
| gaussian_sum-021 | 15 | 1.113 | 198.38 | 54.21 | 132.90 | 14.28 | 171.76 |

Python dispatch/generators, exact rational coordinate transforms, qbox construction and repeated witness membership are substantial. `_score_ranges` includes all exp enclosures AND Python summation yet is not the dominant cumulative category in the profiled case 14. cProfile does not separately expose Arb internal exp/allocation cost, so no fabricated exclusive percentage is assigned. Tracemalloc records Python live/peak objects only; internal GMP/Arb temporary allocation is unmeasured by it.

The old ray code already computes exact quadratic coefficients once. The avoidable repetition is mainly full coordinate/box evaluation of witnesses and Python temporary construction, not recomputing all ray coefficients at every existing subdivision.

## Minimal changes selected from that evidence

1. Add an isolated native module; leave original geometry, old v1 Python queries, references and reports unchanged. Packed immutable kernel inputs use exact binary64 values; GMP computes the inverse exactly.
2. Primary: outward binary64 algebraic filters, correctly directed MPFR exp endpoints, cached ray polynomials and exact-rational/high-precision fallback. System MPFR/GMP headers are installed; Arb development headers are absent. MPFR is a rigorous alternative, not unchecked libm. Preserve UNKNOWN/INVALID and closed contact.
3. Benchmark scalar and batch modes, coefficient caching, inner/outer ellipsoids and conservative grouped tail bounds separately. Retain any negative result.
4. Independent native path: separate cofactor inversion, high-precision natural-coordinate interval bounds, breadth-first subdivision and different region visibility certificates. No primary decisions/supports are reused. Both use MPFR elementary functions; disclose this common dependency and cross-check against old Arb as a third numerical path.
5. Hierarchical small-field producer: distinguish solid boxes from shadowed receiver boxes. Larger sizes are gated on complete independent smaller fields; never publish READY.
6. Only after correctness/performance gates, test joint center/scale/threshold calibration with globally rigorous containment. No new production profile or dataset.

## Numerical proof obligations

IEEE filters require nearest rounding, gradual underflow, no fast-math and no contraction; every elementary algebraic operation is outward enclosed. MPFR functions use RNDD/RNDU, including conversion. Gaussian tails remain present or have a positive certified aggregate remainder. A witness is proved, never inferred from a proxy sample. Tangent equality is handled analytically where reducible and otherwise stays UNKNOWN under bounded precision/subdivision.

Raw profiling: `outputs/gaussian-b2-native-v1/profile/`. The [measured results](GAUSSIAN_B2_RESULTS.md) record implementation, ablations and complete-field gates.

## Implemented numerical architecture

The implementation is in `experiments/gaussian_geometry/b2_native`. Canonical
binary64 centers, matrices and positive weights are copied into immutable native
storage. GMP Gauss–Jordan inversion gives exact rational transforms for primary
fallback; its fast transforms are outward binary64 enclosures of those rationals.
The independent implementation derives its inverse by a separate cofactor path.
Both retain the finite continuous positive sum, including every tail.

The primary uses cached quadratic ray coefficients, outward algebra and directed
MPFR exp endpoints. Strict upper `< tau` proves FREE/MISS; lower `>= tau` or a
proved point witness proves SOLID/HIT. It analytically handles the reducible
single-shape/symbolic-threshold case. General equality can remain UNKNOWN.
Exact-rational/MPFR fallback is triggered on uncertain points and sufficiently
deep ray intervals. INVALID means malformed/nonfinite query input, not numerical
exhaustion. The x86 rounding environment is checked, and fast-math/contraction
are disabled. MPFR's directed semantics are documented in the
[official manual](https://www.mpfr.org/mpfr-current/mpfr.html).

The independent path uses 256-bit natural bounds of local linear coordinates
along a ray and breadth-first subdivision. It never reads primary supports,
decisions or coefficients. It caches its own transform enclosures. Extremely
small terms can retain `[0, w*2^-floor(q_lower/2)]`, from `e>2`; the exponent is
capped conservatively to retain a positive remainder. Uncertain point decisions
retry full exponentials. A tail is never silently set to zero.

These paths share GMP/MPFR libraries and RAII storage; they do **not** have
independent elementary-function libraries. The original Arb implementation is a
third path and completely cross-checks selected 8³/32³ fields. That distinction
must survive any future production-certification proposal.

## Aggregate Taylor certificate

Let `q_i(t)=a_i t²+b_i t+c_i`, `g_i=w_i exp(-q_i/2)`. For real `t`,

`F'(t)=sum_i -(a_i*t+b_i/2)*g_i(t)`.

Cauchy–Schwarz gives `(q_i'/2)² <= a_i*q_i`, hence

`|g_i''(t)| <= a_i*w_i*(q_i+1)*exp(-q_i/2) <= 2*a_i*w_i`.

Thus `M2=2*sum_i a_i*w_i` is a global second-derivative upper bound. For an
interval centered at `m` with half-width `h`,

`F(t) <= F(m)_upper + |F'(m)|_upper*h + M2_upper*h²/2`.

A strict upper below threshold certifies the **whole** interval FREE. Signed
derivative contributions preserve cancellation between different kernels, which
independent term maxima miss. The primary encloses this with binary64 interval
algebra and MPFR exp; the reference separately derives exact derivative
coefficients and uses directed 256-bit arithmetic. Neither treats sampled
variation as a derivative or differentiates binary visibility. This is a
derivative of the analytic defining field only. The two implementations do not
share this decision code.

## Region and source certificates

Spatial q-bounds enclose every transformed coordinate of an arbitrary closed
box, then sum squared-coordinate bounds. Field bounds therefore certify whole
FREE or SOLID regions. Visibility is separate: a FREE hull containing every
source-to-box segment certifies visible receivers; primary shadow certificates
use one contained ellipsoid blocking all eight corners. Different occluders at
different corners do not suffice. Reference shadow certificates instead prove
an entire common ray cross-section SOLID under the full Gaussian sum.

A child may inherit a proved parent mask/visibility property; it never inherits
a property from the other verifier. Unknown children remain in the field and
need refinement/fallback. Outputs carry an independent validity mask. Strict
source clearance uses a whole outward-rounded epsilon cube containing the
epsilon ball (`epsilon=2^-26`), not sampled directions.

## Global containment and generation

`containment.py` first uses the unchanged rigorous exterior-halfspace bound.
If it is inconclusive, the positive-sum theorem gives finite outer supports;
their closed exterior portions are covered and refined until every box is
proved FREE. This proves absence of solid in entire exterior halfspaces, not
just sampled faces. The independent checker reconstructs the partition and
checks every leaf using its own MPFR bounds. The simple halfspace inequality is
also independently rebuilt using Fraction/MPFR rather than primary Arb.

A proved SOLID point on/beyond an inset face is a legitimate **violation
witness**. Candidate points use individual analytic plane minima, but their
actual rounded binary64 coordinates are rigorously rechecked. FREE face points
never certify containment. This distinction allows joint center/scale/threshold
calibration to reject real violations without mistaking budgets for physics.
If containment remains UNKNOWN, the case stops incomplete; it is not replaced
by a faster geometry. The new proposal distribution is versioned and is not
the v0.1 production distribution.

## Optional proof-cover experiment

An independently classified continuous spatial partition yields inner SOLID
boxes and an outer cover retaining every UNKNOWN/pending region. Dyadic cells
can be merged exactly into rectangles. These are auxiliary set bounds, never
sampled voxel truth. Independent canonical convex B can certify inner HIT or
outer MISS; all other pairs still query the Gaussian reference. The unchanged
64-component backend capability constrains the usable cover. Only bounded
segment screening is measured; no hybrid full-field or production integration
is claimed. See the results report for cold versus amortized cost.

Build instructions, actual APIs and all runnable commands are in the
[native experiment guide](../experiments/gaussian_geometry/b2_native/README.md).
