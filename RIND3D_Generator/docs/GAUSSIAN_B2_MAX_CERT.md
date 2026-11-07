# B2 one-dimensional maximum proofs: design and acceptance plan

Base `d67215cce0f6dcfccb504813da5c1e52feb18547`, research branch
`yushu/gaussian-b2-native`. This is an isolated numerical experiment, not a
new renderer default, dataset family or publication contract. RIND3D, previous
B2 experiments and the Phase 3 teacher workspace remain unchanged. No push.

## Characterization before implementation

The unchanged native reference uses 256-bit natural-coordinate interval
bounds, breadth-first subdivision and, from depth four, a signed midpoint
derivative plus a **global absolute** second-derivative bound. The primary
already caches ray quadratics and uses directed binary64 algebra with MPFR
exponentials, rigorous Taylor bounds and GMP/MPFR fallback. Neither is a weak
sampling baseline. The original Arb implementation remains a third path.

Before implementing a new solver, 374 exact trace receivers were frozen from
cases 0, 7, 14, 21: 314 MISS and 60 HIT, with 126 development and 248 evaluation
rays. Selection spans expensive, median and cheap traces, approximate-margin
strata and HIT controls. Saved labels come from complete physical fields;
ordinary numerical extrema are only selection/diagnostic values. All source
coordinates here are the existing synthetic research sources, not private
production labels. Identities and exact binary64 coordinate strings are saved.

The original reference spends 2.386 s on those MISS rays, median 19 nodes,
95th percentile 151, maximum 209 and maximum depth 14. It executes 817,410 exp
endpoints on MISS and already passes 3,870 intervals with its Taylor test.
Timing-only instrumentation of a separate source copy gives 1.757 s natural
bounds, 0.744 s Taylor bounds, 0.098 s derivative preparation and 0.038 s line
preparation across all queries. Total instrumented time is 2.737 s versus
2.637 s uninstrumented (about 3.8% overhead, separate trials). The reference
always uses 256-bit arithmetic; this is not evidence of frequent precision
escalation. No arithmetic expression or decision path in the baseline changes.

A separate 192-bit Arb study on **32 development rays only** analyzes 1,053
natural-inconclusive parameter intervals. It excludes stationary points in
513 and proves strict concavity in 752. Median local signed curvature upper
bound / old global absolute bound is -0.00480. A diagnostic ratio of summed
absolute midpoint derivative contributions to their signed sum has median
10.59; it is a heuristic cancellation measure, not a proof margin. Floating
extrema indicate multiple peaks on 72/374 rays; 349/374 have at least two
material contributors near their approximate maximum. The smallest estimated
MISS margin is 1.16e-5, so separate exact-contact/adversarial tests are needed.

These measurements motivate local signed curvature and derivative/root
exclusion. They do not prove that sophisticated root isolation will be faster.

## Mathematical sufficient conditions

The canonical obstacle is the closed superlevel set of the **positive sum**
`F(y)=sum w_i exp(-||M_i^-1(y-c_i)||²/2)`. Matrices, endpoints, amplitudes
and threshold parameters retain their exact binary64 meanings; matrix inverses
are over the rationals. No Gaussian term or positive tail is discarded.

On the closed segment `r(t)=s+t(x-s)`, write
`q_i(t)=a_i*t²+b_i*t+c_i`, `v_i(t)=a_i*t+b_i/2`, and
`g_i(t)=w_i*exp(-q_i(t)/2)`. Then

```
f = sum g_i
f' = -sum v_i*g_i
f'' = sum (v_i²-a_i)*g_i.
```

All sums include signed terms **before** taking any absolute bound. A natural
interval upper bound below tau proves MISS on that interval. If an interval
enclosure of f' is strictly positive/negative, its maximum is at its right/left
endpoint. Derivative samples never prove monotonicity.

For midpoint m, half-width h, enclosures `f(m)<=A`, `f'(m) in D` and
`f''(I)<=K`, Taylor's theorem gives, for every z with |z|<=h,
`f(m+z)<=A+d*z+K*z²/2`, for some d in D. Therefore a safe upper bound is
`A+abs(D)*h+max(K,0)*h²/2`. If K<0, the whole-real-line maximum of each
concave quadratic gives the additional upper bound
`A+max(D.lo²,D.hi²)/(2*(-K))`. Take the smaller proved upper bound.
This uses a **local signed** full-sum curvature bound, distinct from the old
global `2*sum(w_i*a_i)` absolute bound.

For stationary localization, if `0 not in f''(I)`, every zero of f' in I lies
in `N=m-f'(m)/f''(I)` by the mean value theorem. Let `J=I intersect N`.
The maximum on I occurs at an endpoint or a stationary point in J. Thus
endpoints plus a bound on J suffice. An empty intersection excludes stationary
points; nonempty J is only an outer enclosure, not an approximate root accepted
as truth. Strict curvature means at most one root on that interval, not that
a root exists. Degenerate/multiple roots or failed contraction subdivide.
No bound on the number of roots of the whole mixture is assumed.

HIT requires a certified point/interval lower bound at least tau. MISS requires
strict upper < tau over a complete closed cover of [0,1]. Any budget, uncertain
inequality, equality or numerical failure remains UNKNOWN/INVALID as appropriate.
An optional old-oracle fallback is measured explicitly, not hidden as a success
of the new optimizer. Known analytic contacts retain the old exact reduction.

## Planned matched comparison and gates

Compare the unchanged optimized primary and independent baseline with:
derivative-guided partitioning; stationary localization; and a cheap-first
hybrid using local Taylor, derivative exclusion and stationary contraction.
Use bounded nodes/depth/time and deterministic priority order. Physical proof
criteria never depend on priority scores. Evaluate a development pilot first;
freeze settings before evaluating the independent ray subset. Report adverse
rays and ordinary-ray overhead, not only a favorable average.

A minimal MISS certificate records canonical geometry/threshold identity,
exact source/receiver identity and a lossless dyadic closed interval cover.
Each leaf names a supported enclosure operation; optional stationary leaves
also name the contracted outer interval and its local bounding operation.
Serialized numerical upper bounds are **claims**, independently rechecked.
The checker must validate all identities, finite endpoints, full coverage,
strict inequalities, operations and resource bounds. Overlap is allowed;
even one uncovered point or interval is a failure.

The checker will be separately implemented using exact cofactor inverses and
MPFR enclosures, not the primary optimizer/decisions or a trusted serialized
float. Shared GMP/MPFR libraries, canonical input schema and mathematical
identities are disclosed dependencies. Checking a primary-generated proof is
**not** the existing independent physical rendering contract. The unchanged
independent renderer remains required for scientific comparisons; its cost
must remain in current-contract totals. Alternative primary+checker economics
are a research proposal, never a silent publication-gate change.

The optional E ablation disables only the **initial two HIT witness probes**
at t=0 and t=1. It does not remove either endpoint from the physical segment or
from any MISS partition. Monotone/stationary proofs still evaluate endpoints
when required, and complete closed coverage remains mandatory. E was motivated
after the first full-field timing results; a further 512 cost-unselected,
nonoverlapping residual receivers were frozen before measuring E. This is an
explicit post-hoc optimization with a fresh query holdout, not a claim of
generalization to unseen geometries.

Only after positive ray-level results may an isolated field hook be tested on
32³/64³ case 14/7, followed conditionally by bounded 128³. Require unchanged
solid/response bits, no unresolved accepted voxel, original clearance and the
unchanged independent reference. No complete 1024³ field or READY scene.

## Minimal research certificate format

`rind3d-b2-miss-certificate-v1` is an experimental proof schema, **not a change
to the public RIND3D dataset format**. Its envelope contains:

| Field | Meaning |
|---|---|
| `geometry_id` | Existing lossless canonical parameter-record identity. The checker is separately constructed from the expected geometry. |
| `source_hex`, `receiver_hex` | Three canonical `float.hex()` strings each. The caller also supplies the expected query; stale coordinates fail. |
| `query_id` | SHA256 of canonical JSON containing geometry ID and exact endpoint strings. An identity is never physical evidence by itself. |
| `threshold` | `kind` (`binary64` or `exp`) and exact binary64 parameter `value_hex`. |
| `schema`, `enclosure` | Explicit supported format and enclosure-method versions. Unsupported fields/operations fail. |
| `partition` | Closed intervals and local proof operations, described below. |

Every leaf has `lo`, `hi`, `upper` in canonical finite binary64 hex and an
`op`: `natural`, `increasing`, `decreasing`, `local_taylor` or `stationary`.
Binary64 numbers are exact dyadic rationals; no decimal approximation of a
partition boundary is used. Natural/derivative/local Taylor bounds are
reconstructed independently. The recomputed upper bound must be no larger than
the serialized claimed upper, and **both** must be strictly below the independently
enclosed threshold. Arbitrary small serialized bounds are not trusted.

A stationary leaf also contains `jlo`, `jhi`, `empty`, `inner_op`. The checker
recomputes its own interval Newton enclosure and proves that it fits inside the
specified J, or proves empty intersection when `empty` is true. Nonempty J is
bounded with `natural` or `local_taylor`. Empty J uses canonical dummy endpoints
equal to `lo` and `inner_op=natural`; it is not a zero-width physical interval.

The checker sorts leaves, rejects every uncovered gap (including a single
binary64 step), and requires coverage of exactly [0,1]. Valid overlaps and
duplicate valid leaves are allowed. Limits are 8 MiB serialized JSON, 8,192
leaves and 80–1,024 checker bits, with a separate time budget. Duplicate JSON
keys, nonfinite or noncanonical encodings, malformed/stale identities,
unsupported evidence, insufficient numerical enclosure and unresolved work
fail closed. Rejection means **proof not accepted**, never physical HIT.

The primary encloses rational-inverse ray coefficients in outward binary64
intervals and uses directed MPFR exp at 53 bits. It computes no approximate
Gaussian mesh or clipped tail. The checker instead constructs its own exact
cofactor inverse and rational ray quadratics, completes the square exactly,
and evaluates independent MPFR intervals at 256 bits in the matched comparison
(128 bits is a separate ablation). Its checking code does not invoke the
primary decision routine. Both share GMP/MPFR, canonical input definitions,
RAII storage and the floating-point environment guard. This is a small audited
numerical kernel with adversarial tests, not a formally verified compiler or
an assertion of complete independence from every common software dependency.

For fields, an isolated generated hook changes only the residual primary ray
call in the existing region traversal. Membership, continuous cover/region
certificates and the original independent renderer stay unchanged. Optional
native checking uses one temporary proof at a time and records exact checked
counts; it is not JSON serialization or a portable full-field proof. Region
fills, mask classifications and HITs are not silently claimed as MISS-proof
evidence. Field timings and ray serialization timings must not be confused.
