# B2 interval-local precision

## Measured diagnosis before the solver change

D retains its breadth-first queue. At the first eligible deeper node it
rebuilds every kernel's offset and slope at 256 bits. Closed FREE intervals
have already been removed, and the retained node coordinates are the same
exact dyadics. The configured full fields reach the transition at depth
four. The distinct depth-limit retry for a shallower query can reevaluate a
node, and the exceptional original-A fallback can restart the whole ray.
The six audited fields had zero such original-A fallbacks. The timed
residual sample observed zero depth-limit retries.

D does no 64-to-256 conversion and no per-ray exact rational coefficient
construction at the transition. Each rebuild loads the original binary64
coordinates and evaluates the six directed affine transforms per kernel
with cached independent inverse enclosures. The six affines load 36
binary64 endpoints into MPFR per kernel line, including repeated loads
for equal lower and upper endpoints. That is 22,952,376 repeated loads
on the four audited 64³ fields and 67,573,368 on the two 128³ fields.
The timings charge these loads within the transforms. Source offsets
recur across rays. The vector push copies the already computed line at the same
precision and allocates another set of MPFR objects. Derivative dot
products are evaluated lazily in interval arithmetic.

| Workload | Exact escalations | Exact repeated kernel line preparations | Estimated high line preparation s | Source offset subset s | Vector copy subset s |
|---|---:|---:|---:|---:|---:|
| Four complete 64³ | 41,618 | 637,566 | 4.20 | 1.82 | 0.26 |
| Two complete 128³ | 135,693 | 1,877,038 | 12.27 | 5.33 | 0.78 |

Exact totals come from the preserved complete D traces. Timings use 512
random residual rays of each physical label per field, weighted by the
respective full-field counts. Timing instrumentation is pinned to one
logical CPU and adds roughly 3.5–4.6% to the sampled query time. Offset
and slope measurements include directed loads and endpoint stores.
Copy timings include MPFR allocation and same-precision copying.
The remaining preparation time includes temporary initialization and
destruction, scratch and vector allocation, and loop overhead. In the
128³ sample, 256-bit exp endpoints account for an estimated 61.63 s
of scalar work, and 256-bit derivative preparation about 1.62 s.

A separate diagnostic re-ran the 64-bit natural bound on 3,745 actual
post-transition D intervals. Its result agreed with the 256-bit natural
decision for every probe, including 1,810 certified FREE intervals.
The high-precision natural and midpoint tests evaluate different
supports. D already reuses its natural Gaussian terms in the signed
bound on the same node. The opportunity is to use the sound low
natural enclosure *instead of* a repeated tighter enclosure and to
avoid rebuilding the line unless actual numerical ambiguity demands
it. This is a finite observation and motivates a measured ablation.

E will retain the low enclosures and run the independent 256-bit
midpoint and signed bounds on the remaining nodes. F will additionally
test source-offset and line allocation reuse, whose costs are visible
in the audit. On each tested grid the adoption decision will use
complete-field wall time, agreement, ordinary-ray behavior and peak
RSS. The deeper gate requires a consistent benefit on the frozen and
held-out 64³ workloads and a realistic measured time and memory cap.

## Enclosures and queue invariant

E uses the same 64-bit natural bound on each pending closed interval. A
strict upper bound below the threshold closes that interval. A sufficient
lower bound proves HIT. When the bound is inconclusive, E keeps its
individual Gaussian enclosures and uses the independent 256-bit
midpoint, derivative and curvature arithmetic. Coefficients computed
from the 64-bit offset and slope intervals enclose the exact coefficients.
MPFR accepts their endpoints as exact numerical operands in directed
256-bit operations. The source of uncertainty is the width of each
enclosure, which persists through all these operations.

Each cached Gaussian enclosure is tied to the same ray and exact
interval. It supplies the nonnegative factor in the signed derivative
products even if the transform is later tightened. Positive upper tails
and their lower bound of zero enter these products as before. A midpoint
still straddling the threshold after the full exponential retry triggers
a fresh 256-bit line. This refreshes its local derivative state. A
terminal inconclusive interval receives a high-precision full natural
bound. If the whole query remains unresolved, the original independent
solver can use the remaining time and node budget on the original ray.

The queue stores only three machine values per pending interval.
Its endpoints and midpoint are exact binary64 dyadics under the depth
cap. Closing a FREE node removes it permanently. Bisecting replaces a
closed interval with its two closed children, with the midpoint present
in both. The union of pending and discharged intervals therefore stays
equal to the original closed segment. An integer sum of the discharged
dyadic lengths is checked before returning a whole-ray MISS. An
uncertain result retains UNKNOWN when this invariant or any proof is
insufficient.

F applies E with one field-local set of 64-bit source offsets and direct
construction of each ray's line vector. Offsets are bound to the exact
source, independent engine and precision. The receiver transforms remain
ray-specific. F constructs the same outward offset endpoints as E and
copies them into the line with directed rounding. The source cache
contains O(kernel count) MPFR endpoints and serves only that field
invocation. Scalar microbenchmarks use E because they lack the
within-field source scope.

The previous A, D and Arb implementations remain available. The
experimental additions use the same continuous Gaussian-sum parameters
and require independent numerical proofs for all labels. The output
remains research-only with `ready_capable=False`.
