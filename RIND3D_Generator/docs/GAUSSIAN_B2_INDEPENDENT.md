# Independent B2 ray verification

This study optimizes the verifier that independently renders the continuous
positive Gaussian-sum research geometry. The renderer classifies the closed
segment between two canonical binary64 coordinates. A point is solid when
the sum reaches the threshold. A HIT includes equality, endpoint contact
and tangency. A MISS requires a strict bound on the entire segment. The
experimental module has `ready_capable=False`.

## From the existing reference to the experiment

The original 256-bit MPFR reference derives each inverse matrix by exact GMP
cofactors. It caches outward inverse bounds, constructs each ray's transformed
offset and slope, and applies natural interval arithmetic to the sum of squared
affine coordinates. It computes two directed exponential endpoints per active
kernel or retains a strictly positive upper tail remainder. It searches its
own breadth-first interval partition. From depth four it prepares exact
rational ray derivative coefficients, evaluates a signed midpoint derivative,
and applies a global absolute second-derivative bound. Old box membership,
whole free hulls, continuous cover reuse and solid cross-sections certify
the same regions in every variant. The field experiment changes only the
residual independent scalar call.

For an interval `I`, write each transformed ray as `r_i(t)=U_i+t V_i`.
With `a_i=V_i·V_i`, `b_i/2=U_i·V_i`, `u_i(t)=a_i t+b_i/2`,
and `g_i(t)=w_i exp(-r_i(t)·r_i(t)/2)`, the signed derivatives are

```text
f'(I)  ⊆ sum_i (-u_i(I)) g_i(I)
f''(I) ⊆ sum_i (u_i(I)^2 - a_i) g_i(I)
```

The prototype lazily encloses the dot products from the reference's directed
MPFR offset and slope intervals. It bounds each product with directed
rounding and adds **signed** contributions before comparing against zero.
The nonnegative factor in each Gaussian product permits a sign-aware
two-product enclosure. A far-tail interval includes zero as a lower bound
and a positive upper bound. This preserves the tail's possible contribution
to both derivatives. Original exact cofactor inversion is shared within the
independent implementation. The prototype derives its coefficients, enclosures and decisions entirely
inside the independent implementation.

Let `m` be the midpoint, `h` the interval radius, `F` the upper bound
on `f(m)`, `D` the upper bound on `|f'(m)|`, and `K` the upper bound
on `f''(I)`. Taylor's theorem gives a whole-interval bound

```text
f(t) ≤ F + D |t-m| + K (t-m)^2/2
max_I f ≤ F + D h + max(K,0) h²/2
K < 0  implies  max_I f ≤ F + D²/(-2K)
```

The second upper bound for negative curvature maximizes the concave quadratic
over all real displacements, which includes `I`. A strict sign on the
whole derivative puts the maximum at the corresponding **closed** endpoint.
That endpoint still needs an independent physical enclosure. If curvature
excludes zero, the mean value theorem puts every stationary point inside
`m - f'(m)/f''(I)`. The optional localization attempt checks both
endpoints and the entire outward-rounded enclosure clipped to `I`. A
MISS at this stage requires the endpoint and stationary enclosure proofs.
The quarter-width criterion and four-attempt limit control the cost of
trying this method.

The midpoint and radius of a bisected `[0,1]` interval are exact dyadics
at the permitted depths. A contracted interval is only sent to the natural
bound, using outward MPFR-to-binary64 endpoints. Dyadic midpoint calculations are confined to
bisection intervals. Directed MPFR arithmetic encloses binary64
inputs exactly at 64 bits or more. Exponential bounds are directed, and
any underflow bounds retain the correct side of the exact positive value.
For finite nonsingular binary64 matrices, dyadic determinant denominators
and bounded input exponents also keep the polynomial arithmetic well inside
the default MPFR exponent range. When the sign or upper bound is
insufficient, the algorithm subdivides, retries in the original path within
its remaining budget, or returns UNKNOWN.

## Matched methods

| Method | Initial bounds | At deeper unresolved intervals |
|---|---|---|
| A | Original 256-bit natural bounds | Original exact-coefficient global Taylor and subdivision |
| B | 64-bit natural/point bounds through depth three | Rebuild line intervals at 256 bits and continue the original Taylor search |
| C | Original 256-bit natural/point bounds | Global Taylor with interval coefficients, then local signed bounds |
| D | B's 64-bit prefix | C's 256-bit local path |
| B128 / D128 | 128-bit initial precision | Same 256-bit continuation |
| C_Newton / D_Newton | Same as C / D | Optional stationary localization after the simpler tests fail |

At each interval, the natural decision is attempted first. In C/D at eligible
depths the natural evaluation retains the Gaussian terms. The midpoint test
can certify HIT before any derivative work. An inconclusive midpoint with a
positive tail gets the original full-exp retry. A midpoint already certified
below threshold needs no retry. The global Taylor test precedes local signed
derivative work. Local curvature may certify the entire interval or lead to
the endpoint test. Remaining intervals are bisected. A low-precision proof
is sufficient to discharge an interval. Escalation keeps the unproved
queue and constructs fresh 256-bit line enclosures from canonical inputs.
Previously certified intervals stay discharged. Analytic common-kernel
decisions use the original exact discriminant.

Node and time budgets span the low- and high-precision search. A full
original fallback in C/D receives only the remaining budget. A time limit
is checked between bounded operations. The current library call can finish
before the next check. INVALID, UNKNOWN and certified labels stay distinct.
A numerical proof from C/D can resolve some formerly UNKNOWN queries.
Resolving a query requires an actual numerical proof.

The scalar prototype uses the original independent geometry, GMP/MPFR and
options; the field harness uses the same certificate traversal and timings
for comparison. The third path is the unchanged Python/FLINT Arb evaluator
with separate rational transforms and subdivision. GMP and FLINT ultimately
share integer arithmetic foundations, and both verifiers share the canonical
mathematical inputs. These comparisons test implementation independence
and observed agreement. They are finite checks.

See the [reproduction guide](../experiments/gaussian_geometry/b2_independent/README.md)
and the [measured results](GAUSSIAN_B2_INDEPENDENT_RESULTS.md).
