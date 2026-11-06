# B2 receiver-region research: plan and proof obligations

Baseline: `bd5e0d7956d896eb1a9f4f20334fdb9ad07bc547` on
`yushu/gaussian-b2-native`. Old native source, binaries and measurements remain
unchanged. This work is isolated in `experiments/gaussian_geometry/b2_regions`
and `outputs/gaussian-b2-regions-v1`. No B2 production capability is enabled.

The [measured decision report](GAUSSIAN_B2_REGIONS_RESULTS.md) and
[experiment guide](../experiments/gaussian_geometry/b2_regions/README.md)
contain results and runnable commands.

## Audit and minimal plan

The existing primary uses a whole source/receiver AABB FREE test and same-inner-
ellipsoid corner shadow tests. The independent path uses its own MPFR box
bounds and up to three common SOLID cross-sections. Both repeat substantial
Gaussian arithmetic at small receiver regions. Membership and visibility are
already correctly distinct; their old decisions and scalar fallbacks remain.

Compare unchanged scalar/traversal baselines against (1) subdivided ray volumes,
(2) full-sum cross-sections, and (3) reusable continuous membership proofs.
The cache hypothesis is specifically amortization across a complete field,
unlike the previous negative cold 256-ray proxy experiment. Do not change the
generator distribution. Test a sound fast-exp filter separately, after region
correctness and work accounting are established.

## Required mathematical conditions

For receiver box B, a FREE partition of the complete parameterized ray volume
`{s+t(x-s): x in B, 0<=t<=1}` proves all receivers visible. Every parameter slab
must be covered; one UNKNOWN slab prevents the certificate. A full-sum SOLID
cross-section at a single common t proves all rays blocked. Merely blocking
different corners with different obstacles does not suffice.

A continuous proof tree stores FREE, SOLID and UNKNOWN boxes. The union of
SOLID leaves is an inner set; the union of SOLID and UNKNOWN leaves is an outer
set. Unprocessed budget-limited leaves remain UNKNOWN. These are mathematical
inclusions established by continuous box bounds, not sampled voxel truth.
Whole-ray-volume exclusion from the outer set proves visibility; inclusion in
the shadow of one certified convex inner box proves occlusion. Any missing
proof falls back or remains UNKNOWN. Domain coverage must be checked explicitly.

Primary geometric screening will use outward parametric clipping. Independent
screening will eliminate the ray scale using exact rational halfspaces; it
will construct its own tree using its own Gaussian bounds. No physical labels,
trees or source-dependent classifications cross between implementations.
Shared traversal and scalar types are bookkeeping/numerical primitives only.
Finite-scale projection describes positive ray parameters. The independent
FREE test must separately retain closed t=0 contact whenever its proof box
contains the source; a boundary source directed away from that box is an
explicit regression case. No source-validity assumption hides endpoint contact.

The exponential experiment must derive an enclosure analytically, including
roundoff and positive underflow remainders. Ordinary libm agreement is not a
proof. Any filter must retain MPFR fallback for undecided physical comparisons.

## Acceptance sequence

Prove/test region predicates and small controlled counterexamples. Compare
fixed regions and exact saved rays with cold setup included. Complete four
32³ comparisons, then four 64³ comparisons. Only after those gates attempt
case 14/128³ under a declared bounded budget. Preserve incomplete coverage.
Compare exp calls, rays, certificate sizes, setup, wall/CPU and memory. Do not
claim a scaling law from a single timing or partial field.

The final report will answer the 18 questions in this iteration's request,
preserve negative ablations, and state which predicates are sufficient but
incomplete. Tangency/equality, source validity, clearance, continuous truth and
READY restrictions retain their original semantics. Results are intended for
a later local merge into `yushu/3d`, preserving its teacher WIP; no push.

## Implemented sufficient predicates

The Gaussian field describes the **obstacle**, not a light intensity or
opacity. The output remains binary point-light visibility and a separate solid
mask. Every bound below concerns the finite continuous sum; no kernel is
silently removed.

For a primary kernel, cache exact-rational inverse coefficients as outward
intervals. Write its local coordinates on a receiver box as
`u_i + t*v_i(B)`. On each parameter slab `[a,b]`, interval multiplication
encloses all these coordinates, then their squared norm and positive Gaussian
contribution. Summing the enclosures yields bounds for every ray in the slab.
If all closed slabs are strictly below tau, all rays are FREE. Budget or depth
exhaustion in even one slab returns UNKNOWN. A wholly SOLID slab itself proves
occlusion. The independent path instead encloses the slab in world coordinates
and invokes its own 256-bit natural-coordinate Gaussian bounds.

The fusion predicate proposes a finite set of common parameters from kernel
centers and fixed fractions. These are proposal heuristics only. At each
actual binary 64 t, it proves the entire continuous cross-section SOLID using
the **sum**. A failure to find such a t proves nothing. Both methods can prove
a bridge that is outside every individual kernel's shared-threshold ellipsoid.

For proof reuse, recursively partition `[0,1]^3` into closed dyadic boxes and
classify each with the chosen implementation's continuous Gaussian oracle.
FREE leaves are excluded from the outer cover; SOLID leaves enter the inner
set; UNKNOWN and unprocessed leaves remain in the outer cover. Source/receiver
rays must remain in this covered domain; otherwise the acceleration is disabled.
The inner subset may be bounded for performance without deleting any outer
uncertainty. Shared faces do not create gaps. A whole receiver box can inherit
only a proved property from its own backend.

For primary FREE screening against one outer box `[a,b]`, existence of a ray
intersection requires simultaneous linear inequalities
`s_j+t*(receiver_lo_j-s_j) <= b_j` and
`s_j+t*(receiver_hi_j-s_j) >= a_j`, with `0<=t<=1`.
Outward clipping encloses every feasible t. Empty intersection proves exclusion
of the entire ray volume. To prove shadow from a certified inner box, a
conservative inner parameter interval proves each corner ray HIT against that
same box. The shadow of one convex body is convex for a source outside it, so
the receiver box is included. If the source is in the body, every closed ray
already contacts it. Corners blocked by unrelated bodies are never combined.

The reference independently eliminates lambda from
`lambda>=1` and `lambda*(a-s) <= x-s <= lambda*(b-s)` using exact GMP rational
Fourier–Motzkin projection. It caches outward enclosures of the resulting
halfspaces. A strict separating halfspace proves receiver-box exclusion from
that shadow; containment in all halfspaces proves shadow from one inner body.
This separation test is sufficient, not a complete polyhedral disjointness
solver. Closed source contact is handled separately as stated above. Reference
trees, planes and Gaussian classifications are never supplied by the primary.

## Fast exponential enclosure

The optional primary region filter evaluates `exp(-q/2)` for `q>=0`. Reduce
`u=q/2^(m+1)` to at most 1/32 (and verify its outward upper bound is at most 1/16).
For `P12(u)=sum_(j=0)^12 u^j/j!`, positivity gives

`P12(u) <= exp(u) <= P12(u) + (u^13/13!)/(1-u/14)`.

All polynomial operations, remainder arithmetic and reciprocals are outward
binary 64 intervals. Invert the positive enclosure and square m times. The
procedure needs no assumption about `std::exp` and no empirical ULP constant.
For `q>=2148`, `e>2` proves a positive upper remainder at most 2^-1074;
underflow never means that a finite Gaussian tail is zero. Exact q=0 is one.
The floating-point environment and compiler restrictions are the same as v1.
Positive-exponent/overflow diagnostics use directed MPFR rather than this
nonpositive-domain filter. NaN is invalid. Uncertain physical comparisons
retain the existing MPFR/GMP fallback. The independent geometry verifier does
not use this new polynomial code.

Finite numerical tests use separate 256-bit directed MPFR enclosures and
include subnormals, underflow, overflow, tiny margins and unresolved equality.
Such tests supplement the analytic remainder proof; they do not establish a
universal correctness guarantee for libm. Matched timing compares equal final
resolved statuses, including fallback, separately from enclosure-only costs.

## Bounded tiles

The diagnostic field gate remains at 128³. A separate tile API allocates only
the requested tile (at most 128³) and uses exact integer XYZ origins in a
power-of-two world resolution up to 1024. Receiver positions are still canonical
voxel centers `(global_integer+.5)/world_resolution`; geometry is independent
of that sampling. Unaligned tile tests compare directly with complete small
fields. This is a bounded output primitive, not a 1024³ streaming producer,
RINDPack integration or production-ready resume system. Each current tile call
includes its own cold proof-tree setup; no amortized cross-tile result is claimed.
