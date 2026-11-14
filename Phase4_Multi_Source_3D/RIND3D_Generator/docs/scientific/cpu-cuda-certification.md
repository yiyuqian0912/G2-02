# CPU / CUDA experiment: certification obligations

This experiment is separate from the production backend and publication path.
It keeps continuous canonical binary64 inputs as exact dyadic rational values.
It does not change the stored discrete response, RINDPack, informativeness
index, window selector, equivalence criterion, or READY restrictions.

## Scoped native geometry and direct clipping

Only finite top-level unions of canonical affine boxes or convex-halfspace
leaves are supported. The affine inverse is computed over rationals. The leaf
center must be a strict interior witness both for all local inequalities and
for the open world. Native GMP checks normal rank and the recession cone: a
nontrivial feasible cross-product extreme direction, or rank below three,
rejects the fast scope. Thus each component is bounded and full dimensional.
These obligations replace the predicate-related part of eager mesh/CGAL
initialization inside this experiment, not production's geometry acceptance,
clearance or generator certification. A component merely touching the world
from outside is deliberately unsupported. The canonical fallback stays lazy
and available; unsupported geometry uses the existing Oracle.

For a canonical plane `f_i(x) = n_i·x + d_i <= 0`, a segment from source p to
receiver x satisfies `a_i + t b_i <= 0`, with `a_i=f_i(p)` and
`b_i=f_i(x)-f_i(p)`, `0<=t<=1`. Exact source-plane values are cached once.
Interval clipping encloses each entry/exit parameter. A provably empty
parameter interval certifies MISS. A strict nonempty common interval certifies
HIT only when no sign-indeterminate constraint was skipped. At a point, an
indeterminate sign or endpoint/tangency case falls back to GMP rational
clipping with non-strict inequalities. Such contacts remain blocked.

For a receiver region, sign-indeterminate constraints can be omitted only
when seeking a conservative MISS proof: omitting a constraint enlarges the
feasible set. HIT for an entire region is established by all eight corners
being certified blocked by the **same convex component**. Its shadow is
convex. Corners blocked by different components never form that proof.
Visibility requires MISS for every component. UNKNOWN regions subdivide;
UNKNOWN points use exact predicates.

An additional projected control uses the unchanged exact Fourier–Motzkin
construction described in [the shadow design](shadow-volume-design.md).
It uses the new interval implementation but the same native traversal, making
it possible to distinguish geometric gains from faster arithmetic. Exact
positive-proportional duplicate planes are removed. The direct-clipping path
avoids the O(F²) projected plane set entirely; it does not claim a new minimal
silhouette algorithm. The old unmodified `B_native` remains the diagonal-box
baseline and is reported unsupported for general polyhedra.

CPU interval arithmetic requires round-to-nearest and gradual underflow with
FTZ/DAZ disabled. Each arithmetic operation is expanded by an adjacent binary64
value; bit stepping is tested against `nextafter`, including signed zero,
infinities and subnormals. Fast math and FMA contraction are disabled. Exact
zero products/coefficients can be skipped. NaN/indeterminate arithmetic widens
to the whole interval. Rational coefficient endpoints are independently
checked. Exact point clipping is GMP, not a tolerance decision.

## CUDA predicates

NVRTC compiles a separate PTX module; no CPU execution is presented as CUDA.
The host creates outward binary64 enclosures of every rational coefficient
and verifies the enclosure with Fraction comparisons before uploading.
The kernel uses explicit `__dadd_rd/ru` and `__dmul_rd/ru`. These operations
have documented directed rounding and are not merged into FMA.
[NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
documents the intrinsics; the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
specifies rounding and double-precision subnormal support.

The actual saved PTX must contain `add.rm/rp.f64` and `mul.rm/rp.f64` and no
float32 arithmetic, FTZ or approximate instructions. Compiler options explicitly
disable fast math, FTZ and contraction. On-device tests compare interval sums
and products with exact Fraction answers across random exponents, subnormal
inputs/results, cancellation and overflow. This tests the actual driver/JIT
and device, in addition to relying on the arithmetic specification.

All region coordinates are exact sample-center dyadics `(integer+1/2)/2^k`.
Coordinates are nonnegative and at most one, so sign-specialized interval
products are valid. Coefficients are normalized to absolute magnitude <=1;
four-term plane evaluations cannot overflow in normal rendering. The kernel
still refuses non-finite bounds and returns UNKNOWN. Underflow is handled by
directed double arithmetic, never by flushing a small term to zero.

A convex region is INSIDE only if every plane upper bound is <=0, OUTSIDE
only if some lower bound is >0. Otherwise it is UNKNOWN. Union semantics use
one whole-body inclusion proof or all-body exclusion proofs. UNKNOWN means
refine or invoke the **original canonical CPU oracle**, not infer a label from
a neighboring point. ABORTED/UNRESOLVED canonical results abort the experiment.
Exact non-dyadic silhouette contacts deliberately exercise this fallback.

### Explicit native fallback variant

The default `canonical_cgal` path remains unchanged. A separately selected
`fallback_backend="certified_native"` uses the already implemented complete
CPU convex predicate path for UNKNOWN points. This is CPU interval/GMP work,
not a GPU certificate and not a claim that CGAL ran. No approximate shape or
epsilon is involved. The prerequisite GeometryGroup proves an exact affine
inverse, bounded full-dimensional convex components with strict witnesses in
the world, and top-level union structure. Unsupported expressions cannot
construct this adapter. Because source and receiver are in the convex world,
their closed segment lies in the world; exact component clipping agrees with
world-clipped canonical truth under the strict-witness condition.

Each uncertain ray uses exact rational values of the original canonical planes
to determine the closed parameter interval `0<=t<=1`. A parallel constraint with
positive source value is infeasible; otherwise it adds no restriction.
Nonparallel constraints update entry/exit bounds, and equality remains HIT.
Membership uses the same non-strict original planes. These are complete
predicates within that scope, rather than a heuristic for a general oracle.

The variant is checked against the unchanged independent interval/GMP
implementation at **every** fallback point, plus every small-field voxel. The
32³/64³ near-coplanar fields also match the preserved original-CGAL fallback
fields, whose 134/534 uncertain rays were physically evaluated by CGAL. At
1024³ all 133,240 uncertain receivers were independently rechecked before
accepting the audited field. A second timed field without the extra experimental
checker is compared in its entirety with that audited field and the CPU field.
The original slow/incomplete CGAL experiments and v1 source fingerprints are
retained. This alternative does not implement geometry acceptance or open READY.

GPU work budgets are checked before and after each batch. A blocking canonical
call can still exceed the soft deadline, so a separate process watchdog imposes
the hard experimental bound. Exhaustion leaves an incomplete field; it is not
a physical MISS or a numerical UNRESOLVED result.

Each 64³ brick has levels 64/32/16/8, at most 585 nodes. Batches contain at
most 32 bricks. Final byte threads produce packed response, solid, and two
UNKNOWN bitsets. CPU fallback corrects every unknown bit before RINDPack write.
No 1024³ boolean tensor exists. Device buffers are reused and bounded; optional
source-independent masks use at most 128 MiB of packed host storage. Mask
values are reserialized into distinct source-bound files with unchanged CRC,
SHA and identity semantics, never hardlinked across sources.

## Independent physical evidence and scope of validation

Evidence is keyed by geometry, exact source bytes, canonical policy and input
field hashes. Each relevant brick is freshly physically rendered twice, using
different implementations, and every response and solid bit is compared to
the stored fields. A window can reference all intersecting verified bricks,
including partial and unaligned overlaps. This proves the whole window; no
probe-only equivalence or identity-source shortcut is used. A different source
requires different evidence. File hashes bind an already computed physical
result to the artifact; hashes alone never constitute physical verification.

Small complete fields and spatially stratified 1024³ regions are compared with
the unchanged interval/GMP parametric reference. That reference shares canonical
plane extraction with the prototype; it is **not** an exhaustive replay of CGAL.
Canonical CGAL boundary/random checks are therefore recorded separately.
Missing exhaustive canonical coverage or geometry-family acceptance remains
an explicit publication blocker, even if all sampled-field comparisons agree.

The 256-window diagnostic keeps 85/85/86 windows of sides 64/128/256 and one
verified true source per window (allowed by the 1–8 Lite target). Additional
source search is not measured by that demonstration. Private witnesses,
sources, evidence and selection metadata are absent from the default model
input. Strict dataset loading still excludes the INCOMPLETE diagnostic.
