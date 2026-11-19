# Deterministic Local-first posteriors for finite RIND3D observations

This research labeler accepts one source-separated channel of a continuous polyhedral v3 world. The teacher knows the obstacle halfspaces, the local class array and its position. It starts from the unit cube. The generated source and prior posterior supports play no part in search or certification. The reference distributions in the experiment runner are opened after labeling.

## The target and the issued statement

Let F contain the sources outside the closed window, in the open world, at Euclidean distance strictly greater than 2⁻²⁶ from every closed convex component, that reproduce every FREE receiver. Segment contact, including tangency, blocks a response. The point tested at voxel i is (2i+1)/2048 in each dimension. Obstacles are closed intersections of exact affine halfspaces. The joint obstacle is their union. For positive feasible volume the target q is normalized Lebesgue volume pushed forward onto the 128³ grid. A zero-volume or unproved-positive feasible region cannot earn a posterior. The certificate concerns grid masses. Isolated feasible points and other null sets can be lost in this volume representation.

The maintained entry point is `experiments/credit_solver/cbc_run.py`. A successful envelope has format `astra-xlf-certified-interface/1`, status `CERTIFIED`, an input/build binding, sparse bytes and an upper bound on TV for the distribution obtained by decoding those bytes. An incomplete attempt carries diagnostics and withholds the probability vector. Malformed input or a failed integrity guard raises an error. A time budget is cooperative; a validation operation or final volume snapshot can finish after the deadline. The older experimental programs and their differently versioned envelopes remain historical references.

## How the local observation restricts source space

At a FREE receiver r, the inverse shadow of convex K consists of all sources whose closed segment to r meets K. The kernel derives this polyhedron by eliminating the segment parameter. An inactive receiver allows a union of inverse shadows. An active receiver excludes every obstacle's inverse shadow. Adjacent opposite-response endpoints often cut a much smaller source region than widely separated opposite-response endpoints. Search chooses pairs from the observed grid. Every pair is checked against its actual receiver classes. The kernel processes overlapping inactive explanations in obstacle order, subtracting earlier explanations to keep interiors disjoint. Adjacency itself grants no permission to discard a source. In particular it does not require the whole edge between FREE voxel centres to be physically FREE.

### A projective certificate for a whole mixed patch

There is an additional exact operation on a convex source piece P and a convex receiver box H. Write h for a facet's affine form with interior h ≤ 0. A row nonpositive at both endpoints of every segment in P × H can be removed for this test. A row strictly positive at both ends proves this obstacle misses. For the remaining rows, try one orientation and then, if needed, exchange source and receiver. A front row has strictly positive start h and a nonpositive end. A back row has strictly negative start. If signs are uncertain, this particular reduction can abstain.

A front row f supplies an entry time. A back row g with a positive end supplies an exit time. Comparing them with their positive denominators shows that entry precedes or meets exit exactly when

    D(s,r) = f(s)g(r) − g(s)f(r) ≤ 0.

A back row with nonpositive end imposes no earlier exit and satisfies this inequality automatically. Closed contact is included. Comparing all front/back pairs is sufficient because each row permits an interval in the one shared segment parameter. D is separately affine in s and r. All vertex-pair signs establish a universal sign throughout P × H by convex combination with positive homogeneous weights.

If exactly one possible blocking component remains and just one determinant can vary, the entire visibility test on this product is that single determinant. The reversed traversal uses f = −h_front, g = h_back, thereby expressing its determinant in source-first order again. The strict sign checks give f(s)>0 and f(r)<0. Writing u(z)=g(z)/f(z), the finite classified patch requires

    max u(active receivers) < u(s) ≤ min u(inactive receivers).

A missing class contributes no bound. Each attained extremum yields one affine cut in s. Along a receiver line segment with one denominator sign the ratio lies between the endpoint ratios. Iteratively discarding convex combinations of same-class receivers therefore preserves the extrema, even if a class is disconnected. This describes only the measured lattice samples. Equality on a nonconstant source plane has zero three-dimensional volume; a globally identically-zero strict constraint makes the piece empty and is handled separately. Several pieces can retain different generating occluders. Multiple possible blockers or unresolved facets prevent this particular chart from granting a collective proof.

The optional pass scans dyadic receiver blocks on up to three large source pieces selected *after* endpoint conditioning. Successfully proved bands narrow the piece. If some blocks are unresolved, all earlier receiver obligations remain on the narrowed piece. Only a cover of every relevant receiver can skip subsequent visibility checks. Source legality and clearance still apply. A local budget interruption leaves the old piece intact. Exact rank diagnostics measure the independence of selected cut normals on one source piece. Rank three in a finite observation can still leave strictly positive source volume.

## Why the released bound holds

The kernel begins with the whole world live. Each transition replaces complete handles atomically. Halfspace splits keep both children; clipped interfaces share only null boundaries. Exact union differences own their overlapping interiors once. Visibility screening either contradicts an observation throughout a piece, discharges an obligation throughout the piece, or preserves it. Children only inherit facts established on their containing parent. An outer clearance cover identifies a certainly legal part. An inner forbidden cover removes certainly illegal points. The intervening shell remains live. Search cannot assign a volume or issue a certificate.

An IN piece receives per-cell lower credits rounded down and upper credits rounded up in units of 2⁻¹⁶⁰. Exact octree splits integrate boundary cells. Whole grid blocks inside a validated convex polytope get their exact capacity. Conservation is checked against the piece's rational volume. Each unresolved piece is charged its whole volume rounded *up*. For cell i, let lᵢ be its lower IN credit and U₋ᵢ an upper bound on feasible mass outside i. The implementation sums upper IN credits and live charges, deducting known IN charges inside i and live charges wholly inside i. It may overcount uncertainty outside i but cannot undercount it under the frontier invariant. Then qᵢ ≥ lᵢ/(lᵢ+U₋ᵢ) for lᵢ>0. This realizes the volume-credit theorem without materializing an OUT object per cell.

The sparse vector p is encoded with unique sorted cell IDs and exactly 2⁵² total dyadic ticks. The bound uses the decoded p and rounds the overlap terms down before subtracting from one. The best completed checkpoint remains valid even if later refinement stops. The writer reads its temporary file back, verifies normalization, bound, bytes and binding, then atomically replaces the destination. The certificate assumes an authentic producing process. SHA256 binds identities and detects mismatch against an independently expected input; these are not signatures of authorship.

## Assurance boundary

| Layer | Evidence and remaining assumptions |
|---|---|
| Geometry and analysis | Exact inverse-cone and reciprocal projection derivations above, convex combination, disjoint partition and grid-TV theorem. Null-set simplifications apply to volume only. |
| Input | The canonical decoder interprets finite binary64 coefficients as exact rationals. Each row is positively rescaled to primitive integers; signs are preserved. Only full-dimensional convex components strictly inside the world are accepted. The mask is checked at every centre against closed continuous geometry using directed integer scanline bounds. Ambiguous lines are recomputed with arbitrary-precision integers. Offsets and classes are validated before narrowing. The container path checks direct FREE adjacency. Active/inactive values are the supplied observation X. Establishing their provenance from the *original* private source is a separate generation claim. |
| Exact splits, facets and credits | Python integers and Fraction implement the geometry and per-cell integration. The face ordering, crossing construction and frontier invariants use the existing code-level arguments in XLF_CBC_DESIGN.md and XLF_CBC_IMPLEMENTATION.md, with exact integer side and incidence tests in this branch. Topology and conservation are checked before crediting. No C++ volume routine is reachable from the maintained solver. The repaired GMP helper is an optional differential reference. |
| Fixed-width arithmetic | All decision screen coefficients are bounded by 2⁴⁵ and homogeneous coordinates by 2048. Dot products and row endpoints fit signed int64. At most 64 components contribute to a scanline. Ambiguous integer enclosures do not grant credit. SMT obligations cover arithmetic formulas and directed rounding, including four obligations for the projective reduction. They do not verify the executables. |
| Ownership | Kernel-owned input arrays, masks, derived envelopes, handles and ledgers. Returned hints are detached copies; immutable integer tuples may be shared. A proposal may name receivers or planes, never an IN or OUT label. The same-process model assumes ordinary code using the supported interface, no reflection, monkey patching, native memory corruption or concurrent mutation of code during execution. |
| Runtime | CPython integers/Fraction, NumPy, Numba/LLVM exact integer semantics, compiler/runtime, hardware, canonical codec and serialization remain trusted. Float/Qhull heuristics can select work. Trusted decisions must re-evaluate their supports with exact arithmetic or a directed enclosure. Source hashes and selected loaded binary fingerprints identify the numerical build. This is not a hermetic dependency closure or an end-to-end machine proof. |
| Independent evidence | Exact 214-cell reference, repeated rational rescaling, two legacy failure cases, closed-room 64³, missing sibling, malformed masks, extrema, thin/degenerate cuts, ambiguity and wire corruption tests. Comparisons with point samples or another bounded label are corroboration only. |

The per-label replay verifier is unnecessary for the mathematical implication under these implementation assumptions. For a controlled research release, retain independent differential checks on fixtures. Production adoption needs a pinned clean build, independent review of new trusted paths and larger held-out measurements. A replay run cannot compensate for a shared wrong physical target.

## Running the solver

Use the repository's RIND3D v3 Python environment with the canonical rind decoder, NumPy, Numba, SciPy and Zstandard. From the Phase 4 posterior package:

```sh
cd experiments/credit_solver
export PYTHONPATH="$PWD:$PWD/../../../RIND3D_Generator/src"
python cbc_run.py "$V3_WORK/sampling_impact/v3/scenes/0000/000004.npz" 0 0 /tmp/214.json --seconds 10
python -m pytest -q
python smt_screen.py
python smt_kernel.py
python smt_projective.py
```

Set V3_WORK to the existing v3 experiment-data directory. Use `--patch-depth 1` or `--patch-depth 2` to study collective projective cuts. Both use the same certification rules. To check a released file against an expected input, load the authoritative container with `inputs.load(path, window, channel)` and pass its returned identity with the parsed JSON to `evidence.check_artifact(record, identity)`. No FastLabel or private source is needed.

## Numerical details new to this branch

For one mask scanline and one halfspace (a,b,c,d), write t=a(2oₓ+1)+by+cz+2048d. Along local x the condition is t+2ax≤0. For a>0 the last possible x is floor(−t/2a); for a<0 the first is ceil(−t/2a). For a=0 the line is either entirely allowed or empty. Directed fixed-point rounding gives an inner and an outer integer interval. An empty outer interval proves emptiness. Equal nonempty intervals yield the exact indices. Any disagreement with nonempty outer interval is resolved against the original integer coefficients. The scan sums each closed body's exact interval and compares the resulting occupancy with the actual stored SOLID bits. FREE response values remain observations on which the posterior is conditioned.

The homogeneous volume computation uses the same convex-face fan as the independent Cartesian Fraction specification. For anchor O=(o,w₀) and vertex P=(p,w), subtracting Cartesian coordinates gives (w₀p−wo)/(w₀w). For a triangle with vertices A,B,C, its tetrahedron with anchor O therefore has volume |det(dA,dB,dC)|/(6w₀³wAwBwC), where dP=w₀p−wo. Positive weights ensure the denominator is positive. Faces containing the anchor contribute zero. After face validation the convex fan interiors are disjoint and fill the polytope up to null sets. The implementation performs the determinant in unbounded integers and accumulates exact fractions. This removes repeated fraction operations without introducing another arithmetic backend.

For an explicit example with three distinct projective explanations, the test `cyclic_hinges` uses a unit source cube and three cyclic rotations of a box with opposite corners (4,2,−1) and (6,3,2). Each receiver region applies the same rotation to a pair at (10,7,0) and (10,8,0). The setup can be affinely scaled inside the world. It is a geometric control, not a sampled production window. The exact surviving volumes after successive mixed constraints are 31/60, 43937/162000, and 491/3400. A repeated cut leaves volume unchanged. The three tested lower-bound normals have rank three, yet the third volume is strictly positive. A second possible blocker makes the one-blocker chart decline. These are direct counterexamples to treating plane count as source uniqueness or silently selecting a convenient blocker.
