# XLF CBC labeller: implementation assurance (v2.9)

Code in `experiments/credit_solver/`, plus the exact geometry it reuses from `experiments/global_inverse/` and
`experiments/xlf_assurance/`. Labels as defined in `XLF_CBC_DESIGN.md` (MATHEMATICALLY PROVEN, IMPLEMENTATION-VERIFIED,
EXACT-DIFFERENTIAL-TESTED, RUNTIME-CERTIFIED, EMPIRICALLY TESTED, CONDITIONAL, OPEN).

```sh
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/credit_solver
export PYTHONPATH=".:../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K" OMP_NUM_THREADS=1
native/build.sh "$(which python)"                                         # optional GMP helper
python cbc_run.py CONTAINER.npz WINDOW CHANNEL OUT.json --seconds 10      # one label
CBC_VERIFY_SPLITS=1 python cbc_run.py ...                                 # with per-split certificates
python batch.py OUTDIR ../global_inverse/tasks.json:holdout ... --runner cbc_run.py   # benchmark
python compare.py OUTDIR                                                  # reference comparison
python chaos.py ROOT KEY --seeds 3 --seconds 20                           # search-independence test
python audit_exact.py CONTAINER.npz W S OUT.json --seconds 8              # exact point audit
python smt_screen.py                                                      # z3 proofs
python -m pytest -q test_cbc.py test_wire.py test_poly_exact.py ../global_inverse/test_global.py \
    ../xlf_assurance/test_assurance.py
```

## 1. Trusted computing base

| module | role | trusted? |
|---|---|---|
| `inputs.py` (Astra) | decode one v3 channel: canonical integer obstacle rows, [x,y,z] class array, origin, identity | yes (input) |
| `cbc_kernel.py` | state, transitions, release, `certify` driver, envelope assembly, `separators`, octree credits, encoder, snapshot | **yes** |
| `polycheck.py` | structural certificate `check`, per-split certificate `verify_split` (`check_reference` = test oracle) | **yes** (a checker) |
| `screen2.py` | Numba integer screens, candidate masks, range guard | **yes** |
| `native/cells.cpp` (optional; `CBC_NATIVE=0` disables) | GMP octree cell credits, the same algorithm as `_grid_credit_python` | yes when enabled |
| `global_inverse/poly.py` (Astra + filter, subnormal guard, split hook) | exact homogeneous-integer polytopes | **yes** |
| `global_inverse/solver.py`: `cone`, `apply_pair`, `subtract_many`, `Observation.cleared`, `exclude_window` (Astra) | exact inverse cones, pair conditioning, exclusions | **yes** |
| `global_inverse/envelopes.py`: `Envelope`, `clearance_partition` (Astra) | guaranteed and possible shadows; clearance | **yes** |
| `global_inverse/raster.py`: `fixed` (Astra) | directed fixed-point rounding | **yes** |
| `global_inverse/penumbra.py` (Astra + axis fallbacks) | cone-facet proposals, with exact validation inside | validation only |
| `xlf_assurance/ledger.py` `encode`/`decode`, `kernel.py` `wire_bound`, `ceil_mass`, `single_cell` (Astra) | exact quotas, strict decoding, integer TV bound | **yes** |
| `shadow_facets.py`, `hull_proposals.py`, `cbc_search.py`, `solver.choose_pairs` | proposals and policy | **no** (outputs are validated or re-proved) |
| `run.py` writer, `cbc_run.py`, `batch.py` | atomic writer that re-reads the bytes; CLI | writer re-checks |
| `chaos.py`, `exactcheck.py`, `audit_exact.py`, `compare*.py`, `diag.py`, `hunt.py`, `smt_screen.py` | independent instruments | no (not in the certificate path) |

The trusted base is about 1,700 lines of Python and Numba (plus 360 lines of optional C++). Search code (about 400
lines) can change freely without affecting soundness.

## 2. Code-path review: every operation that can create credit or discard mass

This review was done in full for v2.9 (all files above). For each operation, it says why the operation cannot
create unjustified feasible mass or discard unresolved feasible mass.

| # | operation | code | justification | status |
|---|---|---|---|---|
| 1 | input decoding | `inputs.load` | SOLID ∧ active rejected; window/channel eligibility checked; class array read-only | CONDITIONAL on the container decoder and canonical geometry (Astra's re-render: 229 M voxels agree) |
| 2 | obstacle rows | `certificates.integer_row` | exact rationals of the canonical binary64 parameters | reviewed (Astra); exact |
| 3 | window and obstacle exclusion | `Kernel.exclude_window` → `Observation.exclude_window`, `subtract_many` | F misses the closed window and closed obstacles; exact closed differences | MATHEMATICALLY PROVEN |
| 4 | inverse cone C_K(r) | `solver.cone` | elimination of t ∈ [0, 1]: facets with f(r) ≥ 0 and the pair rows f_i(r) f_j + \|f_j(r)\| f_i (re-derived in this review); a receiver in or on K raises (abstain) | MATHEMATICALLY PROVEN; EXACT-DIFFERENTIAL-TESTED (`test_cone_cuts_match_reference`, exact segment tests in `test_cbc`) |
| 5 | pair conditioning | `apply_pair` | equals P ∩ ⋃C(d) ∖ ⋃C(a) as a first-cover disjoint union | MATHEMATICALLY PROVEN; EMPIRICALLY TESTED (exact audit of discarded points) |
| 6 | polytope split, `within`, `without` | `Poly.split` | Lemma V + FS (design §4g); children's vertices lie in the parent, on the correct side, by construction | MATHEMATICALLY PROVEN; EXACT-DIFFERENTIAL-TESTED (`test_poly_exact`: independent half-space vertex enumeration); RUNTIME-CERTIFIED per split with `CBC_VERIFY_SPLITS=1` |
| 7 | sign filter | `Poly.side`, `_fsign_terms`, `polycheck._sign_matrix` | error ≤ 6u Σ\|terms\| ≪ margin 10⁻¹²; subnormal coordinates, non-finite or tiny sums fall back to exact (v2.9a fix: subnormal vertices previously had no relative error bound; reachable only with split rows ≳ 2¹⁰⁰⁰) | MATHEMATICALLY PROVEN (error analysis); payload byte-identity with the filter off |
| 8 | guaranteed shadow | `Envelope.inner`, from non-contact vertices only | intersection of S_K(v); tightest-pair pruning exact given f_i(r) ≤ 0; vertices in K impose nothing (design §4f) | MATHEMATICALLY PROVEN |
| 9 | possible shadow | `Envelope.outer` + `penumbra.directions` + `separators(mixed_candidates)` | facet/pair rows (design §4); every row that separates K∩cube from P is valid (blocking point lies in the cube); mixed facets complete O = K + cone(K−P) (Lemma O) | MATHEMATICALLY PROVEN; validation exact in the kernel |
| 10 | directed fixed point | `raster.fixed` | rounding up shrinks guaranteed, down grows possible, for positive receiver coordinates | IMPLEMENTATION-VERIFIED (z3) |
| 11 | screens | `screen2._intervals`, `scan_collect`, `scan_ids` | exact scanline endpoints, int64 range, range guard `check_ranges` | IMPLEMENTATION-VERIFIED (z3: endpoints, int64 range) at source level; EXACT-DIFFERENTIAL-TESTED (pointwise int64 reference); CONDITIONAL on Numba compiling int64 `//` as floor division |
| 12 | candidate masks, subset screens | `_mask`, `remap_masks`, `encoded_subset(rel)` | a body outside m(r) cannot block r on the piece or any subset | MATHEMATICALLY PROVEN; tested |
| 13 | inherited obligations | `Piece.pending/done/masks` | monotonicity of both shadows in P; arrays read-only (v2.9a) | MATHEMATICALLY PROVEN |
| 14 | exact enforcement | `Kernel.enforce`, `_exact_cut` | exact cones over the candidate bodies; dark: first-cover union; lit: subtraction; "no volume created" guard | MATHEMATICALLY PROVEN; EXACT-DIFFERENTIAL-TESTED |
| 15 | clearance | `Observation.cleared`, `clearance_partition` | outward offsets with a directed integer sqrt; inward 2⁻²⁶ translates lie within 2⁻²⁶ of K | MATHEMATICALLY PROVEN (reviewed) |
| 16 | settlement (IN credit) | `Kernel._settle` | no pending receivers + I4 + I5 + clearance ⇒ legal parts ⊆ F; every credited polytope passes `polycheck.check` first (v2.9a) | MATHEMATICALLY PROVEN; RUNTIME-CERTIFIED (structure) |
| 17 | volume | `Poly.volume` (fan, \|det\|) | exact for FS-valid polytopes (Lemma FS) | MATHEMATICALLY PROVEN given FS; FS RUNTIME-CERTIFIED for credited and charged polytopes |
| 18 | per-cell credits | `_grid_credit_native` / `_python` | octree with validated facet rows; interior blocks exact; boundary cells floor/ceil of the exact clip; conservation against the exact piece volume | MATHEMATICALLY PROVEN; EXACT-DIFFERENTIAL-TESTED (native ≡ Python); RUNTIME-CERTIFIED (total conservation; the per-cell distribution relies on the octree's own splits) |
| 19 | unknown charges | `snapshot`: `ceil_mass`, `single_cell` | every live piece (deferred shells included) charged ⌈vol⌉; exact grid bounds; FS of each charged polytope checked | MATHEMATICALLY PROVEN; RUNTIME-CERTIFIED (structure) |
| 20 | non-overlap | `_commit` | parents removed before children are added; a stale or duplicate handle raises before any change; children are disjoint by construction (split halves, first-cover, subtraction chains) | by construction; EMPIRICALLY TESTED (chaos misuse) |
| 21 | TV bound | `kernel.wire_bound` | Astra's frontier theorem; integer floors at 2⁻⁹⁶ round the bound outward | MATHEMATICALLY PROVEN; EXACT-DIFFERENTIAL-TESTED (300 random instances: exact bound ≤ wire bound ≤ exact + m·2⁻⁹⁶, true TV ≤ bound) |
| 22 | normalization and serialization | `encode_int`, `ledger.decode` | 2⁵² exact ticks, largest remainder; strict decoding (sorted ids, finite, 0 < p ≤ 1, exact sum 1) | MATHEMATICALLY PROVEN; tested (byte-identical to `ledger.encode` on 200 instances) |
| 23 | status decision | `Kernel.release`, `certify` | only kernel-recorded (payload, bound) pairs; best bound; taint ⇒ ERROR | by construction; EMPIRICALLY TESTED (chaos) |
| 24 | writing | `run.write_result` | re-reads the file; wire bytes, sha256, bound ≤ tolerance, cell count must match; an INCOMPLETE record cannot carry a wire | tested |
| 25 | timeouts, caps | `Budget`, `Limit`, atomic `_commit` | an interrupted transition leaves the state unchanged; any valid state is a valid checkpoint | by construction |
| 26 | overflow and resource limits | `check_ranges`; Python big integers; split rows ≤ 2⁴⁰⁹⁶; native hex vertex keys (v2.9b: the decimal key broke on > 4300-digit integers); `COLLECT_CAP` overflow keeps the full obligation | an exception aborts the run (no certificate); an ArithmeticError taints | by construction |

**Defects found and fixed in this review (v2.9a–b).** None of them was ever observed to produce a wrong
certificate. Each was either a latent soundness gap reachable only by a search outside the current policy, or a
robustness failure.

1. The untrusted search decided CERTIFIED and held the payload/bound pairing. **Fixed:** release discipline, design
   §4h.
2. The kernel's obligation and mask arrays were shared mutably with the search: an in-place write would have
   silently weakened I4 or the masks. **Fixed:** read-only arrays and views, polytope copies.
3. Subnormal vertex coordinates broke the float filter's error model: with a coordinate near 2⁻¹⁰⁷⁴ and a row
   coefficient near 2¹⁰²⁰, the filter could report a wrong sign. **Fixed:** exact fallback.
4. The native helper keyed vertices by decimal `str()`, which fails above 4300 digits. Found by the chaos search.
   **Fixed:** hex keys and the split-row bound.
5. No structural certificate existed for the face structure that volume, octree and split all rely on.
   **Fixed:** `polycheck`, run on every credited and charged polytope, and per split on request.

## 3. Numerical model

| computation | representation | why it is exact or outward |
|---|---|---|
| obstacles | integer rows from the canonical binary64 parameters | exact |
| receivers | homogeneous integers (2(o+i)+1, …, 2048) | exact |
| polytopes | primitive homogeneous integer vertices (≈ 360 bits typical); exact plane intersections | exact; FS runtime-certified (§2 rows 6, 16, 19) |
| sign tests | binary64 filter certain iff \|v\| > 10⁻¹²·Σ\|terms\|, normal coordinates only; exact integers otherwise | exact decision (row 7) |
| receiver screens | rows normalized to 2⁴⁵, directed rounding; int64 with \|value\| < 2⁵⁸ | z3-proved (rows 10, 11) |
| volumes, credits | rationals; floor/ceiling at 2⁻¹⁶⁰; interior cells exactly 2¹³⁹ units | exact + outward integer conversion |
| clearance | outward offsets with a directed integer sqrt; inward rational translates | exact |
| TV bound | integer arithmetic at 2⁻⁹⁶ | outward |
| output | 2⁵² exact ticks, binary64 values summing exactly to 1 | exact |

Floating point appears only in (i) the sign filter (proved margin, exact fallback) and (ii) proposals (support
planes, mixed facets, hull receivers, split planes), all of which are validated exactly or re-proved.

## 4. Tests and checks (v2.9d)

| check | scope | result |
|---|---|---|
| Astra's assurance and geometry tests (`xlf_assurance/test_assurance.py`, `global_inverse/test_global.py`) | ledger, missing sibling, wire, cones, envelopes, clearance | 13 + 25 pass |
| `test_cbc.py` | scan equivalences, pointwise int64 reference, masks, subset screens, split partition, enforced children (exact segments), window exclusion, end-to-end decoded certificate, unvisited mass charged | 12 pass |
| `test_wire.py` | 300 random exact bound instances, tampering, 200 encoder byte-identity instances | 501 pass |
| `test_poly_exact.py` | Poly.split vs independent half-space vertex enumeration (degenerate small rows, dyadic planes, 60-bit random planes, real inverse cones); FS check rejects 4 corruption types; `check` ≡ `check_reference` on valid and corrupted structures; `verify_split` accepts exact halves and rejects swapped or wrong-plane halves; subnormal coordinates use exact signs | 17 pass |
| `smt_screen.py` (z3) | interval endpoints exact, no int64 overflow, directed rounding monotone | 6/6 PROVED |
| byte identity | v2.9a vs v2.8 on the 90-task benchmark (structure checks, release refactor) | 84/84 certified payloads identical |
| chaos search (`chaos.py`) | random legal API sequences + misuse; every snapshot vs exact-certified references | see `XLF_CBC_RESULTS.md` §3 |
| exact differential audit (`audit_exact.py`) | sampled exact points of IN pieces feasible; discarded points infeasible | see `XLF_CBC_RESULTS.md` §3 |
| run-time guards | structure of credited and charged polytopes; split conservation; no volume creation by cuts; octree conservation; facet-row validation; screen ranges; per-split certificate (optional) | abort on violation (never observed) |

## 5. Can per-label physical replay be dropped?

A replay re-checks, after the fact, that the credited regions are feasible and the discarded ones infeasible. In CBC,
the same facts follow from the transition lemmas (design §3–4h), provided the code implements them. The table below
lists the remaining obligations between "the lemmas hold" and "this run's certificate is correct", narrowest first,
with how each is discharged per label.

| obligation | discharged per label by | residual trust |
|---|---|---|
| polytope arithmetic (split, within, without, volumes, faces) | `CBC_VERIFY_SPLITS=1`: every split certified (Lemma V + FS); credited and charged polytopes always FS-checked | the 70-line checker `polycheck` (simpler than the producer; differential-tested against an O(k²) reference) |
| TV bound, encoding, decoding | exact integer computation on the output bytes in every run | `wire_bound`, `encode_int`, `decode` (≈ 60 lines; property-tested) |
| status, overlap, failure handling | release discipline, atomic commits, taint | ≈ 80 lines of kernel bookkeeping (reviewed; chaos-tested) |
| receiver screens | z3 proofs of the integer arithmetic | Numba compilation of ≈ 100 lines; CPython big integers |
| cone and envelope formulas | MATHEMATICALLY PROVEN algebra | ≈ 120 lines (`cone`, `Envelope`, `separators`, `cleared`, `clearance_covers`), reviewed in §2 |
| per-cell octree | conservation against the exact (FS-checked) piece volume | the octree's own splits (native GMP or Python) for the distribution among cells |
| inputs | — | container decoding and canonical geometry (shared with every labeller) |

**Answer.** With `CBC_VERIFY_SPLITS=1`, no per-label physical replay is needed for the polytope arithmetic, the
accounting, the bound or the serialization: these are certified inside the run. The remaining trust is in about
300 lines of closed-form algebra (cones, envelopes, clearance), the Numba compiler and the input decoder. None of
these is per-label data-dependent search code. A machine-checked proof of the algebra (Lean or Coq; no proof
assistant was available in this environment) and a pinned build would close the gap. The independent exact audit
remains a development tool: it found no violation, and it is not needed for each label.

## 6. Remaining gaps (OPEN)

1. **No machine-checked proof** of the cone, envelope, clearance and frontier lemmas. They are manual proofs, and
   the code is reviewed and tested against them.
2. **Per-split certification costs 1.1–2× in Python.** A native checker would make it the default; v2.9 leaves it
   opt-in for production and on for the validation campaigns.
3. **Per-cell distribution inside one IN piece.** The octree's clipped volumes are not individually certified,
   only their total. A native `verify_split` inside the octree would close this.
4. **Build identity.** Numba `cache=True`, g++ and GMP are unpinned.
5. **Interpreter isolation.** The search runs in the kernel's process. Read-only arrays and copies prevent
   accidental interference, not a malicious search.
6. **Inputs.** The decoder and the canonical geometry are trusted inputs (shared with FastLabel and the exact
   verifier).
