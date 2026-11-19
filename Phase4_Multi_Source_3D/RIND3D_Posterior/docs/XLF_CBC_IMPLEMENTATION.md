# XLF CBC labeller: implementation assurance (v3.0, unified)

Code in `experiments/credit_solver/`, plus the exact geometry it reuses from `experiments/global_inverse/` and
`experiments/xlf_assurance/`. Labels (PROVEN, IMPLEMENTATION-VERIFIED, EXACT-TESTED, EMPIRICALLY TESTED, CONDITIONAL,
OPEN) are defined in `XLF_CBC_DESIGN.md`; the code-level proofs are in its §8.

```sh
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/credit_solver
export PYTHONPATH=".:../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K" OMP_NUM_THREADS=1
native/build.sh "$(which python)"                                    # optional GMP octree helper
python cbc_run.py CONTAINER.npz WINDOW CHANNEL OUT.json --seconds 10 # the single certification path
python -m pytest -q test_cbc.py test_wire.py test_poly_exact.py test_adversarial.py test_regression.py \
    ../global_inverse/test_global.py ../xlf_assurance/test_assurance.py
python smt_screen.py && python smt_kernel.py                         # z3: 6 + 16 lemmas
python chaos.py ROOT KEY --seeds 3 --seconds 20                      # search independence, split oracle on
python audit_exact.py CONTAINER.npz W S OUT.json --seconds 8         # exact point audit
python batch.py OUTDIR MANIFEST:SPLIT --runner cbc_run.py && python compare.py OUTDIR
```

## 0. Mode-difference audit (v2.9e) and its resolution

**What differed.** A source search (`grep VERIFY_SPLITS|SPLIT_HOOK|CBC_` over the trusted modules) finds exactly
one difference between v2.9e certified mode and speed mode. With `CBC_VERIFY_SPLITS=1`, `Kernel.__init__`
installed `polycheck.verify_split` as `poly.SPLIT_HOOK`, and `Poly.split` called it after constructing both halves.
Nothing else differed:
* the structure checks of credited and charged polytopes;
* the release discipline;
* the exact cones, screens and credits.

(`CBC_NATIVE` selects between two implementations of the same octree, and `CBC_MIXED` only removed validated rows;
neither changes the trust model.)

| question | answer |
|---|---|
| 1. operation present in one mode only | `verify_split(parent, row, neg, pos)` on every split inside every transition: Lemma V vertex-set equality, plus `polycheck.check` of both halves |
| 2. invariant protected | I1 (cover: no feasible volume lost by a split) and FS (exact volumes, exact facet rows, correct edges) for every polytope, including the intermediate ones inside cut chains that no other check sees |
| 3. needed for soundness? | Only if `Poly.split` can deviate from Lemma V or break FS. It cannot: Theorem P (design §8.1), a code-level proof. The mathematics (Lemma V, FS) was already proven; the check compensated for the *implementation* not being proven |
| 4. compensating for untrusted code? | Yes. It re-proved, at run time, a property of the split implementation that had only been tested |
| 5. correct by construction instead? | Yes: the split primitive is proven (Theorem P) from an FS precondition that holds inductively from the cube, since every polytope the kernel holds comes from `Poly.cube()` or `split` |
| 6. redundancy under the actual representation | Exact Python integers everywhere in `split`. The binary64 filter decides only the early exits, with a proved margin and exact fallback. Theorem P therefore applies to the executed code, and the check is redundant |
| 7. cheap intrinsic replacement | Not needed for splits. The one place where volumes enter the bound keeps a cheap guard (§3) |
| 8. performance of the unified implementation | Equals speed mode: certified mode cost +23% wall time over the canary and 10 certifications at 10 s (XLF_CBC_RESULTS §5–6) |

**Behavioural evidence** (EMPIRICALLY TESTED, consistent with the proof): on all 786 canary channels that both
v2.9e modes certified, the payload bytes and bounds are identical. The check never changed a decision; it never
failed on 494,099 canary splits or in the chaos campaigns.

**What each mode guaranteed in v2.9e.** Both modes shared one algorithmic proof (design §2a), and both trusted the
same cone, envelope, screen, octree and serialization code.
* Certified mode additionally re-established Theorem P's conclusion for every split at run time.
* Speed mode relied on Theorem P's conclusion without a written code-level proof (tests only).

v3.0 writes that proof, removes the switch and the hook from production code, and keeps `verify_split` as an
external oracle (`polycheck.instrument_splits()`, used by `chaos.py` by default and by the tests).

## 1. Trusted computing base (v3.0)

| module (trusted part) | role | lines incl. docstrings |
|---|---|---|
| `cbc_kernel.py` | private problem, state, transitions, release, `certify`, envelope assembly, `separators`, octree credits, encoder, snapshot | ≈ 660 |
| `polycheck.py` `check`, `_topology`, `_sign_matrix` | structural certificate used by the two retained guards (§3) | ≈ 190 |
| `screen2.py` | Numba integer screens, masks, range guard | ≈ 175 |
| `global_inverse/poly.py` | exact polytopes (split, volume, side, filter) | ≈ 190 |
| `global_inverse/solver.py` (cone, apply_pair, subtract_many, Observation.world/exclude_window/cleared) | exact cones and set algebra | ≈ 90 |
| `global_inverse/envelopes.py` (Envelope, clearance_covers, clearance_partition) | shadows, clearance | ≈ 80 |
| `global_inverse/raster.py` (`fixed`), `penumbra.py` (exact validation part) | directed rounding; cone-facet validation | ≈ 30 |
| `xlf_assurance/ledger.py` (encode, decode), `kernel.py` (wire_bound, ceil_mass, single_cell, cell_id) | wire format, integer bound | ≈ 70 |
| `inputs.py` | container decoding (shared input path) | ≈ 60 |
| `native/cells.cpp` (optional) | GMP octree, a transliteration of the Python path | ≈ 340 C++ |

**Not trusted** (may change freely):
* proposal and search code: `cbc_search.py`, `shadow_facets.py`, `hull_proposals.py`, `solver.choose_pairs`,
  `penumbra.directions` proposals;
* the CLI and batch tools;
* instruments: `polycheck.verify_split`/`instrument_splits`, `check_reference`, `chaos.py`, `audit_exact.py`,
  `exactcheck.py`, `compare*.py`.

**Boundary enforcement** (program structure; Python cannot stop deliberate reflection, see §5):
* The kernel builds its own problem copy at construction. The search never holds a reference to any mutable
  trusted object: evidence arrays are read-only and private to the kernel, the search receives read-only views
  and polytope copies, and the State is immutable and replaced atomically.
* A static test (`test_adversarial.py::test_untrusted_modules_never_touch_kernel_private_state`, AST) requires
  that the untrusted modules never access `_`-prefixed attributes, never assign attributes on the kernel, and
  import only `certify` from the kernel module.
* **Gap found and closed in v3.0.** Through v2.9e the kernel cached its obstacle polytopes, facet geometry and
  clearance covers on the observation object it shared with the search. A search that overwrote
  `obs.obstacles` would have made the separating-plane validation check the wrong vertices. This was never
  exercised, but it was a real channel; `private_problem` removes it.

## 2. Primitive-by-primitive assurance (what is proven, what is assumed)

| # | primitive | code | assurance |
|---|---|---|---|
| 1 | input decoding | `inputs.load` | CONDITIONAL on the container decoder and canonical geometry, which are shared with FastLabel and the exact verifier. Astra's re-render agrees on 229 M voxels (EMPIRICALLY TESTED) |
| 2 | obstacle rows | `certificates.integer_row` | exact rationals of the binary64 parameters (reviewed) |
| 3 | split, within, without, subtract_many, apply_pair, exact cuts, window exclusion, clearance partition | `poly.py`, `solver.py`, `envelopes.py`, `cbc_kernel._exact_cut` | **PROVEN, code-level** (Theorem P and §8.2); arithmetic cores IMPLEMENTATION-VERIFIED (z3 S1–S3, Q1–Q2); EXACT-TESTED (`test_poly_exact`: independent vertex enumeration, degenerate cuts, real cones); EMPIRICALLY TESTED (494,099 canary splits and all chaos splits pass the external oracle) |
| 4 | sign filter | `Poly.side`, `_fsign_terms`, `filter_vertices` | PROVEN (error analysis ≤ 6u Σ\|terms\| ≪ 10⁻¹² margin; subnormals and non-finite values exact) |
| 5 | polytope structure and volume | FS, fan volume | PROVEN (Lemma FS, Theorem P); checked at the two guards of §3 |
| 6 | inverse cone | `solver.cone` | PROVEN, code-level (§8.3); elimination core IMPLEMENTATION-VERIFIED (z3 C1–C2); EXACT-TESTED (segment tests) |
| 7 | guaranteed and possible shadows, separators | `Envelope`, `encoded_subset`, `separators`, `penumbra` validation | PROVEN (§4, §4d, §4f, §8.3); pair-row rules and separation IMPLEMENTATION-VERIFIED (z3 E1–E3) |
| 8 | directed rounding and screens | `raster.fixed`, `screen2` | IMPLEMENTATION-VERIFIED at source level (z3, `smt_screen.py`); EXACT-TESTED against a pointwise int64 reference; CONDITIONAL on Numba compiling int64 `//` as floor division and the loops as written |
| 9 | masks, subset screens, inheritance | `_mask`, `remap_masks`, `Piece` | PROVEN (§4b, §4) |
| 10 | per-cell credits | `_grid_credit_python` / `cells.cpp` | PROVEN (octree argument, with clips by Theorem P); native ≡ Python EXACT-TESTED; total conservation checked at run time |
| 11 | accounting, non-overlap | `_commit`, `_settle`, `snapshot` | PROVEN, code-level (§8.4); EMPIRICALLY TESTED (chaos misuse, `test_adversarial`) |
| 12 | posterior theorem, integer bound | `kernel.wire_bound` | PROVEN (Astra); per-cell inequality and floors IMPLEMENTATION-VERIFIED (z3 T1–T3); EXACT-TESTED (300 random instances) |
| 13 | serialization | `encode_int`, `ledger.decode`, `run.write_result` | PROVEN (§8.4); EXACT-TESTED (byte-identical to `ledger.encode` on 200 instances; tampering rejected) |
| 14 | release | `Kernel.release`, `certify` | PROVEN by construction (§4h, §8.4); EMPIRICALLY TESTED (forged results, search exceptions, chaos) |

**The ≈ 300 lines of closed-form algebra** (cone, Envelope, separators, clearance) are now PROVEN at code level, and
their arithmetic cores are IMPLEMENTATION-VERIFIED. What is not mechanized is the correspondence between each z3
formula and its source line: it is a direct transcription, reviewed in §8. A transcription error would show up
in the exact differential tests, which detect it but do not prove its absence.

## 3. Runtime checks that remain, and why

| check | invariant | needed by the proof? | why kept | cost |
|---|---|---|---|---|
| `polycheck.check` of every IN-credited polytope (in `_settle`) and every charged live polytope (in `snapshot`) | FS of the polytopes whose volumes enter the bound | no (Theorem P) | It is the only place where probability mass is created or charged, so a cheap guard there makes a future defect in the split code or its proof fail closed instead of silently. It is the single defence in depth on the credit path | 0.5% of wall |
| exact volume conservation in `Kernel.split` and the octree (Python and native) | exactness of a search split / of the cell distribution total | no | free with cached volumes; fail-closed against a native-build mismatch | < 0.1% |
| "a cut cannot create volume" in `enforce` | subset property of cut children | no | volumes are needed anyway by the scheduler (cached) | < 0.1% |
| `check_ranges` before every screen | int64 range of the z3 proof | **yes**: the z3 proof assumes \|coef\| ≤ 2⁴⁵ and coordinates ≤ 2¹¹ | it is the proof's precondition | negligible |
| argument validation (receivers, rows ≤ 2⁴⁰⁹⁶, handles, exact tolerance) | the transition preconditions of §2a | **yes** | the search is untrusted | negligible |
| taint on ArithmeticError | no certificate after any failed guard | **yes** (fail-closed semantics) | — | none |

Removed: the per-split certificate (moved to the test oracle).

## 4. Tests and checks (v3.0)

| suite | scope | result |
|---|---|---|
| `test_cbc.py` | scan equivalences, pointwise int64 reference, masks, subset screens, split partition, exact segments, exclusion, decoded certificate, unvisited mass | 12 pass |
| `test_wire.py` | bound instances, tampering, encoder identity | 501 pass |
| `test_poly_exact.py` | split vs independent vertex enumeration; FS corruptions rejected; fast vs reference check; split oracle accepts/rejects; subnormals | 17 pass |
| `test_adversarial.py` | the 16 tests listed in its docstring, including forged certificates, double settlement, input mutation after construction, injected structure failure, timeouts and the static boundary check | 16 pass |
| `test_regression.py` | 214-cell posterior, both legacy Tau-pencil defects, hull-only L256 case, wedges task (exact references) | 5 pass |
| Astra's `test_assurance.py`, `test_global.py` | missing-sibling counterexample, closed 64³ room, ledger, cones, envelopes | 13 + 25 pass |
| `smt_screen.py`, `smt_kernel.py` | z3 lemmas | 6/6 + 16/16 PROVED |

## 5. Final proof and risk report

**PROVEN (manual).**
* The frontier posterior theorem.
* Every transition lemma (design §3–§4h).
* Theorem B (algorithmic soundness).
* The code-level proofs of the split primitive, set algebra, cones, envelopes, separators, clearance, accounting,
  serialization and release (§8).

**IMPLEMENTATION-VERIFIED (z3).**
* The screen interval arithmetic, int64 range and directed rounding.
* The split crossing point and the orientation and lexicographic tests.
* The cone elimination core.
* The envelope pair-row rules and the separation lemma.
* The clearance square root and distance.
* The per-cell posterior inequality and the integer TV floors.

**EXACT-TESTED.**
* Split vs independent vertex enumeration.
* Native ≡ Python octree.
* Screens vs the pointwise reference.
* Bound and encoder property tests.
* Regression fixtures against exact references.

**EMPIRICALLY TESTED.**
* 0 inconsistencies against 86 exact references and 786–796 canary labels.
* Chaos campaigns: 0 violations.
* Exact point audits: 0 violations.
* The external split oracle: 0 failures.

**CONDITIONAL.**
* Numba compiles the screen loops faithfully.
* CPython's integer, `Fraction` and container semantics.
* The input decoder and canonical geometry.
* For the native octree: g++/GMP correctness and the transliteration review.
* The search is not malicious at the interpreter level (no reflection into the kernel).

**OPEN.**
* No machine-checked proof of the control flow (Lean/Coq; no proof assistant here).
* Pinned and reproducible builds (Numba cache, LLVM, g++, GMP) and a build hash recorded in the envelope.
* Process-level isolation of the search, which would remove the reflection assumption.
