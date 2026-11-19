# XLF CBC: next steps (evidence-based, v2.9)

Production stays FastLabel v3.4 on `yushu/3d`. Nothing here is a production change; any production use needs a
separate readiness decision. Numbers refer to `XLF_CBC_RESULTS.md` (final build v2.9e, `067d746`).

## 1. Correctness obligations still open (ordered by risk × cost to close)

1. **Machine-checked proofs of the closed-form geometry (≈ 300 lines).** These are `solver.cone`, `Envelope`,
   `separators`, `Observation.cleared` and `clearance_covers`, plus the frontier theorem.
   * Status: manual proofs (design §3–4h), the line-by-line review in IMPLEMENTATION §2, and exact differential
     tests.
   * Next: a Lean 4 (Mathlib) formalization of the elimination lemma for C_K(r), the separation lemma, Lemma O,
     and the frontier TV bound over ℚ.
   * The polytope arithmetic itself does not need this: it is certified per run (`CBC_VERIFY_SPLITS=1`).
2. **Per-cell distribution inside an IN piece.** The octree's boundary-cell clips are certified only in total
   (conservation), not per cell.
   * Close by running `verify_split` on the octree's splits: in Python it costs the same as the kernel's splits;
     in `cells.cpp` it would be nearly free.
3. **Native per-split certificate.** `verify_split` in Python costs 1.1–2× wall time.
   * A C++/GMP port (same two checks; the hex conversion is already in `cells.cpp`) would make the certified mode
     cost ≈ 1.1× and remove the speed/assurance trade-off.
4. **Numba trust.** The screen's integer semantics are proved at source level (z3).
   * Close the compiler gap with a second, independent C++ screen and a byte-for-byte comparison of decisions in
     CI, or with translation validation on the generated LLVM IR.
5. **Pinned builds.** Pin Numba, LLVM, g++, GMP and the `.so` hash. Record the build hash in every label envelope.

## 2. Performance: the remaining tail (v2.9e, 10 s budget)

Remaining failure classes, with the measured cause:

| class | examples | measured cause | candidate fix |
|---|---|---|---|
| huge spread posteriors | `000023_w07_s0/s1` (2 × 10⁵ cells, feasible volume 10% of the world) | exact octree integration (8 s of 10 s) | cheaper certified per-cell enclosures (sub-cell corner tests with a float filter) for interior-heavy pieces; these channels are trivial for FastLabel |
| thin slabs L64/L128 | `000001_w02_s0/s3`, `000001_w05_s0/s3` | integration plus a large frontier of thin pieces | the same as above, plus contact refinement along slab faces |
| L = 256 multi-blocker penumbra | `000008_w11_s2`, `000005_w11_s0`, `000006_w11_s0` | **measured:** of the ambiguous receivers in large screens, 93% (`000008_w11_s2`) and 63% (`000005_w11_s0`) are dark with ≥ 2 possible blockers. These are unions of cones, with no convexity shortcut; the hull step resolves only the single-blocker share | split the piece by planes separating the candidate blockers' cone families (on each side one blocker dominates, so the hull step applies), or a first-cover hull step per blocker after a pencil split |

Current numbers to beat:
* benchmark 8 s: 87/90 with verification off, 84/90 with it on;
* canary 10 s: `XLF_CBC_RESULTS.md` §6.

Ideas measured and **not** adopted (evidence):
* Grid-aligned refinement of live pieces: the per-cell exact decomposition gives exactly the frontier bound on
  all 54 tail tasks with positive mass (design §6).
* A refined worst-case bound: gains at most 2×, and only for few-cell posteriors (design §6).
* Residual bitmasks for overflowing obligation lists: obligation sets at L = 256 are dense (10⁶–10⁷), and the hull
  step removed the need.

## 3. Assurance campaigns to repeat on every kernel change

* `pytest test_cbc.py test_wire.py test_poly_exact.py test_regression.py ../global_inverse/test_global.py
  ../xlf_assurance/test_assurance.py`. The fixtures cover the 214-cell posterior, both legacy Tau-pencil defects,
  the hull-only L256 case and the missing-sibling counterexample (Astra).
* `chaos.py` over the exact-referenced tasks, with `CBC_VERIFY_SPLITS=1`.
* `audit_exact.py` over the L64/L128 benchmark tasks and the newly certified hard tasks.
* Byte identity of certified payloads for refactors that should not change decisions.

## 4. Deployment study (needs a separate decision)

A deterministic-first hybrid labels each channel with CBC under a fixed budget and falls back to FastLabel on
INCOMPLETE. The certification semantics stay separate: a deterministic CERTIFIED envelope versus FastLabel's
`certified_stat`.
* Measured canary share of deterministic labels: `XLF_CBC_RESULTS.md` §6.
* Open measurements: H20 throughput with long-lived workers (the 1–2 s per-process load and JIT is paid once),
  memory at 6–16 workers, and the per-campaign cost of the FastLabel fallbacks.
* A controlled canary of the hybrid on a fresh, unpublished scene set would be the next step. It is not a
  replacement of FastLabel.
