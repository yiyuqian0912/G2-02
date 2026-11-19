# XLF CBC: next steps (evidence-based, v3.0)

Production stays FastLabel v3.4 on `yushu/3d`. Nothing here is a production change; any production use needs a
separate readiness decision.

Numbers refer to `XLF_CBC_RESULTS.md` (unified build v3.0, `a648358`; reference build v2.9e, `067d746`). v3.0 has
one certification path. The items below strengthen that single path; none of them introduces a mode.

## 1. Correctness obligations still open (ordered by risk × cost to close)

1. **Machine-check the code-level proofs (design §8).**
   * Status: manual statement-by-statement proofs; z3 for the arithmetic cores (`smt_kernel.py`, 16 lemmas).
   * Next:
     * a Lean 4 model of `Poly.split` over ℤ (FS precondition, Theorem P);
     * the 1-D Helly step of `cone`;
     * the frontier theorem over ℚ.
   * The split model is about 35 lines of control flow on finite lists; Mathlib's convexity library is not
     needed if FS is stated combinatorially.
2. **Numba compiler trust.** The screen's integer semantics are proved at source level.
   * Close the compiler gap with translation validation of the generated LLVM IR for `_intervals`, or with a
     second independent screen (C) compared decision-for-decision in CI.
3. **Process isolation of the search.** This removes the "not malicious at the interpreter level" assumption.
   * Run the kernel in its own process with a message API (handles, rows, receiver ids in; hints out).
   * The design already allows it: every kernel input is plain integers.
4. **Pinned builds.** Pin Numba, LLVM, g++, GMP and the `.so` hash, and record the build hash in every envelope.
5. **Input decoding** (shared with every labeller). It is a CONDITIONAL dependency.
   * An independent decoder cross-check in CI.
   * Decoding once per scene rather than per channel; this also saves ≈ 1.7 s per L = 256 channel.

## 2. Performance: the remaining tail (v2.9e, 10 s budget)

Remaining failure classes, with the measured cause:

| class | examples | measured cause | candidate fix |
|---|---|---|---|
| huge spread posteriors | `000023_w07_s0/s1` (2 × 10⁵ cells, feasible volume 10% of the world) | exact octree integration (8 s of 10 s) | cheaper certified per-cell enclosures (sub-cell corner tests with a float filter) for interior-heavy pieces; these channels are trivial for FastLabel |
| thin slabs L64/L128 | `000001_w02_s0/s3`, `000001_w05_s0/s3` | integration plus a large frontier of thin pieces | the same as above, plus contact refinement along slab faces |
| L = 256 multi-blocker penumbra | `000008_w11_s2`, `000005_w11_s0`, `000006_w11_s0` | **measured:** of the ambiguous receivers in large screens, 93% (`000008_w11_s2`) and 63% (`000005_w11_s0`) are dark with ≥ 2 possible blockers. These are unions of cones, with no convexity shortcut; the hull step resolves only the single-blocker share | split the piece by planes separating the candidate blockers' cone families (on each side one blocker dominates, so the hull step applies), or a first-cover hull step per blocker after a pencil split |

Current numbers to beat:
* benchmark 8 s: 87/90 with verification off, 84/90 with it on;
* canary 10 s: 786/807 certified mode, 796/807 speed mode (`XLF_CBC_RESULTS.md` §5).

Ideas measured and **not** adopted (evidence):
* Grid-aligned refinement of live pieces: the per-cell exact decomposition gives exactly the frontier bound on
  all 54 tail tasks with positive mass (design §6).
* A refined worst-case bound: gains at most 2×, and only for few-cell posteriors (design §6).
* Residual bitmasks for overflowing obligation lists: obligation sets at L = 256 are dense (10⁶–10⁷), and the hull
  step removed the need.

## 3. Assurance campaigns to repeat on every kernel change

* The test suite:

  ```
  pytest test_cbc.py test_wire.py test_poly_exact.py test_adversarial.py test_regression.py \
      ../global_inverse/test_global.py ../xlf_assurance/test_assurance.py
  ```

  Its fixtures cover the 214-cell posterior, both legacy Tau-pencil defects, the hull-only L256 case, the
  missing-sibling counterexample and the closed 64³ room (both Astra's), plus the adversarial cases.
* `python smt_screen.py && python smt_kernel.py` (z3).
* `chaos.py` over the exact-referenced tasks. It certifies every split with the external oracle by default.
* `audit_exact.py` over the L64/L128 benchmark tasks and the hard tasks.
* Byte identity of certified payloads for refactors that should not change decisions.

## 4. Relation to FastLabel (outside XLF-CBC)

XLF-CBC never falls back to FastLabel. It returns CERTIFIED (with a deterministic bound on its own bytes) or
INCOMPLETE. Whether a production pipeline labels INCOMPLETE channels with FastLabel (`certified_stat`, a different
assurance model) is a separate pipeline decision. It must keep the two semantics distinct in the label envelope.
For that decision, the measured canary share of deterministic labels is in `XLF_CBC_RESULTS.md` §6.
