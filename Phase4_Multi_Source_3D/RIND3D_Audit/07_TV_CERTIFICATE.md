# 07 — What the TV certificate guarantees, and what it does not

Audited code:
* `RIND3D_Dataset1K/rind3d_1k/fastlabel/pipeline.py` (production v3.4, commit b7c8dee; since 2026-11-13 the code is `RIND3D_Posterior/src/rind3d_posterior/fastlabel/pipeline.py`, unchanged algorithm);
* `RIND3D_Posterior/experiments/fastlabel/{adaptive3.py, sampler3.py, CERTIFICATE_AUDIT.md, REPORT.md}`;
* `RIND3D_Posterior/src/rind3d_posterior/rpl/bound.py:tv_bound`;
* `tpl/engine_p2.py:{stage2, emit_region, _shell_volume}`.

## 1. Five statements that must be kept apart

| # | statement | what supports it | class |
|---|---|---|---|
| 1 | **Deterministic TV bound** of a tau-pencil label: TV(q̂, q_F) ≤ `tv_bound`(L, U) | `rpl/bound.py`: the greedy maximum of Σ e_c w_c/(L_t + E) over the box 0 ≤ e ≤ U − L. I re-derived it: it is a valid upper bound, given that the true cell masses lie in [L_c, U_c] | A (bound), conditional on 3 |
| 2 | **Statistical confidence** of a FastLabel label: TV(q_B, q_F) ≤ ε = 0.05 except on an event of probability ≤ δ_label = 0.01/159,802; P(any false statistical certificate in the 1K) ≤ 0.01 | §3 | A, conditional on 3 and on the sampler law |
| 3 | **Numerical correctness of the geometry engine**: OUT decisions are sound (F ⊆ B), polytope volumes are accurate, and the point oracle is exact | float predicates with margins (A1, A2 of CERTIFICATE_AUDIT); directed-rounding GPU oracle with exact fallbacks; Gate A bit-identity (C); E4: 333/333 true sources inside support (C); E8: 0 MC-feasible points outside the engine support in 7 fixtures (C); E12: 0 of 106,170 prior-sampled feasible points outside 18 real label supports (C) | **assumption** + empirical evidence; not proved |
| 4 | **Correctness of the target**: q_F is the posterior that the Student should learn | 03: exact at the label level given G under ω and H1–H8. **Not** the natural-population target for a flattened Student under the current balancing or loader filter | A (label level); the Student level fails (03 §3) |
| 5 | **Student approximation and generalisation error** | nothing in the certificate | none (04 §5) |

A label certificate is a statement of type 1 or 2 **relative to q_F**. Statements 3–5 are separate. The text
"TV ≤ 0.015 vs reference" (REPORT) is an empirical comparison of two labels: TV(ours, q_F) ≤ 0.015 + tv_bound_ref by
the triangle inequality. It is neither a certificate nor a statement about the Student.

## 2. The receiver-subset identity (brief §8.1)

**Claim [A].** Let π be the prior measure, F ⊆ B measurable, 0 < π(F) ≤ π(B) < ∞, and π_A = π(· ∩ A)/π(A). Then

    TV(π_B, π_F) = 1 − π(F)/π(B).

*Proof.*
* For any A: π_F(A) − π_B(A) = π(A∩F)/π(F) − π(A∩B)/π(B) ≤ π(A∩F)(1/π(F) − 1/π(B)) ≤ 1 − π(F)/π(B), since π(A∩B) ≥ π(A∩F).
* Equality holds at A = F. The opposite sign gives the same value at A = B \ F. ∎

**Precise conditions:**
* *Inclusion:* F ⊆ B up to π-null sets. This is soundness of the outer bound. If F ⊄ B, the identity fails, and the sampling test cannot detect it, because it only samples B.
* *Same measure:* both sets are measured by the **same** measure, here Lebesgue on D(G) \ B̄(W), the declared uniform prior. For a non-uniform prior the identity holds with π-measure, and the test must then sample π_B, not U_B.
* *Cells:* q_B and q_F are pushforwards under the same cell map, so TV(q_B, q_F) ≤ 1 − p by data processing. Equality is not guaranteed.

## 3. The statistical certificate, checked against the code (brief §8.2)

| item | code | verdict |
|---|---|---|
| proposal = U_B | `sampler3.BSampler` / `fastlabel/sampler.py`: piece ∝ `hi` (exact polytope volume); tetrahedron of a Delaunay triangulation ∝ volume; Rocchini–Cignoni folding (I re-checked the three branches); v3.4 box fast path | correct law [B]. Discrepancy vs U_B measured ≤ 1e-11 (CERTIFICATE_AUDIT C3) [C] |
| independence | fresh draws from a keyed PRNG within a round; B_r is a function of the history H_{r−1} | valid under the standard idealisation of PRNG output as i.i.d. [A] |
| seeds | `label_rng`: SeedSequence([base_seed, scene, window, source, 3]), spawned into three streams | keyed, deterministic reruns [B] |
| adaptive stopping | chunks 128/384/512; stop if K > a2; stage-1 certify at 512 if K ≤ a1 | the certify event is {K₅₁₂ ≤ a1} ∪ {K₁₀₂₄ ≤ a2}, decreasing in failures. Early stopping only shrinks it. P(certify) is monotone in p (coupling), so its value at p = 1 − ε bounds every p < 1 − ε [A]. `design2` computes it exactly (I re-derived the stage-2 sum) [B] |
| multiple rounds | δ_r = δ·2⁻ʳ, r = 1..12 | Σ δ_r < δ [A] |
| confidence budget | δ_label = 0.01 / N_DATASET with N_DATASET = 159,802 hard-coded | union bound over labels, no independence needed. Valid for the 1K informative labels [A]. **Not valid as-is** for any larger label set (§5) |
| rare feasible regions | label = q_B with B ⊇ F | given soundness, no part of F can be missed. The rare-mode failure of sample-based Teachers (2026-11-09 audit) does not apply. The only error is excess mass on B \ F, ≤ ε [A] |
| oracle | `gpuverify` / `verify.cu`: legality, closed box, every FREE voxel; FP32 filter with E32 margin, directed FP64, exact GMP fallback | rigor argued in CERTIFICATE_AUDIT §4. Gate A: bit-identical on 2684 + 904 + 896 candidates [C]. Not independently re-proved here |
| what is stored | `lo = hi =` cell volume of B; `status = certified_stat`; `tv_bound = ε`; certificate fields | consistent with the theorem [B] |

**The dataset-wide 1% claim is correct, with stated scope [A].** The event "some statistical label is certified while
p_e < 1 − ε" has probability ≤ Σ_e Σ_r δ_r < N_stat · δ_label ≤ 0.01, because N_stat ≤ 159,802. Measured N_stat
is ≈ 67% of labels (02 §6), so the realised bound is ≈ 0.0067. The statement is about the labeller's internal
randomness for a **fixed** dataset. It is simultaneous: with probability ≥ 0.99 every certified statistical label
satisfies TV ≤ 0.05. A per-label 99% policy (v2) would have given only E[#invalid] ≤ 0.01 · N_stat, i.e. up to ~500
invalid labels.

## 4. Critical reading of CERTIFICATE_AUDIT.md

What I confirm:
* the identity and its use;
* the two-stage design and its monotonicity argument;
* the union bound;
* keyed streams;
* the separation of δ (confidence) from ε (bound), which are never added;
* the honest labelling of Monte-Carlo checks as empirical.

Gaps and corrections:
1. **Soundness A1/A2 carries the whole certificate and is not tested by it** (SHOULD). F ⊆ B is assumed, and an unsound OUT decision silently removes part of F. Existing evidence:
   * E4 (all 333 true sources inside the tau-pencil supports);
   * E8 (0 of ≥ 10⁴ MC-feasible points outside the engine support per fixture);
   * accuracy vs references (max TV 0.0164).

   Cheap independent checks to add:
   * assert, per label, that cell(s_j) lies in the stored support. This is a privileged check, and a violation proves unsoundness;
   * for a random sample of labels, draw points from the **prior**, not from B, test them with the exact oracle, and count feasible points outside B.

   E12 ran this second check on 18 broad labels (10 FastLabel, 8 release): 106,170 feasible prior points, **0 outside the support**. That bounds the pooled unsound fraction at ≤ 2.8·10⁻⁵ (95%) [C].
2. **The δ budget is a hard-coded constant** (MUST FIX before any extension). Any extension inherits δ = 0.01/159,802 per label unless changed:
   * uninformative channels: +39,238 in accepted windows;
   * the 100K: 2.94M labels;
   * re-labels.

   With the 100K's 2.94M labels the union bound would be 0.18, not 0.01. The fix is to set δ_label = δ_total / N_planned from the actual task list and store N_planned in every certificate.
3. **Re-runs across configurations** (SHOULD). The "never re-roll" rule covers seeds. A rerun with a different FastLabel configuration (DEV vs HOLDOUT vs production; v3.3 vs v3.4) uses the same seed but makes a different test. If the passing one of several runs is kept, δ must be charged per attempt. Record a configuration hash and an attempt counter per label.
4. **Confidence curves from the observed K** (minor). P(Bin(n, ε′) ≤ K_obs) is a valid p-value-type statement for that label. It is not a pre-specified error rate. Statements at the dataset level should use the design thresholds (a1, a2), as §5 of the audit does.
5. **The target is described as "the exact per-channel posterior of the weighted population"**. True at the label level (03 §1). The audit's "selection law / prior: 0 relative to the contract target" is correct only *relative to the contract target*. It does not address the Student-level mismatch (03 §3), which is of the same order as ε or larger.
6. **Scope limitation found here** (extension only). For all-dark channels, B can contain solid interiors that no receiver constraint removes. Point tests reject them through the legality check, so p = vol(F)/vol(B) can stay below 1 − ε through all rounds:
   * the statistical tier then falls back to exact, and the tau-pencil bound is loose for the same reason (06 §3);
   * measured on 18 real all-dark channels (E11): 9 certified, 3 `uncertified` after fallback (tv_bound 0.13, 0.72, 0.84), 6 timed out at 150 s. By contrast, 18/18 all-lit and 6/6 solid-separated channels certified.

## 5. TV to q_F is not TV to the right target (brief §8.3)

TV(label, desired posterior) ≤ TV(label, q_F) + TV(q_F, desired). The certificate bounds only the first term.

| situation | TV(q_F, desired) at label level | at the Student's optimum |
|---|---|---|
| ω-weighted training, desired = B1K | 0 (+ P_acc ≤ 0.0046 on average) | 0 |
| unweighted training | toy mean 0.022, max 0.34 (E5). SELECTION_LAW cites a Phase-1 measured 0.02–0.11 for the analogous v0.1 factor | not bounded |
| ω-weighted training, desired = N1 (natural single source) | 0 | toy mean 0.031, max 0.087 (balancing, E5). Not removed by any label accuracy |
| default loader filter after backfill | 0 per label | toy mean 0.22, max 0.96 (E5b) |

An extremely accurate approximation of Uniform(F) is therefore worth little at the Student level unless the
population is also right.

## 6. Effect on the Student (brief §8.4)

From 04 Thm 3 [A]:
* TV(E_ω[q̃|X], E_ω[q_F|X]) ≤ E_ω[TV(q̃,q_F) | X] ≤ ε for every X on the simultaneous event;
* because q_B ≥ (1−ε) q_F pointwise, the KL of the shifted optimum is ≤ −log(1−ε) = 0.051 nats;
* weights do not invalidate either bound.

What cannot be concluded: the trained Student inherits **no** TV bound. Its error adds estimation, approximation and
optimisation terms, and the target-definition term above. "Low posterior TV guarantees low Student generalisation
error" is false.
