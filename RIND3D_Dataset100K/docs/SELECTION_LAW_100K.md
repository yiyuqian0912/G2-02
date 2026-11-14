# RIND3D-100K: generative and selection law, per-channel posterior target, training weights

> **Contract v2 (production, provisional).** The law of this document is the v1 law: adaptive κ and m-balanced thinning. Contract v2 replaces two of its parts:
> * **source count:** K ~ κ, declared;
> * **window acceptance:** natural acceptance of every admissible window, plus an optional coverage stream (`rind3d_100k/sampling.py`, `RIND3D_StudentResearch/J_COMPLEXITY_COVERAGE.md` §3–5).
>
> **Labels.** Theorem A1 (`A_TASK_DEFINITION.md`) holds for every stream, because the acceptance depends on s_j only through r_j. The labels need no change.
>
> **Training weights** (`loader.flatten(population="Tstar")`): the balance heuristic, per natural proposal,
> w(m) = 1 / Σ_s n_s c_s(m) / Λ_s.
> * natural stream: c = 1 and Λ = P_A;
> * coverage stream: c(m) and Λ_cov = Σ_m c(m) P(A, m);
> * both are estimated from the exact-m rate stream. P_A may also be estimated with the natural stream's quota estimator (combined_rb).
> * Where a side has only the natural stream, w = P_A / n_nat.
>
> `tests/test_sampling.py` checks the unbiasedness identity E[#windows(m)] · w(m) = P(A, m) exactly.

**Status markers:**
* **[P]** derived here;
* **[M]** measured on the development canaries (laptop);
* **[A]** an assumption, stated.

The 1K derivation (RIND3D_Dataset1K/docs/SELECTION_LAW_1K.md) is **re-derived** below for the 100K law, not assumed.
The 100K law differs from 1K in five places:
1. quotas 4/4/4;
2. per-side rate-stream sizes;
3. epoch-frozen controller plans;
4. lazy, screened evaluation of the acceptance rule;
5. local-only rendering.

## 1. Complete process (contract `rind3d-100k-contract-v1`)

For slot t, every random stream is keyed by (seed = 20261215, t, stream, attempt). These are the same stream ids as 1K.

1. **Geometry.** G ~ family generator `families[t mod 9]` (the v0.2 generator and its acceptance gate). It is drawn
   without reference to any source.
2. **Plan.** (k, c(·, ·, k)) = plan(t) from the epoch-frozen controller (`rind3d_100k/controller.py`). Its inputs:
   * the slot index;
   * a keyed uniform u_t, independent of everything else;
   * the outcomes (m histograms and natural statistics) of scenes in epochs ≤ e(t) − 1;
   * the prior (1K release statistics, an independent campaign).
3. **Sources.** s_1..s_k i.i.d. Uniform(D(G)) by keyed rejection. The only interaction is the 2⁻²⁰ anti-duplicate rule.
4. **Proposals.** For each side L: i.i.d. uniform origins (stream keyed by (t, L, attempt)) and an independent
   U ~ U(0, 1) per proposal.
   * **Rule:** accept iff A(W, S) ∧ U < c(m(W, S), L, k), with A(W, S) = [no s_i ∈ B̄(W)] ∧ [FREE(W) ≠ ∅] ∧
     [m(W, S) ≥ 1].
   * Proposals continue until 4 windows are accepted or the cap (6000) is reached. Repeated origins are skipped.
5. **Rate stream.** R_L proposals per side (R = 256 / 128 / 64 for L = 64 / 128 / 256), from separate keyed streams,
   with the same rule. Λ̂_L = (number accepted) / R_L.
6. **Scene acceptance.** All three quotas filled; otherwise the attempt is rejected and attempt + 1 redraws G and S.
   A worker failure (crash, timeout, memory) retries the **same** attempt, with the same streams.

## 2. Screening does not change the law [P]

**Claim.** `local_eval.evaluate_proposal` computes exactly the indicator 1[A(W, S) ∧ U < c(m, L, k)].

**Facts used:**
* **Probes** are exact evaluations of single receivers by the certified CPU predicate. Each probe is a true fact about
  R_j at that receiver. A lit FREE probe plus a dark FREE probe imply I_j = 1; a FREE probe implies FREE ≠ ∅.
* **The hull certificate** is an exact rational witness that conv({s_j} ∪ receivers) lies outside every closed body. It
  implies every receiver is lit and FREE, so I_j = 0.
* **Lazy rule.** Write m_lo for the number of channels proven informative and u for the number still unresolved.
  - It *accepts* only if m_lo ≥ 1, FREE ≠ ∅, and U < c(m) for **every** m ∈ [m_lo, m_lo + u].
  - It *rejects* only if, for every m in that range, m = 0 or U ≥ c(m).
  - Otherwise it renders exactly. Both lazy outcomes coincide with the exact rule for the true m.
* Accepted windows are then rendered completely, and the exact m is asserted to satisfy U < c(m).
* The probe locations come from an RNG that is not part of the law. They change cost only.

**Consequence.** The accepted windows, their order and Λ̂ are exactly those of an exact evaluation with the same keyed
streams.

**Verified [M]:**
* 12 scenes were rerun with the 1K full-field pipeline: identical windows, identical Λ̂, and 0 mismatches in
  2.9×10⁹ response bits.
* Equivalence held under thinning c ∈ {0.05..1}, k = 1..4, and all families.

**Rejected alternative.** A *coarse 128³ screen* could not be used to reject proposals: it is not a statement about
fine receivers. Measured on 18 scenes [M], coarse rejection would miss these fractions of exactly informative channels:
* 13.6% at L = 64;
* 5.0% at L = 128;
* 2.3% at L = 256.
These are mostly thin shadows and small features. Such a filter would bias the selection against fine structure, which
is a source- and geometry-dependent selection that ω does not correct. It is therefore **not used**.

## 3. The per-channel posterior and the training weight [P]

**Selection probability.** Fix (G, k, c) (the plan) and S. Proposals are i.i.d. from p_nat (uniform origins) and
independent of S. Each proposal is accepted independently with probability A(W, S) c(m(W, S), L, k). Stopping at the
quota, the accepted windows are i.i.d. from

    q(W | G, S) = p_nat(W) A(W, S) c(m(W, S), L, k) / Λ_L(G, S),     Λ_L = E_{p_nat}[A c].

**Weighted population.** Weight every accepted window of side L by ω = Λ_L(G, S). The weighted joint density of
(G, plan, S, W) is

    π(G) · P(plan | earlier epochs) · Π_i u_D(s_i) · n_L p_nat(W) A(W, S) c(m(W,S), L, k) · P_acc(G, S).

The plan factor does not depend on S:
* plans depend only on other scenes, which are independent draws, and on keyed uniforms;
* the within-epoch provisional updates use only *expected* compositions.

**Theorem (unchanged from 1K, re-checked).** If P_acc ≡ 1, the posterior of s_j given (G, W, R_j|W) in this population
is

    ∝ u_D(s_j) · 1[s_j ∉ B̄(W)] · 1[R_j(s_j) = r_j on FREE(W)],

which is the Uniform(F_W^j) target certified by the tau-pencil engine.

*Proof.* The only s_j-dependent factors are u_D(s_j), 1[s_j ∉ B̄(W)] and the consistency of R_j. The factor
c(I_j + m_{−j}, L, k) depends on s_j only through I_j, which is fixed by the observation r_j. Integrating over s_{−j},
the plan and the other quotas leaves a function of (G, W, I_j) only. ∎

**Applies equally to:**
* the 12-window quotas (n_L = 4);
* per-side rate sizes, which affect only the variance of ω̂, not its expectation;
* epoch planning.

**Breaks the theorem, so it is not done:** a plan, c or k depending on this scene's sources, responses or rates; a
proposal distribution depending on S, for example coarse-field-guided proposals; or the coarse rejection above.

**Weight estimators** (`loader.Scene.weights`). Each is unbiased for Λ_L and independent of the accepted windows'
content given (G, S):

| estimator | definition |
|---|---|
| rate (contract default) | Λ̂ = accepts / R_L |
| quota | (r − 1)/(N − 1) with r = 4. Inverse sampling: the i.i.d. gaps between acceptances are independent of the accepted windows. |
| combined | (1 − a_L) rate + a_L quota, with fixed a_L = 0.12 / 0.05 / 0.035 (inverse-variance at the canary medians) |

**Weight noise only costs effective sample size.**
* With relative standard deviation cv, the ESS factor is 1/(1 + cv²).
* With R = 256 / 128 / 64 at the 1K medians of Λ (0.056 / 0.22 / 0.47), cv ≈ 0.26 / 0.17 / 0.13. The ESS factors are
  therefore 0.94 / 0.97 / 0.98 [P, using the 1K medians].
* These per-side sizes cut the rate-stream rendering cost about 4× at L = 256, compared with R = 256 [M].

**Training without ω** gives the biased factor E[1/Λ_L(G, S) | s_j]. This is *not* covered by the label certificate.
Training must use ω.

## 4. Scene rejection and permanent failures [P]

**Conditioning on acceptance.** Let E be "scene accepted". For the dataset-average label error,
TV ≤ P(Eᶜ)/P(E) = r/(1 − r) (1K §3).

**Permanent failures.** Slots whose attempts all failed (retries exhausted) also condition on a geometry/source event.
They are counted as rejections in r.

**Measured [M].** 0 rejections in 96 + 6 + 4 + 12 canary scenes. The one-sided 99% upper bound for n = 96 is
r ≤ 0.047; production will report r̂ with n = 10⁵.

## 5. Error budget of a label

| term | size | kind |
|---|---|---|
| engine certificate | `tv_bound` per label (stored) | deterministic, float-filtered predicates (Phase-1 METHOD §5) |
| anti-duplicate rule | < 1e-14 | derived (1K §4) |
| scene rejection / failures | dataset average ≤ r/(1 − r) | statistical, reported with a 99% bound |
| ω estimation | affects weights, not labels | statistical, unbiased |
| screening / local rendering | 0 (identical decisions, exact certified responses) | proved (§2), verified on 2.9×10⁹ bits |

**Not covered:** bit-identity of GPU and CPU engines is a software equivalence, not a real-arithmetic proof; the
engine's geometric predicates are exact up to documented float margins.

## 6. Balance: raw and weighted (dev canary, 96 scenes, 1152 windows) [M]

| | m = 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| raw share | 27.3% | 24.6% | 24.1% | 24.1% |
| ω-weighted share | 30.5% | 25.3% | 23.1% | 21.1% |

* k histogram 14 / 12 / 17 / 53.
* m is balanced within each side, within ±2 windows per cell of 96 at L = 128 and 256.
* The controller ran with 32-scene epochs. Production uses 1000-scene epochs: per-epoch noise shrinks, and the epoch
  lag delays the deficit correction by one epoch.
