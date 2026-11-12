# RIND3D 1K: sampling law, effective-layer balancing, and the posterior target of each source channel

Status:
* **Derived** (mathematics): sections 2–5.
* **Implemented**: `rind3d_1k/physical.py` and `rind3d_1k/controller.py`.
* **To be checked empirically in the canaries**: the quantities marked *[measured]*.

## 1. The complete generative process (contract `rind3d-1k-contract-v1`)

For campaign slot t, with every random stream keyed by (seed, slot, stream, attempt):

1. **Geometry.**
   * G ~ family generator `FAMILIES[slot mod 9]` (the v0.2 producer), followed by the v0.2 geometry acceptance.
   * The geometry is drawn without reference to sources.
2. **Source count.**
   * k ∈ {1, 2, 3, 4} is chosen by the balancing controller.
   * Its inputs are the controller state: cumulative histograms and pooled estimates from **previous** scenes, plus a keyed random number of this slot.
   * k is therefore independent of this scene's source positions. It may depend on the slot and on earlier scenes.
3. **Sources.**
   * s₁ … s_k are i.i.d. Uniform(D(G)), where D(G) = {p ∈ (0,1)³ : p ∉ solid, certified clearance > 2⁻²⁶} (the v0.1/v0.2 driver rule).
   * They are drawn by rejection from uniform proposals.
   * The only interaction is an anti-duplicate rule: |s_i − s_j|_∞ ≥ 2⁻²⁰.
   * The v0.2 multi-source rule `MIN_SEPARATION = 1/32` is **not** used. Section 4 explains why.
4. **Thinning table.** The controller fixes c(m, L, k) ∈ [c_min, 1] before the sources are drawn.
5. **Window proposals, for each side L ∈ {64, 128, 256}.**
   * i.i.d. uniform origins o ∈ {0, …, N − L}³, from a stream keyed by (seed, slot, L) only.
   * One more independent U ~ Uniform(0, 1) per proposal.
6. **Responses.** R_j(x) = V_G(s_j, x) ∈ {0, 1}: closed-solid, closed-segment visibility at the voxel centres. They are computed exactly as certified full fields, with exact count indices.
7. **Acceptance of one proposal W.**
   * A(W, S) = [s_i ∉ B̄(W) for every i] ∧ [FREE(W) ≠ ∅] ∧ [m(W, S) ≥ 1].
   * B̄(W) is the closed box [o/N, (o + L)/N]³.
   * m = Σ_j I_j, with I_j = [R_j has lit and dark FREE receivers in W].
   * W is accepted iff A(W, S) = 1 and U < c(m, L, k).
   * A repeated origin is skipped.
8. **Quotas.** Proposals continue until n_L ∈ {16, 24, 24} windows are accepted, or the side's cap is reached.
9. **Scene acceptance.** The scene is accepted iff all three quotas are filled. Otherwise it is rejected: the attempt number increases and new geometry and sources are drawn. Every attempt is recorded.
10. **Rate stream.** A further R = 1024 proposals per side, from separate keyed streams, are evaluated with the same rule. They give:
    * Λ̂_L(G, S) = (1/R) Σ 1[accept], used for the training weights;
    * the natural histogram of m among A-windows, used for the controller statistics and diagnostics.

**What a flattened Student example observes:** e = (K_local(W), R_j|_W, W), i.e. the local SOLID mask, one source channel and the window geometry.
* It does not see the other channels, the sources, G outside W, k, or the selection log.

**Privileged label information** (Teacher only): G, and the selection law.

## 2. The posterior of one source channel

Write λ(W | G, S) for the probability that W is one of the accepted windows of side L.

Under the quota design, the n_L accepted windows are i.i.d. draws from

    q(W | G, S) = p_nat(W) · A(W, S) · c(m(W,S), L, k) / Λ_L(G, S),      Λ_L(G, S) = E_{p_nat}[A · c].

Two factors depend on the sources:
* the normalisation 1/Λ_L(G, S), which is the quota effect also seen in v0.1;
* the scene-acceptance probability P_acc(G, S).

### Weighted population (the training distribution)

Give every accepted window of side L in scene (G, S) the weight ω = Λ_L(G, S). In practice the estimate Λ̂_L from the rate stream is used; it is unbiased. The weighted joint density of (G, k, S, W) is then

    π(G) κ(k) Π_i u_D(s_i) · n_L p_nat(W) A(W,S) c(m, L, k) · P_acc(G, S)          (∗)

where u_D is the uniform density on D(G). For clarity this omits the 2⁻²⁰ anti-duplicate factor (section 4).

**Theorem (per-channel posterior).** Under (∗), with P_acc ≡ 1:

    p(s_j | G, W, R_j|_W)  ∝  u_D(s_j) · 1[s_j ∉ B̄(W)] · 1[R_j(s_j) = r_j on FREE(W)].

That is, the label is exactly the source-excluded single-source target Uniform(F_W^j) certified by the tau-pencil engine.

**Proof.**
1. Factorise A(W, S) c(m, L, k) as

       1[s_j ∉ B̄] · Π_{i≠j} 1[s_i ∉ B̄] · 1[m ≥ 1] · c(I_j + m_{−j}, L, k).

2. Given the observation, I_j is fixed because it is a function of r_j.
3. m_{−j} and the exclusion of the other sources depend only on s_{−j}, G and W.
4. Integrating over s_{−j} and k, which given G are independent of s_j, therefore gives a factor h(G, W, I_j) that does not depend on s_j.
5. p_nat(W) and n_L do not depend on S.
6. The remaining s_j-dependence is u_D(s_j) · 1[s_j ∉ B̄] · 1[R_j consistent]. ∎

**Notes.**
* The same statement holds for an **uninformative** channel of an accepted window (I_j = 0, with m ≥ 1 coming from the other channels). Its target is Uniform{s ∈ D(G) \ B̄ : s reproduces the all-lit or all-dark FREE pattern}.
* The theorem also covers **dataset-level balancing**: c may depend on (m, L, k), because it is fixed before the scene's sources are drawn.
* What would break it, and is therefore forbidden:
  * a c that depends on this scene's realised source positions, rates or responses;
  * a source prior that depends on the geometry;
  * any choice of k or c made after looking at S.

### Unweighted counting

If training ignores ω, the per-channel posterior acquires the factor E_{s_{−j}}[1/Λ_L(G, S) | s_j]. This is the v0.1 mechanism. Its size is recorded per scene through Λ̂_L, so it can be audited. It is **not** covered by the label certificate. **Training must use ω** (scene-level, per side); the weights are stored with every window.

## 3. Scene rejection (finite caps)

Let E be the event that the scene is accepted. The actual population is the weighted population (∗) conditioned on E.

**Bound.**
* For a single example, the label error from conditioning on E is TV(p(· | obs, E), p(· | obs)) ≤ P(Eᶜ | obs).
* Averaged over the examples of the dataset, it is at most r/(1 − r), where r = P_ω(Eᶜ) is the ω-weighted rejection probability.

**Proof sketch.** TV(μ(· | E), μ) = μ(Eᶜ) for any event E. Averaging over obs ~ P(· | E) gives E[P(Eᶜ | obs) | E] ≤ P(Eᶜ)/P(E).

**Estimate.**
* r is estimated from all attempted scenes, accepted and rejected, using each attempt's rate-stream Λ̂ values *[measured]*.
* With caps of 6000 proposals per side and the measured acceptance rates, rejection is expected to be rare.
* This average bound is reported in the release statistics. It is statistical: it carries the sampling uncertainty of r̂, reported with a binomial upper 99% bound.

## 4. The anti-duplicate rule (instead of MIN_SEPARATION = 1/32)

A separation constraint couples the sources. Given s_j, each other source is uniform on D \ C(s_j), where C is the cube of half-width δ. Under balancing, both the conditional selection factor h(s_j) and the marginal prior of s_j vary with s_j. The relative variation is ≤ (k − 1) · vol(C) / (vol(D) · ḡ), where ḡ ≥ c_min · (vol(D) − vol(B̄)) is the mean acceptance factor of one other source. The TV error is ≤ 1 − a/b when the factor lies in [a, b].

| δ | vol(C) | TV error (k = 4, c_min = 0.01) |
|---|---|---|
| 1/32 (v0.2) | 2.4·10⁻⁴ | up to ~0.1 (**not negligible**) |
| 2⁻²⁰ (1K) | 6.9·10⁻¹⁸ | < 10⁻¹⁴ (negligible) |

**Decision:** the 1K release uses δ = 2⁻²⁰ and states this bound. Pairs closer than 1/32 are allowed. They occur with probability about C(k, 2) · (4π/3)(1/32)³ / vol(D) ≤ 10⁻³ per scene, and they are physically legal. The realised minimum distances are reported.

## 5. Total error budget of a label, under the weighted population

| Term | Size | Kind |
|---|---|---|
| Certified engine bound (cell volume intervals) | per label, stored (`tv_bound`) | deterministic, given the engine's numerical model |
| Binary64 rounding of the canonical rational rows, and engine margins | covered by the decision margins of the Phase-1 predicates (MARG 1e-11 on unit-normalised planes) | numerical assumption, documented in Phase 1 |
| Anti-duplicate rule | < 1e-14 | derived bound |
| Finite-cap scene rejection | dataset average ≤ r/(1 − r) | statistical estimate *[measured]* |
| Rate-stream estimation of ω | affects training weights, not labels | statistical (relative s.e. ≈ √((1 − Λ)/(1024 Λ))) |
| Output discretisation (128³ cells) | none: the target is the cell-mass vector itself | definition |

**Not covered:**
* bit-identity between the GPU pass and the CPU reference is a software equivalence, not a proof of real-arithmetic correctness;
* the engine's geometric predicates are exact only up to the stated floating-point margins (Phase-1 METHOD.md).

## 6. Effective layers

* **I_j(W) = 1** iff channel j has at least one lit FREE receiver and at least one dark FREE receiver in W.
* **m(W) = Σ_j I_j** is the effective layer count. It is a per-channel informativeness count, **not** the additive overlap R_sum(x) = Σ_j R_j(x).
* R_sum is reported separately: an overlap histogram per window, the lit fraction per channel, and the minimum pairwise Hamming distance between informative channels.
* **Balancing target:** a dataset-wide histogram of m over the 64,000 windows close to (¼, ¼, ¼, ¼). The weighted histogram (under ω) is also reported.

## 7. Relation to the v0.3 selection correction (checked independently)

**v0.3.** The v0.3 weights, `rind3d_training/selection.py`, are w = 1/(n_nat/n + Σ_s (n_s/n) · A_s/P(A_s)). They correct a **mixture of subset quotas**: natural, informative, extreme and rare windows from parallel streams.

**1K v1.** 1K v1 has a single quota stream per side with thinning, so that formula does **not** apply. The 1K correction above is derived from scratch for this law:
* ω = Λ_L(G, S);
* the per-channel theorem in §2.

**Common structure.** Both share the same structure: the quota normalisation 1/P(accept | G, S) must be undone by a weight proportional to the scene-level acceptance rate.

**Bias without the weight.** Phase 1 measured TV 0.02–0.11 for the analogous v0.1 factor on extended posteriors. That is why ω is mandatory for training on the 1K labels.

## 8. Estimating ω in the release (measured on `release_1k_v1`, 1000 scenes × 3 sides)

The theorem of §2 needs a weight with E[ω̂ | G, S] = Λ_L(G, S) that is independent of the accepted windows' content given (G, S). Two independent streams provide one each:

| estimator | definition | why it is valid |
|---|---|---|
| **rate** (contract default) | Λ̂_rate = accepts / 1024 on the rate stream | separate keyed stream; binomial, unbiased |
| **quota** | Λ̂_quota = (r − 1)/(N − 1): r = quota (16 or 24), N = quota proposals (repeats excluded) | inverse (negative-binomial) sampling has this unbiased estimator. For i.i.d. proposals, the accepted windows are independent of the gaps between acceptances, so the estimate does not depend on which windows were accepted. |
| **combined** | (1 − a_L) Λ̂_rate + a_L Λ̂_quota, with fixed a_L = 0.20 / 0.09 / 0.04 | a constant convex mix of two unbiased, mutually independent estimators. It is strictly positive (Λ̂_quota > 0). a_L is the inverse-variance weight at the median Λ of each side. |

**Measured:**

| side | mean Λ̂_rate | mean Λ̂_quota | corr | rate = 0 | median relative s.e.: rate | quota | combined |
|---|---|---|---|---|---|---|---|
| 64 | 0.0746 | 0.0736 | 0.93 | 1 scene | 0.130 | 0.260 | 0.116 |
| 128 | 0.2375 | 0.2382 | 0.93 | 0 | 0.059 | 0.189 | 0.056 |
| 256 | 0.4746 | 0.4770 | 0.94 | 0 | 0.033 | 0.155 | 0.032 |

**Readings:**
* The two estimators agree in mean on every side. That is an empirical check of the rate stream and of the quota bookkeeping.
* One scene-side (L = 64) has Λ̂_rate = 0: no rate-stream accept, although the quota filled. Under the rate estimator its 16 windows get zero weight. This is still unbiased, but it wastes data.
* The combined estimator avoids zero weights and lowers the relative s.e. modestly.

**Interface (`rind3d_1k.loader`):**
* Every window record carries `weight` (rate, the contract default) and `weights` = {rate, quota, combined}.
* `flatten(..., weight="rate" | "quota" | "combined")` attaches the chosen ω to every example.
* ω is a **loss weight**, never a model input.

**Scene rejection:**
* All 1000 slots were accepted at attempt 0; no proposal stream hit the 6000 cap. The largest proposal counts were 4658 (L64), 2678 (L128) and 979 (L256).
* So r̂ = 0/1000. The one-sided 99% binomial upper bound on r is 1 − 0.01^{1/1000} = 0.0046.
* The §3 bound on the dataset-average label error from conditioning on acceptance is therefore r/(1 − r) ≤ 0.0046 (99% confidence).
