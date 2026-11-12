# Phase 2 target: single-source, binary-response, source-excluded posterior under v0.3 sampling

## Scope

* **Phase 2 task.** The source-excluded global task, as corrected by the user on 2026-11-11. The true source and
  every posterior source lie outside the window's closed box B(W) = [o/1024, (o+L)/1024]³.
* **Not in scope.** The `sb-global-v0.3` change (sources inside the window) is not implemented.
* **Deferred.** Natural all-lit, all-dark and no-FREE windows are deferred coverage tests.
* **Phase 2 population.** A declared **informative** population: mixed FREE classes and no source in the closed
  box. It is not the unrestricted natural distribution.

## Generative facts (v0.3 contract and `docs/SELECTION_LAW.md`, read from RIND3D_Training_v03)

* **Source prior.** S ~ Uniform(D(G)), with D(G) = {p ∈ (0,1)³ : p outside the closed solid, certified clearance
  > 2⁻²⁶}. This is the v0.1/v0.2 driver rule.
* **Proposals.** Per (scene, L) and subset, a deterministic uniform proposal stream over origins
  {0..1024−L}³, independent of (G, S).
* **Informative acceptance.** A_inf(W) = [W has lit-FREE and dark-FREE voxels] ∧ [S ∉ closed box]. Given the
  observation O_W and a candidate s with s ∉ B(W), it is a function of O_W only.
* **Stopping.** A stream stops at its **quota** or at its cap.

## Posterior of a selected window

For a selected informative window, write the inclusion probability λ_W(s) = P(W selected | K, s). Then

    p(s | K, W, O_W)  ∝  1[s ∈ F_W] · λ_W(s),     F_W = {s ∈ D(G) \\ B(W) : s reproduces every FREE receiver of W}.

How λ_W(s) behaves depends on the sampling design:

| Design | λ_W(s) | Label |
|---|---|---|
| **Quota, unweighted**: the first n accepted proposals per scene | ≈ n / (N_o · P(A_inf \| K, s)). It depends on s through the **global** acceptance rate, which is exactly the v0.1 mechanism (P_valid). | **Biased**: Uniform(F_W) is NOT the posterior. Phase 1 measured TV of 0.02–0.11 for the analogous v0.1 factor on extended posteriors. |
| **Quota + v0.3 weights** w ∝ P(A_inf \| K, s*) (from the true source's rate stream) | the weighted population has density ∝ p_nat(W) · A_inf(W) | **Uniform(F_W)**, exact up to the estimation error of P(A_inf), which affects the weights, not the label |
| **Fixed proposal budget**: M uniform proposals per scene, keep all accepted, no quota | M · p_nat(W) · A_inf(W), independent of s | **Uniform(F_W) exactly**. The count per scene varies, and its mean is M · E[P(A_inf)]. |
| Natural subset renormalised per (scene, L) after dropping source-containing windows | ∝ p_nat(W) / (1 − P_cs(s)) | Uniform(F_W) × 1/(1 − P_cs(s)). The factor is explicit, varies by ≤ 3.8% for L = 256, and can be applied exactly. |

## Consequences

1. **The labeller's job is the same in every unbiased design.** TPE computes the exact Uniform(F_W) cell
   masses. For the quota design those are the posterior **only** if training uses the v0.3 weights.
2. **P64.** For 64 informative windows per scene, the clean choices are:
   * the fixed proposal budget (variable count, exactly unbiased);
   * a quota of 64 with weights P̂(A_inf | K, s*).

   A raw quota of 64 recreates the v0.1 bias and must not be used unweighted.
3. **Status of these statements.** They are model statements about the documented design [P given the design].
   The labels themselves are exact for Uniform(F_W).
