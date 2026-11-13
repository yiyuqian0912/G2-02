# 02 — Source-Prior and Selection-Bias Audit of the 1K generator, balancing and weights

Scripts and raw outputs: `experiments/e1_weights_selection.py`, `e1b_debalancing_ess.py`, `e2_source_prior_audit.py`
(→ `experiments/out/*.json`). All measurements are on the full release (1000 scenes, 64,000 windows, 159,802 informative
channels) unless stated. Evidence classes as in 01.

## 1. Source prior

**Per-scene uniformity holds [B, C].** `sample_sources` is rejection sampling from the uniform cube (01 §1). E2 tests it
against each scene's own legal volume: for every source, the PIT value u = P_{y∼U(D(G))}(T(y) < T(s)) is estimated
from 6000 Monte-Carlo points per scene (float halfspace rows) for T ∈ {x, y, z, ‖s − ½‖_∞, local solid fraction}.

| statistic | KS p (n = 3110 sources) |
|---|---|
| x | 0.018 |
| y | 0.56 |
| z | 0.73 |
| r = ‖s − ½‖_∞ | 0.17 |
| local solid fraction (cube of half-width 0.06, randomised PIT) | 0.19 |

Per-family and per-k tests (36 + 16): minimum p = 0.019. Nothing is significant after a Bonferroni correction over
the 57 tests (threshold 0.0009). The x-axis p = 0.018 shows a mild excess in the top decile (351 vs 311 expected); it
is reported, not explained, and is compatible with chance at this multiplicity.

**The aggregate histogram is not uniform, and should not be [A, C].** Aggregating scene-conditional uniform sources
gives the density E_G[1_{D(G)}(s)/vol D(G)], which is low where obstacles concentrate (the centre of the cube, since
object centres are drawn away from the walls). Radial counts, by ‖s − ½‖_∞ in bins of 0.1:

| bin | observed | expected, per-scene Uniform(D(G)) | expected, Uniform(cube) |
|---|---|---|---|
| 0–0.1 | 19 | 21.5 | 24.9 |
| 0.1–0.2 | 144 | 158.2 | 174.2 |
| 0.2–0.3 | 479 | 455.0 | 472.7 |
| 0.3–0.4 | 867 | 925.8 | 920.6 |
| 0.4–0.5 | 1601 | 1549.5 | 1517.7 |
| χ² p | | **0.08** | **0.006** |

A flat global histogram would therefore be evidence of a **wrong** sampler here. Uniformity must be tested
relative to each scene's legal volume (PIT), as above.

**The training population barely tilts the marginal source distribution [C].** E3 weights each source by its number of
informative accepted windows (median 53 of 64; every source is informative in at least one window) or by Σω over those
windows:

* the weighted PIT deciles of r stay within 0.086–0.112 (s.e. ≈ 0.006 at Kish ESS 2576–2993);
* the local-solid PIT stays within 0.083–0.108;
* the Spearman correlation between multiplicity and local solid fraction is −0.10 (counts) and −0.05 (Σω).

Conclusion: "balanced layers imply balanced source positions" is false in general, but in this release the measured
tilt of the source marginal is small. The important tilt is not in the source marginal; it is in the hidden geometry
given X (§3).

## 2. Source count and family

* k histogram 154/135/158/553 [C]: the controller chooses k = 4 in almost all unforced slots.
* 69.1% of informative examples come from k = 4 scenes (64.9% after ω) [C].
* k is independent of S of the same scene [B]: chosen before sampling, from earlier scenes only.
* k is not significantly associated with family (p = 0.15) [C].
* Scenes are not i.i.d. across slots, because the controller state is a Markov chain driven by earlier scenes' sources.
  Conditional on the slot, the per-scene law is fixed. This matters only for the definition of the population, which
  is a slot-mixture (01 §4).

## 3. Effective-layer balancing: a hidden selection factor for single-channel examples

The acceptance factor of a window is c_t(M, L) with M = I_j + M_{−j}. For a flattened example of channel j, M_{−j}
depends on the **other, hidden** sources. Measured [C, E1/E1b]:

* **Range of c:** for k ≥ 2 the median c(1) is the floor 0.01 at L = 64 and L = 128, while c(k) = 1. The selection
  factor of a single-channel example therefore spans a ratio of 100 within the same scene.
* **Example composition:**
  * share of informative examples coming from windows with M = 1/2/3/4: 10.1/20.0/30.0/39.9% (pooled);
  * the same shares after removing the thinning (weight ω/c): 27.2/25.4/21.2/26.2%.
  * Raw shares are ∝ M because each window with M informative channels yields M examples; with windows balanced at ¼ per M this gives 1:2:3:4.
* **Consequence:** in the flattened training population, an example whose hidden sources were also informative in the
  same window is up-weighted by up to 100× relative to the natural population. 03 §3 shows that this tilts the
  Student's population optimum; the toy world of E5 measures the size of the tilt.

**Cost of exact debalancing (weight ω/c(M, L, k), computable from stored `m` and `c_table`):**

| side | n examples | Kish ESS with ω | Kish ESS with ω/c |
|---|---|---|---|
| L64 | 38,631 | 24,149 | 3,748 |
| L128 | 60,232 | 47,210 | 5,119 |
| L256 | 60,939 | 52,380 | 14,717 |
| pooled (side-normalised) | 159,802 | 121,869 | **18,788** |

* The top 1% of examples carry 22% of the ω/c weight.
* The balancing deliberately undersampled M = 1 windows by up to 100×, and undoing it costs a factor of 6.5 in ESS.
* A natural single-channel target and a balanced M-histogram cannot both be had cheaply from this release.

## 4. Training weights ω

| quantity | L64 | L128 | L256 |
|---|---|---|---|
| mean / median Λ̂_rate | 0.075 / 0.056 | 0.238 / 0.221 | 0.475 / 0.472 |
| min Λ̂_rate; scenes with Λ̂ = 0; with Λ̂ < 0.01 | 0; 1; 24 | 0.0059; 0; 1 | 0.023; 0; 0 |
| corr(rate, quota) | 0.93 | 0.93 | 0.94 |
| median relative s.e. of rate (binomial, 1024) | 0.129 | 0.059 | 0.033 |
| Kish ESS of window weights (of 16k / 24k / 24k windows) | 9,754 | 18,908 | 20,649 |
| share of total ω held by the top 10% of scenes | 0.29 | 0.20 | 0.17 |
| mean Λ̂ by k = 1..4 | 0.160 / 0.085 / 0.060 / 0.052 | 0.330 / 0.248 / 0.233 / 0.211 | 0.630 / 0.510 / 0.456 / 0.428 |

**Validity [A, B]:**
* The rate and quota estimators are unbiased for Λ_L(G,S) and independent of the accepted windows' content given (G,S): separate keyed stream; and inverse binomial sampling (r−1)/(N−1) with i.i.d. proposals.
* Weights are finite and nonnegative. Only the rate estimator has a zero (one L64 scene; 16 windows wasted).
* The loader's `combined` estimator is strictly positive. It changes the ESS by < 0.5%.

**Scale across sides [C]:**
* Raw ω gives 5.5% / 30.4% / 64.1% of the total weight to L64 / L128 / L256, while the example shares are 24.2 / 37.7 / 38.1%.
* L is an input of the Student, so multiplying ω by any function of L leaves the conditional optimum unchanged [A, 04 Thm 1].
* Normalising ω by its per-side mean raises the pooled ESS from 100,675 to 121,869. It also stops the L64 task from being trained at 1/12 of the L256 weight.

**What ω corrects, and what it does not [A]:**
* ω removes the quota normalisation 1/Λ_L(G,S). That factor depends on s_j and biases the per-channel posterior **given G**: in the E5 toy, without ω the mean TV is 0.022 and the max 0.34; with ω it is 0 to rounding.
* ω does **not** remove the factor c_t(I_j + M_{−j}), which depends on the hidden channels (§3). Nor does it remove label-availability selection (§6).
* Sample reweighting and posterior correction are therefore not interchangeable. ω is a posterior correction at the label level only because the label was derived for the ω-weighted population.

## 5. Scene rejection

All 1000 slots were accepted at attempt 0, so r̂ = 0 with a 99% upper bound of 0.0046 [C]. The unweighted bound
suffices for the ω-weighted rate as long as rejected scenes have below-average Λ, which is the mechanism of rejection
(E, not proved). Three L64 sides needed > 3000 proposals, and the cap is 6000. For 100K, with 4 windows per side, the
cap matters even less.

## 6. Label availability as a selection event

**The default loader filter selects on a privileged variable [B, C].**
* The default is `flatten(..., certified_only=True, accept_statistical=False)` (`loader.py:flatten`, commit b7c8dee). It keeps a label only if its status is `certified`.
* After a FastLabel backfill, `certified` means single-cell support or an exact path. In DEV-18 + HOLDOUT-18 (5082 labels; `RIND3D_Posterior/experiments/fastlabel/runs/{dev18,holdout18}_v34.jsonl`), 67.4% of labels are `certified_stat`. These are exactly the non-single-cell posteriors: median 177–286 cells, against 1 for the kept ones.
* Share retained by the default filter: L64 **5.8%**, L128 **25.4%**, L256 **57.4%**.
* Which path a label takes is a function of (G, W, r_j), not of X alone. So the filter conditions the Student's population on hidden geometry.
* In the E5b toy, keeping only single-cell labels moves the Student optimum by mean TV 0.22 (max 0.96), and 42% of the observation mass vanishes [C].

**Not a label-level problem [A]:**
* For each individual (G, W, r_j), whether its label exists does not depend on s_j beyond (G, W, r_j). So per-label calibration is unaffected (E4).
* The bias is purely in P(G | X): selection on hidden variables (MNAR in the sense of [Rubin1976]; [Zadrozny2004] for selection that depends on more than x).

## 7. Response types present in accepted windows [C]

| type | count | note |
|---|---|---|
| informative (mixed) channels | 159,802 | 17.5% have lit fraction ≥ 0.99 and 4.7% have ≤ 0.01 (a few decisive dark or lit receivers) |
| — of which solid-separated (no 6-adjacent lit/dark pair) | L64 6.7% (all scenes), L128 2.5% (250 scenes), L256 0.39% (40 scenes) | E9 |
| all-lit channels (stored, unlabelled) | 25,795 | exist only in k ≥ 2 scenes, alongside an informative channel |
| all-dark channels (stored, unlabelled) | 13,443 | same |
| windows with any SOLID voxel | 60.2% | mean solid fraction 6.2% |

## 8. Summary of issues

| id | finding | severity | evidence |
|---|---|---|---|
| S1 | The flattened single-channel population carries the hidden balancing factor c(I_j + M_{−j}), which spans 0.01–1. ω does not remove it, so the Student's optimum is a balanced-population posterior, not N1 (03 §3). | **MUST FIX before single-channel Student training** (choose and declare the target; fixes in 09) | A, B, C |
| S2 | The default loader filter drops `certified_stat` labels, a selection on (G, W, r_j). After a backfill it would keep 6% of L64 labels, all single-cell. | **MUST FIX before using any backfill** | B, C |
| S3 | ω has a side-dependent scale; per-side normalisation is free. | SHOULD CHANGE | A, C |
| S4 | The rate estimator has a zero and 13% relative noise at L64; use `combined`. | SHOULD CHANGE (minor) | C |
| S5 | `public/windows.npy` exposes m, mask and k. They are privileged for a single-channel Student. | SHOULD CHANGE (documentation / loader guard) | B |
| S6 | The source prior is per-scene uniform; the aggregate is non-uniform as expected; proposals are independent of S; k and c are fixed before S; P_acc ≈ 1. | KEEP | A, B, C |
