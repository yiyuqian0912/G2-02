# J — RIND3D-100K: natural target with explicit complexity-coverage safeguards

**Request** (2026-11-13). The effective-layer counts m = 1..4 need not be balanced, but every level, especially m = 3 and
m = 4, must be represented strongly enough for training and evaluation. T* stays the primary target. The 12-window
quota and multi-source generation stay. Natural and coverage streams must stay logically distinct.

**Evidence:**
* `experiments/e18_natural_m_breakdown.py`: the natural m-distribution from the 1K rate streams;
* `experiments/e19_coverage_designs.py`: the 100K design simulator: 1K worlds re-weighted to a declared κ, expected values at 100,000 scenes;
* outputs: `experiments/out/e18_natural_m.json`, `e19_coverage_designs.json`.

Nothing in the production contract or the 100K branch was changed.

## 1. Natural m-distribution: where complexity is rare

**Admissible windows** (no source in the box, FREE > 0, m ≥ 1), from 1024 natural proposals per scene and side. Shares are
of windows among admissible ones; channel shares weight a window by its m mixed channels.

| side | K | scenes | P(admissible, m ≥ 1) per proposal | window share m = 1 / 2 / 3 / 4 (%) | channel share m = 1 / 2 / 3 / 4 (%) |
|---|---|---|---|---|---|
| L64 | 1 | 154 | 0.16 | 100 / – / – / – | 100 |
| L64 | 2 | 135 | 0.27 | 80 / 20 / – / – | 67 / 33 |
| L64 | 3 | 158 | 0.36 | 71 / 22 / 7 / – | 52 / 32 / 16 |
| **L64** | **4** | 553 | 0.41 | **65 / 22 / 8.5 / 4.7** | 42 / 29 / 17 / 12 |
| L128 | 3 | 158 | 0.59 | 50 / 30 / 20 / – | 29 / 35 / 35 |
| L128 | 4 | 553 | 0.63 | 43 / 26 / 16 / 14 | 22 / 26 / 24 / 29 |
| L256 | 3 | 158 | 0.81 | 20 / 27 / 54 / – | 8 / 23 / 69 |
| L256 | 4 | 553 | 0.80 | 17 / 16 / 20 / **47** | 6 / 11 / 20 / 63 |

**By geometry family** (K = 4): m = 4 window share at L64, and the fraction of K = 4 scenes with π₄ ≥ 1% per proposal:

| family | L64 m = 4 share | scenes with π₄ ≥ 1% | L128 m = 4 | L256 m = 4 |
|---|---|---|---|---|
| multiscale_convex | **1.8%** | **33%** | 8.2% | 36% |
| barriers | 3.5% | 66% | 12% | 45% |
| occluder_clusters | 3.9% | 63% | 13% | 47% |
| rotated_boxes | 3.9% | 73% | 15% | 50% |
| convex_polyhedra | 4.3% | 52% | 16% | 45% |
| prisms | 5.1% | 63% | 14% | 42% |
| wedges | 5.3% | 76% | 17% | 47% |
| corridors | 5.9% | 84% | 15% | 51% |
| thin_slabs | 8.2% | 96% | 20% | 54% |

**Readings:**
* **Complexity is rare only at L64, and for m = 4 also somewhat at L128.** At L256, m = 4 is the most common level in K = 4 scenes.
* **The source count K bounds m.** K < 4 scenes cannot have m = 4, and K < 3 scenes cannot have m = 3. Per-scene m quotas are impossible in general, as the request anticipated.
* **Family matters.** In multiscale_convex worlds, an L64 m = 4 window is a few-in-a-thousand event in two thirds of the scenes.

## 2. Minimum-coverage targets (declared before choosing a design)

Every target is an absolute count at 100K scale, per side L and level m ≤ 4. These are not shares; no 25% rule is imposed.

| target | value | basis (the 1K pilot, `E_STUDENT_PILOT.md`) |
|---|---|---|
| test: independent scenes with ≥ 1 (L, m) window | **≥ 300** | 121 evaluation scenes gave a ±0.11-nat 95% interval on the information gain. 300 scenes give about ±0.07, enough to resolve the 0.1–0.2-nat effects seen in the pilot per stratum |
| test: windows | **≥ 1,000** | stratified metrics with the bootstrap over ≥ 300 scenes |
| train: independent scenes | **≥ 3,000** | scene count dominated learning: 25% of the scenes lost 0.77 nats, while 25% of the channels lost nothing. 3,000 scenes is 4.4× the pilot's full training set, per rare stratum |
| train: effective channels | **≥ 20,000** | the pilot's 25%-channel arm (26k channels over 684 scenes) matched the full data |
| train: within-stratum T* ESS | **≥ 5,000** | the pilot's T*-release arm (ESS 11.5k) trained as well as the balanced arm |
| per family: m = 4 at every side | **≥ 30 test and ≥ 300 train scenes** | per-family intervals of about ±0.2 nats |

These targets are judgement (class E), calibrated on measured pilot scaling. They can be changed without changing the
method.

## 3. Designs compared (expected values at 100K; four κ candidates)

**Streams.** Both are declared before each scene and logically distinct.
* **natural:** n_nat windows per side; accept every admissible proposal.
* **coverage:** n_cov windows; accept an admissible proposal with probability c(m | K) = ρ^(K − m) for m ≤ K. So c(K) = 1, the highest achievable level is always accepted, and c > 0 at every possible level.
* Coverage slots exist only for scenes with K ≥ K_min and only at the listed sides. Elsewhere all 4 windows are natural.

**T\* weight** of any stored window (balance heuristic): w_m = 1 / (n_nat / P_A + n_cov c(m)/Λ_c).
* It is bounded by P_A / n_nat, i.e. at most 4/n_nat times the pure-natural weight.

**Results.** "Targets" lists the targets missed; "ESS" is the T* ESS fraction at L64. **κ = rising (0.1, 0.2, 0.3, 0.4) and uniform:**

| design (κ) | labels vs natural | proposals vs natural | L64 window share m = 1/2/3/4 (%) | L64 ESS | max weight / natural | targets |
|---|---|---|---|---|---|---|
| D1 natural 4+0 (rising) | 1.00 | 1.00 | 73 / 19 / 6 / 2 | 0.86 | 1.00 | **missed: L64 m = 4 test windows (880); multiscale L64 m = 4 (30 / 239)** |
| **C31: L64 only, K = 4 only, ρ = 0.5 (rising)** | **1.01** | **1.14** | 70 / 19 / 7 / 4 | **0.84** | **1.26** | **all met** |
| C31: L64, K ≥ 3, ρ = 0.5 (rising) | 1.02 | 1.19 | 69 / 20 / 8 / 4 | 0.84 | 1.26 | all met |
| C31: L64, K ≥ 3, ρ = 0.25 (rising) | 1.04 | 1.57 | 65 / 19 / 10 / 6 | 0.76 | 1.32 | all met |
| C31: L64 + L128, K ≥ 3, ρ = 0.25 (rising) | 1.08 | 1.71 | 65 / 19 / 10 / 6 | 0.76 | 1.32 | all met |
| C22: L64 + L128, K ≥ 3, ρ = 0.25 (rising) | 1.15 | 2.43 | 56 / 20 / 14 / 11 | 0.60 | 1.95 | all met |
| C22: all sides, K ≥ 2, ρ = 0.25 (rising) | 1.22 | 2.61 | 53 / 22 / 14 / 11 | 0.59 | 1.95 | all met |
| D1 natural 4+0 (uniform κ) | 1.00 | 1.00 | 79 / 16 / 4 / 1 | 0.83 | 1.00 | **missed: L64 m = 4 test windows and train channels; multiscale** |
| C31: L64, K = 4, ρ = 0.5 (uniform κ) | 1.01 | 1.07 | 77 / 16 / 5 / 2 | 0.82 | 1.26 | **missed: L64 m = 4 test windows** |
| C31: L64, K ≥ 3, ρ = 0.25 (uniform κ) | 1.03 | 1.34 | 73 / 16 / 7 / 4 | 0.75 | 1.32 | all met |

**κ = 1K-realised (0.154, 0.135, 0.158, 0.553) and heavy4 (0.1, 0.1, 0.2, 0.6):** pure natural D1 already meets every
target, including per family (multiscale L64 m = 4: 41 test / 331 train scenes).

**Per-(L, m) detail of the recommended design** (κ rising, C31 L64 K = 4 ρ = 0.5; m = 1 / 2 / 3 / 4):

| side | test scenes | test windows | train scenes | train channels | train within-stratum ESS |
|---|---|---|---|---|---|
| L64 | 9,799 / 5,397 / 2,227 / **1,255** | 28,119 / 7,717 / 2,653 / **1,511** | 78,395 / 43,179 / 17,812 / **10,042** | 224,950 / 123,477 / 63,678 / **48,343** | 187,594 / 109,640 / 56,846 / **40,433** |
| L128 | 9,176 / 6,378 / 3,596 / 1,856 | 22,057 / 10,445 / 4,971 / 2,527 | 73,411 / 51,027 / 28,772 / 14,846 | 176,458 / 167,116 / 119,309 / 80,859 | 155,388 / 151,777 / 108,834 / 69,670 |
| L256 | 6,155 / 5,722 / 4,731 / 3,500 | 12,187 / 10,915 / 9,374 / 7,524 | 49,239 / 45,778 / 37,849 / 28,003 | 97,496 / 174,643 / 224,976 / 240,761 | 90,480 / 166,203 / 216,075 / 225,795 |

* Multiscale_convex, the weakest family: L64 m = 4 rises from 30 / 239 to **55 / 438** (test / train scenes).
* The coverage windows carry 15% of the L64 T* weight mass.
* Coverage proposals per accepted window: median 10, p99 78.

## 4. Statistical checks

**Positivity and support.**
* The natural stream alone covers the full T* support.
* c(m | K) > 0 for every achievable m, so every window type of the coverage stream is also reachable.
* MIS weights exist for every stored window. They are bounded by 4/n_nat times the natural weight: 1.33 for 3+1, 2 for 2+2.

**Weight variance.**
* With 3+1 at L64, the T* ESS fraction at L64 drops from 0.86 to 0.84 at ρ = 0.5, or to 0.76 at ρ = 0.25.
* L128 and L256 are untouched.
* With 2+2 it drops to 0.56–0.60, with weights up to 2× natural.

**Target integrity.** The coverage stream is m-tilted, by construction and on purpose. Its windows enter T* training
only through the balance-heuristic weights, so T* is not redefined. Uniform(F) labels stay exact for each window: the
coverage tilt, like the release's, depends on s_j only through r_j, which is fixed (Theorem A1). The quota
normalisation is handled by the Λ_c and P_A estimates from the rate stream.

**Requirement 8 (keep common windows rather than reject them).**
* Three of the four L64 slots, and every slot at L128 and L256, are untouched natural windows.
* The coverage slot still accepts m = 1 windows with probability 1/8.
* The L64 m = 1 share moves from 73% to 70%; no stratum of natural low-complexity windows is removed.
* By contrast, the 1K's m-balanced design (c(1) = 0.01) cut the L64 T* ESS to 10%.

**Scene diversity.**
* Every scene keeps its 12 windows. No scene is rejected for lack of high-m windows: there is no per-scene m quota.
* A coverage slot simply returns a lower-m window when the scene has none of higher m.
* The independent-scene counts above already account for this.

**Cost.** Proposals are cheap local exact m evaluations (`LOCAL_RENDERING_100K.md`, lazy decisions). Labels dominate the cost.
* ρ = 0.5: +1–2% labels, +7–21% proposals.
* ρ = 0.25: +3–5% labels, +34–73% proposals.
* A cap of 2,000 proposals per coverage window is safe: P(no acceptance) < e⁻¹⁰ at the observed Λ_c. If triggered, the slot is recorded as a natural fallback and weighted with the corresponding mixture density.

## 5. Provenance the production must store (all already planned or cheap)

**Per window:**
* stream (natural / coverage / natural-fallback), proposal index, U;
* m, mask, per-channel counts;
* the c-table row used.

**Per scene and side:**
* the rate-stream m-histogram (for P̂_A and Λ̂_c, estimated as Σ_m rate_m c(m)/rate_n);
* natural and coverage proposal counts;
* n_nat, n_cov, ρ, K_min.

**Per scene:** K and its κ draw.

**Campaign level:**
* the declared κ, the c-tables and the allocation rule;
* if allocation is adaptive (optional; see below), the plan of every epoch, fixed from earlier epochs only.

## 6. Recommendation

1. **Primary target unchanged.** T*: natural multi-source worlds, single-channel posterior.
2. **Choose κ first.** It is the cheapest lever, and it is natural: k is independent of the scene, so T* is not tilted.
   * A κ with ≥ 50% K = 4 scenes, such as the 1K's realised (0.15, 0.14, 0.16, 0.55), makes pure natural sampling meet every target at 100K scale.
   * This keeps 85–94% T* ESS and adds no cost.
3. **Smallest safeguard, recommended as insurance and required if κ has < 50% K = 4:**
   * one coverage window of the four at **L64 only**, in **K = 4 scenes only**, with **c(m) = (1/8, 1/4, 1/2, 1)** (ρ = 0.5);
   * cost: +1% labels, +14% proposals;
   * all targets met for κ = rising; maximum weight 1.26× natural; L64 ESS 0.84.
   * For κ = uniform, use ρ = 0.25 with K ≥ 3: +3% labels, +34% proposals.
   * L128 and L256 need no coverage stream.
4. **Not recommended:**
   * 2+2 designs: twice the weight range, ESS 0.56–0.60, +15–22% labels, with no target that 3+1 misses;
   * any per-scene m quota;
   * thinning of the natural stream.
5. **Adaptive allocation (optional).** A dataset-level controller may raise or lower ρ, or the share of K = 4 scenes receiving a coverage slot, per epoch, to track the coverage targets. This is valid as long as:
   * it acts only on the coverage stream;
   * each epoch's plan depends only on earlier epochs;
   * the plan is stored.
   The MIS weights then stay exact. With the fixed table above, the simulator predicts no need for it.
6. **Students.**
   * Train on both streams with T* (balance-heuristic) weights.
   * Report results per (L, m, family) stratum. Within-stratum weights are the same T* weights, so stratified metrics are T*-conditional.
   * Never use coverage-stream m-shares as the training prior.

**Limits.** The simulator uses the 1K's 1000 worlds as the world model, re-weighted to κ. Expected values are not
production realisations: per-scene π estimates from 1024 proposals are noisy for the rarest levels. The canary of the
100K should verify the predicted L64 m = 4 counts before the contract is frozen.
