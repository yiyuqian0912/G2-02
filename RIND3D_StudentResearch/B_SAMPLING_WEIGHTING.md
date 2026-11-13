# B — Sampling and weighting decision for the multi-source single-channel target T*

**Inputs:**
* scripts: `experiments/e13_target_weights.py`, `e14_exclusion_factor.py`, `e15_100k_designs.py`;
* outputs: `experiments/out/`;
* definitions: `A_TASK_DEFINITION.md`;
* data: all 1000 release scenes, 159,802 mixed (label) examples.

## 1. The natural m-distribution, verified independently (brief §2: "do not treat the audit's estimate as ground truth")

The audit estimated the natural share of examples by effective-layer count m by re-weighting the *release* with ω/c
(audit E1b). Two estimators share none of its assumptions:

* **Rate stream.** Each scene and side has 1024 fixed natural proposals on a separate keyed stream, with their m-histogram (`scene.json: sides[*].rate_m`). They are never thinned and use neither c nor the accepted windows.
* **Quota-stream prefix.** The first r_L = 16 / 24 / 24 proposals of the selection log. Every scene used at least r_L proposals, so this prefix is a fixed-size natural sample with no stopping bias.

Share of mixed examples by m. Scenes are weighted per natural proposal (T*); 95% scene-bootstrap intervals are in
`out/e13_target_weights.json`.

| side | release (raw) | rate stream | ω/c re-weighted (audit method) | quota prefix |
|---|---|---|---|---|
| L64 | 11.4 / 21.0 / 31.2 / 36.4 | **48.9 / 28.5 / 14.1 / 8.6** | 48.4 / 28.2 / 14.5 / 8.8 | 49.0 / 28.5 / 13.8 / 8.7 |
| L128 | 9.8 / 20.1 / 29.7 / 40.4 | **28.9 / 28.6 / 22.5 / 20.0** | 28.0 / 30.4 / 21.4 / 20.1 | 28.5 / 29.3 / 22.0 / 20.2 |
| L256 | 9.5 / 19.3 / 29.5 / 41.7 | **12.5 / 18.5 / 24.9 / 44.1** | 12.9 / 18.6 / 25.2 / 43.3 | 12.5 / 18.3 / 24.9 / 44.3 |
| equal-side pool | 10.2 / 20.2 / 30.1 / 39.5 | 30.1 / 25.2 / 20.5 / 24.2 | 29.8 / 25.7 / 20.4 / 24.1 | 30.0 / 25.4 / 20.2 / 24.4 |

* **The three estimates agree within their intervals.** The audit's figure (27.2 / 25.4 / 21.2 / 26.2, pooled with example-count side shares) is confirmed.
* **The balancing is strongly side-dependent.** It undersamples m = 1 at L64 by a factor of 4.3, but leaves L256 almost natural.

## 2. Estimators of the scene-side weight ω̂ ≈ Λ_L(G, S)

| estimator | definition | unbiased for Λ_L | min over 3000 scene-sides | zeros | median relative s.e. L64 / L128 / L256 |
|---|---|---|---|---|---|
| rate (contract default) | accepted count of 1024 thinned rate proposals / 1024 | yes | 0 | 1 | 0.128 / 0.059 / 0.033 |
| rb (new) | Σ_m rate_m c(m) / 1024: the stream's acceptance *probability* instead of its thinned count (Rao–Blackwellised over the thinning U) | yes, same stream | 0.0020 | 0 | 0.101 / 0.050 / 0.025 |
| quota | (r − 1)/(N − 1), inverse binomial on the quota stream | yes | 0.0032 | 0 | — |
| combined | (1 − a_L) rate + a_L quota | yes | 0.0014 | 0 | — |
| **combined_rb (new default)** | (1 − a_L) rb + a_L quota | yes | 0.0022 | 0 | lowest |

* All estimators are independent of the accepted windows' content given (G, S): separate streams [A, B].
* corr(rate, rb) ≥ 0.998.
* The new `loader.Scene.weights` exposes all five estimators.

## 3. Populations, weights, and their statistical cost (all 159,802 release examples; pooled with per-side normalisation)

| population represented | weight | label correct? | support / positivity | Kish ESS | ESS / n | CV | max / mean | top-1% share |
|---|---|---|---|---|---|---|---|---|
| (none) unweighted accepted examples | 1 | **no**: the 1/Λ_L(G, S) quota factor depends on s_j (audit E5 toy: mean TV 0.022, max 0.34) | — | 159,802 | 1 | 0 | 1 | 0.01 |
| B1K (rate) | ω_rate | yes, for B1K | ω = 0 on 1 L64 scene-side (58 examples carry no weight) | 121,869 | 0.763 | 0.56 | 5.8 | 0.036 |
| B1K (combined_rb) | ω_crb | yes, for B1K | ω > 0 everywhere | 122,245 | 0.765 | 0.55 | 5.6 | 0.036 |
| **T\* (release only)** | ω_crb / (n_L c(m, L)) | yes, for T* (Thm A1) | c ≥ 0.01 > 0 on every (m, L, scene) and every m = 1..4 occurs: positivity holds | **18,877** | **0.118** | 2.73 | 104 | 0.223 |
| tempered, a = 0.5 | ω / c^0.5 | **no** (neither B1K nor T*) | — | 106,640 | 0.667 | 0.71 | 17 | 0.052 |
| tempered, a = 0.75 | ω / c^0.75 | no | — | 57,754 | 0.361 | 1.33 | 44 | 0.110 |
| T* clipped at the 95% quantile | min(ω/c, q95) | no (30% of the T* weight mass removed) | — | 108,722 | 0.680 | 0.69 | 2.8 | 0.028 |
| **T\* (release + NAT-1K, balance heuristic)** | 1 / (n_L c / ω + n_p) | yes, for T* | bounded: w ≤ 1/n_p | see §4 | | | ≤ (n_L + n_p)/n_p | |

**What omitting each factor does:**
* **Omitting ω** breaks the labels themselves. Uniform(F) is the posterior only of the ω-weighted law (audit 03 §1).
* **Omitting 1/c** (training on B1K) leaves every label exact. It changes the Student's optimum: P(G | X) is tilted by the hidden factor H_t(G, W), which spans a 100× range (`A_TASK_DEFINITION.md` §3).
  * The size of that tilt on real data is measured in E16 (tabular estimate) and in the pilot (`E_STUDENT_PILOT.md`).
  * The toy estimate (audit E5) was a mean Student-level TV of 0.031.
* **Tempered or clipped weights** are biased toward B1K by an amount that is not computable without the answer. They are not used as defaults; they are reported only as ESS reference points.

**Hidden-variable shift between B1K and T\*** (example weight shares, pooled; from E13):

| | B1K | T* |
|---|---|---|
| k = 1 / 2 / 3 / 4 | 10.7 / 10.9 / 15.1 / 63.2 % | 5.1 / 8.8 / 15.9 / 70.2 % |
| m = 1 / 2 / 3 / 4 | 15.4 / 21.5 / 28.8 / 34.3 % | 27.2 / 25.4 / 21.2 / 26.2 % |
| family | differences ≤ 1.2 points | |

## 4. Reducing the variance at the source: the natural supplement NAT-1K

The release's thinning is the cause of the 6.5× ESS loss. The cure is to *sample natural windows*, not to re-weight
harder.

The 1K stores every scene's full 1024³ fields. Natural windows can therefore be drawn from the same worlds without
touching the release:
* script: `natural/natural_supplement.py`;
* new keyed stream 57000 + L, declared before any label;
* fixed n_p proposals per side, so the sample is self-weighting;
* crops are read from the stored fields; each is checked against the count index and against release crops of the same window.

**Expected T\* ESS** (E13 design study; per side, release n_L + natural n_p):

| n_p per side | new mixed labels | ESS L64 | ESS L128 | ESS L256 | total | ESS per labelled example |
|---|---|---|---|---|---|---|
| 0 (release only, ω/c) | 0 | 3,651 | 5,212 | 14,752 | 23,615 | 0.09–0.24 |
| 4 | 13,480 | 7,836 | 17,547 | 41,013 | 66,396 | 0.19–0.60 |
| **8 (NAT-1K pilot-train)** | **26,961** | 11,277 | 26,392 | 54,249 | **91,918** | 0.27–0.71 |
| 16 (NAT-1K pilot-select / eval) | 53,922 | 17,378 | 40,558 | 74,565 | 132,501 | 0.38–0.82 |

So about 27k extra FastLabel labels (≈ 45 min) raise the T* ESS by ≈ 3.9×, with every weight bounded by 1/n_p. The
release keeps its value: its m-balanced windows still enter with weights ≤ (n_L + n_p)/n_p relative to natural ones.

NAT-1K also yields the **natural** all-lit / all-dark / solid-separated channels used by the OOD study. The release's
uninformative channels cannot be used for that (audit M4).

## 5. Decision for the 1K

| use | population | weights | data |
|---|---|---|---|
| **Primary Student training (T\*)** | T* | balance heuristic 1/(n_L c/ω̂ + n_p), ω̂ = combined_rb, equal side shares | release + NAT-1K, pilot-train scenes |
| Controlled comparison (same physical samples) | T* release-only | ω̂/(n_L c) | release |
| Controlled comparison (the release's own law) | B1K | ω̂ | release |
| Primary in-distribution evaluation | T* | equal within side | NAT-1K, pilot-eval scenes (no weights at all) |
| Secondary evaluation | B1K | ω̂ | release, pilot-eval scenes |

The B1K and T*-release-only arms stay available for controlled experiments (`loader.flatten(population=...)`, pilot
arms). The permanent training default is decided by the pilot's measured generalisation (E), not by the ESS alone.

## 6. Recommendation for RIND3D-100K: fix the sampling upstream (details in `H_100K_RECOMMENDATIONS.md`)

Design study E15, using the 1K's natural m-distributions, with 4 windows per side:

| design | M-histogram L64 | L128 | L256 | T* ESS / n (L64 / L128 / L256) | mixed labels per scene |
|---|---|---|---|---|---|
| D0 m-balanced thinning (as in the 1K) | 28/25/25/22 | 25/25/25/25 | 24/25/25/27 | 0.10 / 0.09 / 0.24 | 29.8 |
| **D1 natural windows (c ≡ 1), quota 4, ω = P̂(m ≥ 1)** | 73/18/6/3 | 56/23/12/9 | 34/22/19/26 | **0.85 / 0.88 / 0.94** | 22.0 |
| D3 hybrid (2 balanced + 2 natural, balance heuristic) | 50/22/16/12 | 40/24/19/17 | 29/23/22/26 | 0.52 / 0.67 / 0.86 | 25.9 |

**Source-count steering cannot balance m at small windows.**
* Even κ = all-4 gives an L64 M-histogram of 65 / 21 / 8 / 5.
* At L256, a declared κ ≈ (0.05, 0.15, 0.30, 0.50) balances m without any window thinning.
* κ must be fixed in advance: it is part of the declared population.

**Recommendation:**
* The default 100K design is D1: natural windows with a declared fixed κ.
* If a multi-channel study needs m-balanced windows, add them as a separately declared stream and combine by the balance heuristic (D3). Do not thin the single-channel population by hidden-channel counts.
