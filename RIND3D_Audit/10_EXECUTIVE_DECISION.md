# 10 — Executive research decision

Every answer points to its evidence (01–09). Classes: A proved, B code, C measured, D literature, E recommendation.

**1. What should the Student predict?**
* The conditional law of the source cell, p(s | X) = P_T(S_j ∈ cell | X), on the 128³ grid (or a hierarchical head that is exactly normalised on it), outside the window box.
* X is the local SOLID crop, one response channel, the side and the origin. Not m, mask, k or ids.
* T is an **explicitly declared** target population. Recommended: T = N1, natural single-source worlds (G ∼ π, S ∼ Uniform(D(G)), uniform windows with S ∉ B̄), optionally stratified by X-measurable response types.
* The Bayes-optimal Student is the mixture E_T[q_G | X] over unseen worlds, not any single world's posterior. Its loss floor is H(S|X) = H(S|X,G) + I(S;G|X) > 0 (04 Thm 2, A).

**2. Which distribution should the labels represent?**
* Keep the per-world hard posterior q_G = Uniform(F_W^j(G)) (A; 03 §1). It is the exact posterior of S_j given (G, W, r_j) both for the 1K's ω-weighted law and for N1, so the labels need not change when T is declared.
* What must change is the **population** they are averaged over: weights and loader (M1, M2).

**3. Which loss should be the primary objective?**
* Forward cross-entropy to the certified label field, ω_T-weighted, with the label-or-one-hot fallback for any missing or uncertified label.
* Same population functional as one-hot NLL (04 Cor. 1, A). Lower variance: 1 − Σq² = 0.97 at L64 and 0.42 at L256 on real labels (C). About 3× data-efficiency in the exact toy (C).
* Label error ≤ ε = 0.05 in TV, ≤ 0.051 nats KL at the optimum (A).
* Primary evaluation: held-out-scene one-hot NLL, with the label CE as a control variate.
* Run the real-data pilot P1 to confirm the predicted gain before committing GPU budget (E).

**4. What source prior and selection policy?**
* Source prior: keep i.i.d. Uniform(D(G)) with the 2⁻²⁰ anti-duplicate rule and k drawn independently of S (B, C).
* Selection: any thinning must depend only on the example's **own** X (side, own response type), never on other channels. Then P(S|X) is untouched without weights (A, D). Keep ω only for the quota normalisation, side-normalised and `combined`.
* For the existing 1K with T = N1, use ω/c(m) (stored; ESS 121.9k → 18.8k). Alternatively, train the multi-channel Student, for which the balancing cancels (A).

**5. All-lit and all-dark?**
* They are informative, not uniform: fixtures exclude 18–38% of the prior for all-lit and 96% for all-dark; real all-lit posteriors cover 7–95% of the cube (C).
* Include them in a new release as declared, X-measurable strata (Option D), labelled by FastLabel after the solid-interior fix S6:
  * all-lit certifies (C, E11);
  * all-dark certified only 9/18, with 3 uncertified and 6 timeouts, because solid interiors stay inside the outer bound (B, C).
* Do not use the 1K's existing uninformative channels as single-channel examples: they are conditioned on hidden informative channels (M4, A).
* "Distance-based darkness" is a family-specific statistical effect of π(G), not physics (C, E10).

**6. Minimising overfitting, encouraging geometric reasoning:**
* Count scenes, not windows: 333 labels behave like 15 independent units (C).
* Prefer 100K scenes × 12 windows to more windows per scene (C, prior study: −4.43 vs −3.09 NLL at equal windows).
* Use posterior-field targets (variance), scene-level splits, symmetry augmentation where π is invariant (E), and no privileged inputs.
* Monitor the train/held-out gap of NLL and label CE. A training loss below H(S|X) means memorised worlds.
* Evaluate coverage cluster-aware and per stratum. Never use entropy alone.
* Avoid soft response-matching targets: they discard decisive small shadows (C, E6).

**7. What is sound, and what needs correction?**
* **Sound:** source sampling; window proposals; closed-box exclusion; source-separated storage; the per-channel theorem at label level (re-derived, A); ω for the quota factor; the tau-pencil engine on all tested response types (C, E8); the FastLabel certificate mathematics, including the simultaneous 1% (A, 07); scene splits; rejection of coarse screening.
* **Needs correction:**
  * M1, undeclared target and hidden balancing tilt;
  * M2, a loader default that drops 67% of FastLabel labels selectively;
  * M3, a hard-coded δ budget;
  * M4, uninformative channels;
  * S1–S8, including side-normalised ω, soundness asserts and the all-dark solid interiors.

**8. Can we proceed with the current 1K?**
* **Yes for labels:** the FastLabel backfill targets the right per-world object, whatever T is chosen.
* **Not yet for Student training:** first declare T, set the loader to accept every certified status with the one-hot fallback (M2), and choose the weights (ω/c for N1, or multi-channel).
* These are loader and documentation changes. No frozen artefact changes.

**9. Before committing to the 100K:**
* Drop m-balancing for a single-channel Student, or keep it only for an explicitly multi-channel Student.
* Fix κ in advance.
* Allocate δ by the planned label count.
* Store everything needed to rebuild any target weight.
* Decide whether to add all-lit/all-dark strata, and if so, after fix S6.
* Run pilot P1 on 1K labels to size the posterior-vs-one-hot gain and the real balancing tilt.

The rest of the 100K design (local-only exact rendering, scene splits, 12 windows per scene) is supported.
