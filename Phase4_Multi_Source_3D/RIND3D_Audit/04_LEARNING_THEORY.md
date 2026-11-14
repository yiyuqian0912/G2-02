# 04 — Learning Theory: one-hot vs posterior-field supervision, Bayes optimum, irreducible uncertainty, calibration, overfitting

Notation:
* P_acc is the accepted (sampling) population of flattened examples.
* ω ≥ 0 is the loss weight, with tilted measure dP_ω = ω dP_acc / E[ω].
* V = (t, G, W, r_j) are the privileged conditioning variables of a label, and X ⊂ V is the Student input.
* e_S is the one-hot vector of the 128³ cell of S_j, and q = q_G is the label.
* CE(T, p) = −Σ_s T(s) log p(s).

## 1. Population optimum of a weighted cross-entropy

**Theorem 1 [A].** Let T be any target vector (one-hot or soft) and ω ≥ 0 with E[ω] < ∞. Over all conditional laws p(·|x),

    argmin_p E_acc[ω · CE(T, p(·|X))] = p*(s|x) = E_acc[ω T(s) | X = x] / E_acc[ω | X = x] = E_ω[T(s) | X = x].

*Proof.* Write the objective as E_X[E[ω|X] · Σ_s τ(s|X) log 1/p(s|X)] with τ = E_ω[T|X]. For each x, Gibbs'
inequality gives Σ τ log 1/p ≥ Σ τ log 1/τ, with equality iff p = τ. ∎

The unweighted conditional expectation is therefore **not** the optimum when ω depends on privileged variables. The
optimum is the conditional expectation under the tilted measure P_ω.

**Corollary 1: identical objectives, not merely identical optima [A].** Suppose q = P_ω(S_j ∈ · | V) and X is V-measurable. Then for **every** p,

    E_acc[ω · CE(e_S, p(·|X))] = E[ω] · E_ω[−log p(S_j|X)] = E[ω] · E_ω[E_ω[−log p(S_j|X) | V]] = E_acc[ω · CE(q, p(·|X))].

So weighted one-hot NLL and weighted posterior-field CE are the **same functional** of p. They share the minimiser in
every model class, so the equivalence survives misspecification. Their difference is only in finite-sample variance
(§4) and in the practical failure modes below. In the 1K, ω = Λ̂_L(G,S) depends on S, not only on V. The identity still
holds because q was derived as the posterior under P_ω (03 §1).

**Common optimum in the 1K.** With ω weights and Uniform(F) labels, both objectives have the optimum P_ω(S_j | X). This
is the posterior of the balanced flattened population B1K (03 §3), not of a natural world N1.

## 2. When the equivalence fails (brief §4)

| case | one-hot optimum | posterior-field optimum | equal? |
|---|---|---|---|
| ω-weighted, labels exact for P_ω (1K contract) | P_ω(S\|X) | E_ω[q\|X] = P_ω(S\|X) | **yes** (Cor. 1) |
| unweighted training on the 1K | P_acc(S\|X) | E_acc[q\|X] | **no.** q ≠ P_acc(S\|V), because of the 1/Λ factor; neither is P_ω(S\|X) |
| extra weight w(·), e.g. ω/c(M) for N1 | P_{ωw}(S\|X) | E_{ωw}[q\|X] | yes iff, after integrating out non-V variables, w induces a factor constant in s_j on F. True for ω/c(M), since c depends on s_j only via I_j (A) |
| labels missing, kept iff R = 1 | P_ω(S\|X), using all examples | E_ω[q\|X, R=1] | equal iff R ⟂ G \| X. **Fails** for the default loader filter (02 §6) |
| approximate labels q̃ | P_ω(S\|X) | E_ω[q̃\|X] | shifted by at most E_ω[TV(q̃,q)\|X] in TV (Thm 3) |
| source-excluded domains | both live on D(G) \ B̄(W) | same | yes, if the Student's support allows every cell outside the box |
| finite discretisation | e_S on the 128³ cell grid | q on the same grid | yes. With a continuous density head, one-hot uses the exact S while q is uniform within cells: a sub-cell mismatch only |
| multi-source, Student sees all channels | P_ω(S_j\|X_all) | E_ω[q_j\|X_all] | yes: P(s_1..s_k \| G, W, r_1..r_k) = Π_j q_j (A; anti-duplicate omitted) |
| model misspecification | argmin over class | argmin over class | **same functional**, so same argmin (Cor. 1) |

**An exactly unbiased fix for missing labels [A].** Use the target T = q when the label exists and T = e_S otherwise.
Then E_ω[T | V] = q for every V, whatever the missingness mechanism, provided R depends on V only, which holds for label
computation. The optimum is again P_ω(S|X), with no selection bias, at the cost of one-hot variance on the unlabelled
examples. This dominates dropping examples.

## 3. Irreducible ambiguity and the meaning of the loss

**Theorem 2 [A].** For fixed X = x, with q_X = E[q_G | X = x]:

    E_{G|x}[CE(q_G, p)] = E_{G|x}[H(q_G)] + E_{G|x}[KL(q_G ‖ q_X)] + KL(q_X ‖ p)
                       =   H(S | X=x, G)  +   I(S; G | X=x)          + KL(q_X ‖ p).

*Proof.* CE(q_G, p) = H(q_G) + KL(q_G‖p), and KL(q_G‖p) = KL(q_G‖q_X) + Σ_s q_G(s) log(q_X(s)/p(s)). Averaging over
G | x turns the last term into KL(q_X‖p). E_{G|x} KL(q_G‖q_X) is the mutual information between S and G given X = x,
when q_G = P(S|X,G) and q_X = P(S|X). ∎

**Reading the decomposition:**
* **Irreducible because G is hidden:** I(S;G|X). This is exactly the extra information a privileged Teacher has.
* **Irreducible even with G:** H(S|X,G), the width of the feasible set itself.
* **Reducible by the Student:** KL(q_X‖p) only.
* An optimal Student has posterior-CE loss H(S|X) = H(S|X,G) + I(S;G|X) > 0, and the same one-hot NLL. A nonzero loss is the Bayes entropy, not a failure.
* A training loss **below** H(S|X) on training scenes means the Student uses scene identity, i.e. memorised G (§5).
* Lower training loss therefore does not by itself mean better geometric reasoning.
* Computable diagnostic: E[CE(q_G,p)] − E[H(q_G)] = E[KL(q_G‖p)] ≥ I(S;G|X). The first two terms are known from labels, so this gives an upper bound on the hidden-geometry ambiguity remaining at the Student's level.

## 4. What FastLabel posterior fields buy if the optimum is identical (brief §4.1)

**Variance (Rao–Blackwellisation) [A, D].** For a fixed predictor the risk with Bayes soft labels has variance no larger
than with one-hot labels ([Menon2021] Lemma 1, i.i.d. data). For a softmax head with logits z the per-example gradient
is p − T, and

    E[‖p − e_S‖² | V] = ‖p − q‖² + (1 − Σ_c q_c²).

So the one-hot gradient carries an extra conditional variance with trace 1 − Σ q². On the 333 real labels (E4d) [C]:

| grid | mean of 1 − Σq² | median | L64 / L128 / L256 means |
|---|---|---|---|
| 128³ cells | 0.65 | 0.97 | 0.97 / 0.83 / 0.42 |
| 16³ blocks | 0.57 | 0.81 | 0.91 / 0.74 / 0.33 |

The gain is near-total at L64, where posteriors are broad, and small at L256, where 53% of labels are single-cell
(E4c). Rao–Blackwellisation removes the within-world variance only; the between-scene variance of q_G given X remains.

**Finite-data learning (toy) [C].** E7 uses the exact 2D world, a tabular Student with oracle-tuned smoothing, and 24
replicates. Excess risk E_X KL(P(·|X)‖p̂), in nats:

| scenes | one-hot | posterior | posterior, ε = 0.05 worst-case labels | hybrid ½(δ + q) |
|---|---|---|---|---|
| 100 | 3.50 ± 0.01 | 2.79 ± 0.03 | 2.79 ± 0.03 | 2.90 |
| 300 | 3.05 ± 0.01 | 1.75 ± 0.02 | 1.76 ± 0.02 | 1.90 |
| 1000 | 1.89 ± 0.01 | 0.66 ± 0.01 | 0.68 ± 0.01 | 0.75 |
| 3000 | 0.57 ± 0.01 | 0.13 ± 0.005 | 0.17 ± 0.006 | 0.16 |
| ∞ | 0 | 0 | **0.0499** (bound −log 0.95 = 0.0513, Thm 3) | 0 |

* Posterior supervision is worth roughly a 3× larger one-hot dataset: posterior at 300 scenes beats one-hot at 1000, while posterior at 1000 is slightly worse than one-hot at 3000.
* The certified-label bias is visible only once the variance is small (n ≥ 3000), and stays below one-hot at every n tested.
* The hybrid is dominated by pure posterior supervision whenever labels exist.

**Prior real-data evidence** (2026-11-10 Student study; notes in `experiments/prior_student_study_notes.md`) [C]:
* direct one-hot NLL beat Gibbs τ = 0.05 distillation (test NLL −4.46 vs −2.08; hybrid λ = 0.3 −4.27);
* that Teacher was a **soft, biased** target: exp(−C/τ) on a 512-receiver bank, with its own NLL −3.35;
* the same study's synthetic test with exact soft targets favoured soft targets (−5.67 vs −4.85).
* "Distillation hurt" therefore indicts that Teacher, not exact posterior fields. Theorem 1 predicts this: a soft-likelihood target shifts the optimum away from P(S|X) (05).

**Other claimed benefits, and their status:**

| claim | status |
|---|---|
| better multimodal coverage | E: each q_G covers every feasible region of its world, while e_S shows one point. Most real labels are single-component (E4c: 333/333 connected), so multimodality lives in q_X across worlds, not inside labels |
| calibration | E: same optimum, so no asymptotic advantage. Lower variance gives better-calibrated finite-sample estimates in E7 |
| faster convergence | C (toy), E (deep nets) |
| less dependence on individual sources | A: the q-target is invariant to where in F the source happened to fall |
| better generalisation with few scenes | C (toy, through variance). Not tested for memorisation (§5) |

**Costs and disadvantages:**
* compute: ≈ 4.3 h laptop for the 1K with FastLabel v3.4 (REPORT Checkpoint 6; 33× cheaper than tau-pencil);
* storage: up to 10⁶ cells per label;
* exposure to label errors (TV ≤ 0.05 one-sided over-dispersion, Thm 3);
* exposure to label-availability bias (02 §6);
* labels are tied to one prior: any change of the source prior or selection law requires new labels, while one-hot targets need only new weights.

**Controlled experiment** (prepared, not run; full protocol in 08 §3): arms A one-hot, B posterior field, C hybrid with
label-or-one-hot fallback. Same architecture, same scenes, same steps and seeds, same ω, evaluated on held-out scenes by
one-hot NLL plus label CE, coverage, and per-type calibration.

## 5. Label errors and the Student (brief §8.4)

**Theorem 3 [A].** For any nonnegative weights and any labels q̃ with TV(q̃_e, q_e) ≤ ε_e:

    TV(E_ω[q̃|X], E_ω[q|X]) ≤ E_ω[ε_e | X]    (convexity of TV).

If every label satisfies ε_e ≤ ε (FastLabel's dataset-simultaneous event, probability ≥ 0.99), the population optimum
moves by at most ε in TV **for every X**. A per-label 99% certificate gives this only in expectation over labels.
Weights do not break the bound: the conditional expectation is a convex combination under P_ω.

For FastLabel's statistical labels q̃ = q_B = p q_F + (1−p) U_{B\F} with p ≥ 1 − ε, we have q̃ ≥ (1−ε) q_F pointwise. So

    KL(E_ω[q_F|X] ‖ E_ω[q̃|X]) ≤ −log(1 − ε) = 0.0513 nats  (ε = 0.05)   [A; toy: 0.0499, E7].

The error is one-sided over-dispersion: the label is never sharper than the truth.

**What cannot be concluded:**
* The trained Student's TV is bounded by optimisation, approximation and estimation error plus ≤ ε label bias. The certificate covers only the last term.
* The certificate also does not cover the target-definition error of 03 §3. That error is of the same order or larger: toy balancing tilt TV 0.031 mean, label-filter tilt 0.22.

## 6. Correlated examples and overfitting (brief §6.3)

**Facts:**
* Independent units are scenes: 1000 geometries.
* A scene provides 64 windows and ≈ 160 informative examples sharing G and S.
* Labels of one source across windows are strongly dependent (E4b: per-source mean HPD ranks 0.26–0.71). Calibration statistics of the 333 labels have the power of ~15 independent units, not 333 [C].
* 60% of windows contain SOLID voxels (02 §7), so (solid crop, origin) can fingerprint a scene. A high-capacity Student could memorise q_G per scene: privileged information leaking through memorisation.
* The tell-tale sign is a training loss below the held-out loss by more than estimation noise, approaching H(S|X,G) (§3).

**Evidence on scenes vs windows:**
* The prior study found 776 scenes × 16 windows → −4.43 test NLL vs 194 scenes × 64 windows → −3.09, at equal window count (single seed; sources were cut together with geometries) [C].
* The toy E7 found no difference between 1200×1, 300×4 and 100×12 at a fixed source budget. That is a negative result for a tabular learner that cannot exploit scene identity: the toy does not test memorisation [C].
* The direction (more independent scenes, fewer windows) is supported. The size of the effect for the 100K must be measured (pilot in 08 §3) [E].

**Controls:**
* split by scene (already: sha256 split);
* report cluster-robust (scene-bootstrap) intervals for every metric [CameronMiller2015];
* track the train/held-out gap of both one-hot NLL and label CE.

## 7. Calibration and evaluation (brief §6.4)

Recommended, all on held-out scenes, ω-weighted for the declared target population, with scene-bootstrap intervals:

1. **One-hot NLL of the true source**, the log score: strictly proper and local ([Good1952], [Bernardo1979], [GneitingRaftery2007]). The primary metric.
2. **Label CE as a control variate.** By Cor. 1, E[CE(q, p)] = E[−log p(S|X)] exactly. Where labels exist, the label CE estimates the same quantity with lower variance, and the one-hot NLL is the label-free unbiased check. A significant difference flags label or availability bias.
3. **Excess over the label entropy.** E[CE(q,p)] − E[H(q)] = E[KL(q_G‖p)] ≥ I(S;G|X).
4. **Coverage / SBC.** The randomised HPD rank of the true source must be U(0,1) ([Talts2018]; [Hermans2022] for coverage diagnostics). Test with one example per source, or with a scene-level bootstrap; E4b shows why.
5. **Regional reliability.** For a fixed partition (8³ blocks, distance shells from W), bin predicted p(A|X) and compare with the frequency of 1[S ∈ A]. Use the Brier score on regions [Brier1950] for a bounded, decomposable score ([Brocker2009]).
6. **Strata.** Family; side; response type (lit-fraction decile, solid-separated, all-lit/all-dark if included); k and M where the target population includes them.
7. **Entropy vs realised NLL.** Plot predicted entropy against realised log-loss. Entropy alone is not a correctness measure: high entropy is right when H(S|X) is high.
8. **Multimodal coverage.** On the counterfactual families below, the mass assigned to each world-specific feasible region.

## 8. Counterfactual ambiguity experiments (brief §6.5)

**(a) Controlled diagnostic (E; cheap with FastLabel).**
* Keep G inside and near W, the true source s*, and hence X.
* Edit hidden geometry only outside the union of segments [s*, x], x ∈ FREE(W), so r_j is unchanged. This needs an exact check with the certified renderer.
* Each edited world G_i gives a different q_{G_i}, and the union of their supports shows what the Student must cover.
* This is not a sample from G | X. It tests coverage, not calibration.

**(b) Statistically representative (E).**
* Requires a generator that factorises into local and hidden parts, π(G) = π(G_loc) π(G_hid | G_loc). For the v0.1 families this holds only approximately: fixed object counts couple them.
* Draw G_i = (G_loc, G_hid,i) with G_hid,i ∼ π(· | G_loc) and weight w_i = P(X | G_i) = vol F(G_i) / vol D(G_i) × [window factor].
* Then Σ w_i q_{G_i} / Σ w_i is a consistent self-normalised importance estimate of q_X = P(S | X) under the declared prior, and the Student's p(·|X) can be compared with it.
* The construction requires that the edits preserve X, and that w_i includes every selection factor of the target population. Under B1K this is the H_t factor of 03 §3, which is why a natural target (N1) is easier to validate.
