# 05 — Hard geometric posterior vs soft response matching

## 1. The three formulations, and what each one asserts

* **Hard:** q_hard(s | G, W, O) ∝ π(s) 1[R_G(s)|W = O]. This is the exact Bayes posterior under deterministic, noise-free physics with the declared prior [A]. It is the target that FastLabel certifies.
* **Soft:** q_soft ∝ π(s) exp(−β d(R_G(s)|W, O)). This is the exact posterior of some explicit observation model only for special choices of d [A]:
  * with d = d_H, the Hamming count over n FREE receivers, it is the exact posterior under **independent receiver flips** with rate ε, where β = log((1−ε)/ε). This is the ABC/model-error reading of [Wilkinson2013]: the kernel *is* the error distribution;
  * with the **normalised** Hamming d_H/n, it is the flip model with ε_n = 1/(1 + e^{β/n}), which tends to ½ as n grows. A 64³ window has n ≈ 2.6·10⁵, so any moderate β makes every receiver almost uninformative. The implied noise level depends on the window size, which is physically incoherent. **Normalised Hamming is not a defensible likelihood for RIND3D** [A].
* **Hybrid:** L = L_hard + λ L_soft. Its optimum is a λ-mixture of two different conditional targets (04 Thm 1). It is calibrated for neither unless the soft term encodes a real noise process.

## 2. Measured trade-off (exact 2D toy, E6; known geometry, every legal source as truth)

Mean log score of the true source (nats; lower is better) and mass assigned to the exact feasible set F.

**Window 8 × 8 = 64 receivers, noise-free observations:**

| posterior | log score | mass on F | Case A (0 < dark < 10%, n = 89): log score / mass on all-lit candidates | Case B (30–70% dark, n = 91): log score |
|---|---|---|---|---|
| hard | **1.97** | 1.00 | **1.93** / 0.00 | **0.18** |
| flip ε = 0.01 | 1.99 | 0.98 | 2.00 / 0.05 | 0.19 |
| flip ε = 0.10 | 2.12 | 0.89 | 2.42 / 0.21 | 0.26 |
| normalised Hamming β = 100 (ε_eff = 0.17) | 2.23 | 0.83 | 2.74 / 0.29 | 0.34 |
| normalised Hamming β = 20 (ε_eff = 0.42) | 3.36 | 0.38 | 4.48 / **0.59** | 1.59 |
| normalised Hamming β = 5 | 4.78 | 0.20 | 5.31 / 0.55 | 4.09 |

**The same window with true i.i.d. flips at ε_true = 0.01:**

| posterior | share of true sources given zero mass | log score (finite part) |
|---|---|---|
| hard | **0.49** (catastrophic: infinite NLL) | — |
| flip ε = 0.01 (matched) | 0 | 2.08 |
| normalised Hamming β = 100 | 0 | 2.26 |

**Readings** [C]:
* *Small shadows are decisive (brief §5.2).* In Case A (a few dark receivers), the hard posterior gives the all-lit candidates zero mass. Normalised Hamming at β = 20 gives them 59% and loses 2.5 nats. A small percentage of shadow pixels can carry most of the information, so "few dark pixels means little information" is false.
* *Hard is optimal when the physics is exact.* Every soft variant loses log score on noise-free data. The loss is small for a flip model with small ε (+0.02 nats at ε = 0.01) and large for normalised Hamming.
* *Hard is brittle when the physics is not exact (brief §5.3).* At a 1% flip rate on 64 receivers, half of the true sources are excluded. On a 64³ window, any i.i.d. error rate above ~10⁻⁶ would exclude essentially every true source.
* *Robustness requires a noise model, not a distance.* The matched flip posterior is both exact and robust; normalised Hamming is neither.

## 3. What uncertainty is being modelled (brief §5.4)

| type | present in RIND3D? | correct treatment |
|---|---|---|
| A. hidden global geometry | **yes, the dominant ambiguity** | already in the target: p*(s\|X) = E[q_G \| X] mixes worlds (04 Thm 2). It must **not** be emulated by softening q_G. A soft likelihood blurs within-world constraints, whereas hidden geometry produces a mixture of sharp sets: a different shape of uncertainty |
| B. observation noise | **no**, in the synthetic benchmark: exact rendering, certified fields | add a noise model (and relabel) only if real sensors or explicit perturbations become part of the task. Then use an explicit generative noise process: a flip model, or better a spatially correlated boundary-band / geometry-jitter model, for which hard labels are wrong |
| C. labeller numerical error | yes, bounded | certificate ε (TV ≤ 0.05, one-sided over-dispersion; 07) |
| D. finite-data epistemic / model uncertainty | yes | ensembles, scene-level bootstrap, held-out calibration (04 §7). Not a property of the label |

Introducing a soft likelihood "to compensate for missing global geometry" conflates A with B. It changes the target to
the posterior of a world with noisy sensors. That world is not the one the dataset simulates.

## 4. Robustness without changing the target

Hard-label sensitivity is real at the level of a single world: a sub-voxel change in an obstacle can flip a receiver and
change F discontinuously. In the noise-free limit the posterior is not continuous in the data; well-posedness results
need a noise model ([Stuart2010]). The Student's target, however, is q_X = E[q_G | X], which averages over worlds and is
far smoother than any single q_G. So "hard posterior supervision forces overfitting" is false as a population statement
(04 Thm 1; the optimum is the smooth mixture). Overfitting is a finite-scene phenomenon (04 §6).

Robustness can be tested without changing the target (E):
* evaluate the Student on perturbed observations (receiver flips near shadow boundaries, sub-voxel jitter of obstacles, re-rendered);
* report the degradation curve separately from the primary NLL.

## 5. Literature: what applies directly, and what needs extra assumptions

See `LITERATURE.md`.

**Applies directly** [D]:
* the log score is the unique proper local score ([Bernardo1979]), and log score = entropy + KL ([GneitingRaftery2007]). This underlies 04 Thm 2;
* exact ABC / model-error equivalence ([Wilkinson2013]): soft kernels are noise models;
* amortised posterior estimation by maximum likelihood on joint samples converges to the posterior of the *sampling* distribution. Training on a proposal yields a proposal posterior ([PapamakariosMurray2016] Prop. 1, [Greenberg2019]). This is the SBI analogue of 03 §3;
* soft Bayes labels reduce risk variance ([Menon2021]).

**Needs assumptions that RIND3D does not satisfy** [D, E]:
* robust / tempered / coarsened posteriors ([Bissiri2016], [GrunwaldVanOmmen2017], [MillerDunson2019]) address model misspecification. The 1K simulator *is* the model, so there is nothing to be robust against unless a deployment shift is introduced;
* Gaussian-noise well-posedness theory ([Stuart2010]) presumes additive noise.

## 6. Recommendation

* **Primary:** the hard geometric posterior (Uniform(F) under the declared prior and selection law), trained by forward CE or KL (Design A/B in 09). It is the exact Bayes target of the simulated world, and it uses small decisive shadows.
* **Do not adopt** normalised-Hamming soft labels as a target or an auxiliary loss.
* **Optional ablation** (Design D): a small-ε flip-model auxiliary term (ε ≤ 10⁻³ per receiver), introduced **only** together with a perturbation evaluation. Its expected effect is robustness under deployment noise at a small cost in clean log score (toy: +0.02 nats at ε = 0.01).
* **If real sensor noise ever enters the task:** switch to Design C. Use an explicit, physically motivated noise process, ideally correlated near shadow boundaries, and relabel, because hard labels become invalid.
