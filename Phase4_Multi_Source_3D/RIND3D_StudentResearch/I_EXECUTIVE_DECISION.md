# I — Executive decision (2026-11-13)

Evidence is in A–H of this directory. **M** = measured; **A** = proved; **E** = judgement or recommendation.

## 1. Which multi-source population defines the primary Student task?

**T\*: natural multi-source worlds, single-channel examples** (`A_TASK_DEFINITION.md`):
* G ~ π; K ~ κ, declared and independent of G and S;
* S_i i.i.d. Uniform(D(G));
* source-separated responses;
* natural admissible windows: uniform origins, no source in the closed box, FREE > 0;
* every mixed channel is one example;
* the Student sees one channel plus local geometry and predicts P(S_j | X).

**Not chosen:**
* **N1, one-source worlds.** Not a RIND3D population. Under natural sampling it differs from T* only by the other-source exclusion factor: ≤ 1% TV (A + M, E14).
* **B1K, the release's m-balanced law.** Its thinning depends on hidden channels.

**Labels.** Uniform(F) is the exact per-world posterior for T* (Theorem A1). No label changes.

## 2. Which sampling and weighting scheme?

**For the 1K:**
* Train on the release plus the NAT-1K natural-window supplement, with balance-heuristic weights 1/(n_L c/ω̂ + n_p).
  * They are bounded (≤ 1/n_p).
  * ESS is 33% of 124k rows.
  * ω̂ = combined_rb.
* Evaluate on NAT-1K natural windows with no weights.

**Measured** (seed means, `E_STUDENT_PILOT.md` §2.3):
* this design is the best arm on both the T* and the B1K evaluations: +0.14 and +0.13 nats over balanced training;
* release-only ω/c weights (ESS 11%) are not better than balanced ones (−0.04 n.s.);
* the hidden balancing tilt is below the experiment's resolution (< 0.1 nats). The tabular E16 test agrees.

**For the 100K:** fix it upstream (refined in `J_COMPLEXITY_COVERAGE.md`: declared κ with ≥ 50% K = 4, plus a minimal L64 coverage stream for complexity coverage).
* Natural windows (no m-thinning).
* A fixed declared κ.
* ω̂ only for the per-scene quota normalisation: T* ESS 85–94% instead of 10–24% (E15).

## 3. Is FastLabel probability supervision preferable to one-hot under our constraints?

**Yes, decisively** (M):
* **+2.2 to +2.6 nats** held-out information gain at equal data, model and steps;
* +3.7 nats with 25% of the scenes;
* calibrated (coverage MAE 0.03 vs 0.24–0.28);
* no memorisation at full scale. One-hot Students peak after 1–2k steps and then memorise their training worlds.

**Caveat.** No symmetry augmentation was used, and augmentation is the usual one-hot remedy. The size of the gap will shrink at larger scale. Its direction and mechanism (variance removal) are robust.

**Cost.** The 1K labels took 4.33 h on the laptop, already paid.

## 4. Does mixed-only training generalise to all-lit and all-dark observations?

**Partially** (M; `G_GENERALIZATION_AND_EXTREMES.md` §1, against the protocol fixed before training):

| stratum | result |
|---|---|
| solid-separated mixed | **yes**: +0.7 to +1.0 nats; calibrated; 2.6–4× the prior's mass on the feasible region; adaptivity ρ 0.3–0.4 |
| all-dark | **partly**: +0.20–0.26 nats; calibrated; 1.4–1.6× the prior's mass on the feasible region; weak adaptivity; 12–18% of the available information |
| all-lit | **no measurable gain**: near-prior output, slightly overconfident. The full world itself carries only 0.36 nats here, so the near-prior answer is close to correct, but the Student does not detect the constraining all-lit cases |

## 5. Meaningful geometric reasoning, or a trivial prior?

* **Solid-separated: meaningful.** Different response structures receive appropriately different, calibrated, geometry-concentrated predictions.
* **All-dark: a real but coarse geometric rule** ("mass behind visible solids").
* **All-lit:** the Student behaves like its prior, which is mostly the right answer.
* **The decisive contrast** is that one-hot Students with the same architecture and data are confidently wrong on all three unseen types, by up to −1.75 nats. Appropriate uncertainty on unseen observations is a learnt consequence of posterior supervision.
* **This is evidence of transferable geometric reasoning, not proof of complete physical understanding (E).**

## 6. Is the Student ready for larger training experiments?

**Yes, with posterior supervision and the T* design** (E). Conditions:
* **More independent scenes.** Scenes, not channels, are the effective sample size (M: 25% of the channels over all scenes matches the full data; 25% of the scenes loses 0.77 nats).
* **Longer training.** FL Students were still improving at 8k steps.
* **≥ 3 seeds per arm.** The seed spread is 0.06–0.14 nats, the size of the weighting effects.
* **Decide on augmentation** after testing whether π is invariant under the cube symmetries, per family.
* **Keep extremes out of training.** Use them as the standing OOD benchmark.
* **Hardware.** This laptop GPU fits the 2 M-parameter model at batch 32 with block recomputation. Larger models need a bigger GPU. That is not authorised here.

## 7. Problems to resolve before 100K production

From `H_100K_RECOMMENDATIONS.md`:
* **N1** declare T* in the contract;
* **N2** no m-thinning of the single-channel population;
* **N3** a fixed declared κ;
* **N4** an ω̂ estimator with stored rate-stream m-counts;
* **N5** store everything needed to rebuild any weight;
* **N6** δ per label = 0.01 / N_planned;
* **N7** port the M2 loader.

Also:
* S6 review before any all-dark labelling;
* soundness asserts per label;
* budget the solid-separated stratum: 2.5% of labels, ~23% of 1K label compute;
* extremes stay unlabelled in production (cost gate failed: 25–100× a mixed label);
* the 12-window quota is unchanged.

## Status of the brief's deliverables

| | deliverable | where | status |
|---|---|---|---|
| A | revised task definition | `A_TASK_DEFINITION.md` | done |
| B | sampling and weighting decision | `B_SAMPLING_WEIGHTING.md` | done |
| C | corrected loader | `rind3d_1k/loader.py`, `C_LOADER.md`, `tests/test_loader.py` | done; tests pass |
| D | 1K posterior backfill | `D_BACKFILL_REPORT.md` | done: 159,802/159,802 certified, verified |
| E | Student pilot | `E_STUDENT_PILOT.md` | done: 13 arms |
| F | extreme-response OOD benchmark | `F_OOD_BENCHMARK.md`, `ood/` | done: 360 certified references |
| G | generalisation analysis (+ cost-gated extremes) | `G_GENERALIZATION_AND_EXTREMES.md` | done |
| H | 100K recommendations | `H_100K_RECOMMENDATIONS.md` | done; the 100K branch was not modified |
| I | executive decision | this file | done |

**Not done, and why:**
* Controlled counterfactual ensembles of hidden worlds (audit P3): they need a geometry editor; their absence is recorded as a limitation.
* An exact-rational version of S6 and its production review.
* The 1K test split stays unopened for future final evaluations.
