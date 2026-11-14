# E — Student pilot: one-hot vs FastLabel supervision, balanced (B1K) vs corrected (T*) weighting

## 1. Protocol (fixed before training)

**Task.** One source-specific local response plus local geometry → the global 128³ source-cell distribution, in
multi-source physical worlds (`A_TASK_DEFINITION.md`).
* Input: pooled 4 × 32³ window features (lit-FREE, dark-FREE, SOLID and lit/dark-boundary fractions), plus origin and side.
* Not inputs: m, mask, k, ids, weights, status.
* Output: an exactly normalised hierarchical density, 16³ coarse × 8³ fine cells (= the label grid), on the cube minus the closed window box.

**Model** (`pilot/student.py`), identical for every arm:
* about 2.0 M parameters;
* 3D-CNN encoder to 8³ tokens with world-position encodings;
* 4096 coarse-cell queries (window-relative features), 3 cross-attention blocks (d = 128, 4 heads);
* zero-initialised coarse and fine heads, so the initial density is uniform on the support.

**Optimisation**, identical for every arm:
* AdamW, lr 4·10⁻⁴, β = (0.9, 0.98), wd 0.05, 300-step warm-up then cosine, gradient clip 1.0, bf16;
* 8,000 steps of batch 32 (about 2 epochs of the training pool); cross-attention blocks recomputed in the backward pass (memory). Fixed by a throughput benchmark before the first arm;
* no data augmentation: the families are not known to be symmetric, and augmentation would change the target population;
* batches are drawn with probability ∝ the arm's population weight after equal-side normalisation, with an unweighted loss (same expected gradient as the weighted loss);
* the batch sequence is keyed per (seed, step).

**Data** (scene-disjoint roles, `natural/natural_supplement.py`):
* **pilot-train** (≈ 700 train-split scenes): release mixed channels plus NAT-1K mixed channels.
* **pilot-select** (≈ 100 train-split scenes, hash-chosen): early stopping only, using one-hot NLL on 2000 examples of the arm's own population (release examples for B1K, NAT-1K examples for T*).
* **pilot-eval** (the 1K val split, ≈ 100 scenes): final evaluation, once per arm.
* The 1K test split is not opened.

**Training stratum:** boundary-mixed channels only. Solid-separated, all-lit and all-dark channels are held out for
`F_OOD_BENCHMARK.md`.

**Arms** (`pilot/arms.json`):

| arm | target | weighting (population) | data | purpose |
|---|---|---|---|---|
| FL_B1K | certified FastLabel / tau-pencil label field (label-or-one-hot fallback, counted) | ω̂ (B1K) | release | posterior supervision, current balanced law |
| OH_B1K | one-hot true-source cell | ω̂ (B1K) | release | one-hot baseline, same samples |
| FL_TREL | label field | ω̂/(n_L c) (T*, release only) | release, the same physical samples as FL_B1K | balancing correction with identical data |
| FL_MIS | label field | balance heuristic (T*) | release + NAT-1K | the recommended corrected design |
| OH_MIS | one-hot | balance heuristic (T*) | release + NAT-1K | one-hot under T* |
| FL/OH_B1K_sc25 | as FL/OH_B1K | B1K | 25% of the pilot-train scenes, all their channels | scene count |
| FL/OH_B1K_ex25 | as FL/OH_B1K | B1K | all scenes, 25% of the channels | channel count at an equal example budget |
| seed 1 of FL_B1K, OH_B1K, FL_MIS, OH_MIS | | | | seed variability |

**Metrics** (`pilot/evaluate.py`, `pilot/analyze.py`; pilot-eval; scene-bootstrap 95% intervals; paired differences
on identical examples):
* held-out one-hot NLL and information gain over the uniform-legal prior: the primary score, proper for p*;
* posterior-field CE and KL to the certified label, only where a certified label exists;
* mass on the label support;
* HPD coverage and calibration MAE;
* coarse Brier score;
* entropy and effective support;
* spatial reliability;
* strata: side, family, k, lit fraction.

Each metric is computed under **two evaluation populations**:
* T*: NAT-1K natural windows, equal weights within side;
* B1K: release windows, ω̂.

**Predictions** (from `RIND3D_Audit/04` and the E7 toy; to be confirmed or refuted):
1. FL ≤ OH in held-out NLL at equal steps, with the largest gap at L64.
2. With fewer scenes the FL advantage grows.
3. T*-weighted arms beat B1K arms on the T* evaluation, and the reverse holds on the B1K evaluation. The size of the crossover measures the real balancing tilt.
4. FL_TREL is noisier than FL_MIS (ESS 12% vs ≈ 60%).

## 2. Run log and results

**Runs.** All 13 arms trained to 8,000 steps (17–28 min each on the laptop RTX 3060) and were evaluated once on pilot-eval.
* One arm was rerun after a selection-set bug, which was fixed before any T* arm ran: FL_TREL.
* One arm was restarted after an interruption: FL_MIS_s1.
* Fallbacks: 0 label-or-one-hot fallbacks in every posterior arm (every training label exists).
* Evaluation sets:

| set | rows |
|---|---|
| T* | 6,321 natural boundary-mixed rows |
| B1K | 6,000 of 18,976 release rows (declared subsample) |
| all-lit | 3,000 of 9,110 rows, including all 180 references |
| all-dark | 2,043 rows (180 references) |
| solid-separated (natural) | 59 rows |
| solid-separated (release) | 440 rows |

* Full tables: `experiments/out/pilot_tables.md` (from `pilot_analysis.json`); seed summaries: `pilot_seed_effects.json`; train/held-out gaps: `pilot_train_gap.json`.

**Training pools** (pilot-train, 684 scenes):

| weighting | rows | Kish ESS | ESS share |
|---|---|---|---|
| B1K (ω̂) | 106,080 | 73,035 | 69% |
| T*, release only | 106,080 | 11,451 | 10.8% |
| T*, MIS (release + NAT-1K) | 124,502 | 41,225 | 33% |

### 2.1 Headline numbers

Held-out **information gain** = NLL(uniform-legal prior) − NLL(Student), in nats; higher is better. The prior NLL is about
14.55 nats. Brackets are 95% scene-bootstrap intervals; s0 / s1 are the two seeds.

| arm | T* (in-distribution) | B1K (in-distribution) | KL(q_G ‖ p) on T* | calibration MAE (T*) |
|---|---|---|---|---|
| **FL_MIS** (posterior, T*-MIS) s0 / s1 | **3.98 [3.86, 4.10] / 3.96 [3.83, 4.08]** | 3.90 / 3.85 | 6.37 / 6.38 | 0.035 / 0.032 |
| FL_B1K (posterior, balanced) s0 / s1 | 3.77 [3.66, 3.88] / 3.88 [3.78, 3.99] | 3.66 / 3.83 | 6.58 / 6.45 | 0.031 / 0.034 |
| FL_TREL (posterior, T* release-only) s0 | 3.81 [3.69, 3.92] | 3.72 | 6.53 | — |
| OH_MIS (one-hot, T*-MIS) s0 / s1 | 1.60 / 1.88 | 1.54 / 1.78 | 8.80 / 8.53 | 0.244 / — |
| OH_B1K (one-hot, balanced) s0 / s1 | 1.14 [0.99, 1.30] / 1.38 [1.21, 1.54] | 1.09 / 1.36 | 9.23 / 8.96 | 0.275 / — |
| FL_B1K_ex25 (all scenes, 25% of channels) | 3.80 | 3.72 | 6.55 | — |
| FL_B1K_sc25 (25% of scenes, all channels) | 3.04 | 2.96 | 7.31 | 0.211 |
| OH_B1K_ex25 | 1.42 | 1.38 | 8.96 | — |
| OH_B1K_sc25 | **−0.62** (worse than the prior) | −0.62 | 11.01 | — |

### 2.2 FastLabel probability-field supervision vs one-hot (same samples, same model, same budget)

**Effect size** (seed means, T*):

| comparison | gain difference |
|---|---|
| FL_B1K − OH_B1K | **+2.56 nats** |
| FL_MIS − OH_MIS | **+2.23 nats** |
| FL − OH, 25% of the channels | +2.39 |
| FL − OH, 25% of the scenes | +3.66 |

* Every paired difference has a CI far from 0, for example OH_B1K_s0 − FL_B1K_s0: NLL +2.63 [2.49, 2.77].
* The seed-to-seed spread is 0.06–0.14 nats.

**Calibration.** HPD-coverage MAE is 0.03 (FL) vs 0.24–0.28 (OH).

**Mechanism: memorisation.** One-hot Students reach their best selection NLL at step 1,000–2,000 and then overfit. Gains
on training-scene rows vs held-out rows:

| arm | training-scene rows | held-out | gap | training NLL at step 8,000 |
|---|---|---|---|---|
| FL_B1K | 4.22 | 3.69 | 0.52 | — |
| OH_B1K (best checkpoint) | 3.15 | 1.10 | 2.05 | 5.7 nats |
| OH_B1K_sc25 | 4.85 | −0.73 | 5.6 | 0.8 nats |

* Posterior targets spread mass over the feasible region and remove most of the variance of the one-hot target. This is the Rao–Blackwell effect predicted by `RIND3D_Audit/04` and the E7 toy, and it is far larger here than in the toy.
* **Caveat.** No symmetry augmentation was used: the families are not known to be invariant, and augmentation would change the target population. Augmentation (as in the 2026-11-10 study) is the standard remedy for one-hot memorisation. The one-hot deficit at larger scale and with augmentation is therefore expected to shrink. The pilot shows the direction and the mechanism, not the asymptotic gap.
* **Per side** (T* gain; FL is ahead at every side):

| side | FL_B1K | OH_B1K |
|---|---|---|
| L64 | 3.31 | 0.90 |
| L128 | 3.84 | 1.19 |
| L256 | 4.16 | 1.35 |

### 2.3 Balanced (B1K) vs corrected (T*) weighting

**Same physical samples, release only.**
* FL_TREL − FL_B1K (seed 0) on T*: NLL −0.036 [−0.077, +0.009] (not significant); KL −0.050 [−0.084, −0.013].
* On the B1K evaluation it is also slightly better: NLL −0.060 [−0.107, −0.014].
* Despite its 10.8% ESS, ω/c weighting costs nothing measurable.

**Recommended design** (release + NAT-1K, balance heuristic).
* FL_MIS − FL_B1K (seed means) on T*: **+0.14 nats** gain, positive in both seed pairs (+0.21, +0.07).
* On the B1K evaluation also: +0.13.
* KL to the labels drops by 0.06–0.21 (seed pairs).
* For one-hot: +0.48 (it benefits more from the extra natural windows).

**Reading.**
* The predicted crossover (T* arms better on T*, worse on B1K) **did not appear**. T*-weighted Students are as good or better on *both* evaluation populations.
* At this scale (684 scenes, 2.0 M-parameter Student, 8k steps), the hidden balancing tilt is below the resolution of the experiment. It is smaller than the 0.06–0.14-nat seed spread. This matches E16, which also found no detectable tilt.
* MIS's advantage comes mainly from the additional natural windows: about 18k more examples on the same scenes. Part of it may also be the target correction; the two cannot be separated with these arms.
* **Consequence.** The corrected population can be adopted as the default. It is scientifically the declared target, it costs no measurable performance, and with the natural supplement it is slightly better. Severe release-only importance weights are not needed: the balance-heuristic design keeps every weight ≤ 1/n_p.

### 2.4 Scene count vs channel count (equal 25% example budgets)

* FL: 25% of the channels over all scenes reach **3.80**, the same as the full data (3.77–3.88). 25% of the scenes reach **3.04**.
  * Paired difference sc25 − ex25 on T*: NLL +0.77 [0.68, 0.86].
* OH: 1.42 vs −0.62.
* **Independent scenes, not channels, are the effective sample size.**
  * Within a scene, extra windows and channels add little at this budget.
  * With few scenes, even posterior supervision memorises: FL_B1K_sc25 has a training-scene gain of 7.84 vs 2.97 held-out. Its train-split scenes outside the subset score 2.92, the same as held-out.
* This supports 100K scenes × 12 windows over more windows per scene (`H_100K_RECOMMENDATIONS.md`).

### 2.5 Multimodality and spatial calibration

**In-distribution, FL Students:**
* the certified support receives 200–270× the prior's mass;
* the Student's information content tracks the full-world label's, ρ(KL(p ‖ u), KL(q_G ‖ u)) = 0.63–0.65 [CI ±0.02];
* HPD coverage: mean absolute deviation from nominal 0.03 over the 10–90% levels; the largest deviation is 0.056, a slight under-coverage at the 20–30% levels.

**One-hot Students:**
* support-mass ratio 18–28×;
* coverage MAE 0.24–0.28, overconfident.

Per-arm reliability histograms (16³ coarse cells) are stored in `pilot_analysis.json`.

### 2.6 What the Student does and does not learn

**Information recovered** (`experiments/out/pilot_info_recovered.json`):
* the full-world posterior carries 10.3 nats about the source on T* windows;
* the best Students recover 3.9–4.0 nats (37–39%).

The rest is the uncertainty caused by hidden global geometry: I(S; G | X) plus finite-capacity loss. **The Student is not expected to reach the full-world label.**
* Its KL to individual labels stays at 6.4 nats.
* That distance is what a Student that knows only X *should* keep. KL to a single world's posterior is not a target.
* At full scale (684 scenes) its training loss on posterior targets stayed at about 10.3 nats, consistent with the H(S | X) floor and no memorisation. With 171 scenes it fell to 8.1 nats: memorisation of the training worlds (§2.4).
