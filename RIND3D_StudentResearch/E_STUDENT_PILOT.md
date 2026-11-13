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

(filled in after the runs)
