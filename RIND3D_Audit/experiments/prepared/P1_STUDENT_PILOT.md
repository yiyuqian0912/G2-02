# P1 — Controlled Student pilot: one-hot vs posterior field vs label-or-one-hot (PREPARED, NOT RUN)

**Purpose.** Measure on real 1K data what 04 predicts from theory and the toy (E7):
* the same population optimum for all arms;
* lower variance and better finite-data NLL for posterior fields;
* no bias from the label-or-one-hot fallback;
* the size of the target-population effect (ω vs ω/c).

## Data
* Labels: `p1_label_subset.py` → 160 train + 40 val scenes, all informative channels. About 32k FastLabel labels, ≈ 45 min on the full laptop (15 shards).
* Test: 40 **val** scenes as the held-out evaluation. The 1K test split is not opened. The 2026-11-10 study already used its own test split once.
* Sources: `privileged/sources.json` (one-hot targets, evaluation only).

## Student (fixed across arms)
* The 2026-11-10 architecture (Drive `student_training_study/code/s3d/`, see `../prior_student_study_notes.md`):
  * input: 4×32³ pooled channels (lit-FREE, dark-FREE, SOLID, boundary) + origin + side;
  * output: a hierarchical 16³ × 8³ exactly normalised density over the cube minus the window box.
* Its 16 × 8 = 128 grid equals the label grid, so label cell masses map 1:1 onto the fine cells.
* 8k steps, batch 64, AdamW 4e-4 cosine, 48-element symmetry augmentation, 3 seeds per arm. Same steps and same scenes for every arm.
* No m, mask, k, or scene ids as inputs.

## Arms (identical data order and weights within a seed)

| arm | target T_e | weight |
|---|---|---|
| A | one-hot e_S (cell of the true source) | ω (combined, side-normalised) |
| B | label q_e (all certified statuses accepted) | same |
| C | q_e where a label exists, e_S otherwise (simulate missingness: drop 30% of labels, selected by label width) | same |
| B′ | label q_e | ω/c(m) (natural target N1) |
| B″ | label q_e, default loader filter (`accept_statistical=False`) | ω. Expected to be biased: the M2 demonstration |

## Metrics (held-out val scenes, scene-bootstrap 95% intervals)
1. One-hot NLL under ω and under ω/c. Primary; both are reported because they target different populations.
2. Label CE (control variate) and the label-CE minus label-entropy gap.
3. Cluster-aware HPD coverage, as in E4b (one example per source, repeated draws).
4. Regional Brier on 8³ blocks.
5. Strata: side, family, lit-fraction decile, solid-separated.
6. Training curves: NLL vs steps, and the gradient-norm variance per step (Rao–Blackwell check).

## Predictions (from 04, to be confirmed or refuted)
* B ≤ A in held-out NLL at equal steps. The gap is largest at L64 (mean 1 − Σq² is 0.97) and smallest at L256 (0.42).
* C ≈ B (unbiased fallback).
* B″ worse than B, most at L64.
* B′ better than B on the ω/c-weighted NLL and worse on the ω-weighted NLL. The size of this crossover measures the real-data balancing tilt.

## Cost
* Labels ≈ 45 min laptop.
* Training: 5 arms × 3 seeds × 8k steps = 15 runs of the 2026-11-10 configuration. The per-run GPU time is not verified here: take it from that study's logs on Drive, multiply by 15, and add about 10% for evaluation.
