# F — Extreme-response OOD benchmark (protocol fixed on 2026-11-13, before any Student was trained)

## Purpose

Experiment B tests whether a Student trained **only on boundary-mixed channels** infers sensible source distributions
on response types it never saw:
* all-lit;
* all-dark;
* solid-separated mixed (lit and dark FREE regions separated only by SOLID).

The benchmark is evaluation-only.
* No OOD example is used for training, early stopping, model selection or hyperparameters.
* Every Student is evaluated on it **once**, after its training run has finished.

## Population and data

| item | choice | why |
|---|---|---|
| scenes | pilot-eval = the 1K **val** split (about 100 scenes). Never used for training or selection. The 1K test split stays closed. | scene-disjoint from training |
| windows | NAT-1K natural windows of those scenes (16 proposals per side, stream 57000 + L) | natural frequencies. The release's own all-lit and all-dark channels exist only next to a hidden mixed channel (audit M4), so they are not used. |
| channels | every channel of every admissible natural window, by X-measurable type | the T* stratum of that type |
| additional (C only) | release solid-separated channels of pilot-eval scenes | they carry certified backfill labels |

**Strata** (all computed from X, or from privileged data for analysis only):

| id | brief case | definition |
|---|---|---|
| A0 | all-lit, unobstructed window | all-lit, no SOLID voxel in the window |
| A1 | all-lit with occluders in the window | all-lit, SOLID present |
| A-ig | all-lit, hidden occluders constrain vs. don't | split by the reference posterior's information gain KL(q_G ‖ u) (privileged, analysis only) |
| B1 | all-dark with a visible local wall | all-dark, SOLID present in the window |
| B0 | all-dark caused by geometry outside the window | all-dark, no SOLID in the window |
| C | mixed, no 6-adjacent lit/dark FREE pair | solid-separated |
| D | disconnected feasible regions | reference support with ≥ 2 26-connected components at 128³ (where a reference exists) |
| E | source near a geometric boundary | SOLID within 32 voxels (∞-norm) of the true source (privileged, analysis only). Within 8 voxels holds for only 30 of the 3110 sources (1%), so 32 was fixed before training; 156 sources qualify |
| H | the same local X under different hidden worlds | every natural example is a draw from P(G \| X). The Student is scored against the marginal through proper scores (below). No controlled counterfactual ensemble was built (limitation). |

## References and metrics

**Ground truth that needs no constructed reference.** The true source S_j of each held-out example is a draw from the
T* conditional P(S | X) of its stratum.
* The held-out one-hot NLL is a strictly proper score: its expectation is minimised exactly by the true conditional p*(s | X).
* p* is the marginal over the hidden worlds compatible with X.
* So "NLL relative to the prior" quantifies the Student against the correct marginal posterior. It does not require that posterior to be computed, and it is the primary OOD metric.

**Certified full-world references, where available.** For a declared random sample per (side, type) of 60 channels:
* all-lit: FastLabel v3.4;
* all-dark: the S6 prototype (solid interiors removed from B; `s6/fastlabel_s6.py`);
* solid-separated: the backfill labels and FastLabel on NAT-1K.

They give q_G = Uniform(F_W(G)), the posterior **given the complete geometry**. The Student is not expected to match an
individual q_G. These references are used for:
* the mass the Student puts on F_W(G) (the region physically consistent with the full world);
* the Student's information gain versus the reference's information gain across examples (adaptivity);
* label CE (also a proper score for p* in expectation, with lower variance).

Tasks that do not certify are recorded as unresolved. **No uniform or heuristic reference is substituted.**

**Metrics, per stratum** (scene-bootstrap 95% intervals; 1 scene = 1 cluster):

| metric | definition | passes if |
|---|---|---|
| information gain over the prior | NLL(uniform-legal prior) − NLL(Student) | > 0, with the CI excluding 0 |
| calibration | randomised HPD rank of the true cell; coverage at 10…90%; mean absolute error | MAE ≤ 0.10 (the in-distribution MAE is reported alongside) |
| mass on the reference support | p_Student(supp q_G), compared with the prior's u(supp q_G) | ratio > 1 |
| adaptivity | Spearman ρ between the Student's KL(p ‖ u) and the reference's KL(q_G ‖ u) across examples | ρ > 0, with the CI excluding 0 |
| entropy and effective support | H(p), exp H(p) | reported, not a criterion (entropy alone proves nothing) |
| spatial reliability | predicted coarse-cell mass vs. hit frequency (16³ cells, log bins) | reported |

**Reading the outcome.**
* **A constant-prior Student fails** information gain (it is 0) and adaptivity (ρ undefined or 0). It passes calibration trivially, so calibration alone is never sufficient.
* **"Meaningful geometric generalisation"** requires all four criteria on a stratum. Partial passes are reported as partial.
* **A failure is a documented limitation, not hidden.**

## Analytic and controlled fixtures

The audit's seven exact fixtures (E8: tau-pencil vs independent Monte Carlo, N = 32) establish the physics references
for all-lit, all-dark and solid-separated observations. They are not at the Student's input scale (1024³ worlds,
L ≥ 64), so they validate the reference engines, not the Student.

Controlled counterfactual ensembles of hidden worlds compatible with one X (audit P3) need a geometry editor that
preserves the window response. They were **not built** in this phase. Case H is evaluated only through proper scores on
natural draws, which are exact samples of P(G | X). A controlled ensemble would not be representative of P(G | X)
anyway.
