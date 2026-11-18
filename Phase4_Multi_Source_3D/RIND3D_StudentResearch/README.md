# RIND3D Student research phase (2026-11-13): multi-source task, loader, 1K backfill, Student pilot, OOD extremes

The brief is "RIND3D — Scientific Decisions, Loader Corrections, 1K Backfill, and Student Generalization Research",
with its addendum on cost-gated extreme responses. **Start with `I_EXECUTIVE_DECISION.md`.**

> **Terminology (2026-11-18).** These records keep their original light/shadow wording: lit = response-active, dark = response-inactive (so "all-dark S6" is the all-inactive prototype); see `../docs/RIND3D/GLOSSARY.md`.

| file | content |
|---|---|
| `A_TASK_DEFINITION.md` | multi-source worlds, the single-channel target T*, four populations compared, labels exact for T* |
| `B_SAMPLING_WEIGHTING.md` | the natural m-distribution verified; weight estimators; ESS; NAT-1K supplement; 100K design study; E16 |
| `C_LOADER.md` | M2 loader correction (`RIND3D_Dataset1K/rind3d_1k/loader.py`) |
| `D_BACKFILL_REPORT.md` | 1K FastLabel backfill and NAT-1K labels: run, verification, cost by type |
| `E_STUDENT_PILOT.md` | 13-arm pilot: one-hot vs posterior, B1K vs T*, scenes vs channels |
| `F_OOD_BENCHMARK.md` | OOD protocol, fixed before training |
| `G_GENERALIZATION_AND_EXTREMES.md` | OOD results and interpretation; S6 all-dark fix; cost gate for extremes |
| `H_100K_RECOMMENDATIONS.md` | changes before the 100K contract is frozen |
| `I_EXECUTIVE_DECISION.md` | answers to the brief's seven questions; deliverable status |
| `J_COMPLEXITY_COVERAGE.md` | follow-up: natural T* plus a minimal complexity-coverage stream for the 100K (E18, E19) |

**Code:**

| directory | contents |
|---|---|
| `experiments/` | E13–E17 analyses and outputs |
| `natural/` | NAT-1K generator, task count, labelling launcher |
| `ood/` | declared OOD sample and labelling driver |
| `s6/` | all-dark solid-interior fix prototype, its runs and validation |
| `pilot/` | features, model, training, evaluation, analysis, tables |

**Data** (outside the repository, under `RIND3D_1K_data/`):

| directory | contents |
|---|---|
| `release_1k_v1` | labels backfilled; physical data unchanged |
| `natural_supplement_v1` | NAT-1K |
| `pilot_features_v1` | 194,532 rows, 24 GB |
| `pilot_runs_v1` | checkpoints and evaluations |
| `backfill_verification_v1` | full verification outputs, SHA256SUMS |
