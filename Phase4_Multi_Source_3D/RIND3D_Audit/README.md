# RIND3D Scientific Foundations Audit (2026-11-13)

> **Status 2026-11-13.** M1 resolved by declaring the target T* (`../RIND3D_StudentResearch/A_TASK_DEFINITION.md`); M2 fixed
> (the loader accepts every certified status, explicit one-hot fallback; `C_LOADER.md`); M3 fixed (δ = family_delta /
> n_tasks); S6 prototyped (`../RIND3D_StudentResearch/s6/`). Statements below about the loader default describe the
> state at audit time. Paths under `RIND3D_Research/` are in the archive tag `archive/rind3d-research-2026-11-13`.

This is an independent audit of the RIND3D program, written against the user brief `BRIEF.md`. Every claim carries an
evidence class: **A** proved, **B** verified in code (file:function), **C** measured (configuration and uncertainty
given), **D** literature (`LITERATURE.md`), **E** hypothesis or recommendation.

## Deliverables

| # | file | content |
|---|---|---|
| 1 | `01_MODEL_SPEC.md` | the generative process as implemented (brief §2 items 1–18), random variables, factorisation, graphical model, the four distributions |
| 2 | `02_SELECTION_AUDIT.md` | source prior (PIT tests), k and family, the effective-layer balancing tilt, weights and ESS, label availability, response-type counts |
| 3 | `03_POSTERIOR_VALIDITY.md` | Bayes derivation of Uniform(F) with hypotheses H1–H8; where it holds (label level) and where it fails (flattened Student level) |
| 4 | `04_LEARNING_THEORY.md` | weighted CE optimum; one-hot ≡ posterior as functionals; failure cases; entropy + MI decomposition; Rao–Blackwell gain; label-error propagation; overfitting; evaluation |
| 5 | `05_HARD_VS_SOFT.md` | hard vs soft vs hybrid; normalised Hamming is incoherent; toy measurements; uncertainty types; recommendation |
| 6 | `06_EXTREME_RESPONSES.md` | all-lit, all-dark, solid-separated: definitions, exact 3D fixtures, real-data frequencies, FastLabel probe, inclusion policy |
| 7 | `07_TV_CERTIFICATE.md` | what the deterministic and statistical certificates guarantee; critical reading of CERTIFICATE_AUDIT.md; target vs approximation error |
| 8 | `08_EXPERIMENTS.md` | index of all experiments (E1–E12), configurations, results, prepared heavier experiments (P1–P4) |
| 9 | `09_CONTRACT_vNEXT.md` | design comparison A–G; MUST FIX / SHOULD CHANGE / OPTIONAL / KEEP; 100K recommendations |
| 10 | `10_EXECUTIVE_DECISION.md` | answers to the nine executive questions |

Supporting files:
* `LITERATURE.md`: primary sources with verification notes (prepared by a literature sub-agent; key results read from the PDFs);
* `experiments/prior_student_study_notes.md`: extraction and critique of the 2026-11-10 Student/Teacher study (from the local review ZIP);
* `experiments/`: scripts (`e*.py`, `toyworld.py`, `e11/`), outputs (`out/`), prepared experiments (`prepared/`);
* `PROGRESS.md`: work log.

## Headline findings

* **Label level: correct.** Uniform(F) is the exact posterior of a channel given G under the ω-weighted 1K law (A; re-derived). Without ω it is not (toy: mean TV 0.022, max 0.34).
* **Student level: MUST FIX.**
  * Effective-layer balancing thins windows by up to 100× according to the *hidden* channels. ω cannot undo this, so a flattened single-channel Student learns a balanced-population posterior, not a natural one (toy: TV 0.03 mean, 0.09 max). Declare the target; use ω/c(m) or a multi-channel Student.
  * The default loader drops `certified_stat` labels. That is a privileged selection (67% of FastLabel labels; only 6% of L64 labels kept).
* **One-hot NLL and posterior-field CE are the same population functional** (A). Posterior fields cut variance: on real labels the removed gradient-variance term 1 − Σq² is 0.97 at L64, 0.42 at L256. Their label error at the optimum is ≤ 0.051 nats.
* **The hard posterior is right for this noise-free world.** Normalised-Hamming soft targets ignore decisive small shadows (toy: 59% mass on infeasible all-lit candidates).
* **All-lit and all-dark are informative.** The engine handles them, but all-dark certificates fail because solid interiors stay inside the outer bound.
* **The FastLabel simultaneous 1% claim is valid** for the 1K informative labels. Its δ budget is hard-coded and must be re-planned for any extension. Soundness (F ⊆ B) is an untested assumption that deserves independent asserts.
