# RIND3D Scientific Audit — Progress

Started 2026-11-13 (fresh restart). Compute limits were relaxed by the main agent mid-run (4 CPU processes, 15 min, short GPU).

## Done (all deliverables written)
- 01–10 deliverables, README.md, LITERATURE.md (sub-agent), experiments/prior_student_study_notes.md (sub-agent).
- Experiments E1, E1b, E2/E3, E4, E4b, E4c, E4d, E5, E5b, E6, E7, E8 (7 fixtures), E9, E10, E11 (GPU, 42 tasks),
  E12 (GPU, 18 labels). Scripts in experiments/, outputs in experiments/out/.
- Prepared, not run: P1 Student pilot (prepared/P1_STUDENT_PILOT.md, prepared/p1_label_subset.py); P3, P4 (08 §3).

## Open / next (for the main agent)
- Run P1 (labels about 45 min laptop + 15 training runs) to measure the real-data Rao–Blackwell gain and balancing tilt.
- Decide the target population T (M1) and the loader default (M2) before any Student training.
- Fix S6 (solid-interior exclusion), then rerun E11 (P4).
