# RIND3D Training-Ready v0.3 candidate

This directory is a separately versioned candidate that turns the accepted v0.2 pipeline into training-ready, Teacher-compatible data. Three things are unchanged and are only read:
- the v0.2 canary;
- the frozen contract `rind3d-production-contract-v0.2`;
- RIND3D v0.1 and all prior research.

Preservation is proven by SHA256 trees in `outputs/preservation/compare.json`.

Start with `RIND3D_TRAINING_READY_DECISION.md`.

| Deliverable | File |
|---|---|
| Decision and go/no-go | `RIND3D_TRAINING_READY_DECISION.md` |
| Training contract (normative) | `RIND3D_TRAINING_CONTRACT.json` |
| Student / target schema | `RIND3D_STUDENT_TARGET_SCHEMA.md` |
| Dataset ↔ Teacher interoperability | `RIND3D_DATASET_TEACHER_INTEROP.md` |
| Loader benchmarks | `RIND3D_LOADER_BENCHMARKS.json` |
| 102 vs 256 vs dynamic vs hybrid | `RIND3D_102_VS_256_WINDOW_STUDY.md` |
| Scale cost | `RIND3D_TRAINING_SCALE_COST.json` |
| Source domains / selection law | `docs/SOURCE_DOMAINS.md`, `docs/SELECTION_LAW.md` |
| Reproducible commands | `docs/REPRODUCE.md` |
| Next-campaign proposal (not launched) | `configs/next-campaign-sb-10k-v0.3.json`, `docs/NEXT_CAMPAIGN_PROPOSAL.md` |
| Evidence | `EVIDENCE_MANIFEST.json`, `outputs/RIND3D_TRAINING_READY_REVIEW.zip` |

## Code (`rind3d_training/`)

| Module | Purpose |
|---|---|
| `index` | obs-index-v0.3; P102 is the exact v0.2 prefix |
| `selection` | selection law and population weights |
| `export` | two-view export: Student and privileged |
| `student` | NumPy/zstd Student reader, scene-balanced sampler, GPU unpack |
| `teacher` | TeacherPhysics v0.3 |
| `phase3_adapter` | Phase 3 Teacher adapter |
| `leakage` | file-level audit |
| `study` | window study |
| `bench_fields`, `bench_student`, `blocksparse` | benchmarks |
| `interop_audit` | Teacher interoperability audit |
| `scale_cost`, `report`, `evidence`, `preservation` | reports and evidence |

Tests: `tests/test_training_ready.py`, run on the actual exported files.

Nothing here is committed or pushed. No production campaign was launched.
