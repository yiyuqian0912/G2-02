# Cleanup manifest: `yushu/3d` consolidation of 2026-11-13

## 1. References

| reference | value |
|---|---|
| pre-cleanup state | `yushu/3d` at `0270712` |
| archive tag (annotated, pushed) | `archive/rind3d-research-2026-11-13`. Tag object `f0271b6`, points to `0270712` |
| archive branch (pushed) | `archive/yushu-3d-research-2026-11-13` at `0270712` |
| working branch | `yushu/3d-consolidation`, fast-forwarded into `yushu/3d` after validation |
| consolidation commits | `52ac18c` (1/4, removals), `6697c6d` (2/4, configuration), 3/4 docs, 4/4 handoff. Final IDs are in `VALIDATION_RECORD.md` §0 |

**Recovery:**

```bash
git fetch origin --tags
git log --oneline -1 archive/rind3d-research-2026-11-13
git show archive/rind3d-research-2026-11-13:<path>
git checkout archive/rind3d-research-2026-11-13 -- <path>
```

The archive holds 505 tracked files under the removed paths; recovery is checked in `VALIDATION_RECORD.md`.

## 2. Removed from the active branch (585 tracked files; all in the archive)

| path | files | why removed | dependency check |
|---|---|---|---|
| `RIND3D_Research/` | 253 | production v0.2 (`rind3d_production`), training v0.3 (`rind3d_training`), posterior-label studies (study1_cepl, phase1_tau_pencil, phase2_gpu, rpl_shared). Superseded by `RIND3D_Dataset1K` (pipeline, loader) and `RIND3D_Posterior` (packaged engines) | no import from any active module on `yushu/3d` or `yushu/3d-100k-h20` (grep) |
| `RIND3D_Generator/experiments/gaussian_geometry/` | 200 | Gaussian-B2 geometry research. Closed for RIND3D; it continues on its own branch `yushu/gaussian-b2-native` | not in `provenance/ORIGINAL_SOURCE.json`; referenced only by its own docs |
| `RIND3D_Generator/docs/GAUSSIAN_*.md`, `docs/assets/` | 33 + 22 | reports and figures of that line | assets referenced only by GAUSSIAN docs |
| `RIND3D_Posterior/experiments/fastlabel/{runs/, run4*.jsonl, eval*.jsonl, e1.jsonl, refs.jsonl, jobs*.txt, random120.txt, gatea/old_flow/}` | 67 | benchmark outputs and job lists behind the numbers in `REPORT.md` | outputs only; no code reads them |
| `RIND3D_Posterior/experiments/twophase/{sweep*, prof1.py, emitprof.py, bench_*, subset_variants.jsonl}` | 8 | profiling and benchmark files of the two-phase research | `twophase.py` kept: the Gate A harness imports it |
| `RIND3D_StudentResearch/{chain_stdout.log, post_backfill_chain*.sh, watch*.sh, RESUME.md, chain/chain.log, pilot/{final_eval,rolling_eval,finish_pilot,train_gap}.sh, s6/{alldark_remaining,rerun}.txt}` | 13 | one-off orchestration and status files of the 2026-11-13 runs | reproducible entry points (`run_arms.py`, `final_analysis.sh`, `run_all_s6.sh`) kept |

Also cleared from the working tree: bytecode caches and empty directories under the removed paths (regenerable).

**My own untracked run leftovers were moved, not deleted:**
* 13 files, the pid and log files of earlier runs;
* destination `RIND3D_1K_data/worktree_leftovers_20261113/`, with the list in `MOVED_FILES.txt`.

**Untouched pre-existing local work:**
* the root `.gitignore` modification;
* uncommitted `Phase3_Single_Source_3D` adaptive-teacher work (another session);
* `RIND3D_Generator/probe/`.

## 3. Retained, consolidated, changed

**Retained as is (algorithms, formats, tests):**
* `RIND3D_Generator` (byte-stable imported source);
* `RIND3D_Posterior/src` and tests;
* `RIND3D_Dataset1K` packages and tests;
* `RIND3D_Audit`;
* `RIND3D_StudentResearch` docs A–J and tools;
* the FastLabel research docs and Gate A harness.

**Preferred implementations, unchanged in place** (rationale in `RIND3D_IMPLEMENTATION_FACTS.md`):

| subsystem | preferred implementation |
|---|---|
| geometry, rendering, storage | `rind3d` |
| exact labels | `rind3d_posterior.label_window` |
| FastLabel | `rind3d_1k.fastlabel`, the production copy of the research pipeline v3.4 |
| loader | `rind3d_1k.loader` (populations, estimators, explicit fallback) |
| natural windows | `RIND3D_StudentResearch/natural/natural_supplement.py`, kept in place: it is a 1K-derived tool, and the 100K will sample natural windows natively |
| Student | `RIND3D_StudentResearch/pilot/` (reference implementation, pilot scale) |

**Changed (configuration only, `6697c6d`):**
* hard-coded `/home/...` repository paths became file-relative;
* data paths come from `RIND3D_1K_DATA`, and interpreters from `RIND3D_LABEL_PY` / `RIND3D_TORCH_PY`;
* shell entry points source `RIND3D_HANDOFF/env.sh`;
* local `.gitignore` files were added for run artifacts.

No algorithm, distribution, physical semantics or label format changed. Analysis outputs and a regenerated NAT-1K scene are byte-identical (`VALIDATION_RECORD.md`).

**Documentation updated:**
* `RIND3D_Dataset1K/README.md`, `docs/{PRODUCTION_LOG_1K_v1, CONTRACT_1K, REPRODUCE, SELECTION_LAW_1K, OPTIMIZATION_REPORT, GEOMETRY_DIVERSITY}.md`;
* `RIND3D_Posterior/README.md`;
* `RIND3D_Audit/README.md`;
* `docs/RIND3D/README.md`, `docs/RIND3D/PHASE3_SUMMARY.md`;
* the root `README.md` (RIND3D section);
* stale "v3.3" docstrings.

Each change adds a dated status note, or corrects an interface description, and points to the archive where a file moved.

## 4. Data outside Git (inventory; nothing moved or modified)

| location (`/home/yushu/Projects/RIND/`) | size | content | integrity record |
|---|---|---|---|
| `RIND3D_1K_data/release_1k_v1` | 20 GB | frozen 1K release: physical data and 159,802 certified labels | per-scene `SEAL.json` (25,550 files); `labels_log.jsonl` sha256; `backfill_verification_v1/post_backfill_integrity.json` |
| `RIND3D_1K_data/natural_supplement_v1` | 3.1 GB | NAT-1K: 907 scenes, natural windows, 30,349 labels, 360 OOD references (`labels/ood/`) | `labels_log.jsonl` sha256; `verify_nat1k.json` |
| `RIND3D_1K_data/pilot_features_v1` | 24 GB | Student pilot features: 194,532 rows (regenerable by `build_features.py`) | shard row tables |
| `RIND3D_1K_data/pilot_runs_v1` | 0.9 GB | the 13 pilot arms: checkpoints, train logs, eval npz | `arms.json`, `queue.log` |
| `RIND3D_1K_data/backfill_verification_v1` | 74 MB | full verification outputs | `SHA256SUMS` |
| `RIND3D_1K_data/{canary128_A, canary1024_A, smoke_controller, geom128}` | 1.1 GB | 1K canaries, controller smoke, geometry study | manifests inside |
| `RIND3D_100K_handoff/` | 39 MB | tarball of the 100K handoff (5560a20) | `.sha256`, `HANDOFF_MANIFEST.json` |
| `RIND3D_PosteriorLabels*`, `RIND3D/`, `reports/` | 4–14 GB each | earlier label studies, the original v0.1 generator checkout and reports | `EVIDENCE_MANIFEST.json` where present |

## 5. Remaining ambiguity / unresolved choices

1. **Should FastLabel move into `RIND3D_Posterior` as a package next to the tau-pencil engine?** It currently sits inside `rind3d_1k`, which ties it to the 1K scene layout. The 100K port would need either that move or an adapter. Left as is (no change without an integration decision; `100K_READINESS.md`).
2. **`RIND3D_StudentResearch` mixes decision documents with reusable tools** (NAT-1K generator, Student reference, OOD, S6). A later split into `RIND3D_Student/` and `RIND3D_Tools/` is possible. It was not done, to avoid renaming without a concrete ambiguity.
3. **The Gate A harness depends on the FastLabel research modules**, not the production package. Porting it to `rind3d_1k.fastlabel` would let the research code be archived fully.
4. **`RIND3D_Generator` contains many research modules inside the `rind3d` package** (`*_research.py`, campaign tooling, `legacy/`). They are kept because the package's CLI and tests import them, and the tree is provenance-tracked.
