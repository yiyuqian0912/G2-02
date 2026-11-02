# RIND3D integration handoff — 2026-11-02

RIND3D v0.1 contains 1,000 certified 3D scenes and 256,000 observation windows. The existing release was integrated without regenerating fields, changing scientific data, or adding reference-source pools. The permanent dataset name is **RIND3D**. Historical `Lite-Core-1K` identifiers remain unchanged as internal provenance.

The integration is on `yushu/3d`, based on fetched `origin/main` commit `37b1f045d1c78ccc2b28364fa9e4bed5e6a65642`. The original working directory is preserved on `tmp/pre-3d-integration` at WIP commit `8bfecd8f8c06cc85d834d3df11301dfcf32b879e`. Its prior HEAD was `1591783558e04e1188e97ee008932b11421db52b`; the local `main` ref remains `0908ceb7fe7ed8dcf89c1f1a69315e1168408e24`. The integration uses a separate sibling worktree, `G2-02-3d`, so ignored local datasets and the old environment were not moved. All eleven preserved changed/untracked files retain their pre-integration hashes. Phase 1/2 source on this branch is byte-identical to the fetched base.

## Delivered interfaces

| Directory | Responsibility |
| --- | --- |
| `RIND3D_Generator/` | Original generation, canonical geometry, CPU/CUDA rendering, RINDPack, window selection, A/B physical evidence, campaign/recovery and reader source |
| `RIND3D_Generator/vendor/` | Required project-owned canonical geometry and RINDRay source; no sibling checkout dependency |
| `Phase3_Single_Source_3D/` | Restricted student data, explicit teacher geometry, certified physical queries, costs and bounded octree |

The student sample is whitelisted to scene/window IDs, integer XYZ origin, side, packed local response and packed local solid mask. Full geometry is accessed explicitly through `teacher_scene`. The public archive itself includes full fields/geometry under `full-field-training-v1`; it is not a blind hidden-field benchmark. True sources, private references, witnesses, production seeds and private certification evidence are absent from the release.

`PhysicalOracle` validates FREE sources with original clearance and observation exclusion, evaluates bounded Cartesian source/receiver batches, and rerenders arbitrary unaligned 64/128/256 windows. Native raster calls are tiled at their existing 64³ limit. Point queries use certified CPU or the original canonical reference; CUDA is an optional local-ROI backend. Invalid sources have a validity mask and response -1; unresolved/numerical failures raise rather than becoming physical labels. The reference backend also uses the original canonical receiver-membership classifier, so the solid-mask comparison is independent. Sparse mismatch, full-window mismatch and independently verified equivalent-source certification remain distinct. Nearby teacher candidates have no reference-label minimum-separation constraint.

The diagnostic octree starts with eight 512³ cells, samples k=4 candidates per cell, and can split a selected cell into eight 256³ children. Uniform and exploratory variation policies are runnable. It records unresolved cells and query budgets without pruning them or inventing a complete probability target. The teacher/model objective is still research; no 3D neural architecture or training job was added.

## Download and setup

[RIND3D-v0.1.tar.zst on Google Drive](https://drive.google.com/file/d/1PmMp39En-uc4QTgACd_EpS4mTZ3wO7yx/view?usp=drivesdk) is **633,273,091 bytes**. The applied permission was read back as `type=anyone`, `role=reader`, `allowFileDiscovery=false`. Two complete unauthenticated downloads passed the original archive SHA256:

```text
1f5a2ff2bd9ac4a479a074b33e0d06f160706e4369b199144ee117693913ee1e
```

From a checkout of this branch:

```bash
cd Phase3_Single_Source_3D
uv sync --extra dev --extra train
uv run python examples/download_dataset.py --output data
export RIND3D_DATA_ROOT="$PWD/data/RIND3D-v0.1"
uv run python examples/inspect_dataset.py
uv run python examples/pytorch_dataloader.py --workers 2
```

Reader-only setup was tested before any generator or canonical module was installed. It needs no GPU or native build. Physics adds `uv sync --extra dev --extra train --extra physics`, followed by `RIND_PYTHON="$PWD/.venv/bin/python" ../RIND3D_Generator/scripts/build.sh`. See the generator README for C++/GMP/MPFR prerequisites and optional CUDA driver/NVRTC requirements. Keep the 2D and 3D environments separate.

## Acceptance evidence

| Check | Actual result |
| --- | --- |
| Existing repository/2D checks | 202 passed, 6 skipped; 38 shell entrypoints passed |
| Relocated generator complete regression | 348 passed, 7 skipped, 0 failed, 4 inherited fork-deprecation warnings; 263.70 s |
| Phase 3, Linux x86_64 / Python 3.12 | 33 passed, no skips/failures; 21.55 s |
| Fresh clone, Linux x86_64 / Python 3.13 | 33 passed, no skips/failures; 21.78 s |
| Fresh canonical native build | Both CGAL capability probes passed; all five native modules imported |
| Fresh extracted release | 13,013 sealed files SHA-verified, 1,000 scenes, 256,000 windows; 8.38 s |
| Reader-only access | All splits, unaligned 79³ ROI, local Y/M loading and two PyTorch workers passed |
| Relocated scientific fixtures | Nine families × complete 32³/64³/128³ responses and masks; independent CPU/CUDA/canonical checks passed, including hard contacts; 595.09 s |
| Real-data smoke | Nine families × 64/128/256 observations: 27 windows, 3,456 native pairs, 864 original-reference comparison pairs; all compared labels agree |
| Complete local ROI comparison | Selected barriers windows at all three sizes: CPU/CUDA response and solid bits agree |
| Whole-process octree example | 0.429 s wall, 0.428 s user+system CPU, 92,995,584-byte peak child RSS; 64 attempts and one refinement |

The seven generator skips require private original production/campaign artifacts, intentionally excluded from the repository. They do not grant new publication authority. Existing 2D data-dependent skips are retained. Historical first runs exposed missing copied configuration/scripts and the ROI-size adapter issue; these were fixed before the final passing runs. The source inventory verifies 292 copied files, with explicit path/documentation/package-description adaptations. Canonical predicates, rendering algorithms, numerical policy and publication-gate source are unchanged.

The real smoke selects the least-component scene among the first eight scenes per family, then descriptive windows spanning dark/mixed/bright observations. It uses 32 native FREE probes and eight original-reference receivers per window. This is a bounded correctness check, not a representative throughput benchmark. Sampled lit/free ratios ranged from 0.00139 to 0.99697. Native batches had a 0.000796 s median and 0.001892 s maximum; original-reference batches had a 9.626 s median and 35.709 s maximum, with different batch sizes and setup costs. No speedup ratio is inferred. That initial visibility-reference run took 373.26 s wall and 373.12 s CPU with a 516,087,808-byte peak RSS. After making canonical solid classification independent as well, the complete rerun passed all 27 cases in 372.58 s wall / 372.51 s CPU, with a 516,100,096-byte peak RSS.

For the three local ROIs, native CPU times were 0.0154/0.0789/0.2018 s; CUDA times were 0.4112/0.2445/0.2484 s, including each example's cold renderer setup. Small ROI CUDA is not assumed faster. The internal octree loop was about 0.023 s; the separately measured 0.429 s process includes imports, dataset/geometry loading and diagnostic writing. These are different timing scopes.

Two larger original-reference diagnostics were stopped as incomplete after 533 and 858 seconds. Their logs are preserved; neither was classified as physical rejection or reported as a completed comparison. The scientific fixture protocol and full-window criteria were not relaxed. Only the integration diagnostic's scene/sample work was bounded. Full downloaded-file SHA validation was rerun; the fresh verification figure above does not claim another exhaustive decoded-CRC scan. Random ROI reads validate the touched bricks, and the original sealed release's prior full CRC/physical acceptance remains separate provenance.

Aggregate, source-coordinate-free records are in [INTEGRATION_CHECKS.json](../RIND3D_Generator/provenance/INTEGRATION_CHECKS.json). The original frozen data build is `b038ba8842f5ea5e059f42173dbbb840cdef4544e52f02dab3cec82f0d8e19a2`; this relocated Python 3.12 fixture build is `e19b3004fc9e733cdb7cb345846c9ed14fea8eebd447a304f2ba41927a5aed3e`. They are deliberately not presented as the same identity. Full operational logs remain outside Git in the local integration evidence directory.

## Remaining decisions

Dataset/software license and citation authors remain pending. Public link access does not create a permissive license. New production campaigns require their own complete scientific, quality and recovery authority; no original certificate was copied and no new campaign authority was issued here. Arbitrary-point CUDA, a finished teacher distribution, 3D student architecture and equivalent-source enrichment remain outside this integration. The original complete dataset, old READY outputs, main/tmp branches and sibling projects are preserved.
