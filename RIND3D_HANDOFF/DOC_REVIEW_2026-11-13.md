# Review of `RIND3D_数据集设计与技术说明` (2026-11-13)

**Input:** `GPT-Artifacts/RIND3D__…GitHub版.zip`, i.e. `docs/RIND3D_数据集设计与技术说明.md` plus 8 PNGs.

**Output:**
* `docs/RIND3D_数据集设计与技术说明.md`, the revised text;
* `docs/assets/01–09_*.png`, the new figures (sources in `RIND3D_HANDOFF/figures/`).

**Checked against:**
* `RIND3D_IMPLEMENTATION_FACTS.md`, `README.md`, `100K_READINESS.md` and `100K_FASTLABEL_INTEGRATION.md`;
* `RIND3D_StudentResearch/H_100K_RECOMMENDATIONS.md` and `J_COMPLEXITY_COVERAGE.md`;
* the code on `yushu/3d-fastlabel-core` and `yushu/3d-100k-h20@5560a20`.

**Overall.** The draft is largely accurate. Every number it quotes matches the sources: 4.7 / 1.8 / 14 / 47%; +1% labels and +14% proposals; 25× and 80–100×; 13.4 s, 2.5% and 23%; 3.7 s, 0.36 MB, 47 KB and 0.2 TB; 159,802 labels, 52,523 single-cell + 107,279 statistical; 4.33 h; 10.2 labels/s; 447 references at max TV 0.0164; 2.2–2.6 nats; 2 M parameters.

The corrections concern four things:
* what is implemented and what is only recommended (the selection law and κ);
* two places where the mechanism was stated loosely;
* the status changes of this task (the FastLabel move, the 100K patches);
* the figures.

Structure kept: one title, eight sections, no subtitles, headers or footers, four bold-red open items.

## The 14 audit points

| # | point | verdict | change in the revised text |
|---|---|---|---|
| 1 | physics and visibility semantics | consistent (closed segment, closed solids, touching = blocked, SOLID mask separate) | §2: the source law spelled out. Uniform on D(G) by rejection sampling, open cube, outside the closed solid, certified clearance > 2⁻²⁶; anti-duplicate ‖·‖∞ ≥ 2⁻²⁰ (fact sheet §2, `physical.py:67-83`) |
| 2 | 1024³ vs 128³ | consistent | none |
| 3 | multi-source canonical data vs single-channel input | consistent | none |
| 4 | T* | essentially consistent, but missing that natural means **no m-thinning** | §3: T* defined with natural acceptance (c ≡ 1); listed the per-window and per-scene provenance needed for the weights (stream, proposal index, U, c-row, rate-stream m counts) |
| 5 | κ and the L64 coverage policy | consistent | §3: states that 2+2 is not recommended (ESS ≈ 0.6, weights up to 2×) |
| 6 | implemented vs recommended | **correction.** The draft described only admissibility plus the recommended natural / coverage design. The 1K and the current 100K draft also **thin by m** (m-balanced c-tables from an adaptive controller), and K comes from an adaptive controller (15% floor per k; 1K realised 15.4 / 13.5 / 15.8 / 55.3%) | §2 and §3: the implemented rule is stated, and the natural + coverage design is marked as the recommended contract change. Figure 03 is labelled "推荐候选" |
| 7 | local-only renderer | consistent | §5: adds the 1K full-field cost (24.6 s per scene) for comparison. Figure 05 uses a real 100K scene |
| 8 | 100K storage vs RINDPack | consistent | §5: durability gaps made concrete (a torn manifest line blocks resume; no sha256 of the consolidated labels). A minimal patch is tested on a copy and not applied (`100K_FASTLABEL_INTEGRATION.md` §4) |
| 9 | tau-pencil and FastLabel roles | consistent, status outdated | §6 and §8: FastLabel is now the scene-format-independent core in `RIND3D_Posterior` with 1K / 100K adapters, byte-identical to the backfill. The 100K engine patch is tested on a copy; the unpatched 100K loader silently drops `certified_stat` labels. Section 8 table updated |
| 10 | deterministic and statistical TV | **correction of a formula statement.** The draft wrote TV(q_B, q_F) = 1 − Vol(F)/Vol(B). Equality holds for the continuous uniform laws; the cell-level label TV is ≤ that value | §6: written as ≤, with the one-line reason (aggregation does not increase TV). Added the two-stage design (512 / 1024, δ_r = δ·2⁻ʳ), that statistical labels store q_B itself (`certified_stat`) and the deterministic exits |
| 11 | costs by response type | consistent. The solid-separated mechanism was loose ("无法找到直接亮暗边界 receiver") | §4: by definition no 6-adjacent lit/dark pair **exists**. Phase 1 starts from the 128 interior receivers, B stays broad (hundreds of thousands of pieces and cells), and 72–78% of the time is phase-1 BSP and emission. The all-dark failure is given its cause (B keeps solid interiors) |
| 12 | no H20 validation | consistent | §6 and §8: the Gate A procedure is named (three builds: default, forced FP64, forced GMP), plus the accuracy check on the 100K mix |
| 13 | CE supervision and weights | consistent | §7: the calibration claim is now quantified (coverage MAE 0.03 vs 0.24–0.28, `I_EXECUTIVE_DECISION.md`). "严格后验" became "认证后验", because a q_B label is certified, not exact |
| 14 | Teacher necessity | consistent | none |

## Other corrections

* **§4, OOD results:** the information gains are now quantified (all-dark +0.20 to +0.26 nats, solid-separated +0.7 to +1.0; `G_GENERALIZATION_AND_EXTREMES.md`).
* **§6, new figure 07 (a real exact posterior).** The main task's F is a very thin wedge sheet: total volume ≈ 0.098 cell volumes over 3,862 cells. Cell fill fractions are median 1.9·10⁻⁵, from 1.6·10⁻⁶ (10%) to 6.0·10⁻⁵ (90%). This makes "uniform density, not uniform cell mass" concrete.
* **§8:** the code table is updated (FastLabel core location, 1K entry points, the patches directory, the figure sources).
* **Language:** the wording is tightened throughout. No content was removed; vague statements were made precise.

## Figures

Every previous figure was replaced; the audit table is in `RIND3D_HANDOFF/figures/README.md`. The main defects were:
* decorative content: Gaussian blobs, a random checkerboard, sources outside the world box;
* a missing glyph (`q□`) and a cropped panel in the FastLabel figure;
* window sizes not to scale;
* a 2D-only definition of a 3D adjacency;
* title bands.

The new figures:
* use real 1K / 100K data and production code;
* are drawn at the document's 14.8 cm width and checked at that size (`out/contact_sheet.png`);
* have every displayed crop re-rendered bit for bit with the certified renderer.

## Open questions (kept bold red in the document)

1. κ, whether to use the L64 coverage slot, and the minimum coverage targets. They need the 100K canary statistics.
2. Phase-1 narrowing for solid-separated windows without changing F or the certificate. S6 also needs a production review.
3. FastLabel replacing TPE in the 100K. It is pending the H20 Gate A, cross-device bit-exactness, throughput and the accuracy check.
4. Student architecture and scale, augmentation, and OOD generalisation at 100K scale.
5. The integration steps listed in §8: merge the core, apply the patches, freeze κ, run a canary of at least 256 scenes on H20.

## Not resolved here (no evidence either way)

* **The total 100K storage after the switch to FastLabel and natural sampling.** The draft's 0.2 TB is the TPE-label projection. FastLabel's sparse statistical labels have a different size distribution (median cells per label: L64 2,997, L128 121, L256 1). It must be measured on the canary.
