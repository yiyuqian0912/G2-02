# H — RIND3D-100K: scientific sampling changes before the contract is frozen

**Scope.** These are recommendations only. The 100K branch (`yushu/3d-100k-h20`) was read, not modified, and no
campaign was started. Evidence:
* `B_SAMPLING_WEIGHTING.md` (E13, E15);
* `A_TASK_DEFINITION.md`;
* `E_STUDENT_PILOT.md` (the measured effect of the weighting on a real Student).

## Kept (mandatory or validated)

| item | keep |
|---|---|
| scale | 100,000 scenes; 1024³ worlds; 12 windows per scene, 4 each of L = 64, 128, 256 |
| multi-source worlds | 1–4 physical sources per scene, source-separated binary responses, one source-aligned posterior slot per channel, canonical storage that never flattens |
| source prior | i.i.d. Uniform(D(G)) given G, with the 2⁻²⁰ anti-duplicate rule (audit E2: per-scene PIT uniform) |
| window geometry | uniform origins; closed-box exclusion of every source; FREE > 0 |
| labels | hard Uniform(F) per channel on 128³ cells; FastLabel certificates with δ = δ_total / N_planned (M3) |
| splits | scene-level hash splits; a test split never used for optimisation |

## Necessary changes

| # | change | why | how in the 100K code (`RIND3D_Dataset100K/rind3d_100k`) |
|---|---|---|---|
| N1 | **Declare the target population T\* in the contract.** Multi-source worlds with a declared κ; natural admissible windows; every mixed channel is one single-channel example | It defines what the Student learns. The 1K left it implicit (audit M1) | contract text plus `SELECTION_LAW_100K.md` |
| N2 | **No thinning by the effective-layer count m** (c ≡ 1). Windows are accepted iff no source is in the box, FREE > 0 and m ≥ 1 | c(m) depends on the *other* (hidden) channels and tilts P(G \| X) of a single-channel Student. In the 1K it costs 88% of the ESS to undo (ω/c ESS 11.8%). With c ≡ 1, the T* ESS is 85–94% (E15 D1) | controller plan: c_table ≡ 1; `select_side` unchanged |
| N3 | **Fixed, declared source-count law κ**, keyed by slot, independent of every scene (e.g. κ = (0.10, 0.20, 0.30, 0.40): 90% multi-source scenes) | An adaptive controller makes the population a function of the run history. A fixed κ is reproducible and k ⟂ (G, S) holds by construction | `controller.plan_epoch` returns the fixed κ draw per slot |
| N4 | **Weights:** ω̂ = an unbiased estimate of P(m ≥ 1 \| G, S, L) per scene-side, from the rate stream (count or Rao–Blackwellised). Report the ESS | The quota of 4 windows per side needs the 1/Λ correction for exact labels (Thm A1). With c ≡ 1 only the scene-level spread remains | `rate` stream already in `select_side`. Store the per-proposal m counts (as the 1K does), so `rb` is available |
| N5 | **Store everything needed to rebuild any weight:** per window m, mask, k, U; per side the rate-stream m-histogram and quota proposals; κ | Any later target change must be a re-weighting, not a regeneration | already mostly present. Keep `rate_m` and per-proposal logs |
| N6 | **δ per label = 0.01 / N_planned**, with N_planned the planned mixed-channel count | union bound over the whole campaign (M3) | `fastlabel.DEFAULTS["n_tasks"]` = N_planned |
| N7 | **Loader:** accept every certified status; declared `population`; explicit one-hot fallback for unresolved labels | M2; the 100K canary had 1.4% timeouts | port `rind3d_1k.loader.flatten` |

## Recommended

| # | recommendation | evidence |
|---|---|---|
| R1 | **Effective-layer balance for a multi-channel study:** if it needs m-balanced windows, add a *separately declared* m-balanced stream and combine it with the natural stream by the balance heuristic (E15 D3: 2 + 2 windows per side gives an L64 M-histogram of 50/22/16/12 at ESS 0.52 / 0.67 / 0.86). Do not thin the primary population | E15 |
| R2 | **A heavy-k κ partly balances m without thinning.** With 1K worlds, κ = all-4 gives natural M-histograms of 65/21/8/5 (L64), 44/25/15/16 (L128), 19/16/18/47 (L256). κ cannot balance L64, where most windows have one mixed channel | e15b |
| R3 | **Labelling cost:** natural windows carry fewer mixed channels (22.0 vs 29.8 labels per scene, E15). About 26% fewer label tasks for the same 12 windows | E15 |
| R4 | **Extreme channels.** All-lit and all-dark channels of stored windows are kept as stored channels with status `not_computed`. They are **not** labelled in production: the cost gate is not met (`G_GENERALIZATION_AND_EXTREMES.md` §3). Under N2 they are still tilted, because they exist only in windows with m ≥ 1 | audit M4 |
| R5 | **Natural OOD reservoir.** The 100K renders locally and stores no full fields, so natural extreme windows cannot be derived later. If an OOD benchmark is wanted from 100K worlds, draw it on held-out (test-split) scenes only, as a separately declared stream without the m ≥ 1 condition. It is evaluation-only and needs no labels (proper one-hot scores). The 12-window production quota is unchanged | F protocol |
| R6 | **S6 before any all-dark labelling.** Exclude solid interiors from the outer bound. The prototype certified every one of the audit's all-dark tasks that had failed (§3 of G) | `s6/` |
| R7 | **Soundness asserts in production** (audit S4): every label's support must contain the true source cell. This is checked for all 1K labels in `D_BACKFILL_REPORT.md` | — |
| R8 | **Training defaults** follow from the pilot (`E_STUDENT_PILOT.md`): the supervision type and the weighting | E |

## What the 100K design must not do

* Replace multi-source worlds with single-source ones. The single-channel target needs no such change: under natural sampling, the multi-source exclusion factor changes the conditional by ≤ 1% TV (A §3).
* Thin windows by the number of informative channels of other sources for the single-channel population.
* Use adaptive κ without declaring the resulting population.
* Add mandatory all-lit or all-dark labelling, or change the 12-window quota.
