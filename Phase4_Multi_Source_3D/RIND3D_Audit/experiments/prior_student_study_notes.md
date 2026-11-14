# Prior Student/Teacher training study (2026-11-10): audit notes

Author: audit sub-agent, 2026-11-13. Read-only extraction from review archives; no code was run.

Archive roots (unzipped to scratch, paths below are relative to them):
- `ST/` = `RIND3D_STUDENT_TRAINING_REVIEW.zip` -> `RIND3D_STUDENT_TRAINING_REVIEW/`
- `RM/` = `RIND3D_RARE_MODE_AUDIT_REVIEW.zip`
- `RL/` = `RIND3D_REVIEW_LATEST.zip`

Main report: `ST/RIND3D_3D_STUDENT_TRAINING_DECISION.md` (388 lines), cited as DECISION:Lnn.

(Work in progress; sections are appended as evidence is read.)

## 1. Data

- Dataset: **RIND3D v0.1, 1K single-source scenes** (not v0.2/v0.3/1K-production). DECISION:L5-16; ABLATIONS.json:L2 ("RIND3D v0.1 1K, train/val/test 776/116/108 scenes").
- Split: **by scene**, the v0.1 release *geometry-group* splits (DECISION:L6). Train 776 scenes / 49,664 windows; val 116 / 1,856; test 108 / 1,728 (DECISION:L9-13; `results/out/support_checks.json`).
- Windows per scene: v0.1 stores 256 windows/scene (OBJECTIVES:L27 "256,000 windows"); the study uses a seeded random pick of **64/scene for train, 16/scene for val and test** (`code/s3d/build_feats.py:L31,L41-42`, seed `seed_of(sid, split, "window-pick")`).
- Window sizes: sides {64, 128, 256} voxels in a 1024^3 world (OBJECTIVES:L23; `code/s3d/grid.py` header). Test group sizes: side64 604, side128 556, side256 568 (CALIBRATION.json test group_sizes, ~L134055).
- Sources per scene: **one** (single-source); s* ~ uniform on FREE(K) (OBJECTIVES:L20-22). 776 distinct training sources (OBJECTIVES:L40).
- Window selection law (v0.1): proposals rejected if they contain s* or are "not informative" (no lit or no shadowed FREE voxel); then a lit-ratio mixture natural 0.3 / ratio-balanced 0.5 / extreme 0.2, plus a spatial-coverage greedy rule (OBJECTIVES:L23-28). So the training/test law is a *selected* law, p(s | O, W, selected) (OBJECTIVES:L37-39).
- Teacher pool: 16 windows/scene of train (12,416 windows), plus all val (1,856) and test (1,728) windows = 16,000 Teacher windows (`code/s3d/teacher.py:L218-220`, `--per-scene 16`; TEACHER_VALUE:L17-21).
- v0.2 production canary (108 unseen scenes, 10,980 windows) used only for OOD evaluation (DECISION:L16, §8 L245-261).
- Test used once? Claimed: "Every model was evaluated once on test, after FROZEN_PROTOCOL.json was frozen (08:42 UTC)" (DECISION:L20; `code/configs/FROZEN_PROTOCOL.json:L2-3`; README:L91). Not independently verifiable from logs (test eval logs carry no timestamps). Caveats: (i) test Teacher targets and support checks (which use K and s*) were computed before training (`code/scripts/queue_vm1.sh:L5` waits for `test_teacher.npz`); this is label access, not model selection. (ii) D_sc25_l03 was added after the freeze (09:52 UTC) and also evaluated on test (`RUN_DEVIATIONS.json:L18-22`). (iii) The 16k extension was registered at 08:29, before the freeze (`EXTENSION_BUDGET.json:L2`).

## 2. Student input X

- Exactly: pooled window features **4 x 32^3 uint8** (channels: lit-FREE, shadow-FREE, SOLID, lit/shadow-boundary fractions), the window **origin** (integer voxels, world-aligned) and **side** (bucketed into 3 classes). ARCHITECTURE:L7; `code/s3d/data.py:L3-5, L56-62` (`Split.get` returns only feats, origin, side, src[label], exclude[support flag], and world only for A_priv); `code/s3d/model.py:L70-80` (encoder gets feats, side_idx, origin; query encoder gets coarse-cell centres + origin + side, with absolute-position Fourier features, `model.py:L27-38`).
- Features are computed by `rind3d_teacher.features.window_features(observed_response, local_solid_mask)` (`build_feats.py:L10-20`); that function is NOT in the zip, so its exact pooling could not be checked.
- Leakage check: `lit_ratio`, `free`, `family`, `component`, `ratio_bin`, scene_id are loaded but are not returned by `Split.get` -> not inputs (`data.py:L22-29, L48-49, L56-62`). No K, s*, source count, or selection info enters the network. Positive: no leakage found in the code paths read.
- Weak leakage-like channels (by design, not bugs): absolute window origin and side are inputs; the support Omega_S = cube \ window is computed from W, which encodes the generator rule "s* not in window" (selection information baked into the support). The selected-law lit-ratio component is not an input but shapes the training distribution.
- Privileged control only (A_priv): a second encoder over a 32^3 world occupancy grid [near, occ, window-mask, 0] (ARCHITECTURE:L26; `model.py:L41-50, L73-75`).
- Augmentation: independent random one of 48 cube symmetries per sample, applied to feats, origin, s*, Teacher points and world grid (`data.py:L91-131`).

## 3. Output parameterisation

- Two-level piecewise-constant density on a fixed partition of the whole unit cube: 16^3 coarse cells (64 voxels) x 8^3 fine cells (8 voxels) = 128^3 = 2^21 fine cells. `model.py:L1-12`; `grid.py:L1-25`; OBJECTIVES §6 L137-151.
- Normalisation: P(c) = V_c e^{l_c} / sum V_c' e^{l_c'} (V_c = legal volume fraction of coarse cell), P(f|c) = v_cf e^{m_cf} / sum (v_cf = legal fraction of fine cell), p(s) = P(c)P(f|c)/(v_cf |f|) on the legal part. Exactly normalised (cell masses sum to 1 within 4e-7, MC integral ~1). `model.py:L7-11, L93-110`; OBJECTIVES:L141-149.
- Coverage: whole cube **minus the window box** for v0.1 (`exclude=True` for every v0.1 window, `data.py:L41`; `grid.py:L8-13`). Solids are NOT excluded (support uses W only, never K). A flag `exclude_window=False` gives Omega_S = cube for natural v0.2 windows (ARCHITECTURE:L11).
- Coarse logit from cross-attention (3 blocks, d=192, 6 heads) of 4,096 coarse-cell-centre queries onto window tokens; fine 512-way logits = Linear(d->512) on the coarse-cell embedding; both heads zero-initialised so the initial density is uniform on Omega_S (ARCHITECTURE:L17-25; `model.py:L53-80`). 8.41 M params (B: 8.31 M).
- Units: log density per unit-cube volume; "uniform = 0" means uniform on the cube. Uniform on Omega_S is in fact -log(1-|W|) <= 0.016 nats (side 256), negligible.
- Resolution limit: 8 voxels (1/128 of the world); sub-8-voxel feasible structure (slit throats, thin umbras) is not resolvable (ARCHITECTURE:L44; DECISION:L383).

## 4. Loss arms (17 training runs + B; 15 distinct arm configs incl. dropped ones)

Common to all trained arms (`code/configs/student_arms.json:L3-17`; `code/s3d/train.py`): HierDensity d=192, 3 layers, 6 heads; batch 64 windows/step; **8,000 steps** (16k for the two extension arms); AdamW lr 4e-4, betas (0.9, 0.98), wd 0.05, 500-step warmup + cosine to 1e-5, grad clip 1.0; bf16; 48-symmetry augmentation; windows sampled **uniformly with replacement** from the row pool each step (not scene-balanced) (`train.py:L134-141`). Trained from scratch (no B weights loaded). **No importance / sample weights** anywhere in the losses (`train.py:L154-158`). Checkpoint = best of the arm's OWN val objective every 1,000 steps (`FROZEN_PROTOCOL.json:L4`; `train.py:L171-177`); in practice every 8k A/C/D run had its best at step 8000, except A_sc25 (step 4000).

Loss definitions (`train.py:L6-13, L32-43, L154-158`):
- l_A(w) = -log p_theta(s*_w | O_w, W_w) (fine-cell piecewise-constant density at s*).
- l_G(w) = -(1/m) sum_j log p_theta(t_wj), m = 256 points drawn uniformly with replacement from the M = 2,048 stored resampled Teacher points.

| Run | Loss | NLL windows | Gibbs | lambda | steps | seeds |
|---|---|---|---|---|---|---|
| A_full | mean l_A over 64 | all 49,664 | - | - | 8k | 0,1,2 |
| A_sub | mean l_A | 12,416 Teacher-pool windows (776 x 16) | - | - | 8k | 0 |
| A_sc25 | mean l_A | 194 scenes x 64 = 12,416 (seeded scene subset, rng 20261112, `train.py:L114-121`) | - | - | 8k | 0 |
| A_sc50 | as A_sc25 at 50% | - | - | - | - | **dropped before start** |
| A_priv | mean l_A + 32^3 world map input | all | - | - | 8k | 0 |
| C_pool | mean l_G over 64 pool windows (coefficient 1) | none | all 64 from pool | - | 8k | 0 (s1 dropped) |
| D_all (l=1) | mean l_A over 64 + lambda * mean l_G over 16 of them | 48 random + 16 pool | 16/64 | 1.0 | 8k | 0 |
| D_all_l03 | same | same | 16/64 | 0.3 | 8k | 0 |
| D_all_l01 | same | same | 16/64 | 0.1 | 8k | 0 |
| D_sel | same; Gibbs windows from declared "hard" 25% of pool (3,104) | | 16/64 | 0.3 (override; config says 1.0) | 8k | 0 |
| D_rand | same; random 25% of pool (3,104) | | 16/64 | 0.3 (override) | 8k | 0 |
| A_full_16k | A | all | - | - | 16k | 0 |
| D_all_l03_16k | D | all | 16/64 | 0.3 | 16k | 0 |
| D_sc25_l03 | D restricted to A_sc25's 194 scenes (Gibbs pool 3,104) | 12,416 | 16/64 | 0.3 | 8k | 0 (exploratory, post-freeze) |
| B / B_priv | existing Phase 3 cost-regression checkpoints (MSE on C + ranking), normalised post hoc as p ∝ v_f exp(-C_hat/tau'), tau' chosen on val from {0.01..0.5}: B 0.05, B_priv 0.03 | | | | | |

Sources: `student_arms.json:L18-113`; run configs `results/runs/*/config.json` (lambda values confirmed); `RUN_DEVIATIONS.json`; `EXTENSION_BUDGET.json`; DECISION:L99-120; OBJECTIVES §4 L90-103; `code/s3d/evaluate.py:L24, L169-178`.
lambda* = 0.3 was chosen by val NLL from the completed sweep {1.0: -3.92, 0.3: -4.30} before lambda=0.1 finished (`results/teacher/lambda_choice.json`); lambda=0.1 later scored better (test -4.41).
D_sel policy: >= 2 of {screen multimodal, screen zero-cost volume <= 2/4096 ("rare", true for 60% of pool), lit-ratio bin 0/9, screen non-locality >= 75th pct}, cheap screen = first 4,096 Sobol points x first 128 receivers (`results/teacher/selection_report.json`).

## 5. Results

Test (1,728 windows, 108 scenes), scene-cluster bootstrap 95% CI (2,000 reps, `code/s3d/analyze.py:L14-25`), single seed except A. Full table DECISION:L137-157; paired diffs L159-178; TEACHER_VALUE:L37-47.

| Model | Test NLL [CI] | Cov 50/80/90/95 | Cal. err | HPD-90 vol | Paired vs A-mean |
|---|---|---|---|---|---|
| A s0/s1/s2 | -4.49 / -4.46 / -4.45 (CIs ~±0.18) | 0.55/0.86/0.94/0.96 (mean) | 0.04 | 0.047 | 0 |
| A_sub | -4.43 | 0.53/0.85/0.92/0.96 | 0.037 | 0.045 | +0.04 [-0.01,0.08] |
| A_sc25 (194 sc.) | -3.09 [-3.34,-2.82] | 0.22/0.60/0.76/0.87 | 0.198 | 0.049 | +1.37 [1.17,1.58] |
| A_priv | -4.53 | 0.54/0.86/0.94/0.97 | 0.034 | 0.045 | -0.07 [-0.14,-0.00] |
| A_full_16k | -5.13 [-5.33,-4.92] | 0.59/0.88/0.94/0.97 | 0.059 | 0.036 | -0.67 |
| D l=0.1 | -4.41 | 0.63/0.91/0.97/0.99 | 0.085 | 0.094 | +0.05 [0.00,0.10] |
| D l=0.3 | -4.27 | 0.73/0.96/0.99/1.00 | 0.148 | 0.204 | +0.20 [0.14,0.25] |
| D_sel / D_rand | -4.33 / -4.31 | | 0.126 / 0.154 | | +0.13 / +0.15 |
| D l=0.3 16k | -4.92 | | 0.157 | 0.182 | vs A_16k +0.21 [0.16,0.26] |
| D_sc25 l=0.3 | -3.05 | 0.34/0.79/0.94/0.98 | 0.095 | 0.168 | vs A_sc25 +0.04 [-0.12,0.20] |
| D l=1 | -3.90 | 0.88/0.99/0.99/1.00 | 0.249 | 0.376 | +0.56 |
| C Gibbs only | -2.08 | 0.82/0.96/0.99/1.00 | 0.270 | 0.483 | +2.38 |
| B tau 0.05 | -1.67 | 0.62/0.80/0.87/0.91 | 0.098 | 0.281 | +2.79 |
| B_priv tau 0.03 | -2.29 | | 0.100 | | +2.17 |
| Gibbs Teacher (knows K) | -3.35 [-3.47,-3.22] | | | | |
| Bank-feasible uniform (knows K; 1,239/1,728 = 72% windows) | -8.44 [-8.66,-8.20] | | | | |

- Calibration metrics: HPD-based "PIT" u = P(rho > rho(s*)) + 0.5 P(cell of s*) (`evaluate.py:L44-45`); coverage at a = mean(u <= a); calibration error = mean |coverage - a| over a in {0.05,...,0.95} (`analyze.py:L28-32`). Also "s* outside 99% HPD" and "worse than uniform" rates (DECISION:L197-204). No reliability-bin ECE, no per-coordinate PIT.
- Per-family / strata (test, A_full_s0 NLL, cal-err in parentheses; from `RIND3D_CALIBRATION_AND_MODE_COVERAGE.json` by_split_and_group.test.models.A_full_s0, ~L118755-120665): barriers -4.17, convex_polyhedra -4.79, corridors -4.08, multiscale_convex -4.91, occluder_clusters -4.74, prisms -4.67, rotated_boxes -4.65, thin_slabs -3.59, wedges -5.08; side 64/128/256: -4.11/-4.52/-4.85; component natural/ratio-balanced/extreme: -3.95/-5.11/-3.72; Teacher-multimodal (174 w) -4.85; feasible volume >= 1e-2 (173 w) -1.71 vs < 1e-3 (1,374 w) -5.00. Ordering A > D0.3 > C > B holds in every family (e.g. thin_slabs A -3.59, D -3.38, C -1.74, B -1.54). Families have only 96-272 test windows (6-17 scenes) each.
- 194 vs 776 experiment: A_sub = 776 scenes x 16 windows (the Teacher pool) vs A_sc25 = 194 scenes (seeded 25% scene subset) x all 64 windows; both 12,416 windows, 8k steps x 64 = 512k window draws (~41 passes); test -4.43 vs -3.09; A_sc25 train NLL -7.24 vs val -2.66 at 8k (overfit), best checkpoint step 4000 (val -2.876) (`results/runs/A_sc25_s0/metrics.jsonl`; DECISION:L74, L270; `RUN_DEVIATIONS.json`). Single seed, single subset.
- Physical validity, mode audit and v0.2 canary: DECISION §6-§8 (L210-261). Canary: A transfers on informative windows (-4.39) but is worse than uniform and overconfident on natural all-lit windows (+0.11, cal 0.15).

## 6. The Gibbs tau Teacher target (mathematically)

- q_tau(s | O, K) = 1[s in Omega_T(K,W)] exp(-C_O(s;K)/tau) / Z, tau = 0.05 (fixed Phase 3 value, never tuned), Omega_T = FREE(K) minus box(W) with clearance 2^-26; Z = ∫_cube 1_legal e^{-C/tau} ds, so q is a density per unit-cube volume (`code/s3d/teacher.py:L1-17, L23`; OBJECTIVES:L58-62).
- C_O(s;K) = fraction of FREE receivers in the window whose binary lit/shadow response under candidate s differs from the observed one (normalised Hamming distance; Phase 3 `ADAPTIVE_TEACHER.md:L9-13`). In this study it is estimated on a declared **512-receiver stratified bank**: minority class (lit or shadow) gets all its receivers if <= 128 else up to 256, majority the rest; inverse-probability weights N_class/n_class; C = sum w_i mis_i / sum w_i (`teacher.py:L32-52, L83-90`). Not a full-receiver cost; banks only add phantom feasibility (RM `notes/proof_notes.md:L113-119`, Prop. O).
- Candidate set: 2^14 scrambled-Sobol uniform points + 2^14 from defensive proposal g2 = 0.5 U + 0.5 sum_c q1(c) U(cell c) over 16^3 cells; weights w = 1_legal e^{-C/tau}/g; Z_hat = (mean1 w + mean2 w)/2; q_hat self-normalised over the 2^15 points; M = 2,048 points kept by systematic resampling (no weights stored) plus fp16 16^3 coarse masses (`teacher.py:L102-138`; OBJECTIVES §5 L105-131).
- Interpretation given by the authors: e^{-C/tau} = e^{-m/(N tau)} is the likelihood of an i.i.d. per-voxel flip model with eps ~ 0.49995 for N ~ 1e5 FREE voxels, i.e. a tempered pseudo-posterior, not a calibrated posterior; p_B = E_{K|O}[q_tau] != p* (OBJECTIVES:L68-74). tau -> 0 with full receivers gives Unif(Z_O(K)), the Rao-Blackwellised form of the NLL target (OBJECTIVES:L76-80) - **not trained** (DECISION:L382).
- No soft-Hamming / k-error / zero-cost target; no mode re-weighting (OBJECTIVES:L83-86; README:L92).
- Teacher NLL at s* is log Z_hat (since C(s*) = 0 on all 16,000 windows), i.e. a pure softness measure of the window, independent of where s* sits inside the feasible set (`teacher.py:L161`).

## 7. Stated caveats, negative results, unresolved issues (as written by the study)

- Only tau = 0.05 Gibbs supervision trained; the tau -> 0 full-receiver (Rao-Blackwellised) Teacher, the only Teacher with the right population target, was not trained (DECISION:L382; TEACHER_VALUE:L143-145).
- 8-voxel resolution; no sub-8-voxel test (DECISION:L383; ARCHITECTURE:L44).
- 8k-step budget; A still improving at 16k (DECISION:L384). One density family only (no flows/mixtures) (L385).
- Canary is OOD by observation law; no natural-law training; B on a 1,500-window subset (L386). The v0.1 Student is invalid for windows that may contain s* (ARCHITECTURE:L46; DECISION:L256).
- A_priv uses only a coarse 32^3 map: measures learnability with 776 sources, not the information value of K (L387).
- Single-seed C, D, A_priv; paired CIs cover scene sampling, not seed variance (L388).
- Dropped runs: C_pool_s1, A_sc50; synthetic tempered-Teacher experiment implemented but not run (`RUN_DEVIATIONS.json`).
- Selection law is response-dependent; models estimate p(s | O, W, selected); the spatial-coverage greedy step leaves a "second order" s-dependence that is ignored (OBJECTIVES:L37-39, L46-50).
- The Teacher itself misses s*'s region (outside its 90% HPD) in 4-12.5% of mode-audit windows (DECISION:L237, L243).
- Negative results: every Gibbs arm worse than A in NLL, monotone in lambda; selective ~ random; Gibbs-trained Students under-confident; distillation makes samples less physically consistent (DECISION §1, §5, §6).

## 8. Auditor's critical observations

1. **Same selection law for train and test.** NLL is evaluated on v0.1 test windows drawn by the same selected law (s* not in W, informative, lit-ratio mixture) as training, with the same support Omega_S = cube \ W. So the comparison between arms is internally valid, but every number is for p(s | O, W, selected), not the natural law. The canary shows the Student fails (worse than uniform, overconfident) on all-lit natural windows. No importance weights were used, so nothing here estimates natural-law performance.
2. **Compute is matched only in optimizer steps.** All A/C/D arms use 8k x 64 windows (D/C also run a 256-point fine-head gather; ~same step time). Teacher generation (~2 training runs of GPU) is excluded from the "equal compute" claim, which only strengthens the negative verdict. Checkpoint rules differ by arm (own objective). For the main 8k A/C/D runs the best checkpoint was always step 8000, so the rule did not matter there. It did matter for D_sc25: selected at step 4000 (val NLL -2.77) while its val NLL peaked at -2.83 at step 6000. That slightly handicaps the hybrid in the 194-scene comparison.
3. **"Distillation hurts" is evidence against this target, not against soft labels.** The tau = 0.05 Teacher's own NLL at s* (-3.35) is worse than A (-4.46), and its NLL at s* equals log Z_hat (C(s*) = 0), which is a pure measure of how diffuse it is. Any lambda > 0 pulls a proper-score learner toward a distribution that scores worse at s*, so monotone degradation is what you would expect. Several things are confounded: (a) tempering (tau = 0.05 on a fraction-valued C, which is extremely soft per voxel); (b) the 512-receiver bank, which adds phantom feasibility (Prop. O; the bank shifts coarse masses by median TV 0.015 vs full FREE); (c) the Teacher's support (FREE(K) \ W) differs from the Student's (cube \ W); (d) the arithmetic K-mixture. The study's own synthetic Gate 2 shows that *exact* soft targets help a lot (-5.67 vs -4.85 single-sample NLL vs analytic -5.79; ABLATIONS.json gates.synthetic). So the right conclusion is "this biased Teacher hurts", and the Rao-Blackwellised tau -> 0 / full-receiver variant remains untested. The authors say this (OBJECTIVES §3; DECISION:L56-62), but the headline can be misread.
4. **The "K is worth ~4 nats" comparison is not like-for-like.** -8.44 is the mean of log V0_hat over only the 72% of windows where V0_hat > 0 (`analyze.py:L101-103`), while A's -4.46 is over 100% of windows. That number has three biases: (i) Jensen bias of log of an unbiased volume estimate, which is optimistic; (ii) a hit-resolution floor near log(2^-15), with the smallest sets excluded entirely; (iii) bank-feasible ⊇ exact feasible, so the true hard posterior is sharper. A on the same 72% subset is not reported (not found).
5. **Teacher "completeness" is relative to the same method class.** The 0 discovery and 0 candidate failures are measured against a 2^19-point plain Sobol reference with the same bank and tau, at the coarse >= 5%-mass level, on 48 windows per stratum. The rare-mode audit shows Sobol can miss tiny dominant modes silently, with independent scrambles agreeing (RM `RIND3D_RARE_MODE_TEACHER_AUDIT.md:L38-41, L86`), although real windows at tau = 0.05 and 65k points were found accurate (L61, L349). The production Teacher here used 2^15 (32k) points, not 65k.
6. **The scene-count experiment confounds sources with geometries.** In v0.1 every scene has one geometry and one source, so 194 vs 776 scenes cuts both the number of labels and K-diversity by 4x. "Number of distinct sources dominates" could equally be "number of geometries dominates". Other limits: single seed, a single seeded subset, A_sc25 early-stopped at step 4000, and A_sub's 16 windows/scene are a random sub-pick while A_sc25 uses all 64. The deviation note misstates the A_sc25 peak (it says -2.81 at step 3000; metrics show -2.876 at step 4000; `RUN_DEVIATIONS.json:L21` vs `results/runs/A_sc25_s0/metrics.jsonl`).
7. **The lambda sweep was incomplete when lambda\* was fixed.** lambda\* = 0.3 was chosen from {1, 0.3} before lambda = 0.1 finished, and lambda = 0.1 was better on test (-4.41, +0.05 vs A). So the D_sel/D_rand/16k/sc25 hybrids all use a lambda that is not even the best of the sweep. Given the monotone trend, the optimum is lambda -> 0.
8. **Validation Teacher CE uses a sorted prefix.** `train.py:L64` scores only `pts[:, :512]`. Systematic resampling returns indices in increasing order over [stage-1 Sobol, stage-2], so this prefix is roughly the stage-1/early-Sobol part of q_hat, not a uniform subsample. It affects only checkpoint selection for C/D and the logged val Teacher CE; test Teacher CE uses all 2,048 points (`evaluate.py:L183, L66-68`). The effect is probably small, but it is a protocol deviation.
9. **Input pooling discards information.** X is 4 x 32^3 pooled fraction channels, so a side-256 window loses 8x per axis. The later v0.3 decision rules out lossy pooling ("2x majority pool erases minority receivers in 96 of 120 L256 windows"; `RIND3D_Research/training_v03/RIND3D_TRAINING_READY_DECISION.md:L147`). Fraction pooling keeps counts but not positions. NLL improves with side (-4.11 / -4.52 / -4.85), so this is not obviously binding, but the Student's information ceiling is below that of the raw observation.
10. **Leakage: none found.** Splits are by geometry-group hash (1 geometry = 1 scene in v0.1; Phase3 `docs/DATASET.md` split paragraph), and no K, s*, count, selection tag or lit-ratio reaches the network (`data.py:L56-62`). Unverified: `window_features` (the code is not in the zip or the repo), and whether the B checkpoint was trained on the same train split only.
11. **Uncertainty is under-stated for non-A arms.** Bootstrap CIs are over scenes only. The A seed spread is 0.04 nats, so differences of 0.05 (D lambda = 0.1) or 0.02-0.05 (D_sel vs D_rand) are within seed noise. Only effects of >= 0.1-0.2 nats are robust.
12. **Selection-strata naming.** The "rare" screen criterion is true for 60% of the pool (`selection_report.json`), so "rare" in D_sel is not rare. Only 10% of test windows are Teacher-multimodal (174/1,728), so the multimodality claims rest on few windows (and on 48-window mode-audit strata).

## Not found / not verifiable
- `rind3d_teacher.features.window_features` source (exact pooling definition): not in the zip or the project tree.
- Model B training details (train split used, loss weights, ranking term) are in the earlier Teacher study, not this zip.
- Timestamps proving test was evaluated after 08:42 UTC: logs carry none; the claim rests on README/DECISION.
- A's NLL restricted to the 72% V0 > 0 subset (needed for a like-for-like comparison with -8.44).
- Seed variance for C/D/A_priv/A_sc25 (single seeds).
