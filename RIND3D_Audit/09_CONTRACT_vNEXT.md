# 09 — Recommended RIND3D-vNext scientific contract

Nothing here changes a frozen artefact. Every 1K item is a loader, weight, metadata or documentation change. 100K items
concern a contract that is not frozen.

## 1. Candidate scientific designs (brief §10)

Ratings: ++ strong, + adequate, − weak, −− unsound for the stated purpose.

| criterion | A: current hard posterior + ω | B: corrected hard posterior | C: explicit noise posterior | D: hard + auxiliary soft | E: one-hot NLL (with correct weights) | F: hybrid / curriculum | G: multi-channel Student, hard labels |
|---|---|---|---|---|---|---|---|
| clarity | + | ++ | + | − | ++ | − | + |
| correct probabilistic meaning | label level ++; Student level − (B1K target, 03 §3) | ++ (declared target T, weights ω/c or a natural selection law) | ++ only if noise is real; −− here (05 §3) | − (mixture of two targets) | ++ (same functional as B, 04 Cor. 1) | − unless label-or-one-hot fallback, which equals B | ++ (balancing is observable and cancels) |
| identifiability / use of geometry | ++ | ++ | − (blurs decisive small shadows; E6) | + | + (same optimum, noisier) | + | ++ |
| multimodality | ++ (q_X mixture) | ++ | + | + | ++ (needs a flexible head) | + | ++ |
| robustness to perturbations | − (brittle per world; smooth after the mixture) | − (same) | ++ if the noise model is right | + | − | + | − |
| calibration | + (target is B1K) | ++ | ++ under its own noise model | − | ++ | − | ++ |
| dataset bias | − (balancing tilt, loader filter) | ++ | − | − | + (no label-availability bias; still needs the target weights) | − | + |
| generalisation, finite data | ++ (Rao–Blackwell; toy ≈ 3× data) | ++ | + | + | − (variance) | + | ++ |
| compute | labels ≈ 4.3 h per 1K (FastLabel) | same + weights | relabel with a new likelihood | labels + soft term | none for labels | labels | same as B |
| feasible with current labels | yes | yes (weights from stored m, c) | no | partly | yes (sources are stored) | yes | yes |
| scalable to 100K | labels CPU/GPU-bound (≈ 33k thread-h TPE; FastLabel ≪) | same | worse | worse | trivially | same | same |
| novelty / publishability | + | ++ (certified amortised posterior with a stated selection law) | − | − | − (standard NPE) | − | ++ (source-separated multi-layer inference) |

**Primary formulation (recommended): Design B.**
* Labels: hard Uniform(F) per world (unchanged).
* Target: an explicitly declared population T. Recommended: N1-type, natural single-source, optionally stratified by X-measurable response types.
* Loss: forward CE to the certified labels, with a one-hot fallback where a label is missing; weights making the training law equal T.
* Evaluation: one-hot NLL on held-out scenes (label CE as a control variate).

**Ablations:**
* E (one-hot only), to measure the Rao–Blackwell gain on real data;
* G (multi-channel Student), for the effective-layer research question;
* D with a small-ε flip term, only together with a perturbation benchmark.

**Not recommended:** C (no noise in the simulated world); normalised-Hamming soft targets; λ-hybrids without a noise model.

## 2. MUST FIX (demonstrated inconsistencies; fix before Student training or before the extension concerned)

| id | inconsistency | minimal reproducible case | proposed fix (no frozen artefact changes) |
|---|---|---|---|
| M1 | **The Student target population is undeclared, and the flattened ω-weighted population is not a natural one.** The hidden-channel thinning c(I_j + M₋ⱼ) ∈ [0.01, 1] tilts P(G\|X). ω does not remove it. | E5 toy: TV(B1K optimum, N1 optimum) mean 0.031, max 0.087; X-marginal TV 0.24. Real data: c spans 100× within scenes; example shares by M are 10/20/30/40% vs natural 27/25/21/26% (E1b) | Declare T in the contract. For T = N1, add `flatten(weight="natural")` = ω/c(m, L, k) from stored `m` and `c_table` (ESS 121.9k → 18.8k), **or** train the multi-channel Student (Design G), where the factor cancels. For T = B1K, state that the Student's posterior depends on the controller history, and evaluate only on B1K |
| M2 | **The default loader filter selects on privileged label paths.** `flatten(certified_only=True, accept_statistical=False)` keeps only `certified`, i.e. single-cell or exact labels after a FastLabel backfill | DEV + HOLDOUT (5082 labels): 67.4% `certified_stat`; retention 5.8% / 25.4% / 57.4% at L64/128/256, all single-cell. E5b toy: mean TV 0.22 and 42% of X-mass lost | Accept every certified status by default (`certified`, `certified_stat`; `certified_loose` with its bound recorded). For missing or uncertified labels use the unbiased **label-or-one-hot** target (04 §2), never drop. Log the dropped ω-mass per stratum |
| M3 | **The FastLabel δ budget is hard-coded** to N_DATASET = 159,802 | the same δ_label on 2.94M 100K labels gives a union bound of 0.18; on 1K plus uninformative channels, 0.0125 | δ_label = δ_total / N_planned from the actual task list; store N_planned and the policy in every certificate; charge reruns (07 §4) |
| M4 | **Uninformative channels of the release are not natural single-channel examples.** They exist only if another hidden channel is informative (h_t(G,W,0) requires M₋ⱼ ≥ 1; zero probability in k = 1 scenes, so no weight can correct it) | 03 §3; 25,795 all-lit and 13,443 all-dark channels, all from k ≥ 2 scenes (02 §7) | Never feed them to a single-channel Student under T = N1. Include all-lit/all-dark only through a selection law designed for them (§5 below, 06) |

## 3. SHOULD CHANGE (strongly supported improvements)

| id | change | why |
|---|---|---|
| S1 | Normalise ω by its per-side mean (any function of L only) | optimum unchanged (04 Thm 1); weight shares go from 5/30/64% to 24/38/38%; pooled ESS 100.7k → 121.9k |
| S2 | Use the `combined` ω (strictly positive) as the default | removes the Λ̂ = 0 scene-side; ESS +0.4% |
| S3 | Mark `m`, `mask`, `k` in `public/windows.npy` as privileged for single-channel Students (and add a loader guard) | they encode the hidden channels |
| S4 | Per-label soundness asserts: cell(s_j) ∈ support; plus a periodic prior-sample soundness check (E12) | A1 is the foundation of every certificate and is not tested by it (07 §4) |
| S5 | Record the configuration hash, attempt count and N_planned in every certificate | re-rolls and config reruns must be charged to δ |
| S6 | Exclude solid interiors explicitly in the window-side (wmode) path of the tau-pencil engine, and in FastLabel's B, before any all-dark labelling | all-dark fixture: tv_bound 0.125 although the label is exact (hi − lo = solid volume). Real all-dark channels: 9/18 certified, 3 uncertified, 6 timeouts (E11, 06 §3, §6) |
| S7 | Evaluation protocol of 04 §7: one-hot NLL + label CE control variate; cluster-aware SBC/coverage; strata by side, family, lit fraction, solid-separated | 333 labels = 15 independent sources (E4b) |
| S8 | State the scene count, not the example count, as the dataset size in all reports | 04 §6 |

## 4. OPTIONAL (ablations, further research)

* O1: one-hot vs posterior vs label-or-one-hot pilot on real data (08 §3, prepared).
* O2: multi-channel Student (Design G) for the effective-layer question.
* O3: small-ε flip auxiliary loss + perturbation benchmark (05 §6).
* O4: counterfactual hidden-geometry families (04 §8) for the ambiguity diagnostics.
* O5: FastLabel `point_items` boundary set extended to FREE voxels adjacent to SOLID. Efficiency only: 6.7% of L64 informative channels are solid-separated (E9).
* O6: fixed-proposal-budget sampling (no quotas), which is self-weighting and needs no ω for the 1/Λ factor (§5).

## 5. RIND3D-100K: necessary corrections vs optional improvements

**Necessary, before freezing the 100K contract:**
1. **Declare the Student and its target population.**
   * Single-channel Student: **drop effective-layer balancing from the acceptance rule** (c ≡ 1), or make thinning depend only on X-measurable features of the *same* channel (side, lit-fraction bin, response type, solid-separated flag). X-measurable thinning leaves P(S|X) unchanged without weights ([Zadrozny2004]; 03 §1) and only changes P(X), which evaluation can re-weight by a function of X.
   * Multi-channel Student: M-balancing is acceptable (it cancels), but the per-channel labels must then be used with all channels as input.
2. **Loader default:** M2 above. In the 100K canary, 1.4% of tasks timed out. Those must enter through the label-or-one-hot target, not be dropped.
3. **δ budget:** M3.
4. **Store everything needed to reproduce any target weight:**
   * per window: m, mask, c-table, k, and U (already in the selection log);
   * per side: rate and quota counts;
   * per channel: response-type flags;
   * per label: status, path, certificate, and (privileged) cell(s_j) ∈ support.

**Recommended:**
* **Source prior:** keep i.i.d. Uniform(D(G)) with the anti-duplicate rule [KEEP].
* **Source count:** use a fixed κ (e.g. uniform on 1..4, keyed by slot) instead of the adaptive controller. The population is then a fixed, reproducible distribution rather than a function of the run history, and k is independent of every scene. Adaptive k is valid but makes T harder to state.
* **Scenes vs windows:** keep 100K scenes × 12 windows. Independent scenes are the effective sample size (04 §6); the prior study measured −4.43 vs −3.09 test NLL for 4× more scenes at equal windows.
* **All-lit / all-dark:** include as an explicit stratum with a declared X-measurable thinning rate (06 §5), labelled by FastLabel after S6. All-lit labels are broad (10⁵–2·10⁶ cells, 3–24 s each, E11), so budget them.
* **Weights:** keep ω for the quota normalisation, side-normalised and `combined`. Alternatively use a fixed proposal budget per side with no quota stop: accepted windows then have density ∝ p_nat A c, and no 1/Λ correction is needed.
* **Train/test shift evaluation:** report every metric under T (ω-weighted) and under the raw accepted law; report per stratum; scene-bootstrap intervals; a separate perturbation set.

## 6. KEEP (supported by the audit)

* Per-scene i.i.d. uniform legal sources; proposals independent of S; k and c fixed before S (H1–H3, E2).
* The anti-duplicate rule instead of MIN_SEPARATION.
* Closed-box exclusion, implemented identically in generator and labeller.
* Source-separated channels, stored canonically and never pre-flattened.
* Hard Uniform(F) per-world labels on 128³ cells (03, 05).
* ω for the quota normalisation (required: toy label TV 0.022 mean / 0.34 max without it).
* The FastLabel certificate design: B ⊇ F, two-stage exact α, keyed streams, simultaneous δ (07).
* Scene-level hash splits.
* Rejection of coarse 128³ screening in the 100K (it would select against fine shadows).
