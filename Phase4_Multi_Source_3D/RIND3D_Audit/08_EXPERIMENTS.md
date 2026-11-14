# 08 — Experimental evidence: scripts, configurations, results

All scripts are in `experiments/`, outputs in `experiments/out/`. Common settings:
* interpreter `/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python`;
* `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba`, `nice -n 19`;
* the release is read-only, and nothing is written outside `RIND3D_Audit/`.

Runs before the main agent's limit update used ≤ 1 process; later runs used ≤ 4 CPU processes and short GPU jobs.

## 1. Experiments run

| id | script | configuration | key result | uncertainty / caveat |
|---|---|---|---|---|
| E1 | `e1_weights_selection.py` | all 1000 scenes; scene.json, windows, windows_full | Λ̂ stats per side; Kish ESS; k by family (χ² p = 0.15); example shares by k; 25,795 all-lit + 13,443 all-dark channels; lit-fraction quantiles; c-table ranges (c(1) median 0.01 for k ≥ 2 at L64/L128) | exact counts; binomial s.e. of Λ̂ reported |
| E1b | `e1b_debalancing_ess.py` | informative examples, weights ω and ω/c(m) | example shares by m 10/20/30/40% → natural 27/25/21/26%; ESS 121.9k → 18.8k | exact given stored c, m |
| E2/E3 | `e2_source_prior_audit.py` | 6000 MC points per scene; PIT for x, y, z, r, local solid fraction; 3110 sources | per-scene uniformity not rejected (min p 0.018 of 57 tests); aggregate ≠ cube-uniform (p = 0.006), = per-scene expectation (p = 0.08); training-weighted PIT deciles within 0.083–0.112 | MC error of each PIT ≈ 0.007; float rows |
| E4 | `e4_label_coverage.py` | 333 certified release labels | true source in support 333/333; swapped axes 0/333; tv_bound max 1.7e-7 | — |
| E4b | `e4b_label_coverage_clusters.py` | one label per source, 2000 draws | 15 independent sources; KS rejects in 4.65% (nominal 5%) | low power (n = 15) |
| E4c | `e4c_label_support_structure.py` | 26-connected components at 128³ | 333/333 supports connected; single-cell 30% (L256 53%); median cells 93 | 4 scenes only |
| E4d | `e4d_rao_blackwell_gain.py` | trace of the removed gradient covariance, 1 − Σq² | mean 0.65 (128³), 0.57 (16³); L64 0.97, L256 0.42 | — |
| E5 | `e5_selection_tilt_toy.py` with `toyworld.py` | exact 2D toy: 16², 10 random obstacles (1024 worlds), 4 windows of 3 × 3, 1K κ and median c-table | label level (k = 2): ω-weighted TV ≤ 2.8e-16; unweighted mean 0.022, max 0.34. Student level: B1K vs N1 TV mean 0.031, p90 0.053, max 0.087; excess log-loss 0.0042 nats; with ω/c TV 2.5e-5 | exact enumeration; toy H range 0.18–1.32 vs 100× in the real c |
| E5b | `e5b_label_availability_toy.py` | the same toy, label filters | single-cell-only: mean TV 0.22, max 0.96, 42% of X-mass lost; drop broadest 5%: mean 0.004 | exact |
| E6 | `e6_hard_vs_soft_toy.py` | exact 2D, known G, 24² grid, windows 4 × 4 and 8 × 8; hard, flip-ε, normalised Hamming | noise-free: hard best (1.97 nats, w8); nHam β = 20: 3.36, 59% mass on all-lit candidates in Case A. With 1% true flips: hard gives the truth zero mass in 49% of cases | exact enumeration |
| E7 | `e7_onehot_vs_posterior_toy.py` | E5 world, natural law, tabular Student, oracle α, 24 replicates | excess KL at 1000 scenes: one-hot 1.89, posterior 0.66, ε-labels 0.68; infinite-data floor of ε-labels 0.0499 (bound 0.0513). Scenes vs sources per scene at a fixed budget: no difference (negative) | s.e. in the JSON; the tabular toy cannot memorise scenes |
| E8 | `e8_extreme_fixtures.py <fixture> 4000000` | 7 exact 3D fixtures, N = 32, L = 4; tau-pencil CPU vs MC | full agreement (table in 06 §3); all-dark tv_bound 0.125 caused by retained solid interior | MC s.e. per fixture |
| E9 | `e9_solid_separated_scan.py` | L64 all scenes; L128 250 scenes; L256 40 scenes | solid-separated informative channels: 6.7% / 2.5% / 0.39% | L256 sample small (2331 channels) |
| E10 | `e10_distance_visibility.py` | 3000 random FREE pairs per scene; dataset distances | P(blocked \| ℓ) rises, family-specific, exponential misfit up to 0.126; median source distance all-dark 0.71 > informative 0.62 > all-lit 0.52 | float rows |
| E11 | `e11/run_all.sh` → `run_task.py` (GPU) | FastLabel v3.4 on 36 uninformative + 6 solid-separated channels from train scenes; 150 s cap per task; labels not written | see 06 §6 | small sample; timeouts are reported as such |
| E12 | `e12_soundness_prior_mc.py e12_tasks.txt <M>` (GPU) | prior-sampled exact point tests vs label support; 10 broad FastLabel + 8 broad release labels | see §2 | |

**Reproduce (CPU):** `cd RIND3D_Audit/experiments && nice -n 19 $PY <script> > out/<name>.json`. The E8 fixtures each
take ~20–60 s (four in parallel ≈ 3 min). E11/E12 need `RIND_NVRTC_LIBRARY` (see `e11/run_all.sh`).

## 2. E12 results: independent soundness test of F ⊆ support (GPU, M = 20,000 prior points per label)

* **Labels:** 10 FastLabel `certified_stat` labels and 8 release tau-pencil labels, chosen as the broadest available (2.4·10⁴ to 9.8·10⁵ cells).
* **Feasible points:** 106,170 prior-sampled points were feasible under the exact verifier. **0 lie outside the label support** [C]. If the outer bound dropped a fraction f of F, the chance of seeing no violation in this sample is ≤ (1 − f)^106,170; this bounds the pooled unsound fraction by f ≤ 2.8·10⁻⁵ at 95% (rule of three over all points; per-label sample sizes 456–8950).
* **Volumes:**
  * Release (exact) labels: the MC volume agrees with Σ lo within 2.1 s.e. on all 8. Largest deviation: 000000/w12/s3, 0.2815 ± 0.0032 vs 0.2882.
  * FastLabel labels: vol_MC(F)/vol(B) = 0.983–1.002, i.e. p ≥ 0.95 on all 10. Consistent with every certificate. This is empirical, not a proof.
* **What it supports:** this is the first test of A1 that does not sample from B. It complements E4 (true sources) and E8 (fixtures). Recommended as a periodic production check (09 S4).

## 3. Heavier experiments prepared but not run

| id | what | command | estimated cost |
|---|---|---|---|
| P1 | Controlled Student pilot A/B/C/B′/B″ (protocol `prepared/P1_STUDENT_PILOT.md`) | labels: `prepared/p1_label_subset.py --out <external dir> --train 160 --val 40 --shard i/15` (15 shards); training with the 2026-11-10 Student code | labels ≈ 45 min full laptop (≈ 32k FastLabel labels at 12 labels/s); training 15 runs of the prior configuration (per-run time from that study's logs) |
| P2 | Real-data balancing tilt | inside P1: arms B vs B′ evaluated under ω and ω/c | included in P1 |
| P3 | Counterfactual hidden-geometry families (04 §8a) | not scripted. Needs a geometry editor that preserves the true source's visibility cone, checked with the certified renderer, plus FastLabel per edited world | ≈ 1–20 s per label; 50 windows × 20 worlds ≈ 1000 labels ≈ 2–5 min GPU-laptop, plus engineering |
| P4 | FastLabel after the all-dark fix S6 | rerun `e11/run_all.sh e11/tasks.txt` with the patched engine | ≈ 15–40 min GPU |

## 4. Audit of the existing Student experiments (brief §9.5)

Source: the 2026-11-10 Student study, from the local review ZIP. Full notes with file and line references:
`experiments/prior_student_study_notes.md`.

| item | finding | consequence |
|---|---|---|
| data | RIND3D **v0.1**, single-source scenes, not the 1K. Train 776 scenes / 49,664 windows; val 116; test 108. Split by scene. Windows from a *selected* law: informative only, lit-ratio mix 30/50/20, source never inside | the results transfer to the 1K only qualitatively |
| input features | 4×32³ pooled channels (lit-FREE, dark-FREE, SOLID, boundary) + origin + side. No k, s*, lit ratio or selection tag | no privileged leakage found (`window_features` source not available, so unchecked) |
| output | hierarchical 16³ × 8³, exactly normalised over the cube minus the window box (solids not excluded) | same grid as the labels (16 · 8 = 128) |
| losses | A: one-hot NLL; B: energy model; C: Gibbs τ = 0.05 CE on 256 of 2048 resampled points; D: A + λ·Gibbs | the Gibbs target is a soft normalised-mismatch likelihood on a 512-receiver bank, i.e. a *biased* target (05 §1) |
| weights | none (v0.1 has no ω) | the optimum is the posterior of the selected v0.1 law, evaluated on the same law. A consistent comparison, but not a natural-population claim |
| results | test NLL: A −4.46 (3 seeds, spread ≈ 0.04); D λ = 0.1/0.3/1: −4.41/−4.27/−3.90; C −2.08; B −1.67; the Teacher's own NLL at s* −3.35. Scene count: 776×16 → −4.43 vs 194×64 → −3.09 (single seed) | "direct NLL wins" against **this** Teacher. It does not test exact posterior fields. The study's own synthetic check with exact soft targets favoured them (−5.67 vs −4.85) |
| calibration | mean \|coverage − nominal\| over HPD levels 0.05–0.95; A: 0.04 | not cluster-aware (04 §7) |
| distribution shift | the Student fails on natural all-lit windows (out-of-distribution canary) | consistent with 06 §5 option A: undefined on excluded X |
| caveats from the notes | single seeds except A; λ* fixed before λ = 0.1 finished; one exploratory arm evaluated after the freeze; the scene-count test confounds sources and geometries | effects below ≈ 0.1 nats are not robust |

## 5. Limitations of this audit's evidence

* The source-tilt audit (E3) is stratified by family and k, but not by side or M.
* No explicit grazing-ray fixture was built.
* The disconnected-support fixture turned out connected (through the holes).
* E4–E4d use the only labels that exist: 333 labels from 4 scenes and 15 sources.
* All toy-world magnitudes (E5–E7) show mechanisms and orders of magnitude. They are not predictions of the 3D effect sizes, which P1 is designed to measure.
