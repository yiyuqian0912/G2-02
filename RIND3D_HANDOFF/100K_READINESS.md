# RIND3D-100K readiness notes (2026-11-13)

**Source:** a read-only review of branch `yushu/3d-100k-h20` at `5560a20` (`RIND3D_Dataset100K/`), plus the design studies on `yushu/3d`. **Nothing on the 100K branch was changed, merged or launched.**

## 1. What is implemented on the 100K branch

| component | state |
|---|---|
| target scale in code and contract | 100,000 scenes, 1024³, 12 windows (4 × L = 64/128/256), K = 1..4, source-separated responses, 128³ posteriors (`contract/RIND3D_100K_CONTRACT_v1.json`, status "frozen-draft") |
| physical stage | **Local-only** (`rind3d_100k/local_eval.py`): exact probes, exact hull certificates for all-lit channels, lazy acceptance, progressive certified 64³ tiles with early stop, full crops of accepted windows only. **No full-world response field.** 3.7 s per scene on the laptop |
| equivalence with the 1K full-field selection | documented on 12 scenes: identical windows and Λ̂, 0 mismatches in 2.9·10⁹ bits. The comparison harness is not committed; `scripts/smoke_test.py` checks determinism against fixtures |
| selection law | same as the 1K: m-balanced c-tables from an epoch-frozen controller (1000-slot epochs, lag 1), adaptive k with a 15% floor. Rate stream R = 256/128/64, exact m only for the first 8 rate proposals |
| storage | one npz container per scene (0.36 MB), R3LJ label journal (47 KB per label), ≈ 0.2 TB projected. Gaps: manifest without fsync; no sha256 for consolidated label files |
| labeller | tau-pencil TPE v3.1 (`rind3d_posterior.label_window`), CPU and GPU-assisted workers. Canary: 98.6% tier-1 coverage, ≈ 41 CPU-s per label, ≈ 33.4 k laptop-thread-h for 2.94 M labels [proj]. Tier 2 is in the contract but not wired |
| orchestration | coordinator, worker pool, per-GPU workers, supervision of crash / timeout / RSS, resume, STOP and grace; multi-node runs use disjoint slots, each with **its own controller history** |
| tests | none under `RIND3D_Dataset100K/tests`. Validation = `smoke_test.py` (6 steps) plus `validate.py` / `verify_root.py` |
| hardware | **everything measured on the laptop (RTX 3060).** Nothing has run on an H20 |

## 2. Shared code with `yushu/3d`

* No file is changed on both branches: a merge would be textually conflict-free.
* `RIND3D_Generator` and `RIND3D_Posterior/src` + `tests` are identical on both.
* The 100K imports from the shared packages:
  * `rind3d_1k.families` and `rind3d_1k.physical`: identical on both branches;
  * `rind3d_posterior`;
  * `rind3d`.
* It has its own loader, labels and controller.
* `yushu/3d` adds what the 100K lacks: FastLabel (`rind3d_1k/fastlabel`), the `certified_stat` status, the M2 loader (populations, `combined_rb`, explicit fallback), sha256 per label file, and the GPU semaphore / worker init hook.

## 3. Proposed scientific contract changes (`RIND3D_StudentResearch/H_100K_RECOMMENDATIONS.md`, `J_COMPLEXITY_COVERAGE.md`)

**Settled by evidence (recommended before freezing):**
* declare T* (natural multi-source worlds, single-channel posterior) in the contract;
* no m-thinning of the primary population;
* store full rate-stream m counts, so the Rao–Blackwellised weight is computable;
* per-label δ = 0.01 / N_planned;
* port the M2 loader;
* extremes stay unlabelled and stored. They fail the cost gate at 25–100× a mixed label.

**Open choices (preferred direction stated, not locked):**

| choice | options | preferred direction | basis |
|---|---|---|---|
| source-count law κ | 1K-realised (0.154, 0.135, 0.158, 0.553); rising (0.1, 0.2, 0.3, 0.4); uniform | ≥ 50% K = 4 scenes: pure natural sampling then meets every coverage target | E19 simulation with the 1K worlds |
| complexity coverage | pure natural 4+0; **3+1 at L64 only**: one coverage window in K = 4 scenes, c(m) = (1/8, 1/4, 1/2, 1); 3+1 K ≥ 3 with c = ρ^(K−m), ρ = 0.25; 2+2 | 3+1 at L64 as insurance; required if κ has < 50% K = 4. Cost: +1% labels, +14% proposals, max weight 1.26× natural. L128 and L256 need none. **2+2 is not recommended** (ESS ≈ 0.6, weights up to 2×) | E18, E19 |
| coverage-target thresholds | the declared absolute targets, per (L, m): test ≥ 300 independent scenes and ≥ 1,000 windows; train ≥ 3,000 scenes, ≥ 20,000 channels, T* ESS ≥ 5,000; per family m = 4 ≥ 30 test / 300 train scenes | adopt, or revise after the canary | judgement calibrated on the 1K pilot |
| labeller | TPE (implemented) vs FastLabel (validated at 1K scale, not integrated) | FastLabel, if the port passes Gate A and the accuracy and throughput checks on H20 | 1K: 10.2 labels/s vs TPE ≈ 0.35 labels/s on the same laptop |
| all-dark S6 | research prototype | only for an optional supplementary labelled set, after production review | `G_GENERALIZATION_AND_EXTREMES.md` §3 |

**Coverage streams must stay logically distinct.** Per window, store the stream id, proposal index, U and c-row. Per scene-side, store the rate m-histogram and the proposal counts. Combine streams only through the balance-heuristic weights 1/(n_nat/P_A + n_cov c(m)/Λ_c).

## 4. Integration work if FastLabel is adopted for the 100K (not performed)

1. **Code.** Bring `rind3d_1k/fastlabel/` and the `labels.py` / `supervise.py` / loader changes into the 100K branch, or move FastLabel into `RIND3D_Posterior` first (`CLEANUP_MANIFEST.md` §5).
2. **Scene adapter.** `label_task` reads the 1K directory layout. It needs the 100K `SceneFile` (meta resolution, geometry, crops) and a slot-based RNG scene index.
3. **Workers.** CUDA in every label worker, with one GPU semaphore per device. The 100K CPU workers currently run with `CUDA_VISIBLE_DEVICES=""`, and the 100K pool has no init hook.
4. **Statuses.** The 100K labels, validator and loader must handle `certified_stat`, kind 4 and the certificate meta.
5. **δ and seed.** Set `n_tasks` to the planned 100K label count, with a new `base_seed`.
6. **Loader constants.** Quota 4/4/4 and the 100K α values. `rb` must not be ported until full rate m counts exist.

## 5. Required canary validation on the target hardware (before production)

1. Build the environment; check NVRTC against the driver; probe FP64 and FP32 throughput. The renderer and the verifier rely on FP64 directed rounding, and H20 FP64 is reported low.
2. Cross-device bit-exactness: the smoke label digests and physical fixtures. For FastLabel, also Gate A (`RIND3D_Posterior/experiments/fastlabel/validate_verify.py`, ≈ 10 min) and `RIND3D_Dataset1K/tests/test_fastlabel.py`.
3. A canary of ≥ 256 scenes under the final contract (κ, coverage stream), checking:
   * the realised per-(L, m, family) counts against the E19 predictions, especially L64 m = 4;
   * the weight distributions and the ESS;
   * the label coverage and timeout tail;
   * per-scene and per-label throughput;
   * RAM per worker and storage I/O.
4. Accuracy of the chosen labeller on the 100K mix against certified exact references (1K: 447 references, max TV 0.0164).
5. Multi-GPU scaling and the multi-node controller history. Prefer one root, or an explicitly declared per-node plan.
