# Resume state (stopped on user request, 2026-11-13 ~17:25 EST)

## Done and verified

* **1K backfill:** complete. 159,802/159,802 labels certified; verified (`D_BACKFILL_REPORT.md`).
* **NAT-1K natural supplement:** 907 scenes, 30,349 labels, verified.
* **OOD references:** 360/360 certified (`ood/`).
* **S6 prototype:** 18/18 on the audit's all-dark tasks.
* **Docs:** A, B (with §7, E16), C, D, F, H written; G §3 (extremes, cost gate) written.
* **Pilot features:** `RIND3D_1K_data/pilot_features_v1`, 194,532 rows.

## Pilot arms (`RIND3D_1K_data/pilot_runs_v1`): 11 of 13 trained and evaluated

* **Done:** FL_B1K_s0, OH_B1K_s0, FL_TREL_s0, FL/OH_B1K_sc25_s0, FL/OH_B1K_ex25_s0, FL_MIS_s0, OH_MIS_s0, FL_B1K_s1, OH_B1K_s1 (each with done.json and eval.npz).
* **Missing:** FL_MIS_s1 (stopped at step 7150 of 8000; its directory was removed, so it restarts) and OH_MIS_s1.
* **Train vs held-out gaps:** `experiments/out/pilot_train_gap.json` (4 arms).
* **Interim analyses:** `experiments/out/pilot_interim_*.json` (`pilot/tables.py` prints them).

## Interim results (T* in-distribution; information gain over the uniform-legal prior; scene-bootstrap 95% CI)

Prior NLL ≈ 14.55 nats.

| arm | T* gain | all-lit | all-dark | solid-sep. (natural) |
|---|---|---|---|---|
| FL_B1K s0 / s1 | 3.77 / 3.88 | −0.02 / −0.03 | +0.20 / +0.24 | +0.89 / +0.95 |
| OH_B1K s0 / s1 | 1.14 / 1.38 | −0.30 / −1.04 | −0.64 / −0.74 | +0.14 / +0.02 |
| FL_TREL s0 | 3.81 | −0.04 | +0.17 | +0.92 |
| FL_MIS s0 | **3.98** | −0.04 | +0.26 | +1.04 |
| OH_MIS s0 | 1.60 | −0.96 | −0.36 | +0.26 |
| FL_B1K_sc25 / ex25 | 3.04 / 3.80 | — | — | — |
| OH_B1K_sc25 / ex25 | −0.62 / 1.42 | — | — | — |

* **Posterior beats one-hot** by 2.4–3.7 nats in every comparison, with calibration MAE 0.03 vs 0.28.
* **One-hot memorises:** best checkpoints at step 1000–2000. At step 8000 the training NLL is ~5.7 nats, and 0.8 nats with 25% of the scenes.
* **Scenes matter, channels don't.** Holding 25% of the channels over all scenes matches the full data. Holding 25% of the scenes costs 0.77 nats.
* **Seed noise is ~0.12 nats (FL) and ~0.24 nats (OH).**
  * MIS vs B1K: −0.21 (s0 vs s0) and −0.09 (vs B1K s1).
  * TREL vs B1K: −0.04.
  * **The weighting effect is small and comparable to seed noise.** Conclude only after FL_MIS_s1.
* **OOD:** posterior Students beat the prior on solid-separated (+0.9) and all-dark (+0.2–0.26), and sit at the prior on all-lit (≈ −0.03). One-hot Students are worse than the prior on all-lit and all-dark.

## Commands to resume (about 45 min of training, 12 min of evaluation)

```bash
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1; F=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1
PY=/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python
$PY run_arms.py $F $R $R/arms.json --train-only >> $R/queue.log 2>&1     # FL_MIS_s1, OH_MIS_s1
$PY run_arms.py $F $R $R/arms.json --eval-only  >> $R/eval_queue.log 2>&1
$PY analyze.py ../experiments/out/pilot_analysis.json FL_B1K=$R/FL_B1K_s0/eval.npz ... --pairs A:B,...
$PY tables.py ../experiments/out/pilot_analysis.json
```

Then:
* fill `E_STUDENT_PILOT.md` §2;
* write `G_GENERALIZATION_AND_EXTREMES.md` §1–2 and `I_EXECUTIVE_DECISION.md`;
* commit.

## Notes

* **NVIDIA:** fixed by the reboot. Driver and library are both 620.70.
* **Process matching:** never use `pgrep -f` with a pattern that appears in the waiting shell's own command line. Use PIDs.
