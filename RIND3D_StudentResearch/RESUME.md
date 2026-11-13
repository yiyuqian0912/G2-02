# Resume state (stopped on user request, 2026-11-13 ~10:55 EST)

## Done and verified

* **1K backfill:** complete. 159,802/159,802 labels certified; verified (`D_BACKFILL_REPORT.md`).
* **NAT-1K natural supplement:** 907 scenes, 30,349 labels, verified.
* **OOD references:** 360/360 certified (`ood/`).
* **S6 prototype:** 18/18 on the audit's all-dark tasks.
* **Docs:** A, B, C, D, F, H written; G §3 (extremes, cost gate) written.
* **Pilot features:** `RIND3D_1K_data/pilot_features_v1`, 194,532 rows.

## Pilot arms (`RIND3D_1K_data/pilot_runs_v1`, `arms.json` there)

| arm | trained | evaluated |
|---|---|---|
| FL_B1K_s0 | yes (best select NLL 11.01 @ 8000) | **no** (eval was interrupted) |
| OH_B1K_s0 | yes (best 13.59 @ 2000, then overfits) | yes |
| FL_B1K_sc25_s0 | yes (best 11.74) | no |
| OH_B1K_sc25_s0 | yes (best 15.50 @ 1000: worse than the prior) | no |
| FL_B1K_ex25_s0 | yes | no |
| FL_TREL_s0 | no (failed on a selection-set bug, since fixed) | no |
| OH_B1K_ex25_s0 | no (interrupted at step ~4000; directory removed) | no |
| FL_MIS_s0, OH_MIS_s0, FL_B1K_s1, OH_B1K_s1, FL_MIS_s1, OH_MIS_s1 | no | no |

**First evaluation, OH_B1K_s0.** Information gain over the uniform-legal prior, in nats:

| set | gain |
|---|---|
| T* | 1.24 |
| B1K | 1.14 |
| all-lit | −0.30 |
| all-dark | −0.68 |
| natural solid-separated | +0.23 |
| release solid-separated | −0.42 |

## Commands to resume (about 2.7 h of training plus 1.3 h of evaluation on the laptop GPU)

```bash
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1; F=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1
PY=/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python      # torch + CUDA (read-only use of that venv)
$PY run_arms.py $F $R $R/arms.json --train-only >> $R/queue.log 2>&1      # trains only arms without done.json
$PY run_arms.py $F $R $R/arms.json --eval-only  >> $R/eval_queue.log 2>&1 # evaluates arms without eval.npz
$PY analyze.py ../experiments/out/pilot_analysis.json FL_B1K=$R/FL_B1K_s0/eval.npz OH_B1K=$R/OH_B1K_s0/eval.npz ...
```

Then:
* fill `E_STUDENT_PILOT.md` §2;
* write `G_GENERALIZATION_AND_EXTREMES.md` §1–2 and `I_EXECUTIVE_DECISION.md`;
* commit.

## Notes

* **NVIDIA:** the userspace library is 620.70 while the kernel module is 615.78 (a system update at 04:55).
  * `nvidia-smi` fails, but CUDA works.
  * Do not use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`: it needs NVML.
  * A reboot fixes it, and is safe now: nothing is running.
* **Process matching:** never use `pgrep -f` with a pattern that appears in the waiting shell's own command line. Use PIDs.
