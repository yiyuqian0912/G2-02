#!/bin/bash
# Train the remaining arms, then evaluate every arm without eval.npz (sequential, one process at a time).
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1; F=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1
PY=/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot
echo "resume $(date)" >> $R/queue.log
$PY run_arms.py $F $R $R/arms.json --train-only >> $R/queue.log 2>&1
$PY run_arms.py $F $R $R/arms.json --eval-only 2>&1 | grep -v '"trained": false' >> $R/eval_queue.log
echo "finished $(date) $(ls $R/*/eval.npz | wc -l) evaluated" >> $R/eval_queue.log
