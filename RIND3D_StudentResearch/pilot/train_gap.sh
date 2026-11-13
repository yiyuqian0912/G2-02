#!/bin/bash
# Same metrics on pilot-train rows (release, B1K; 6000-row declared subsample) for four arms: train vs held-out gap.
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1; F=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1
for a in FL_B1K_s0 OH_B1K_s0 FL_B1K_sc25_s0 OH_B1K_sc25_s0; do
  [ -f $R/$a/eval_train.npz ] || /home/yushu/Projects/Post-Attention-Arch/.venv/bin/python evaluate.py $F $R/$a/best.pt $R/$a/eval_train.npz --sets b1k --role pilot-train > $R/$a/eval_train.out 2>&1
done
echo done > $R/train_gap.done
