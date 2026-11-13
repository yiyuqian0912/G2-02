#!/bin/bash
# Single evaluator: repeat eval-only passes (each evaluates trained arms without eval.npz) until every arm is evaluated.
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot
n=$(/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python -c "import json; print(len(json.load(open('$R/arms.json'))))")
while [ $(ls $R/*/eval.npz 2>/dev/null | wc -l) -lt $n ]; do
  /home/yushu/Projects/Post-Attention-Arch/.venv/bin/python run_arms.py /home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1 $R $R/arms.json --eval-only 2>&1 | grep -v '"trained": false' >> $R/eval_queue.log
  sleep 120
done
echo "all evaluated $(date)" >> $R/eval_queue.log
