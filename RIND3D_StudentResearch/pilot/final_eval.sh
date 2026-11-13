#!/bin/bash
# Wait until every arm has trained (done.json) and no evaluation is running, then evaluate every arm without eval.npz.
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot
names=$(/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python -c "import json; print(' '.join(a['name'] for a in json.load(open('$R/arms.json'))))")
while true; do
  ok=1; for n in $names; do [ -f $R/$n/done.json ] || ok=0; done
  [ $ok = 1 ] && ! pgrep -f "evaluate.py" > /dev/null && break
  sleep 60
done
echo "final eval start $(date)" >> $R/eval_queue.log
/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python run_arms.py /home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1 $R $R/arms.json --eval-only >> $R/eval_queue.log 2>&1
echo "final eval done $(date)" >> $R/eval_queue.log
