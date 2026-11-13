#!/bin/bash
# Emits new lines of the training and evaluation queue logs (arm finished / failed), plus a heartbeat every 25 min.
R=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1
q0=$(wc -l < $R/queue.log); e0=$(wc -l < $R/eval_queue.log); last=$(date +%s)
while true; do
  q=$(wc -l < $R/queue.log); e=$(wc -l < $R/eval_queue.log)
  [ "$q" != "$q0" ] && { tail -n $((q - q0)) $R/queue.log | sed 's/^/TRAIN /' | cut -c1-160; q0=$q; }
  [ "$e" != "$e0" ] && { tail -n $((e - e0)) $R/eval_queue.log | sed 's/^/EVAL /' | cut -c1-160; e0=$e; }
  now=$(date +%s)
  if [ $((now - last)) -ge 1500 ]; then echo "HEARTBEAT $(ls $R/*/done.json 2>/dev/null | wc -l) trained, $(ls $R/*/eval.npz 2>/dev/null | wc -l) evaluated"; last=$now; fi
  sleep 20
done
