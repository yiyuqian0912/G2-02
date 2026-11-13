#!/bin/bash
# Emits a line on: new chain step, new queue line (arm finished), NAT/OOD labelling failures, E16 finished; plus a heartbeat.
S=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch
Q=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_runs_v1/queue.log
c0=$(wc -l < $S/chain/chain.log); q0=$(wc -l < $Q 2>/dev/null || echo 0); f0=0; e16=0; last=$(date +%s)
while true; do
  c=$(wc -l < $S/chain/chain.log); q=$(wc -l < $Q 2>/dev/null || echo 0)
  [ "$c" != "$c0" ] && { tail -n $((c - c0)) $S/chain/chain.log | sed 's/^/CHAIN /'; c0=$c; }
  [ "$q" != "$q0" ] && { tail -n $((q - q0)) $Q | sed 's/^/ARM /' | cut -c1-200; q0=$q; }
  f=$(cat $S/chain/nat_labels.out $S/chain/ood_labels.out 2>/dev/null | grep -c '"failure"')
  [ "$f" != "$f0" ] && { echo "LABEL-FAILURES $f"; f0=$f; }
  if [ $e16 = 0 ] && [ -s $S/experiments/out/e16_tabular_tilt.json ]; then echo "E16 done"; e16=1; fi
  now=$(date +%s)
  if [ $((now - last)) -ge 1500 ]; then
    echo "HEARTBEAT nat=$(grep '"done"' $S/chain/nat_labels.out 2>/dev/null | tail -1 | cut -c1-60)"; last=$now
  fi
  sleep 20
done
