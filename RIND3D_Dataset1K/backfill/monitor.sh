#!/bin/bash
# Emits one progress line every ${1:-900} s, and immediately on failures, Python errors, or process exit.
R=/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1
LOG=$R/labels_fastlabel_backfill.log
PID=$(cat /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K/backfill/backfill.pid)
nf0=$(grep -c '"failure"' $LOG); ne0=$(grep -c '"error"' $R/labels_log.jsonl)
last=0
while true; do
  now=$(date +%s)
  nf=$(grep -c '"failure"' $LOG); ne=$(grep -c '"error"' $R/labels_log.jsonl)
  if [ "$nf" != "$nf0" ] || [ "$ne" != "$ne0" ]; then
    echo "ALERT failures=$nf errors=$ne $(grep '"failure"' $LOG | tail -1 | cut -c1-300)"; nf0=$nf; ne0=$ne
  fi
  if ! kill -0 $PID 2>/dev/null; then echo "EXITED: $(tail -2 $LOG | cut -c1-400)"; exit 0; fi
  if [ $((now - last)) -ge ${1:-900} ]; then
    p=$(grep '"done"' $LOG | tail -1)
    m=$(free -m | awk '/Mem:/{print $3"/"$2" MiB"}')
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo "progress $p mem=$m gpu=$g"; last=$now
  fi
  sleep 20
done
