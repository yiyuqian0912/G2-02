#!/bin/bash
# S6 prototype on the audit's 18 all-dark E11 tasks (train-split scenes), sequential, 600 s cap per task.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/RIND3D_HANDOFF/env.sh"   # RIND3D_REPO, RIND3D_1K_DATA, interpreters, NVRTC
cd $RIND3D_REPO/RIND3D_StudentResearch/s6
while read sc w s L t; do
  r=$(timeout 600 nice -n 5 $RIND3D_LABEL_PY run_s6.py $sc $w $s 2>>s6.err | tail -1)
  [ -z "$r" ] && r="{\"scene\": \"$(basename $sc)\", \"window\": $w, \"source\": $s, \"L\": $L, \"error\": \"no_output (timeout 600 s or crash; see s6.err)\"}"
  echo "$r" >> s6_alldark.jsonl
done < ${1:-alldark_tasks.txt}
echo done > s6_alldark.done
