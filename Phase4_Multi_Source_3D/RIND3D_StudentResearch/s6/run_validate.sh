#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/RIND3D_HANDOFF/env.sh"   # RIND3D_REPO, RIND3D_1K_DATA, interpreters, NVRTC
cd $RIND3D_REPO/RIND3D_StudentResearch/s6
R=$RIND3D_1K_DATA/release_1k_v1/scenes
for t in "000234 5 2" "000014 2 3" "000257 45 0"; do
  set -- $t
  timeout 1800 nice -n 5 $RIND3D_LABEL_PY validate_s6.py $R/$1 $2 $3 4096 60000 2>>validate.err | tail -1 >> s6_validation.jsonl
done
echo done > s6_validation.done
