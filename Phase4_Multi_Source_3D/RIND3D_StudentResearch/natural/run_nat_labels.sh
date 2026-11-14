#!/bin/bash
# FastLabel v3.4 labels for every MIXED channel of NAT-1K (separate delta family: 0.01 / N_SUP; base seed 20261114).
# N_SUP is the planned task count, fixed before labelling (natural/N_SUP.json).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/RIND3D_HANDOFF/env.sh"   # RIND3D_REPO, RIND3D_1K_DATA, interpreters, NVRTC
cd $RIND3D_REPO/RIND3D_Dataset1K
N=$($RIND3D_LABEL_PY -c "import json; print(json.load(open('$RIND3D_REPO/RIND3D_StudentResearch/natural/N_SUP.json'))['n_tasks'])")
exec $RIND3D_LABEL_PY scripts/run_labels.py $RIND3D_1K_DATA/natural_supplement_v1 ${1:-15} 0 --engine fastlabel --scene-order --kw "{\"fastlabel\": {\"base_seed\": 20261114, \"n_tasks\": $N}}"
