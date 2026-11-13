#!/bin/bash
# FastLabel v3.4 labels for every MIXED channel of NAT-1K (separate delta family: 0.01 / N_SUP; base seed 20261114).
# N_SUP is the planned task count, fixed before labelling (natural/N_SUP.json).
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src
N=$(/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python -c "import json; print(json.load(open('/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/natural/N_SUP.json'))['n_tasks'])")
exec /home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python scripts/run_labels.py /home/yushu/Projects/RIND/RIND3D_1K_data/natural_supplement_v1 ${1:-15} 0 --engine fastlabel --scene-order --kw "{\"fastlabel\": {\"base_seed\": 20261114, \"n_tasks\": $N}}"
