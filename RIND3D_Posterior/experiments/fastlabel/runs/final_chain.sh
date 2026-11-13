#!/bin/bash
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/experiments/fastlabel
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
PY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
DEV=$($PY -c "import json; print(','.join(str(int(x)) for x in json.load(open('bench_plan.json'))['dev']))")
HOLD=$($PY -c "import json; print(','.join(str(int(x)) for x in json.load(open('bench_plan.json'))['holdout']))")
$PY runner3.py 0 15 0 runs/dev18_v34.jsonl --gpu-permits 1 --scenes $DEV > runs/dev18_v34.log 2>&1
$PY runner3.py 0 15 0 runs/holdout18_v34.jsonl --gpu-permits 1 --scenes $HOLD > runs/holdout18_v34.log 2>&1
$PY runner3.py 0 15 0 runs/acc_v34.jsonl --gpu-permits 1 --tasks runs/acc_tasks.txt --kw '{"validate_n": 4096}' > runs/acc_v34.log 2>&1
echo chain-done > runs/final_chain.done
