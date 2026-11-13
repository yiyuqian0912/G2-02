#!/bin/bash
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/experiments/fastlabel
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
PY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
run() { $PY runner3.py 4 ${3:-15} 7 runs/$1.jsonl --gpu-permits 1 --kw "$2" > runs/$1.log 2>&1; }
H='"race_nodes": 0, "nbd": 256, "nin": 128, "small_records": 0'
run v3l "{$H, \"s1_tests\": 320, \"s1_nb\": 256}"
run v3m "{$H, \"s1_d1\": 3}"
run v3n '{"race_nodes": 0, "nbd": 128, "nin": 64, "small_records": 0}'
run v3o "{$H}" 16
