#!/bin/bash
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/s6
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src:/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K
R=/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes
for t in "000234 5 2" "000014 2 3" "000257 45 0"; do
  set -- $t
  timeout 1800 nice -n 5 /home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python validate_s6.py $R/$1 $2 $3 4096 60000 2>>validate.err | tail -1 >> s6_validation.jsonl
done
echo done > s6_validation.done
