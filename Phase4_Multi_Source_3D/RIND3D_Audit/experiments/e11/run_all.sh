#!/bin/bash
# E11 driver: one process at a time, each task capped at 150 s wall; results appended to out/e11_fastlabel_extreme.jsonl
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
PY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
cd "$(dirname "$0")/.."
while read sc w s L t; do
  r=$(timeout 150 nice -n 19 $PY e11/run_task.py $sc $w $s $L $t 2>>out/e11.err)
  if [ -z "$r" ]; then r="{\"scene\": \"$(basename $sc)\", \"window\": $w, \"source\": $s, \"L\": $L, \"type\": \"$t\", \"error\": \"timeout_150s\"}"; fi
  echo "$r" >> out/e11_fastlabel_extreme.jsonl
done < ${1:-e11/tasks.txt}
