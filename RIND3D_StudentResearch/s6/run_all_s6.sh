#!/bin/bash
# S6 prototype on the audit's 18 all-dark E11 tasks (train-split scenes), sequential, 600 s cap per task.
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/s6
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src:/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K
while read sc w s L t; do
  r=$(timeout 600 nice -n 5 /home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python run_s6.py $sc $w $s 2>>s6.err | tail -1)
  [ -z "$r" ] && r="{\"scene\": \"$(basename $sc)\", \"window\": $w, \"source\": $s, \"L\": $L, \"error\": \"no_output (timeout 600 s or crash; see s6.err)\"}"
  echo "$r" >> s6_alldark.jsonl
done < ${1:-alldark_tasks.txt}
echo done > s6_alldark.done
