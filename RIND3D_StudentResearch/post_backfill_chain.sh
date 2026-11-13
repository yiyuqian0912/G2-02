#!/bin/bash
# After the 1K backfill exits: verify it, finish the pilot features, then label NAT-1K (mixed) and the OOD sample.
# Each step logs to RIND3D_StudentResearch/chain/. Resumable: every step skips finished work.
set -u
S=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch
D=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K
R=/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1
F=/home/yushu/Projects/RIND/RIND3D_1K_data/pilot_features_v1
PY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python
mkdir -p $S/chain
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src
while kill -0 56390 2>/dev/null; do sleep 30; done
echo "backfill exited $(date)" >> $S/chain/chain.log
# 1. remaining feature shards (8..15) at normal priority
cd $S/pilot
for i in $(seq 8 15); do $PY build_features.py $F $i 16 >> $F/build_$i.log 2>&1 & done
# 2. verification (6 processes)
cd $D && $PY backfill/verify_backfill.py $R backfill/pre_backfill_integrity.json $S/chain/verify_backfill.json 6 > $S/chain/verify_backfill.out 2>&1
$PY backfill/integrity_snapshot.py $R $S/chain/post_backfill_integrity.json > $S/chain/post_integrity.out 2>&1
echo "verify done $(date)" >> $S/chain/chain.log
wait
while pgrep -f "build_features.py" > /dev/null; do sleep 20; done      # shards 0..7 were started earlier at nice 19
echo "features done $(date)" >> $S/chain/chain.log
# 3. NAT-1K mixed labels (12 workers), then the OOD reference sample (12 workers, 900 s cap)
$S/natural/run_nat_labels.sh 12 > $S/chain/nat_labels.out 2>&1
echo "nat labels done $(date)" >> $S/chain/chain.log
cd $S/ood && $PY label_ood.py run ood_tasks.txt 12 900 > $S/chain/ood_labels.out 2>&1
echo "ood labels done $(date)" >> $S/chain/chain.log
