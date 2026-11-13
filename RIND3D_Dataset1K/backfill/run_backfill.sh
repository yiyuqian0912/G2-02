#!/bin/bash
# Approved 1K FastLabel v3.4 posterior backfill (2026-11-13). Resumable: rerunning this script skips finished task files.
cd /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K
export RIND_NVRTC_LIBRARY=/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/yushu/.cache/rind3d_numba
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src
exec /home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python scripts/run_labels.py /home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1 15 0 --engine fastlabel --scene-order --retry-failed
