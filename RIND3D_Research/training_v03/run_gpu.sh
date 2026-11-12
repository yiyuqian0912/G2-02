#!/bin/bash
# Benchmark-only CUDA interpreter (.venv-gpu: Python 3.12, numpy, zstandard, torch 2.14.1+cu130 from the local uv cache).
# Student-side modules only (rind3d_training.student*, bench_train): NumPy/zstd readers of exported student shards.
HERE=/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Training_v03
export PYTHONPATH=$HERE
export PYTHONDONTWRITEBYTECODE=1
if [[ "$1" == *.py ]]; then exec $HERE/.venv-gpu/bin/python "$@"; else exec $HERE/.venv-gpu/bin/python -m "$@"; fi
