#!/bin/bash
# Run a v0.3 training-readiness module with the certified producer interpreter (read-only checkout).
# v0.2 production code (RIND3D_Production) is imported read-only; it is never modified by v0.3.
HERE=/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Training_v03
export PYTHONPATH=$HERE:/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Production
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == *.py ]]; then exec /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator/.venv/bin/python "$@"; else exec /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator/.venv/bin/python -m "$@"; fi
