#!/bin/bash
# Run a production module with the certified producer interpreter (read-only checkout).
export PYTHONPATH=/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Production
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator/.venv/bin/python -m "$@"
