#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="../../src:../../../RIND3D_Generator/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" research.py "$@"
