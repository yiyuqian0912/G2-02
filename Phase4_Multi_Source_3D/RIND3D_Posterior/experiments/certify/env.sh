# Environment for the certified-labelling experiments (staging worktree). Source from bash.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RIND3D_REPO="$(cd "$_here/../../.." && pwd)"
export PY="${RIND3D_LABEL_PY:-/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/rind3d_numba_cert}"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH="$RIND3D_REPO/RIND3D_Posterior/src:$RIND3D_REPO/RIND3D_Dataset1K:$RIND3D_REPO/RIND3D_Dataset100K:$RIND3D_REPO/RIND3D_Generator/curved"
export WORK="${RIND3D_V3_WORK:-/home/yushu/Projects/RIND/RIND3D_100K_v3_work}"
