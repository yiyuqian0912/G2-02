# RIND3D environment for the 1K reference pipeline. Source it from bash:  source RIND3D_HANDOFF/env.sh
# Every variable can be overridden before sourcing; the defaults are the values of the original development machine.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RIND3D_REPO: the Phase4 root (Phase4_Multi_Source_3D/ in the G2-02 repository), which holds every RIND3D_* component
export RIND3D_REPO="${RIND3D_REPO:-$(cd "$_here/.." && pwd)}"
export RIND3D_1K_DATA="${RIND3D_1K_DATA:-/home/yushu/Projects/RIND/RIND3D_1K_data}"       # release_1k_v1, natural_supplement_v1, pilot_*
# labeller interpreter: Python 3.12 with numpy, numba, scipy, zstandard, the built rind3d package (RIND3D_Generator)
export RIND3D_LABEL_PY="${RIND3D_LABEL_PY:-/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/bin/python}"
# Student interpreter: Python 3.12 with CUDA PyTorch and numpy/scipy (no zstandard needed)
export RIND3D_TORCH_PY="${RIND3D_TORCH_PY:-/home/yushu/Projects/Post-Attention-Arch/.venv/bin/python}"
# CUDA runtime compiler used by the certified CUDA backend and the FastLabel verifier (NVRTC 12)
export RIND_NVRTC_LIBRARY="${RIND_NVRTC_LIBRARY:-/home/yushu/Projects/RIND/G2-02-3d-100k/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/rind3d_numba}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export PYTHONPATH="$RIND3D_REPO/RIND3D_Posterior/src:$RIND3D_REPO/RIND3D_Dataset1K:$RIND3D_REPO/RIND3D_Generator/curved:$RIND3D_REPO/RIND3D_Generator/voxel${PYTHONPATH:+:$PYTHONPATH}"
