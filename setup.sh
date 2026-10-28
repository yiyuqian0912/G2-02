#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="${RIND_UV_VERSION:-0.11.7}"
PYTHON_VERSION="${RIND_PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${RIND_TORCH_BACKEND:-cu124}"
TOOLS="$ROOT/.tools"
UV="$TOOLS/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/runtime_cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.python}"
mkdir -p "$TOOLS" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
if [[ ! -x "$UV" ]]; then
  installer="$ROOT/runtime_cache/uv-install.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer" "https://astral.sh/uv/${UV_VERSION}/install.sh"
  else
    echo 'setup.sh needs curl or wget.' >&2; exit 1
  fi
  env UV_UNMANAGED_INSTALL="$TOOLS" UV_NO_MODIFY_PATH=1 sh "$installer"
fi
case "$TORCH_BACKEND" in cu118|cu121|cu124|cpu) ;;
  *) echo 'RIND_TORCH_BACKEND must be cu118, cu121, cu124, or cpu.' >&2; exit 2 ;;
esac
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --managed-python --python "$PYTHON_VERSION" "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$UV" pip install --python "$PY" -r "$ROOT/requirements.txt"
"$UV" pip install --python "$PY" 'torch==2.4.1' --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}"
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/Phase1_Single_Source_Single_Intensity/vendor/rind-dataset"
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/Phase1_Single_Source_Single_Intensity"
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/Phase2_Multi_Source"
# Existing independent environments are useful local state, not disposable
# links. Create a shared-environment link only when no entry exists.
for phase in Phase1_Single_Source_Single_Intensity Phase2_Multi_Source; do
  if [[ ! -e "$ROOT/$phase/.venv" && ! -L "$ROOT/$phase/.venv" ]]; then
    ln -s ../.venv "$ROOT/$phase/.venv"
  fi
done
"$PY" - <<'PY_CHECK'
import sys, numpy, numba, torch
print('Python:',sys.version.split()[0])
print('NumPy:',numpy.__version__)
print('Numba:',numba.__version__)
print('PyTorch:',torch.__version__)
print('CUDA available:',torch.cuda.is_available())
print('Visible GPUs:',torch.cuda.device_count())
PY_CHECK
echo 'Setup complete. Use this environment explicitly:'
printf 'export RIND_PYTHON=%q\n' "$PY"
