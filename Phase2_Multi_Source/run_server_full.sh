#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

command_name="${1:-all}"
if [[ "$command_name" != "prepare" && "$command_name" != "train" && "$command_name" != "all" ]]; then
  echo "Usage: ./run_server_full.sh [prepare|train|all]" >&2
  exit 2
fi

if [[ -n "${RIND_PYTHON:-}" ]]; then
  training_python="$RIND_PYTHON"
else
  training_python="$PROJECT_DIR/.venv/bin/python"
fi
if [[ ! -x "$training_python" ]]; then
  echo "Missing server environment: $training_python" >&2
  echo "Run: uv sync --locked --extra train --extra diagnostics --extra browser" >&2
  exit 1
fi

if [[ "$command_name" == "prepare" || "$command_name" == "train" || "$command_name" == "all" ]]; then
  "$training_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available in this environment"; print("Visible CUDA devices:", torch.cuda.device_count())'
fi

export RIND_PYTHON="$training_python"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PROJECT_DIR/outputs/.numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/outputs/.matplotlib}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
exec "$PROJECT_DIR/run_combined_training.sh" "$command_name" \
  --config "$PROJECT_DIR/configs/layered_combined_server_full.json" "${@:2}"
