#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIND_MASKS_CONFIG="${RIND_MASKS_CONFIG:-$ROOT/Phase2_Multi_Source/configs/masks_expanded.json}"
export RIND_TMUX_SESSION="${RIND_TMUX_SESSION:-rind_masks_expanded}"
export RIND_TMUX_FRONTEND=./run_masks_expanded_tmux.sh
exec "$ROOT/run_masks_tmux.sh" "$@"
