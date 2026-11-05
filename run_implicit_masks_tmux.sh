#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIND_TMUX_SESSION="${RIND_TMUX_SESSION:-rind_masks_implicit_v6}"
export RIND_TMUX_FRONTEND=./run_implicit_masks_tmux.sh
exec "$ROOT/run_sequential_masks_tmux.sh" "$@"
