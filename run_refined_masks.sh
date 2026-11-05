#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIND_MASKS_CONFIG="${RIND_MASKS_CONFIG:-$ROOT/Phase2_Multi_Source/configs/masks_boundary_refined.json}"
exec "$ROOT/run_masks.sh" "$@"
