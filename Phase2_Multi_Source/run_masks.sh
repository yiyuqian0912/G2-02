#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIND_DECOMPOSITION_CONFIG="${RIND_MASKS_CONFIG:-$ROOT/configs/masks_only.json}"
exec "$ROOT/run_decomposition.sh" "$@"
