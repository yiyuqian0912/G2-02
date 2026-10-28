#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIND_DECOMPOSITION_CONFIG="$ROOT/Phase2_Multi_Source/configs/decomposition_union.json"
exec "$ROOT/run_decomposition.sh" "$@"
