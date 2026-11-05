#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec uv run --extra viewer python -m rind_phase3.browser --root "${RIND3D_DATA_ROOT:-$project_dir/data/RIND3D-v0.1}" "$@"
