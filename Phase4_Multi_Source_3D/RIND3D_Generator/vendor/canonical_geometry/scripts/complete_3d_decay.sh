#!/usr/bin/env bash
# Expand the fully audited 128-slot 3D binary pilot to two additive variants.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_root="${1:-${project_root}/dataset/artifacts/smoke-3d-impl2-v2}"
output_root="${2:-${project_root}/dataset/artifacts/decay-3d-v1}"
rind="${project_root}/dataset/.venv/bin/rind"
python="${project_root}/dataset/.venv/bin/python"

"${rind}" validate-core --dataset "${source_root}" > /dev/null
"${python}" - "${source_root}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
snapshot = (root / "CURRENT").read_text().strip()
manifest = json.loads((root / "snapshots" / snapshot / "manifest.json").read_text())
if manifest["supported_dimensions"] != [3] or manifest["slots"] != [0, 128]:
    raise SystemExit("3D decay expansion requires a complete 0:128 source")
PY

# expand-decay performs its own deep source audit before publication.
"${rind}" expand-decay --source "${source_root}" --output "${output_root}"
"${rind}" validate-core --dataset "${output_root}" --deep --replay
