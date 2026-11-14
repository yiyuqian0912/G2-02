#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${RIND_PYTHON:-${root}/.venv/bin/python}"
export RIND_PYTHON="${python}"
"${root}/vendor/canonical_geometry/scripts/build_native.sh"
# uv's editable build contains the three generator extensions; rebuilding the
# canonical dependency above does not modify any sibling checkout.
"${python}" -c 'import rind._exact2d, rind._exact3d, rind3d._native, rind3d._convex, rind3d._research; print("RIND3D native and canonical backends import successfully")'
