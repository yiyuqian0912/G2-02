#!/usr/bin/env bash
set -euo pipefail
PHASE2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="$(cd -- "$PHASE2_ROOT/.." && pwd)"
archive="${1:-$PACKAGE_ROOT/RIND-random-v3-100000scenes-multi-unit-256views.zip}"
teacher_root="${2:-$PHASE2_ROOT/outputs/full_training_h20_v2}"
expected_sha256="44015d37e6abb2093228380c237beeea1e71266b87640a5e8cdc05fe4a36d174"
if [[ -n "${RIND_PYTHON:-}" ]]; then
  python_bin="$RIND_PYTHON"
elif [[ -x "$PACKAGE_ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  python_bin="$PACKAGE_ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python"
elif [[ -x "$PACKAGE_ROOT/.venv/bin/python" ]]; then
  python_bin="$PACKAGE_ROOT/.venv/bin/python"
else
  echo "Python environment missing; set RIND_PYTHON=/absolute/path/to/python" >&2
  exit 1
fi
if [[ ! -f "$archive" ]]; then
  echo "Dataset ZIP missing: $archive" >&2
  exit 1
fi
export PYTHONPATH="$PHASE2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -u -m dataset_update.manage_dataset install \
  --archive "$archive" \
  --destination "$PHASE2_ROOT/data/raw/phase2-unit-random-100k" \
  --sha256 "$expected_sha256"
"$python_bin" -u -m dataset_update.manage_dataset randomize \
  --base-root "$PHASE2_ROOT/data/raw/phase2-unit-random-100k" \
  --destination "$PHASE2_ROOT/data/raw/phase2-random-strength-100k" \
  --seed 20261028
"$python_bin" -u -m dataset_update.manage_dataset append \
  --base-root "$PHASE2_ROOT/data/raw/phase2-multi-source" \
  --appended-root "$PHASE2_ROOT/data/raw/phase2-random-strength-100k" \
  --destination "$PHASE2_ROOT/data/raw/phase2-merged-110k" \
  --teacher-root "$teacher_root"
echo "Merged dataset ready: $PHASE2_ROOT/data/raw/phase2-merged-110k"
echo "Old IDs/caches preserved. New windows require local labels; candidate rendering is needed only for full localization teachers."
