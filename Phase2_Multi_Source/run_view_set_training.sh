#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$project_dir/.." && pwd)"

if [[ -n "${RIND_PYTHON:-}" ]]; then
  python_bin="$RIND_PYTHON"
elif [[ -x "$repo_dir/Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  python_bin="$repo_dir/Phase1_Single_Source_Single_Intensity/.venv/bin/python"
else
  python_bin="$project_dir/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment is missing. Set RIND_PYTHON or create a project .venv." >&2
  exit 1
fi

export PYTHONPATH="$repo_dir/Phase1_Single_Source_Single_Intensity/src:$project_dir/src:$project_dir/vendor/rind-dataset:$project_dir/research/layered_residual${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$project_dir/research/layered_residual/train_view_sets.py" "$@"
