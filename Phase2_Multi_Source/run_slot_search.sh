#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="$(cd "$ROOT/.." && pwd)"
task="${1:-all}"
[[ $# -eq 0 ]] || shift
case "$task" in plan|search|fit|all|evaluate) ;; *) echo 'Usage: run_slot_search.sh {plan|search|fit|all|evaluate} [--gpus auto|N] [--gpu-ids 0,1]' >&2; exit 2 ;; esac
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "$PACKAGE/Phase1_Single_Source_Single_Intensity/.venv/bin/python" "$PACKAGE/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate"; break; fi
  done
fi
[[ -n "$python_bin" ]] || { echo 'Python environment missing. Run ./setup.sh or set RIND_PYTHON.' >&2; exit 1; }
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PACKAGE/runtime_cache/numba-slot-search}"
mkdir -p "$NUMBA_CACHE_DIR"
exec "$python_bin" -u "$ROOT/research/implicit_slots/architecture_search.py" "$task" \
  --config "${RIND_SEARCH_CONFIG:-$ROOT/configs/slot_search.json}" "$@"
