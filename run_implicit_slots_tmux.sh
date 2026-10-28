#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$ROOT/$(basename -- "${BASH_SOURCE[0]}")"
SESSION_DEFAULT="${RIND_TMUX_SESSION:-rind_phase2_slots}"
OUTPUT_ROOT="${RIND_TMUX_OUTPUT_ROOT:-$ROOT/Phase2_Multi_Source/outputs/implicit_contribution_slots_v1}"
RUNNER="${RIND_TMUX_RUNNER:-$ROOT/run_implicit_slots.sh}"
TASK="${RIND_TMUX_TASK:-train}"
FRONTEND="${RIND_TMUX_FRONTEND:-./run_implicit_slots_tmux.sh}"
TMUX_ROOT="$OUTPUT_ROOT/tmux_runs"
LATEST_POINTER="$TMUX_ROOT/latest_run.txt"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run_implicit_slots_tmux.sh start [--session NAME] [GPU options]
  ./run_implicit_slots_tmux.sh status [--session NAME]
  ./run_implicit_slots_tmux.sh attach [--session NAME]
  ./run_implicit_slots_tmux.sh logs
  ./run_implicit_slots_tmux.sh stop [--session NAME]

GPU options are passed to run_implicit_slots.sh, for example:
  --gpus 8
  --gpu-ids 2,3
  --max-used-memory-mib 2048
  --max-utilization 10
EOF
  exit 2
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Install it once with: sudo apt-get update && sudo apt-get install -y tmux" >&2
    exit 1
  fi
}

validate_session() {
  if ! [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Session name may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
  fi
}

latest_run() {
  if [[ ! -f "$LATEST_POINTER" ]]; then
    echo "No tmux run has been recorded yet." >&2
    exit 1
  fi
  local value
  value="$(<"$LATEST_POINTER")"
  if [[ -z "$value" || ! -d "$value" ]]; then
    echo "Latest tmux run directory is unavailable: $value" >&2
    exit 1
  fi
  printf '%s\n' "$value"
}

worker() {
  local session="$1"
  local run_dir="$2"
  shift 2
  local log="$run_dir/training.log"
  set +e
  (
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tmux worker started | session=$session | pid=$$"
    echo "Results: $OUTPUT_ROOT"
    echo "Command: $RUNNER $TASK $*"
    printf '%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$run_dir/started_at.txt"
    "$RUNNER" "$TASK" "$@"
    code=$?
    printf '%s\n' "$code" > "$run_dir/exit_code.txt"
    printf '%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$run_dir/finished_at.txt"
    if [[ "$code" -eq 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] training finished successfully"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] training failed | exit_code=$code"
    fi
    exit "$code"
  ) 2>&1 | tee -a "$log"
  local code=${PIPESTATUS[0]}
  set -e
  exit "$code"
}

action="${1:-}"
[[ -n "$action" ]] || usage
shift

if [[ "$action" == "__worker" ]]; then
  [[ $# -ge 2 ]] || usage
  worker "$@"
fi

session="$SESSION_DEFAULT"
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      [[ $# -ge 2 ]] || usage
      session="$2"
      shift 2
      ;;
    *)
      arguments+=("$1")
      shift
      ;;
  esac
done
validate_session "$session"

case "$action" in
  start)
    require_tmux
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session already exists: $session" >&2
      echo "Use './run_implicit_slots_tmux.sh status --session $session' or choose another name." >&2
      exit 1
    fi
    if [[ "${#arguments[@]}" -eq 0 ]]; then
      arguments=(--gpus auto)
    fi
    mkdir -p "$TMUX_ROOT"
    run_id="$(date '+%Y%m%d_%H%M%S')"
    run_dir="$TMUX_ROOT/$run_id"
    mkdir -p "$run_dir"
    printf '%s\n' "$run_dir" > "$LATEST_POINTER"
    printf '%s\n' "$OUTPUT_ROOT" > "$run_dir/results_path.txt"
    worker_environment=(env "RIND_TMUX_OUTPUT_ROOT=$OUTPUT_ROOT" "RIND_TMUX_RUNNER=$RUNNER" "RIND_TMUX_TASK=$TASK")
    for variable in RIND_PYTHON RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG NUMBA_CACHE_DIR; do
      if [[ -n "${!variable:-}" ]]; then
        worker_environment+=("$variable=${!variable}")
      fi
    done
    printf -v worker_command '%q ' "${worker_environment[@]}" "$SELF" __worker "$session" "$run_dir" "${arguments[@]}"
    printf '%s\n' "$worker_command" > "$run_dir/command.txt"
    tmux new-session -d -s "$session" "$worker_command"
    echo "Started background training."
    echo "tmux session: $session"
    echo "log: $run_dir/training.log"
    echo "results: $OUTPUT_ROOT"
    echo "You may now disconnect SSH/VPN."
    echo "Attach: $FRONTEND attach --session $session"
    echo "Status: $FRONTEND status --session $session"
    ;;
  status)
    require_tmux
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "RUNNING | tmux session=$session"
    else
      echo "NOT RUNNING | tmux session=$session"
    fi
    if [[ -f "$LATEST_POINTER" ]]; then
      run_dir="$(<"$LATEST_POINTER")"
      echo "latest run: $run_dir"
      if [[ -f "$run_dir/exit_code.txt" ]]; then
        echo "exit code: $(<"$run_dir/exit_code.txt")"
      fi
      if [[ -f "$run_dir/training.log" ]]; then
        echo "--- last 20 log lines ---"
        tail -n 20 "$run_dir/training.log"
      fi
    fi
    ;;
  attach)
    require_tmux
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session is not running: $session" >&2
      exit 1
    fi
    exec tmux attach-session -t "$session"
    ;;
  logs)
    [[ "${#arguments[@]}" -eq 0 ]] || usage
    run_dir="$(latest_run)"
    touch "$run_dir/training.log"
    exec tail -n 100 -f "$run_dir/training.log"
    ;;
  stop)
    require_tmux
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session is not running: $session" >&2
      exit 1
    fi
    tmux send-keys -t "$session" C-c
    echo "Sent Ctrl-C to tmux session: $session"
    ;;
  *) usage ;;
esac
