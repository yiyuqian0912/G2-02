#!/bin/bash
# RIND3D 1K production driver: physical stage (sequential, controller) + continuously re-launched label stage.
# Usage: production_loop.sh <contract.json> <root> <slot_from> <slot_to> <workers> <gpu_workers>
# Stop safely: touch <root>/STOP   (the current label pass finishes; everything is resumable)
# Label passes run under the crash/hang supervisor (scripts/run_labels.py): failed tasks are quarantined in
# <root>/labels_failures.jsonl and reported as engine_failure / timeout slots, never as labels.
set -u
CFG=$1; ROOT=$2; A=$3; B=$4; K=$5; KG=$6
NICE=${NICE:-10}; TLIM=${TLIM:-1800,3600,7200}   # RK: tier-2 retry workers (default 6, memory)
HERE=$(cd "$(dirname "$0")/.." && pwd); REPO=$(cd "$HERE/.." && pwd)
PROD_PY=$REPO/RIND3D_Generator/.venv/bin/python
LAB_PY=${LAB_PY:-/home/yushu/Projects/RIND/RIND3D_PosteriorLabels3_Phase2/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/claude-1000/numba_pkg}
mkdir -p "$ROOT"
( cd "$REPO/RIND3D_Generator" && PYTHONPATH="$HERE" "$PROD_PY" "$HERE/scripts/run_physical.py" "$CFG" "$ROOT" "$A" "$B" ) >> "$ROOT/physical.log" 2>&1 &
PHYS=$!
echo "physical pid $PHYS" >> "$ROOT/production.log"
while true; do
  [ -f "$ROOT/STOP" ] && break
  PYTHONPATH="$REPO/RIND3D_Posterior/src:$HERE" "$LAB_PY" "$HERE/scripts/run_labels.py" "$ROOT" "$K" "$KG" --scene-order --nice $NICE --timeout $TLIM \
      --kw '{"all": {"nq_cap": 33554432, "ne_cap": 67108864}, "gpu": {"gpu_cap": 4194304}}' >> "$ROOT/labels.log" 2>&1
  if ! kill -0 $PHYS 2>/dev/null; then
    # physical stage finished: one more pass picks up the last scenes, then tier-2 retries, then stop
    PYTHONPATH="$REPO/RIND3D_Posterior/src:$HERE" "$LAB_PY" "$HERE/scripts/run_labels.py" "$ROOT" "$K" "$KG" --scene-order --nice $NICE --timeout $TLIM \
        --kw '{"all": {"nq_cap": 33554432, "ne_cap": 67108864}, "gpu": {"gpu_cap": 4194304}}' >> "$ROOT/labels.log" 2>&1
    PYTHONPATH="$REPO/RIND3D_Posterior/src:$HERE" "$LAB_PY" "$HERE/scripts/retry_uncertified.py" "$ROOT" "${RK:-6}" --nice $NICE >> "$ROOT/labels.log" 2>&1
    break
  fi
  sleep 60
done
echo "done $(date)" >> "$ROOT/production.log"
