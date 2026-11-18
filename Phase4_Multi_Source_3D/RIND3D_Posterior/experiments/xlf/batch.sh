#!/bin/bash
# usage: batch.sh OUT.jsonl TIMELIMIT KEY...
source "$(dirname "$0")/../certify/env.sh"
out=$1; tl=$2; shift 2
for k in "$@"; do
  XLF_CAP_FINE=${XLF_CAP_FINE:-16384} timeout $((tl + 300)) $PY "$(dirname "$0")/run_one.py" $k $tl 2>/dev/null | tail -1 >> "$out" || echo "{\"key\": \"$k\", \"error\": true}" >> "$out"
done
echo done
