#!/usr/bin/env bash
# Continue the exact 3D pilot in adjacent, immutable eight-slot snapshots.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dataset_root="${1:-${project_root}/dataset/artifacts/smoke-3d-impl2-v2}"
rind="${project_root}/dataset/.venv/bin/rind"
python="${project_root}/dataset/.venv/bin/python"
log_dir="${dataset_root}/expansion-logs"
mkdir -p "${log_dir}"

while true; do
  "${rind}" validate-core --dataset "${dataset_root}" > "${log_dir}/current-quick.json"
  next_slot="$("${python}" - "${dataset_root}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
snapshot = (root / "CURRENT").read_text().strip()
manifest = json.loads((root / "snapshots" / snapshot / "manifest.json").read_text())
if manifest["supported_dimensions"] != [3] or manifest["slots"][0] != 0:
    raise SystemExit("expected 3D pilot starting at slot zero")
print(manifest["slots"][1])
PY
)"
  if (( next_slot == 128 )); then
    break
  fi
  if (( next_slot < 0 || next_slot > 128 || next_slot % 8 != 0 )); then
    echo "unexpected pilot slot boundary: ${next_slot}" >&2
    exit 1
  fi
  stop=$(( next_slot + 8 ))
  echo "Building exact 3D slots ${next_slot}:${stop}"
  "${rind}" resume --dataset "${dataset_root}" --slots "${next_slot}:${stop}" \
    > "${log_dir}/resume-${next_slot}-${stop}.json" \
    2> "${log_dir}/resume-${next_slot}-${stop}.stderr"
done

"${rind}" validate-core --dataset "${dataset_root}" --deep --replay \
  > "${log_dir}/final-deep-replay.json" \
  2> "${log_dir}/final-deep-replay.stderr"
echo "3D pilot reached 128 slots and passed deep replay."
