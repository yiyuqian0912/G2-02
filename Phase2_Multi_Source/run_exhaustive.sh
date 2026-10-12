#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --with scipy --with matplotlib python scripts/exhaustive_teacher.py "$@"
