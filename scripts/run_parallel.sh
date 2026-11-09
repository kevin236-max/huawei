#!/usr/bin/env bash
# Run tasks x repeats method searches concurrently; see scripts/run_parallel.py --help.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HUAWEI_PYTHON="${HUAWEI_PYTHON:-/kling_test/gen4/tangjiafu/miniconda3/envs/huawei/bin/python}"

if ! command -v -- "$HUAWEI_PYTHON" >/dev/null 2>&1; then
    echo "Python not found: $HUAWEI_PYTHON. Set HUAWEI_PYTHON to your environment's Python." >&2
    exit 1
fi

exec "$HUAWEI_PYTHON" "$PROJECT_ROOT/scripts/run_parallel.py" "$@"
