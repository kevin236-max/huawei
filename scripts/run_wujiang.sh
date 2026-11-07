#!/usr/bin/env bash
# Run the Wujiang monthly runoff task through the existing harness.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$PROJECT_ROOT/data/wujiang"
HUAWEI_PYTHON="${HUAWEI_PYTHON:-/kling_test/gen4/tangjiafu/miniconda3/envs/huawei/bin/python}"

if ! command -v -- "$HUAWEI_PYTHON" >/dev/null 2>&1; then
    echo "Python not found: $HUAWEI_PYTHON. Set HUAWEI_PYTHON to your environment's Python." >&2
    exit 1
fi

# Default to a full search; also accept run.py's inspection/evaluation commands.
if [[ $# -eq 0 || "$1" == -* ]]; then
    set -- run "$@"
fi

# Keep caller-relative --config/--candidate paths intact and pin the task to Wujiang.
exec "$HUAWEI_PYTHON" "$PROJECT_ROOT/run.py" "$@" --task "$TASK_DIR"
