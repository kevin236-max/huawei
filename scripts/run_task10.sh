#!/usr/bin/env bash
# Run every data/task_* task (prompt: prompts/main.md) through scripts/run_parallel.sh.
#
# Usage:
#   bash scripts/run_task10.sh [run_parallel options]
#
# Defaults: config/config_qwen3.8-max.yaml (prompts.main = prompts/main.md), experiment name
# task10; repeat, parallel and staggers come from that config's batch section. Any option is
# passed through and overrides these, e.g. --repeat 3, --parallel 10, --dry-run, or
# --config config/config_qwen3.8-27B.yaml. Ctrl-C stops the run as in run_parallel.sh.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

tasks=()
for dir in "$PROJECT_ROOT"/data/task_*/; do
    [[ -f "$dir/benchmark.yaml" ]] && tasks+=("${dir%/}")
done
if [[ ${#tasks[@]} -eq 0 ]]; then
    echo "No data/task_* directories with benchmark.yaml found." >&2
    exit 1
fi
task_list="$(IFS=,; echo "${tasks[*]}")"

exec bash "$PROJECT_ROOT/scripts/run_parallel.sh" \
    --config "$PROJECT_ROOT/config/config_qwen3.8-max.yaml" \
    --tasks "$task_list" --name task10 "$@"
