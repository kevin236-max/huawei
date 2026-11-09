#!/usr/bin/env bash
# Run several parallel experiments one after another, one per config file.
#
# Usage:
#   bash scripts/run_sequence.sh [config ...] [-- run_parallel options]
#
# Without configs, runs the modeling-reference comparison:
#   config/config_qwen3.8-max.yaml, then config/config_qwen3.8-max_no_modeling_ref.yaml
# Each experiment is named after its config file (runs/<config name>_<time>/). Options after
# `--` (e.g. --tasks, --repeat, --parallel, --dry-run) are passed to every experiment.
# A failed experiment does not stop the next one; the exit code is non-zero if any failed.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

configs=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    configs+=("$1")
    shift
done
[[ "${1:-}" == "--" ]] && shift
if [[ ${#configs[@]} -eq 0 ]]; then
    configs=("$PROJECT_ROOT/config/config_qwen3.8-max.yaml"
             "$PROJECT_ROOT/config/config_qwen3.8-max_no_modeling_ref.yaml")
fi

for config in "${configs[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config" >&2
        exit 1
    fi
done

status=0
# Ctrl-C/SIGTERM reaches run_parallel, which interrupts its jobs and exits normally; without this
# trap bash would then start the next config. The trap runs once the current experiment returns.
trap 'echo "=== $(date "+%F %T") interrupted; remaining configs skipped"; exit 130' INT TERM
for config in "${configs[@]}"; do
    name="$(basename -- "$config" .yaml)"
    echo "=== $(date '+%F %T') start $name"
    bash "$PROJECT_ROOT/scripts/run_parallel.sh" --config "$config" --name "$name" "$@"
    code=$?
    echo "=== $(date '+%F %T') end $name (exit $code)"
    [[ $code -ne 0 ]] && status=1
done
exit $status
