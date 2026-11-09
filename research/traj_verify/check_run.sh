#!/usr/bin/env bash
# Research only: run capture_requests.mjs + render_check.py on every trajectory of one local-27B run.
# Usage: bash check_run.sh <run repeat dir> [effort]   (effort defaults to the run's config.json value)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
RUN="$(cd -- "$1" && pwd)"
NODE="$ROOT/.runtime/node/bin/node"
PY="${QWEN_PY:-/kling_test/gen4/tangjiafu/miniconda3/envs/qwen38-cu12/bin/python}"
MODEL_DIR="${MODEL_DIR:-/kling_test/gen4/tangjiafu/model_list/Qwen3.8-27B}"
read -r MODEL EFFORT MAXTOK < <(python3 -c "import json,sys;a=json.load(open(sys.argv[1]))['api'];print(a['model'],a.get('reasoning_effort') or 'off',a['max_tokens'])" "$RUN/config.json")
CHECK_EFFORT="${2:-$EFFORT}"
OUT="$HERE/out/$(basename "$(dirname "$RUN")")_$(basename "$RUN")"
mkdir -p "$OUT"
find "$RUN" -name trajectory.jsonl | sort | while read -r traj; do
    name="$(realpath --relative-to="$RUN" "$(dirname "$traj")" | tr / _)"
    (cd "$HERE" && timeout 120 "$NODE" capture_requests.mjs "$traj" "$OUT/$name.requests.jsonl" "$MODEL" "$EFFORT" "$MAXTOK")
    echo "== $name"
    "$PY" -B "$HERE/render_check.py" "$traj" "$OUT/$name.requests.jsonl" "$MODEL_DIR" --effort "$CHECK_EFFORT" 2>/dev/null | tail -1
done
