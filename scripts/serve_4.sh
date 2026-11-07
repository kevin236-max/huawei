#!/usr/bin/env bash
# Serve the existing Qwen model using the read-only, patched qwen38-cu12 env.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash agent/scripts/serve.sh [--dry-run] [vLLM options...]

Defaults: Qwen3.8-27B, GPUs 0-7, TP=8, DP=1, 127.0.0.1:18000.
Cross-request prefix caching is enabled with Mamba cache mode align.
The script activates qwen38-cu12 automatically. No installation is performed.
--dry-run must be first; prints the command without activating or starting vLLM,
importing GPU libraries, or creating logs/cache directories.

Environment overrides:
  CONDA_ROOT, QWEN_ENV, QWEN_PY, CUDA_VISIBLE_DEVICES, MODEL_PATH, MODEL_NAME
  HOST, PORT, TP_SIZE, DP_SIZE, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION
  MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS, MIN_FRAME_TOKENS, MAX_FRAME_TOKENS
  SERVER_LOG_DIR, XDG_CACHE_HOME, VLLM_CACHE_ROOT, HF_HOME

Extra arguments are forwarded last, e.g. --data-parallel-size 2.
Logs and caches default to agent/results/Qwen3.8-27B/server and agent/.cache.
Same-host client: base_url=http://127.0.0.1:18000/v1, model=Qwen3.8-27B.
api_key=EMPTY is a client placeholder only when neither --api-key nor
VLLM_API_KEY configures server authentication.
HELP
    exit 0
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/kling_test/gen4/tangjiafu/miniconda3}"
QWEN_ENV="${QWEN_ENV:-qwen38-cu12}"
MODEL_PATH="${MODEL_PATH:-/kling_test/gen4/tangjiafu/model_list/Qwen3.8-27B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.8-27B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MIN_FRAME_TOKENS="${MIN_FRAME_TOKENS:-128}"
MAX_FRAME_TOKENS="${MAX_FRAME_TOKENS:-768}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${PROJECT_ROOT}/results/Qwen3.8-27B/server}"

# These values participate in shell arithmetic below.
for name in MAX_MODEL_LEN MIN_FRAME_TOKENS MAX_FRAME_TOKENS; do
    value="${!name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ || ${#value} -gt 9 ]]; then
        printf '%s must be a positive integer of at most 9 digits\n' "$name" >&2
        exit 2
    fi
done
if (( MIN_FRAME_TOKENS > MAX_FRAME_TOKENS )); then
    echo 'MIN_FRAME_TOKENS must not exceed MAX_FRAME_TOKENS' >&2
    exit 2
fi

# Keep bytecode and runtime caches out of the external source/env/model trees.
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/cuda}"

if (( DRY_RUN )); then
    # Resolve the normal named environment without executing its activation.
    QWEN_PY="${QWEN_PY:-${CONDA_ROOT}/envs/${QWEN_ENV}/bin/python}"
else
    source "${CONDA_ROOT}/bin/activate" "$QWEN_ENV"
    QWEN_PY="${QWEN_PY:-$(command -v python)}"
    if [[ ! -d "$MODEL_PATH" ]]; then
        printf 'Model directory does not exist: %s\n' "$MODEL_PATH" >&2
        exit 1
    fi
fi

COMMAND=("$QWEN_PY" -B -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME"
    --host "$HOST" --port "$PORT"
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-prefix-caching --mamba-cache-mode align
    --allowed-local-media-path /
    --limit-mm-per-prompt "{\"image\":0,\"video\":$((MAX_MODEL_LEN / MIN_FRAME_TOKENS + 1))}"
    --mm-processor-kwargs "{\"size\":{\"shortest_edge\":$((MIN_FRAME_TOKENS * 2048)),\"longest_edge\":$((MAX_FRAME_TOKENS * 2048))}}"
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
    "$@")

if (( DRY_RUN )); then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p -- "$SERVER_LOG_DIR"
SERVER_LOG="$(mktemp "${SERVER_LOG_DIR}/$(date -u +%Y%m%d_%H%M%S)_XXXXXX.log")"
printf 'Server log: %s\n' "$SERVER_LOG"
exec > >(tee -a -- "$SERVER_LOG") 2>&1

# This import verifies the existing editable vLLM installation's video patch.
"$QWEN_PY" -B -c 'from vllm.multimodal.media.frame_sequence import load_frame_sequence'
exec "${COMMAND[@]}"
