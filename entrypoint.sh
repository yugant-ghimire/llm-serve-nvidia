#!/usr/bin/env bash
# Builds the `vllm serve` command from environment variables (see .env).
# Only flags whose variable is set get passed, so vLLM's own defaults apply
# for anything you leave blank.
set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
  echo "ERROR: MODEL is not set. Set it in .env (e.g. MODEL=Qwen/Qwen2.5-7B-Instruct)." >&2
  exit 1
fi

ARGS=("$MODEL")

# add_arg FLAG VALUE — appends "--flag value" if VALUE is non-empty
add_arg() {
  if [[ -n "${2:-}" ]]; then
    ARGS+=("$1" "$2")
  fi
}

# add_flag FLAG VALUE — appends bare "--flag" if VALUE is "true"/"1"
add_flag() {
  if [[ "${2:-}" == "true" || "${2:-}" == "1" ]]; then
    ARGS+=("$1")
  fi
}

add_arg  --host                       "${HOST:-0.0.0.0}"
add_arg  --port                       "${PORT:-8000}"
add_arg  --served-model-name          "${SERVED_MODEL_NAME:-}"
add_arg  --api-key                    "${API_KEY:-}"

# Parallelism / memory
add_arg  --tensor-parallel-size       "${TENSOR_PARALLEL_SIZE:-}"
add_arg  --pipeline-parallel-size     "${PIPELINE_PARALLEL_SIZE:-}"
add_arg  --gpu-memory-utilization     "${GPU_MEMORY_UTILIZATION:-}"
add_arg  --swap-space                 "${SWAP_SPACE:-}"
add_arg  --cpu-offload-gb             "${CPU_OFFLOAD_GB:-}"

# Model / context
add_arg  --max-model-len              "${MAX_MODEL_LEN:-}"
add_arg  --max-num-seqs               "${MAX_NUM_SEQS:-}"
add_arg  --max-num-batched-tokens     "${MAX_NUM_BATCHED_TOKENS:-}"
add_arg  --dtype                      "${DTYPE:-}"
add_arg  --quantization               "${QUANTIZATION:-}"
add_arg  --kv-cache-dtype             "${KV_CACHE_DTYPE:-}"
add_arg  --tokenizer                  "${TOKENIZER:-}"
add_arg  --revision                   "${MODEL_REVISION:-}"
add_arg  --seed                       "${SEED:-}"

# Chat / tool calling
add_arg  --chat-template              "${CHAT_TEMPLATE:-}"
add_arg  --tool-call-parser           "${TOOL_CALL_PARSER:-}"
add_flag --enable-auto-tool-choice    "${ENABLE_AUTO_TOOL_CHOICE:-}"
add_arg  --reasoning-parser           "${REASONING_PARSER:-}"

# Behavior toggles
add_flag --trust-remote-code          "${TRUST_REMOTE_CODE:-}"
add_flag --enforce-eager              "${ENFORCE_EAGER:-}"
add_flag --enable-prefix-caching      "${ENABLE_PREFIX_CACHING:-}"
add_flag --disable-log-requests       "${DISABLE_LOG_REQUESTS:-}"

# Anything not covered above, passed through verbatim
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi

echo "Starting: vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
