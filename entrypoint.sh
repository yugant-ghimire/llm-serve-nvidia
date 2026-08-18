#!/usr/bin/env bash
# Builds the `sglang.launch_server` command from environment variables
# (see config.env). Only flags whose variable is set get passed, so
# sglang's own defaults apply for anything left blank.
set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
  echo "ERROR: MODEL is not set. Set it in config.env." >&2
  exit 1
fi

ARGS=(--model-path "$MODEL")

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

# Server
add_arg  --host                        "${HOST:-0.0.0.0}"
add_arg  --port                        "${PORT:-8000}"
add_arg  --served-model-name           "${SERVED_MODEL_NAME:-}"
add_arg  --api-key                     "${API_KEY:-}"

# Memory / capacity
add_arg  --tp-size                     "${TP_SIZE:-}"
add_arg  --mem-fraction-static         "${MEM_FRACTION_STATIC:-}"
add_arg  --context-length              "${CONTEXT_LENGTH:-}"
add_arg  --max-running-requests        "${MAX_RUNNING_REQUESTS:-}"
add_arg  --kv-cache-dtype              "${KV_CACHE_DTYPE:-}"

# Throughput
add_arg  --chunked-prefill-size        "${CHUNKED_PREFILL_SIZE:-}"
add_flag --enable-mixed-chunk          "${ENABLE_MIXED_CHUNK:-}"
add_arg  --num-continuous-decode-steps "${NUM_CONTINUOUS_DECODE_STEPS:-}"
add_arg  --attention-backend           "${ATTENTION_BACKEND:-}"

# CUDA graphs (decode only — prefill capture crashes under the host shim)
add_arg  --cuda-graph-max-bs           "${CUDA_GRAPH_MAX_BS:-}"
add_arg  --cuda-graph-backend-prefill  "${CUDA_GRAPH_BACKEND_PREFILL:-}"
add_arg  --cuda-graph-backend-decode   "${CUDA_GRAPH_BACKEND_DECODE:-}"

# Speculative decoding (MTP)
add_arg  --speculative-algorithm       "${SPEC_ALGORITHM:-}"
add_arg  --speculative-num-steps       "${SPEC_NUM_STEPS:-}"
add_arg  --speculative-eagle-topk      "${SPEC_EAGLE_TOPK:-}"
add_arg  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS:-}"

# Hybrid (Mamba) cache
add_arg  --mamba-ssm-dtype             "${MAMBA_SSM_DTYPE:-}"
add_arg  --mamba-full-memory-ratio     "${MAMBA_FULL_MEMORY_RATIO:-}"
add_arg  --mamba-radix-cache-strategy  "${MAMBA_RADIX_CACHE_STRATEGY:-}"

# Chat / parsers / shim workarounds
add_arg  --mm-feature-transport        "${MM_FEATURE_TRANSPORT:-}"
add_arg  --reasoning-parser            "${REASONING_PARSER:-}"
add_arg  --tool-call-parser            "${TOOL_CALL_PARSER:-}"

# Reliability
add_arg  --watchdog-timeout            "${WATCHDOG_TIMEOUT:-}"
add_flag --trust-remote-code           "${TRUST_REMOTE_CODE:-}"

# Anything not covered above, passed through verbatim
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi

echo "Starting: python3 -m sglang.launch_server ${ARGS[*]}"
exec python3 -m sglang.launch_server "${ARGS[@]}"
