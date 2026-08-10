# llm-serve-nvidia

Containerized LLM serving on NVIDIA GPUs using [vLLM](https://docs.vllm.ai)'s OpenAI-compatible server. All configuration lives in **`config.env`**.

**Current target: `Qwen/Qwen3.6-27B-FP8` on a single 48 GB L40S** — FP8 weights (~28 GB) + 128K context KV cache, thinking mode and tool calling enabled. `make up` is all it takes.

## 1. One-time host setup

- NVIDIA driver installed (`nvidia-smi` works)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Sanity-check GPU passthrough:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

## 2. Run

```bash
git clone <this-repo> && cd llm-serve-nvidia
make up
```

First boot pulls the vLLM image (~10 GB) and downloads ~28 GB of model weights, so expect 15–45 minutes depending on bandwidth. Weights persist in the `hf-cache` Docker volume — later restarts take only a few minutes (model load + CUDA graph capture).

Watch startup:

```bash
make logs
```

You'll see, in order: weight download → `Loading model weights` → KV-cache profiling (a line like `GPU KV cache size: ... tokens`) → CUDA graph capture → **`Application startup complete`**. The server is ready at that point.

## 3. Test

```bash
make test          # hits /v1/models and sends a chat request
```

Or manually:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "What is the capital of Nepal?"}],
    "max_tokens": 512
  }'
```

Qwen3.6 is a **thinking model**: with `REASONING_PARSER=qwen3` set (it is), the chain-of-thought arrives in the response's `reasoning_content` field and the final answer in `content`. Any OpenAI SDK works — set `base_url` to `http://<server>:8000/v1`. Tool calling is enabled (`qwen3_coder` parser, auto tool choice).

## 4. Monitor

| What | How |
|---|---|
| Container + health state | `make status` (healthcheck polls `/health` every 30 s) |
| Live logs (requests, errors) | `make logs` |
| GPU utilization / VRAM | `watch -n1 nvidia-smi` on the host |
| Health endpoint | `curl localhost:8000/health` → HTTP 200 when healthy |
| Throughput, latency, queue depth, KV-cache usage | `curl localhost:8000/metrics` (Prometheus format) |

Useful `/metrics` series: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, `vllm:time_to_first_token_seconds`. Point Prometheus/Grafana at `:8000/metrics` if you want dashboards.

vLLM also prints a throughput line in the logs every few seconds (`Avg prompt throughput ... tokens/s, Running: N reqs`), which is the quickest way to eyeball load.

The container restarts automatically (`unless-stopped`) if it crashes or the host reboots.

## 5. Reconfigure

Edit `config.env`, then:

```bash
make restart
```

Settings chosen for the L40S (all in `config.env`, annotated there):

| Setting | Value | Why |
|---|---|---|
| `MAX_MODEL_LEN` | `131072` | 128K context fits the ~15 GB KV budget left after weights; Qwen recommends ≥128K for thinking quality. Native max is 262144. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | ~44 GB for vLLM, ~2.5 GB headroom for CUDA context |
| `QUANTIZATION` | empty | Checkpoint is pre-quantized FP8; vLLM auto-detects it |
| `REASONING_PARSER` | `qwen3` | Separates `<think>` blocks into `reasoning_content` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Qwen3.6's tool-call format, with auto tool choice on |
| `VLLM_VERSION` | `v0.19.1` | Qwen3.6 requires vLLM ≥ 0.19.0 |

Optional extra speed: enable multi-token prediction (speculative decoding) by uncommenting the `EXTRA_ARGS=--speculative-config ...` line in `config.env`.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing on the host — install it, run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`, verify with the `nvidia-smi` passthrough check in section 1 |
| `GLIBC_2.38 not found (required by ...libhaishare.so)` spam | The cloud host injects a GPU-sharing shim needing a newer glibc than Ubuntu 22.04 images have — use an `-ubuntu2404` `VLLM_VERSION` tag (already the default here) |
| Crash during `Capturing CUDA graphs` with `cudaErrorStreamCaptureInvalidated` (often after a shim `cuCtxSynchronize failed (rc=900)` warning) | The host's GPU-sharing shim synchronizes the CUDA context mid-capture, killing graph capture — set `ENFORCE_EAGER=true` (already the default here) |
| CUDA out-of-memory at startup | Lower `MAX_MODEL_LEN` to `65536`, or set `KV_CACHE_DTYPE=fp8`, or lower `GPU_MEMORY_UTILIZATION` to `0.90` |
| OOM under load (not at startup) | Set `MAX_NUM_SEQS=32` to cap concurrency |
| "unrecognized model" / architecture error | Base image too old — check `VLLM_VERSION` is ≥ `v0.19.0`, then `make restart` |
| Slow first startup | Normal: image pull + 28 GB weight download. Watch progress in `make logs` |
| Container unhealthy but logs look fine | Healthcheck allows 45 min for first boot; check `curl localhost:8000/health` |
| Want raw `<think>` text in `content` instead | Clear `REASONING_PARSER` in `config.env` and `make restart` |

## Repo layout

| File | Purpose |
|---|---|
| `config.env` | **All** model + runtime settings (committed; keep secrets out of it) |
| `docker-compose.yml` | GPU wiring, ports, weight-cache volume, healthcheck |
| `Dockerfile` | Thin layer over the official `vllm/vllm-openai` image |
| `entrypoint.sh` | Turns `config.env` variables into `vllm serve` flags |
| `Makefile` | `up` / `down` / `logs` / `restart` / `status` / `test` / `shell` |
