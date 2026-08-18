# llm-serve-nvidia

Containerized LLM serving on NVIDIA GPUs using [sglang](https://docs.sglang.io)'s OpenAI-compatible server. All configuration lives in **`config.env`**.

**Current target: `RadixArk/Qwen3.8-27B-NVFP4` on a single 48 GB L40S** behind a cloud provider's GPU-sharing shim — NVFP4 weights (~15.5 GB), 128K context, MTP speculative decoding, decode CUDA graphs, thinking mode and tool calling enabled. Measured: **~70 tok/s per stream, ~366 tok/s aggregate at 5 concurrent users.** `make up` is all it takes.

The full debugging history of this host (the shim, its failure modes, and every workaround baked into this repo) lives in [LEARNINGS.md](LEARNINGS.md) — read it before changing infra-level settings.

## 1. One-time host setup

- NVIDIA driver installed (`nvidia-smi` works)
- Docker + Compose plugin — run `sudo bash docker-install.sh` (follows the
  GPU provider's prescribed install path)

**Provider-specific rules (this cloud):** the platform injects GPU access and
a GPU-sharing shim into every container automatically. Do **not** add
`deploy.resources` GPU blocks, `runtime: nvidia`, `--gpus`, or
`CUDA_VISIBLE_DEVICES` anywhere — they conflict with the injection. This
repo's compose file follows that rule. On a standard NVIDIA host, add a
normal GPU reservation block to `docker-compose.yml` and drop the
shim-related settings flagged in `config.env`.

## 2. Run

```bash
git clone <this-repo> && cd llm-serve-nvidia
make up
make logs
```

First boot pulls the sglang image and downloads weights to the NFS share
(`HF_CACHE_DIR`), then captures decode CUDA graphs — allow 10–30 minutes.
Ready when the log shows **`Uvicorn running on http://0.0.0.0:8000`**.

## 3. Test

```bash
make test          # hits /v1/models and sends a chat request
```

Or manually:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RadixArk/Qwen3.8-27B-NVFP4",
    "messages": [{"role": "user", "content": "What is the capital of Nepal?"}],
    "max_tokens": 1000
  }'
```

This is a **thinking model**: chain-of-thought arrives in `reasoning_content`,
the answer in `content`. Use generous `max_tokens` (1000+) — short caps get
consumed by reasoning before any visible answer appears. Tool calling is
enabled (`qwen3_coder` parser). Any OpenAI SDK works — set `base_url` to
`http://<server>:8000/v1`.

## 4. Monitor

| What | How |
|---|---|
| Container + health state | `make status` |
| Live logs (shim spam filtered) | `make logs` |
| GPU utilization / VRAM | `watch -n1 nvidia-smi` on the host |
| Health endpoint | `curl localhost:8000/health` → HTTP 200 when healthy |
| Stuck-or-working check | `docker exec sglang-server sh -c 'ps -eo pid,time,args \| grep sglang \| grep -v grep'` twice — TIME climbing = working |

If requests hang while `/health` still answers, the engine is wedged (see
LEARNINGS.md); the watchdog aborts after `WATCHDOG_TIMEOUT` seconds and
Docker restarts the container automatically.

## 5. Reconfigure

Edit `config.env`, then `make restart`. Key settings (all annotated in the file):

| Setting | Value | Why |
|---|---|---|
| `MEM_FRACTION_STATIC` | `0.88` | 0.90 starves CUDA graph capture; 0.85 wastes KV |
| `CONTEXT_LENGTH` | `131072` | Cheap on this hybrid arch (~32 KB/token KV) |
| `MAX_RUNNING_REQUESTS` | `8` | The cookbook's `1` serializes all users |
| `CUDA_GRAPH_BACKEND_PREFILL` | `disabled` | Prefill capture crashes under the shim |
| `SPEC_*` (MTP) | steps 5 / draft 6 | ~1.3–1.4× decode via speculative decoding |
| `MM_FEATURE_TRANSPORT` | `cpu` | CUDA IPC crashes on the shim's managed memory |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `GLIBC_2.38 not found (required by ...libhaishare.so)`, instant exit | Image too old for the provider's shim — keep a cu129+/Ubuntu-24.04-based `SGLANG_VERSION` |
| Crash at `Capture ... CUDA graph begin` with `cudaErrorStreamCaptureInvalidated` | The shim broke graph capture — ensure `CUDA_GRAPH_BACKEND_PREFILL=disabled`; if decode capture also fails, the node is having a bad day: retry later and log the timestamp for the provider |
| `cudaErrorUnknown` in a flashinfer path on first request | JIT kernel collision — verify the image build ran the `nvidia-cutlass-dsl` uninstall (rebuild with `make build`) |
| `cudaErrorInvalidValue` creating `MmItemMemoryPool` | `MM_FEATURE_TRANSPORT=cpu` missing |
| Boot sees far less free VRAM than expected (`Load weight begin. avail mem=...`) | Stale UVM from a killed container — wait 2–5 min and `make restart` (see LEARNINGS.md) |
| Requests hang, `/health` fine, GPU 0%, `HAI-9473` warnings every 30 s | Provider's scheduler lock jammed — watchdog will self-heal; report timestamp to provider |
| `Not enough GPU memory for hybrid ... state cache` | Lower `CONTEXT_LENGTH` or raise `MEM_FRACTION_STATIC` cautiously |

## Repo layout

| File | Purpose |
|---|---|
| `config.env` | **All** model + runtime settings (committed; keep secrets out) |
| `docker-compose.yml` | Host networking, NFS weight cache, healthcheck — **no GPU overrides** |
| `Dockerfile` | Thin layer over `lmsysorg/sglang` + shim workaround baked in |
| `entrypoint.sh` | Turns `config.env` variables into `sglang.launch_server` flags |
| `Makefile` | `up` / `down` / `logs` / `restart` / `status` / `test` / `shell` / `bash` |
| `LEARNINGS.md` | Post-mortem of the GPU-sharing shim saga; read before infra changes |
| `docker-install.sh` | Provider-prescribed Docker install |
