# Why sglang worked and vLLM didn't: a post-mortem

Serving `Qwen/Qwen3.6-27B-FP8` on a rented 48 GB L40S turned into a multi-day
debugging exercise. The model now runs reliably under **sglang with the host
shim in passthrough mode**, after every vLLM configuration eventually hung.
This document records what we learned, with evidence, so the next person
doesn't rediscover it the hard way.

## TL;DR

The GPU host is not a plain VM. It is a Kubernetes pod on a cloud platform
that force-injects a **GPU-sharing shim** (`libhaishare.so`, a fork of the
open-source [nvshare](https://github.com/grgalex/nvshare) project) into every
container via `/etc/ld.so.preload`. The shim interposes the CUDA driver API to
implement multi-tenant GPU sharing: UVM-managed memory, VRAM "memswap"
eviction, and a node-level scheduler lock held in tens-of-seconds time slices.

Almost every failure we hit was a collision between an inference engine and
that shim — not a bug in our Dockerfile, compose file, or model config:

| Symptom | Root cause |
|---|---|
| Crash at CUDA graph capture (`cudaErrorStreamCaptureInvalidated`) | Shim calls `cuCtxSynchronize` mid-capture |
| Requests hang forever; API healthy; GPU 0% | Engine blocked behind the shim's jammed scheduler lock |
| 25–30 min boots (healthy host: ~5 min) | Every `cuMemGetInfo` waits 30 s for the shim lock (`HAI-9473` warnings) |
| Engine saw only 19 GiB free of 44.5 | Shim scheduler pushed a halved "fair-share" VRAM budget (likely caused by our container requesting the GPU through two paths at once) |
| Pod RAM slammed its 48 GiB ceiling 103k times | Shim converts VRAM allocations to host-backed UVM memory: host RAM usage scales with VRAM claim |
| sglang: crash creating multimodal feature pool (`cudaErrorInvalidValue`) | CUDA IPC handles cannot be created on UVM/managed memory |
| sglang: crash in flashinfer CuTe-DSL rmsnorm JIT init (`cudaErrorUnknown`) | JIT library loading is incompatible with the shim's `cuGetProcAddress`/`dlsym` interposition — the shim's own strings admit this |

The fix that finally worked was the shim's own escape hatch:
**`HAISHARE_PASSTHROUGH=1`** — documented in the shim binary as
"transparent mode (no scheduler connection, no background threads, hooks pass
through)". With interposition out of the way, sglang boots in ~3 minutes and
serves reliably.

## The environment (what we discovered about the host)

- The "VM" is a **k8s pod**: overlay root filesystem, `/32` pod IP
  (`192.168.100.195`), `KUBERNETES_SERVICE_PORT` in the environment, hostname
  `i-38e719c2-...`. Public IP `45.115.219.97` is the provider's NAT gateway;
  SSH port 30991 is a gateway mapping. **Port 8000 is not reachable from the
  internet until a mapping is added in the provider dashboard** — nothing
  inside the pod can open it.
- `/etc/ld.so.preload` injects `/usr/lib/haishare/libhaishare.so` into every
  process. A container runtime hook (`haishare-runtime mutate config.json`)
  also injects GPU access + the shim into every Docker container automatically
  — which is why the provider's guide says **never** add `--gpus`,
  `deploy.resources` GPU reservations, `runtime: nvidia`, or
  `CUDA_VISIBLE_DEVICES` (double wiring corrupts CUDA state and appears to
  make their scheduler count one pod as two tenants).
- The shim is an nvshare fork: identical scheduler protocol strings
  (`REQ_LOCK` / `DROP_LOCK` / `LOCK_OK` / `SET_TQ`, `scheduler.sock`).
  nvshare's design: convert `cuMemAlloc` to `cuMemAllocManaged` (UVM) so
  tenants can oversubscribe VRAM, and serialize GPU use with a per-device
  lock held for tens of seconds per tenant.
- The shim exposes tenant-side env knobs (recovered via `strings` on the
  binary): `HAISHARE_PASSTHROUGH`, `HAISHARE_PASSTHROUGH_TEMPORAL`,
  `HAISHARE_MEMSWAP`, `HAISHARE_RAW_MEMGETINFO`, `HAISHARE_AUTO_FAIRSHARE`,
  `HAISHARE_DISABLE_MEM_WATCHER`, `HAISHARE_VRAM_BUDGET_MIB`, and more.

## Why vLLM kept failing

vLLM itself was never the bug — four separate shim interactions were:

1. **The shim explicitly targets vLLM.** Its binary contains
   `"vLLM workload detected; reporting to scheduler for auto-memswap"`.
   Setting `HAISHARE_MEMSWAP=0` was overridden within microseconds
   (`memswap: feature DISABLED` → `feature ENABLED` in the same log
   millisecond). vLLM on this host is *forced* into memswap mode; sglang is
   not auto-detected, which is half the reason it behaves better.
2. **vLLM queries GPU memory constantly** — during its memory-profiling boot
   phase and from its periodic stats loop while serving. Every one of those
   `cuMemGetInfo` calls goes through the shim's scheduler lock; when the lock
   is contended (which on this node is chronic), each call stalls 30 s
   (`HAI-9473: memswap cuMemGetInfo lock gate exceeded 30s grace`). This
   stretched 1–2 minute boots to 25–75 minutes and repeatedly wedged the
   engine in steady state: API server healthy, `/health` 200, but zero
   requests reaching the engine, GPU 0%.
   Mitigations that helped but didn't cure: `--kv-cache-memory-bytes`
   (skips the profiling phase), `--disable-log-stats` (removes the serving-
   time memory polling).
3. **CUDA graph capture is impossible under the shim** — its memswap restore
   path calls `cuCtxSynchronize` during capture, invalidating it
   (`rc=900` / `cudaErrorStreamCaptureInvalidated`). `ENFORCE_EAGER=true`
   was required (costing decode throughput).
4. **Host RAM pressure.** Because the shim makes GPU allocations host-backed
   (UVM), claiming 37.8 GiB of VRAM (`GPU_MEMORY_UTILIZATION=0.92`) also
   consumed tens of GB of the pod's 48 GiB RAM; the pod hit its cgroup memory
   ceiling >100k times during a single boot cycle. Lowering to 0.75 (33.4 GiB
   claim) plus 32K context was the calculated safe point:
   weights 28.51 GiB + ~2.35 GiB overhead + ~2.5 GiB KV.

The one vLLM boot that fully served requests worked for ~10 minutes after
startup, then wedged idle — the shim evicted our VRAM ("memswap") and its
restore deadlocked. A keep-alive pinger did not prevent later wedges: the
lock jams independently of idleness.

## Why the working setup works

The final configuration (see `sglang.env` / `make sglang-up`):

- **`HAISHARE_PASSTHROUGH=1`** — the decisive change. The shim goes fully
  transparent: no scheduler connection, no lock gates, no memswap, no UVM
  conversion. Every previously-blocked path (JIT kernel init, meminfo, IPC)
  talks to real CUDA. The shim logs it clearly on startup:
  `HAISHARE_PASSTHROUGH: interposer transparent (SINGLE-TENANT ONLY; no
  temporal sharing/memswap)`.
- **sglang instead of vLLM** — sglang boots without a long profiling phase
  (weights load in ~10 s from cache; engine end-to-end ~45 s), allocates KV
  explicitly, and isn't singled out by the shim's vLLM detector. Two
  shim-specific crashes had to be worked around even so:
  - `--mm-feature-transport cpu` — the default `cuda_ipc` transport for
    multimodal features can't create IPC handles on managed memory
    (`cudaErrorInvalidValue` in `storage._share_cuda_()`).
    (Passthrough likely also fixes this, but cpu transport is harmless
    insurance for text serving.)
  - `--disable-cuda-graph` — same graph-capture hazard as vLLM. Cheap
    insurance even in passthrough mode.
- **`--mem-fraction-static 0.75`, `--context-length 32768`** — same memory
  math as vLLM's safe point; keeps everything inside real free VRAM with
  host-RAM headroom.
- **`--kv-cache-dtype fp8_e4m3`** — halves KV memory; yields ~66K cache
  tokens. `--mamba-ssm-dtype bfloat16` halves Mamba state size, raising the
  concurrent-request cap (this hybrid Mamba/attention model caps concurrency
  by Mamba state slots — 7 concurrent requests with current settings).
- **`--watchdog-timeout 600`** + `--restart unless-stopped` — if a forward
  pass ever wedges >10 min, sglang kills itself and Docker restarts it:
  self-healing instead of silent hangs.
- **`--network host`** — Docker's default bridge has broken DNS on this pod
  (containers couldn't resolve `huggingface.co`); host networking inherits
  the pod's DNS and binds port 8000 directly.

Measured after bring-up: first token in ~2 s, ~10–11 tokens/s decode
(eager mode), correct reasoning-parser split (`reasoning_content` vs
`content`), `finish_reason: stop`.

## Caveats and open items

- **Passthrough is "SINGLE-TENANT ONLY"** (the shim's own words). We opt out
  of the provider's sharing mechanics entirely. If the provider co-schedules
  another tenant on this card, there is no arbitration — expect interference
  both ways. If your plan is supposed to be a dedicated GPU, this is exactly
  what you paid for; confirm with the provider.
- **Port 8000 still needs a provider-dashboard mapping** to be reachable
  publicly. Until then, test via SSH tunnel:
  `ssh -p 30991 -N -L 18000:localhost:8000 ubuntu@45.115.219.97`.
- **Provider ticket evidence:** chronic `HAI-9473` lock-gate warnings every
  30 s for hours; `cuCtxSynchronize failed (rc=900)` during graph capture;
  engine hangs with an idle GPU; 25–75 minute boots. All reproduce with
  plain vLLM under their default shim mode.
- The vLLM stack (`config.env` + compose) is retained and boots, but on this
  host it remains vulnerable to the scheduler-lock wedge even with the
  profiling-skip and stats-off flags. Prefer `make sglang-up` here. On a
  normal NVIDIA host (no shim), the vLLM path should work as originally
  designed — re-enable CUDA graphs and 0.92 utilization there.

## Debugging toolkit that proved useful

```bash
# Live logs without shim spam
docker logs -f --tail 50 <container> 2>&1 | grep --line-buffered -vE "HAI-9473"

# Wedged vs working: engine CPU time must climb between samples
docker exec <container> sh -c 'ps -eo pid,time,args | grep EngineCore | grep -v grep'
# (note: `ps -eo comm` truncates to 15 chars and silently misses
#  "VLLM::EngineCore" — always match on args, not comm)

# Bounded liveness probe — http=200 fast = alive; http=000 at timeout = wedged
curl -s -m 30 localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
  -o /dev/null -w "http=%{http_code} in %{time_total}s\n"

# True vs virtualized VRAM view
docker run --rm -e HAISHARE_RAW_MEMGETINFO=1 --entrypoint python3 <image> \
  -c "import torch; f,t=torch.cuda.mem_get_info(); print(f/2**30, t/2**30)"

# Shim knobs / behavior (the binary documents itself)
sudo strings /usr/lib/haishare/libhaishare.so | grep -i haishare
```
