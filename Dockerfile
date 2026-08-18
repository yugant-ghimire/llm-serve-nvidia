# sglang OpenAI-compatible server, GPU-ready.
# The base image ships with CUDA, PyTorch, and sglang preinstalled.
ARG SGLANG_VERSION=latest-cu129-runtime
FROM lmsysorg/sglang:${SGLANG_VERSION}

# The host provider's GPU-sharing shim crashes on JIT-compiled CuTe-DSL
# kernels (cudaErrorUnknown in cuda_dialect_init_library_once). Removing
# the package forces sglang onto precompiled Triton/Marlin fallbacks,
# which work under the shim. See LEARNINGS.md.
RUN pip uninstall -y nvidia-cutlass-dsl >/dev/null 2>&1 || true

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
