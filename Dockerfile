# vLLM OpenAI-compatible server, GPU-ready.
# The base image ships with CUDA, PyTorch, and vLLM preinstalled.
ARG VLLM_VERSION=v0.19.1
FROM vllm/vllm-openai:${VLLM_VERSION}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
