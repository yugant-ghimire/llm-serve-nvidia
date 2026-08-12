# All configuration lives in config.env.
# --env-file makes compose read it for ${...} interpolation too (port, GPU count).
COMPOSE = docker compose --env-file config.env

.PHONY: up down build logs restart status test shell bash sglang-up sglang-down sglang-logs

up: ## Build (if needed) and start the server in the background
	$(COMPOSE) up -d --build

down: ## Stop and remove the server
	$(COMPOSE) down

build: ## Rebuild the image
	$(COMPOSE) build

logs: ## Follow server logs
	$(COMPOSE) logs -f vllm

restart: ## Restart (picks up config.env changes)
	$(COMPOSE) up -d --build --force-recreate

status: ## Container + health status
	$(COMPOSE) ps

shell: ## Shell inside the running container
	$(COMPOSE) exec vllm bash

bash: ## Bash inside the container; falls back to a one-off debug container if it isn't running
	@$(COMPOSE) exec vllm bash || \
	  (echo "--- vllm-server not running; starting one-off debug container ---"; \
	   $(COMPOSE) run --rm --entrypoint bash vllm)

# ---------------------------------------------------------------
# sglang stack — the engine that works on the current GPU host.
# All settings live in sglang.env; see LEARNINGS.md for the story.
# ---------------------------------------------------------------

sglang-up: ## Start the sglang server (config: sglang.env)
	@set -a; . ./sglang.env; set +a; \
	docker rm -f $$SGLANG_CONTAINER 2>/dev/null || true; \
	docker run -d --name $$SGLANG_CONTAINER \
	  --network host --ipc=host \
	  -v $$HF_CACHE_VOLUME:/root/.cache/huggingface \
	  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
	  -e HAISHARE_PASSTHROUGH=$$HAISHARE_PASSTHROUGH \
	  --restart unless-stopped \
	  $$SGLANG_IMAGE \
	  python3 -m sglang.launch_server \
	  --model-path $$MODEL --tp $$TP_SIZE \
	  --context-length $$CONTEXT_LENGTH \
	  --mem-fraction-static $$MEM_FRACTION_STATIC \
	  --host 0.0.0.0 --port $$PORT \
	  --kv-cache-dtype $$KV_CACHE_DTYPE \
	  --mamba-ssm-dtype $$MAMBA_SSM_DTYPE \
	  --watchdog-timeout $$WATCHDOG_TIMEOUT \
	  --disable-overlap-schedule --disable-cuda-graph \
	  --mm-feature-transport cpu \
	  --reasoning-parser $$REASONING_PARSER \
	  --tool-call-parser $$TOOL_CALL_PARSER \
	  $$SGLANG_EXTRA_ARGS; \
	echo "Started $$SGLANG_CONTAINER — follow with: make sglang-logs"

sglang-down: ## Stop and remove the sglang server
	@set -a; . ./sglang.env; set +a; docker rm -f $$SGLANG_CONTAINER

sglang-logs: ## Follow sglang logs (shim warnings filtered out)
	@set -a; . ./sglang.env; set +a; \
	docker logs -f --tail 50 $$SGLANG_CONTAINER 2>&1 | grep --line-buffered -vE "HAI-9473"

test: ## Smoke-test the OpenAI-compatible endpoint
	@. ./config.env; \
	PORT=$${PORT:-8000}; \
	AUTH=""; [ -n "$$API_KEY" ] && AUTH="-H \"Authorization: Bearer $$API_KEY\""; \
	echo "--- /v1/models ---"; \
	eval curl -s $$AUTH http://localhost:$$PORT/v1/models; echo; \
	MODEL_ID=$${SERVED_MODEL_NAME:-$$MODEL}; \
	echo "--- /v1/chat/completions ---"; \
	eval curl -s $$AUTH http://localhost:$$PORT/v1/chat/completions \
	  -H '"Content-Type: application/json"' \
	  -d "'{\"model\": \"$$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}], \"max_tokens\": 50}'"; echo
