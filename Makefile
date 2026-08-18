# All configuration lives in config.env.
# --env-file makes compose read it for ${...} interpolation too.
COMPOSE = docker compose --env-file config.env

.DEFAULT_GOAL := help

.PHONY: help up down build logs restart status test shell bash

help: ## List available commands
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Build (if needed) and start the server in the background
	$(COMPOSE) up -d --build

down: ## Stop and remove the server
	$(COMPOSE) down

build: ## Rebuild the image
	$(COMPOSE) build

logs: ## Follow server logs (host-shim spam filtered out)
	@$(COMPOSE) logs -f sglang 2>&1 | grep --line-buffered -vE "HAI-9473"

restart: ## Restart (picks up config.env changes)
	$(COMPOSE) up -d --build --force-recreate

status: ## Container + health status
	$(COMPOSE) ps

shell: ## Shell inside the running container
	$(COMPOSE) exec sglang bash

bash: ## Bash inside the container; falls back to a one-off debug container if it isn't running
	@$(COMPOSE) exec sglang bash || \
	  (echo "--- sglang-server not running; starting one-off debug container ---"; \
	   $(COMPOSE) run --rm --entrypoint bash sglang)

test: ## Smoke-test the OpenAI-compatible endpoint
	@. ./config.env; \
	PORT=$${PORT:-8000}; \
	AUTH=""; [ -n "$$API_KEY" ] && AUTH="-H \"Authorization: Bearer $$API_KEY\""; \
	echo "--- /v1/models ---"; \
	eval curl -s $$AUTH http://localhost:$$PORT/v1/models; echo; \
	MODEL_ID=$${SERVED_MODEL_NAME:-$$MODEL}; \
	echo "--- /v1/chat/completions ---"; \
	eval curl -s -m 120 $$AUTH http://localhost:$$PORT/v1/chat/completions \
	  -H '"Content-Type: application/json"' \
	  -d "'{\"model\": \"$$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}], \"max_tokens\": 400}'"; echo
