# Local LLM deployment: find -> deploy -> use
# Run `make help` to see the steps.

# setup.sh installs uv and Node under ~/.local; make sure we find them even
# before your shell has been reloaded
export PATH := $(HOME)/.local/bin:$(HOME)/.local/node/bin:$(PATH)

PY := uv run python
export LLM_BASE_URL ?= http://localhost:8000/v1

.PHONY: help setup check find deploy use use-curl use-python use-pi bench bench-compare stop clean all test

help:  ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup:  ## One-time install of uv, Node, vLLM (+ container toolkit if RUNTIME=docker)
	./scripts/setup.sh

check:  ## Check GPU, driver, uv, Node, disk (+ Docker if RUNTIME=docker)
	./scripts/check_env.sh

find:  ## Stage 1: pick a model that fits your GPU -> deployment.json
	$(PY) -m scripts.find_model $(if $(MODEL),--model $(MODEL)) $(if $(MAX_B),--max-params $(MAX_B))

deploy:  ## Stage 2: serve the model with vLLM on :8000 (RUNTIME=docker for Docker)
	$(PY) -m scripts.deploy $(if $(RUNTIME),--runtime $(RUNTIME))

use: use-curl use-python use-pi  ## Stage 3: call the model three ways

use-curl:  ## Stage 3a: raw HTTP with curl
	./use/01_curl.sh

use-python:  ## Stage 3b: the OpenAI Python client
	$(PY) use/02_openai_client.py

use-pi:  ## Stage 3c: a Pi agent using the model with tools
	cd use/03_pi_agent && npm start

bench:  ## Stage 4: throughput under 32 concurrent requests
	$(PY) -m scripts.bench --engine vllm

bench-compare:  ## Stage 4b: same benchmark, vLLM vs Ollama on this GPU
	./scripts/bench_compare.sh

stop:  ## Stop the vLLM server
	$(PY) -m scripts.deploy stop

clean: stop  ## Stop the server and delete generated files
	rm -rf deployment.json bench-results use/03_pi_agent/workspace/output

all: check find deploy use bench  ## Run every stage in order

test:  ## Run the unit tests
	uv run pytest -q
	@if command -v node >/dev/null && [ -d use/03_pi_agent/node_modules ]; then cd use/03_pi_agent && npm test --silent; fi
