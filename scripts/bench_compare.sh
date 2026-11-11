#!/usr/bin/env bash
# Same benchmark, same GPU, same base model: vLLM vs Ollama.
# vLLM must be running (make deploy). It is stopped to free the GPU for Ollama.
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_MODEL="${OLLAMA_MODEL:-$(uv run python -m scripts.bench --ollama-model)}"
if [ -z "$OLLAMA_MODEL" ]; then
  echo "No Ollama equivalent known for this model; set OLLAMA_MODEL=<tag> (see ollama.com/library)"
  exit 1
fi

echo "### 1/2 vLLM"
uv run python -m scripts.bench --engine vllm

echo; echo "### Stopping vLLM to free the GPU"
uv run python -m scripts.deploy stop
sleep 5

echo; echo "### 2/2 Ollama ($OLLAMA_MODEL)"
./fallback/ollama.sh up "$OLLAMA_MODEL"
./fallback/ollama.sh pull "$OLLAMA_MODEL" | tail -3
curl -sf http://localhost:11434/api/generate -d "{\"model\": \"$OLLAMA_MODEL\", \"prompt\": \"hi\", \"stream\": false}" >/dev/null
uv run python -m scripts.bench --check-ollama-gpu "$OLLAMA_MODEL" || { ./fallback/ollama.sh down "$OLLAMA_MODEL"; exit 1; }
uv run python -m scripts.bench --engine ollama --base-url http://localhost:11434/v1 --model "$OLLAMA_MODEL"
./fallback/ollama.sh down "$OLLAMA_MODEL"

echo; echo "### Results (same prompts, 32 concurrent users)"
uv run python -m scripts.bench --compare
echo
echo "vLLM was stopped for this comparison. Restart it with: make deploy"
