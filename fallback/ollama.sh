#!/usr/bin/env bash
# Ollama: the laptop-friendly way to run a model locally. It serves the same
# OpenAI-compatible API (on port 11434), so every script in use/ works unchanged:
#
#   ./fallback/ollama.sh up              # start Ollama (uses an installed one, else Docker)
#   ./fallback/ollama.sh pull qwen3:8b   # download a model
#   export LLM_BASE_URL=http://localhost:11434/v1
#   make use-python
#   ./fallback/ollama.sh down [model]    # unload the model / stop the container
set -euo pipefail

IMAGE=ollama/ollama:0.40.11
URL=http://localhost:11434
MODEL="${2:-${OLLAMA_MODEL:-qwen3:8b}}"

running() { curl -sf "$URL/api/version" >/dev/null 2>&1; }

up() {
  if running; then
    echo "Ollama is already running at $URL (version $(curl -s "$URL/api/version" | python3 -c 'import sys,json;print(json.load(sys.stdin)["version"])'))"
  elif command -v ollama >/dev/null 2>&1; then
    echo "Starting the installed Ollama..."
    (ollama serve >/tmp/ollama-serve.log 2>&1 &)
  elif command -v docker >/dev/null 2>&1; then
    gpu=()
    docker run --rm --gpus all ubuntu:24.04 true >/dev/null 2>&1 && gpu=(--gpus all)
    echo "Starting Ollama in Docker ${gpu[*]:-(CPU only)}..."
    docker run -d --name ollama "${gpu[@]}" -p 127.0.0.1:11434:11434 -v ollama:/root/.ollama "$IMAGE" >/dev/null
  else
    echo "Install Ollama first: https://ollama.com/download  (macOS, Linux, Windows)"
    exit 1
  fi
  for _ in $(seq 30); do running && break; sleep 1; done
  running || { echo "Ollama did not start"; exit 1; }
  echo "Ollama is up at $URL. Next: ./fallback/ollama.sh pull $MODEL"
}

pull() {
  echo "Pulling $MODEL (a GGUF, 4-bit quantized build)..."
  curl -sf "$URL/api/pull" -d "{\"model\": \"$MODEL\"}" | python3 -c '
import sys, json
last = ""
for line in sys.stdin:
    s = json.loads(line).get("status", "")
    if s != last:
        print("  " + s); last = s'
  echo
  echo "Use it with:  export LLM_BASE_URL=$URL/v1"
}

down() {
  # unload the model from GPU memory (keep_alive: 0), then stop our container if we started one
  curl -sf "$URL/api/generate" -d "{\"model\": \"$MODEL\", \"keep_alive\": 0}" >/dev/null 2>&1 || true
  if command -v docker >/dev/null 2>&1 && docker ps -q -f name='^ollama$' 2>/dev/null | grep -q .; then
    docker rm -f ollama >/dev/null && echo "Stopped the Ollama container."
  else
    echo "Unloaded $MODEL from GPU memory."
  fi
}

case "${1:-}" in
  up) up ;;
  pull) pull ;;
  down) down ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
