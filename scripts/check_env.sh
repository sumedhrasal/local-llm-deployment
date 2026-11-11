#!/usr/bin/env bash
# Check that this machine can run the workshop. Prints ✓ / ✗ per item.
# Exit code 1 if anything required is missing.
set -uo pipefail

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; FAILED=1; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
FAILED=0

echo "Checking environment..."

# 1. NVIDIA GPU + driver
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  while IFS=, read -r idx name total free cc; do
    ok "GPU $idx:$name |${total} MiB total,${free} MiB free | compute capability${cc}"
  done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
  if [ "$(printf '13.0\n%s\n' "$cuda" | sort -V | head -1)" = "13.0" ]; then
    ok "NVIDIA driver $drv (supports CUDA $cuda; vLLM 0.33 needs >= 13.0)"
  else
    bad "NVIDIA driver $drv supports only CUDA $cuda; vLLM 0.33 needs >= 13.0 (driver 580+). Use a newer GPU AMI or update the driver."
  fi
else
  bad "No NVIDIA GPU found (nvidia-smi missing or failing)."
  echo "      vLLM needs an NVIDIA GPU. On a laptop, use the Ollama path instead: ./fallback/ollama.sh up"
fi

# 2. Docker, and Docker can see the GPU (only for RUNTIME=docker)
docker_bad() { bad "$*"; }
if [ "${RUNTIME:-native}" != "docker" ]; then
  ok "Runtime: native (vLLM installed by uv, no Docker needed; RUNTIME=docker to use Docker)"
elif command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
    if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then
      ok "Docker can access the GPU (NVIDIA Container Toolkit works)"
    else
      docker_bad "Docker cannot access the GPU. Install the NVIDIA Container Toolkit: make setup, or drop RUNTIME=docker"
    fi
  else
    docker_bad "Docker is installed but you can't use it. Fix: sudo usermod -aG docker \$USER, then log out and back in"
  fi
else
  docker_bad "Docker not installed. See https://docs.docker.com/engine/install/"
fi

# 3. uv (runs the Python scripts)
if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version | awk '{print $2}')"
else bad "uv not installed. Fix: make setup"; fi

# 4. Node (only needed for the Pi agent example)
if command -v node >/dev/null 2>&1; then
  v=$(node --version | tr -d v)
  if [ "$(printf '22.19.0\n%s\n' "$v" | sort -V | head -1)" = "22.19.0" ]; then
    ok "Node $v"
  else
    warn "Node $v is older than 22.19 (needed for the Pi agent step). Fix: make setup"
  fi
else
  warn "Node not installed (needed only for the Pi agent step). Fix: make setup"
fi

# 5. Disk space for model weights + the vLLM image
free_gb=$(df -Pk "${HF_HOME:-$HOME}" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$free_gb" -ge 40 ]; then ok "Disk: ${free_gb} GB free"
else bad "Disk: only ${free_gb} GB free (need ~40 GB for the vLLM image + model weights)"; fi

echo
if [ "$FAILED" -eq 0 ]; then echo "Environment looks good."; else echo "Fix the ✗ items above, then re-run: make check"; fi
exit "$FAILED"
