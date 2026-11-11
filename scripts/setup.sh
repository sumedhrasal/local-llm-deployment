#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu GPU machine (e.g. an EC2 Deep Learning AMI).
# Safe to re-run: every step is skipped if it is already done.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE_VERSION=22.23.2

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "uv (Python package manager)"
if command -v uv >/dev/null 2>&1; then
  echo "already installed: $(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

step "Node.js (for the Pi agent example)"
if command -v node >/dev/null 2>&1 && [ "$(printf '22.19.0\n%s\n' "$(node --version | tr -d v)" | sort -V | head -1)" = "22.19.0" ]; then
  echo "already installed: node $(node --version)"
else
  arch=$(uname -m); case "$arch" in x86_64) arch=x64;; aarch64) arch=arm64;; esac
  mkdir -p "$HOME/.local/node"
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.xz" \
    | tar -xJ -C "$HOME/.local/node" --strip-components=1
  grep -q '.local/node/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.local/node/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$HOME/.local/node/bin:$PATH"
  echo "installed node $(node --version) to ~/.local/node (added to ~/.bashrc)"
fi

step "NVIDIA Container Toolkit (lets Docker use the GPU)"
if [ "${RUNTIME:-native}" != "docker" ]; then
  echo "skipped (only needed for RUNTIME=docker)"
elif docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null 2>&1; then
  echo "Docker can already see the GPU"
elif command -v apt-get >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -q && sudo apt-get install -y -q nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
else
  echo "Not an apt-based system. Install the toolkit manually:"
  echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

step "Python dependencies"
uv sync

if [ "${RUNTIME:-native}" != "docker" ]; then
  step "vLLM (downloaded once into uv's cache; takes a few minutes)"
  v=$(sed -n 's/^VLLM_VERSION = "\(.*\)".*/\1/p' scripts/deploy.py)
  d=$(sed -n 's/^EXCLUDE_NEWER = "\(.*\)".*/\1/p' scripts/deploy.py)
  uv tool run --python 3.12 --exclude-newer "$d" --from "vllm==$v" vllm --version
fi

step "Pi agent dependencies"
(cd use/03_pi_agent && npm ci --no-audit --no-fund)

step "Checking everything"
exec ./scripts/check_env.sh
