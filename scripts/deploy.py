"""Stage 2: serve the model from deployment.json with vLLM.

    uv run python -m scripts.deploy                    # vLLM installed by uv (default)
    uv run python -m scripts.deploy --runtime docker   # the official vLLM Docker image
    uv run python -m scripts.deploy --dry-run          # only print the command
    uv run python -m scripts.deploy stop               # stop the server

`native` needs only the NVIDIA driver and uv: no Docker, no sudo. `docker` freezes
the whole environment (OS libraries included) and is how vLLM usually runs in
production; it needs the NVIDIA Container Toolkit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

VLLM_VERSION = "0.33.0"                       # pinned: vLLM changes fast
EXCLUDE_NEWER = "2026-11-10"                  # ...and so do its dependencies: resolve them as of this date
IMAGE = f"vllm/vllm-openai:v{VLLM_VERSION}"
CONTAINER = "llm-server"
PORT = 8000
NATIVE_DIR = Path(".vllm")                    # pid + log file for the native runtime


class DeployError(Exception):
    pass


def vllm_flags(card: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """The `vllm serve` flags, plus a reason for every one."""
    args = ["--model", card["model_id"]]
    reasons = [("--model", "the Hugging Face repo to download and serve")]

    args += ["--max-model-len", str(card["max_model_len"])]
    reasons.append(("--max-model-len", "longest prompt+answer in tokens; sizes the KV cache (from Stage 1)"))

    args += ["--gpu-memory-utilization", str(card["gpu_memory_utilization"])]
    reasons.append(("--gpu-memory-utilization",
                    "fraction of GPU memory vLLM claims up front for weights + KV cache"))

    if card.get("tensor_parallel_size", 1) > 1:
        args += ["--tensor-parallel-size", str(card["tensor_parallel_size"])]
        reasons.append(("--tensor-parallel-size", "split every layer across this many GPUs"))

    args += ["--enable-auto-tool-choice", "--tool-call-parser", card["tool_call_parser"]]
    reasons.append(("--enable-auto-tool-choice", "let the model decide when to call a tool (agents need this)"))
    reasons.append(("--tool-call-parser", "how to read this model family's tool-call format"))

    if card.get("reasoning_parser"):
        args += ["--reasoning-parser", card["reasoning_parser"]]
        reasons.append(("--reasoning-parser", "separate the model's <think> reasoning from its answer"))
    return args, reasons


def build_command(card: dict, runtime: str = "native") -> tuple[list[str], list[tuple[str, str]]]:
    """The full command for the chosen runtime, plus a reason for every flag."""
    flags, flag_reasons = vllm_flags(card)

    if runtime == "native":
        argv = ["uv", "tool", "run", "--python", "3.12", "--exclude-newer", EXCLUDE_NEWER,
                "--from", f"vllm=={VLLM_VERSION}", "vllm", "serve",
                "--host", "127.0.0.1", "--port", str(PORT)]
        reasons = [
            ("uv tool run", f"run vLLM {VLLM_VERSION} in its own cached environment (no Docker needed)"),
            ("--exclude-newer", "install vLLM's dependencies as they were on this date, so the setup can't drift"),
            ("--host", "listen on localhost only (use SSH port-forwarding from your laptop)"),
            ("--port", "where the OpenAI-compatible API is served"),
        ]
        return argv + flags, reasons + flag_reasons

    argv = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
            "-p", f"127.0.0.1:{PORT}:8000",
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-v", f"{hf_home()}:/hf-cache", "-e", "HF_HOME=/hf-cache"]
    reasons = [
        ("--gpus", "give the container access to the GPU(s)"),
        ("--ipc", "PyTorch workers share memory through /dev/shm; the default 64 MB is too small"),
        ("-p", f"expose the API on localhost:{PORT} only (use SSH port-forwarding from your laptop)"),
        ("--user", "run as you, not root, so downloaded files in your cache stay yours"),
        ("-v", "share your Hugging Face download cache, so weights download only once"),
    ]
    if os.environ.get("HF_TOKEN"):
        argv += ["-e", "HF_TOKEN"]  # pass by name: the secret never appears in the command
        reasons.append(("-e", "pass your HF_TOKEN through (needed for gated models)"))
    return argv + [IMAGE] + flags, reasons + flag_reasons


def hf_home() -> str:
    return os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def refresh_utilization(card: dict, gpus) -> dict:
    """Recompute --gpu-memory-utilization from memory free *now* (Stage 1 may be stale)."""
    from scripts.find_model import GiB, MiB, gpu_memory_utilization
    util = min(gpu_memory_utilization(g) for g in gpus)
    budget = util * min(g.total_mib for g in gpus) * MiB
    need = card["est_vram_gib"] * GiB / card.get("tensor_parallel_size", 1)
    if need > budget:
        raise DeployError(f"{card['model_id']} needs ~{need / GiB:.1f} GiB per GPU but only "
                          f"{budget / GiB:.1f} GiB is free now. Free the GPU or run `make find` again.")
    return dict(card, gpu_memory_utilization=util)


# --- server state and logs, per runtime (replaced in tests) ------------------

def _pid() -> int | None:
    try:
        return int((NATIVE_DIR / "server.pid").read_text())
    except (FileNotFoundError, ValueError):
        return None


def server_state(runtime: str) -> str:
    if runtime == "native":
        pid = _pid()
        if pid is None:
            return "missing"
        try:
            os.kill(pid, 0)
            return "running"
        except (ProcessLookupError, PermissionError):
            return "exited"
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "missing"


def server_logs(runtime: str, n: int) -> str:
    if runtime == "native":
        try:
            return "\n".join((NATIVE_DIR / "server.log").read_text(errors="replace").splitlines()[-n:])
        except FileNotFoundError:
            return ""
    r = subprocess.run(["docker", "logs", "--tail", str(n), CONTAINER], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def last_log_line(runtime: str) -> str:
    lines = [ln for ln in server_logs(runtime, 5).splitlines() if ln.strip()]
    return lines[-1][:110] if lines else ""


def server_ready(base_url: str) -> bool:
    import httpx
    try:
        return httpx.get(f"{base_url}/models", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def wait_ready(base_url: str, runtime: str, timeout_s: int = 1800, poll_s: float = 5) -> None:
    """Poll /v1/models until the server answers; fail fast if the server dies."""
    start, previous = time.time(), None
    while True:
        if server_ready(base_url):
            print(f"\n✓ Server ready after {time.time() - start:.0f}s")
            return
        if server_state(runtime) != "running":
            raise DeployError("The server stopped while starting. Last log lines:\n" + server_logs(runtime, 40))
        if time.time() - start >= timeout_s:
            raise DeployError(f"Timed out after {timeout_s}s. Last log lines:\n" + server_logs(runtime, 40))
        line = last_log_line(runtime)
        if line and line != previous:  # only show progress when something new happened
            print(f"  [{time.time() - start:4.0f}s] {line}", flush=True)
            previous = line
        time.sleep(poll_s)


def start(cmd: list[str], runtime: str) -> None:
    if runtime == "native":
        NATIVE_DIR.mkdir(exist_ok=True)
        log = open(NATIVE_DIR / "server.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (NATIVE_DIR / "server.pid").write_text(str(proc.pid))
        print(f"Server started (pid {proc.pid}). Full logs: tail -f {NATIVE_DIR}/server.log")
        return
    Path(hf_home()).mkdir(parents=True, exist_ok=True)  # else Docker creates it owned by root
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise DeployError("docker run failed:\n" + r.stderr)
    print(f"Container started. Full logs: docker logs -f {CONTAINER}")


def stop() -> None:
    """Stop whichever runtime is running."""
    if shutil_which("docker"):
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    pid = _pid()
    if pid is not None:
        try:
            os.killpg(pid, 15)  # the whole process group: vLLM spawns workers
            for _ in range(60):  # wait until it has really exited and freed the GPU
                if server_state("native") != "running":
                    break
                time.sleep(1)
        except (ProcessLookupError, PermissionError):
            pass
        (NATIVE_DIR / "server.pid").unlink(missing_ok=True)
    print("Stopped the vLLM server (if it was running).")


def shutil_which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def pretty(argv: list[str]) -> str:
    """Format the command one flag per line, like you'd write it by hand."""
    lines, cur = [], []
    for a in argv:
        if (a.startswith("-") or a == IMAGE) and cur and not cur[-1] in ("--from", "--python"):
            lines.append(" ".join(cur))
            cur = []
        cur.append(shlex.quote(a))
    lines.append(" ".join(cur))
    return " \\\n    ".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="deploy", choices=["deploy", "stop"])
    ap.add_argument("--card", default="deployment.json")
    ap.add_argument("--runtime", choices=["docker", "native"], default=os.environ.get("RUNTIME") or "native")
    ap.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    args = ap.parse_args(argv)

    if args.action == "stop":
        return stop()

    if not Path(args.card).exists():
        print(f"{args.card} not found. Run `make find` first (Stage 1).")
        sys.exit(1)
    card = json.loads(Path(args.card).read_text())
    if not args.dry_run:
        from scripts.find_model import detect_gpus
        stop()  # free the GPU first, then size against what is free now
        try:
            card = refresh_utilization(card, detect_gpus())
        except DeployError as e:
            print(f"✗ {e}")
            sys.exit(1)
    cmd, reasons = build_command(card, args.runtime)

    where = IMAGE if args.runtime == "docker" else f"vllm {VLLM_VERSION} via uv, no Docker"
    print(f"== Deploying {card['model_id']} with vLLM ({where}) ==\n")
    print("Why each flag:")
    for flag, why in reasons:
        print(f"  {flag:<26} {why}")
    if card.get("quantization"):
        print(f"  {'(no --quantization)':<26} vLLM reads '{card['quantization']}' from the model's "
              "config.json and picks its fastest kernel")
    print("\nCommand:\n  " + pretty(cmd) + "\n")
    if args.dry_run:
        return

    base_url = f"http://localhost:{PORT}/v1"
    try:
        start(cmd, args.runtime)
        print("First start downloads the weights (and vLLM itself, for native) and compiles "
              "CUDA graphs: a few minutes.")
        wait_ready(base_url, args.runtime)
    except DeployError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    print(f"""
The model is served at {base_url} (OpenAI-compatible).
Try:  curl -s {base_url}/models
Next: make use""")


if __name__ == "__main__":
    main()
