# Deploy an LLM on your own GPU

Take a GPU machine from nothing to a self-hosted LLM that your code, and an
agent, can use. Each step is a single command, and each one prints what it
decided and why.

```
 Stage 1: find            Stage 2: deploy             Stage 3: use              Stage 4: bench
┌──────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐   ┌──────────────────┐
│ How much GPU     │     │ vllm serve <model>  │     │ curl                │   │ 64 requests,     │
│ memory is free?  │ ──▶ │   + flags chosen    │ ──▶ │ OpenAI Python client│   │ 32 at a time:    │
│ Which Hub model  │     │     from Stage 1    │     │ Pi agent (tools)    │   │ vLLM vs Ollama   │
│ fits? → .json    │     │ on localhost:8000   │     │                     │   │                  │
└──────────────────┘     └─────────────────────┘     └─────────────────────┘   └──────────────────┘
   make find                make deploy                 make use                  make bench
```

**What you need:** a Linux machine with an NVIDIA GPU (24 GB recommended; the
outputs below come from a 24 GB card) and a driver that supports CUDA 13.0 or
newer. There's an [AWS setup](#1-get-a-gpu-machine-on-aws) below. No GPU? See
[Laptop option: Ollama](#laptop-option-ollama).

---

## 1. Get a GPU machine (on AWS)

| Setting | Value |
|---|---|
| Instance type | `g6.xlarge` (NVIDIA L4, 24 GB) or `g5.xlarge` (NVIDIA A10G, 24 GB): roughly $0.80–1.00/hour on demand, check current pricing |
| AMI | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**: comes with the NVIDIA driver |
| Storage | 100 GB gp3 (vLLM + model weights take about 20 GB; leave room) |
| Security group | SSH (port 22) from your IP only. **Don't** open port 8000: the model server has no password. |

Connect with a port forward, so `localhost:8000` on your laptop reaches the server:

```bash
ssh -i your-key.pem -L 8000:localhost:8000 ubuntu@<instance-public-ip>
```

> ⚠️ **A GPU instance costs money every hour it runs.** Stop it when you're done
> (see [Clean up](#7-clean-up)).

## 2. Set up

```bash
git clone https://github.com/sumedhrasal/local-llm-deployment.git
cd local-llm-deployment
make setup        # installs uv, Node.js and vLLM, then checks everything
source ~/.bashrc  # pick up uv and node in this shell (or log out and back in)
```

`make setup` can be re-run safely, and skips anything already installed. Then
check the machine at any time with `make check`:

```
Checking environment...
  ✓ GPU 0: NVIDIA GeForce RTX 3090 Ti | 24564 MiB total, 24100 MiB free | compute capability 8.6
  ✓ NVIDIA driver 595.71.05 (supports CUDA 13.2; vLLM 0.33 needs >= 13.0)
  ✓ Runtime: native (vLLM installed by uv, no Docker needed; RUNTIME=docker to use Docker)
  ✓ uv 0.10.9
  ✓ Node 22.23.2
  ✓ Disk: 771 GB free

Environment looks good.
```

## 3. Stage 1: find a model that fits

```bash
make find
```

`scripts/find_model.py` asks `nvidia-smi` how much memory is **free**, lists the most
downloaded text-generation models on the Hugging Face Hub (plus quantized
variants), and filters out models it can't use:

- **gated**: needs an access request on the Hub
- **license**: not a permissive license
- **publisher**: not the model maker or a well-known quantizer
- **architecture**: no tool-call parser for this model family
- **base model**: not chat-tuned
- **no tool support**: the chat template has no tools

Then it estimates the GPU memory each model needs:

```
GPU memory  ≈  weights            (size of the .safetensors files)
            +  KV cache           (context length × bytes per token)
            +  ~2 GiB overhead    (activations, CUDA graphs)

KV cache bytes per token = 2 (K and V) × layers × KV heads × head size × 2 bytes
```

It picks the **biggest model up to 8B parameters** that fits. Bigger isn't
always better: an 8B model downloads faster, answers faster, and leaves more
memory for other users. Raise the cap with `make find MAX_B=14`, or check one
specific model with `make find MODEL=Qwen/Qwen3-14B-AWQ`.

Expected output (24 GB GPU):

```
== Your hardware ==
GPU 0: NVIDIA GeForce RTX 3090 Ti, 23.5 of 24.0 GiB free, compute capability 8.6
vLLM may use up to 90% of each GPU (free memory, capped at 90%)

== Candidate models ==
Hub: 191 popular text-generation models (top 100 overall + top 100 quantized)
Filtered out 166 (gated, license, architecture, no tool support, ...); 25 candidates left
Memory estimate = weights + KV cache (context x bytes/token) + 2 GiB overhead
Rule: the biggest model up to 8B parameters that fits (change with --max-params)
  ✓ Qwen/Qwen3-14B-AWQ: ~16.3 GiB at 32k context  (fits, but above the 8B cap)
  ✓ Qwen/Qwen2.5-14B-Instruct-AWQ: ~17.3 GiB at 32k context  (fits, but above the 8B cap)
  ✓ Qwen/Qwen3-8B-AWQ: ~12.2 GiB at 32k context  ← chosen
  ✓ Qwen/Qwen3-8B: ~19.5 GiB at 16k context
  ✓ Qwen/Qwen2.5-7B-Instruct-AWQ: ~8.9 GiB at 32k context
  ...
  ✗ Qwen/Qwen3-32B-AWQ: needs ~22.0 GiB even at 8k context → skip
  ✗ Qwen/Qwen3-14B: needs ~30.8 GiB even at 8k context → skip

== Deployment card (deployment.json) ==
  model            Qwen/Qwen3-8B-AWQ
  why              biggest model up to 8B that fits (8B parameters)
  precision        awq
  weights          5.7 GiB
  context length   32768 tokens  (KV cache 4.5 GiB at 144 KiB/token)
  GPU memory       ~12.2 GiB estimated, vLLM allowed 90%
  GPUs             1
  tool calling     --tool-call-parser hermes
  reasoning        --reasoning-parser qwen3
```

**What to notice**
- `Qwen3-8B` (16-bit) and `Qwen3-8B-AWQ` (4-bit) are the same model. The 4-bit
  version is a third of the size, which leaves room for a 32k context instead of 16k.
- The 32B models need about 18 GiB for the weights alone, so they can't fit in 24 GB.
- The results depend on *free* memory. If another program is using the GPU,
  you'll get a smaller model (or a shorter context).

## 4. Stage 2: deploy with vLLM

```bash
make deploy
```

`scripts/deploy.py` turns `deployment.json` into a `vllm serve` command,
explains every flag, starts the server in the background and waits until
`GET /v1/models` answers. vLLM runs from a pinned version (`0.33.0`) installed
by `uv`, so no Docker is needed.

```
== Deploying Qwen/Qwen3-8B-AWQ with vLLM (vllm 0.33.0 via uv, no Docker) ==

Why each flag:
  uv tool run                run vLLM 0.33.0 in its own cached environment (no Docker needed)
  --exclude-newer            install vLLM's dependencies as they were on this date, so the setup can't drift
  --host                     listen on localhost only (use SSH port-forwarding from your laptop)
  --port                     where the OpenAI-compatible API is served
  --model                    the Hugging Face repo to download and serve
  --max-model-len            longest prompt+answer in tokens; sizes the KV cache (from Stage 1)
  --gpu-memory-utilization   fraction of GPU memory vLLM claims up front for weights + KV cache
  --enable-auto-tool-choice  let the model decide when to call a tool (agents need this)
  --tool-call-parser         how to read this model family's tool-call format
  --reasoning-parser         separate the model's <think> reasoning from its answer
  (no --quantization)        vLLM reads 'awq' from the model's config.json and picks its fastest kernel

Command:
  uv tool run \
    --python 3.12 \
    --exclude-newer 2026-11-10 \
    --from vllm==0.33.0 vllm serve \
    --host 127.0.0.1 \
    --port 8000 \
    --model Qwen/Qwen3-8B-AWQ \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3

Server started (pid 132680). Full logs: tail -f .vllm/server.log
...
✓ Server ready after 75s

The model is served at http://localhost:8000/v1 (OpenAI-compatible).
```

**What to notice**
- The first start takes a few minutes: it downloads the weights, then compiles
  and captures CUDA graphs. Restarts are faster, because everything is cached.
- `nvidia-smi` now shows about 90% of the GPU in use. vLLM reserves that memory
  up front, and fills what the weights don't use with KV cache (more cache =
  more users at once).
- **The tool-calling gotcha:** without `--enable-auto-tool-choice` and the right
  `--tool-call-parser`, the model writes tool calls as plain text and agents
  silently stop working.

Stop the server with `make stop`.

### Run it in Docker instead

The official `vllm/vllm-openai` image freezes the whole environment, and it's how
vLLM usually runs in production. It needs Docker plus the NVIDIA Container Toolkit
(`RUNTIME=docker make setup` installs the toolkit):

```bash
make deploy RUNTIME=docker
```

It uses the same vLLM flags, plus `--gpus all`, `--ipc=host`, a port bound to
`127.0.0.1`, `--user` (so files in your cache stay yours), and your Hugging Face
cache mounted into the container.

## 5. Stage 3: use the model

The server speaks the **OpenAI API**. Every tool built for OpenAI works by
changing the base URL. `make use` runs all three examples.

### 3a. Raw HTTP (`make use-curl`)

```
== POST http://localhost:8000/v1/chat/completions ==
{
  "model": "Qwen/Qwen3-8B-AWQ",
  "messages": [
    {"role": "user", "content": "In two sentences: why does a GPU need memory for a KV cache?"}
  ],
  "max_tokens": 1000
}

== Response ==
(the model also returned its reasoning in a separate field: reasoning_content)

A GPU needs memory for a KV cache to store keys and values from previous tokens, enabling
efficient attention computation during sequence generation. ...

tokens: 23 in, 319 out
```

### 3b. The OpenAI Python client (`make use-python`)

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-locally")
```

That's the only change from calling OpenAI. [`use/02_openai_client.py`](use/02_openai_client.py)
shows three things: a chat completion, streaming, and **tool calling**. In the
tool-calling example, the model asks *your code* to run a function, then uses the result:

```
== 3. Tool calling ==
Model asked to call: get_weather({'city': 'Chicago'})
We ran it and got:   {'city': 'Chicago', 'temperature_c': 4, 'conditions': 'windy'}
Final answer:        The current weather in Chicago is 4°C with windy conditions.
```

### 3c. An agent: Pi (`make use-pi`)

[Pi](https://pi.dev) is a minimal coding agent. [`use/03_pi_agent/agent.ts`](use/03_pi_agent/agent.ts)
embeds it with its SDK, registers your server as a provider, and gives it a task.
The model decides which tools to call, and Pi runs them inside `use/03_pi_agent/workspace/`:

```
Task:  Read notes.md. Then write a summary of it to output/summary.md: exactly 5 bullet points, ...

[tool] read {"path":"notes.md"}
[tool] write {"path":"output/summary.md","content":"- vLLM uses PagedAttention to store the KV cache in fixed-siz...
done

Finished in 1.7s

== use/03_pi_agent/workspace/output/summary.md ==
- vLLM uses PagedAttention to store the KV cache in fixed-size blocks, reducing GPU memory waste ...
- Continuous batching keeps the GPU busy by adding new requests and removing finished ones ...
- Prefix caching reuses cached KV blocks for requests with shared prefixes ...
- Quantization techniques like AWQ/GPTQ and FP8 reduce memory usage and maintain speed ...
- CUDA graphs record and replay the sequence of GPU kernels for decoding ...
```

The agent gets only the `read` and `write` tools, not a shell: give an agent
only the access its task needs. [`sandbox.ts`](use/03_pi_agent/sandbox.ts) also
confines both tools to `workspace/`. Any path that leads outside it, whether
absolute, through `..`, or through a symlink, is refused before the disk is
touched. That covers file access only; for full isolation, run agents in a
container or a VM.

**Interactive Pi:** copy [`models.example.json`](use/03_pi_agent/models.example.json)
to `~/.pi/agent/models.json` (set the model `id` to yours), then:

```bash
cd use/03_pi_agent
npx pi --model local/Qwen/Qwen3-8B-AWQ                 # chat with the agent
npx pi --model local/Qwen/Qwen3-8B-AWQ --print "How many bullet points are in workspace/notes.md?"
```

## 6. Stage 4: benchmark

```bash
make bench
```

[`scripts/bench.py`](scripts/bench.py) sends 64 requests, 32 at a time, and
streams every answer to measure:
- **time to first token**: how long a user waits before text appears
- **latency**: how long until the whole answer is done
- **throughput**: output tokens per second, summed over all users

```
== Benchmark: vllm (Qwen/Qwen3-8B-AWQ) ==
64 requests, 32 at a time, up to 256 tokens each

  throughput      2764 output tokens/s  (10.80 requests/s)
  latency         p50 2.9s, p95 3.0s per answer
  first token     p50 0.12s
```

### vLLM vs Ollama on the same GPU

```bash
make bench-compare
```

This runs the same benchmark against vLLM and against Ollama, with the same base
model (Ollama's 4-bit GGUF build) on the same GPU. vLLM is stopped first, to free
the GPU.

```
### Results (same prompts, 32 concurrent users)
engine   model                         tok/s  req/s  p50 lat  p95 lat  p50 TTFT  vs first
vllm     Qwen/Qwen3-8B-AWQ              2824  11.03     2.9s     2.9s     0.10s
ollama   qwen3:8b                        139   0.54    58.7s    59.0s    56.89s     20.3x
```

(RTX 3090 Ti, vLLM 0.33.0, Ollama 0.31.2 with default settings. Your numbers will differ.)

**What to notice**
- One user at a time would see similar speeds from both. The gap comes from
  **concurrency**: vLLM's *continuous batching* runs all 32 requests together
  and adds new ones every decoding step. Ollama's default settings process
  requests (almost) one at a time, so most users wait in a queue (look at the
  time to first token).
- Ollama can run requests in parallel too (`OLLAMA_NUM_PARALLEL`), which narrows
  the gap but doesn't close it. *PagedAttention* is what lets vLLM fit this many
  KV caches in memory at once.

### Which engine when?

All three serve the same OpenAI-compatible API, so switching between them
doesn't change your application code.

| | llama.cpp | Ollama | vLLM |
|---|---|---|---|
| Built for | running anywhere, efficiently | developer convenience | high-throughput serving |
| Hardware | CPU, Apple Silicon, any GPU | same as llama.cpp (built on it) | NVIDIA GPUs (AMD, TPU, CPU less mature) |
| Model format | GGUF (heavily quantized) | GGUF, from its own library | Hugging Face weights, AWQ/GPTQ/FP8 |
| Many users at once | limited | limited | excellent |
| Setup | build or download a binary | one installer | Python/CUDA (`uv`) or Docker |
| Best for | laptops, edge devices, tinkering | a personal assistant, prototyping | serving many users, production, research |

## Laptop option: Ollama

No NVIDIA GPU? [Install Ollama](https://ollama.com/download) and use the same scripts:

```bash
./fallback/ollama.sh up
./fallback/ollama.sh pull qwen3:8b
export LLM_BASE_URL=http://localhost:11434/v1
make use-curl use-python use-pi
```

## 7. Clean up

```bash
make clean        # stop the server, delete generated files
```

> ⚠️ **On AWS: stop (or terminate) your instance in the EC2 console.** Logging
> out does not stop the billing. Terminating also deletes the disk; stopping
> keeps it (and you pay a little for storage).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` while the server starts | Something else is using the GPU (`nvidia-smi`). Free it and re-run `make find`, or pick a shorter context / smaller model: `make find MAX_B=4`. |
| The model answers but never calls tools | The server needs `--enable-auto-tool-choice` and the right `--tool-call-parser` (`make deploy` adds both). |
| `401` / `403` downloading a model | It's gated: accept its license on huggingface.co and `export HF_TOKEN=...`. |
| `Could not reach http://localhost:8000/v1` | Is the server running (`curl localhost:8000/v1/models`)? Check `.vllm/server.log`. On your laptop, is the SSH tunnel (`-L 8000:localhost:8000`) open? |
| Driver too old (CUDA < 13.0) | Use a current GPU AMI, or update the NVIDIA driver to 580 or newer. |
| Server takes very long to start | The first start downloads the vLLM package and the weights. Watch progress with `tail -f .vllm/server.log`. |

## Repo layout

```
Makefile                  every step is a make target (make help)
scripts/
  check_env.sh, setup.sh  environment check and one-time setup
  find_model.py           Stage 1: hardware + Hub search + memory math → deployment.json
  models_catalog.py       per-model-family parsers, trusted publishers, offline fallback list
  deploy.py               Stage 2: vllm serve (native or Docker) + health check
  bench.py                Stage 4: concurrent benchmark
  bench_compare.sh        Stage 4b: vLLM vs Ollama
use/
  01_curl.sh              Stage 3a: raw HTTP
  02_openai_client.py     Stage 3b: chat, streaming, tool calling
  03_pi_agent/            Stage 3c: a Pi agent on your model (sandbox.ts keeps it in workspace/)
fallback/ollama.sh        start Ollama, pull a model
tests/                    unit tests (make test)
```
