"""Stage 4: how fast is the server when many people use it at once?

    uv run python -m scripts.bench --engine vllm                 # 64 requests, 32 at a time
    uv run python -m scripts.bench --engine ollama --base-url http://localhost:11434/v1
    uv run python -m scripts.bench --compare                     # table of saved results

Each request streams its answer, so we can measure:
  * time to first token (TTFT): how long a user waits before text appears
  * latency: how long until the whole answer is done
  * throughput: total output tokens per second across all users
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path("bench-results")

TOPICS = ["a lighthouse keeper", "a robot learning to paint", "the first day of college",
          "a city that floats", "a chess tournament", "a lost dog finding home",
          "a scientist on Mars", "a jazz musician in Chicago"]


@dataclass
class ReqResult:
    latency_s: float
    ttft_s: float
    output_tokens: int
    ok: bool


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (same as numpy's default)."""
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(results: list[ReqResult], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r.ok]
    tokens = sum(r.output_tokens for r in ok)
    return {
        "requests": len(ok),
        "failed": len(results) - len(ok),
        "concurrency": concurrency,
        "wall_s": round(wall_s, 2),
        "req_per_s": len(ok) / wall_s if wall_s else 0,
        "output_tok_per_s": tokens / wall_s if wall_s else 0,
        "p50_latency_s": percentile([r.latency_s for r in ok], 0.50),
        "p95_latency_s": percentile([r.latency_s for r in ok], 0.95),
        "p50_ttft_s": percentile([r.ttft_s for r in ok], 0.50),
    }


async def one_request(client, model: str, i: int, max_tokens: int) -> ReqResult:
    # "/no_think" is Qwen3's switch to skip its reasoning phase; other models ignore it.
    prompt = f"Write a short story about {TOPICS[i % len(TOPICS)]} (story #{i}). /no_think"
    start = time.perf_counter()
    ttft, tokens, chunks = None, None, 0
    try:
        stream = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.7, stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and ttft is None:
                ttft = time.perf_counter() - start
            if chunk.choices:
                chunks += 1
            if chunk.usage:
                tokens = chunk.usage.completion_tokens
        return ReqResult(time.perf_counter() - start, ttft or 0.0, tokens if tokens is not None else chunks, True)
    except Exception as e:
        print(f"  request {i} failed: {e.__class__.__name__}: {e}", file=sys.stderr)
        return ReqResult(time.perf_counter() - start, 0.0, 0, False)


async def run(base_url: str, model: str, n: int, concurrency: int, max_tokens: int) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=600, max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def limited(i):
        nonlocal done
        async with sem:
            r = await one_request(client, model, i, max_tokens)
        done += 1
        if sys.stdout.isatty():  # live counter in a terminal, quiet in logs
            print(f"\r  {done}/{n} requests done", end="", flush=True)
        return r

    # one warm-up request so we don't measure loading the model
    await one_request(client, model, 0, 16)
    start = time.perf_counter()
    results = await asyncio.gather(*(limited(i) for i in range(n)))
    wall = time.perf_counter() - start
    print()
    return summarize(results, wall, concurrency)


def compare_table(rows: list[dict]) -> str:
    base = rows[0]["output_tok_per_s"] or 1
    out = [f"{'engine':<8} {'model':<26} {'tok/s':>8} {'req/s':>6} {'p50 lat':>8} {'p95 lat':>8} "
           f"{'p50 TTFT':>9} {'vs first':>9}"]
    for r in rows:
        ratio = base / r["output_tok_per_s"] if r["output_tok_per_s"] else float("inf")
        out.append(f"{r['engine']:<8} {r['model'][-26:]:<26} {r['output_tok_per_s']:>8.0f} {r['req_per_s']:>6.2f} "
                   f"{r['p50_latency_s']:>7.1f}s {r['p95_latency_s']:>7.1f}s {r['p50_ttft_s']:>8.2f}s "
                   f"{'' if r is rows[0] else f'{ratio:.1f}x':>9}")
    return "\n".join(out)


def ollama_on_gpu(ps: dict, model: str) -> bool:
    """From Ollama's GET /api/ps: is this model loaded (at least partly) in GPU memory?"""
    return any(m.get("name", "").startswith(model) and m.get("size_vram", 0) > 0 for m in ps.get("models", []))


def ollama_equivalent(model_id: str) -> str | None:
    """Map a Hugging Face model id to the same base model in Ollama's library."""
    name = model_id.split("/")[-1].lower()
    m = re.match(r"(qwen3|qwen2\.5)-(\d+(?:\.\d+)?)b", name)
    return f"{m.group(1)}:{m.group(2)}b" if m else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="vllm", help="label for the results file")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", help="default: the first model the server lists")
    ap.add_argument("-n", "--requests", type=int, default=64)
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--compare", action="store_true", help="print a table of saved results")
    ap.add_argument("--ollama-model", action="store_true", help="print the Ollama tag matching deployment.json")
    ap.add_argument("--check-ollama-gpu", metavar="MODEL", help="exit 1 unless Ollama runs MODEL on the GPU")
    args = ap.parse_args(argv)

    if args.check_ollama_gpu:
        import httpx
        ps = httpx.get("http://localhost:11434/api/ps", timeout=5).json()
        if not ollama_on_gpu(ps, args.check_ollama_gpu):
            sys.exit(f"Ollama is running {args.check_ollama_gpu} on the CPU, not the GPU, so a comparison "
                     "would be unfair. Give Ollama GPU access (install Ollama natively, or the "
                     "NVIDIA Container Toolkit for Docker) and try again.")
        print(f"Ollama has {args.check_ollama_gpu} on the GPU.")
        return

    if args.ollama_model:
        card = json.loads(Path("deployment.json").read_text())
        print(ollama_equivalent(card["model_id"]) or "")
        return

    if args.compare:
        rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
        rows.sort(key=lambda r: r["engine"] != "vllm")
        if not rows:
            sys.exit("No results yet. Run: make bench")
        print(compare_table(rows))
        return

    from openai import OpenAI
    try:
        model = args.model or OpenAI(base_url=args.base_url, api_key="local").models.list().data[0].id
    except Exception:
        sys.exit(f"Could not reach {args.base_url}. Is the server running?")

    print(f"== Benchmark: {args.engine} ({model}) ==")
    print(f"{args.requests} requests, {args.concurrency} at a time, up to {args.max_tokens} tokens each")
    s = asyncio.run(run(args.base_url, model, args.requests, args.concurrency, args.max_tokens))
    s.update(engine=args.engine, model=model)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.engine}.json").write_text(json.dumps(s, indent=2))
    print(f"""
  throughput      {s['output_tok_per_s']:.0f} output tokens/s  ({s['req_per_s']:.2f} requests/s)
  latency         p50 {s['p50_latency_s']:.1f}s, p95 {s['p95_latency_s']:.1f}s per answer
  first token     p50 {s['p50_ttft_s']:.2f}s
  failed          {s['failed']}
Saved to {RESULTS_DIR}/{args.engine}.json""")


if __name__ == "__main__":
    main()
