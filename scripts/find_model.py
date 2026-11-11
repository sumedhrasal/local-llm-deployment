"""Stage 1: find a model that fits your GPU.

    uv run python -m scripts.find_model            # search the Hugging Face Hub
    uv run python -m scripts.find_model --model Qwen/Qwen3-8B-AWQ   # check one model

How it decides:
  1. Ask nvidia-smi how much GPU memory is *free* right now.
  2. List popular text-generation models on the Hub (plus quantized variants).
  3. Drop models we can't or shouldn't use: gated, restrictive license,
     an architecture we have no tool-calling parser for, no tool support in
     the chat template, or FP8 on a GPU without FP8 hardware.
  4. For each remaining model, estimate GPU memory:
        weights (size of the .safetensors files)
      + KV cache (bytes per token x context length)
      + ~2 GiB overhead (activations, CUDA graphs)
     and keep the longest context (32k / 16k / 8k) that fits.
  5. Pick the biggest model (most parameters) that fits.
  6. Write deployment.json for Stage 2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

from scripts.models_catalog import ARCH_PARSERS, FALLBACK_MODELS, PERMISSIVE_LICENSES, TRUSTED_PUBLISHERS

GiB = 1024**3
MiB = 1024**2
OVERHEAD_BYTES = 2 * GiB          # activations, CUDA graphs, PyTorch allocator slack
CONTEXT_OPTIONS = [32768, 16384, 8192]
MAX_GPU_UTIL = 0.90               # never ask vLLM for more than 90% of a GPU
HUB = "https://huggingface.co"


class NoModelFits(Exception):
    pass


@dataclass
class GPU:
    index: int
    name: str
    total_mib: int
    free_mib: int
    compute_cap: float


@dataclass
class Candidate:
    model_id: str
    arch: str
    quant: str | None
    params_b: float
    weights_bytes: int
    kv_bytes_per_token: int
    max_context: int
    downloads: int = 0
    license: str | None = None


@dataclass
class Fit:
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    est_bytes: int


# --- 1. hardware --------------------------------------------------------------

def parse_nvidia_smi(text: str) -> list[GPU]:
    gpus = []
    for line in text.strip().splitlines():
        idx, name, total, free, cc = [p.strip() for p in line.split(",")]
        gpus.append(GPU(int(idx), name, int(total), int(free), float(cc)))
    return gpus


def detect_gpus() -> list[GPU]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return parse_nvidia_smi(out.stdout) if out.returncode == 0 else []


# --- 2-3. candidates from the Hub ---------------------------------------------

def params_from_name(model_id: str) -> float | None:
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", model_id.split("/")[-1])
    return float(m.group(1)) if m else None


def kv_bytes_per_token(config: dict) -> int:
    """K and V vectors, for every layer and KV head, stored in fp16 (2 bytes)."""
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads") or config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * kv_heads * head_dim * 2


def _quant_method(cfg: dict | None) -> str | None:
    q = (cfg or {}).get("quantization_config") or {}
    return q.get("quant_method")


def skip_reason(listing: dict, gpus: list[GPU]) -> str | None:
    """Why we can't use this model (None = looks usable)."""
    name = listing["id"].split("/")[-1].lower()
    cfg = listing.get("config") or {}
    arch = (cfg.get("architectures") or ["?"])[0]
    license_ = ((listing.get("cardData") or {}).get("license") or "unknown")
    template = ((cfg.get("tokenizer_config") or {}).get("chat_template")) or ""
    if isinstance(template, list):
        template = " ".join(t.get("template", "") for t in template)

    if listing["id"].split("/")[0] not in TRUSTED_PUBLISHERS:
        return "publisher not in our trusted list"
    if listing.get("gated"):
        return "gated (needs an access request on the Hub)"
    if license_ not in PERMISSIVE_LICENSES:
        return f"license '{license_}' is not in our permissive list"
    if arch not in ARCH_PARSERS:
        return f"architecture {arch} has no tool-call parser in our catalog"
    if "base" in name.split("-") or (
            ARCH_PARSERS[arch]["chat_only_if_instruct"] and "instruct" not in name):
        return "base model (not chat-tuned)"
    if "tools" not in template:
        return "chat template has no tool support"
    if _quant_method(cfg) == "fp8" and min(g.compute_cap for g in gpus) < 8.9:
        return "FP8 needs compute capability >= 8.9 (Ada/Hopper); this GPU is older"
    return None


def candidate_from_hub(listing: dict, config: dict, files: dict[str, int]) -> Candidate:
    weights = sum(size for path, size in files.items() if path.endswith(".safetensors"))
    return Candidate(
        model_id=listing["id"],
        arch=(config.get("architectures") or listing["config"]["architectures"])[0],
        quant=_quant_method(config) or _quant_method(listing.get("config")),
        params_b=params_from_name(listing["id"]) or round(weights / 2 / 1e9, 1),
        weights_bytes=weights,
        kv_bytes_per_token=kv_bytes_per_token(config),
        max_context=config.get("max_position_embeddings", 8192),
        downloads=listing.get("downloads", 0),
        license=(listing.get("cardData") or {}).get("license"),
    )


def _hub_get(client, path: str, **params):
    r = client.get(f"{HUB}{path}", params=params)
    r.raise_for_status()
    return r.json()


def fetch_candidates(gpus: list[GPU], limit: int, log=print) -> list[Candidate]:
    import httpx

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    expand = [("expand[]", f) for f in ("config", "gated", "downloads", "cardData")]
    searches = [None, "AWQ"] + (["FP8"] if min(g.compute_cap for g in gpus) >= 8.9 else [])

    listings: dict[str, dict] = {}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        for search in searches:
            params = [("pipeline_tag", "text-generation"), ("sort", "downloads"),
                      ("direction", "-1"), ("limit", str(limit))] + expand
            if search:
                params.append(("search", search))
            r = client.get(f"{HUB}/api/models", params=params)
            r.raise_for_status()
            for m in r.json():
                listings.setdefault(m["id"], m)
        log(f"Hub: {len(listings)} popular text-generation models "
            f"(top {limit} overall + top {limit} quantized)")

        cands, skipped = [], 0
        for listing in listings.values():
            if skip_reason(listing, gpus):
                skipped += 1
                continue
            try:
                tree = _hub_get(client, f"/api/models/{listing['id']}/tree/main")
                files = {f["path"]: f["size"] for f in tree if f.get("type") == "file"}
                config = _hub_get(client, f"/{listing['id']}/resolve/main/config.json")
                cands.append(candidate_from_hub(listing, config, files))
            except Exception as e:  # one broken repo shouldn't stop the search
                log(f"  ! {listing['id']}: could not read metadata ({e})")
        log(f"Filtered out {skipped} (gated, license, architecture, no tool support, ...); "
            f"{len(cands)} candidates left")
    return cands


def load_candidates(gpus: list[GPU], limit: int, log=print) -> tuple[list[Candidate], list[str]]:
    try:
        cands = fetch_candidates(gpus, limit, log)
        if cands:
            return cands, [f"Searched the Hugging Face Hub (top {limit} models)"]
        msg = "Hub returned no usable model details (rate-limited?); using the built-in fallback list"
    except Exception as e:
        msg = f"Hub unreachable ({e.__class__.__name__}: {e}); using the built-in fallback list"
    log(f"  ! {msg}")
    return [Candidate(**m) for m in FALLBACK_MODELS], [msg]


def server_running(url: str = "http://localhost:8000/v1/models") -> bool:
    """Is a model server from an earlier `make deploy` still holding GPU memory?"""
    import httpx
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def fetch_one(model_id: str, gpus: list[GPU]) -> Candidate:
    import httpx

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        listing = _hub_get(client, f"/api/models/{model_id}",
                           **{"expand[]": ["config", "gated", "downloads", "cardData"]})
        if reason := skip_reason(listing, gpus):
            print(f"  ! warning: {model_id}: {reason}")
        tree = _hub_get(client, f"/api/models/{model_id}/tree/main")
        files = {f["path"]: f["size"] for f in tree if f.get("type") == "file"}
        config = _hub_get(client, f"/{model_id}/resolve/main/config.json")
    if (listing.get("config") or {}).get("architectures", ["?"])[0] not in ARCH_PARSERS:
        raise NoModelFits(f"{model_id}: architecture not supported by this workshop's catalog")
    return candidate_from_hub(listing, config, files)


# --- 4. memory math -----------------------------------------------------------

def estimate_vram(c: Candidate, context: int) -> int:
    return c.weights_bytes + c.kv_bytes_per_token * context + OVERHEAD_BYTES


def gpu_memory_utilization(g: GPU) -> float:
    """Fraction of the GPU vLLM may claim: free memory (minus a margin), capped at 90%."""
    frac = (g.free_mib - 512) / g.total_mib
    return math.floor(min(MAX_GPU_UTIL, frac) * 100) / 100


def fit(c: Candidate, gpus: list[GPU]) -> Fit | None:
    """Longest context that fits, using 1 GPU if possible, else tensor parallelism."""
    util = min(gpu_memory_utilization(g) for g in gpus)
    budget_per_gpu = util * min(g.total_mib for g in gpus) * MiB
    for tp in (n for n in (1, 2, 4, 8) if n <= len(gpus)):
        for ctx in [x for x in CONTEXT_OPTIONS if x <= c.max_context] or [c.max_context]:
            # weights and KV cache are split across GPUs; overhead is per GPU
            per_gpu = (c.weights_bytes + c.kv_bytes_per_token * ctx) / tp + OVERHEAD_BYTES
            if per_gpu <= budget_per_gpu:
                return Fit(ctx, util, tp, int(per_gpu * tp))
    return None


# --- 5. choose ----------------------------------------------------------------

# Bigger is not always better: an 8B model downloads faster, starts faster, answers
# faster, and leaves more memory for concurrent users. Raise with --max-params 14.
DEFAULT_MAX_PARAMS_B = 8


def rank_key(cf: tuple[Candidate, Fit]):
    """Most parameters; then the longest context (most room for users); then the
    smallest weights; then the most downloads."""
    c, f = cf
    return (-c.params_b, -f.max_model_len, c.weights_bytes, -c.downloads)


def select(cands: list[Candidate], gpus: list[GPU], forced: bool = False,
           max_params_b: float = DEFAULT_MAX_PARAMS_B):
    fitting, too_big = [], []
    for c in cands:
        f = fit(c, gpus)
        if f is None:
            too_big.append(c)
        else:
            fitting.append((c, f))
    if not fitting:
        raise NoModelFits("No candidate fits in the free GPU memory." if not forced
                          else f"{cands[0].model_id} does not fit in the free GPU memory.")

    fitting.sort(key=rank_key)
    allowed = fitting if forced else [cf for cf in fitting if cf[0].params_b <= max_params_b]
    if not allowed:
        raise NoModelFits(f"No model up to {max_params_b:g}B parameters fits. Try --max-params with a larger value.")
    chosen, chosen_fit = allowed[0]

    reasons = []
    for c, f in fitting:
        note = ""
        if c is chosen:
            note = "  ← chosen"
        elif not forced and c.params_b > max_params_b:
            note = f"  (fits, but above the {max_params_b:g}B cap)"
        reasons.append(f"✓ {c.model_id}: ~{f.est_bytes / GiB:.1f} GiB at {f.max_model_len // 1024}k context"
                       + (f" on {f.tensor_parallel_size} GPUs" if f.tensor_parallel_size > 1 else "") + note)
    for c in sorted(too_big, key=lambda c: c.weights_bytes):
        need = estimate_vram(c, 8192) / GiB
        reasons.append(f"✗ {c.model_id}: needs ~{need:.1f} GiB even at 8k context → skip")
    return chosen, chosen_fit, reasons


# --- 6. deployment card -------------------------------------------------------

def make_card(c: Candidate, f: Fit, gpus: list[GPU], reasons: list[str]) -> dict:
    parsers = ARCH_PARSERS[c.arch]
    g = gpus[0]
    return {
        "model_id": c.model_id,
        "quantization": c.quant,
        "max_model_len": f.max_model_len,
        "gpu_memory_utilization": f.gpu_memory_utilization,
        "tensor_parallel_size": f.tensor_parallel_size,
        "tool_call_parser": parsers["tool_call_parser"],
        "reasoning_parser": parsers["reasoning_parser"],
        "est_vram_gib": round(f.est_bytes / GiB, 1),
        "weights_gib": round(c.weights_bytes / GiB, 2),
        "gpu": {"name": g.name, "total_mib": g.total_mib, "free_mib": g.free_mib,
                "compute_cap": g.compute_cap, "count": len(gpus)},
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="skip the search and check this Hub model id")
    ap.add_argument("--limit", type=int, default=100, help="how many popular models to look at")
    ap.add_argument("--max-params", type=float, default=DEFAULT_MAX_PARAMS_B,
                    help=f"largest model size to pick, in billions of parameters (default {DEFAULT_MAX_PARAMS_B})")
    ap.add_argument("--out", default="deployment.json")
    args = ap.parse_args(argv)

    print("== Your hardware ==")
    gpus = detect_gpus()
    if not gpus:
        print("No NVIDIA GPU found. vLLM needs one.\n"
              "On a laptop, try the Ollama path instead: ./fallback/ollama.sh up")
        sys.exit(1)
    for g in gpus:
        print(f"GPU {g.index}: {g.name}, {g.free_mib / 1024:.1f} of {g.total_mib / 1024:.1f} GiB free, "
              f"compute capability {g.compute_cap}")
    if server_running():
        print("\nA model server is already running on localhost:8000 and holding GPU memory,\n"
              "so the free-memory numbers above are not what a new deployment would get.\n"
              "Run `make stop` first, then `make find` again.")
        sys.exit(1)
    print(f"vLLM may use up to {gpu_memory_utilization(gpus[0]):.0%} of each GPU "
          f"(free memory, capped at {MAX_GPU_UTIL:.0%})\n")

    print("== Candidate models ==")
    try:
        if args.model:
            cands, notes = [fetch_one(args.model, gpus)], [f"Model chosen by you: {args.model}"]
        else:
            cands, notes = load_candidates(gpus, args.limit)
        print("Memory estimate = weights + KV cache (context x bytes/token) + 2 GiB overhead")
        if not args.model:
            print(f"Rule: the biggest model up to {args.max_params:g}B parameters that fits "
                  "(change with --max-params)")
        chosen, f, reasons = select(cands, gpus, forced=bool(args.model), max_params_b=args.max_params)
    except NoModelFits as e:
        print(f"\n✗ {e}\nFree up GPU memory (check nvidia-smi) or try a smaller model with --model.")
        sys.exit(1)
    except Exception as e:  # e.g. httpx errors when checking one model with --model
        print(f"\n✗ Could not read {args.model or 'models'} from the Hugging Face Hub "
              f"({e.__class__.__name__}: {e}). Check the model id and your network.")
        sys.exit(1)
    ok = [r for r in reasons if r.startswith("✓")]
    i = next(n for n, r in enumerate(ok) if "← chosen" in r) if not args.model else 0
    shown = ok[max(0, i - 3): i + 5]  # a few bigger ones, the choice, a few smaller ones
    if ok.index(shown[0]) > 0:
        print(f"  ... {ok.index(shown[0])} bigger models that also fit")
    for r in shown:
        print("  " + r)
    if len(ok) > ok.index(shown[-1]) + 1:
        print(f"  ... and {len(ok) - ok.index(shown[-1]) - 1} smaller models that also fit")
    for r in reasons:
        if not r.startswith("✓"):
            print("  " + r)

    card = make_card(chosen, f, gpus, notes + reasons)
    with open(args.out, "w") as fh:
        json.dump(card, fh, indent=2)

    kv_gib = chosen.kv_bytes_per_token * f.max_model_len / GiB
    print(f"""
== Deployment card ({args.out}) ==
  model            {chosen.model_id}
  why              {"your choice" if args.model else f"biggest model up to {args.max_params:g}B that fits"} ({chosen.params_b:g}B parameters)
  precision        {chosen.quant or 'bf16 (full precision)'}
  weights          {chosen.weights_bytes / GiB:.1f} GiB
  context length   {f.max_model_len} tokens  (KV cache {kv_gib:.1f} GiB at {chosen.kv_bytes_per_token / 1024:.0f} KiB/token)
  GPU memory       ~{f.est_bytes / GiB:.1f} GiB estimated, vLLM allowed {f.gpu_memory_utilization:.0%}
  GPUs             {f.tensor_parallel_size}
  tool calling     --tool-call-parser {card['tool_call_parser']}
  reasoning        {('--reasoning-parser ' + card['reasoning_parser']) if card['reasoning_parser'] else 'n/a'}

Next: make deploy""")


if __name__ == "__main__":
    main()
