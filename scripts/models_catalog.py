"""What we know about model families, beyond what the Hugging Face Hub tells us.

vLLM needs two extra hints to turn a model's raw text output into structured
data, and they depend on the model family:

* tool_call_parser: how the model writes a tool call (e.g. Qwen wraps JSON in
  <tool_call> tags, the "hermes" format). Without it, an agent never sees
  any tool calls.
* reasoning_parser: how a "thinking" model marks its reasoning (Qwen3 uses
  <think>...</think>). vLLM moves that text into a separate field.

Only architectures listed here are considered by find_model.py.
"""

# chat_only_if_instruct: in this family the plain-named repos are base models,
# and only "...-Instruct" ones are chat-tuned (Qwen3 names are chat models).
ARCH_PARSERS: dict[str, dict] = {
    "Qwen3ForCausalLM": {"tool_call_parser": "hermes", "reasoning_parser": "qwen3", "chat_only_if_instruct": False},
    "Qwen2ForCausalLM": {"tool_call_parser": "hermes", "reasoning_parser": None, "chat_only_if_instruct": True},
    "LlamaForCausalLM": {"tool_call_parser": "llama3_json", "reasoning_parser": None, "chat_only_if_instruct": True},
    "MistralForCausalLM": {"tool_call_parser": "mistral", "reasoning_parser": None, "chat_only_if_instruct": True},
}

# Publishers we take models from: the model makers plus well-known quantizers.
# Anyone can upload to the Hub, so popularity alone is not a trust signal.
TRUSTED_PUBLISHERS = {
    "Qwen", "meta-llama", "mistralai", "microsoft", "google", "ibm-granite",
    "hugging-quants", "RedHatAI", "neuralmagic", "casperhansen", "TheBloke",
}

# Licenses that let you use the model freely (including commercially).
PERMISSIVE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause"}

# Used when the Hub can't be reached. Numbers were read from the Hub
# (safetensors file sizes and config.json), the same way find_model.py does live.
GiB = 1024**3
FALLBACK_MODELS: list[dict] = [
    dict(model_id="Qwen/Qwen3-14B-AWQ", arch="Qwen3ForCausalLM", quant="awq", params_b=14,
         weights_bytes=int(9.29 * GiB), kv_bytes_per_token=163840, max_context=40960),
    dict(model_id="Qwen/Qwen3-8B-AWQ", arch="Qwen3ForCausalLM", quant="awq", params_b=8,
         weights_bytes=int(5.68 * GiB), kv_bytes_per_token=147456, max_context=40960),
    dict(model_id="Qwen/Qwen2.5-7B-Instruct-AWQ", arch="Qwen2ForCausalLM", quant="awq", params_b=7,
         weights_bytes=int(5.19 * GiB), kv_bytes_per_token=57344, max_context=32768),
    dict(model_id="Qwen/Qwen3-4B-AWQ", arch="Qwen3ForCausalLM", quant="awq", params_b=4,
         weights_bytes=int(2.48 * GiB), kv_bytes_per_token=147456, max_context=40960),
]
