# Notes: how vLLM serves models fast

- **PagedAttention.** The KV cache (the attention keys and values of every token
  seen so far) is stored in fixed-size blocks, like pages of virtual memory.
  Requests no longer reserve one big contiguous chunk for their maximum length,
  so far less GPU memory is wasted and more requests fit at once.
- **Continuous batching.** Instead of waiting for a whole batch to finish, the
  scheduler adds new requests and removes finished ones after every decoding
  step. The GPU stays busy even when answers have very different lengths.
- **Prefix caching.** Requests that share a prefix (the same system prompt, the
  same document) reuse its cached KV blocks instead of recomputing them.
- **Quantization.** AWQ/GPTQ store weights in 4 bits, FP8 in 8 bits. Smaller
  weights leave more memory for the KV cache, and fast kernels (e.g. Marlin)
  keep the speed close to 16-bit.
- **CUDA graphs.** The fixed sequence of GPU kernels for a decoding step is
  recorded once and replayed, removing per-step launch overhead.
- **OpenAI-compatible server.** Any OpenAI client works by changing `base_url`.
  The trade-off: vLLM shines with many concurrent users on NVIDIA GPUs; for a
  single user on a laptop, llama.cpp or Ollama is simpler.
