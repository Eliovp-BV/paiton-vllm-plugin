# Server steps: rc8img-0937-p

## KV cache configuration (vLLM log)

- Initial free memory 31.17 GiB, reserved 3.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_me
- Initial free memory 31.17 GiB, reserved 3.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_me
- GPU KV cache size: 236,090 tokens, Maximum concurrency for 200,000 tokens per request: 1.18x

## CUDA-graph capture sizes

- [1, 2, 4, 8, 16]

- Initializing a V1 LLM engine (v0.29.0) with config: model='/models/view', speculative_config=None, tokenizer='/models/view', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_re
- builds 1 (uniform 0, mixed/prefill 1) reuses 0
- {"world": 2, "rank": 0, "enabled": true, "reason": "", "abi_minor": 27, "counters": {"native": 0, "native_f32": 0, "fallback": 1, "declined_dtype": 0, "declined_shape": 1, "declined_disabled": 0, "nat

- batch-size snapshots 0, per-second engine stats 29
