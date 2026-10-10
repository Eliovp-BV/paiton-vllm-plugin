# Server steps: rc8img-0937-a

## KV cache configuration (vLLM log)

- Initial free memory 31.17 GiB, reserved 1.62 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_m
- Initial free memory 31.17 GiB, reserved 1.62 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_m
- GPU KV cache size: 112,347 tokens, Maximum concurrency for 98,304 tokens per request: 1.14x

## CUDA-graph capture sizes

- [4, 8, 16, 24, 32]

- non-default args: {'host': '0.0.0.0', 'port': 18982, 'model': '/models/view', 'tokenizer': '/models/view', 'dtype': 'bfloat16', 'max_model_len': 98304, 'served_model_name': ['fn'], 'tensor_parallel_si
- Initializing a V1 LLM engine (v0.29.0) with config: model='/models/view', speculative_config=SpeculativeConfig(method='mtp', model='/models/view', num_spec_tokens=3), tokenizer='/models/view', skip_to
- builds 1 (uniform 0, mixed/prefill 1) reuses 0

- batch-size snapshots 0, per-second engine stats 36
