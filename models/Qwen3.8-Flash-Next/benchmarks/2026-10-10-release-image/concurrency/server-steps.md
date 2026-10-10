# Server steps: rc8img-0937-c

## KV cache configuration (vLLM log)

- Initial free memory 31.17 GiB, reserved 1.62 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_m
- Initial free memory 31.17 GiB, reserved 1.62 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_m
- GPU KV cache size: 112,347 tokens, Maximum concurrency for 98,304 tokens per request: 1.14x

## CUDA-graph capture sizes

- [4, 8, 16, 24, 32]

- non-default args: {'host': '0.0.0.0', 'port': 18982, 'model': '/models/view', 'tokenizer': '/models/view', 'dtype': 'bfloat16', 'max_model_len': 98304, 'served_model_name': ['fn'], 'tensor_parallel_si
- Initializing a V1 LLM engine (v0.29.0) with config: model='/models/view', speculative_config=SpeculativeConfig(method='mtp', model='/models/view', num_spec_tokens=3), tokenizer='/models/view', skip_to
- builds 1 (uniform 0, mixed/prefill 1) reuses 0

- batch-size snapshots 0, per-second engine stats 29

### conc-warmup.json (T 0.7, top-p 0.95, top-k 20) level 1 round 0: wall 0.76 s, aggregate 167.5 tok/s, ok 1, TTFT ms [125]

- update gaps per request (ms, p50 each): [13.0]; median 13.0; worst p99 13.51; first tokens (s after send of the wave): [0.13]; tokens 128
- container CPU during the round (mean % of one core per process): APIServer[1314320] 13, EngineCore[1314674] 72, Worker[1314783] 171, Worker[1314891] 102, other[1314673] 0

### conc-warmup.json (T 0.7, top-p 0.95, top-k 20) level 2 round 0: wall 0.88 s, aggregate 289.7 tok/s, ok 2, TTFT ms [91, 92]

- update gaps per request (ms, p50 each): [15.5, 15.56]; median 15.530000000000001; worst p99 16.19; first tokens (s after send of the wave): [0.09, 0.09]; tokens 256
- container CPU during the round (mean % of one core per process): APIServer[1314320] 4, EngineCore[1314674] 102, Worker[1314783] 211, Worker[1314891] 118, other[1314673] 0

### conc-warmup.json (T 0.7, top-p 0.95, top-k 20) level 4 round 0: wall 2.04 s, aggregate 251.1 tok/s, ok 4, TTFT ms [61, 1053, 1054, 1054]

- update gaps per request (ms, p50 each): [20.62, 20.63, 20.68, 20.68]; median 20.655; worst p99 21.84; first tokens (s after send of the wave): [0.06, 1.05, 1.05, 1.05]; tokens 512
- container CPU during the round (mean % of one core per process): APIServer[1314320] 1, EngineCore[1314674] 100, Worker[1314783] 156, Worker[1314891] 109, other[1314673] 0

### conc-warmup.json (T 0.7, top-p 0.95, top-k 20) level 8 round 0: wall 1.58 s, aggregate 647.5 tok/s, ok 8, TTFT ms [108, 109, 110, 110, 110, 111, 111, 111]

- update gaps per request (ms, p50 each): [29.11, 29.15, 29.31, 29.32, 29.34, 29.36, 29.36, 29.4]; median 29.33; worst p99 30.76; first tokens (s after send of the wave): [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]; tokens 1024
- container CPU during the round (mean % of one core per process): APIServer[1314320] 3, EngineCore[1314674] 101, Worker[1314783] 200, Worker[1314891] 111, other[1314673] 0

### conc-ladder-t0.json (T 0.0, top-p 0.95, top-k 20) level 1 round 0: wall 0.73 s, aggregate 175.6 tok/s, ok 1, TTFT ms [77]

- update gaps per request (ms, p50 each): [12.99]; median 12.99; worst p99 13.36; first tokens (s after send of the wave): [0.08]; tokens 128
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 96, Worker[1314783] 209, Worker[1314891] 122, other[1314673] 0
- engine stats per second: running [4], waiting [0], gen tok/s [211], KV % [33.8]

### conc-ladder-t0.json (T 0.0, top-p 0.95, top-k 20) level 2 round 0: wall 0.87 s, aggregate 294.6 tok/s, ok 2, TTFT ms [54, 138]

- update gaps per request (ms, p50 each): [15.67, 15.69]; median 15.68; worst p99 16.12; first tokens (s after send of the wave): [0.06, 0.14]; tokens 256
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 95, Worker[1314783] 188, Worker[1314891] 118, other[1314673] 0
- engine stats per second: running [4], waiting [0], gen tok/s [211], KV % [33.8]

### conc-ladder-t0.json (T 0.0, top-p 0.95, top-k 20) level 4 round 0: wall 1.12 s, aggregate 456.3 tok/s, ok 4, TTFT ms [106, 107, 107, 107]

- update gaps per request (ms, p50 each): [20.3, 20.3, 20.38, 20.42]; median 20.34; worst p99 22.06; first tokens (s after send of the wave): [0.11, 0.11, 0.11, 0.11]; tokens 512
- container CPU during the round (mean % of one core per process): APIServer[1314320] 3, EngineCore[1314674] 101, Worker[1314783] 203, Worker[1314891] 114, other[1314673] 0
- engine stats per second: running [4], waiting [0], gen tok/s [211], KV % [33.8]

### conc-ladder-t0.json (T 0.0, top-p 0.95, top-k 20) level 8 round 0: wall 1.64 s, aggregate 625.3 tok/s, ok 8, TTFT ms [72, 208, 209, 209, 209, 210, 210, 210]

- update gaps per request (ms, p50 each): [29.19, 29.22, 29.26, 29.31, 29.32, 29.33, 29.35, 29.37]; median 29.314999999999998; worst p99 30.51; first tokens (s after send of the wave): [0.07, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21]; tokens 1024
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 100, Worker[1314783] 199, Worker[1314891] 113, other[1314673] 0

### conc8.json (T 0.7, top-p 0.95, top-k 20) level 8 round 0: wall 3.22 s, aggregate 635.3 tok/s, ok 8, TTFT ms [236, 351, 352, 352, 353, 353, 354, 354]

- update gaps per request (ms, p50 each): [29.33, 29.38, 29.43, 29.46, 29.47, 29.48, 29.49, 29.5]; median 29.465; worst p99 31.09; first tokens (s after send of the wave): [0.24, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 92, Worker[1314783] 195, Worker[1314891] 111, other[1314673] 0
- engine stats per second: running [0], waiting [0], gen tok/s [202], KV % [0.0]

### conc8.json (T 0.7, top-p 0.95, top-k 20) level 8 round 1: wall 3.01 s, aggregate 680.3 tok/s, ok 8, TTFT ms [109, 110, 110, 111, 111, 111, 112, 112]

- update gaps per request (ms, p50 each): [29.45, 29.49, 29.49, 29.53, 29.54, 29.58, 29.58, 29.62]; median 29.535; worst p99 31.3; first tokens (s after send of the wave): [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 100, Worker[1314783] 203, Worker[1314891] 111, other[1314673] 0

### conc8-t0.json (T 0.0, top-p 0.95, top-k 20) level 8 round 0: wall 3.2 s, aggregate 639.2 tok/s, ok 8, TTFT ms [224, 367, 367, 368, 369, 369, 369, 370]

- update gaps per request (ms, p50 each): [29.66, 29.67, 29.7, 29.73, 29.79, 29.79, 29.84, 29.85]; median 29.759999999999998; worst p99 32.0; first tokens (s after send of the wave): [0.23, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 2, EngineCore[1314674] 95, Worker[1314783] 197, Worker[1314891] 112, other[1314673] 0
- engine stats per second: running [8], waiting [0], gen tok/s [615], KV % [67.6]

### conc8-topp1.json (T 0.7, top-p 1.0, top-k 20) level 8 round 0: wall 3.33 s, aggregate 615.5 tok/s, ok 8, TTFT ms [246, 345, 470, 470, 472, 472, 473, 473]

- update gaps per request (ms, p50 each): [29.59, 29.64, 29.65, 29.65, 29.69, 29.74, 29.78, 29.81]; median 29.67; worst p99 100.23; first tokens (s after send of the wave): [0.25, 0.35, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 3, EngineCore[1314674] 93, Worker[1314783] 194, Worker[1314891] 111, other[1314673] 0
- engine stats per second: running [8], waiting [0], gen tok/s [615], KV % [67.6]

### conc8-same.json (T 0.0, top-p 0.95, top-k 20, identical rows) level 8 round 0: wall 2.81 s, aggregate 727.6 tok/s, ok 8, TTFT ms [239, 327, 429, 430, 431, 432, 432, 433]

- update gaps per request (ms, p50 each): [24.85, 24.92, 24.92, 24.92, 24.93, 24.93, 24.94, 24.95]; median 24.925; worst p99 86.57; first tokens (s after send of the wave): [0.24, 0.33, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 3, EngineCore[1314674] 92, Worker[1314783] 194, Worker[1314891] 112, other[1314673] 0

### conc8-nostream.json (T 0.7, top-p 0.95, top-k 20, non-streaming) level 8 round 0: wall 3.26 s, aggregate 628.4 tok/s, ok 8, TTFT ms [2848, 2940, 3074, 3098, 3164, 3229, 3230, 3256]
- NON-STREAMING: per-request tok/s over the whole request (TTFT included) [78.6, 79.3, 79.3, 80.9, 82.6, 83.3, 87.1, 89.9]; aggregate 628.4 tok/s; wall 3.26 s; tokens 2048
- update gaps per request (ms, p50 each): []; median -; worst p99 None; first tokens (s after send of the wave): [2.85, 2.94, 3.07, 3.1, 3.17, 3.23, 3.23, 3.26]; tokens 2048
- container CPU during the round (mean % of one core per process): APIServer[1314320] 1, EngineCore[1314674] 93, Worker[1314783] 194, Worker[1314891] 111, other[1314673] 0
- engine stats per second: running [0], waiting [0], gen tok/s [614], KV % [0.0]
