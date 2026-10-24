# Benchmarks (measured on the shipped release image through this directory's launcher; BetterBench 0.6.0 standard profile through the vLLM OpenAI server, exact bf16 wire, cold prefix cache, 2x R9700)

Run of 10 Oct 2026 09:40-10:10 UTC on the release image with its baked compile caches: parts single-stream + concurrency in `--mode decode`,
prefill in `--mode prefill-long`.

## Decode mode (`--mode decode`): single stream and concurrency
Decode per category (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1, summarization 239.8 (weighted 216.3)
Concurrency ladder 1/2/4/8 x 48 requests: 204.6 / 315.1 / 449.2 / 565.1 tok/s aggregate (48/48 completed at 8).
TTFT p50 short prompts 117 ms; update p99 13.9 ms per stream update (MTP: about 2.9 tokens per update).

## Prefill mode (`--mode prefill-long`): depth sweep (W8A8 prompt rows, default prefill path)
Prefill by depth, exact wire, 2,048-token chunks (tok/s): 2,000: 7,925; 8,000: 8,321; 16,000: 8,311; 32,000: 8,232; 64,000: 8,161; 128,000: 7,928
Prefill at exactly 64,000 prompt tokens (raw completions, cold prefix cache): 8,212 tok/s full (TTFT median 7.793 s); 8,204 tok/s steady with the first 2,048-token chunk removed; exact token count verified through the server's tokenizer.
Exact-arithmetic prefill (`--mode prefill-long-nopf`), development-stack measurement of 10 Oct 2026 (not re-run on the image): 2,000: 7,575; 8,000: 7,785; 16,000: 7,762; 32,000: 7,711; 64,000: 7,607; 128,000: 7,422; exactly 64,000 tokens 7,673 / 7,666.

## Long context and caching (served, 10 Oct 2026 gates)
Needles: 40/40 at 131,072 and at 200,000 tokens. Prefix caching (opt-in `--prefix-caching`, align mode, 2,048-token blocks): 64K repeat 9.6 s -> 0.35 s TTFT, identical output.
Single-stream decode in the 200K mode (prefill-long, speculation off): 105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K).

## Native prefix-caching modes (20261024-pc-rc2, 24 Oct 2026)
BetterBench 0.6.0 standard profile through this launcher, one run per mode, unique prompts (no prefix hits); decode / TTFT p50 /
prefill @64K / 8 requests / 4 requests:
`4bit-long-cache-mtp-native` (the `--fn-worker native --prefix-caching` default, speculative decoding on) with `--native-kv-tier off`
232.6 / 104 ms / 7,270 / 487.5 / 397.6, with the RAM tier 207.3 / 107 ms / 7,223 / 482.8 / 395.5; speculative decoding off (20261023,
unchanged files): `4bit-long-cache-native` 94.2 / 110 ms / 6,918 / 313.0 / 233.8; `3bit-cache-native` 101.0 / 73 ms / 8,731 / 403.4 / 279.7;
`3bit-long-cache-native` 100.5 / 73 ms / 8,725 / 401.5 / 279.0; `3bit-long-cache-mtp-native` (speculative decoding on, RAM tier)
213.5 / 85 ms / 8,346 / 611.0 / 460.4, with `--native-kv-tier off` 250.0 / 81 ms / 8,404 / 657.7 / 505.1.
Repeated prompts (cold → hit TTFT, one request at a time): 4-bit `4bit-long-cache-mtp-native` 200K document 33.62 → 0.84 s, 90K
14.72 → 0.78 s, from the RAM tier 30.39 → 0.96 s (a hit resumes one block early with speculative decoding); `4bit-long-cache-native`
200K document 38.14 → 0.55 s, from the RAM tier 32.72 → 0.63 s; 3-bit 90K 10.53 → 0.32 s, 32K 3.70 → 0.27 s, from the RAM tier
10.48 → 0.42 s; 3-bit long 200K 24.53 → 0.42 s.
Idle sessions resumed from the RAM tier (`4bit-long-cache-native`): 0.57-0.62 s at 145K to 1.07 s at 260K ([overview](README.md#prefix-caching-on-the-native-worker)).

## Host conditions
Quiet host (load average median ~5 on 48 threads; no concurrent builds or uploads).

## Method
20 passes per category after 3 warm-ups, temperature 0.7 / top-p 0.95 / top-k 20, unique nonce per request (no prefix-cache hits),
Under MTP the server streams several tokens per update, so "update p99" is per stream update.
