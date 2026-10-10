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

## Host conditions
Quiet host (load average median ~5 on 48 threads; no concurrent builds or uploads).

## Method
20 passes per category after 3 warm-ups, temperature 0.7 / top-p 0.95 / top-k 20, unique nonce per request (no prefix-cache hits),
Under MTP the server streams several tokens per update, so "update p99" is per stream update.
