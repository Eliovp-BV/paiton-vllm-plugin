# 1 October 2026: the 262K long-context mode on the 3-bit weights, full BetterBench

One R9700 (300 W), the 29 September r2 image, 3-bit W3A4 weights, BetterBench 0.6.0 **full** settings (20 passes per
category, 48 requests per concurrency level, 2 + 8 prefill runs per depth), temperature 0.7, cold prefix cache. Two
fresh server processes: `run-3bit.sh --context 262144` (262K mode: FP8 KV, prefix caching, eight requests) and
`run-3bit.sh` (65K default: 4-bit KV, no prefix caching). Reports: [262K mode](report-262k-mode.md),
[262K mode deep prefill](report-262k-deep-prefill.md), [65K default](report-65k-default.md);
[numbers.json](numbers.json).

| BetterBench row | 262K mode | 65K default |
|---|---:|---:|
| combined single-stream decode (weighted) | 174.3 tok/s | 159.3 tok/s |
| update p99 (gap between stream updates) | 29.3 ms | 27.9 ms |
| TTFT p50, batch 1 | 85 ms | 89 ms |
| aggregate @ 1 / 2 / 4 / 8 concurrent | 144.9 / 249.3 / 345.3 / 458.9 tok/s | 146.6 / 251.7 / 371.0 / 485.8 tok/s |
| prefill @ 2K / 8K / 16K / 32K / 64K | 4,179 / 4,087 / 3,915 / 3,836 / 3,549 tok/s | 3,886 / 4,162 / 4,095 / 3,941 / 3,623 tok/s |
| prefill @ 94K / 184K tokens (BetterBench depths 128K / 250K) | 3,040 / 2,395 tok/s (31.0 s / 76.7 s) | – |

Single-stream decode by category, tok/s (262K mode / 65K default): chat 148.9 / 132.4, code 202.9 / 164.2, file edit 221.3 / 198.1, json 246.4 / 196.7, math 211.6 / 208.7, prose 92.1 / 110.3, reasoning 131.5 / 125.4, summarization 142.2 / 190.5.
Every request completed (160/160 decode, 192/192 concurrency, 40/40 + 20 prefill) in
both arms.

Reading the two columns: the 262K mode is 5.5% below the 65K default at eight concurrent requests and 2% at the 64K
prefill depth (prefix-caching bookkeeping and 3,520-row chunks), and above it on weighted decode on this seed. The
65K default's weighted decode is lower than the 26 September quick run (184.4) because BetterBench's fixed seed
replays different sampled answers through the 4-bit cache, as documented on 28 September; it is not a slowdown.

For reference, a BetterBench screenshot posted for another R9700 stack at the same 262,144 setting read 118.8 tok/s
combined decode, 38.9 ms update p99, 92 ms TTFT p50, 249.1 tok/s at eight concurrent and 2,853 tok/s prefill at 64K.
