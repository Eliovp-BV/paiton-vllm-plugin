### rc8img-0937 (2026-10-10 10:10 UTC; profile standard; BetterBench 0.6.0; max-model-len 98304; server start 301 s)

Config: release image rc8 (sha256:c7e76bc0d7d9; launched through the image's launch-flashnext.py, seeded caches; default modes decode / prefill-long)

release-image validation: pass = within 2 % of the lane numbers 219.4 / 573.5 / 122 ms / 13.8 ms / 8,165

| Source | Weighted decode (tok/s) | Update p99 (ms) | TTFT p50, short prompts (ms) | Aggregate @8 (tok/s) | Prefill at 64,000 tokens (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference engine 1.2.3, published (BetterBench 0.6.0, 2x R9700; prefill over a lossy 6-bit wire) | 177.3 | 17.1 | 192 | 457.5 (48/48 ok) | 6,418 |
| ours, run `rc8img-0937` (standard profile; exact wire); release image rc8 (sha256:c7e76bc0d7d9; launched through the image's launch-flashnext.py, seeded caches; default modes decode / prefill-long) | 216.3 | 13.9 | 117 | 565.1 (48/48 ok) | 8,161 |

Prefill by depth, exact wire, 2,048-token chunks (tok/s): 2,000: 7,925, 8,000: 8,321, 16,000: 8,311, 32,000: 8,232, 64,000: 8,161, 128,000: 7,928

Decode per category (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1, summarization 239.8

Multi-seed decode probe (seeds [11, 22, 33], 4 prompts x 256 tokens, T 0.7): median 210.6 tok/s (min 196.0, max 260.7; per seed 11: 209.7, 22: 211.1, 33: 205.9); TTFT median 0.088 s

BetterBench @8 per request: TTFT p50 183 ms, max 278 ms; per-request decode tok/s min 64.8, median 86.4; wall 34.1 s for 48 requests

Server step at 8 rows (client inter-token gap p50, 256-token requests): T 0.7 / top-p 0.95 / top-k 20: 29.5 ms; T 0 greedy: 29.8 ms; T 0.7 / top-k 20 / top-p 1.0: 29.7 ms; T 0, 8 identical rows: 24.9 ms; non-streaming @8 aggregate 628.4 tok/s

Prefill at exactly 64,000 prompt tokens (raw completions, cold prefix cache): 8,212 tok/s full (TTFT median 7.793 s); 8,204 tok/s steady with the first 2,048-token chunk's 0.2418 s removed; exact token count yes

Host during the run (59 samples, 10 s): load average min 2.3 / median 5.1 / max 23.9; concurrent CPU-heavy foreign processes: ssh (testbench) peak 116 % in 2 of 59 samples, docker (testbench) peak 125 % in 1 of 59 samples, node (testbench) peak 105 % in 1 of 59 samples

Cold first wave per concurrency level (warm-up ladder before the timed passes, 128 tokens): @1: TTFT ms [125], all first tokens by 0.126 s; @2: TTFT ms [91, 92], all first tokens by 0.092 s; @4: TTFT ms [61, 1053, 1054, 1054], all first tokens by 1.055 s; @8: TTFT ms [108, 109, 110, 110, 110, 111, 111, 111], all first tokens by 0.112 s

Concurrency probe round 0 (8 x 256 tokens): ok 8, wall 3.22 s, aggregate 635.3 tok/s, max in flight 8, all first tokens by 0.355 s, earliest finish 2.719 s, TTFT ms [236, 351, 352, 352, 353, 353, 354, 354], per-request decode tok/s [88.9, 90.3, 92.5, 93.0, 94.0, 98.1, 100.2, 107.9], update gap p50 (median) 29.46 ms, p99 (max) 31.09 ms
Concurrency probe round 1 (8 x 256 tokens): ok 8, wall 3.01 s, aggregate 680.3 tok/s, max in flight 8, all first tokens by 0.113 s, earliest finish 2.535 s, TTFT ms [109, 110, 110, 111, 111, 111, 112, 112], per-request decode tok/s [88.0, 90.6, 92.7, 95.1, 95.1, 97.0, 99.1, 105.2], update gap p50 (median) 29.54 ms, p99 (max) 31.3 ms

Settings: temperature 0.7, top-p 0.95, top-k 20, passes 20 (warm-up 3), unique nonce True, concurrency [1, 2, 4, 8] x 48, prefill depths [2000, 8000, 16000, 32000, 64000, 128000] x 8. Combined-score keys seen: ['decode', 'itl_low1', 'ttft_p50', 'update_p50', 'update_p99'].

Inputs archived per run folder: profile.env (the server config file), cmd.txt (the docker command), config.txt, server.log (per-second vLLM stats and batch-size histograms), betterbench.json/.log, conc-warmup.json, conc8.json, multiseed.json, vram.json: `concurrency`, `decode`, `prefill`

