# 10 October 2026: the release image, BetterBench 0.6.0 standard profile (the published numbers)

Two R9700, the pushed image (`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1`, digest in
`../../runtime.lock.json`), started through `launch-flashnext.py` with the baked compile caches; exact bf16 wire, cold prefix cache,
temperature 0.7 / top-p 0.95 / top-k 20, unique prompts. Three server starts: `decode/` (single-stream, 20 passes x 8 categories) and
`concurrency/` (1/2/4/8 x 48 requests + warm-up ladder, multi-seed and 8-row probes) in `--mode decode`; `prefill/` (depths 2K-128K x 8
+ the exact 64,000-token probe) in `--mode prefill-long`. `merged/` holds the combined BetterBench result, the per-request probe outputs,
the server step statistics and `validation.txt` (every line within 2 % of the development-stack run that preceded it).

| BetterBench row | value |
|---|---:|
| combined single-stream decode (weighted) | 216.3 tok/s |
| aggregate @ 1 / 2 / 4 / 8 concurrent | 204.6 / 315.1 / 449.2 / 565.1 tok/s (48/48) |
| TTFT p50, batch 1 | 117 ms |
| update p99 (gap between stream updates; MTP: ~2.9 tokens per update) | 13.9 ms |
| prefill @ 2K / 8K / 16K / 32K / 64K / 128K | 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928 tok/s |
| exactly 64,000 prompt tokens | 8,212 tok/s full, 8,204 steady (TTFT 7.79 s) |

Single-stream decode by category (tok/s): chat 202.1, code 214.3, file edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1,
summarization 239.8. [numbers.json](numbers.json) has the machine-readable rows; the raw BetterBench outputs are in the subfolders.
