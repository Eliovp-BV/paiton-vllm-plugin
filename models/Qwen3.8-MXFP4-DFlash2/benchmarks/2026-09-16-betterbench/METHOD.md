# BetterBench comparison methodology

These measurements compare two complete serving stacks on one **AMD Radeon AI PRO R9700, 32 GB VRAM**, with **16 GB host RAM**, on 16 September 2026. Servers ran sequentially. The headline comparison uses the Paiton release repeat and the stronger GGZ14 repeat with matched HIP settings and reused compilation caches. All four runs remain available in [data](data/) and [reports](reports/).

## Models and serving settings

Both stacks load the same original target and drafter files, tokenizer and chat template. The [checkpoint lock](../../checkpoint.lock.json) records every file hash; [verification evidence](evidence/checkpoint-verification.json) records the reference runtime's independent checks.

| Role | Checkpoint | Revision |
|---|---|---|
| Target | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | `5233554c5fa56afda40150556b95573c2d7d29c0` |
| DFlash2 draft | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f` |

Both use tensor parallelism 1, an 8,192-token context limit, seven speculative tokens, unpadded draft batches, FP8 KV, at most eight sequences, 4,096-token prefill chunks, synchronous scheduling and disabled prefix caching. Both allocate **5 GiB to cache**. Identical checkpoint storage does not imply identical runtime arithmetic: attention implementations and draft policies differ.

Paiton uses its published image with vLLM `0.28.0+rocm723`; the GGZ14 stack uses vLLM `0.27.1` and repository revision `92eed82fcbb1cca31f7a9108c6371a0bba647ee2`. Effective graph modes are respectively `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`. [Provenance](provenance.json) pins the images, source and libr4d revision. This is a serving-stack comparison, not an isolated kernel experiment.

The stronger GGZ repeat adds the four HIP settings used by Paiton: `DEBUG_HIP_DYNAMIC_QUEUES=0`, `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1` and `HSA_ENABLE_IPC_MODE_LEGACY=1`. It also reuses compilation caches. Its improvement cannot establish any individual flag's causal effect. Some first-use framework compilation warnings remained in both repeated runs; this is not a fully warmed sustained-load qualification.

## Workload and measurements

The client is unmodified [BetterBench 0.6.0](https://github.com/GGZ14/BetterBench/tree/d00ad5ec8098c06584a88ec3468bacd37d5ed098). [Source hashes](betterbench-source-manifest.json), the [profile](profile-quick.json) and the [bounded corpus manifest](bounded-corpus-manifest.json) preserve the workload. All original prompt contents are retained; serial and concurrent output budgets are capped at **128 tokens**, with 16 tokens for prefill probes.

Each run measures **52 requests** and discards ten warmups: two serial samples in each of eight categories, eight requests at each concurrency level 1/2/4/8, and two prefill samples at each nominal depth 2,000/7,000. Concurrency uses the corpus's first eight entries: three chat, four code and one file-edit prompt. Actual prefill input counts are 1,560/1,530 and 5,211/5,257 tokens, giving medians **1,545 and 5,234**, identically on both sides.

Serial and concurrent requests use temperature 0.7, top_p 0.95, top_k 20 and seed 42. Prefill requests use temperature zero with the other fields unchanged. Request seeds override engine defaults; equal sampled settings do not guarantee identical generated sequences across implementations. Unique nonces are enabled and prefix caching is disabled.

BetterBench defines the reported metrics as follows:

- **Decode throughput:** `(completion_tokens − 1) / (last − first streamed update time)`. The weighted serial score combines category medians with weights code 0.30, reasoning 0.20, prose 0.15, JSON 0.15, file editing 0.10 and summarization 0.10. Math and chat are measured but have zero score weight.
- **End-to-end throughput:** completion tokens divided by request wall time, including TTFT; the weighted score combines category medians with the same weights.
- **Aggregate throughput:** all completed output tokens divided by the concurrency phase's wall time. This is distinct from median per-request decode speed.
- **TTFT:** request start to first content update, including HTTP and server queue time, excluding client semaphore waiting.
- **Prefill throughput:** prompt tokens divided by TTFT. This includes first-token and HTTP overhead; it is not isolated prefill-kernel timing.
- **Stream-update gaps:** gaps between streamed updates, which can contain several speculative tokens. They are not individual token latencies.

## Capacity, completeness and limits

Equal cache bytes produce different capacities: startup reports **74,430 tokens for Paiton and 25,746 for GGZ14**. These are allocator-reported figures, not a maximum-context stress test. [Allowlisted log excerpts](evidence/cache-log-excerpts.json) include the capacity lines and an initial GGZ run with three running requests and five waiting. Admission and memory efficiency contribute to the C8 aggregate/TTFT difference. Neither server was separately tuned for its maximum practical cache allocation.

All **208 measured requests** completed successfully. This establishes request completeness, not answer quality. Two separate untimed, non-thinking smoke checks produced the expected arithmetic and JSON responses on both engines; [their evidence](evidence/) is only a basic output check. Timed serial generations hit the length cap in **15/16 Paiton and 14/16 GGZ** requests in the final pair. Completed-answer quality and distributional equivalence were not evaluated.

Paiton's final result includes short-prompt TTFT regressions of 3.6–12.0% in five categories, C1/C2 median TTFT increases of 7.3%/11.0%, and C4/C8 per-request decode reductions of 3.1%/15.5%. [The complete comparison](COMPARISON.md) preserves these alongside the throughput gains. Small differences may be noise: this short, sequential screen supports descriptive comparisons, not confidence intervals, p99 conclusions or universal superiority.

This uses shared **AMD MXFP4 weights**, not the [Reddit report](https://www.reddit.com/r/LocalLLaMA/comments/1whcik4/nvfp4_qwen38_27b_amd_r9700s_running_5809_toks/)'s NVFP4 conversion and two-GPU setup. It establishes neither multi-GPU scaling nor a conclusion about the author's choice of published tests.

Published raw results preserve every non-environment value. Hostname, local endpoint, hardware inventory identifiers and the private-scope note are replaced with public metadata. [Provenance](provenance.json) records original and published hashes plus unchanged payload hashes. HTML reports are regenerated from these sanitized JSON files, with trailing whitespace normalized; [SHA256SUMS](SHA256SUMS) covers the publication package. Full private logs, compiler source and generated implementation details are not included.
