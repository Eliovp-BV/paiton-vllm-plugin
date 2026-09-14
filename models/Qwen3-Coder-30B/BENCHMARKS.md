# Local R9700 qualification — 2026-09-08

Paiton improved aggregate output throughput by **21.3% at concurrency 1**
and **70.1% at concurrency 2** over the fastest stock configuration tested.
These are results for the pinned checkpoint and short workload below. They
are not a comparison against the external Hyperloom/LinkedIn report, whose
checkpoint, quantization and workload details are not equivalent evidence.

| Engine | Concurrency | Run 1 / run 2 output tokens/s | Pooled output tokens/s | Requests/s | Mean per-request output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock vLLM | 1 | 104.53 / 104.31 | 104.42 | 0.4079 | 104.44 |
| Paiton | 1 | 126.75 / 126.61 | 126.68 | 0.4948 | 126.70 |
| Stock vLLM | 2 | 101.30 / 101.91 | 101.61 | 0.3969 | 50.82 |
| Paiton | 2 | 172.92 / 172.73 | 172.82 | 0.6751 | 86.45 |

| Engine | Concurrency | E2E mean / p95 (s) | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Stock vLLM | 1 | 2.451 / 2.458 | 85.91 / 86.69 | 9.276 / 9.298 |
| Paiton | 1 | 2.020 / 2.023 | 86.02 / 86.93 | 7.586 / 7.597 |
| Stock vLLM | 2 | 5.038 / 5.142 | 171.68 / 272.67 | 19.083 / 19.541 |
| Paiton | 2 | 2.961 / 2.965 | 155.64 / 167.59 | 11.003 / 11.055 |

Aggregate throughput divides actual output tokens by run makespan; pooled
values divide total tokens by the sum of both makespans. Per-request output
rate divides output tokens by that request's end-to-end latency, including
TTFT. TTFT ends at the first token-bearing stream event. TPOT is
`(E2E-TTFT)/(output_tokens-1)`, including final completion overhead.

Two runs of 16 completed requests per row give 32 request samples for latency
percentiles. All 128 measured requests completed without errors or token-count
mismatches. Each row has 7,524 input and 8,192 output tokens. Prompts span
eight coding tasks with two variants, 229–241 actual input tokens, and 256
output tokens per request. Identical request bodies include greedy decoding,
seed 1201, `ignore_eos=true`, and logprobs. Every performance request ends at
the forced length cap; partial code from these requests is not quality scored.

All 16 prompts were warmed before C1, with two additional warmup requests at
C2. Download, startup, compilation and warmup are excluded. No new JIT warning
was observed during the final measured runs. The same 4096 context cap,
BF16 KV pool, scheduler settings, tokenizer, weights, graph settings and GPU
profile were used for both engines. No speculative decoding is enabled.

Stock O2 graphs were qualified after an eager preview (34.48 C2 tokens/s)
and a bounded stock MoE tile tuning experiment (81.93/84.18 C2 tokens/s).
The faster stock graph preview reached 100.24 and was selected for the final
comparison. The stock image's relevant Qwen, quantization and platform code
was audited against upstream vLLM commit
`39bd959b582c85e78e7e0326d49042ce7c3c07ed`. Other attention backends, runtime
versions and quantized checkpoints were not exhaustively benchmarked.

At matched concurrency, mean request latency decreased 17.6% and 41.2%.
Within Paiton, increasing concurrency from 1 to 2 raises aggregate throughput
36.4% while increasing mean request latency from 2.020 to 2.961 seconds.
Concurrency tuning and execution-engine gains are separate comparisons.

The GPU stayed at AUTO/COMPUTE with no power, clock, fan or voltage changes.
Sampled VRAM peaked at about 20.1 GiB with full GPU weight residency. Maximum
sampled temperatures were 60 C edge, 86 C junction and 76 C memory; maximum
sampled power was 302 W. Mean core clocks were 3154/3228 MHz for stock/Paiton
C1 and 3240/3051 MHz for C2. Automatic boost was not normalized away. The C1
gain has a small margin over the 20% objective, so independent repetition is
useful before making broader claims.

Both engines passed all four sandboxed coding checks and scored 7/8 on the
small natural-EOS quality suite, without truncation. Both failed the JSON
sorting check by returning strings instead of numbers. Generated code can
differ while passing the executable checks. The compiled operation also
passed independent numerical, routing edge-case and graph-replay checks.
Its random-case differences from stock were approximately 0.36–0.425%
relative L2; the implementation is not bit exact.

Common generated prefixes averaged 129.8 tokens at C1 and 89.8 at C2, with
mean absolute selected-token logprob differences near 0.0096 and 0.0097 on
those prefixes. These measurements are not full-vocabulary logit equivalence
tests. Neither this small suite nor text agreement establishes comprehensive
quality parity. Long-context quality/performance, other GPUs, higher
concurrency, LoRA and speculation remain unqualified.

Machine-readable aggregate measurements are in [results.json](results.json).
Exact requests, streamed responses, code checks, raw telemetry, commands and
profiles are retained locally outside the repositories. Reproduction commands,
runtime versions, model/license provenance and supported requirements are in
[README.md](README.md). Raw session evidence remains local; the release bundle
contains the reusable benchmark harness and aggregate results.

The local container recipe was built and its offline launch loaded the model,
captured graphs and served the same eight quality cases (7/8, all coding checks
passed). Its terminal chat answered the deterministic arithmetic check, and
its pinned tokenizer regenerated the identical 16 benchmark requests. The
table above measures the same runtime/artifact through the experiment launch;
the packaged launch was smoke tested rather than separately rebenchmarked.

The public launcher also passed cache/container reuse, rejection of a conflicting
stock/Paiton mode, and terminal chat checks. The release enables vLLM's
`qwen3_xml` tool parser; both ordinary and streamed automatic function calls
produced the expected tool name and parsed JSON arguments in the API check.
