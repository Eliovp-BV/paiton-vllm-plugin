# Local qualification

Paiton completed the primary scenario in **2.214 s median** (32 requests,
0.00686 s standard deviation). Compared with the fastest earlier stock vLLM
reference of **4.819 s**, this is **54.0% lower end-to-end latency (2.176×)**.
The same-image stock repeat was slower, so its larger gain is not the headline.
The original-runtime stock recheck measured 6.265 s median across 32 requests
(SD 0.085 s); it also passed 18/20 quality tasks and all ten API checks.

## Corrected release-image matrix

Both columns below use the identical corrected runtime image and pinned original
MXFP4 checkpoint. There were no request errors or token-count mismatches.

| Input / output | Concurrency | N per mode | Stock E2E median ± SD | Paiton E2E median ± SD | Reduction |
|---|---:|---:|---:|---:|---:|
| 512 / 256 | 1 | 32 | 6.110 ± 0.310 s | 2.214 ± 0.007 s | 63.8% |
| 512 / 256 | 2 | 16 | 8.100 ± 0.105 s | 3.078 ± 0.033 s | 62.0% |
| 128 / 128 | 1 | 8 | 3.095 ± 0.003 s | 1.112 ± 0.002 s | 64.1% |
| 2048 / 128 | 1 | 8 | 3.344 ± 0.008 s | 1.356 ± 0.002 s | 59.5% |
| 6144 / 128 | 1 | 8 | 4.001 ± 0.016 s | 2.007 ± 0.002 s | 49.8% |
| 512 / 1024 | 1 | 4 | 24.264 ± 0.293 s | 8.660 ± 0.004 s | 64.3% |

The corrected-image stock primary was 6.110 s, versus the earlier 4.819 s
reference. The slower stock timing also reproduced on the original developer runtime.
Its cause is not established;
using 6.110 s as the only baseline would overstate the practical compiler gain.
The independent llama.cpp original-MXFP4 screen measured 2.663 s (four requests).
It is a separate runtime screen, not a statistically established Paiton advantage.

| Scenario / mode | TTFT median | TPOT median | Aggregate output tokens/s | Requests/s |
|---|---:|---:|---:|---:|
| p512-o256-c1 / stock | 83.9 ms | 23.63 ms | 42.46 | 0.166 |
| p512-o256-c1 / paiton | 83.9 ms | 8.34 ms | 115.61 | 0.452 |
| p512-o256-c2 / stock | 152.8 ms | 31.11 ms | 63.36 | 0.248 |
| p512-o256-c2 / paiton | 104.3 ms | 11.67 ms | 166.14 | 0.649 |
| p2048-o128-c1 / stock | 295.0 ms | 24.01 ms | 38.24 | 0.299 |
| p2048-o128-c1 / paiton | 278.9 ms | 8.48 ms | 94.35 | 0.737 |
| p6144-o128-c1 / stock | 964.0 ms | 23.92 ms | 31.98 | 0.250 |
| p6144-o128-c1 / paiton | 906.1 ms | 8.67 ms | 63.76 | 0.498 |
| p512-o1024-c1 / stock | 84.1 ms | 23.64 ms | 42.04 | 0.041 |
| p512-o1024-c1 / paiton | 80.9 ms | 8.38 ms | 118.22 | 0.115 |

All output rates above include reasoning and control tokens. The raw summary
separates every channel, post-stop continuation and final-answer token count.

## Natural-stop chat

Four measured requests per prompt and mode after one warmup; identical requests.

| Prompt | Stock first visible answer / E2E | Paiton first visible answer / E2E | Completion tokens stock / Paiton |
|---|---:|---:|---:|
| Binary search | 0.376 / 1.690 s | 0.166 / 0.635 s | 70 / 70 |
| 17 + 25 | 0.370 / 0.393 s | 0.167 / 0.175 s | 15 / 15 |
| Palindrome function | 0.362 / 2.098 s | 0.166 / 0.785 s | 88 / 88 |

All 24 measured streams completed with a visible answer. Summed reasoning, final,
other, control and post-stop counts exactly match reported completion usage.

## Engine prefill and decode timings

These are per-request engine histogram means, not sums of GPU kernel durations.

| Shape / concurrency | Stock prefill / decode | Paiton prefill / decode |
|---|---:|---:|
| p512-o256-c1 | 0.080 / 5.946 s | 0.078 / 2.131 s |
| p512-o256-c2 | 0.125 / 7.918 s | 0.097 / 2.964 s |
| p2048-o128-c1 | 0.291 / 3.053 s | 0.275 / 1.077 s |
| p6144-o128-c1 | 0.957 / 3.041 s | 0.898 / 1.104 s |

Bulk prefill retains stock experts; the measured serving gain is concentrated in
one/two-token expert decode. Routing and attention remain stock.

## Hardware and memory

One Radeon AI PRO R9700, gfx1201, 64 compute units / 32 WGPs, 32 GB VRAM;
Intel Core i5-8400, six logical CPUs; Linux 6.17.0-1028-oem; 15.52 GiB RAM
and 4 GiB swap. No clock, power, voltage or cooling controls were changed.
During this matrix, sampled peak junction temperatures were 89°C stock and
86°C Paiton. Telemetry retains actual clocks and memory usage for each run.

Peak driver VRAM was 17.066 / 16.965 GiB stock/Paiton. Torch peak allocated
memory was 16.283 / 16.285 GiB; peak reserved was 16.43 GiB in both modes.
Whole-host unavailable RAM peaked at 5.279 / 5.159 GiB and whole-host swap
usage at 1.950 / 1.977 GiB during these timed runs. These are whole-host
observations, not isolated process RSS. Earlier qualification used more swap.
Fresh compilation-cache startup with already cached model weights took 154.5 /
142.3 seconds (one observation each). Initial model download is additional.

## Method

Primary scenario declared before tuning: median warm end-to-end request latency,
512 exact input tokens / 256 forced output tokens, concurrency one. Objective:
at least 10% lower latency, aim 20%, versus the fastest stable matched stock vLLM
configuration qualified on this R9700. Final primary sampling uses two runs of
16 requests after two warmups. Concurrency two uses two runs of eight requests;
secondary shapes use eight requests, or four for sustained 1,024-token output.
Warmups use the same concurrency as the measured workload. The length-controlled
requests ask for a hash-table explanation and Python example, padded with neutral
reference notes. They characterize serving shapes; the separate natural chat and
quality prompts provide task variety.

Both modes use the same pinned model, tokenizer, upstream chat template with a
fixed benchmark date, low reasoning effort, temperature zero, seed 1201, BF16 KV,
8,192-token context, two scheduled sequences, 512-token prefill chunks, prefix
cache disabled, O2 compilation and graph sizes one/two. No speculation is used.
The server uses TRITON_ATTN and the stock OAI Triton MXFP4 loader. Hardware
settings remain unchanged. Driver memory and hardware telemetry are sampled;
Torch peak allocation/reservation are collected through a private qualification
endpoint. The endpoint is absent from the normal public launch.

Report medians, means, standard deviations and sample counts. No tail percentile
is estimated from this small sample. Throughput includes reasoning and forced
post-stop continuation, whose counts are explicitly separated from final-channel
answer tokens. Natural-stop chat separately measures the first visible answer.
Input tokens divided by TTFT include queueing and first-token work; this is not
an isolated GPU prefill rate. Engine prefill timings are retained separately.

## Bounded alternatives and rejected approaches

| Experiment | Result / decision |
|---|---|
| Stock vLLM 0.26 O2, original MXFP4 | Initial 4.83 s median; selected stock baseline |
| Paiton expert decode, same weights | Initial 2.20 s median; corrected release image reproduced 2.214 s |
| Newer vLLM 0.28, original MXFP4 | Initial 7.59 s median; slower, excluded as baseline |
| llama.cpp, lossless original-MXFP4 GGUF | Quiet four-request screen 2.66 s median; stronger independent runtime reference, reported separately |
| Gemma 4 12B BF16 eager | Initial 13.12 s median; 10/10 small non-thinking quality screen, higher memory and latency |
| Gemma 4 12B graph startup | Bounded initialization stopped after over six minutes without serving |
| Intel AutoRound INT4 loader | Qualified vLLM rejects quantization method; no claim of measured INT4 quality or speed |
| Early Paiton BF16 router pointer | Incorrect ABI produced invalid output; rejected and fixed by conversion after stock route rounding |
| Monolithic container runtime layer | Exceeded GHCR 10 GB/layer limit; replaced with layers that preserve all 2,984 SDK hard-link groups |

Cross-model screens use different tokenizers and reasoning modes and do not prove
general model quality superiority. The separate llama.cpp runtime is pinned to
`434ddbbc0e30522e897670681e503b797c12b7c1`, retains BF16 dense weights and KV,
and disables prompt caching. Its initial screen uses four measured requests after
two warmups. Runtime changes are not counted as compiler gains.

## Numerical and task checks

The compiled expert region is checked against both an independent FP32 formula
with explicit BF16 boundaries and stock OAI Triton. Tests cover synthetic weights,
real checkpoint layers 0 and 23, one/two tokens, random/tied/skewed routing, large
inputs, zeros, and the BF16 router regression. Invalid ABI shapes are rejected;
graph replay must match eager Paiton bitwise. Unsupported serving shapes fall
back to stock.

A strict elementwise threshold initially flagged two near-cancellation values in
the final layer. That failure is retained. The revised checks retain the largest
outliers and require relative L2 error below 0.001 and error normalized by the
sum of absolute weighted expert contributions below 0.01. This handles
cancellation without treating a small aggregate error as sufficient by itself.
The initial final-layer check's maximum stock relative L2 was 0.000215 and maximum
contribution-normalized error was 0.00692. Generated text can still diverge.

The fixed 20-task suite covers instruction following, coding, reasoning, factual
questions, structured output and tools. Generated code runs without network in
an unprivileged container with memory, CPU, process and wall-time bounds. No
hosted judge or paid API is used. Long-context tests place facts near the beginning,
middle and end of approximately 7,675-token prompts, including two concurrent
requests; this is a bounded retrieval check, not a comprehensive context benchmark.

The first final-container chat evaluation crossed midnight. Inspection of returned
prompt IDs showed that the Harmony renderer ignored the supplied Jinja date.
Those unmatched chat results are retained but excluded from the paired quality
comparison. The corrected evaluation uses `VLLM_SYSTEM_START_DATE=2026-09-09`;
performance requests supply fixed token IDs and were unaffected.

Code cases evaluate execution after removing an optional outer Markdown fence.
Exact-output and JSON cases enforce their formatting predicates directly. Raw
responses remain available for checking instructions beyond those predicates.

## Final correctness results and profiling limits

The corrected-image 20-task evaluation passed **18/20 in both modes**, with
identical request bodies and prompt token IDs and no task-level regressions.
Both fail `code-balanced` (malformed final code channel) and `instruction-filter`
(JSON instruction produces a Markdown fence). All ten API checks pass in each
mode, including three reasoning efforts, streaming tools, tool continuation,
long-context retrieval and two concurrent long requests. This limited suite does
not establish universal quality parity.

All 24 numerical cases pass on the corrected image, including checkpoint layers
0 and 23, routing edge cases and graph replay. Detailed outliers are retained in
the result archive. No new quantization was applied, so there is no requantization
quality claim.

The original whole-runtime layer exceeded GHCR's per-layer limit. Splitting SDK
wheel components separately broke hard-link groups and was rejected. The final
layout separates Torch while preserving all 2,984 runtime hard-link groups.
The rejected layout's 3.429-second Paiton result is historical, not release data.

ROCm profiler attachment fails because its register library paths conflict;
an earlier startup-instrumented trace also reported inverted GPU timestamps.
Those traces are excluded from kernel-time claims. The retained engine histograms
separate prefill and decode, and focused device-event tests measured the expert
region. No isolated-kernel gain is presented as the serving improvement.

The raw archive includes requests, token IDs, outputs, per-request distributions,
telemetry, engine metrics, numerical outliers and the paired quality audit. See
[the archive index](benchmark/results/index.json), [machine-readable results](results.json),
[exact images](container-images.json), and [reproduction commands](REPRODUCE.md).

## Final public launcher check

The final image was tested through `serve-docker.sh --offline` in stock and
Paiton modes. Both passed terminal chat, curl chat, streaming, JSON-schema output
and the check that `/collective_rpc` is absent. The final image additionally
passed vocabulary initialization with Docker networking disabled. Its serving
module and compiled artifact match the full-benchmark image byte-for-byte.
Image IDs and the differences are recorded in `container-images.json`.
