# NEO Q4_K_M RDNA4 comparison

The optimized Paiton package provides native text and image execution through
vLLM's loader, scheduler, sampling and streaming API. This report compares it
with the initial native Paiton/vLLM implementation and a working unmodified
llama.cpp server. **The initial implementation is not stock vLLM.** Stock vLLM
support for this exact mixed-GGUF hybrid model was not qualified.

For 128 generated tokens, complete-request medians improve **2.38–3.00×**
over the initial native implementation. The working unmodified llama.cpp
server remains faster than the final Paiton/vLLM package. The initial/optimized
comparison includes an independently qualified prefill precision change.

## Measurement contract

One idle Radeon AI PRO R9700 / gfx1201, 34,208,743,424 bytes VRAM; Linux
6.17.0-1028-oem, amdgpu 7.1.3.31500000, approximately 15.52 GiB host RAM and
4 GiB swap. Both serving engines use the same pinned ROCm 7.14.60850 runtime
base `sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9`.
No host ROCm libraries are injected. vLLM is revision
`39bd959b582c85e78e7e0326d49042ce7c3c07ed`; clean llama.cpp is revision
`434ddbbc0e30522e897670681e503b797c12b7c1`, Release/HIP, gfx1201,
HIP graphs and MMQ matrix support enabled.

All runs use the exact pinned author Q4_K_M file, its BF16 output head,
source tokenizer, greedy sampling, seed 711, 8,192 total context, prefill chunks
of 512, one active sequence, BF16 attention KV and MTP disabled. vLLM prefix
caching and llama.cpp prompt reuse are disabled. Concurrent HTTP requests queue;
this is not a batch-two GPU throughput claim. GPU ownership is checked before
each request. No builds or audits run alongside the final measured workloads.

Text prompts have exactly 128, 1,024 or 4,096 tokens: repeated ` vertex` plus
one of six single-token suffixes. All 36 tokenizer probes match exactly between
the source tokenizer and llama.cpp. One warmup and five measured requests are
used per workload, with exactly 1 or 128 output tokens and EOS ignored in both
engines. p95 is the linearly interpolated empirical percentile of five samples;
this small sample does not establish production tail-latency guarantees.
TTFT starts before sending the HTTP request. Decode rate excludes the first
output token and ends at the last token-bearing stream event. Total time includes
the streaming HTTP request through completion.

The original token-ID-array measurements remain separate: llama.cpp's OpenAI
completions endpoint rejected arrays, so they are not mixed into these matched
text-string tables. Natural completion lengths are reported separately below.

## Precision and attribution

The initial native implementation uses FP32 projection operands and simpler
packed-weight reductions, without native decode graphs. The optimized package
uses vectorized packed Q4/Q6 decode, FP16 temporary prefill matrix operands with
FP32 accumulation, native attention, and native HIP decode graphs. Original
quantized weight values and the BF16 output head remain unchanged. FP32
activations, small coefficients and GDN state are retained; the model is not
expanded into FP32 weights.

These are explicitly different activation-arithmetic profiles, independently
qualified for quality. Their full-request improvement must not be attributed
solely to faster kernels at identical arithmetic. Isolated same-contract native
GEMV tests provide that narrower evidence: representative Q4 gate projection
0.415264 → 0.109301 ms, Q6 down projection 0.385562 → 0.092075 ms. FP16 prefill
is a separate tested profile: at M=512, representative Q4 7.601863 → 1.643029 ms
and Q6 7.790576 → 1.524459 ms. Its relative operator change is about 0.03%.

Unmodified llama.cpp's MMQ path quantizes activations internally. Its speed and
quality are useful cross-engine comparisons, but its arithmetic is not
bit-equivalent to Paiton. No source re-quantization, AWQ substitution, reasoning
reduction or shorter output is used to claim a speedup.

## Text streaming latency

Seconds, median / p95. Decode rate is median tokens/second.

| Engine | Input / output tokens | Complete HTTP | TTFT | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Initial native vLLM | 128 / 1 | 0.850 / 0.853 | 0.850 / 0.853 | — |
| Initial native vLLM | 128 / 128 | 17.492 / 17.508 | 0.834 / 0.834 | 7.62 |
| Initial native vLLM | 1024 / 1 | 4.902 / 4.920 | 4.902 / 4.920 | — |
| Initial native vLLM | 1024 / 128 | 22.650 / 22.660 | 4.924 / 4.926 | 7.17 |
| Initial native vLLM | 4096 / 1 | 26.259 / 26.265 | 26.259 / 26.265 | — |
| Initial native vLLM | 4096 / 128 | 47.551 / 47.555 | 26.260 / 26.261 | 5.96 |
| Optimized Paiton/vLLM | 128 / 1 | 0.434 / 0.436 | 0.434 / 0.436 | — |
| Optimized Paiton/vLLM | 128 / 128 | 7.352 / 7.356 | 0.438 / 0.439 | 18.37 |
| Optimized Paiton/vLLM | 1024 / 1 | 1.446 / 1.448 | 1.446 / 1.448 | — |
| Optimized Paiton/vLLM | 1024 / 128 | 8.945 / 8.954 | 1.470 / 1.471 | 16.99 |
| Optimized Paiton/vLLM | 4096 / 1 | 6.491 / 6.508 | 6.491 / 6.507 | — |
| Optimized Paiton/vLLM | 4096 / 128 | 15.875 / 15.886 | 6.503 / 6.503 | 13.55 |
| Unmodified llama.cpp | 128 / 1 | 0.239 / 0.240 | 0.239 / 0.240 | — |
| Unmodified llama.cpp | 128 / 128 | 5.258 / 5.262 | 0.414 / 0.415 | 26.22 |
| Unmodified llama.cpp | 1024 / 1 | 1.030 / 1.035 | 1.030 / 1.035 | — |
| Unmodified llama.cpp | 1024 / 128 | 5.976 / 5.977 | 1.063 / 1.064 | 25.85 |
| Unmodified llama.cpp | 4096 / 1 | 4.016 / 4.030 | 4.016 / 4.030 | — |
| Unmodified llama.cpp | 4096 / 128 | 9.149 / 9.243 | 4.066 / 4.160 | 24.99 |

Inter-token latency in milliseconds; median / p95 over individual token intervals from the five measured 128-output-token streams.

| Engine | 128 input | 1,024 input | 4,096 input |
| --- | ---: | ---: | ---: |
| Initial native vLLM | 131.174 / 131.885 | 139.657 / 140.277 | 167.645 / 168.547 |
| Optimized Paiton/vLLM | 54.475 / 54.930 | 58.754 / 59.828 | 73.720 / 74.637 |
| Unmodified llama.cpp | 37.730 / 37.966 | 38.265 / 38.511 | 39.592 / 39.822 |

## FP32 projection profile

This additional native text-only build retains FP32 prefill operands while using the optimized packed decode and attention paths. It is a separately identified deployment profile, not an identical artifact with only a runtime switch. The compiler revision is `1c8aa55aff0b0d620c93879a3ed766e62a44b9b8`, artifact SHA256 `942a611e836f397d2624316e80418bc38064e7b5d876e02d006608463474a1b7`. The initial pre-commit artifact SHA256 is `0c5f2cac4dd0a6da41a82e66643535daafbc6353bd5ab040b5957018104fdfdd`; its compiler revision was not recorded, so source-level reproducibility of that historical baseline is limited.

| Input / output | HTTP median / p95 (s) | TTFT median / p95 (s) | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 128 / 1 | 0.879 / 0.880 | 0.879 / 0.880 | — |
| 128 / 128 | 6.429 / 6.442 | 0.869 / 0.871 | 22.84 |
| 1024 / 1 | 4.785 / 4.800 | 4.785 / 4.800 | — |
| 1024 / 128 | 10.941 / 10.950 | 4.809 / 4.816 | 20.74 |
| 4096 / 1 | 19.823 / 19.830 | 19.823 / 19.830 | — |
| 4096 / 128 | 27.810 / 27.812 | 19.833 / 19.836 | 15.92 |

Held-out perplexity for this profile: 6.86218558, over 770 predictions.

## Queued HTTP requests

Two simultaneous requests, each 128 input / 128 output tokens; one active GPU sequence. One warmup pair and five measured pairs.

| Engine | Request latency median / p95 (s) | Pair makespan median / p95 (s) | Pair throughput (tok/s) |
| --- | ---: | ---: | ---: |
| Initial native vLLM | 26.250 / 35.011 | 35.009 / 35.013 | 7.31 |
| Optimized Paiton/vLLM | 11.032 / 14.712 | 14.701 / 14.714 | 17.41 |
| Unmodified llama.cpp | 7.935 / 10.596 | 10.571 / 10.606 | 24.22 |

## Image streaming requests

Identical generated PNG images, text prompt, source template, reasoning disabled, greedy sampling and fixed output counts. The image bytes vary slightly between repetitions to prevent image-cache reuse. Both engines see the same bytes in each matched run. The 256-square image contributes 64 image embeddings; the 1,024-square image contributes 1,024. One warmup plus five measured requests per case. Image TTFT measures first visible content. Full-request output rate is completion tokens / complete HTTP time, including image processing and prefill. Chat content events can coalesce or hide tokens, so no image inter-token rate is inferred from visible chunks.

| Engine | Image | Input / output tokens | Complete HTTP median / p95 (s) | TTFT median / p95 (s) | Full-request output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paiton/vLLM | 256 × 256 | 95 / 1 | 0.459 / 0.459 | 0.458 / 0.459 | — |
| Paiton/vLLM | 256 × 256 | 95 / 128 | 7.278 / 7.288 | 0.458 / 0.459 | 17.59 |
| Paiton/vLLM | 1024 × 1024 | 1055 / 1 | 2.435 / 2.437 | 2.435 / 2.437 | — |
| Paiton/vLLM | 1024 × 1024 | 1055 / 128 | 9.837 / 9.844 | 2.433 / 2.433 | 13.01 |
| llama.cpp | 256 × 256 | 95 / 1 | 0.476 / 0.483 | 0.476 / 0.483 | — |
| llama.cpp | 256 × 256 | 95 / 128 | 5.128 / 5.141 | 0.490 / 0.504 | 24.96 |
| llama.cpp | 1024 × 1024 | 1055 / 1 | 1.774 / 1.791 | 1.773 / 1.789 | — |
| llama.cpp | 1024 × 1024 | 1055 / 128 | 6.474 / 6.484 | 1.758 / 1.772 | 19.77 |

The last two large-image 128-output requests completed in 8.431 and 8.408 s,
versus approximately 9.84 s for the other three. All samples are retained;
that variation is not presented as a new steady-state optimization.

## Memory

Peaks sampled every 100 ms. GiB = 2³⁰ bytes. Driver VRAM includes runtime/cache allocations. Host memory is container cgroup memory, not process RSS; file cache and swap are reported separately. Column maxima may occur at different instants and must not be added together.

| Profile | Driver VRAM | Cgroup memory | Anonymous | File cache | Swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial native vLLM | 20.195 | 10.791 | 2.711 | 8.016 | 0.000 |
| Optimized Paiton/vLLM | 21.093 | 10.829 | 3.739 | 7.084 | 0.000 |
| Unmodified llama.cpp | 18.584 | 8.781 | 4.916 | 6.612 | 0.000 |
| Paiton/vLLM images | 22.568 | 11.007 | 3.955 | 7.005 | 0.000 |
| llama.cpp images | 18.679 | 9.947 | 8.747 | 1.169 | 0.074 |

The target tensors occupy 18,036,258,816 bytes; 451,319,808 bytes of draft tensors
remain unused in the upstream file. The projector file is 931,145,920 bytes.
The language artifact plans 149,028,864 bytes of activation storage and
196,083,712 bytes of shared scratch. vLLM reserves a 2 GiB hybrid cache pool.
The native image workspace is bounded at 4,096 patches. Measured peaks include
additional framework, graph, preprocessing and allocator overhead. A separate
maximum-image state/isolation test peaked at 22.691 GiB driver VRAM. The runtime
image size reported by Docker is 17.799 GB; its saved OCI archive is 5.143 GB
(4.790 GiB), retaining the
pinned external vLLM/ROCm stack; the 19.43 GB language/projector download is
separate. Neither figure is a minimum host RAM or VRAM requirement.

## Preparation and startup

These are single observations, not repeated median/p95 benchmarks. Native
language compilation took 281.829 s using static kernel selection; no autotuning
profilers ran. An isolated prepared-cache package start reached health in
167.834 s (2 s polling; OS file cache not flushed). Startup inventory and
checkpoint verification took 56.048 s, target copy/binding another 38.769 s,
and the reported vLLM model-loading phase 98.079 s including projector loading.
These overlapping phases must not be added together. The first start with an
empty named cache, including downloads, reached health in 431.993 s (5 s polling).
An immutable GHCR pull followed by a new container with `HF_HUB_OFFLINE=1`
reused the named cache and reached health in 201.078 s
(5 s polling). Text, PNG/JPEG and the 11-task suite repeated successfully,
with the same single code-tracing failure. No whole-model dequantization or
weight conversion is required.

## Numerical and quality qualification

Criteria were fixed before judging candidates: exact decoded representative
weights, FP64 operator error bounds, full logits cosine ≥0.9999 and max absolute
error ≤0.1, no lost baseline task passes, and held-out perplexity increase ≤1%.

The selected FP16-prefill Paiton model has full-logit cosine 0.99999958 and max
absolute difference 0.01401 against a precision-matched FP32 reference. Default
llama.cpp MMQ logits exceed the cross-engine tolerance (about 0.34 max error);
an isolated FP32 reference resolves this difference without relaxing the limit.
That unresolved bit-level difference from default MMQ is disclosed rather than
called numerical equivalence.

The frozen original prose/code evaluation contains 770 next-token predictions:
default llama.cpp perplexity 6.92827299, precision-matched FP32 reference
6.86116330, initial FP32 Paiton 6.86210834, and the qualified FP16-prefill package
6.86223329. These are narrow reproducible checks, not a broad capability ranking.

The independent vision reference uses the exact GGUF projector weights and
its tanh-GELU merger contract. Full 27-layer square/rectangular probes have max
absolute errors 0.00032044 / 0.00103760 and relative L2 errors
2.7315e-6 / 4.3097e-6, within the frozen 0.003 / 1e-5 limits. Standalone C++
changed-input graph replay is bit-exact with fresh eager execution. The selected
GGUF merger arithmetic is retained; it is not claimed bit-equivalent to a new
run of the author-source Transformers checkpoint's merger.

A supplemental native C++ three-axis mRoPE oracle covers FP32 positions through
8K, with a bound accounting for FP32 phase rounding. Maximum absolute error
is 0.000229744; maximum fraction of the explicit error bound is 0.184.
Changed-input HIP graph replay is bit-exact with fresh execution. This test
build and execution require neither Torch nor Triton. The supplemental test
commit is `965c10c060c3d1a9e837d23629880c08863533d6`; packaged compiler code
remains pinned to `0b7ca75aec37b4049dc7c13027774e9efd84794d`.

Five image API fixtures cover red/blue colors, spatial position, counting and
OCR. Both engines pass all five with identical prompt/output token counts.
Native image A–B–A requests, queued requests, cancellation, maximum-size chunked
prefill and the single-image limit are tested. Text checks cover state isolation,
shared-prefix fresh requests, stream parity, cancellation, queueing, the 8,192
context boundary and rejection above the limit. Prefix caching itself is disabled.

## Natural task completion

Frozen 11-task suite; actual completion tokens and full non-streaming HTTP time.
These lengths may differ across engines and are not kernel-throughput evidence.
Both engines pass 10/11, failing the same code-tracing question.

| Task | llama.cpp pass / tokens / seconds | Paiton/vLLM pass / tokens / seconds |
| --- | ---: | ---: |
| instruction_literal | pass / 3 / 0.484 | pass / 3 / 0.440 |
| instruction_json | pass / 10 / 0.699 | pass / 10 / 0.725 |
| code_trace | fail / 3 / 0.458 | fail / 3 / 0.436 |
| code_slice | pass / 10 / 0.717 | pass / 10 / 0.723 |
| code_fib | pass / 44 / 1.969 | pass / 44 / 2.191 |
| code_unique | pass / 52 / 2.972 | pass / 52 / 2.534 |
| multi_turn | pass / 6 / 0.699 | pass / 6 / 0.563 |
| reasoning_off | pass / 3 / 0.576 | pass / 3 / 0.427 |
| reasoning_on | pass / 197 / 7.698 | pass / 201 / 9.091 |
| long_context | pass / 8 / 6.033 | pass / 8 / 9.195 |
| tool_call | pass / 28 / 2.076 | pass / 28 / 1.738 |

## Limits and unsuccessful experiments

MTP is not implemented or advertised: native draft execution, target verification,
acceptance/rejection and accepted-prefix KV/GDN commit/rollback remain necessary.
No Paiton MTP acceptance rate or speedup is claimed. Separate earlier llama.cpp
MTP observations do not qualify Paiton speculation. Video, multi-image requests,
multiple active sequences and prefix caching remain outside the release contract.
Q6_K received metadata and representative-tensor inspection, not full-model
qualification. A newly quantized author-source checkpoint would be a distinct
derivative and is not this release.

The native BF16 output-head alternatives passed sampled FP64 and changed-input graph checks but did not improve batch-one decode: current 4.0219 ms, alternatives 4.0320 / 4.4597 ms. They were not adopted. Initial cold outliers are retained and are not used to claim a p95 gain.

An activation-Q8 prototype gave only a small Q4 improvement and a Q6 regression,
with about 0.53% relative activation error, and was not adopted. Persistent graph
metadata buffers regressed warm request latency; tracing showed only two initial
graph captures across 127 decode steps, so that change was discarded. An earlier
text-only graph pilot reached about 5.94 seconds for one short workload; it was
not reproduced in the full packaged profile and is excluded from final gains.

ROCprof attempts did not produce a valid complete worker trace: attachment
missed the worker, and a later run reported invalid timestamps and failed to
finalize worker output. Instrumented timings are excluded. The full multimodal
package decodes more slowly than the separate optimized FP32 text-only build;
this gap remains unresolved and is reported without claiming FP16 is always faster.

## Reproduction and release

Use the immutable model launcher and revisions in [paiton-release.json](paiton-release.json).
The public standard-library harness preserves existing result files and checks
GPU ownership before each request:

```bash
python3 models/Qwen3.8-NEO-CODER-MAX/benchmark_http.py \
  --engine paiton --label paiton-repeat --base-url http://127.0.0.1:8000/v1 \
  --owned-container paiton-qwen38-neo --output-dir /tmp/neo-comparison
```

For llama.cpp, build the pinned revision with `-DGGML_HIP=ON
-DGPU_TARGETS=gfx1201 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
-DGGML_HIP_MMQ_MFMA=ON -DCMAKE_BUILD_TYPE=Release` in the
compatible runtime. Use `llama-server -m MODEL --mmproj PROJECTOR --alias qwen38-neo
-c 8192 -np 1 -b 512 -ub 512 -ngl 999 --load-mode none --fit off -fa on
-ctk bf16 -ctv bf16 --jinja --image-max-tokens 1024 --spec-type none`, then run
the same harness with `--engine llama`, its endpoint and owned container name.
The harness disables llama prompt reuse per request. Preserve the exact pinned
weight/projector hashes and tokenizer; do not substitute another Qwen fine-tune.

Raw results, fixed evaluation inputs, exact local commands and rejected
experiments are retained outside tracked source. Compiler source remains private;
only allowlisted native artifacts, sanitized provenance and notices are exported.
See [RELEASE_CHECKS.md](RELEASE_CHECKS.md) for publication checks and scope.
