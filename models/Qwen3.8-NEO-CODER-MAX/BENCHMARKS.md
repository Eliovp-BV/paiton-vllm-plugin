# Native GGUF through vLLM: NEO v1.1.0 results

Paiton executes the author's mixed GGUF weights through the actual vLLM loader,
scheduler, sampling and streaming API, using native HIP artifacts. The qualified
Qwen3.8 NEO CODER MAX Q4_K_M profile supports text and one image. It has lower
median latency than working llama.cpp in all three tested 128-output-token text
workloads and both image sizes. The long-text lead is small: about 0.8%.

## Matched text comparison

Both current engines use the exact pinned Q4_K_M and BF16 head, source tokenizer,
8,192 total context, 2,048-token prefill chunks, one active sequence, BF16 KV,
MTP off, prefix caching off and `GPU_MAX_HW_QUEUES=1`. Fixed prompts have
128/1,024/4,096 tokens; outputs are exactly 1 or 128 tokens. Sampling is greedy,
seed 711, EOS ignored. Repeated ` vertex` prompts end with six distinct
single-token suffixes; API token usage is asserted. One warmup and five measured
requests per workload; engines run serially on the same idle GPU, with ownership
checks before requests and no overlapping CPU builds or image audits.

Seconds, median / p95. p95 is linearly interpolated from five samples and is not
a production tail-latency guarantee. Decode rate excludes the first token;
HTTP timing includes network/serving work and is not a kernel-only measurement.

| Engine | Input / output | HTTP median / p95 | TTFT median / p95 | Decode tok/s median |
|---|---:|---:|---:|---:|
| Paiton v1.1.0 | 128 / 1 | 0.271 / 0.273 | 0.271 / 0.273 | — |
| Paiton v1.1.0 | 128 / 128 | 4.925 / 4.929 | 0.272 / 0.272 | 27.295 |
| Paiton v1.1.0 | 1024 / 1 | 0.923 / 0.924 | 0.923 / 0.924 | — |
| Paiton v1.1.0 | 1024 / 128 | 5.631 / 5.637 | 0.945 / 0.946 | 27.103 |
| Paiton v1.1.0 | 4096 / 1 | 4.184 / 4.203 | 4.184 / 4.203 | — |
| Paiton v1.1.0 | 4096 / 128 | 9.023 / 9.030 | 4.233 / 4.235 | 26.496 |
| llama.cpp | 128 / 1 | 0.239 / 0.239 | 0.238 / 0.239 | — |
| llama.cpp | 128 / 128 | 5.264 / 5.289 | 0.416 / 0.419 | 26.197 |
| llama.cpp | 1024 / 1 | 1.001 / 1.004 | 1.001 / 1.004 | — |
| llama.cpp | 1024 / 128 | 5.933 / 5.936 | 1.022 / 1.024 | 25.857 |
| llama.cpp | 4096 / 1 | 3.995 / 4.006 | 3.995 / 4.006 | — |
| llama.cpp | 4096 / 128 | 9.099 / 9.101 | 4.013 / 4.016 | 24.972 |

Relative to the fresh llama.cpp comparison, 128-output request medians are
6.4%, 5.1% and 0.8% lower. **Some prefill-only cases still favor llama.cpp:**
short/long requests with one output token are 0.239/3.995 seconds versus
Paiton 0.271/4.184 seconds. Faster decode offsets that gap in the measured
128-output workloads. Different contexts, output lengths or concurrency can
change the comparison; this is not a universal speed claim.

Inter-token latency, milliseconds median / p95, pooled across the 635 intervals
from five measured 128-output streams:

| Engine | 128 input | 1024 input | 4096 input |
|---|---:|---:|---:|
| Paiton v1.1.0 | 36.631 / 36.891 | 36.899 / 37.128 | 37.758 / 37.987 |
| llama.cpp | 37.741 / 37.967 | 38.238 / 38.485 | 39.611 / 39.851 |

## Images and queued clients

Identical PNG bytes, source chat template, reasoning disabled, greedy sampling
and fixed token counts. Image bytes vary between runs to avoid cache reuse.
Image HTTP time includes the native image encoder, language prefill and decode.
The 256-square / 1,024-square images produce 95 / 1,055 total prompt tokens.
One warmup and five measured streams per case. Image TTFT is first visible
content; image chunks do not establish token-level inter-token latency.

| Engine | Image / output | HTTP median / p95 (s) | TTFT median / p95 (s) |
|---|---:|---:|---:|
| Paiton v1.1.0 | 256² / 1 | 0.297 / 0.297 | 0.297 / 0.297 |
| Paiton v1.1.0 | 256² / 128 | 4.909 / 4.911 | 0.297 / 0.297 |
| Paiton v1.1.0 | 1024² / 1 | 1.629 / 1.770 | 1.629 / 1.770 |
| Paiton v1.1.0 | 1024² / 128 | 6.279 / 6.281 | 1.626 / 1.626 |
| llama.cpp | 256² / 1 | 0.472 / 0.483 | 0.472 / 0.483 |
| llama.cpp | 256² / 128 | 5.111 / 5.128 | 0.474 / 0.490 |
| llama.cpp | 1024² / 1 | 1.730 / 1.733 | 1.727 / 1.730 |
| llama.cpp | 1024² / 128 | 6.455 / 6.473 | 1.747 / 1.768 |

With two simultaneous HTTP clients, each requesting 128/128 tokens, Paiton's
pair makespan is 9.905 / 9.907 seconds versus llama.cpp 10.466 / 10.661 seconds;
aggregate output is 25.845 versus 24.460 tokens/s. Both execute only one active
sequence: these are queued clients, not batch-two GPU throughput.

## Precision and quality

Original Q4_K, Q6_K, small FP32 coefficients and the BF16 output head remain
intact. This is native consumption of the selected GGUF, not AWQ repacking or
a newly quantized source checkpoint. v1.1.0 uses the separately identified
`q4_decode_q8_1` activation profile for large Q4 decode projections; prefill
uses temporary FP16 operands and FP32 accumulation. Small operations and GDN
state retain FP32. Native implementation and build dependencies add neither
Torch nor Triton; existing external vLLM integration is separate.

The inherited 770-token teacher-forced decode evaluation measures perplexity
6.87108244564 against the older FP32 profile's 6.86215184876: +0.1301%, below
the preset 1% bound. This is a narrow held-out set, not task accuracy or a new
v1.1.0 quality improvement. The final prefill sweep introduces no further
activation-arithmetic change. Its 31,784,960 compared full-vocabulary logits
(4,096-token prefill plus 128 greedy steps) match the qualified baseline exactly.
All 36 text API requests preserve 2,322 token events and log probabilities;
all 24 image benchmark requests preserve streamed text and usage. Prefill
held-out perplexity is unchanged at 6.86244447185.

Both engines score 10/11 on the fixed coding/instruction/tool/multi-turn/reasoning
suite, failing the same code-trace question, and 5/5 on image fixtures. JPEG,
stream parity, cancellation, A–B–A request isolation, queued requests, 8K context
boundaries and image limits pass. These checks do not establish broad capability
preservation. llama.cpp uses different activation arithmetic; bit equivalence
between engines is not claimed. The historical FP32-reference error result
(0.01401 max error) is not relabeled as a new Q8-decode measurement.

## Deployment and memory

Qualified hardware: R9700/gfx1201, 34,208,743,424-byte VRAM, Linux 6.17.0-1028-oem,
amdgpu 7.1.3.31500000. Both engines use ROCm 7.14.60850 from pinned base
`sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9`.
vLLM revision `39bd959b582c85e78e7e0326d49042ce7c3c07ed`; unmodified llama.cpp
revision `434ddbbc0e30522e897670681e503b797c12b7c1`, HIP Release gfx1201,
graphs and MMQ/MFMA enabled. No host ROCm libraries are injected.

| Full qualification scope | Peak GPU GiB | Summed process RSS GiB | Cgroup peak GiB |
|---|---:|---:|---:|
| Paiton v1.1.0 candidate | 23.736 | 5.114 | 10.603 |
| llama.cpp reference | 19.034 | 10.148 | 10.557 |

200ms sampling includes startup, text, images, queueing and quality tests;
Paiton also runs cancellation/context-limit checks. RSS may double-count shared
mappings; cgroup includes file cache. These peaks are not minimum hardware
requirements. Paiton uses more VRAM in this comparison. Its native graph plans
596,115,456 bytes for activations and 249,561,088 bytes for shared workspace;
the hybrid KV allocation is 2 GiB. Context remains 8K, one active request,
one PNG/JPEG image; MTP, video and prefix caching remain off.

## Preparation, provenance and reproduction

The qualified artifact built in 330.930 seconds with framework imports blocked;
an unrelated two-core build overlapped, so this is not a build-speed comparison.
A clean offline candidate restart took 170.216 seconds; a separate short-request
repeat was 4.914 / 4.943 seconds. Initial first-request warmups are excluded from
steady tables; this sweep does not isolate every graph's capture time. No new
offline re-quantization or autotuning stage was introduced.

v1.1.0 promotes the exact qualified rc4 payload. Its filesystem layers and all
runtime configuration are byte-identical; only version/source labels change.
[Release metadata](paiton-release.json) binds source, artifact, image, notices
and post-publication verification. The compiler remains private. Only explicitly
allowlisted binaries and sanitized metadata are exported.

The source benchmark harness retains each exact request and streamed event:

```bash
python3 models/Qwen3.8-NEO-CODER-MAX/benchmark_http.py \
  --engine paiton --label local-check --output-dir /tmp/neo-benchmark \
  --base-url http://127.0.0.1:8000/v1 --owned-container paiton-qwen38-neo
```

The matched reference used `llama-server -c 8192 -np 1 -b 2048 -ub 2048 -ngl 999
--load-mode none --fit off -fa on -ctk bf16 -ctv bf16 --jinja --image-max-tokens 1024
--spec-type none`, the same pinned GGUF/projector and `GPU_MAX_HW_QUEUES=1`.
Raw reports, logs, weights and intermediate builds are retained outside tracked
source. The release summary records hashes of the underlying evidence. Natural
completion results are kept separate; shorter reasoning is not counted as
faster kernels.
