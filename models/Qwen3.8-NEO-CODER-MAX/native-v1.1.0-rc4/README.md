# Native v1.1.0-rc4 local candidate

This locally qualified candidate improves native prefill while preserving the
qualified rc1/rc2 arithmetic. It uses 2,048-token prefill chunks with bounded
GEMM tiles. The exact author mixed Q4_K_M weights, BF16 output head, tokenizer,
reasoning controls and image projector are retained. No compiler dependencies
were added. vLLM remains the serving engine; llama.cpp is a separate reference.

## Measured behavior

Streaming HTTP seconds, median / p95; one warmup and five measured runs per
workload on the same idle R9700. Total context is 8K, temperature 0, seed 711,
fixed output length, BF16 KV and one active sequence. rc1 uses 1,024-token
chunks; rc4 and the fresh llama.cpp reference both use 2,048. Five samples do
not establish production tail latency.

| Input / output tokens | Previous rc1 | Paiton rc4 | llama.cpp |
|---|---:|---:|---:|
| 128 / 128 | 5.041 / 5.044 | 4.925 / 4.929 | 5.264 / 5.289 |
| 1024 / 128 | 5.778 / 5.779 | 5.631 / 5.637 | 5.933 / 5.936 |
| 4096 / 128 | 9.716 / 9.720 | 9.023 / 9.030 | 9.099 / 9.101 |

Against rc1, complete-request medians improve by 2.3%, 2.6% and 7.1%.
Against the fresh reference, the corresponding reductions are 6.4%, 5.1%
and 0.8%; the long-request lead is small. Prefill remains a limitation:
with just one output token, short/long medians are 0.271/4.184 seconds for
Paiton versus 0.239/3.995 for llama.cpp. Medium single-output latency improves
(0.923 versus 1.001 seconds). Faster decode offsets the remaining prefill gap
in the measured 128-output workloads. Neither engine wins every workload.

For a 1,024-square image and 128 output tokens, Paiton is 6.279 / 6.281 seconds
versus llama.cpp 6.455 / 6.473. The previous rc1 median was 6.694 seconds.
Peak residency across startup and qualification is 23.74 GiB for Paiton
versus 19.03 GiB for the reference.
Full raw reports and commands stay outside tracked product source.

## Correctness and limits

All 31,784,960 full-vocabulary logits from 4,096-token prefill plus 128 greedy
decode steps match the qualified 1,024-token baseline exactly. All 36 text API
requests preserve 2,322 token events and log probabilities exactly; all 24
image benchmark requests preserve streamed text and usage. The unqualified
[rc3 experiment](../native-v1.1.0-rc3/README.md) is retained separately.

The fixed text suite remains 10/11, with the same code-trace failure. PNG 5/5,
JPEG, state/isolation/cancellation, streaming, queued requests and context/image
limits pass. Held-out prefill perplexity remains 6.86244447185. These are narrow
checks, not a broad accuracy claim. The unchanged Q8 decode arithmetic inherits
rc1's separately measured +0.1301% held-out perplexity change against the older
FP32 profile; no new decode quality gain or cross-engine bit identity is claimed.

Supported: R9700/gfx1201, ROCm 7.14, 8,192 total context tokens, 2 GiB BF16 KV,
one active sequence with additional HTTP requests queued, text and one PNG/JPEG
image up to 4,096 patches / 1,048,576 resized pixels. MTP, prefix caching and
video are off. Native draft execution, verification and accepted-prefix KV/GDN
commit or rollback remain unfinished; no speculative speedup is claimed.

## Launch and rollback

The image is **local and not published**. After making the GPU available, run
from the repository root:

```bash
PAITON_NEO_IMAGE=sha256:8e491d58690adb8713b0ff7445cc9ea5043c386502b1aaf357c54d515f6c88af \
  ./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

Endpoint `http://127.0.0.1:8000/v1`, model `qwen38-neo`. The existing pinned
weight cache is reused. No compiler checkout is required. [candidate.json](candidate.json)
records the packaged compiler/plugin revisions, native hashes and full contract.
Runtime allowlisting, checksums, local integrity signature, all image layers
and offline startup with a read-only cache were checked. A clean restart took
170.2 seconds; a separate short-request repeat measured 4.914 / 4.943 seconds.

Stop only this candidate's container and use the [rc2](../native-v1.1.0-rc2/README.md)
image `sha256:7c2fdcf5055f5d8beb8b96c7ac0c30c9dc4e5f506443eba0236faf37bb558e27`
with the same launcher to roll back. Preserve the weight cache. Existing
published defaults, tags and packages remain unchanged. [Image API](../IMAGE_API.md)
and [notices](../THIRD_PARTY_NOTICES.md) apply.
