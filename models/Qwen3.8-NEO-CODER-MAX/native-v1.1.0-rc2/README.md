# Native v1.1.0-rc2 local candidate

This locally qualified candidate vectorizes Q4_K/Q6_K conversion into the
existing FP16 prefill buffers. Original packed weights, FP16 operands,
accumulation, Q8 decode arithmetic, GDN and vision remain unchanged from
[rc1](../native-v1.1.0-rc1/README.md). No new compiler dependencies are added.

Across36 matched text requests, all2,322 output token events and log probabilities
matched rc1 exactly. All24 matched image requests preserved streamed text and
token usage. The fixed text suite retains10/11 (same code-trace failure);
PNG5/5, JPEG, state/queue/cancellation, streaming and context checks pass.
Prefill held-out perplexity remains6.86244447185. These are narrow qualification
checks, not a broad capability benchmark.

Matched128-output-token HTTP medians are4.932/5.628/9.168
seconds for128/1024/4096 input tokens. Long text remains slightly behind the
previous matched llama.cpp measurement. Full raw reports stay outside tracked
product source.

The supported contract remains R9700/gfx1201, ROCm7.14,8K total context,1024-token
prefill chunks,2GiB BF16 KV,one active sequence, queued HTTP requests, text and
one PNG/JPEG image (up to4096 patches). MTP, prefix caching and video are off.
Native MTP draft/verification and accepted-prefix KV/GDN rollback remain unfinished.

## Launch and rollback

This image is local and **not published**. After making the GPU available, run
from the repository root:

```bash
PAITON_NEO_IMAGE=sha256:7c2fdcf5055f5d8beb8b96c7ac0c30c9dc4e5f506443eba0236faf37bb558e27 \
  ./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

Endpoint`http://127.0.0.1:8000/v1`, model`qwen38-neo`; existing pinned weight
cache is reused. No compiler checkout is required. [candidate.json](candidate.json)
pins source revisions, hashes and limits. Artifact checksums, local integrity
signature, all image layers and clean offline startup were verified.

Stop only rc2 and use rc1 image ID
`sha256:23dd74dc1a81915a1501c49e49ca09890b16925274ab9311f2ead417b4970274`
with the same launcher to roll back. Preserve the cache. Existing published
defaults remain unchanged. [Image API](../IMAGE_API.md) and
[notices](../THIRD_PARTY_NOTICES.md) apply.
