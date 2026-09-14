# Native v1.1.0-rc1 local candidate

This candidate keeps actual vLLM serving and the original mixed Q4_K_M GGUF.
It is built and tested locally, **not published**. The existing v1.0.0 launcher,
GHCR digest and other model releases remain unchanged.

The native compiler adds Q4 decode with Q8_1 activations, partitioned decode
attention and parallel GDN prefill. Original packed weight values and the BF16
output head are retained. Q8 activation arithmetic is a separate precision
profile, not bit-equivalent to the previous FP32 decode or llama.cpp. Prefill
uses temporary FP16 operands; normalization, softmax and GDN state retain FP32.
No new Torch/Triton dependency is introduced into the compiler or native artifacts.
The external vLLM serving stack retains its existing framework dependencies.

## Supported contract

R9700/gfx1201, pinned ROCm 7.14, TP1, 8,192 total tokens, 1,024-token prefill
chunks, 2 GiB BF16 KV cache and one active sequence. Additional HTTP requests
queue. Text and one PNG/JPEG still image are supported, up to 4,096 vision
patches / 1,048,576 resized pixels. Reasoning controls, streaming and the existing
Qwen3 Coder tool parser are retained. Prefix caching, video and MTP are disabled.

MTP needs native draft execution, target verification and accepted-prefix KV/GDN
commit or rollback. The present GDN path retains final state only. Draft tensors
are present but no speculative acceptance or speedup is claimed. See the
[parent model guide](../README.md) for exact tokenizer, model, image and notice
contracts; its performance/arithmetic discussion describes v1.0.0 unless stated.

The candidate passes the fixed 10/11 baseline text-task result (same code-trace
failure), 5/5 image fixtures, JPEG input, request isolation, cancellation, queued
requests, reasoning stream parity and the context boundary. Native decode
held-out perplexity increases 0.1301% versus the FP32 baseline, below the preset
1% bound. These are narrow qualification checks, not a broad quality guarantee.

Matched 128-output-token requests take 5.041 / 5.778 / 9.716 seconds median for
128 / 1,024 / 4,096 input tokens. Working llama.cpp takes 5.261 / 5.939 / 9.096
seconds with the same context, token counts, sampling and prefill capacity.
Short and medium prompts are faster; long prefill remains slower. Full raw
reports stay outside tracked product source. The candidate is not an overall
performance win across all workloads.

## Launch and rollback

The local immutable image ID, source revisions, native hashes and feature
contract are recorded in [candidate.json](candidate.json). After making the GPU
available, run from the repository root:

```bash
PAITON_NEO_IMAGE=sha256:23dd74dc1a81915a1501c49e49ca09890b16925274ab9311f2ead417b4970274 \
  ./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

This requires the locally built image; no GHCR pull is implied. Endpoint
`http://127.0.0.1:8000/v1`, model `qwen38-neo`. The existing named weight cache
is reused and its full hashes are checked. No compiler checkout is required.
The image sets `GPU_MAX_HW_QUEUES=1` and `PAITON_GGUF_PREFILL_TOKENS=1024`.
The explicit server option `--prefill-chunk-tokens 512` selects the smaller
validated scheduling profile without changing weights or total context.

Stop only the candidate container and run the same launcher without
`PAITON_NEO_IMAGE` to roll back to published v1.0.0, digest
`sha256:2a8fed46bc19fe8ca7164139c66f78e80f841120d42112b2968ba060c319fdbe`.
Keep the weight cache. The existing [image API example](../IMAGE_API.md) and
[notices](../THIRD_PARTY_NOTICES.md) apply.

The local bundle's allowlist, checksums and integrity signature, full image
layer review, and clean offline startup passed. Publication and a new immutable
registry pull remain on hold at the user's instruction.
