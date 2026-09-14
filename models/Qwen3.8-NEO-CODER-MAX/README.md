# Qwen3.8 NEO CODER MAX on RDNA4

Published and qualified for text and one still image on R9700/gfx1201.
The immutable GHCR image passed an offline start with cached weights, text
generation and PNG/JPEG image requests without compiler or plugin checkouts.

A separately identified [native v1.1.0-rc4 local candidate](native-v1.1.0-rc4/README.md)
adds faster decode and prefill. It is not published and does not change the
default image described below.

This profile runs the requested DavidAU fine-tune through vLLM's model loader,
scheduler, sampling and streaming API, with the target model executing Paiton's
native HIP artifacts. GGUF is the weight container; llama.cpp is a separate
reference engine and is not a serving dependency.

## Model and precision

The [selected author GGUF revision](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/tree/89230607b3708bc1efe174e8fd34d4b164476c88)
is `89230607b3708bc1efe174e8fd34d4b164476c88`. The exact Q4_K_M filename, SHA256,
and [author-source revision](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU/tree/895782677541896947ea136714b45dbed994daca)
`895782677541896947ea136714b45dbed994daca` are pinned in
[checkpoint.lock.json](checkpoint.lock.json).

The 18,498,573,856-byte file contains 433 Q4_K, 64 Q6_K, eight Q8_0,
360 FP32 and one BF16 tensor. Q8_0 matrices belong to the unused MTP block.
The output head remains BF16. The 64-layer target has 48 GDN layers and
16 full-attention layers, using the source's Qwen3.5 compatibility identifiers.
Hidden width is 5,120, MLP width 17,408 and vocabulary size 248,320.
Full attention uses 24 query / four KV heads with dimension 256; GDN uses
16 key / 48 value heads with dimension 128 and a four-tap convolution.
The rotary dimension is 64, theta 10,000,000, with interleaved mRoPE
sections [11, 11, 10]. This is the NEO fine-tune, distinct from the library's
AMD Qronos release.

Paiton loads the original packed tensor bytes with bounded host buffers. It does
not expand the whole model to BF16, re-quantize the source checkpoint, apply AWQ
packing, change the already-multiplicative normalization values, or reapply the
stored GDN decay transform. Temporary prefill matrix operands use FP16 and
accumulate in FP32. Decode consumes packed Q4/Q6 weights with vectorized FP32
arithmetic. Small coefficients, activations and recurrent state remain FP32;
attention KV storage remains BF16. Attention uses two BF16 query components
with FP32 softmax/value accumulation. These arithmetic choices are separate
from the unchanged weight quantization.

## Supported profile

| Setting | Contract |
| --- | --- |
| GPU | Radeon AI PRO R9700, gfx1201, approximately 32 GiB VRAM |
| Host tested | Linux 6.17.0-1028-oem; amdgpu 7.1.3.31500000 |
| Container runtime | Pinned ROCm 7.14.60850 stack; no host-library injection |
| vLLM revision | 39bd959b582c85e78e7e0326d49042ce7c3c07ed |
| Parallelism | TP1; one active sequence; additional HTTP requests queue |
| Context | 8,192 total tokens; prefill chunks up to 512 tokens |
| KV allocation | 2 GiB |
| Execution | Native HIP decode graphs; eager prefill and vLLM scheduling |
| API | Completions, chat, streaming, Qwen3 reasoning and Qwen3 Coder tool parsing |
| Modality | Text and one still image per request; up to 4,096 vision patches (1,048,576 resized pixels); video disabled |
| Prefix caching | Disabled; fresh requests sharing text prefixes are tested |
| MTP | Disabled; native draft/verification/state rollback remains unfinished |

The native 27-layer image encoder uses the pinned 931,145,920-byte
`mmproj-BF16.gguf`, including its BF16 matrices and FP32 patch/position/norm
exceptions. Vision computation runs in HIP/rocBLAS; vLLM and Transformers
provide the existing image preprocessing and API handling. Image embeddings
retain FP32 precision through vLLM staging and enter the compiled three-axis
mRoPE language path. Larger images are resized within the stated pixel budget.
Image tokens and generated tokens share the 8,192-token context budget.

The exact source tokenizer, template and generation config are retained.
Five API tokenizer checks match the pinned tokenizer exactly, including code,
Unicode and special tokens. The GGUF adds 243 unused padding entries after
ID 248076 that are absent from the source tokenizer; these are not advertised
as user tokens. The inherited tokenizer warning about a Mistral
regex was investigated; changing that regex would change this model's pinned
tokenization and is not part of this release. EOS IDs remain 248046 and 248044.
Use `chat_template_kwargs.enable_thinking` to control reasoning explicitly.
The preserved template opens a thinking section when that setting is omitted;
pass `false` to request the source template’s non-thinking form. The retained
generation configuration specifies temperature 1.0, top-k 20 and top-p 0.95;
all qualification comparisons explicitly use greedy sampling instead.

## Launch

Use the immutable image selected by:

```bash
./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

The default endpoint is `http://127.0.0.1:8000/v1`, and the served model name is
`qwen38-neo`. `PAITON_NEO_PORT` selects another host port;
`PAITON_NEO_CACHE` selects another Docker volume. The first start downloads the
pinned language GGUF and projector into the cache (19.43 GB combined).
Subsequent starts reuse them and verify their full hashes.
A compiler checkout, build tools and reference engine are not required.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-neo","messages":[{"role":"user","content":"Write a Python function that preserves the first occurrence of each integer in a list."}],"temperature":0,"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

For an existing GGUF file, mount it read-only into the container and pass
`--checkpoint /path/in/container/model.gguf`. It must match the pinned filename's
contents and SHA256. Mount the pinned projector and pass
`--projector /path/in/container/mmproj-BF16.gguf` to reuse an existing projector
file as well. No alternative quantization is selected automatically.

For image input, use the same `/v1/chat/completions` endpoint with an
`image_url` content item containing an image URL or a base64 data URL, plus a
`text` content item. See the complete example in [IMAGE_API.md](IMAGE_API.md).

## Correctness and performance evidence

The qualification thresholds were fixed before evaluating candidates: exact
decoded weight values, explicit operator error bounds, full logits cosine at
least 0.9999 and maximum absolute difference at most 0.1, no lost baseline task
passes, and held-out perplexity increase at most 1%.

The native full-model logits have cosine 0.99999958 and maximum absolute error
0.01401 against a precision-matched native reference. Standalone Paiton and
actual vLLM logits are bit-for-bit equal for the fixed 128-token probe. Native
graph replay, changed inputs, nonzero GDN state, poisoned cache padding,
prefill/decode transitions, request isolation, cancellation, queued requests
and the context boundary are tested. An 8,191-token prompt plus one output token
succeeds; oversized prompts are rejected.

The fixed 11-task suite scores 10/11 in both the reference and Paiton. Both fail
the same code-tracing question. The independent held-out evaluation contains
770 predicted tokens of original prose/code: perplexity is 6.92827 for the
unmodified reference and 6.86223 for Paiton. These are narrow reproducible tests,
not a broad capability benchmark. Graph mode preserves the same likelihood.

Unmodified llama.cpp uses different activation arithmetic: its default MMQ path
quantizes activations. Its raw logits differ from Paiton beyond the above
numerical tolerance (maximum difference about 0.34 in the probe). Forcing FP32
matrix arithmetic in an isolated reference build resolves that discrepancy
without changing the tolerance. Quality checks also compare against the
unmodified engine. Cross-engine performance is therefore reported separately
from comparisons of Paiton arithmetic profiles; it is not a claim of identical
arithmetic or universally identical generation.

The matched timing tables and arithmetic distinctions are recorded in
[BENCHMARKS.md](BENCHMARKS.md).
All timing workloads hold input/output token counts fixed, use temperature zero,
seed 711, one active sequence, the same context limit and unchanged weights.
Reasoning-token reductions are not used to claim kernel speedups.

## MTP, vision and alternatives

The GGUF includes an MTP block with Q8_0 draft matrices, and the draft/source
metadata and tokenizer compatibility were inspected. The pinned author source
also indexes `model-mtp-restored.safetensors`; its inspected header contains
BF16 `mtp.fc.weight` with shape [5120, 10240]. Paiton's current target
path exposes final GDN state only and rejects speculative metadata. Native MTP
still needs a draft-block entrypoint, target verification, acceptance/rejection
handling and accepted-prefix KV/GDN commit or rollback. Snapshot-and-replay is
one possible implementation, with overhead that must be measured. The package
does not silently enable upstream speculation or advertise unmeasured benefit.
Separate llama.cpp MTP measurements do not establish Paiton MTP support.

Native vision matches an independent FP32 reference using the exact projector
weights: full-encoder relative L2 error is below 0.0005% on square and
rectangular inputs. A standalone C++ process verifies changed-input HIP graph
replay with no Torch or Triton imports. Fixed image tests cover colors, spatial
identification, counting and OCR. Image A–B–A requests and queued requests
preserve sampled log probabilities exactly; cancellation and maximum-size
chunked image prefill pass. These small tests establish the advertised serving
contract, not comprehensive visual reasoning accuracy.

The Q6_K variant received metadata and representative-tensor inspection, not a full-model
release qualification. Quantizing the author's source checkpoint would create
a separately identified derivative; this package uses the native GGUF route.

## Release and rollback

[Release metadata](paiton-release.json) pins the runtime, compiler, plugin,
model and artifact hashes. Startup checks the payload inventory, native ABI,
architecture, runtime versions and complete checkpoint SHA256. Only allowlisted
binaries and sanitized metadata leave the private compiler. The image retains
applicable [notices](THIRD_PARTY_NOTICES.md) and a dependency SBOM.

Published tag: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-neo-coder-max-q4km-rdna4-v1.0.0`.

Immutable digest: `sha256:2a8fed46bc19fe8ca7164139c66f78e80f841120d42112b2968ba060c319fdbe`.

The new tag does not replace existing Qronos or other model releases. To roll
back, stop only `paiton-qwen38-neo` and launch the previously selected model's
existing launcher or immutable image. The NEO cache uses its own named volume;
rollback does not require deleting weights or changing other services.
