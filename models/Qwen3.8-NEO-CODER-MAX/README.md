# Qwen3.8 NEO CODER MAX on RDNA4

## Model weights and existing downloads

Run these commands from the repository root. This release requires both the
selected **Q4_K_M language GGUF** and **`mmproj-BF16.gguf`** from revision
`89230607b3708bc1efe174e8fd34d4b164476c88` of the
[author's GGUF repository](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/tree/89230607b3708bc1efe174e8fd34d4b164476c88).
The image supplies the matching tokenizer and runtime configuration. A different
GGUF quantization is not interchangeable.

### First download

```bash
./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

This downloads the two pinned files into its named Docker volume. It does not
search the host's Hub cache automatically.

### Already in a local folder

Select the directory containing both files, then mount it read-only:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/neo-gguf"
docker run --rm --name paiton-qwen38-neo-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=paiton-qwen38-neo-cache,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/source,readonly" \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:534287969135f581744ae481b578599468b0bf7ac9a4051b0941500e4c18da4d \
  --checkpoint /models/source/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf \
  --projector /models/source/mmproj-BF16.gguf
```

The two files are hash-verified by the runtime. For linked Hub snapshots,
use the cache mount below instead of mounting just the snapshot directory.
The [native CLI](#native-serving) also supports an explicitly selected complete
local model directory with `paiton --model-dir /absolute/path/to/neo-gguf serve qwen38-neo`.

### Already in the Hugging Face cache

Mount your Hub root, then tell this image to resolve both pinned files there:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
docker run --rm --name paiton-qwen38-neo-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=paiton-qwen38-neo-cache,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/hf-hub,readonly" \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:534287969135f581744ae481b578599468b0bf7ac9a4051b0941500e4c18da4d \
  --cache-dir /hf-hub
```

For another disk, set `HF_HUB_CACHE="/absolute/path/to/your/hub-cache"` first.
Keep both files and their blobs at the pinned revision. The writable runtime
volume is separate from the read-only model cache. `PAITON_NEO_CACHE` in the
ordinary launcher names a **Docker volume**, not a host model directory.
[Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve qwen38-neo
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve qwen38-neo`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve qwen38-neo` to prepare without starting the
server, then `paiton --offline serve qwen38-neo` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `neo-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`, revision `89230607b3708bc1efe174e8fd34d4b164476c88`.
- **Existing runtime:** Python 3.12, vLLM `0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714`, Torch `2.12.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `qwen38-neo` / `qwen38-neo-image-8k`.
- **Serving behavior:** Explicit original mixed Q4_K_M GGUF, native FP32 activations and BF16 weight metadata; text plus one image (max 1,048,576 pixels), 8K/C1, 2048-token prefill, 2 GiB KV, APC and MTP off. Released native internal decode graphs; vLLM eager execution.
- **API:** `http://127.0.0.1:8000/v1`, model name `qwen38-neo`. Wait for readiness; `curl http://127.0.0.1:8000/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile qwen38-neo-image-8k vllm serve /models/existing-qwen38-neo`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


**GGUF weights, vLLM serving, native Paiton execution.** This release runs the
DavidAU Qwen3.8 NEO CODER MAX mixed Q4_K_M fine-tune through vLLM's actual model
loader, scheduler, sampling and streaming API, with native HIP language and
image execution. llama.cpp is a separate benchmark reference, not a dependency
of this serving path.

Against fresh working llama.cpp measurements, complete streaming request medians
are **6.4%, 5.1% and 0.8% lower** at 128, 1,024 and 4,096 input tokens. This is
near parity on the long workload; some prefill-only cases still favor llama.cpp.
[Full results and arithmetic distinctions](BENCHMARKS.md) ·
[Why native GGUF through vLLM matters](NATIVE_GGUF.md).

Support is qualified for this pinned model and quantization, including one
PNG/JPEG image. This does not imply support for every GGUF architecture or file.

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
accumulate in FP32. Large Q4 decode projections use the qualified native Q8_1 activation
profile; small Q4 and Q6 operations retain floating arithmetic. Small coefficients, activations and recurrent state remain FP32;
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
| Context | 8,192 total tokens; prefill chunks up to 2,048 tokens |
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

The final prefill sweep compares 31,784,960 full-vocabulary logits exactly
against the preceding qualified arithmetic profile. All 36 text API requests
preserve 2,322 token events and log probabilities; all 24 image requests preserve
streamed text and usage. Native graph replay, live state, request isolation,
cancellation, queueing and the 8,192-token boundary pass.

The fixed suite remains 10/11 in both engines, with the same code-trace failure;
five PNG fixtures and JPEG pass. Prefill held-out perplexity remains
6.86244447185. The inherited Q8 decode evaluation is +0.1301% perplexity versus
the older FP32 decode profile over 770 held-out predictions, below the preset
1% bound. These narrow tests are not a broad accuracy claim. Original packed
weights are retained; v1.1.0 is not claimed bit-equivalent to the older FP32
activation profile or to llama.cpp's different activation arithmetic.

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
applicable [notices](THIRD_PARTY_NOTICES.md) and a dependency SBOM. [Publication checks](PUBLICATION_CHECKLIST.md) and
[evidence hashes](qualification-evidence.json) record the qualification.

Published tag: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-neo-coder-max-q4km-rdna4-v1.1.0`.

Immutable digest: `sha256:534287969135f581744ae481b578599468b0bf7ac9a4051b0941500e4c18da4d`.

The new tag preserves v1.0.0, Qronos and other model releases. To roll back,
stop only the NEO container and select the previous immutable image:

```bash
docker stop paiton-qwen38-neo
PAITON_NEO_IMAGE=ghcr.io/eliovp/paiton-vllm-plugin@sha256:2a8fed46bc19fe8ca7164139c66f78e80f841120d42112b2968ba060c319fdbe \
  ./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh
```

The launcher uses `--rm`; if a separately created stopped container retains the
same name, remove only that stopped NEO container before relaunching. The named
cache volume is retained.
