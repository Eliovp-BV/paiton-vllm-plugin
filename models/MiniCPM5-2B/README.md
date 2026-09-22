# MiniCPM5-2B: quick local chat and code on R9700

## Model weights and existing downloads

Choose one option below. Run commands from the repository root. This release
requires `openbmb/MiniCPM5-2B-GPTQ` at revision `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`, including its configuration,
tokenizer and complete weight files. The compiled runtime is included in the
container; the weights are separate.

### First download

The launcher downloads the pinned checkpoint on first use and reuses its own
persistent cache afterward:

```bash
./models/MiniCPM5-2B/serve-docker.sh
```

Use `--download-only` to prepare ahead of time. To reuse an earlier download,
choose one of the following options before starting the container.

### Already in a local folder

In the [supported native environment](#native-serving), point directly to it:

```bash
paiton --model-dir /absolute/path/to/minicpm5 serve minicpm5
```

For Docker, bind your complete standalone checkpoint into the pinned snapshot
location expected by this image. This does not copy or download the weights:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/minicpm5"
docker run --rm --name paiton-minicpm5-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8036:8036 \
  --mount type=volume,src=paiton-minicpm5-local-runtime,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/cache/huggingface/hub/models--openbmb--MiniCPM5-2B-GPTQ/snapshots/6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0 --offline
```

Use the exact checkpoint above. The mount path selects the release's expected
location; it does not make a different model compatible. For a Hub snapshot
containing links to blobs, use the cache option below instead.

### Already in the Hugging Face cache

For a previous `hf download openbmb/MiniCPM5-2B-GPTQ` without `--local-dir`,
mount the **Hub cache root**, keeping snapshots and blobs together. Select your
configured cache, or replace the first line with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
docker run --rm --name paiton-minicpm5-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8036:8036 \
  --mount type=volume,src=paiton-minicpm5-cached-runtime,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/cache/huggingface/hub,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0 --offline
```

The required revision must already be complete. If it is missing, download that
revision on the host with `hf download openbmb/MiniCPM5-2B-GPTQ --revision 6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`,
then retry. `--offline` prevents model downloads inside this container.
These examples start the same API in the foreground. Run one server on this
port at a time. [Cache paths and Docker mounts](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve minicpm5
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve minicpm5`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve minicpm5` to prepare without starting the
server, then `paiton --offline serve minicpm5` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `minicpm5-awq-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `openbmb/MiniCPM5-2B-GPTQ`, revision `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`.
- **Existing runtime:** Python 3.14, vLLM `0.26.1.dev1+g396cd1a43.rocm714`, Torch `2.11.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `minicpm5` / `minicpm5-awq-text-8k`.
- **Serving behavior:** MiniCPM5 asymmetric AWQ G128, FP16 activations, lossless in-memory nibble transpose; upstream prefill and native 1/2-token decode. Explicit 8K context, C2, APC off, reasoning disabled by the released chat-template default, no speculation. Source weights and quantization are unchanged.
- **API:** `http://127.0.0.1:8036/v1`, model name `minicpm5-2b`. Wait for readiness; `curl http://127.0.0.1:8036/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile minicpm5-awq-text-8k vllm serve /models/existing-minicpm5`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


A small Apache-2.0 instruction model, with Paiton-compiled W4A16 decode and an
OpenAI-compatible vLLM API. Qualified target: one Radeon AI PRO R9700, gfx1201,
32 GB VRAM. This separate image preserves every existing community package.

The exact checkpoint is [openbmb/MiniCPM5-2B-GPTQ](https://huggingface.co/openbmb/MiniCPM5-2B-GPTQ/tree/6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805),
revision `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805`. Despite the repository name,
its tensors use **AWQ GEMM packing, asymmetric INT4, group size 128**. Activations
and KV cache use FP16; embeddings and output head are unquantized. It has 2.517B
dense parameters. The original checkpoints remain unchanged.

This is for short chat, concise coding help and lightweight tool requests.
Small-model limitations matter: deterministic direct-answer quality was 14/20
on the retained task set, versus 15/20 for the BF16/FP16 source. Paiton matched
all 20 stock W4 responses, including its failures. Arithmetic, strict filtering,
long reasoning and unfamiliar code need review. This is not a replacement for
a larger reasoning model. [Selection and alternatives](SELECTION.md). [Quantization evidence](QUANTIZATION.md).

## Run

The versioned GHCR image is published. [Compiled artifacts and manifests](https://huggingface.co/EliovpAI/MiniCPM5-2B-W4A16-Paiton-RDNA4/tree/v1.0.0) are also available on Hugging Face; weights download from the pinned upstream checkpoint.

```bash
./serve-docker.sh
```

The server listens on `0.0.0.0:8036`. Downloads are pinned and SHA-256 verified.
Weights and runtime caches persist in `~/.cache/paiton-minicpm5-2b`.
Use `PAITON_CACHE` to choose another directory, or `PAITON_PORT` to change the
published port. Do not run competing GPU workloads during benchmarking.

```bash
./serve-docker.sh --download-only     # download and verify ahead of time
./serve-docker.sh --offline           # reuse the prepared checkpoint
./serve-docker.sh --offline --stock   # same checkpoint, native vLLM projections
python3 chat.py
```

```bash
curl http://localhost:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"max_tokens":256,"temperature":0,"stream":true}'
```

Default: thinking disabled, 8,192-token context, two sequences, 512 scheduled
prefill tokens, 1 GiB FP16 KV cache, decode graphs for batches one and two,
no prefix caching or speculation. The 128K advertised upstream context is not
qualified here. Requests may set `chat_template_kwargs.enable_thinking=true`,
but thinking-on W4 is experimental and excluded from this release qualification. Reasoning can consume the output budget; the fast-chat profile is tested
with thinking off. Reasoning parsing uses `qwen3`, tools use `minicpm5`.

Studio uses the existing chat adapter, pinned image/checkpoint identities,
read-only weight mount, persistent runtime cache and exclusive GPU lease.
A ready model is reused for 120 seconds while eligible; switching releases the
owned model before loading another. The separate Studio change must be installed
with this image; it does not modify an active Studio service.

## Optional thinking mode — experimental for W4

The pinned template supports thinking. This release defaults it **off in both the
server and terminal client**. Direct mode is the qualified fast-chat profile;
thinking-on W4 is available for experimentation, not a qualified reasoning result.

Using the updated repository client against the existing published container:

```bash
python3 models/MiniCPM5-2B/chat.py --thinking --max-tokens 4096 --timeout 600
```

Run that command from the repository root. The extra budget/timeout flags belong
to the updated host script; the immutable v1.0.0 image still contains its original
client. No image rebuild is required to send these API settings. Omit `--thinking`
for direct answers. The client displays final-answer text, so it may stay quiet
while reasoning is generated.

The same setting works directly through the API:

```bash
curl --fail http://127.0.0.1:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Solve 3x + 7 = 22 and check the result."}],"max_tokens":4096,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}'
```

Set `"enable_thinking":false` to turn it off. The parser separates reasoning from
`message.content`; API clients can inspect `reasoning`/`reasoning_content` and the
corresponding streaming delta fields. Reasoning and the final answer share the
output budget, and prompt plus output must fit the tested 8,192-token context.
Reduce the output budget as conversation history grows. Thinking may repeat or
consume the budget before a useful answer; 4,096 tokens does not guarantee completion.
The published 14/20 quality result and loading/speed figures use thinking **off**.

## Reproduce

[Loading and generation measurements](BENCHMARKS.md) include cache boundaries,
repeats, quality results and rejected approaches. [Build and benchmark commands](REPRODUCE.md)
include stock/Paiton selection. [Licenses and provenance](THIRD_PARTY_NOTICES.md)
cover redistribution and the publisher's quantization metadata limitations.
