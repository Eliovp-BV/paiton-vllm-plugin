# Qwen3.8 27B Qronos on RDNA4

## Model weights and existing downloads

Run these examples from the repository root. Use
`amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`, revision
`649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2`. Other Qwen quantizations are separate
packages. The image includes the runtime, not the original checkpoint.

### First download

```bash
./models/Qwen3.8/serve-docker.sh
```

The launcher checks the host's configured Hugging Face cache first. If the
pinned files are absent, it downloads them into its persistent Docker volume.
It prints which location it selected.

### Already in a local folder

Use your complete standalone AMD checkpoint, including tokenizer and processor
files. With the [native environment](#native-serving), use
`paiton --model-dir /absolute/path/to/qronos serve qwen38-qronos`.
For Docker, mount that folder explicitly:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/qronos"
docker run --rm --name paiton-qwen38-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=paiton-qwen38-cache,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/base,readonly" \
  -e PAITON_BASE_MODEL=/models/base -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0
```

The runtime verifies the source and creates its prepared copy in the writable
cache. It preserves your original files. If your folder is a Hub snapshot with
blob links, use the next option to keep those links accessible.

### Already in the Hugging Face cache

The default launcher recognizes `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` and
the legacy `HUGGINGFACE_HUB_CACHE`. Require an existing complete checkpoint:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
./models/Qwen3.8/serve-docker.sh --require-cache
```

For a cache on another drive, replace the export with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`.
The helper mounts the complete Qronos repository cache read-only, preserving its
snapshot links, and disables model downloads. `--dry-run --require-cache`
prints the checked paths and Docker mount command without starting the server.
An incomplete revision stops before Docker starts. [Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve qwen38-qronos
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve qwen38-qronos`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve qwen38-qronos` to prepare without starting the
server, then `paiton --offline serve qwen38-qronos` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `qronos-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`, revision `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2`.
- **Existing runtime:** Python 3.12, vLLM `0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714`, Torch `2.12.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `qwen38-qronos` / `qwen38-qronos-text-8k`.
- **Serving behavior:** Explicit text-only 8K/C1 native compiled model; original source checkpoint unchanged, APC off, no speculation. Released Qronos W4 LM head and serialized external graphs.
- **Runtime conversion:** the released LM-head path performs lossy BF16-to-W4 quantization during loading. The original checkpoint files stay unchanged; this preparation can take several minutes on the host CPU.
- **API:** `http://127.0.0.1:8000/v1`, model name `qwen38`. Wait for readiness; `curl http://127.0.0.1:8000/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile qwen38-qronos-text-8k vllm serve /models/existing-qwen38-qronos`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


This package serves AMD's public
[`Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
checkpoint on one Radeon AI PRO R9700.

## Scope

| Component | Value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700, `gfx1201`, 32 compute units |
| ROCm | 7.14 |
| Model revision | `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` |
| Tensor parallelism | 1 |
| Active batch size | 1 |
| Maximum context | 8,192 tokens |
| Input | Text |

The runtime processes one active request. Additional requests wait in the
server queue. The package is intended for local chat, coding, and generation
by one user.

## Docker

Requirements:

- Linux x86-64
- Docker
- `/dev/kfd` and `/dev/dri` access
- an AMD Radeon AI PRO R9700

Start the server:

```bash
docker run -d \
  --name paiton-qwen38 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --mount type=volume,src=paiton-qwen38-cache,dst=/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0
```

The first start downloads the 19.9 GB model from Hugging Face. The named
volume keeps the download. With cached weights, model assembly takes roughly
10 to 12 minutes.

Follow startup:

```bash
docker logs -f paiton-qwen38
```

The server is ready when the log reports `Application startup complete.`

## Launch helper

The repository includes a helper that checks the GPU, Docker access, model
cache, port, and image:

```bash
./models/Qwen3.8/serve-docker.sh
```

Useful options:

```bash
./models/Qwen3.8/serve-docker.sh --dry-run
./models/Qwen3.8/serve-docker.sh --download
./models/Qwen3.8/serve-docker.sh --require-cache
./models/Qwen3.8/serve-docker.sh --with-hf-token
```

AMD's model repository is public. A Hugging Face token is optional. The
`--with-hf-token` option passes an existing token only when requested.

To use an unpacked checkpoint directory:

```bash
docker run --rm \
  --name paiton-qwen38 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  -e PAITON_BASE_MODEL=/models/base \
  -v /absolute/path/to/amd-qwen38:/models/base:ro \
  -v paiton-qwen38-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1.3.0
```

## Chat

After startup:

```bash
docker exec -it paiton-qwen38 paiton-chat
```

Commands:

- `/reset` clears the conversation
- `/help` shows client help
- `/quit` exits

## Thinking / reasoning mode

The pinned checkpoint's chat template supports thinking, and the packaged server
uses the `qwen3` reasoning parser. The **terminal client disables thinking by
default**. Enable it per chat session with a larger output budget:

```bash
docker exec -it paiton-qwen38 paiton-chat --model qwen38 --thinking --max-tokens 4096
```

Omit `--thinking` for direct answers. No server restart or different checkpoint is
required. The terminal client displays the final answer, not the separate reasoning
stream, so it may stay quiet while the model thinks.

API callers should explicitly choose the mode. The server does not impose the
terminal client's thinking-off default; an omitted setting follows the upstream
chat template. To enable thinking:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Solve 3x + 7 = 22 and check the result."}],"max_tokens":4096,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":true}}'
```

Set `"enable_thinking":false` for direct mode. Reasoning is returned separately
from `message.content` (in `reasoning` or `reasoning_content`, depending on the
runtime); streaming clients should handle the equivalent delta fields. Do not
assume a `reasoning_effort` setting replaces this template toggle.

The token budget includes reasoning **and** the final answer. Keep prompt plus
output within the configured 8,192-token context; reduce the output budget for
long conversations. A budget exhausted during thinking can leave no final answer.
The 4,096-token example is a starting budget, not a quality or completion guarantee.

Thinking availability does not establish a separate reasoning-quality benchmark. Use the direct-mode release measurements only for their documented settings.

## API

The server uses model name `qwen38` and provides an OpenAI-compatible API:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Explain why the sky is blue."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Existing ROCm environment

Local installation requires Python 3.12, ROCm 7.14, PyTorch 2.12 for ROCm
7.14, and vLLM commit
`39bd959b582c85e78e7e0326d49042ce7c3c07ed` built for `gfx1201`.

```bash
python3 -m pip install --no-deps \
  "git+https://github.com/Eliovp-BV/paiton-vllm-plugin.git@paiton-qwen38-qronos-w4a16-gfx1201-v1.3.0"
paiton-qwen38-serve --check-runtime
paiton-qwen38-serve
```

The server downloads the Paiton runtime overlay from the tagged GitHub
release and resolves the original AMD checkpoint from the normal Hugging Face
cache.

## Runtime behavior

The package uses:

- O2 full-and-piecewise graph capture at batch size 1
- model-specific merged projection paths
- AOT decode attention
- tiled recurrent decode
- a compiled W4 language-model head
- fitted MLP decode shadows
- the source checkpoint W4 path for prefill

Fitted decode shadows intentionally change decode quantization. Outputs are
deterministic at temperature zero in the supported configuration, but they are
not specified as bit-identical to the checkpoint's W4 execution path.

The release does not claim support for other GPUs, ROCm versions, larger
batches, tensor parallelism, or multimodal inputs.
