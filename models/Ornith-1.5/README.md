# Paiton Ornith 1.5 on Radeon AI PRO R9700

## Model weights and existing downloads

Run these commands from the repository root. The Docker release needs the full
`Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4` checkpoint at revision
`9e488f46c0f7969f84c9923ee0256311cd50316e`, including its `dflash-draft/`
directory. Allow writable space for the prepared shards as described below.

### First download

```bash
./models/Ornith-1.5/serve-docker.sh --chat
```

This downloads missing weights into the `paiton-ornith-cache` Docker volume.
It does not automatically mount your host's Hugging Face cache.

### Already in a local folder

With the [supported native environment](#native-serving), use
`paiton --model-dir /absolute/path/to/ornith serve ornith` for the non-speculative
native preset. To retain the Docker release's DFlash profile, mount the full
standalone checkpoint:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/ornith"
docker run --rm --name paiton-ornith-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/base,readonly" \
  -e PAITON_BASE_MODEL=/models/base -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0
```

Your source stays read-only. The runtime verifies it and keeps its prepared
shards in the Docker volume. Use the cache example for linked Hub snapshots.

### Already in the Hugging Face cache

Mount the entire Hub cache so the snapshot's links to `blobs/` remain valid:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
docker run --rm --name paiton-ornith-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/hf-hub,readonly" \
  -e PAITON_BASE_MODEL=/hf-hub/models--Capicua25x--Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4/snapshots/9e488f46c0f7969f84c9923ee0256311cd50316e \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0
```

Set `HF_HUB_CACHE` to an absolute path on another drive if needed. Both target
and draft files must already be present at the pinned revision. These direct
Docker commands start the API in the foreground; use the [chat commands](#chat)
in a second terminal with the selected container name. [Cache path guide](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve ornith
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve ornith`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve ornith` to prepare without starting the
server, then `paiton --offline serve ornith` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `ornith-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4`, revision `9e488f46c0f7969f84c9923ee0256311cd50316e`.
- **Existing runtime:** Python 3.12, vLLM `0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714`, Torch `2.12.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `ornith` / `ornith-text-8k`.
- **Serving behavior:** Explicit text-only 8K/C1 native compiled model; original source checkpoint unchanged, APC off, no speculation. Lossless cached resharding to fit host RAM. This non-speculative profile does not reproduce the separate DFlash benchmark; the existing DFlash launcher remains available.
- **API:** `http://127.0.0.1:8000/v1`, model name `ornith`. Wait for readiness; `curl http://127.0.0.1:8000/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile ornith-text-8k vllm serve /models/existing-ornith`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


This package runs the public
[`Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4)
checkpoint on one 32 GB Radeon AI PRO R9700.

## Start

For a one-line local chat experience, clone the public package and let the
helper start the server, wait for it, and open the terminal client:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git && cd paiton-vllm-plugin && ./models/Ornith-1.5/serve-docker.sh --chat
```

To start only the OpenAI-compatible server, run the container directly:

```bash
docker run -d \
  --name paiton-ornith \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  --mount type=volume,src=paiton-ornith-cache,dst=/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin:ornith15-mxfp4-rdna4-v1.0.0
```

The first start downloads the pinned 22.9 GB target checkpoint and its 772 MB
DFlash draft from Hugging Face. Paiton losslessly splits the target checkpoint
into smaller shards once. Allow about 48 GB of free disk space for the download
and ready-to-run copy. The Docker volume preserves both for later starts.

Follow startup:

```bash
docker logs -f paiton-ornith
```

## Chat

After a detached server reports that it is ready:

```bash
docker exec -it paiton-ornith paiton-chat --model ornith
```

Use `/reset` to clear the conversation and `/quit` to exit. The server also
provides an OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1/chat/completions`.

## Thinking / reasoning mode

The pinned checkpoint's chat template supports thinking, and the packaged server
uses the `qwen3` reasoning parser. The **terminal client disables thinking by
default**. Enable it per chat session with a larger output budget:

```bash
docker exec -it paiton-ornith paiton-chat --model ornith --thinking --max-tokens 4096
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
  -d '{"model":"ornith","messages":[{"role":"user","content":"Solve 3x + 7 = 22 and check the result."}],"max_tokens":4096,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":true}}'
```

Set `"enable_thinking":false` for direct mode. Reasoning is returned separately
from `message.content` (in `reasoning` or `reasoning_content`, depending on the
runtime); streaming clients should handle the equivalent delta fields. Do not
assume a `reasoning_effort` setting replaces this template toggle.

The token budget includes reasoning **and** the final answer. Keep prompt plus
output within the configured 8,192-token context; reduce the output budget for
long conversations. A budget exhausted during thinking can leave no final answer.
The 4,096-token example is a starting budget, not a quality or completion guarantee.

The published throughput benchmark explicitly used **thinking disabled**. The toggle is available, but thinking-on quality and throughput are not qualified by those results, including the default DFlash configuration.

## Qualified scope

- Radeon AI PRO R9700 with `gfx1201`
- ROCm 7.14
- one GPU, tensor parallel size 1
- batch size 1 and up to 8,192 tokens
- MXFP4 target model with 16-token DFlash speculation
- public checkpoint revision `9e488f46c0f7969f84c9923ee0256311cd50316e`

DFlash is enabled by default. To serve the same Paiton target without
speculation, add `-e PAITON_ORNITH_DFLASH=0` to `docker run`, or export that
variable before using the helper.

The container downloads model weights directly from the public Hugging Face
repository. It contains no checkpoint weights and no Paiton compiler source.

## Performance

The qualified package averaged 44.628 output tokens per second across two
complete runs on one R9700. This is 27.03% faster than the fastest obtained
stock vLLM result of 35.132 output tokens per second. At equal hardware cost,
that corresponds to 27.03% more output tokens per unit of time and a 21.28%
lower hardware cost per generated token.

The [benchmark record](BENCHMARKS.md) documents the stock settings, Paiton
settings, workload, individual results, and reproduction command.
