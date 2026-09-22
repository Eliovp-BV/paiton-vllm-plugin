# Qwen3-Coder 30B for local coding on R9700

## Model weights and existing downloads

Choose one option below. Run commands from the repository root. This release
requires `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` at revision `4bd30395b72ea6045edd04806c4fea448d4467b3`, including its configuration,
tokenizer and complete weight files. The compiled runtime is included in the
container; the weights are separate.

### First download

The launcher downloads the pinned checkpoint on first use and reuses its own
persistent cache afterward:

```bash
./models/Qwen3-Coder-30B/serve-docker.sh
```

Use `--download-only` to prepare ahead of time. To reuse an earlier download,
choose one of the following options before starting the container.

### Already in a local folder

In the [supported native environment](#native-serving), point directly to it:

```bash
paiton --model-dir /absolute/path/to/qwen3-coder serve qwen3-coder
```

For Docker, bind your complete standalone checkpoint into the pinned snapshot
location expected by this image. This does not copy or download the weights:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/qwen3-coder"
docker run --rm --name paiton-qwen3-coder-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8010:8010 \
  --mount type=volume,src=paiton-qwen3-coder-local-runtime,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/cache/huggingface/hub/models--cyankiwi--Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/snapshots/4bd30395b72ea6045edd04806c4fea448d4467b3,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen3-coder-30b-awq-rdna4-v1.0.0 --offline
```

Use the exact checkpoint above. The mount path selects the release's expected
location; it does not make a different model compatible. For a Hub snapshot
containing links to blobs, use the cache option below instead.

### Already in the Hugging Face cache

For a previous `hf download cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` without `--local-dir`,
mount the **Hub cache root**, keeping snapshots and blobs together. Select your
configured cache, or replace the first line with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
docker run --rm --name paiton-qwen3-coder-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8010:8010 \
  --mount type=volume,src=paiton-qwen3-coder-cached-runtime,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/cache/huggingface/hub,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen3-coder-30b-awq-rdna4-v1.0.0 --offline
```

The required revision must already be complete. If it is missing, download that
revision on the host with `hf download cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit --revision 4bd30395b72ea6045edd04806c4fea448d4467b3`,
then retry. `--offline` prevents model downloads inside this container.
These examples start the same API in the foreground. Run one server on this
port at a time. [Cache paths and Docker mounts](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve qwen3-coder
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve qwen3-coder`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve qwen3-coder` to prepare without starting the
server, then `paiton --offline serve qwen3-coder` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `coder-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit`, revision `4bd30395b72ea6045edd04806c4fea448d4467b3`.
- **Existing runtime:** Python 3.12, vLLM `0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714`, Torch `2.12.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `qwen3-coder` / `qwen3-coder-text-4k`.
- **Serving behavior:** Explicit released C2 text profile, APC off, no speculation, graph sizes 1/2. Original checkpoint quantization unchanged; native expert decode with existing upstream prefill and attention. 4K context, qwen3_xml tools.
- **API:** `http://127.0.0.1:8010/v1`, model name `qwen3-coder`. Wait for readiness; `curl http://127.0.0.1:8010/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile qwen3-coder-text-4k vllm serve /models/existing-qwen3-coder`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


Run Qwen3-Coder-30B-A3B-Instruct on one **Radeon AI PRO R9700, 32 GB**, with
terminal chat and an OpenAI-compatible coding API. The prebuilt container
includes Paiton and the compiled artifact; no compiler or local build is needed.

**[Download the ready-to-run bundle](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/qwen3-coder-30b-awq-rdna4-v1.0.0/paiton-qwen3-coder-r9700-v1.0.0.tar.gz)** ·
[Checksums and release files](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/tag/qwen3-coder-30b-awq-rdna4-v1.0.0) ·
[Hugging Face artifacts and launcher](https://huggingface.co/EliovpAI/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit-Paiton-RDNA4) ·
[Measured performance and quality](BENCHMARKS.md)

## Start coding

You need Linux, Docker, a working AMD GPU driver and the R9700. From a new
checkout, this starts the server and opens terminal chat:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/Qwen3-Coder-30B/serve-docker.sh --chat
```

Or download and extract the bundle, then run `./run.sh --chat` inside its
folder. The launcher pulls the prebuilt image and downloads the pinned
18.1 GB INT4 checkpoint on first use. Later launches reuse the cache. Enter a
coding question, use `/reset` for a new conversation, or `/quit` to leave chat.
The API continues running after chat closes.

You can also download the launcher and compiled artifacts from Hugging Face:

```bash
hf download EliovpAI/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit-Paiton-RDNA4 \
  --revision v1.0.0 --local-dir ./paiton-qwen3-coder
bash ./paiton-qwen3-coder/serve-docker.sh --chat
```

The Hub package contains artifacts, manifests, checksums, the launcher and
documentation only. Weights are downloaded directly from the pinned cyankiwi
checkpoint into your local cache. The container already includes the same
compiled artifact; see the Hub model card for optional external artifact use.

To start only the API, omit `--chat`. Check progress with
`docker logs -f paiton-qwen3-coder`; stop it with
`docker stop paiton-qwen3-coder`. Running the launcher again restarts the same
container. It reports a conflict if the existing container uses another image,
mode, port or cache, instead of silently serving the wrong setup.

For example: “Write a Python function that merges overlapping intervals,
without mutating its input. Include tests for empty and touching intervals.”

## Connect a coding client

Choose the client's **OpenAI-compatible** provider and enter:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8010/v1` |
| Model | `qwen3-coder` |
| API key, if the client requires a value | `local` |
| Context budget | `4096` tokens, including output |
| Suggested maximum output | `1024` tokens |

The local endpoint does not require authentication and is bound to localhost.
Chat completions, streaming and automatic function tool calls are enabled with
vLLM's `qwen3_xml` parser. This is a 4096-token release: use focused functions,
files and coding questions. Whole-repository agent workloads and long-context
quality have not been qualified. No particular editor extension is certified.

A direct API request:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"Write a Python binary search with tests."}],"max_tokens":1024,"temperature":0}'
```

Run the included API and tool-call check with host Python 3:

```bash
python3 models/Qwen3-Coder-30B/check-api.py
```

A client on another machine can use an SSH tunnel:
`ssh -L 8010:127.0.0.1:8010 your-gpu-host`, then the same local base URL.

## Cache and launch options

Weights and runtime caches persist under `~/.cache/paiton`; set
`PAITON_CACHE_DIR` to change that directory. Existing Hugging Face downloads
can be reused when `$PAITON_CACHE_DIR/huggingface` contains their `hub/` cache.
For an existing `~/.cache/huggingface` snapshot, use
`PAITON_CACHE_DIR="$HOME/.cache" ./models/Qwen3-Coder-30B/serve-docker.sh --offline`.
Only the cache is mounted; your source repository is not sent to the model
unless your coding client includes it in a request.

| Option | Behavior |
| --- | --- |
| `--chat` | Wait for startup, then open terminal chat |
| `--download-only` | Download/cache the pinned checkpoint without loading the GPU |
| `--offline` | Use the cached checkpoint; no model download |
| `--stock` | Use stock vLLM for the matched comparison |

`PAITON_HOST_PORT` changes the host API port; `PAITON_CONTAINER_NAME` selects
another container name. Stop the active server before switching stock/Paiton
on this single GPU. `PAITON_QWEN3_CODER_IMAGE` overrides the image for a local
build or a digest pin. The versioned release image is:

`ghcr.io/eliovp/paiton-vllm-plugin:qwen3-coder-30b-awq-rdna4-v1.0.0`

## Requirements, model and supported scope

The qualification used 16 GB host RAM with 4 GB swap, an i5-8400, Linux kernel
6.17.0-1028-oem and AMD driver 6.19.14.31400000. Allow **30 GB free disk minimum**
for the checkpoint, roughly 5.1 GB image and caches; 40 GB gives more build room.
Lazy safetensors loading allows the 18.1 GB checkpoint to load on this host.
Observed loading was about 46 seconds; cached startup takes roughly two minutes,
with additional time for fresh graph/compilation caches. First-use downloads
are separate. Warm GPU usage peaked near **20.1 GiB**, with full GPU residency.

Pinned model: [cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tree/4bd30395b72ea6045edd04806c4fea448d4467b3),
revision `4bd30395b72ea6045edd04806c4fea448d4467b3`: compressed-tensors symmetric
INT4 groups of 32, BF16 activations. It quantizes the official
[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct).
Both model cards declare Apache-2.0. The quantizer does not pin its upstream
weight revision; this exact quantized snapshot is pinned for both engines.
Qwen3-Coder is a non-thinking model.

Runtime: HIP 7.14.60850, PyTorch 2.12.0+rocm7.14.0, Transformers 5.15.1 and
vLLM commit `39bd959b582c85e78e7e0326d49042ce7c3c07ed` in the pinned base image.
Relevant stock source files were audited against that upstream commit; the
image also carries unrelated Quark support patches. Both modes use the same
4096 context cap, two sequence slots, 512 batched tokens, 2 GiB BF16 KV pool,
ROCM_ATTN, O2 graph sizes 1/2, and disabled prefix caching.

Measured gains were **21.3% at concurrency 1** and **70.1% at concurrency 2**
against the fastest stock configuration tested on the fixed short coding
workload. Both engines passed four executable coding checks and scored 7/8
on the small quality suite. Numerical and text differences remain. See
[BENCHMARKS.md](BENCHMARKS.md) for latency, counts, raw-data scope and limits.
No LoRA, expert parallelism, speculation, other GPU or long-context performance
is qualified. No power, clock, fan or voltage changes are required.

## Reproduce benchmarks or rebuild

[REPRODUCE.md](REPRODUCE.md) contains the benchmark commands and offline
container recipe. The downloadable bundle includes the compiled artifact,
manifest, licensed AMD SMI runtime, public runtime source and model instructions.
Private compiler source and model weights are excluded from the bundle/image.

Model license: [QWEN_LICENSE](QWEN_LICENSE). Runtime and component notices:
[NOTICE.md](NOTICE.md), repository [LICENSE](../../LICENSE), and
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
