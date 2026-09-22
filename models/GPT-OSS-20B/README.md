# GPT-OSS-20B on Radeon AI PRO R9700

## Model weights and existing downloads

Choose one option below. Run commands from the repository root. This release
requires `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, including its configuration,
tokenizer and complete weight files. The compiled runtime is included in the
container; the weights are separate.

### First download

The launcher downloads the pinned checkpoint on first use and reuses its own
persistent cache afterward:

```bash
./models/GPT-OSS-20B/serve-docker.sh
```

Use `--download-only` to prepare ahead of time. To reuse an earlier download,
choose one of the following options before starting the container.

### Already in a local folder

In the [supported native environment](#native-serving), point directly to it:

```bash
paiton --model-dir /absolute/path/to/gpt-oss serve gpt-oss-20b
```

For Docker, bind your complete standalone checkpoint into the pinned snapshot
location expected by this image. This does not copy or download the weights:

```bash
export PAITON_MODEL_DIR="/absolute/path/to/gpt-oss"
docker run --rm --name paiton-gpt-oss-local \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8020:8020 \
  --mount type=volume,src=paiton-gpt-oss-local-runtime,dst=/models/cache \
  --mount "type=bind,src=$PAITON_MODEL_DIR,dst=/models/cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:gpt-oss-20b-mxfp4-rdna4-v1.0.0 --offline
```

Use the exact checkpoint above. The mount path selects the release's expected
location; it does not make a different model compatible. For a Hub snapshot
containing links to blobs, use the cache option below instead.

### Already in the Hugging Face cache

For a previous `hf download openai/gpt-oss-20b` without `--local-dir`,
mount the **Hub cache root**, keeping snapshots and blobs together. Select your
configured cache, or replace the first line with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`:

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
docker run --rm --name paiton-gpt-oss-cached \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p 127.0.0.1:8020:8020 \
  --mount type=volume,src=paiton-gpt-oss-cached-runtime,dst=/models/cache \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/models/cache/huggingface/hub,readonly" \
  ghcr.io/eliovp/paiton-vllm-plugin:gpt-oss-20b-mxfp4-rdna4-v1.0.0 --offline
```

The required revision must already be complete. If it is missing, download that
revision on the host with `hf download openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee`,
then retry. `--offline` prevents model downloads inside this container.
These examples start the same API in the foreground. Run one server on this
port at a time. [Cache paths and Docker mounts](../../docs/MODEL_WEIGHTS.md).

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve gpt-oss-20b
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve gpt-oss-20b`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve gpt-oss-20b` to prepare without starting the
server, then `paiton --offline serve gpt-oss-20b` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `gptoss-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
- **Existing runtime:** Python 3.14, vLLM `0.26.1.dev1+g396cd1a43.rocm714`, Torch `2.11.0+rocm7.14.0`, ROCm SDK 7.14.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `gpt-oss-20b` / `gptoss-text-8k`.
- **Serving behavior:** Explicit released C2 text profile, APC off, no speculation, graph sizes 1/2. Original checkpoint quantization unchanged; native expert decode with existing upstream prefill and attention. 8K context, Harmony reasoning/tools, bundled pinned Harmony vocabulary.
- **API:** `http://127.0.0.1:8020/v1`, model name `gpt-oss-20b`. Wait for readiness; `curl http://127.0.0.1:8020/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile gptoss-text-8k vllm serve /models/existing-gpt-oss-20b`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


Free community release: original OpenAI MXFP4 weights with Paiton
expert decode kernels, served through vLLM's OpenAI-compatible API. One R9700
(gfx1201), 32 GB VRAM.

**2.18× speedup — 54% lower end-to-end latency** in the tested generation
scenario, using the original checkpoint. Stock already uses **O2 and full decode
graph capture**. [Compiled artifacts on Hugging Face](https://huggingface.co/EliovpAI/GPT-OSS-20B-MXFP4-Paiton-RDNA4).

The model is pinned to `6cee5e81ee83917806bbde320786a8fb61efebee`.
It has approximately 21B total / 3.6B active parameters. All experts remain on
the GPU. This package does not requantize the checkpoint.

## Measured performance

Primary scenario: 512 input / 256 output tokens, one request at a time, original
MXFP4 and BF16 KV. Paiton measured 2.214 s median over 32 requests.

| Configuration | Median end-to-end latency |
|---|---:|
| Fastest earlier stock vLLM reference | 4.819 s |
| Corrected release image, stock | 6.110 s |
| Corrected release image, Paiton | 2.214 s |

The conservative comparison is **54.0% lower latency** against the faster stock
reference. The stock timing shift also occurs on the original developer runtime;
its cause is not established. Output throughput includes explicitly counted
reasoning tokens. See [all measurements and limitations](BENCHMARKS.md).

## Launch

Run from this model directory:

```bash
./serve-docker.sh
```

`./serve-docker.sh` uses `ghcr.io/eliovp/paiton-vllm-plugin:gpt-oss-20b-mxfp4-rdna4-v1.0.0`.
The first launch
downloads only the pinned model files. Model, Triton and vLLM
caches persist in `~/.cache/paiton-gpt-oss`. The pinned Harmony vocabulary is
embedded in the image, so cached-model offline launches also support chat. The API listens on `0.0.0.0:8020`
inside the container and is mapped to port 8020 on the host. Keep access limited
to your intended clients; this launch has no authentication configured.

```bash
# Same checkpoint and serving settings, stock expert implementation:
PAITON_CONTAINER=paiton-gpt-oss-stock ./serve-docker.sh --stock

# Reuse an existing Hugging Face cache without modifying it:
PAITON_HF_CACHE="$HOME/.cache/huggingface" ./serve-docker.sh --offline

# Terminal client (Python standard library only):
python3 chat.py
python3 chat.py 'Explain binary search in two sentences.'

curl http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"What is 17 + 25?"}],"reasoning_effort":"low","max_tokens":512,"temperature":0,"stream":false}'
```

For streamed output and schema-constrained JSON:

```bash
curl --no-buffer http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"reasoning_effort":"low","max_tokens":256,"temperature":0,"stream":true}'

curl http://localhost:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Return 17 plus 25 as JSON with integer field answer."}],"reasoning_effort":"low","max_tokens":256,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"sum","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}}'
```

Stop the existing instance before switching modes. Override `PAITON_IMAGE`,
`PAITON_PORT`, `PAITON_CONTAINER` or `PAITON_CACHE` as needed.

## Resources

Tested on a Linux host with 15.52 GiB usable RAM and 4 GiB swap. Whole-host swap
usage reached about 3 GiB during qualification; a swap-free 16 GB deployment is
not qualified. Reserve roughly 70 GB of free disk for the downloaded container,
its extracted runtime, 13.8 GB of model tensor data and persistent caches. Building
from the full developer base needs substantially more temporary disk than running
the compact image.

At the tested settings, sampled driver VRAM peaked at about 17.1 GiB stock and
17.0 GiB Paiton. Torch peak allocated memory was about 16.3 GiB and peak reserved
memory 16.4 GiB. These include all experts, padded weights, scales, nonquantized
weights, workspaces, graphs and the 2 GiB KV allocation. Original checkpoint
tensors occupy 12.82 GiB; the padded loaded model occupies about 14.17 GiB.

One startup with cached model weights and empty compilation caches took 155 s
stock and 142 s Paiton. These are individual startup observations, not a startup
speedup claim; initial download time is additional and network-dependent.

## Scope and limits

The release configuration uses 8,192 tokens **including prompt and generated
reasoning/answer**, at most two scheduled requests, a 2 GiB BF16 KV cache,
512-token prefill chunks, and graph capture for decode batches one and two.
Longer prompts and larger batches use the stock expert fallback. The advertised
131,072-token architecture limit is not this package's tested serving limit.

Attention, positional encoding, sinks, sliding windows, top-4 routing, sampling,
Harmony formatting and parsers remain in vLLM. Paiton replaces only supported
one/two-token expert computation. Low-bit storage is distinct from arithmetic:
Paiton unpacks original MXFP4 and accumulates in FP32, preserving the BF16
activation and weighted-partial rounding boundaries. KV cache stays BF16.

Reasoning effort `low`, `medium` and `high`, streaming, tool calls and JSON-schema
output are covered by local checks. Reasoning consumes the output token budget;
a small budget can finish before any answer. A strict local coding prompt can
produce a malformed `final code` channel even in stock, leaving no final answer.
An unconstrained JSON-only instruction can produce Markdown fences; request a
JSON schema when machine-readable output is required. This is a small regression
suite, not a claim of general quality parity.

This entry qualifies one GPU, no LoRA, no speculative decoding, and no
multi-GPU serving. It does not change clock, power, fan or voltage settings.

See [selection](SELECTION.md), [measured results](BENCHMARKS.md),
[reproduction](REPRODUCE.md), and [licenses/provenance](NOTICE.md).

For an immutable image reference, set `PAITON_IMAGE` to
`ghcr.io/eliovp/paiton-vllm-plugin@sha256:0cf9c11304ade58e91d97db876a4462781fa2c18d2d4846f26d56fe5d207669d`.
To rebuild locally with the compiled overlay from Hugging Face, see
[REPRODUCE.md](REPRODUCE.md).
