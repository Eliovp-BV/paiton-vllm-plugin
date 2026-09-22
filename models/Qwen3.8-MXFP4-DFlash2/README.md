# Qwen3.8 MXFP4 + DFlash2 on regular vLLM

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve qwen38-nvfp4
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve qwen38-nvfp4`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve qwen38-nvfp4` to prepare without starting the
server, then `paiton --offline serve qwen38-nvfp4` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `qwen38-rocm10-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`.
- **Existing runtime:** Python 3.12, vLLM `0.29.0`, Torch `2.12.0+rocm10.0.0`, ROCm SDK 10.0.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `qwen38-nvfp4` / `qwen38-nvfp4-w4a8-text-65k`.
- **Serving behavior:** Explicit text-only NVFP4→MXFP4 requantization, FP8 activations, FP8 KV, FP16 recurrent cache, 65K context, APC off, no speculation. Original weights stay unchanged; conversion is lossy and occurs only in device memory at model load. Existing upstream warmup/graph compilation remains.
- **API:** `http://127.0.0.1:18982/v1`, model name `Qwen3.8`. Wait for readiness; `curl http://127.0.0.1:18982/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile qwen38-nvfp4-w4a8-text-65k vllm serve /models/existing-qwen38-nvfp4`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


## Current release

**20 September 2026: updated 65K image, ROCm 10 and vLLM 0.29.0.**
The combined Paiton target and DFlash2 runtime measured **154.42 tok/s weighted
decode**, **218.1 tok/s median JSON decode**, and **421.20 tok/s aggregate at
concurrency eight**. Weighted decode improved **5.16%** against a fresh run of
the released image; large-prefill throughput improved **4.37–5.61%**.

| Image default | Total context limit | Maximum scheduled requests | Image |
| --- | ---: | ---: | --- |
| Updated 65K | 65,536 tokens | 8 | `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260920-r2` |
| Existing 200K | 200,000 tokens | 1 | [18 September 200K package](https://github.com/users/Eliovp/packages/container/paiton-vllm-plugin/1266757900) |

Both image defaults use FP8 KV caching and disable automatic prefix caching (APC).
APC remains an optional launcher setting. The existing 200K `chat` profile below
is the separately validated long-conversation configuration; its image has not
been replaced by this 65K throughput update.
The launchers pin images by digest under `ghcr.io/eliovp/paiton-vllm-plugin`.
No registry login is required.

**Context is configurable at startup.** The image names select defaults;
`--context` changes the target and draft limits without rebuilding. Available
VRAM and the supported model range still determine what fits. The performance
results below use the unchanged 65,536-token benchmark profile.

## Run the current release

Use Linux x86-64, Python 3, Docker, and one Radeon AI PRO R9700 with 32 GB VRAM and working
AMD GPU device access. Run the following commands from the repository root.
The images contain the runtime; download the target and draft weights separately
using the Hugging Face CLI (`hf`), or point the variables at existing copies of
these exact snapshots.

The main setup below uses the **updated 65K image with APC off**, matching the
benchmark configuration. The optional long-context profile remains available below.

This release loads the **Unsloth NVFP4 checkpoint through the MXFP4 runtime path**.
The AMD checkpoint and automatic downloader in the historical release below
are for the older images.

## Model weights and existing downloads

Choose **one** of the following paths. The current container needs both:

| Component | Repository | Required revision |
| --- | --- | --- |
| Target | `unsloth/Qwen3.8-27B-NVFP4` | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` |
| DFlash2 draft | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f` |

The native `paiton serve qwen38-nvfp4` preset above is non-speculative. The
instructions here preserve the Docker release's **DFlash2** profile.

### First download

Run this complete block from the repository root. The `hf download` commands
populate the model directories; the exports and `mkdir` commands alone do not.
Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
if `hf` is not available.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-65k-20260920"
mkdir -p "$PAITON_TARGET_DIR" "$PAITON_DRAFT_DIR" "$PAITON_CACHE_DIR"

hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --local-dir "$PAITON_TARGET_DIR"
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
  --revision ee0cb26a8279b7910cc28d82a8a3e15e4728d56f \
  --local-dir "$PAITON_DRAFT_DIR"
```

After both downloads complete, use [Launch with prepared local folders](#launch-with-prepared-local-folders).
If either download fails, resolve that failure before starting the server.

### Already in a local folder

Point to your complete **standalone** target and draft directories. These can
be folders created with `hf download --local-dir` or complete local copies of
the same pinned files. Do not point to the parent directory holding both models.

```bash
export PAITON_TARGET_DIR="/absolute/path/to/qwen38-nvfp4"
export PAITON_DRAFT_DIR="/absolute/path/to/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-65k-20260920"
mkdir -p "$PAITON_CACHE_DIR"
```

Each model directory must contain its own `config.json` and complete weight
files; keep the target tokenizer too. The launcher mounts these directories
read-only as `/models/target` and `/models/draft`. It uses `/cache` for the
separate writable runtime cache. Then [launch below](#launch-with-prepared-local-folders).

### Already in the Hugging Face cache

If you previously ran `hf download` without `--local-dir`, select the Hub cache
that contains both pinned snapshots. The following **65K Docker command** mounts
the entire cache read-only and selects the snapshots inside it, preserving their
links to `blobs/`. It uses the same image and inference settings as the default
65K launcher; only the weight paths differ.

For a cache on another drive, replace the first export with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`.

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-65k-20260920"
mkdir -p "$PAITON_CACHE_DIR"

docker run --rm --name paiton-qwen38-65k-cached --network host \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/hf-hub,readonly" \
  --mount "type=bind,src=$PAITON_CACHE_DIR,dst=/cache" \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260920-r2@sha256:791c09ec96626fcd33fa873d3e92dca208bb43b513317a00d7584ec730b4dabd \
  serve /hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --tokenizer /hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --served-model-name Qwen3.8 \
  --host 127.0.0.1 \
  --port 18982 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 6535819798 \
  --gpu-memory-utilization 0.98 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --language-model-only \
  --safetensors-load-strategy lazy \
  --attention-backend R4D \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true}}' \
  --speculative-config '{"method":"dflash","model":"/hf-hub/models--tcclaviger--Qwen3.8-27B-DFlash2-FP8/snapshots/ee0cb26a8279b7910cc28d82a8a3e15e4728d56f","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"TRITON_ATTN","max_model_len":65536,"disable_padded_drafter_batch":true,"draft_sample_method":"greedy"}' \
  --mamba-cache-mode align \
  --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.95, "top_k": 20}' \
  --seed 42
```

Both snapshots must be complete; a different cached revision is not selected
implicitly. If one is missing, run the corresponding pinned `hf download`
command from [First download](#first-download) **without `--local-dir`** to fill
your configured Hub cache, then retry this command. Do not mount only a linked
snapshot at `/models/target` or `/models/draft`: that can break its blob links.
[Cache layout and path guide](../../docs/MODEL_WEIGHTS.md).

### Launch with prepared local folders

After completing **First download** or **Already in a local folder**, run:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh
```

The current launcher checks that directories exist, but does not check that
checkpoint files are present. An error about `/models/target` and `config.json`
means the mounted target is missing, incomplete, or has inaccessible links.
Check the selected host directory and both downloads before retrying. The Hub
cache option above launches Docker directly and does not need this extra step.

The server runs in the foreground at `http://127.0.0.1:18982/v1`, with API model
name **`Qwen3.8`**. First startup loads/converts weights and compiles runtime
components; wait for readiness before sending requests or measuring throughput.
The persistent cache is reused on subsequent starts. The first image pull is
approximately 9.6 GB, excluding model weights.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function that removes duplicate items while preserving order."}],"temperature":0.7,"top_p":0.95,"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

For the separately validated 200K chat configuration, use the **standalone
local-folder** setup above. Stop your 65K container (`paiton-qwen38-65k` for the
launcher, or `paiton-qwen38-65k-cached` for the direct Hub-cache example), keep
`PAITON_TARGET_DIR` and `PAITON_DRAFT_DIR` pointing to the standalone folders,
and select the existing long-context image. The direct Hub-cache command above
is specifically the 65K profile:

```bash
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-200k"
mkdir -p "$PAITON_CACHE_DIR"
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-200k.sh --profile chat --context 200000
```

Select one profile at a time on the GPU. The 200K image passed a **198,989-token
prompt** followed by generation, a fresh short request, and both ordinary and
streaming XML tool calls. The performance tables below are for the **65K profile**;
they are not measurements of 200K throughput.

No private compiler checkout is needed to run these images. The full qualified
runtime payload is distributed in the images; the public repository alone is
not a complete build context for this release.

## GPU, context, and memory controls

The 65K and 200K names select startup presets. Context is **not compiled into the
image**: the launchers can set both the target and DFlash draft limits at startup,
using the existing images. The limit includes input and generated tokens. A larger
limit still requires sufficient cache and VRAM; it does not guarantee useful
model quality at that length.
The checkpoint's configured ceiling is 262,144 tokens; the largest serving
limit tested here is **220,000**, not a claim that 262K fits this GPU.

The launcher exposes `/dev/kfd` and all of `/dev/dri`, adds the `video` group,
and uses host IPC. Select the GPU with your usual ROCm environment variables.
For example, if the R9700 is ROCm GPU 1:

```bash
export ROCR_VISIBLE_DEVICES=1
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-200k.sh --profile chat
```

Use the index or GPU UUID appropriate to your system. `--list-gpus` lists physical
render devices and PCI addresses for identification; render-device numbers are
not ROCm visibility indices. The launcher does not choose a GPU automatically.

The launcher forwards your `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` values unchanged. If a variable is unset on the host, it
is also unset in the container, overriding the image's default. The launcher
does not force a GPU index or UUID. An explicitly empty mask remains empty;
it does not mean "show all GPUs".

On Linux, `ROCR_VISIBLE_DEVICES` also filters ROCr tools such as `rocminfo`;
`HIP_VISIBLE_DEVICES` applies at the HIP layer. If you set both, HIP indices refer
to the GPUs remaining after the ROCr filter. Check any existing exports before
launching: setting both variables to `1` does not necessarily select the second
physical card. Leave unused variables unset rather than assigning empty strings.
These controls do not make unsupported GPU architectures compatible with this image.

For a GPU shared with a desktop, start with the smaller preset:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh \
  --profile desktop
```

This selects 32,768 tokens, one scheduled request, 1,024-token prefill chunks,
smaller graph captures, and a 2 GiB KV allocation. It is a starting point
for sharing VRAM, not a guarantee against memory exhaustion. The unchanged
benchmark presets reserve a fixed KV pool and target a dedicated GPU.

Customize the limits without rebuilding or downloading another image:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh \
  --profile desktop --context 16384
```

Setting memory utilization switches to automatic KV sizing unless you explicitly
provide `--kv-cache-memory-bytes`. Lowering context alone does **not** reduce a
fixed KV allocation. `--kv-cache-memory-bytes auto` also enables automatic sizing.
Automatic profiling can leave insufficient cache for the requested context in
this runtime; startup reports the required and available cache sizes.
Use `--dry-run` to inspect the complete Docker command, or `--help` for all options.
Customized settings are separate from the benchmark configuration below.

### 200K and 220K with prefix caching

With your GPU visibility configured as above, use the chat profile for 200K,
or set a larger context on the **same image**:

```bash
# 200K total context, including generated tokens
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-200k.sh \
  --profile chat

# Stop the existing server before selecting 220K instead
docker stop paiton-qwen38-200k
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-200k.sh \
  --profile chat --context 220000
```

The chat profile uses one scheduled request, an 8 GiB KV pool, 1,024-token
prefill chunks, prefix caching, and thinking disabled. It also applies the
memory-allocation setting needed by this configuration. It leaves little spare
VRAM: select a dedicated R9700 rather than a card driving a busy desktop.
These settings differ from the 65K throughput benchmark configuration.

On one R9700, the 220K configuration correctly retrieved a value from a
**215,005-token prompt**. The identical repeat reused **213,840 cached tokens**
and completed in **1.77 seconds versus 130.00 seconds cold**; both answers were
nine tokens at temperature zero. These are complete response times from one
functional probe, not general latency or decode-throughput claims. Changed-prefix
retrieval, 380- and 409-token answers without observed looping, a subsequent XML tool call,
and a fresh short request also passed.

The unchanged release presets disable APC, so zero hits with their defaults is
expected. The persistent disk cache used during startup is separate from the
in-memory conversation prefix cache.

`--prefix-caching on` enables the experimental APC configuration and selects the
compatible recurrent-state settings. This changes memory requirements; do not
assume the release preset's fixed cache budget remains sufficient. Use
`--profile chat` for the complete long-context configuration. Cache hits require an
unchanged token prefix that is still resident. They reduce repeated prompt work,
not the cost of generating each new token.

The chat profile reports cached tokens in `usage.prompt_tokens_details.cached_tokens`.
Streaming clients must also request `"stream_options":{"include_usage":true}`
to receive usage in the stream. Server-side cache counters are available at
`http://127.0.0.1:18982/metrics`.

These checks are not a full long-conversation quality evaluation. Repetition
penalties and sampling settings belong in each client's API requests. Report the
prompt, settings and server logs when diagnosing loops; a sampling workaround is
not a general fix.

The release preset's model template enables thinking when the client omits that
setting. The chat profile and our benchmarks disable it. Use `--thinking off` to
set the server default, or send
`"chat_template_kwargs":{"enable_thinking":false}` in each request.

### Images and vision

These presets serve **text only** with `--language-model-only`. They do not accept
a llama.cpp `mmproj` file. A separate single-image smoke test with an 8K context
limit used the checkpoint's vision weights and passed color identification and
subsequent text and tool requests. That configuration needs additional VRAM and
is not included in these launchers; 200K/220K multimodal use has not been validated.

## Current benchmark results

R9700, 300 W; vLLM 0.29 / ROCm 10; 65,536 context; maximum eight sequences; APC off; thinking off. Temperature 0.7, top-p 0.95, top-k 20, seed 42. Each arm uses a fresh process and the same fixed warmup.

| Category | GGZ published update p50 ms | New Paiton update p50 ms | GGZ published tok/s | Previous post Paiton tok/s | Fresh release control tok/s | New package tok/s |
|---|---:|---:|---:|---:|---:|---:|
| chat | 42.3 | 33.2 | 67.1 | 123.6 | 123.2 | 121.3 |
| code | 42.5 | 33.4 | 120.5 | 169.1 | 168.7 | 180.6 |
| file edit | 42.5 | 33.3 | 138.0 | 185.7 | 185.5 | 181.2 |
| json | 42.4 | 33.4 | 153.1 | 216.0 | 215.8 | 218.1 |
| math | 42.5 | 33.5 | 140.0 | 182.5 | 182.7 | 184.3 |
| prose | 42.3 | 33.4 | 69.2 | 75.4 | 75.4 | 79.8 |
| reasoning | 34.3 | 33.4 | 123.9 | 112.4 | 112.5 | 117.8 |
| summarization | 34.2 | 33.3 | 141.7 | 115.0 | 115.1 | 138.8 |

Weighted decode: **146.85 → 154.42 tok/s (+5.16%)**.

| Concurrent requests | GGZ published | Historical GGZ local | Previous post Paiton | Fresh release control | New package |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 110.20 | 115.96 | 115.43 | 122.62 |
| 2 | 215 | 193.01 | 202.85 | 200.55 | 207.18 |
| 4 | 322 | 289.02 | 301.75 | 300.15 | 308.17 |
| 8 | 471 | 396.45 | 406.49 | 410.71 | 421.20 |

Concurrency values are aggregate generated tokens per second over each complete 48-request workload.

| Nominal prefill depth | GGZ published input tok/s | Previous post Paiton | Fresh release control | New package |
|---:|---:|---:|---:|---:|
| 2,000 | 3,552 | 3,510 | 3501 | 3501 |
| 8,000 | 3,536 | 3,532 | 3535 | 3692 |
| 16,000 | 3,619 | 3,535 | 3540 | 3694 |
| 32,000 | 3,437 | 3,389 | 3394 | 3584 |
| 64,000 | 3,192 | 3,116 | 3118 | 3293 |

Update p50 is the median streamed-update gap, not per-token latency or TTFT.
Sampled output content and accepted-token work can differ; these are serving-throughput measurements, not identical-output timing.
Nominal prefill depths correspond to median actual prompt lengths 1516.5, 5894.5, 11802, 23549.5 and 47016.5.
GGZ comparison values are historical references from the previous post; GGZ was not rerun.

Both arms: 290/290 successful benchmark requests. Candidate matches all 12/12 greedy controls before timing. Each arm repeats 12/12 greedy outputs after the benchmark. No unhandled serving errors.

Decode has five scored requests per category after one warmup. Prefill has eight scored requests per depth after two warmups. These are individual complete runs; category values are not selected across repeats.

[Machine-readable results](benchmarks/2026-09-20-combined/numbers.json).

## Historical releases and comparisons

The sections below describe earlier images, checkpoints, launchers and benchmark
settings. Their numbers are separate from the current release. In particular,
`serve.py`, `runtime.lock.json`, and `checkpoint.lock.json` below describe the
legacy release; use the `run-rocm10-*.sh` launchers above for the current images.

<details>
<summary>Earlier releases, APC investigation, benchmarks and reproduction instructions</summary>

## Long coding conversations: prefix caching can remove most repeat-turn waiting

**New APC investigation — 17 September 2026.** Automatic prefix caching reuses
computation for an unchanged conversation prefix. It can make a large practical
difference when a coding agent sends the same growing history on every turn.
Our released profiles currently disable it because compact native GDN replay
does not yet support prefix reuse.

On one R9700, the stock-GDN APC path reduced **40K cold-to-repeat first-token
latency from 15.58 s to 1.13 s**. Our experimental native-prefill APC candidate
reduced **150K cold-to-repeat latency from 88.46 s to 2.04 s**, reproduced with a
second distinct prefix at **88.61 s to 2.03 s**. Those are **43–44× faster response
starts on cache hits**, not decode-throughput gains or new competitor results.
The 150K tests used an 8 GiB cache and one active request.

APC is a workload choice: it helps repeated documents and growing conversations,
while the compact native path retains better decode performance and cache
capacity for fresh prompts. The report includes these tradeoffs and the cache
miss, branch, tool-result and concurrency checks.

**Availability:** the existing GHCR images remain unchanged. They do **not**
implement the new `PAITON_PREFIX_CACHING=1` convenience flag yet. Users can try
the measured stock-GDN APC configuration on the published 64K image with the
[explicit profile override](benchmarks/2026-09-17-prefix-caching/REPRODUCE.md).
The native-prefill APC candidate is not yet distributed. Do not apply the
fallback to the 200K image with its existing 8 GiB cache: that budget does not
meet the stock-APC cache requirement at 200K.

[APC results, raw evidence and limitations](benchmarks/2026-09-17-prefix-caching/README.md)
· [Try stock-GDN APC and reproduce the controls](benchmarks/2026-09-17-prefix-caching/REPRODUCE.md).

## 64K and 200K images, tool-call fix, and quick benchmark — 17 September 2026

New v1.1.0 images add Qwen XML tool-call parsing and configurable context limits.
The **64K profile** uses a 5 GiB cache and up to eight scheduled requests; the
**200K profile** uses an 8 GiB cache and one active request, with little spare
VRAM on the 32 GB R9700. The original v1.0.0 image remains available.

The 64K image completed a 52-request BetterBench quick run: **304.1 tok/s aggregate
at concurrency eight**, with **67.8 tok/s median per request**. These are short
prompts with a 128-token output cap and thinking disabled. Separate long-context
retrieval and OpenCode tool tests are documented alongside the benchmark.

[New benchmark page, visuals, evidence, and GHCR package links](benchmarks/2026-09-17-agentic-64k/README.md)
· [64K and 200K launch commands](LAUNCH-agentic-v1.1.0.md)
· [Tool calling and context details](SUPPORT.md).

The comparison sections below retain the measurements from the earlier 8K release.

## Earlier release benchmark suites

Paiton delivers **22% higher weighted decode throughput**, **57% more throughput
at eight concurrent requests**, and **12.5–17.3% faster prefill** than Radiance +
DFlash2 on the same Radeon AI PRO R9700. Paiton leads all eight task categories
and all four concurrency throughput levels in the full 188-request comparison.

At eight concurrent requests, median time to first token falls from **6.59
seconds to 195 ms**, including queueing. The same 5 GiB cache pool provides
**2.89× the estimated token capacity**, with logs recording eight active requests
for Paiton versus three for Radiance.

Paiton combines native HIP kernels, adapted Radiance techniques, and vLLM's
DFlash2 support. The official vLLM installation stays in place: a model-specific
plugin supplies the native implementation. There is no separate Radiance engine
or DFlash package to install.

![Stock vLLM O2, Radiance + DFlash2, and Paiton + DFlash2 throughput and prefill](three-engine-throughput.png)

The common 54-request matrix includes stock vLLM O2: C8 throughput is
**33.7 / 175.6 / 328.5 tok/s** for stock / Radiance / Paiton respectively. Its
128-token generation cap differs from the longer full preset, reported
separately below.

![Full-workload comparison against Radiance + DFlash2](full-confirmation.png)

[Complete three-engine tables, latency, and measurement scope](BENCHMARKS.md).
Radiance retains an approximately 10–11 ms TTFT advantage at one and two
concurrent requests; Paiton's latency advantage appears under concurrent load.

## Run the v1.0.0 benchmark release

[Context overrides, structured tool calls, and coding-agent setup](SUPPORT.md)
cover the reproduced parser mismatch and the local configuration workarounds.
The corrected 64K profile has functional API/client validation separate from the
historical benchmarks.

Release **v1.0.0** is available on GHCR. The launcher pins the qualified image
by digest in [runtime.lock.json](runtime.lock.json).

Use Linux x86-64, Docker, and one Radeon AI PRO R9700 with working AMD GPU device
access. The dedicated image includes the pinned official vLLM 0.28 ROCm runtime,
the Paiton adapter, and all qualified native libraries.

```bash
docker run -d --name paiton-qwen38-mxfp4 \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 2g \
  -p 127.0.0.1:8000:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:9b2dae214076d35de785e073b31294b033a376b16e6bc1ec1fdada4e54d96c59
```

The first image pull downloads approximately 11.4 GB of runtime layers when
those layers are not already cached. The first start then downloads approximately
21.9 GB of target and draft weights
from their original repositories and verifies the locked file hashes. Later
starts reuse the cache. Follow `docker logs -f paiton-qwen38-mxfp4` until startup
completes, then check `curl --fail http://127.0.0.1:8000/health`.

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Quark-AWQ-MXFP4","messages":[{"role":"user","content":"Write a Python function that preserves the first occurrence of each item in a list."}],"temperature":0,"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

No compiler checkout or build step is required. The image starts the ordinary
vLLM OpenAI API server after checkpoint verification.

## Launcher and cached models

From this repository, [serve.py](serve.py) starts the same image and defaults to
verified downloads with a persistent Docker cache volume:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py --detach
```

Use `--cache /path/to/cache` for a host cache directory, `--port` for another
local port, and `--name` for a different container name. `--dry-run` prints the
Docker argument list without downloading files or starting a container.

To download and verify the snapshots without accessing the GPU:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py --download-only
```

After downloading, `--offline` requires cached or explicitly mounted files.
Existing snapshot directories can be supplied independently; each is mounted
read-only and verified inside the container:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py \
  --target /path/to/pinned-target-snapshot \
  --draft /path/to/pinned-draft-snapshot \
  --offline --detach
```

Use one launch example at a time for the same GPU. `docker stop
paiton-qwen38-mxfp4` stops the default serving container; the cache persists.

## Model and v1.0.0 profile

| Setting | Tested configuration |
|---|---|
| Target | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| Draft | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` |
| Hardware | One Radeon AI PRO R9700, RDNA 4 / `gfx1201`, 32 GB |
| Runtime | Official vLLM 0.28 ROCm image plus the Paiton plugin |
| Input | Text |
| Maximum context | 8,192 total tokens per request |
| Concurrent requests | Up to eight |
| Cache | FP8 KV, explicit 5 GiB pool, prefix caching disabled |
| Prefill chunk | Up to 4,096 tokens |
| Speculation | DFlash2, seven speculative tokens, unpadded drafting |
| Benchmark sampling | Greedy; no requested log probabilities |

Target and draft revisions, file sizes, and SHA256 hashes are pinned in
[checkpoint.lock.json](checkpoint.lock.json). This checkpoint is distinct from
the Qronos and NEO CODER MAX releases; their speeds are not used as model-matched
baselines here. The token-cache capacity estimate does not extend the tested
8,192-token per-request context limit.

The native overlay and manifests are also distributed through
[Hugging Face](https://huggingface.co/EliovpAI/Qwen3.8-27B-Quark-AWQ-MXFP4-DFlash2-Paiton-RDNA4).
The weight files remain in the original model repositories.

## Integration and provenance

The dedicated image installs the model-specific plugin with `--no-deps` onto
the pinned official vLLM image. It uses vLLM's extension interfaces for the
model, loader, linear kernels, attention backend, and worker. It does not replace
the installed vLLM library. The main repository's general installation has a
different historical vLLM pin; use this model's qualified image and profile.

Paiton's native artifacts use HIP and load independently of PyTorch, Triton,
and Radiance. The serving adapter retains vLLM's existing framework dependencies
and upstream DFlash2 scheduling. The compiler and implementation source stay
private; the runtime package contains only the external adapter, allowlisted
native libraries, and runtime metadata.

[Radiance](https://github.com/magiccodingman/vllm-radiance) and
[StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) are credited for
the adapted kernel techniques. [Third-party attribution and terms](THIRD_PARTY_NOTICES.md).

The ordinary CLI deployment check passes streaming, eight concurrent requests,
and generation at the 8K context boundary followed by a fresh request.
[Deployment check](deployment-check.json) · [Unchanged-vLLM audit](runtime-audit.json) · [Published-image audit](release-audit.json).

## Reproduce the image context

The image can be rebuilt from the public adapter and the pinned native overlay.
The context preparer verifies an explicit file allowlist; it does not require
the private compiler. Download the companion release with the Hugging Face CLI,
then prepare a new build directory:

```bash
hf download EliovpAI/Qwen3.8-27B-Quark-AWQ-MXFP4-DFlash2-Paiton-RDNA4 \
  --revision v1.0.0 --local-dir qwen38-paiton-runtime
python3 models/Qwen3.8-MXFP4-DFlash2/prepare_image_context.py \
  --overlay qwen38-paiton-runtime/overlay --output qwen38-image-context
docker build -t paiton-qwen38-mxfp4:local qwen38-image-context
```

Use this model's `runtime-pyproject.toml`, copied automatically by the preparer;
the repository-wide package targets other runtime versions. The published
container digest identifies the tested distribution; a local rebuild creates
its own image identity.

</details>
