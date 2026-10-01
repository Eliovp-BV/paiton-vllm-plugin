# Qwen3.8 27B on one R9700

Choose **MXFP4** or **3-bit**. Both use vLLM with DFlash2 on a Radeon AI PRO R9700
(32 GB), with a 65,536-token context and up to eight scheduled requests by default.

| Run | What you get |
| --- | --- |
| [run-mxfp4.sh](run-mxfp4.sh) | Higher weight precision and stronger measured knowledge recall; FP8 KV cache. |
| [run-3bit.sh](run-3bit.sh) | Faster generation and more cache capacity; 3-bit W3A4 weights and a 4-bit KV cache. Requires the extra weights below. |

Both choices use the same NVFP4 target download: the MXFP4 path requantizes it
at load time; the 3-bit path uses our additional weights. The scripts select the
weight mode explicitly, even if you have downloaded both.

[Compare measured speed and cache capacity](#current-benchmark-results).

## Before you start

- Linux x86-64, Python 3, [Hugging Face CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli), and Docker usable without `sudo`.
- One R9700 with 32 GB VRAM and AMD device access (`/dev/kfd` and `/dev/dri`).
- About **75 GB free disk**, including the container, target, drafter and optional 3-bit weights.

Run the commands below from the repository root, in the same terminal:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

Already cloned it? Use your existing checkout. The launcher pins the
**29 September r2** image, including the DFlash2 draft-head fix.

## Model weights and existing downloads

Both variants need these two downloads. Set the paths again when opening a new
terminal; the downloaded files and runtime cache are reused.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20260928"
mkdir -p "$PAITON_TARGET_DIR" "$PAITON_DRAFT_DIR" "$PAITON_CACHE_DIR"

hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --local-dir "$PAITON_TARGET_DIR"
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
  --revision ee0cb26a8279b7910cc28d82a8a3e15e4728d56f \
  --local-dir "$PAITON_DRAFT_DIR"
```

Already downloaded these revisions? Set the variables to your complete local
folders instead. Each needs its own `config.json` and weights; keep the target's
tokenizer too. For linked Hugging Face cache snapshots, use the
[cache setup instructions](REFERENCE.md#already-in-the-hugging-face-cache).

### Optional: 3-bit W3A4 weights

For `run-3bit.sh`, also download these **9.55 GB** of weights. Keep the target
and drafter above; this is an add-on.

```bash
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
mkdir -p "$PAITON_W3ROT_DIR"
hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision d74ae7d5f5f1b4b45dd12fb1271e3664283a2ec1 \
  --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)
```

## Start the server

Choose **one**:

```bash
# MXFP4
bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh

# Or 3-bit
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh
```

The server runs in the foreground. First startup with an empty runtime cache
can take about **seven minutes**; wait for `/health` to succeed. The container
includes the runtime; no private compiler checkout is needed.

**API:** `http://127.0.0.1:18982/v1` · **Model:** `Qwen3.8`

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop with `docker stop paiton-qwen38` before switching variants or settings.
Add `--detach` to launch in the background; `docker logs -f paiton-qwen38` follows its log.

## Common options

Add these to **either** script:

| Option | What it changes |
| --- | --- |
| `--vision` | Enables image input in the 65K mode and automatically reduces the KV allocation to make room. |
| `--context 200000` | One long conversation: 200K total context, prefix caching on, FP8 KV, thinking off. |
| `--context 220000` | Same long-context mode at the largest context tested here. |
| `--profile desktop` | Smaller starting preset for a shared GPU: 32K context, one request, 2 GiB KV. |
| `--thinking off` | Disables thinking by default; clients can override it. |
| `--kv-cache fp8` | Uses FP8 instead of the 3-bit preset's default 4-bit KV cache. |
| `--port 18082` | Changes the API port. |
| `--help` / `--dry-run` | Lists all options / prints the Docker command without starting it. |

For example:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --vision
# After stopping that server, run a long text conversation:
bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh --context 200000
```

<a id="images-and-vision"></a><a id="long-context-200k-and-220k"></a>

**Vision at 200K is not supported by the current launcher.** Use `--vision`
in the 65K mode (or desktop preset). The vision encoder and image processing
need extra VRAM, leaving less memory for context. Combining very long prompts
with images can cause out-of-memory errors, even when the text-only run fits.
The launcher reduces the default KV budget for vision; custom memory settings
replace that budget. Vision with prefix caching is also not yet qualified,
which is why the standard 200K mode refuses `--vision`.

Context includes **input plus output**. The default 65K mode has prefix caching
off; long context turns it on and uses FP8 KV because KV4 with prefix caching
is not qualified. Up to eight scheduled requests does not mean eight full 65K
prompts will fit. See the [reference](REFERENCE.md#gpu-context-and-memory-controls)
for GPU selection, custom memory budgets and measured capacity.

## Current benchmark results

Measured on **one 300 W R9700**. Speed depends on prompt length, workload and
concurrency; use these as reference measurements for each choice.

**65K serving: speed comparison with FP8 KV in both arms, 26 September.**

| Measurement | MXFP4 | 3-bit |
| --- | ---: | ---: |
| Weighted single-stream decode | 156.1 tok/s | 184.4 tok/s |
| Eight concurrent requests, aggregate output | 428.0 tok/s | 492.1 tok/s |
| Prefill, approximately 5.9K input tokens | 3,831 tok/s | 4,165 tok/s |

These runs predate the current image. `run-mxfp4.sh` still uses FP8 KV;
`run-3bit.sh` now defaults to KV4, so this is **not an exact benchmark of today's
two default commands**. Add `--kv-cache fp8` to select the 3-bit FP8 path.
[Full settings, results and subsequent KV4 checks](REFERENCE.md#current-benchmark-results).

**Long-context text: add `--context 200000` to either script.** Both use FP8 KV.
Measured with a 198,989-token prompt:

| Measurement | MXFP4 | 3-bit |
| --- | ---: | ---: |
| New prompt, time to first token | 102 s | 94 s |
| Follow-up reusing the cached prompt | 1.2 s | 1.2 s |
| Decode at this length | 63–68 tok/s | 73–78 tok/s |

[Long-context test details](REFERENCE.md#long-context-200k-and-220k).

**65K default cache capacity:** MXFP4 uses FP8 KV; 3-bit uses KV4.

| Pooled cache tokens | `run-mxfp4.sh` | `run-3bit.sh` |
| --- | ---: | ---: |
| Text only | 174,634 | 393,216 |
| With `--vision` | 120,277 | 278,050 |

These are shared pool totals, not context per request. Vision takes memory from
that pool; neither the capacity figures nor the text benchmarks establish
200K vision support. [Vision checks](REFERENCE.md#images-and-vision).

In the paired FP8 KV comparison, the 3-bit weights traded some accuracy for
speed and memory: the MMLU-Pro subset fell **2.86 points** versus MXFP4.
Math, coding and retrieval differences
were within the reported uncertainty. [Quality results](REFERENCE.md#faster-decode-and-prefill-3-bit-w3a4-weights-optional).

## Advanced setup and release history

[Release reference](REFERENCE.md) contains image pins, release notes, detailed
benchmarks, long-context checks, KV tuning and older releases. The existing
`run-rocm10*.sh` scripts remain available; `run-rocm10.sh` automatically chooses
3-bit when `PAITON_W3ROT_DIR` is set and MXFP4 otherwise.

### Native serving

Already have the exact supported vLLM environment? `paiton serve qwen38-nvfp4`
uses the separate **text-only, non-speculative** preset: 65K, FP8 KV, no DFlash2
or 3-bit weights. Follow [native setup](REFERENCE.md#native-serving) for installation.
Its bundle is `qwen38-rocm10-native-20260921`, profile
`qwen38-nvfp4-w4a8-text-65k`, with the same pinned target revision above.

[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[3-bit weight license and calibration](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
