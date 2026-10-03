# Qwen3.8 27B on one R9700

Qwen3.8 27B with DFlash2 speculative decoding in vLLM, on one Radeon AI PRO R9700 (32 GB).
Two weight choices:

- **MXFP4**: the most accurate; slower, with a smaller KV cache and a long context of 200,000 tokens. It needs only
  the base download, an NVFP4 checkpoint that the server converts to MXFP4 when it starts.
- **3-bit W3A4**: the fastest, up to 262,144 tokens of context, slightly less accurate. An extra 9.55 GB download on
  top of the base one.

Download the weights, then [pick how to run it](#pick-how-to-run-it).
[Measured speed and accuracy](#current-benchmark-results).

## Before you start

- Linux x86-64, Python 3, [Hugging Face CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli), and Docker usable without `sudo`.
- One R9700 with 32 GB VRAM that does not drive your desktop (for a shared card, see `--profile desktop` in
  [Advanced options](#advanced-options)), and AMD device access (`/dev/kfd` and `/dev/dri`).
- About **75 GB free disk**, including the container, target, drafter and optional 3-bit weights.

Clone the repository, then run every command below from its root, in the same terminal:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

Already cloned it? Run `git pull` in your checkout. The launcher pins the
**2 October r1** image,
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261002-r1@sha256:82a24a1926bc01a134b106401390650b9e0ddb0aa8cf6a613ba3a615ce46b840`
([what changed](REFERENCE.md#release-notes-2-october-2026)).

## Model weights and existing downloads

Both weight choices need these two downloads. The launcher finds them, and its runtime cache, through these
variables: set them again in every new terminal; the downloads and the runtime cache are reused.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261002"
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

For the `run-3bit.sh` rows of the [table below](#pick-how-to-run-it), also download these **9.55 GB**. They replace
part of the target's weights, so the target and drafter above stay required. `run-3bit.sh` finds them through
`PAITON_W3ROT_DIR`; set it again in a new terminal too.

```bash
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
mkdir -p "$PAITON_W3ROT_DIR"
hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision 7a2e3d702144f744c1c5a46eaa42c7a2309d5cf7 \
  --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)
```

<a id="want-longer-context-run-this"></a><a id="images-and-vision"></a><a id="long-context-200k-and-220k"></a>

## Pick how to run it

Run **one** of these commands from the repository root, with the paths above set:

| You want | Run | Context per request | At once (shared KV cache) | Prefix cache | Images (`--vision`) |
| --- | --- | ---: | --- | --- | --- |
| The most accurate answers | `bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh` | 65,536 | up to 8 requests, 174,634 tokens in total | no | yes |
| The most accurate answers, one long document | `bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh --mode long` | 200,000 | 1 request | yes | no |
| **Recommended:** fast everyday chat, coding and tools | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh` | 65,536 | up to 8 requests, 393,216 tokens in total | no | yes |
| One long document, many follow-up questions | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long` | 262,144 | up to 8 requests, 281,665 tokens in total: one full-length document plus short requests | yes | yes, up to 245,000 context |
| Several long conversations at once | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4` | 262,144 | up to 8 requests, 458,922 tokens in total | no | no |

- **Context** counts prompt plus output; a longer request is refused (HTTP 400). Running requests share the KV
  cache; when it is full, further requests wait.
- **Prefix cache: yes.** A new question about the document the server has just read starts in under 2 s instead
  of 1 to 2 minutes. The cache holds about one full-length document; a new long document replaces it.
  **No:** every request reads its whole prompt again, earlier turns included (`--mode long-kv4`: about 10 s at 32K
  tokens, 2 minutes at 258K).
- **`--mode long` or `--mode long-kv4`?** Add up the tokens of everything running at once: up to 281,665,
  `--mode long`; more, `--mode long-kv4`.
- **Images:** add `--vision` to a row with "yes", e.g.
  `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long --vision`.

Each row sets its own limits; to change them, see [Advanced options](#advanced-options).

## Start the server

Run the command you picked. The server runs in the foreground; add `--detach` to run it in the background
(`docker logs -f paiton-qwen38` follows its log). The first start with an empty runtime cache takes about
**seven minutes**; wait for `/health` to succeed.

**API:** `http://127.0.0.1:18982/v1` · **Model:** `Qwen3.8`

Thinking is on by default in the 65,536-token rows and off in the long rows. Each request can set it with
`"chat_template_kwargs":{"enable_thinking":true}` (or `false`); `--thinking on|off` changes the server default.
The example below and all benchmarks use thinking off.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop with `docker stop paiton-qwen38` before switching to another row.

## Current benchmark results

Measured on **one 300 W R9700**, thinking off. Speed depends on prompt length, workload and concurrency; use
these as reference measurements for each row.

**The 65,536-token rows.** Speed from BetterBench; accuracy with greedy decoding on the same questions for each
column.

| | `run-mxfp4.sh` | `run-3bit.sh --kv-cache fp8` | `run-3bit.sh` |
| --- | ---: | ---: | ---: |
| KV cache | FP8 | FP8 | 4-bit |
| Speed measured on | 26 September image | 26 September image | 2 October r1 image |
| Weighted single-stream decode | 156.1 tok/s | 184.4 tok/s | 179.7 tok/s |
| Eight concurrent requests, aggregate output | 428.0 tok/s | 492.1 tok/s | 488.5 tok/s |
| Prefill, approximately 5.9K input tokens | 3,831 tok/s | 4,165 tok/s | – |
| GSM8K 5-shot (1,319) | 95.68 | 95.30 | 95.22 |
| HumanEval pass@1 (164) | 95.12 | 93.90 | 94.51 |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 59.71 | 60.57 |

Paired MXFP4 against 3-bit (both FP8 KV): only the MMLU-Pro gap (−2.86 points, 95 % interval −4.81 to −0.90) is
beyond the uncertainty; math and coding are within it. The 4-bit KV cache scores the same as FP8. On the r1
image, 48/48 requests completed at each concurrency level (peak 30.85 GiB).
[Full settings and results](REFERENCE.md#current-benchmark-results) ·
[Quality results](REFERENCE.md#faster-decode-and-prefill-3-bit-w3a4-weights-optional).

**One long document: `run-3bit.sh --mode long`, 1 October.**

| | 198,989-token prompt | 257,992-token prompt |
| --- | ---: | ---: |
| New prompt, time to first token | 85 s | 126 s |
| Follow-up question on the cached document, time to first token | 1.4 s | 1.7 s |
| Decode at this depth | 73–78 tok/s | 72–76 tok/s |

Accuracy of the 3-bit weights with every question placed after a 257K-token document, against the same weights at
short context (paired per question; all differences are within the 95 % intervals):

| Benchmark | 3-bit, short context | 3-bit, after 257K tokens |
| --- | ---: | ---: |
| GSM8K 5-shot (1,319) | 95.30 | 95.83 |
| HumanEval pass@1 (164) | 93.90 | 91.46 |
| MMLU-Pro subset, 0-shot (14 × 100) | 59.71 | 60.29 |

Short requests in `--mode long` against `run-3bit.sh`, 1 October: the full 20-pass BetterBench, cold prefix cache,
fresh processes, the same tool and settings as the September tables.

| BetterBench row | `--mode long` | `run-3bit.sh` |
|---|---:|---:|
| Weighted single-stream decode | 174.3 tok/s | 178.9 tok/s |
| Gap between stream updates, p99 | 29.3 ms | 28.0 ms |
| Time to first token, p50 (short prompts) | 85 ms | 86 ms |
| Eight concurrent requests, aggregate output | 458.9 tok/s | 478.8 tok/s |
| Prefill at 47K input tokens (64K depth) | 3,549 tok/s | 3,481 tok/s |
| Prefill at 94K / 184K input tokens | 3,040 / 2,395 tok/s | – |

[Reports and numbers](benchmarks/2026-10-01-262k/README.md).

**Several long conversations: `run-3bit.sh --mode long-kv4`, 2 October r1 image, 3 October.** BetterBench
with the same settings as above:

| BetterBench row | `--mode long-kv4` |
|---|---:|
| Weighted single-stream decode | 174.3 tok/s |
| Time to first token, p50 (short prompts) | 87 ms |
| Four / eight concurrent requests, aggregate output | 364.6 / 462.4 tok/s |
| Prefill at 2K / 32K / 64K depth | 4,167 / 3,954 / 3,633 tok/s |

Accuracy in `--mode long-kv4` with standard-length questions, paired per question against MXFP4:

| Benchmark | `run-mxfp4.sh` | `--mode long-kv4` |
| --- | ---: | ---: |
| GSM8K 5-shot (1,319) | 95.68 | 95.53 |
| HumanEval pass@1 (164) | 95.12 | 92.07 |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 60.57 |
| Needle at 61,440 tokens (80) | 80/80 | 80/80 |

Only the MMLU-Pro gap is beyond the uncertainty: the 3-bit weights' gap, as in the 65,536-token rows. Over five
131K-token documents the 4-bit cache predicts the text as well as the FP8 cache at every depth.
[Details](REFERENCE.md#the-4-bit-cache-in-the-long-context-mode).

**Images with long context, and MXFP4 long context.** `run-3bit.sh --mode long --vision` (245,000 tokens): a
200K-token prompt with a chart, 87 s cold and 1.3 s from the cache. `run-mxfp4.sh --mode long` (200,000 tokens,
one request, 28 September): a 199K-token prompt, 102 s cold, 1.2 s from the cache, 63–68 tok/s.
[Long-context details and accuracy](REFERENCE.md#long-context-200k-and-220k) ·
[Vision checks and how to send an image](REFERENCE.md#images-and-vision).

<a id="common-options"></a>

## Advanced options

Every row of the [table](#pick-how-to-run-it) already sets its context, cache and request limits. Add these only
to tune; the launcher refuses flags that contradict the chosen `--mode`. `--help` lists every option and
`--dry-run` prints the Docker command without starting it.

| Option | What it changes |
| --- | --- |
| `--context TOKENS` | A smaller context than the row's (input plus output). Only MXFP4 `--mode long` goes higher: up to 220,000, the largest tested. |
| `--max-num-seqs COUNT` | Fewer concurrent requests than the row's. |
| `--long-prefill-threshold 2048` | 3-bit long rows shared by several users: short requests are answered within seconds while a long prompt is processed. The long prompt takes about 16 % longer (`--mode long`: 146 s instead of 126 s at 258K). |
| `--thinking on` / `--thinking off` | The server default for thinking (on in the 65,536-token rows, off in the long rows). Requests can override it. |
| `--kv-cache fp8` | `run-3bit.sh` without `--mode`: the FP8 instead of the 4-bit KV cache (250,578 tokens). |
| `--kv-cache-memory-bytes BYTES` / `--gpu-memory-utilization FRACTION` | Your own KV budget instead of the measured one, also with `--vision`. |
| `--profile desktop` | For an R9700 that also drives your desktop: 32,768 context, one request, 2 GiB KV. |
| `--port PORT` / `--name NAME` / `--detach` | API port (default 18982), container name, run in the background. |

Commands from earlier releases still work and give the same Docker command: `--context 262144` is `--mode long`
(MXFP4: `--context 200000`; with `--vision`: `--context 245000 --vision`), `--context 262144 --kv-cache kv4` is
`--mode long-kv4`, and `run-rocm10.sh` picks the 3-bit weights when `PAITON_W3ROT_DIR` is set and MXFP4 otherwise.

In `--mode long`, concurrent questions about the same ~257K-token document are effectively handled one at a time.
See the reference for [GPU selection and memory budgets](REFERENCE.md#gpu-context-and-memory-controls) and
[long-context details](REFERENCE.md#long-context-200k-and-220k).

## Advanced setup and release history

[Release reference](REFERENCE.md) contains image pins, release notes, detailed
benchmarks, long-context checks, KV tuning and older releases.

### Native serving

Already have the exact supported vLLM environment? `paiton serve qwen38-nvfp4` serves a separate preset: text
only, 65K, FP8 KV, without DFlash2 or the 3-bit weights. Follow [native setup](REFERENCE.md#native-serving) for
installation. Its bundle is `qwen38-rocm10-native-20260921`, profile `qwen38-nvfp4-w4a8-text-65k`, with the same
pinned target revision above.

[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[3-bit weight license and calibration](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
