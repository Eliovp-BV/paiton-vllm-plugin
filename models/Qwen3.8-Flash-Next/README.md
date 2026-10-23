# Qwen3.8 Flash Next on two R9700

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in two Paiton releases for two AMD Radeon AI PRO R9700, served
by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin. One launcher runs both; pick a page:

- **[3-bit: the fastest](README-3bit.md)**: everything in VRAM.
- **[4-bit: the accuracy release](README-4bit.md)**: the 16-bit model's scores; part of the experts in system RAM.

Every published command serves as in the previous release (r3), on the torch worker. Opt-in: the
[native Paiton worker](#native-worker-opt-in) (`--fn-worker native` on `--mode decode` and `--mode 4bit`). New in 20261023-pc-rc1:
[prefix caching on the native worker](#prefix-caching-on-the-native-worker) with a RAM prefix tier sized from the host
(`--fn-worker native --prefix-caching`).

## Which one do I want?

| | [3-bit](README-3bit.md) | [4-bit](README-4bit.md) |
|---|---|---|
| Choose it for | Speed | Accuracy |
| Task benchmarks (16-bit: GSM8K 96.9 %, HumanEval 96.3 %, MMLU-Pro 84.7 %) | not measured yet | **GSM8K 96.8 %, HumanEval 96.3 %, MMLU-Pro 83.7 %** |
| Long documents (NLL vs bf16) / needles | +0.55 % / 40 of 40 at 131K and 200K | **+0.16 %** / 40 of 40 at 131K, identical to bf16 |
| Single-request decode, default (torch worker, checkpoint drafter) | **216 tok/s** | ~190 tok/s |
| Single-request decode, `--fn-worker native` (distilled drafter) | 272.5 tok/s | 240.4 tok/s (`--kv-profile speed`); 240.3 tok/s (default `--kv-profile capacity`) |
| 8 concurrent requests, default (torch worker, checkpoint drafter) | **565 tok/s** | 421 tok/s |
| 8 concurrent requests, `--fn-worker native` (distilled drafter) | 685.4 tok/s | 566.7 tok/s (`--kv-profile speed`); 561.9 tok/s (default `--kv-profile capacity`) |
| Prefill, 64K-token prompt, default (torch worker) | **8,161 tok/s** | 6,991 tok/s |
| Prefill, 64K-token prompt, `--fn-worker native` (distilled drafter) | 8,392 tok/s | 7,759 tok/s (`--kv-profile speed`); 7,757 tok/s (default `--kv-profile capacity`) |
| Weights | 116 GB, all in VRAM | 133 GB, about 21.4 GiB of experts per GPU in VRAM (21.2 with `--vision`), the rest in pinned system RAM |
| System RAM | 128 GB or more (native: about 60 GiB of free GTT, about 128 GB+ of RAM) | 256 GB as benchmarked; 128 GB, 96 GB and 64 GB with the automatic layouts ([4-bit page](README-4bit.md#system-ram)); native: a GTT limit of about 98 GiB (about 196 GB+ of RAM) |
| Context window | 98K (speculative decoding on), 200K (off); native: 98K | 98K or 262K (on), 200K (off); native: 98K |
| KV cache | about 112K tokens (`--mode decode`); native: about 551K | about 305K tokens (`--mode 4bit`); native: about 305K (default `--kv-profile capacity`) or 183K (`--kv-profile speed`) |
| Agents and coding tools | `--prefix-caching` (torch worker); native: `--fn-worker native --prefix-caching` | `--mode 4bit-cache` (torch worker); native: `--fn-worker native --prefix-caching` |
| Repeated 90K-token prompt, time to first token (native prefix caching: cold → hit) | 10.53 → 0.32 s | 16.82 → 0.46 s (200K document: 38.14 → 0.55 s) |
| Image input | no | `--vision` (torch worker; `4bit`, `4bit-cache`, `4bit-long`) |
| Start | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh` | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh` |
| Start, native worker | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native` | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native` |
| Hugging Face | [W3A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4) | [W4A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4) |

Speed: BetterBench 0.6.0 standard profile on the shipped images through this launcher ([how](#how-the-numbers-were-taken)). Default
rows: the torch worker as published in r3, with the checkpoint's own drafter (decode and 8 requests in `--mode decode` /
`--mode 4bit`, prefill in `--mode prefill-long` / `--mode 4bit-200k`). Native rows: every number in `--mode decode --fn-worker native`
/ `--mode 4bit --fn-worker native` (with the `--kv-profile` named), with the distilled drafter B4gh5-mtpft that ships in the image.
Task benchmarks: same prompts, sampling seeds and thinking-on template for the 4-bit and the 16-bit model; method and significance on
the [4-bit page](README-4bit.md#accuracy). No custom engine was built: this is regular vLLM with the Paiton plugin and our own kernels.
We keep tuning, we keep optimizing. What changed in this release: [RELEASE-NOTES.md](RELEASE-NOTES.md).

## Prerequisites
- Two AMD Radeon AI PRO R9700 (32 GiB each) on a ROCm 10 host driver; Docker with GPU device access (`/dev/kfd`, `/dev/dri`; groups
  `video`, `render`). System RAM per release: see the table above.
- `python3` 3.10 or newer and the Hugging Face hub package: `pip install huggingface_hub` (the launcher downloads the weights and the
  base model's tokenizer files through it; the `hf` / `huggingface-cli` binaries are only a fallback).
- Disk: the weights (table above), a few MB for the runtime view and ~0.3 GB per mode of compile cache; the 3-bit image is about
  41 GB; the 4-bit / native image is larger: it also carries the native worker and the distilled drafter (about 7.5 GB).
  The 4-bit / native image is about 56 GB on disk.
- The native worker has its own host requirements: [Native worker (opt-in)](#host-requirements).
- `launch-flashnext.py --dry-run` checks these prerequisites (RAM, GTT limit, peer-to-peer link between the GPUs) and prints exactly
  what is missing.

## Use the server
The wrappers and modes are on the [3-bit](README-3bit.md#run-it) and [4-bit](README-4bit.md#run-it) pages. The server runs in the
foreground; add `--detach` to run it in the background (`docker logs -f paiton-flashnext` follows its log). The first start of a mode
copies the baked compile cache and takes about **five to six minutes** (weight loading dominates). Wait for the launcher's
`launcher: READY on port …` line (with `--detach` the command returns at that point): after `/health` answers, the launcher drops the
checkpoint files from the page cache and waits until the kernel has finished compacting host memory (at most 30 s), because that
compaction can pause the GPUs for a few seconds right after start. `--settle-s 0` and `--keep-page-cache` turn this off.

**API:** `http://127.0.0.1:18982/v1` (or `http://<this-machine-ip>:18982/v1` from another system) · **Model:** `Qwen3.8-Flash-Next` ·
**API key:** none unless you set one (enter any value if a client asks)

**Thinking and tool calls.** The model thinks before it answers unless the request sets
`"chat_template_kwargs": {"enable_thinking": false}`. The server returns the thinking in `reasoning` (streamed as
`delta.reasoning`) and the answer in `content`; with `tools` in the request, tool calls come back as `tool_calls`.

The server listens on all interfaces (0.0.0.0); use `--host 127.0.0.1` to keep it local. To require an API key, start it with the key
in the environment: `PAITON_API_KEY=<key> bash models/Qwen3.8-Flash-Next/run-flashnext.sh` (clients then send it as the bearer token).
`--api-key KEY` works too, but the key is then visible in the process list (`ps`) of the machine. Use a key or a firewall if the
machine is reachable from untrusted networks.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true}'
```

- **Stop** with `docker stop paiton-flashnext` (or the name you started it with) before switching to another mode or release.
- **Long prompts:** send prompts of 100K+ tokens from a file (`-d @request.json`) with a client timeout of a few minutes (a
  190K-token prompt takes about 25 s to read in the 200K mode).

## Native worker (opt-in)
The default of every mode (`decode`, `prefill-long`, `decode-nopf`, `prefill-long-nopf`, `4bit`, `4bit-cache`, `4bit-long`,
`4bit-200k`, `4bit-xl`) is the torch worker, exactly as in the previous release (r3): a published command serves as before.
`--fn-worker native` runs the model in the native Paiton worker instead; vLLM keeps the API server, the scheduler and the KV cache
manager. Two modes have it, plus their prefix-caching variants ([below](#prefix-caching-on-the-native-worker)):

| | 3-bit | 4-bit |
|---|---|---|
| Command | `--mode decode --fn-worker native` | `--mode 4bit --fn-worker native` (`--kv-profile capacity`, the default) or `... --kv-profile speed` |
| Window | 98,304 tokens, speculative decoding on (MTP depth 3) | 98,304 tokens, speculative decoding on (MTP depth 3) |
| KV cache | about 551K tokens (torch default `--mode decode`: about 112K) | `capacity`: 2.5 GiB per GPU, about 305,338 tokens (3.11 concurrent 98K contexts, as r3 `--mode 4bit`); `speed`: 1.5 GiB per GPU, about 183,202 tokens (1.86 contexts), more experts in VRAM, faster |
| Image | `qwen38-flashnext-rocm10-vllm029-20261023-pc-rc1` | `qwen38-flashnext-rocm10-vllm029-20261023-pc-rc1` |

The other modes have no native release path. `--dry-run` prints which worker a command runs and how to get the other.

| Configuration | Drafter | Decode, 1 request | TTFT p50 | Prefill, 64K | 8 requests | 4 requests |
|---|---|---:|---:|---:|---:|---:|
| 3-bit `--mode decode --fn-worker native` | distilled (B4gh5-mtpft) | 272.5 tok/s | 56 ms | 8,392 tok/s | 685.4 tok/s | 531.0 tok/s |
| 3-bit `--mode decode --fn-worker native --drafter stock` | checkpoint's own | 249.9 tok/s | 56 ms | 8,401 tok/s | 659.4 tok/s | 516.4 tok/s |
| 3-bit default `--mode decode` (prefill: `--mode prefill-long`), torch worker, r3 | checkpoint's own | 216.3 tok/s | 117 ms | 8,161 tok/s | 565.1 tok/s | 449.2 tok/s |
| 4-bit `--mode 4bit --fn-worker native --kv-profile speed` | distilled (B4gh5-mtpft) | 240.4 tok/s | 65 ms | 7,759 tok/s | 566.7 tok/s | 456.0 tok/s |
| 4-bit `--mode 4bit --fn-worker native` (`--kv-profile capacity`, the default) | distilled (B4gh5-mtpft) | 240.3 tok/s | 65 ms | 7,757 tok/s | 561.9 tok/s | 456.9 tok/s |
| 4-bit `--mode 4bit --fn-worker native --drafter stock` (`--kv-profile capacity`, the default) | checkpoint's own | 237.6 tok/s | 65 ms | 7,748 tok/s | 545.4 tok/s | 443.4 tok/s |
| 4-bit default `--mode 4bit` (prefill: `--mode 4bit-200k`), torch worker, r3 | checkpoint's own | 189.8 tok/s | 121 ms | 6,991 tok/s | 420.7 tok/s | – |

BetterBench 0.6.0 standard profile ([how](#how-the-numbers-were-taken)). Native prefill rows are measured in the native mode itself
(98,304-token window, speculative decoding on); the torch rows' prefill comes from the 200K modes (speculative decoding off).

**Drafter.** The native modes use the distilled MTP drafter B4gh5-mtpft by default. It ships inside the image, one copy shared by
3-bit and 4-bit. The launcher assembles `<checkpoint dir>-b4gh5-mtpft` next to the downloaded checkpoint: hard links to the checkpoint's
files (copies across file systems) plus links to the drafter files in the image; the downloaded checkpoint itself is never modified.
`--dry-run` checks the image's drafter files against their pinned sha256. `--drafter stock` uses the checkpoint's own drafter (the
checkpoint-only path).

**Image.** From 20261023-pc-rc1 on, every native mode runs `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261023-pc-rc1`
@ `sha256:156177eee1297f658f87f59210c0e7eec11603f19590795b591ee999c2820eb7`: the 20261017-r1 image plus one layer (5.34 MB) with the prefix-caching executor and the cache modes' files. The two
native modes above keep their executor and files byte-identical and were re-measured on it (inside the range of the 20261017
measurements). The torch 4-bit modes keep `...-20261017-r1`, the 3-bit torch modes `...-20261010-r2`.

**No image input.** `--vision` is a torch-worker option (4-bit modes); it is not available with `--fn-worker native`. The 3-bit has no
image input in this release.

**Repeatability (4-bit).** On `--mode 4bit --fn-worker native`, greedy output can vary from run to run while
a long prompt (tens of thousands of tokens) is being read next to other requests; a request on its own, or several short ones
together, repeat exactly. Errors and quality are not affected. The 3-bit native worker repeats exactly in all of these cases.

### Prefix caching on the native worker
New in 20261023-pc-rc1. An agent or coding tool resends the whole conversation every turn; with prefix caching the shared part is
reused instead of read again. On the native worker, `--prefix-caching` switches to the mode's prefix-caching variant and prints the
switch; the three cache modes can also be named directly:

| | 3-bit | 3-bit, long window | 4-bit |
|---|---|---|---|
| Command | `--mode decode --fn-worker native --prefix-caching` (= `--mode 3bit-cache-native`) | `--mode 3bit-long-cache-native` | `--mode 4bit --fn-worker native --prefix-caching` (= `--mode 4bit-long-cache-native`; also from `--mode 4bit-long-native`) |
| Window | 98,304 tokens | 262,144 tokens | 262,144 tokens |
| Speculative decoding | off | off | off |
| RAM prefix tier | auto, up to 16 GiB per GPU | auto, up to 16 GiB per GPU | auto, up to 16 GiB per GPU |
| Host RAM | from about 63 GiB (tier off at 64 GiB); tier 2 GiB per GPU from about 79 GiB, 16 GiB from about 103 GiB | as 3-bit | a GTT limit of about 98 GiB (about 196 GB of RAM at the default GTT limit) |

Prefix-cache blocks are 2,048 tokens; a hit resumes on a block boundary and gives the same tokens as a cold start of the same request.
Blocks evicted from VRAM go to the RAM prefix tier in host memory and are loaded back on a hit. Time to first token on a repeated
prompt, cold → hit (served gates, 23 Oct, one request at a time):

| Prompt | 3-bit `3bit-cache-native` | 3-bit `3bit-long-cache-native` | 4-bit `4bit-long-cache-native` |
|---|---:|---:|---:|
| 32K tokens | 3.70 → 0.27 s | – | – |
| 90K tokens | 10.53 → 0.32 s | 10.53 → 0.32 s | 16.82 → 0.46 s |
| 200K-token document | – (98K window) | 24.53 → 0.42 s | 38.14 → 0.55 s |
| Hit from the RAM tier (evicted from VRAM) | 90K: 10.48 → 0.42 s | – | 200K: 32.72 → 0.63 s |

**Idle sessions.** Eight 4-bit sessions of 145K-260K tokens (1.62M tokens in total) left idle until their blocks had moved to the
RAM tier, then resumed one at a time: 0.57-0.62 s at 145K, 0.67-0.70 s at 161K, 0.74-0.78 s at 178K, 0.75-0.80 s at 194K,
0.96-0.98 s at 211K, 0.98-0.99 s at 227K, 0.99 s at 244K and 1.07 s at 260K (cold: 23-51 s); every session gave the same tokens as
its cold twin. The time grows with the depth because a resume reads the last 2.0-2.45K tokens again (from the last full block), and
those tokens attend over the whole context; the copy from RAM itself takes about 0.15 s. A stream that is decoding while another
session is restored keeps going: inter-token gap p50 10.5 → 12.5 ms, at most one gap of about 0.46 s.

**Speed: speculative decoding is off in the cache modes.** BetterBench 0.6.0 standard profile on the release image, no prefix hits
(unique prompts), compared with the speculative-decoding native modes:

| Configuration | Decode, 1 request | TTFT p50 | Prefill, 64K | 8 requests | 4 requests |
|---|---:|---:|---:|---:|---:|
| 3-bit `--mode 3bit-cache-native` | 101.0 tok/s | 73 ms | 8,731 tok/s | 403.4 tok/s | 279.7 tok/s |
| 3-bit `--mode 3bit-long-cache-native` | 100.5 tok/s | 73 ms | 8,725 tok/s | 401.5 tok/s | 279.0 tok/s |
| 3-bit `--mode decode --fn-worker native` (speculative decoding on) | 272.5 tok/s | 56 ms | 8,392 tok/s | 685.4 tok/s | 531.0 tok/s |
| 4-bit `--mode 4bit-long-cache-native` | 94.2 tok/s | 110 ms | 6,918 tok/s | 313.0 tok/s | 233.8 tok/s |
| 4-bit `--mode 4bit --fn-worker native` (speculative decoding on) | 240.3 tok/s | 65 ms | 7,757 tok/s | 561.9 tok/s | 456.9 tok/s |

Choose a cache mode when long conversations are resent (agents, coding tools, chat with long documents); choose the speculative
mode for the fastest decode of new prompts.

**RAM prefix tier sizing.** `--native-kv-tier auto` (the default of the cache modes) sizes the tier at start from the host and
prints the formula with its numbers on the `native RAM prefix tier:` line:

```
tier per GPU = min(16, floor((available RAM - the mode's resident RAM - start margin - 8 GiB) / 2)) GiB, off below 2 GiB
```

With the tier off, prefix caching stays on with hits from VRAM only. `--native-kv-tier GIB` or `off` overrides the size. The tier is
filled at start (10-27 s), so a RAM shortfall shows at start, not under load. 16 GiB per GPU holds about 2M tokens of idle
sessions; beyond that the least recently used blocks are dropped and their next turn is read again from the start (same tokens).

**`--memory GIB`** caps the server container's RAM (docker `--memory`, no swap): the RAM checks, the GTT limit (half of it) and the
tier are then sized as on a host with GIB of RAM, e.g. to leave RAM for other work. Tested hosts (the 4-bit pinned GTT tables are
not charged to a container limit, so for 4-bit the limit bounds only the pageable part):

| Host RAM | `4bit-long-cache-native` | `3bit-cache-native` |
|---|---|---|
| 252 GiB | tier 16 GiB per GPU; all gates pass (15-minute soak, 0 errors) | tier 16 GiB per GPU; all gates pass (also `3bit-long-cache-native`) |
| 200 GiB | tier 16 GiB per GPU; all gates pass | – |
| 192 / 128 / 64 GiB | refused before start: `GTT limit ... < 98 GiB ...; this host fits --mode 3bit-cache-native or --mode 3bit-long-cache-native` | 128 GiB: tier 16 GiB per GPU, all gates pass; 64 GiB: tier off, VRAM hits, all gates pass (tight: about 7 GiB left after the host tables) |

**Limits of the cache modes.** Speculative decoding is off (speculative decoding with prefix caching is in development); the
4-bit cache mode runs only with its host tables pinned in GTT (no `--ple-host pageable|filebacked`, `--expert-cold`); no `--vision`;
the same request-API limits as the other native modes (below). Greedy output is identical between a hit and a cold start when the
request is served alone; next to other running requests the greedy tokens can differ, with or without a cache hit (the batch
composition changes the arithmetic; quality is not affected).

### Request-API limits
The native worker answers these request fields with HTTP 400 and a message that names the field; the torch default serves all of them:

- `logprobs`, `top_logprobs`, `prompt_logprobs`
- `response_format`, structured outputs, guided JSON
- tool calls with `strict: true` tools, or `tool_choice` `"required"` or a named tool (plain `auto` tool calling works)
- `presence_penalty`, `frequency_penalty`, `repetition_penalty`
- `top_k` outside 1..32 at temperature > 0 (the server default 20 is fine)
- `min_tokens`, `allowed_token_ids`, `bad_words`
- `thinking_token_budget`

Served as on the torch worker: `temperature`, `top_p`, `top_k` 1..32, `seed`, `n` > 1, stop strings and stop token ids, `max_tokens`,
streaming, usage stats, the chat template including `enable_thinking`, the reasoning parser and `auto` tool calling. `min_p` and
`logit_bias` are refused by vLLM itself with speculative decoding, on both workers. We are working on these limits for a next release.

### Host requirements
When you ask for the native worker explicitly, there is no automatic fallback: if the host does not meet these, the launcher stops and
prints the reason (`--dry-run` prints it as a warning).

- **4-bit native:** the host tables pinned in GTT memory: a GTT limit of at least about 98 GiB. At the default GTT limit (half the
  RAM) that means about 196 GB of RAM or more. Hosts with less keep the torch default, which has the 128 GB / 96 GB / 64 GB layouts
  ([4-bit page](README-4bit.md#system-ram)). `--ple-host pageable|filebacked`, `--expert-cold` and `--host-pools userptr` are
  torch-worker layouts.
- **3-bit native:** about 60 GiB of free GTT for the host tables (about 128 GB of RAM or more at the default GTT limit).
- **Native prefix caching:** `4bit-long-cache-native` as 4-bit native (a smaller host is refused with the cache modes that fit it);
  `3bit-cache-native` / `3bit-long-cache-native` from about 63 GiB of RAM, with the RAM prefix tier from about 79 GiB
  ([above](#prefix-caching-on-the-native-worker)).
- Two R9700 (32 GiB each), as before.

## Options
`--mode decode|prefill-long|decode-nopf|prefill-long-nopf|4bit|4bit-200k|4bit-cache|4bit-long|4bit-xl|3bit-cache-native|3bit-long-cache-native|4bit-long-cache-native`,
`--fn-worker torch|native`, `--native-kv-tier auto|GIB|off`, `--memory GIB`,
`--kv-profile capacity|speed`, `--drafter distilled|stock`, `--prefix-caching`, `--vision`, `--vision-max-pixels N|2mp|4k`,
`--host-pools auto|gtt|userptr`, `--kv-cache-memory-bytes`, `--expert-vram-gib`, `--ple-host auto|pinned|pageable|filebacked`,
`--ple-raw-dir DIR`, `--expert-cold auto|off|SHARE`, `--expert-cold-dir DIR`, `--port`, `--host ADDRESS`, `--api-key KEY`, `--name`,
`--weights DIR`, `--cache DIR`, `--image REF`, `--devices GPU-...,GPU-...`, `--served-model-name`, `--detach`, `--settle-s`,
`--keep-page-cache`, `--dry-run`. GPU selection detects AMD devices by vendor id; two are required. The launcher picks the right image
per mode: the 3-bit torch modes run the published 3-bit image, the 4-bit torch modes the 20261017-r1 image, every native mode the
20261023-pc-rc1 image.

## Troubleshooting
- **"It is slow" during a long prompt.** Every 10 s the server log prints a line like
  `Avg prompt throughput: 6812.4 tokens/s, Avg generation throughput: 12.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 41.0%, Prefix cache hit rate: 0.0%`.
  *Prompt throughput* is the server reading your prompt (prefill); *generation throughput* is it writing the answer. Before the first
  token of a long request the server is reading: a 64K-token prompt takes about 8 s on the 3-bit and about 9 s on the 4-bit, and in
  that window generation throughput is near zero by design. Slow generation shows as low generation throughput while nothing is being
  read. `docker logs -f <name>` follows the log.
- **Agents re-reading the whole conversation.** With prefix caching, a repeated prefix is served from the cache; the log's
  *Prefix cache hit rate* shows it. If the KV cache is small relative to the window, the launcher warns at start
  (`WARNING prefix caching: the KV cache holds about 112K tokens < 2 x the 98K window ...`): long agent requests then queue, and a
  conversation that was evicted from the cache is read again from the start on its next turn. `--mode 4bit-cache` keeps evicted
  prefixes in a RAM tier; on the native worker `--prefix-caching` does the same ([above](#prefix-caching-on-the-native-worker)). Lowering `--kv-cache-memory-bytes` below the mode value shrinks the cache further, and the launcher says by how much.
- **Waiting requests.** `Waiting: N reqs` above 0 means the server is at its concurrency (8 requests) or its KV cache is full; requests
  start as others finish.

## How the numbers were taken
BetterBench 0.6.0, standard profile, on 2x R9700 against our OpenAI-compatible server through this launcher; prefill with exact bf16
activations on the wire (no compressed wire); 20 passes per category after warm-up, unique prompts (no prefix-cache hits), concurrency
1/2/4/8 x 48 requests, prefill depths 2K-128K. Raw outputs: `benchmarks/`; depth tables and host conditions: [BENCHMARKS.md](BENCHMARKS.md).

## Files
`run-flashnext*.sh` (the wrappers), `launch-flashnext.py` (also builds the runtime view `<weights>-view` once: base config + tokenizer
at the pinned revision, see RELEASE-NOTES.md), `modes.json` (environment + engine arguments of every mode), `README.md` (this
overview), `README-3bit.md`, `README-4bit.md`, `RELEASE-NOTES.md`, `BENCHMARKS.md`, `THIRD_PARTY_NOTICES.md`, `runtime.lock.json`
and `checkpoint.lock.json` (what this release was built from and tested with), `Dockerfile` + `prepare_image_context.py` +
`build-context.lock.json` (rebuild the image from locked inputs, see REPRODUCE.md), `release-audit.json`, `deployment-check.json`,
`metrics.json` and `benchmarks/` (the raw BetterBench outputs).
