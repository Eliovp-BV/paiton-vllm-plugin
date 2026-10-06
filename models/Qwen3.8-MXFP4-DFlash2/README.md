# Qwen3.8 27B on one R9700

Qwen3.8 27B with DFlash2 speculative decoding in vLLM, on one Radeon AI PRO R9700 (32 GB).
Two weight choices:

- **MXFP4**: the most accurate, from the base download alone. Up to 200,000 tokens of context, one request at a
  time in that mode; images only in the default 65,536-token mode.
- **3-bit W3A4**: the fastest, with room for more and longer requests: up to 262,144 tokens of context (524,288
  experimental); images up to 245,000 tokens (not in the coding or 512K modes). Slightly less accurate; an extra
  9.55 GB download.

Download the weights, then [pick how to run it](#pick-how-to-run-it).
[Measured speed and accuracy](#current-benchmark-results).

## Before you start

- Linux x86-64, Python 3, [Hugging Face CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli), and Docker usable without `sudo`.
- One R9700 with 32 GB VRAM that does not drive your desktop (for a shared card, see `--profile desktop` in
  [Advanced options](#advanced-options)), and AMD device access (`/dev/kfd` and `/dev/dri`).
- About **75 GB free disk**, including the container, target, drafter and optional 3-bit weights; the optional
  [`--extend-cache`](#extend-cache) SSD folder takes up to 64 GiB more.
- **16 GB of system RAM** is enough for every row (tested). Two rows pin 2.4 GiB of it; see
  [system RAM](#system-ram).

Clone the repository, then run every command below from its root, in the same terminal:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

Already cloned it? Run `git pull` in your checkout. The launcher pins the **5 October r1** image,
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261005-r1@sha256:245c71d54f046f89b74dbfbd4c894003754561e55c9d215a8d3ce65364fb0f80`, and Docker pulls it on the first
start ([what changed](REFERENCE.md#release-notes-5-october-2026)). The previous release stays runnable:
[run an earlier release](#run-an-earlier-release).

## Model weights and existing downloads

Both weight choices need these two downloads, the base download: the target, an NVFP4 checkpoint that the server
converts to MXFP4 when it starts, and the DFlash2 drafter. The launcher finds them, and its runtime cache (files the
server builds on its first start and reuses later), through these variables: set them again in every new terminal;
the downloads and the runtime cache are reused.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261005"
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

Find what you want, run that command from the repository root with the paths above set, and check what you get and
what you give up. Run **one** at a time.

**MXFP4 weights** (the base download only):

| You want | Run | Context per request | At once (shared KV cache) | Prefix cache | Images (`--vision`) | You give up |
| --- | --- | ---: | --- | --- | --- | --- |
| The most accurate answers | `bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh` | 65,536 | up to 8 requests, 174,634 tokens in total | no | yes | speed and room, against 3-bit: 156 instead of 180 tok/s, 174,634 instead of 393,216 cache tokens |
| The most accurate answers, one long document | `bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh --mode long` | 200,000 | 1 request | yes | no | one request at a time; no images |

**3-bit W3A4 weights** (the base download plus the [3-bit download](#optional-3-bit-w3a4-weights)):

| You want | Run | Context per request | At once (shared KV cache) | Prefix cache | Images (`--vision`) | You give up |
| --- | --- | ---: | --- | --- | --- | --- |
| **Recommended:** fast everyday chat, coding questions and tools (for coding agents see `--mode long-kv4`) | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh` | 65,536 | up to 8 requests, 393,216 tokens in total | no | yes | a little accuracy (MMLU-Pro 60.6 instead of 62.6) |
| One long document, many follow-up questions, also with images (screenshots) | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long` | 262,144 | up to 8 requests, 281,665 tokens in total: one full-length document at a time plus short requests | yes | yes, with the context lowered to 245,000 | about 3 to 4 % speed on short requests (against the recommended row); half the cache of `--mode long-kv4` |
| **Coding agents**, or several long conversations at once | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4` | 262,144 | up to 8 requests, 569,878 tokens in total: two full-length requests | yes | yes (496,129 tokens cached) | 2.4 GiB of pinned system RAM; reading a new long prompt is about 3 % slower than without prefix caching (7 to 13 % for short prompts, a fraction of a second) |
| Coding agents that return to more documents than the GPU cache holds, also after a restart | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 --extend-cache` | 262,144 | as the row above, or 451,879 tokens where the launcher moves the embedding table to the GPU (it says so at start) | yes, plus system RAM or an NVMe/SSD | no | system RAM, or up to 64 GiB of NVMe/SSD space, sized by the launcher for your host ([more cache](#extend-cache)) |
| One request longer than 262,144 tokens (experimental) | `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-512k` | 524,288 | up to 8 requests, 594,290 tokens in total: one full-length request plus short ones | yes | no | the model's long-context position scaling on every request; 2.4 GiB of pinned system RAM, no fallback; a cold 500K-token read takes about 6 minutes |

- **Context** counts prompt plus output: keep the prompt plus `max_tokens` within it, or the request is refused
  (HTTP 400).
- **Shared KV cache:** running requests share it; when it is full, further requests wait. **Prefix cache: yes**
  means that finished requests stay in that same cache until their space is needed, least recently used first: a
  new question about a document the server has already read starts in a few seconds (1.2 to 4.4 s in our runs)
  instead of 1.5 to 6 minutes, and the next turn of a conversation reads only its new part. **No:** every request
  reads its whole prompt again.
- **Images:** add `--vision` to a row with "yes"; not with MXFP4 `--mode long` or `--mode long-512k`. Images up to 4K are
  read at full resolution, larger ones are downscaled. For screenshots of code prefer `--mode long-kv4 --vision`: the
  FP8-cache `--mode long --vision` can misread single characters of tiny (about 15 px) text in 4K screenshots. For example
  `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long --vision`. The vision encoder's 0.88 GiB comes out of
  the shared cache (the "At once" figures above are without it): with `--vision` the cache holds 278,050 (3-bit,
  65,536-token row), 120,277 (MXFP4, 65,536-token row) or 253,298 (`--mode long`) tokens. Send images as
  OpenAI-style `image_url` content; a 1920 × 1080 screenshot costs about 2,000 tokens
  ([how to send one](REFERENCE.md#images-and-vision)).
- <a id="system-ram"></a>**System RAM.** `--mode long-kv4` and `--mode long-512k` keep the 2.4 GiB embedding table
  in pinned RAM. The launcher warns when less than 6 GiB of RAM is available at start (the server and the table
  need about that much): close programs, or with `--mode long-kv4` add `--no-system-memory-weights`
  (`--mode long-512k` needs the RAM and refuses that flag). A host with less than about 14.25 GiB
  of RAM in total cannot pin the table: there `--mode long-kv4` keeps it on the GPU and prints a `Note:` line at
  start. Requests still get 262,144 tokens and the prefix cache, but the cache holds 451,879 tokens (one
  full-length request plus short ones); `--no-system-memory-weights` chooses the same on purpose. `--mode long-512k`
  refuses to start there.
- **`--mode long-512k`** is experimental: it extends the model's position range to 524,288 tokens with the model's
  official long-context scaling, applied to every request in this mode, short ones included. Short-question scores
  stayed within noise of `--mode long-kv4`, and a small needle test found 4 of 4 planted facts at 300K and at 500K
  tokens. Its cache is only 4 % larger than that of `--mode long-kv4`: use it only for single requests above 262,144
  tokens. With a `--context` of 262,144 or less, the launcher serves exactly `--mode long-kv4` (without the scaling).

**For coding agents.** Start the server with `--mode long-kv4` and point the agent at `http://127.0.0.1:18982/v1`,
model `Qwen3.8`, any API key, context window 262,144 tokens (prompt plus output). An agent sends its whole
conversation again on every turn; the prefix cache reads only what is new. In a 20-turn session that grew from 50K
to 253K tokens, the first token came after 7.8 s on average from turn 2 on instead of 63.6 s, and three agents
sharing a 100K-token repository each started in 2.6 to 2.7 s ([measurements](#coding-mode-results)). Cache hits
need an unchanged start of the prompt: the same system prompt and earlier turns, no timestamp at the top. Each
response's `usage.prompt_tokens_details.cached_tokens` shows how many prompt tokens came from the cache. For MXFP4
accuracy with one agent, `run-mxfp4.sh --mode long` also keeps the prefix cache (200,000 tokens, one request at a
time, no images). To keep more documents than the GPU cache holds, also across restarts, add
[`--extend-cache`](#extend-cache).

Each row sets its own limits; to change them, see [Advanced options](#advanced-options).

<a id="extend-cache"></a>

### More cache for coding agents: `--extend-cache`

`--mode long-kv4` keeps its prefix cache on the GPU. When that is full, the oldest documents are dropped and the
next question about them reads them again. Add `--extend-cache` to keep them in system RAM or on an NVMe/SSD
instead: a document comes back from there in seconds, with the same output.

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 --extend-cache
```

| Option | What it does |
| --- | --- |
| `--extend-cache` (the same as `--extend-cache auto`) | Uses system RAM when the host has enough of it for RAM to add capacity beyond the GPU cache; otherwise an NVMe/SSD folder behind a small RAM staging tier. |
| `--extend-cache ram` | Always system RAM, also where it adds little capacity (the start line then says so). |
| `--extend-cache disk` | Always the NVMe/SSD folder: the most capacity (about 2.8M tokens at the 64 GiB cap), somewhat slower restores. |

What `auto` picks:

| Host RAM | `auto` picks | What you get |
| --- | --- | --- |
| 16 GB | NVMe/SSD with 4 GiB of RAM staging; the embedding table moves to the GPU (GPU cache 451,879 tokens) | documents up to about 220K tokens restore from disk |
| 24 GB | NVMe/SSD with 4.5 GiB of RAM staging | documents up to about 250K tokens restore from disk |
| 32 GB | RAM, a tier of about 15.5 GiB; the embedding table moves to the GPU | about 677K tokens of cache |
| 48 GB | RAM, a tier of about 21 GiB | about 917K tokens |
| 64 GB | RAM, about 29 GiB | about 1.27M tokens |
| 96 GB and more | RAM | about 1.9M tokens and up |

The token counts are computed from the 24 KB a cached token takes in a tier; only the 16 GB SSD restores below were
measured.

One line at start states the choice and the capacities.

**Measured** on a 16 GB host with a SATA SSD, `--extend-cache` (auto), after a full server restart:

| Document | Read cold | Restored from the SSD |
| --- | ---: | ---: |
| 130K tokens | 46.9 s | **6.8 s** |
| 200K tokens | 86.4 s | **9.0 s** |

Both outputs were identical to a fresh read. NVMe should be faster; we have not measured it.

In a session that keeps more documents than the GPU cache holds (six 100K-token documents asked in turn, twice,
16 GB host), every re-read came back from the SSD: 4.3 s to the first token instead of 33.5 s, all 12 answers
identical to a fresh read.

- **Same output.** A reopened document gives output bit-identical to the same document served from the GPU cache,
  at every length tested (32K to about 258K tokens). A hit the server cannot place is read again instead. Without
  `--extend-cache`, outputs are identical to the 3 October image.
- **RAM copies the GPU cache.** The RAM tier keeps a copy of what the GPU cache holds, so it adds capacity only when
  it is bigger than the GPU cache; `auto` accounts for that. A RAM tier of N GiB restores a recently read document
  of up to about N × 55K tokens; with several large documents competing, older ones are read again, correctly,
  just not faster.
- **The disk folder** is `PAITON_CACHE_DIR/kv-disk`, or `--disk-cache-dir DIR`. It is capped at 64 GiB or a quarter
  of the free space, whichever is smaller, checked at start; a folder that grew past its cap is removed at the next
  start. It stores about 24 KB per newly read prompt token (64 GiB holds about 2.8M new tokens); tokens served from
  the cache are not stored again. A restore passes through the RAM staging tier, which sets the largest document it
  restores (the table above); longer ones are read again. The staging tier also limits how many big documents come
  back at the same moment (4 GiB stages about two 100K-token documents at once); concurrent re-reads beyond that are
  read again. Spinning disks are refused.
- **First reads are a little slower.** With a cache tier, every new prompt block is also written to the tier, so
  the first read of a long prompt takes longer: about 8 % for a 131K-token prompt in our test (one run). Decode
  speed without a concurrent long prompt is unchanged within noise.
- **16 GB hosts.** A RAM tier with the embedding table in system RAM needs 32 GB, so the launcher refuses it there;
  with a tier it moves the table to the GPU. It refuses to start when free memory is short, naming the shortfall,
  and warns when memory is tight.
- **A RAM tier smaller than the GPU cache** (chosen with `--extend-cache ram` or `--host-cache-gib`) gets a warning at
  start: it keeps a copy of what the GPU already holds, so it rarely helps with several large documents in turn;
  `--extend-cache disk` keeps them across the session.
- **Only with `--mode long-kv4`** and the 4 October image or later. Older images refuse it.

More: [cache tiers in the reference](REFERENCE.md#cache-tiers).

## Start the server

Run the command you picked. The server runs in the foreground; add `--detach` to run it in the background
(`docker logs -f paiton-qwen38` follows its log). The first start takes about **three minutes** on our reference host (the image ships a
seeded compile cache; later starts about two minutes); wait for `/health` to succeed.

**API:** `http://127.0.0.1:18982/v1` · **Model:** `Qwen3.8` · **API key:** none (enter any value if a client asks)

Thinking is on by default in the 65,536-token rows and off in every `--mode long*` row. Each request can set it with
`"chat_template_kwargs":{"enable_thinking":true}` (or `false`); `--thinking on|off` changes the server default.
The example below and all benchmarks use thinking off; add `--thinking off` to match them.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

Send a prompt of hundreds of thousands of tokens from a file (`-d @request.json`) and give the client a timeout of
several minutes: reading 500K tokens for the first time takes about 6 minutes.

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
beyond the uncertainty; math and coding are within it. The 4-bit KV cache scores the same as FP8. On the 2 October
r1 image, 48/48 requests completed at each concurrency level (peak 30.85 GiB).
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

<a id="coding-mode-results"></a>

**Coding agents: `run-3bit.sh --mode long-kv4`, 3 October.** Compared with the same mode on the 2 October image,
which ran without prefix caching and with the embedding table on the GPU. 3-bit weights, thinking off.

| Capacity | 2 October image | `--mode long-kv4` | `--mode long-512k` |
| --- | ---: | ---: | ---: |
| Tokens per request | 262,144 | 262,144 | 524,288 |
| Shared KV cache | 458,922 tokens | **569,878 tokens** | **594,290 tokens** |
| Full-length requests at once | 1, plus short ones | **2** (two 261K-token requests, none preempted) | 1, plus short ones |
| Prefix cache | no | yes | yes |
| Embedding table in system RAM | – | 2.4 GiB | 2.4 GiB |

BetterBench, the full 20-pass run with the same tool and settings as the tables above:

| BetterBench row | 2 October image | `--mode long-kv4` | `--mode long-512k` |
| --- | ---: | ---: | ---: |
| Weighted single-stream decode | 174.3 tok/s | 173.4 tok/s (−0.5 %) | 173.3 tok/s (−0.6 %) |
| Speculative acceptance length | 4.098 | 4.103 | 4.132 |
| Aggregate output at 1 / 2 / 4 / 8 requests | 144.5 / 247.3 / 364.6 / 462.4 tok/s | 143.9 / 246.2 / 355.4 / 478.5 tok/s | 144.6 / 247.5 / 360.7 / 463.5 tok/s |
| Time to first token p50 at 1 / 2 / 4 / 8 requests (short prompts) | 86.7 / 126.3 / 134.3 / 149.7 ms | 87.1 / 124.8 / 132.4 / 147.8 ms | 86.9 / 126.9 / 135.9 / 152.8 ms |
| Cold prefill at 2K / 8K / 16K / 32K / 64K depth | 4,167 / 4,165 / 4,102 / 3,954 / 3,633 tok/s | 3,641 / 3,856 / 3,769 / 3,824 / 3,536 tok/s | 3,638 / 3,846 / 3,959 / 3,829 / 3,541 tok/s |


With prefix caching the server stops each prefill step where a later request can resume from the cache. On long
prompts that costs about 3 % (32K to 64K); short prompts pay a fixed 7 to 13 % (a 2K-token prompt takes two steps
instead of one). Decode and short-prompt latency are unchanged.

What the prefix cache buys in coding workloads. Both columns ran on a pre-release build of these configurations,
before the release cut the number of prefill steps with prefix caching; the left column is the 2 October form of
the mode (no prefix caching, the embedding table on the GPU):

| Workload | 2 October form (no prefix caching) | `--mode long-kv4` | Faster |
| --- | ---: | ---: | ---: |
| One conversation growing from 50K to 253K tokens over 20 turns: whole session | 1,237 s | **204 s** | **6.1×** |
| … mean time to first token (turns 2 to 20) / at turn 20 | 63.6 s / 120.8 s | **7.8 s / 11.2 s** | 8–11× |
| Three agents sharing a 100K-token repository prefix, 5 turns each: whole session | 655 s | **111 s** | **5.9×** |
| … first reply of each agent | 33 / 68 / 101 s | **2.6–2.7 s** | up to 38× |
| … later turns, mean time to first token | 84 s | **7.0 s** | 12× |
| Prompt tokens served from the cache | 0 % | 91–92 % | |
| New question about a 258K-token document already read | cold read 134 s | **2.7 s** (the same output as a cold read) | 49× |

Accuracy with standard-length questions, greedy, paired per question:

| Benchmark | `run-mxfp4.sh` | 2 October image | `--mode long-kv4` | `--mode long-512k` |
| --- | ---: | ---: | ---: | ---: |
| GSM8K 5-shot (1,319) | 95.68 | 95.53 | 95.45 | 95.53 |
| HumanEval pass@1 (164) | 95.12 | 92.07 | 92.68 | 94.51 |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 60.57 | 59.93 | 60.64 |
| Needle at 61,440 tokens (80) | 80/80 | 80/80 | 80/80 | 80/80 |
| Needles at 300K / 500K tokens (4 facts each) | – | – | – | 4/4 / 4/4 |

Each 3 October column is within noise of the column to its left (paired, p > 0.05 for every benchmark), and an
answer from the cache scores like a cold read (log-likelihood difference +0.008 nats per token, 95 % interval
−0.055 to +0.083; the same top token at 59 of 60 positions). The 3 October columns and this cache check ran on
pre-release builds of these configurations. The MMLU-Pro gap to MXFP4 is the 3-bit weights' own gap, as in the
65,536-token rows. Over five 131K-token documents the 4-bit cache predicts the text as well as the FP8 cache at
every depth (pre-release build).

`--mode long-512k`: reading a 300K-token document cold took 158 s, a 500K-token one 361 s; follow-up questions
took 2.8 to 2.9 s and 4.2 to 4.4 s with 297,600 and 497,600 tokens from the cache. Its weighted BetterBench decode
matches `--mode long-kv4`, but the chat category decoded 16 % slower than on the 2 October image (116.1 against
137.9 tok/s, one run).
[Coding mode details](REFERENCE.md#coding-mode) · [`--mode long-512k` details](REFERENCE.md#mode-long-512k).

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
| `--max-num-seqs COUNT` | Fewer concurrent requests than the row's (1 to 8; above the row's default, such as the 1 of MXFP4 `--mode long` and `--profile desktop`, is untested). |
| `--long-prefill-threshold TOKENS|off` | Experimental: cap on the prefill tokens a long prompt takes per step so short requests from other users are answered while it is processed. Off by default; it changes long-prompt outputs slightly and its accuracy is not yet validated. |
| `--thinking on` / `--thinking off` | The server default for thinking (on in the 65,536-token rows, off in the `--mode long*` rows). Requests can override it. |
| `--kv-cache fp8` | `run-3bit.sh` without `--mode`: the FP8 instead of the 4-bit KV cache (250,578 tokens). |
| `--no-system-memory-weights` | `--mode long-kv4` with nothing pinned, for a host short of free RAM: the embedding table stays on the GPU and the cache holds 451,879 tokens (one full-length request plus short ones); prefix caching stays on. |
| `--extend-cache [auto\|ram\|disk]` | `--mode long-kv4`: keep prefix-cache blocks that leave the GPU in system RAM or on an NVMe/SSD; see [more cache](#extend-cache). |
| `--kv-cache-memory-bytes BYTES` / `--gpu-memory-utilization FRACTION` | Your own KV budget instead of the measured one, also with `--vision`. |
| `--profile desktop` | For an R9700 that also drives your desktop: 32,768 context, one request, 2 GiB KV. |
| `--port PORT` / `--name NAME` / `--detach` | API port (default 18982), container name, run in the background. |
| `--devices N` / `--list-gpus` | The GPU to run on (default: the first R9700) and the list to choose from. |
| `--ignore-vram-check` | Start although VRAM is already in use on the selected GPU (the launcher otherwise waits up to 30 s for a stopping container, then refuses with the amount in use). |

Command lines from earlier releases still start the same server:

| Earlier command | Same as |
| --- | --- |
| `run-3bit.sh --context 262144` | `run-3bit.sh --mode long` |
| `run-3bit.sh --context 245000 --vision` | `run-3bit.sh --mode long --vision` |
| `run-mxfp4.sh --context 200000` | `run-mxfp4.sh --mode long` |
| `run-3bit.sh --context 262144 --kv-cache kv4` | the 2 October `--mode long-kv4`: no prefix caching, 458,922 tokens (the launcher prints a note) |
| `run-rocm10.sh` | `run-3bit.sh` when `PAITON_W3ROT_DIR` is set, `run-mxfp4.sh` otherwise |

Experimental options, such as setting the cache tiers by hand (`--host-cache-gib`, `--disk-cache-dir`), are in
the [reference](REFERENCE.md#system-memory-and-experimental-options); no row needs them.
See the reference for [GPU selection and memory budgets](REFERENCE.md#gpu-context-and-memory-controls) and
[long-context details](REFERENCE.md#long-context-200k-and-220k).

## Advanced setup and release history

[Release reference](REFERENCE.md) contains image pins, release notes, detailed
benchmarks, long-context checks, KV tuning and older releases.

### Run an earlier release

Add `--image` with the exact reference below to the command of your row. The weights stay the same; give each image
its own runtime cache folder. With an earlier reference, the launcher builds the Docker command of that release;
`--dry-run` prints it without starting anything. Copy the reference exactly, including `@sha256:`, so that
Docker runs exactly that image. The launcher recognises an earlier image by its tag (`…-20261004-r1`, `…-20261003-r1`,
`…-20261002-r1s`, `…-20260929-r2`), with or without the digest and also under a local re-tag that keeps the tag,
and by its digest alone; an image under any other tag or an image ID counts as the current image.

```bash
# 4 October r1, the previous release: every row except images in the coding mode (--mode long-kv4 --vision)
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261004"
mkdir -p "$PAITON_CACHE_DIR"
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 \
  --image ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261004-r1@sha256:6a97d65fda17c1c48b36d3423c6f3709bd65a3849227e45a8a24a552d4c81b9d

# 3 October r1: every row; refuses --extend-cache
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261003"
mkdir -p "$PAITON_CACHE_DIR"
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 \
  --image ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261003-r1@sha256:fb71b59eb29f3341dd10e9972920e75073a03fefc7bf6d2f2e1966b91a730f53

# 2 October (r1s): every row except --mode long-512k
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261002"
mkdir -p "$PAITON_CACHE_DIR"
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh \
  --image ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261002-r1s@sha256:a1c1025052f84a009428709bfe7e9431281ab5d5c0f723a49d79eecafe519dad

# 29 September r2: the 65,536-token rows and --mode long, also with --vision
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20260929"
mkdir -p "$PAITON_CACHE_DIR"
bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh \
  --image ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2@sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045
```

To return to the current release, run `export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261005"` again
and drop `--image`.

The 3 October image runs every row as it was released. With the 4-bit cache it refuses the RAM and SSD cache tiers
(`--extend-cache`, and `--host-cache-gib` with `--mode long-kv4`): there a cache hit could resume from the wrong
recurrent state. `--host-cache-gib` with `--mode long` (FP8 cache) works there as before.

On the 2 October image, `--mode long-kv4` runs as it was released: no prefix caching, the embedding table on the
GPU, a 458,922-token cache. The 29 September image refuses `--mode long-kv4`; both refuse `--mode long-512k` and
`--host-cache-gib`.

### Native serving

Already have the exact supported vLLM environment? `paiton serve qwen38-nvfp4` serves a separate preset: text
only, 65K, FP8 KV, without DFlash2 or the 3-bit weights. Follow [native setup](REFERENCE.md#native-serving) for
installation. Its bundle is `qwen38-rocm10-native-20260921`, profile `qwen38-nvfp4-w4a8-text-65k`, with the same
pinned target revision above.

[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[3-bit weight license and calibration](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
