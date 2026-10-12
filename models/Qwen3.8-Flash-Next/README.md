# Qwen3.8 Flash Next on two R9700 (draft, local staging; not published)

Paiton's Flash Next package, two checkpoints served by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin (native grouped
MoE, native collectives, native hidden-chain and sampled MTP drafts):
- **3-bit** (`EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4`): the fastest; everything in VRAM.
- **4-bit** (`EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4`): the most accurate (within 0.16 % of the bf16 log-likelihood on long documents,
  needles 40/40 identical to bf16); part of the experts is served from pinned system RAM, so it needs a host with about 256 GB of RAM.
Model details, accuracy and format: the two model cards.

## About this release
The second release of this model on Paiton: the 4-bit container with the expert host tier, and launcher fixes for the 3-bit modes. We keep
tuning, we keep optimizing. No custom engine was built: this is regular vLLM with the Paiton plugin and our own kernels.

## What you get (official numbers, measured on the shipped image through this launcher; BetterBench 0.6.0 standard profile, exact bf16 wire, cold prefix cache, 2x R9700)

<!-- [[REL3S]]: the 3-bit rows are the published 10 Oct numbers (same image and config); the morning validation with the new launcher confirms them -->

| 4-bit (`--mode 4bit`, `--mode 4bit-200k`) | |
|---|---:|
| Weighted single-stream decode | 189.7 tok/s |
| Eight concurrent requests, aggregate | 420.7 tok/s (48/48) |
| Prefill, 64K-token prompt (BetterBench 64K row) | 6,991 tok/s |
| Prefill, exact 64,000-token prompt, cold cache | 7,162 tok/s |
| TTFT p50, short prompts | 121 ms |
| Update p99 | 17.7 ms per stream update |

4-bit per-category decode (tok/s): chat 171.4, code 187.1, file_edit 207.5, json 217.5, math 217.2, prose 170.4, reasoning 170.0,
summarization 206.1. Prefill by depth (tok/s): 2K 5,564 / 8K 6,771 / 16K 6,641 / 32K 7,044 / 64K 6,991 / 128K 6,857. The 4-bit modes use
an fp8 (E4M3) KV cache; `--mode 4bit` (MTP depth 3, 98,304-token window) also uses the INT6 trunk shipped in the 4-bit repository,
`--mode 4bit-200k` (200,000-token window, MTP off) the 8-bit trunk. Measured on a PCIe 4.0 x16 host with 251 GiB of RAM.

| 3-bit (`--mode decode`, `--mode prefill-long`) | |
|---|---:|
| Weighted single-stream decode | 216.3 tok/s |
| Eight concurrent requests, aggregate | 565.1 tok/s (48/48; ladder 1/2/4/8 = 204.6 / 315.1 / 449.2 / 565.1) |
| Prefill, 64K-token prompt | 8,161 tok/s (exactly 64,000 tokens: 8,212) |
| TTFT p50, short prompts | 117 ms |
| Update p99 | 13.9 ms per stream update (MTP: ~2.9 tokens per update) |

Per-category decode (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1,
summarization 239.8. Prefill by depth (tok/s): 2K 7,925 / 8K 8,321 / 16K 8,311 / 32K 8,232 / 64K 8,161 / 128K 7,928.
Decode and concurrency: `--mode decode` (MTP depth 3, 98,304-token window). Prefill: `--mode prefill-long` (200,000-token window,
MTP off). Both modes use the bf16 KV cache (no released mode uses fp8 KV). Both modes also exist with exact-arithmetic prefill, about 7 % slower (`--mode decode-nopf`, `--mode prefill-long-nopf`;
prefill @64K 7,607 measured on the development stack, not re-run on the image). Served long-context needles: 40/40 at 131K and at 200K.

**Prefix caching (opt-in, `--prefix-caching`, 3-bit modes; the 4-bit modes use `--mode 4bit-cache`)**: align-mode caching of the recurrent state per 2,048-token block. A repeated 64K-token
prompt goes from 9.6 s to 0.35 s time-to-first-token with identical output. Off by default because the official numbers were measured
without it.

## Long context
Two modes: `decode` (98,304-token window, speculative decoding on) and `prefill-long` (200,000-token window, speculative decoding off).
Tested end to end at 200K: needle retrieval at 200,000 tokens 40/40, identical to bf16; prefix caching (opt-in `--prefix-caching`) with
byte-identical hits (a 128K repeat goes from 20 s to 0.41 s time-to-first-token); prefill measured to 128K (see BENCHMARKS.md);
single-stream decode in the 200K mode (`prefill-long`, speculation off): 105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token
prompts (TTFT 24.8 s at 190K); 200K with speculation on does not fit the stock pool.

## How the numbers were taken
BetterBench 0.6.0, standard profile, on this machine (2x R9700) against our OpenAI-compatible server; prefill with exact bf16
activations on the wire (no compressed wire); 20 passes per category after warm-up, unique prompts (no prefix-cache hits), concurrency
1/2/4/8 x 48 requests, prefill depths 2K-128K.

## Next release
We keep tuning, we keep optimizing.

## Prerequisites
- Two AMD Radeon AI PRO R9700 (32 GiB each) on a ROCm 10 host driver; Docker with GPU device access (`/dev/kfd`, `/dev/dri`; groups `video`, `render`).
- `python3` 3.10 or newer and the Hugging Face hub package: `pip install huggingface_hub` (the launcher downloads the weights and the base
  model's tokenizer files through it; the `hf` / `huggingface-cli` binaries are only a fallback).
- Disk: about 116 GB for the weights plus a few MB for the runtime view and ~0.3 GB per mode of compile cache; the image is 41 GB.
- 4-bit modes only: about 96 GiB of pinned host memory (the n-gram table and other host tables, about 55 GiB, plus about 41 GiB of expert
  copies). The benchmarked configuration keeps it in GPU-mapped GTT memory, which needs about 98 GiB of free GTT; the GTT limit is half
  the RAM by default, so in practice **a host with 256 GB of RAM**. On smaller hosts the launcher falls back to ROCm's user-pointer
  memory and says so: it runs, but slower (186.0 vs 191.9 tok/s single-stream, 388.9 vs 423.1 at 8 requests in our validation) and can
  pause for several seconds when the kernel compacts host memory. Raising the GTT limit is a system setting the host's administrator may
  choose; the launcher never changes it. It also warns when the two GPUs have no direct peer-to-peer link.
- `launch-flashnext.py --dry-run` checks these prerequisites and prints exactly what is missing.

## Set up
- `pip install huggingface_hub`; the first start downloads and verifies the weights (SHA256SUMS) into `--weights` (default `~/paiton-models/Qwen3.8-Flash-Next-W3A8`).
- `docker pull ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1` (digest `sha256:c7e76bc0d7d9f8a3d575b940d23f9b5219eada13b1960e10f3aaa996b5ad9191`, pinned in modes.json; the launcher pulls it by digest if missing).

## Ways to run

### Start the server

Pick a row in [What you get](#what-you-get) and run its wrapper with the weights downloaded (the first run downloads and verifies them):

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext.sh            # decode mode (default): 98,304-token window, speculative decoding on
bash models/Qwen3.8-Flash-Next/run-flashnext-200k.sh       # 200,000-token mode (prefill-long): speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-exact.sh      # exact-arithmetic prefill, about 7 % slower
bash models/Qwen3.8-Flash-Next/run-flashnext-cached.sh     # decode mode with the prefix-caching opt-in
```

The wrappers call the launcher; the same thing by hand:

```bash
python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode decode|prefill-long|decode-nopf|prefill-long-nopf [--prefix-caching] [--dry-run]
```

The server runs in the foreground; add `--detach` to run it in the background (`docker logs -f paiton-flashnext` follows its log). The
first start of a mode copies the baked compile cache and takes about **five to six minutes** (weight loading dominates). Wait for the
launcher's `launcher: READY on port …` line (with `--detach` the command returns at that point): after `/health` answers, the launcher
drops the checkpoint files from the page cache and waits until the kernel has finished compacting host memory (at most 30 s), because
that compaction can pause the GPUs for a few seconds right after start. `--settle-s 0` and `--keep-page-cache` turn this off.

**API:** `http://127.0.0.1:18982/v1` · **Model:** `Qwen3.8-Flash-Next` · **API key:** none (enter any value if a client asks)

**4-bit experts (more accurate, slower; uses system RAM):**

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh        # 4-bit experts with the expert host tier: 98,304-token window, speculative decoding on
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-200k.sh   # 4-bit experts, 200,000-token window, speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-cache.sh  # opt-in cache mode for agents and coding tools: prefix caching + 8 GiB RAM prefix tier
```

They download the 4-bit weights (`EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4`, about 133 GB) into `~/paiton-models/Qwen3.8-Flash-Next-W4A8`.
About 20.8 GiB of experts per GPU stay in VRAM (the most-used ones); the rest are read from pinned system memory when a token needs them
(`--expert-vram-gib` and `--host-pool on|off` change the split; the launcher checks the RAM and the GTT size before it starts).

**4-bit cache mode (opt-in, `--mode 4bit-cache`).** For agents and coding tools, which send the whole conversation again on every turn:
the 4-bit decode mode with prefix caching (2,048-token blocks) and an 8 GiB RAM prefix tier that keeps prefix blocks evicted from VRAM in
pinned system memory. Repeated long prompts start in about 1 s instead of 6-16 s (32K 6.3 -> 1.0 s, 64K 12.3 -> 1.1 s, 90K 16.2 -> 1.1 s to the first
token); hits from the RAM tier give the same output as a cold start. It pins about 8 GiB more host memory (about 108 GiB of free GTT on the first GPU, in practice a 256 GB host); without that the launcher leaves the RAM tier
off, says so, and keeps prefix caching in VRAM. The cost: about 22 % lower multi-user throughput than `--mode 4bit` (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. The official numbers above are measured in `--mode 4bit`; `--prefix-caching`
on `--mode 4bit` / `4bit-200k` is refused (not measured there).

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true}'
```

- **Stop** with `docker stop paiton-flashnext` (or the name of the wrapper you started) before switching to another mode.
- **Long prompts:** send prompts of 100K+ tokens from a file (`-d @request.json`) with a client timeout of a few minutes (a 190K-token prompt
  takes about 25 s to read in the 200K mode).

## Options
`--mode decode|prefill-long|decode-nopf|prefill-long-nopf|4bit|4bit-200k|4bit-cache`, `--prefix-caching`, `--host-pools auto|gtt|userptr`, `--kv-cache-memory-bytes`, `--expert-vram-gib`, `--host-pool on|off`, `--port`, `--name`, `--weights DIR`, `--cache DIR`, `--image REF`, `--devices GPU-...,GPU-...`,
`--served-model-name`, `--detach`, `--settle-s`, `--keep-page-cache`, `--dry-run`. GPU selection detects AMD devices by vendor id; two are required.
Host pools: the 3-bit modes keep ROCm's user-pointer memory for their ~55 GiB of host tables (as published); `--host-pools gtt` is an
opt-in that removes the occasional memory pauses under load and costs ~1 GiB VRAM in decode. The 4-bit modes use GTT memory (see Prerequisites).

## Files
`run-flashnext*.sh` (the wrappers above), `launch-flashnext.py` (also builds the runtime view `<weights>-view` once: base config + tokenizer at the pinned revision, see RELEASE-NOTES.md), `modes.json` (environment + engine arguments of the modes and the prefix-caching opt-in), `README.md`, `RELEASE-NOTES.md`, `BENCHMARKS.md` (depth table,
per-category decode, host conditions), `THIRD_PARTY_NOTICES.md`, `runtime.lock.json` and `checkpoint.lock.json` (what this release was built from and tested with),
`Dockerfile` + `prepare_image_context.py` + `build-context.lock.json` (rebuild the image from locked inputs, see REPRODUCE.md), `release-audit.json`,
`deployment-check.json`, `metrics.json` and `benchmarks/` (the raw BetterBench outputs of the published run).

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
  prefixes in a RAM tier. Lowering `--kv-cache-memory-bytes` below the mode value shrinks the cache further, and the launcher says by how much.
- **Waiting requests.** `Waiting: N reqs` above 0 means the server is at its concurrency (8 requests) or its KV cache is full; requests
  start as others finish.

## Limitations (draft)
- Known issue (3-bit, draft for approval): In very long agentic sessions (beyond ~30K tokens) the 3-bit can end replies early. Use `--prefix-caching`, and prefer the 4-bit for long coding sessions; a fix is planned for the next 3-bit release.
- 98,304-token window in decode mode (MTP pool); 200,000 in prefill-long mode with MTP off; vision not validated; prefix caching is an
  opt-in flag (official numbers measured without it). The prompt rows run W8A8; accuracy numbers are in the model card's table. The `-nopf` modes use
  exact-arithmetic prefill, about 7 % slower.
