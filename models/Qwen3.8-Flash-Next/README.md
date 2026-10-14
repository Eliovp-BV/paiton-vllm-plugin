# Qwen3.8 Flash Next on two R9700

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in two Paiton releases for two AMD Radeon AI PRO R9700, served
by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin. One launcher runs both; pick a page:

- **[3-bit: the fastest](README-3bit.md)**: everything in VRAM.
- **[4-bit: the accuracy release](README-4bit.md)**: the 16-bit model's scores; part of the experts in system RAM.

## Which one do I want?

| | [3-bit](README-3bit.md) | [4-bit](README-4bit.md) |
|---|---|---|
| Choose it for | Speed | Accuracy |
| Task benchmarks (16-bit: GSM8K 96.9 %, HumanEval 96.3 %, MMLU-Pro 84.7 %) | not measured yet | **GSM8K 96.8 %, HumanEval 96.3 %, MMLU-Pro 83.7 %** |
| Long documents (NLL vs bf16) / needles | +0.55 % / 40 of 40 at 131K and 200K | **+0.16 %** / 40 of 40 at 131K, identical to bf16 |
| Single-request decode | **216 tok/s** | ~190 tok/s |
| 8 concurrent requests | **565 tok/s** | 421 tok/s |
| Prefill, 64K-token prompt | **8,161 tok/s** | 6,991 tok/s |
| Weights | 116 GB, all in VRAM | 133 GB, about 20.8 GiB of experts per GPU in VRAM, the rest in pinned system RAM |
| System RAM | 128 GB or more | 256 GB |
| Context window | 98K (speculative decoding on), 200K (off) | 98K or 262K (on), 200K (off) |
| Agents and coding tools | `--prefix-caching` | `--mode 4bit-cache` |
| Start | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh` | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh` |
| Hugging Face | [W3A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4) | [W4A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4) |

Speed: BetterBench 0.6.0 standard profile on the shipped images through this launcher ([how](#how-the-numbers-were-taken)). Task
benchmarks: same prompts, sampling seeds and thinking-on template for the 4-bit and the 16-bit model; method and significance on the
[4-bit page](README-4bit.md#accuracy). No custom engine was built: this is regular vLLM with the Paiton plugin and our own kernels.
We keep tuning, we keep optimizing. What changed in this release: [RELEASE-NOTES.md](RELEASE-NOTES.md).

## Prerequisites
- Two AMD Radeon AI PRO R9700 (32 GiB each) on a ROCm 10 host driver; Docker with GPU device access (`/dev/kfd`, `/dev/dri`; groups
  `video`, `render`). System RAM per release: see the table above.
- `python3` 3.10 or newer and the Hugging Face hub package: `pip install huggingface_hub` (the launcher downloads the weights and the
  base model's tokenizer files through it; the `hf` / `huggingface-cli` binaries are only a fallback).
- Disk: the weights (table above), a few MB for the runtime view and ~0.3 GB per mode of compile cache; each image is about 41 GB.
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

## Options
`--mode decode|prefill-long|decode-nopf|prefill-long-nopf|4bit|4bit-200k|4bit-cache|4bit-long|4bit-xl`, `--prefix-caching`, `--host-pools auto|gtt|userptr`,
`--kv-cache-memory-bytes`, `--expert-vram-gib`, `--ple-host auto|pinned|pageable`, `--port`, `--host ADDRESS`, `--api-key KEY`, `--name`,
`--weights DIR`, `--cache DIR`, `--image REF`, `--devices GPU-...,GPU-...`, `--served-model-name`, `--detach`, `--settle-s`,
`--keep-page-cache`, `--dry-run`. GPU selection detects AMD devices by vendor id; two are required. The launcher picks the right image
per mode (3-bit and 4-bit run different images).

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
