# Qwen3.8 Flash Next 3-bit on two R9700: the fastest

[Overview: 3-bit or 4-bit?](README.md) · [Release notes](RELEASE-NOTES.md) · [Benchmarks](BENCHMARKS.md) ·
weights: [EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4)

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with mixed 3-bit experts, served by vLLM 0.29 on ROCm 10
with tensor parallel 2 and the Paiton plugin (native grouped MoE, native collectives, native hidden-chain and sampled MTP drafts).
Everything runs from VRAM: **216 tok/s for a single request, 565 tok/s at 8 requests, 8,161 tok/s prefill at 64K.**
No custom engine: regular vLLM with the Paiton plugin and our own kernels.

This release: the same image, weights, modes, speed and accuracy as the 10 October release, with launcher fixes (below).

## Accuracy
Negative log-likelihood on 44 held-out long documents (8K-32K tokens): 1.7671 against bf16's 1.7574 (+0.55 %). Needle retrieval as
served: 40 of 40 at 131K and at 200K tokens, identical to bf16. Full tables (KL, top-1 agreement, per-length buckets): the
[model card](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4). The prompt rows run W8A8.

## Speed
Official numbers, measured on the shipped image through this launcher (BetterBench 0.6.0 standard profile, exact bf16 wire, cold
prefix cache, 2x R9700):

| `--mode decode` / `--mode prefill-long` | |
|---|---:|
| Weighted single-stream decode | 216.3 tok/s |
| Eight concurrent requests, aggregate | 565.1 tok/s (48/48; ladder 1/2/4/8 = 204.6 / 315.1 / 449.2 / 565.1) |
| Prefill, 64K-token prompt | 8,161 tok/s (exactly 64,000 tokens: 8,212) |
| TTFT p50, short prompts | 117 ms |
| Update p99 | 13.9 ms per stream update (MTP: ~2.9 tokens per update) |

Per-category decode (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1,
summarization 239.8. Prefill by depth (tok/s): 2K 7,925 / 8K 8,321 / 16K 8,311 / 32K 8,232 / 64K 8,161 / 128K 7,928.
Decode and concurrency: `--mode decode` (MTP depth 3, 98,304-token window). Prefill: `--mode prefill-long` (200,000-token window,
MTP off). Both modes use the bf16 KV cache. How the numbers were taken: [overview](README.md#how-the-numbers-were-taken).

## Long context
- `decode`: 98,304-token window, speculative decoding on; `prefill-long`: 200,000-token window, speculative decoding off.
- Tested end to end at 200K: needles 40/40 at 200,000 tokens, identical to bf16; prefix caching with byte-identical hits (a 128K repeat
  goes from 20 s to 0.41 s time-to-first-token); prefill measured to 128K ([BENCHMARKS.md](BENCHMARKS.md)).
- Single-stream decode in the 200K mode: 105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K).
  200K with speculative decoding on does not fit the KV pool.

## Run it
Requirements: [Hardware](#hardware) below and the common [prerequisites](README.md#prerequisites). The first start downloads and
verifies the weights (SHA256SUMS) into `~/paiton-models/Qwen3.8-Flash-Next-W3A8` (`--weights DIR` to change it).

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext.sh            # decode mode (default): 98,304-token window, speculative decoding on
bash models/Qwen3.8-Flash-Next/run-flashnext-200k.sh       # 200,000-token mode (prefill-long): speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-exact.sh      # exact-arithmetic prefill, about 7 % slower
bash models/Qwen3.8-Flash-Next/run-flashnext-cached.sh     # decode mode with prefix caching (agents and coding tools)
```

By hand: `python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode decode|prefill-long|decode-nopf|prefill-long-nopf [--prefix-caching] [--dry-run]`.

**Agents and coding tools: add `--prefix-caching`.** An agent resends the whole conversation every turn; with the prefix cache the
shared part is reused instead of read again: a repeated 64K-token prefix reaches its first token in 0.35 s instead of 9.6 s, with
identical output. It is opt-in because the speed table is measured without it (the benchmark never repeats a prefix). The launcher
warns when the KV cache (about 112K tokens in `decode`) is small relative to the window.

**Exact-arithmetic prefill:** `--mode decode-nopf` / `--mode prefill-long-nopf` use the exact prefill path, about 7 % slower (prefill
@64K 7,607 on the development stack); decode is identical.

The first start of a mode takes about five to six minutes; wait for `launcher: READY on port …`. **API:** `http://127.0.0.1:18982/v1`
(or `http://<this-machine-ip>:18982/v1`) · **Model:** `Qwen3.8-Flash-Next`. Network, API key, stopping and troubleshooting:
[overview](README.md#use-the-server).

## Hardware
- Two AMD Radeon AI PRO R9700 (32 GiB each), ROCm 10 host driver.
- **System RAM: 128 GB or more.** The host tables (n-gram table and others, about 55 GiB over both GPUs) live in pinned system
  memory. By default they use ROCm's user-pointer memory, as published; `--host-pools gtt` puts them in GPU-mapped GTT memory (needs
  about 120 GiB of RAM or more), which removes the occasional memory pauses under load and costs ~1 GiB VRAM in decode.
- Disk: about 116 GB for the weights; the image is 41 GB: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2`
  @ `sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b` (pinned in modes.json; the launcher pulls it by digest).
- Peak VRAM per card: 30.3 GiB in `decode`, 31.5 GiB in `prefill-long` under 64K-128K prefill load.

## Known issues and limitations
- In very long agentic sessions (beyond ~30K tokens) the 3-bit can end replies early. Use `--prefix-caching`, and prefer the
  [4-bit](README-4bit.md) for long coding sessions; a fix is planned for the next 3-bit release.
- 98,304-token window in `decode` (MTP pool), 200,000 in `prefill-long` with MTP off. Vision is not validated.
