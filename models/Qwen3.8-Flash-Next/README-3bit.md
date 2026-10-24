# Qwen3.8 Flash Next 3-bit on two R9700: the fastest

[Overview: 3-bit or 4-bit?](README.md) · [Release notes](RELEASE-NOTES.md) · [Benchmarks](BENCHMARKS.md) ·
weights: [EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4)

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with mixed 3-bit experts, served by vLLM 0.29 on ROCm 10
with tensor parallel 2 and the Paiton plugin (native grouped MoE, native collectives, native hidden-chain and sampled MTP drafts).
Everything runs from VRAM: **216 tok/s for a single request, 565 tok/s at 8 requests, 8,161 tok/s prefill at 64K.**
No custom engine: regular vLLM with the Paiton plugin and our own kernels.

This release: the default commands serve the same image, weights, modes, speed and accuracy as before (torch worker). New, opt-in:
the native Paiton worker, `--mode decode --fn-worker native` ([below](#native-worker-opt-in)).

## Accuracy
New in this release: more accurate 3-bit weights (Hugging Face revision `71faf406`), a drop-in replacement for the first
3-bit release (`829b089b`): same format, file sizes, kernels and speed. They are the default download from this release on; on a host with the first release, the launcher downloads them into a fresh folder (`~/paiton-models/Qwen3.8-Flash-Next-W3A8-71faf406`), never mixed with the first release, and says once that the old folder can be deleted; an explicit `--weights DIR` with other weights is served as is, with a warning.

| Against the bf16 model | New 3-bit | First 3-bit (`829b089b`) | [4-bit](README-4bit.md) |
|---|---:|---:|---:|
| GSM8K / HumanEval / MMLU-Pro (16-bit: 96.9 / 96.3 / 84.7 %) | 97.4 / 94.5 / 83.6 % | 96.8 / 95.7 / 84.2 % | 96.8 / 96.3 / 83.7 % |
| KL t+a, free / routing locked | **0.0550 / 0.0367** | 0.0604 / 0.0413 | 0.0306 / 0.0166 |
| Top-1 agreement | **91.18 %** | 90.74 % | 93.36 % |
| Long documents (44 held-out, 8K-32K tokens), NLL vs bf16 1.7574 | **1.7636 (+0.35 %)** | 1.7671 (+0.55 %) | 1.7603 (+0.16 %) |
| Served KL on two R9700 (top-64; paired difference −0.0061 [−0.0124, −0.0012]) | **0.0563** | 0.0624 | – |
| Needles at 131K tokens | 25 of 25 prompts | 40 of 40 (also 40 of 40 at 200K) | 40 of 40 |
| AI2D with images, emulated (16-bit 85.3 %; not served in the 3-bit modes) | **81.8 %** | 77.7 % | – |

Task scores of the two 3-bit weight sets do not differ significantly (new vs first: GSM8K +0.6, HumanEval −1.2, MMLU-Pro
−0.6 points; paired p 0.17 / 0.50 / 0.38); both are within 1.8 points of the 16-bit model, with no significant difference. AI2D: +4.2 points
[+1.8, +6.5], McNemar p 0.0008, measured with an emulation of the served arithmetic (the 3-bit modes take no image input).

Full tables and method: the [model card](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4). The prompt rows run
W8A8.

## Speed
Official numbers of the default (torch worker, the checkpoint's own drafter), measured on the shipped image through this launcher
(BetterBench 0.6.0 standard profile, exact bf16 wire, cold prefix cache, 2x R9700):

| `--mode decode` / `--mode prefill-long` (default, torch worker) | |
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

## Native worker (opt-in)
`--mode decode --fn-worker native` serves `--mode decode`'s configuration (98,304-token window, MTP depth 3) through the native
Paiton worker, with a KV cache of about 551K tokens (about 112K on the torch default).

| `--mode decode --fn-worker native`, new 3-bit weights, 24 Oct | Distilled drafter B4gh5-mtpft (default) |
|---|---:|
| Weighted single-stream decode | 258.7 tok/s |
| TTFT p50, short prompts | 56 ms |
| Prefill, 64K-token prompt (in this mode) | 8,433 tok/s |
| Eight concurrent requests, aggregate | 687.5 tok/s |
| Four concurrent requests, aggregate | 532.6 tok/s |

The first 3-bit weights measured 272.5 / 56 ms / 8,392 / 685.4 / 531.0 (17 Oct); `--drafter stock` (the checkpoint's own
drafter) 249.9 / 56 ms / 8,401 / 659.4 / 516.4. Before and after: [BENCHMARKS.md](BENCHMARKS.md).

No image input, request-API limits (no logprobs, structured outputs, penalties, ...) and the host requirement (about 60 GiB of free
GTT): [overview](README.md#native-worker-opt-in). From 20261024 on, every native mode runs
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc3` @ `sha256:5bca558833ad230ab1e8db7e6a64f538a7396c44dd660b69716a5aec4e714640` (the 20261017-r1 image plus
the prefix-caching layers; this mode's executor and files are unchanged).

**Prefix caching on the native worker (new in 20261024-pc-rc2): `--mode decode --fn-worker native --prefix-caching`** switches to
`--mode 3bit-cache-native` (98,304-token window); `--mode 3bit-long-cache-native` has a 262,144-token window. Both: prefix caching
with 2,048-token blocks, speculative decoding off, and a RAM prefix tier sized from the host (`--native-kv-tier auto`, up to 16 GiB
per GPU). A repeated 90K-token prompt reaches its first token in 0.32 s instead of 10.53 s (32K: 0.27 s instead of 3.70 s; from the
RAM tier 0.42 s instead of 10.48 s; 200K document in `3bit-long-cache-native`: 0.42 s instead of 24.53 s). Speed without prefix hits
(BetterBench standard profile): `3bit-cache-native` 101.0 tok/s single-request decode, 403.4 tok/s at 8 requests, 8,731 tok/s prefill at 64K (`--mode decode --fn-worker native` with speculative decoding: 272.5 / 685.4 / 8,392); `3bit-long-cache-native` 100.5 / 401.5 / 8,725. `--mode 3bit-long-cache-mtp-native` (new in 20261024-pc-rc2) keeps speculative decoding on with prefix caching: 262,144-token window, 213.5 tok/s single-request decode, 611.0 tok/s at 8 requests with the RAM tier on (250.0 / 657.7 with `--native-kv-tier off`), 200K-token document 25.81 → 0.68 s; it needs the same host as `--mode decode --fn-worker native`. The other two cache modes start from about 63 GiB of RAM (RAM
tier off at 64 GiB, 16 GiB per GPU at 128 GiB). Details: [overview](README.md#prefix-caching-on-the-native-worker).

## Long context
- `decode`: 98,304-token window, speculative decoding on; `prefill-long`: 200,000-token window, speculative decoding off.
- Tested end to end at 200K: needles 40/40 at 200,000 tokens, identical to bf16; prefix caching with byte-identical hits (a 128K repeat
  goes from 20 s to 0.41 s time-to-first-token); prefill measured to 128K ([BENCHMARKS.md](BENCHMARKS.md)).
- Single-stream decode in the 200K mode: 105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K).
  200K with speculative decoding on does not fit the KV pool.

## Run it
Requirements: [Hardware](#hardware) below and the common [prerequisites](README.md#prerequisites). The first start downloads and
verifies the weights (SHA256SUMS) into `~/paiton-models/Qwen3.8-Flash-Next-W3A8-71faf406` (`--weights DIR` to change it).

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext.sh            # decode mode (default): 98,304-token window, speculative decoding on
bash models/Qwen3.8-Flash-Next/run-flashnext-200k.sh       # 200,000-token mode (prefill-long): speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-exact.sh      # exact-arithmetic prefill, about 7 % slower
bash models/Qwen3.8-Flash-Next/run-flashnext-cached.sh     # decode mode with prefix caching (agents and coding tools)
bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native   # opt-in: decode mode on the native worker
bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native --prefix-caching   # native prefix caching (3bit-cache-native)
```

By hand: `python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode decode|prefill-long|decode-nopf|prefill-long-nopf [--prefix-caching] [--dry-run]`
(`--mode decode` also takes `--fn-worker native [--drafter stock] [--prefix-caching]`; `--mode 3bit-cache-native|3bit-long-cache-native`
take `--native-kv-tier auto|GIB|off` and `--memory GIB`).

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
- Disk: about 116 GB for the weights; the image of the torch modes is 41 GB: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2`
  @ `sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b` (pinned in modes.json; the launcher pulls it by digest).
  `--fn-worker native` runs `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc3` @ `sha256:5bca558833ad230ab1e8db7e6a64f538a7396c44dd660b69716a5aec4e714640` instead.
- Native worker: about 60 GiB of free GTT for the host tables (about 128 GB of RAM or more at the default GTT limit); without it
  the launcher stops with the reason. Native prefix caching (`3bit-cache-native`, `3bit-long-cache-native`): tested under 128 GiB
  (RAM prefix tier 16 GiB per GPU) and 64 GiB memory limits (tier off, hits from VRAM; tight: about 7 GiB left after the host tables).
- Peak VRAM per card: 30.3 GiB in `decode`, 31.5 GiB in `prefill-long` under 64K-128K prefill load.

## Known issues and limitations
- In very long agentic sessions (beyond ~30K tokens) the 3-bit can end replies early. Use `--prefix-caching`, and prefer the
  [4-bit](README-4bit.md) for long coding sessions. The new weights reduce it but do not remove it: probability of an early stop
  at real stop points beyond 28K tokens 0.452 (first weights 0.465, bf16 0.328); on empty-reply points 0.420 (0.637, bf16 0.385).
- 98,304-token window in `decode` (MTP pool, also with `--fn-worker native`), 200,000 in `prefill-long` with MTP off. No image
  input in this release.
- The native worker is opt-in and has request-API limits ([overview](README.md#request-api-limits)). Native prefix caching runs with
  speculative decoding off; speculative decoding with prefix caching is in development.
