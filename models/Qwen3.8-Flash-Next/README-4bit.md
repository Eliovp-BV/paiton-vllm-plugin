# Qwen3.8 Flash Next 4-bit on two R9700: the accuracy release

[Overview: 3-bit or 4-bit?](README.md) · [Release notes](RELEASE-NOTES.md) · [Benchmarks](BENCHMARKS.md) ·
weights: [EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)

**The accuracy of the 16-bit model, on two R9700: the same GSM8K and HumanEval scores, and MMLU-Pro within one point.** The 16-bit
model is 335 GiB (360 GB) and can't run on this hardware at all. This is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
with 4-bit experts, served by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin; part of the experts is served from
pinned system RAM, so it needs a host with about 256 GB of RAM. No custom engine: regular vLLM with the Paiton plugin and our own kernels.

## Accuracy

| Task benchmark | Questions | 16-bit | **4-bit** |
|---|---:|---:|---:|
| GSM8K (grade-school math) | 1,319 | 96.9 % | **96.8 %** |
| HumanEval (Python, pass@1) | 164 | 96.3 % | **96.3 %** |
| MMLU-Pro (100 per category × 14) | 1,400 | 84.7 % | **83.7 %** |

Same prompts, sampling seeds and thinking-on chat template for both. GSM8K and HumanEval are within the 16-bit model's own run-to-run
variation (0.0 and 1.2 points; paired McNemar p = 1.00). The MMLU-Pro gap of 1.0 point is not statistically significant (p = 0.16) and
comes from replies that reach the 16,384-token limit (the 4-bit model thinks about 12 % longer); on the 1,271 questions where neither
model reached the limit the scores are 91.1 % (4-bit) and 91.0 % (16-bit).

| Perplexity (context length) | 16-bit | **4-bit, as served** | change |
|---|---:|---:|---:|
| wikitext-2 test (512 / 4096) | 4.589 / 3.929 | **4.629 / 3.961** | +0.87 % / +0.80 % |
| unseen text (512 / 4096) | 6.938 / 6.163 | **6.954 / 6.187** | +0.23 % / +0.38 % |

Long documents (44 held-out documents of 8K-32K tokens): negative log-likelihood within 0.16 % of bf16. Needle retrieval at 131,072
tokens: 40 of 40, every bf16 answer reproduced token for token. Measured on the configuration the launcher serves (INT6 trunk, int8
prompt activations, fp8 KV cache). Method, KL, top-1 agreement and per-category scores: the
[model card](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4).

## Speed
Official numbers, measured on the shipped image through this launcher (BetterBench 0.6.0 standard profile, exact bf16 wire, cold
prefix cache, 2x R9700, PCIe 4.0 x16 host with 251 GiB of RAM):

| `--mode 4bit` / `--mode 4bit-200k` | |
|---|---:|
| Weighted single-stream decode (median of 3 runs) | 189.8 tok/s |
| Eight concurrent requests, aggregate | 420.7 tok/s (48/48) |
| Prefill, 64K-token prompt (BetterBench 64K row) | 6,991 tok/s |
| Prefill, exact 64,000-token prompt, cold cache | 7,162 tok/s |
| TTFT p50, short prompts | 121 ms |
| Update p99 | 17.7 ms per stream update |

Per-category decode (tok/s): chat 171.4, code 187.1, file_edit 207.5, json 217.5, math 217.2, prose 170.4, reasoning 170.0,
summarization 206.1. Prefill by depth (tok/s): 2K 5,564 / 8K 6,771 / 16K 6,641 / 32K 7,044 / 64K 6,991 / 128K 6,857.
Decode and concurrency: `--mode 4bit` (MTP depth 3, 98,304-token window, INT6 trunk). Prefill: `--mode 4bit-200k` (200,000-token
window, MTP off, 8-bit trunk). Both use an fp8 (E4M3) KV cache. How the numbers were taken: [overview](README.md#how-the-numbers-were-taken).

## Run it
Requirements: [Hardware](#hardware) below and the common [prerequisites](README.md#prerequisites). The first start downloads and
verifies the weights (about 133 GB) into `~/paiton-models/Qwen3.8-Flash-Next-W4A8` (`--weights DIR` to change it).

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh        # 98,304-token window, speculative decoding on (the speed table)
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-200k.sh   # 200,000-token window, speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-cache.sh  # agents and coding tools: prefix caching + 8 GiB RAM prefix tier
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-long.sh   # 262,144-token window, speculative decoding on, ~461K tokens of KV cache
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-xl.sh     # 262,144-token window, speculative decoding on, ~595K tokens of KV cache
```

By hand: `python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode 4bit|4bit-200k|4bit-cache|4bit-long|4bit-xl [--dry-run]`.

**Agents and coding tools: `--mode 4bit-cache`.** An agent resends the whole conversation every turn. This mode is the 4-bit decode
mode with prefix caching (2,048-token blocks) and an 8 GiB RAM prefix tier that keeps prefix blocks evicted from VRAM in pinned system
memory. Repeated long prompts start in about 1 s instead of 6-16 s (32K 6.3 -> 1.0 s, 64K 12.3 -> 1.1 s, 90K 16.2 -> 1.1 s to the
first token); hits from the RAM tier give the same output as a cold start. The cost: about 22 % lower multi-user throughput than
`--mode 4bit` (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. It needs
about 108 GiB of free GTT; without that the launcher leaves the RAM tier off, says so, and keeps prefix caching in VRAM.
`--prefix-caching` on `--mode 4bit` / `4bit-200k` is refused (not measured there).

**Long windows with speculative decoding: `--mode 4bit-long` / `--mode 4bit-xl`.** The 4-bit decode mode with a 262,144-token
window: it moves 1 or 2 GiB of experts per GPU from VRAM to the host tier to make room for a larger KV cache. Measured 12-13 October
on the released image with the same protocol for all three rows (allocator cap 0.925):

| Mode | Window | KV cache | Experts in VRAM per GPU | Decode, 1 request | 8 requests | 64K prefill on this server | Needles at ~258K tokens | Peak VRAM per card |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `4bit` | 98,304 | 305,338 tokens | 20.75 GiB | 189.4 tok/s | 416.1 tok/s | 4,527 tok/s | – | 30.6 GiB |
| `4bit-long` | 262,144 | 461,566 tokens | 19.75 GiB | 187.8 tok/s | 410.7 tok/s | 4,161 tok/s | 5 / 5 | 30.6 GiB |
| `4bit-xl` | 262,144 | 595,050 tokens | 18.75 GiB | 184.9 tok/s | 403.3 tok/s | 3,862 tok/s | 5 / 5 | 30.7 GiB |

Prompts are read faster in `--mode 4bit-200k` (speculative decoding off: 6,991 tok/s at 64K). `--prefix-caching` is refused in
both modes (not measured there); `--mode 4bit-cache` is the prefix-caching mode.

**Expert placement:** about 20.8 GiB of experts per GPU stay in VRAM (the most-used ones); the rest are read from pinned system
memory when a token needs them. `--expert-vram-gib` and `--host-pool on|off` change the split.

The first start of a mode takes about five to six minutes; wait for `launcher: READY on port …`. **API:** `http://127.0.0.1:18982/v1`
(or `http://<this-machine-ip>:18982/v1`) · **Model:** `Qwen3.8-Flash-Next`. Network, API key, stopping and troubleshooting:
[overview](README.md#use-the-server).

## Hardware
- Two AMD Radeon AI PRO R9700 (32 GiB each), ROCm 10 host driver.
- **System RAM: 256 GB.** The 4-bit modes pin about 96 GiB of host memory (about 55 GiB of host tables plus about 41 GiB of expert
  copies). The benchmarked configuration keeps it in GPU-mapped GTT memory, which needs about 98 GiB of free GTT; the GTT limit is
  half the RAM by default, so in practice a host with 256 GB of RAM (a 200 GB host has about 93 GiB).
- On hosts with less GTT the launcher falls back to ROCm's user-pointer memory and says so: it runs, but slower (186.0 vs 191.9 tok/s
  single-stream, 388.9 vs 423.1 at 8 requests in our validation run) and can pause for several seconds when the kernel compacts host
  memory. Raising the GTT limit (`amdgpu.gttsize`, `ttm.pages_limit`) is a system setting the host's administrator may choose; the
  launcher never changes it. `--dry-run` prints the RAM, the GTT limit and what it chooses.
- **128 GB hosts: `--ple-host pageable`** (opt-in, `--mode 4bit` and `--mode 4bit-cache`). The per-layer embedding tables (about
  49 GiB of the host tables) then stay in ordinary pageable host memory instead of pinned memory, so only about 47 GiB is pinned and
  the GTT pools need about 49 GiB of free GTT, which the default limit of a 128 GB host covers. The tables still need the RAM: the
  launcher checks for about 112 GiB of available memory, as in the default mode. Same output as the default; in our check on a
  128 GB memory budget, 176.2 tok/s single-stream and 403.4 at 8 requests (default pinned: 177.2 / 400.7). It needs an image with
  this runtime; the launcher refuses the flag on images without it.
- Disk: about 133 GB for the weights. Image: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261011-r2`
  (digest pinned in modes.json; the launcher pulls it by digest).
- Peak VRAM per card in `--mode 4bit`: 30.6 GiB under a stream of long prompts, 30.0 GiB in the decode benchmark
  (allocator cap 0.925).

## Known issues and limitations
- 98,304-token window in `4bit` (MTP pool), 262,144 in `4bit-long` / `4bit-xl`, 200,000 in `4bit-200k` with MTP off. `--mode 4bit-200k` serves the same experts and fp8
  KV cache with the 8-bit trunk; its accuracy was not measured separately. Vision is not validated.
- `--mode 4bit-cache` is opt-in: the speed table is measured in `--mode 4bit`.
