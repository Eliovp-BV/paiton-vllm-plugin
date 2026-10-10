# Qwen3.8 Flash Next on two R9700 (draft, local staging; not published)

Paiton's 3-bit Flash Next package: the mixed 3-bit expert / W8A16 trunk checkpoint
`EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4` served by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin
(native grouped MoE, native collectives, native hidden-chain and sampled MTP drafts). Model details, accuracy and format:
the model card (handoffs/release-cards-20261009/README-W3A8.md draft).

## About this release
This is the first release of this model on Paiton; more will follow: the RAM/SSD cache tier next, then the 4-bit container. We keep
tuning, we keep optimizing. No custom engine was built: this is regular vLLM with the Paiton plugin and our own kernels.

## What you get (official numbers, measured on the shipped image through this launcher; BetterBench 0.6.0 standard profile, exact bf16 wire, cold prefix cache, 2x R9700)

| | this package |
|---|---:|
| Weighted single-stream decode | 216.3 tok/s |
| Eight concurrent requests, aggregate | 565.1 tok/s (48/48; ladder 1/2/4/8 = 204.6 / 315.1 / 449.2 / 565.1) |
| Prefill, 64K-token prompt | 8,161 tok/s (exactly 64,000 tokens: 8,212) |
| TTFT p50, short prompts | 117 ms |
| Update p99 | 13.9 ms per stream update (MTP: ~2.9 tokens per update) |

Per-category decode (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1,
summarization 239.8. Prefill by depth (tok/s): 2K 7,925 / 8K 8,321 / 16K 8,311 / 32K 8,232 / 64K 8,161 / 128K 7,928.
Decode and concurrency: `--mode decode` (MTP depth 3, 98,304-token window). Prefill: `--mode prefill-long` (200,000-token window,
MTP off). Both modes also exist with exact-arithmetic prefill, about 7 % slower (`--mode decode-nopf`, `--mode prefill-long-nopf`;
prefill @64K 7,607 measured on the development stack, not re-run on the image). Served long-context needles: 40/40 at 131K and at 200K.

**Prefix caching (opt-in, `--prefix-caching`)**: align-mode caching of the recurrent state per 2,048-token block. A repeated 64K-token
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
The RAM and SSD KV-cache tier (prefix-cache offload outside the VRAM, as on the 27B release), then the 4-bit container. We keep tuning,
we keep optimizing.

## Set up
- Two AMD Radeon AI PRO R9700 (32 GiB each), ROCm 10 host driver, Docker with `/dev/kfd` and `/dev/dri` access (groups `video`, `render`).
- `pip install -U huggingface_hub` for the `hf` download command; ~110 GB of disk for the weights.
- `docker pull ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1` (digest `sha256:c7e76bc0d7d9f8a3d575b940d23f9b5219eada13b1960e10f3aaa996b5ad9191`, pinned in modes.json; the launcher pulls it by digest).

## Run
```bash
./launch-flashnext.py                      # decode mode on port 18982, downloads + verifies the weights on first use
./launch-flashnext.py --mode prefill-long  # 200K-token prompts
./launch-flashnext.py --mode decode-nopf   # exact GDN prefill path (also prefill-long-nopf)
./launch-flashnext.py --prefix-caching     # opt-in prefix caching (any mode)
./launch-flashnext.py --dry-run            # print the docker command only
```
The OpenAI-compatible API serves model name `Qwen3.8-Flash-Next` at `http://127.0.0.1:18982/v1`. First start of a mode copies the
baked compile cache (no cold compile); starts take ~3-5 minutes (weight load dominates).

## Options
`--mode decode|prefill-long|decode-nopf|prefill-long-nopf`, `--prefix-caching`, `--port`, `--name`, `--weights DIR`, `--cache DIR`, `--image REF`, `--devices GPU-...,GPU-...`,
`--served-model-name`, `--detach`, `--dry-run`. GPU selection detects AMD devices by vendor id; two are required.

## Files
`launch-flashnext.py` (also builds the runtime view `<weights>-view` once: base config + tokenizer at the pinned revision, see RELEASE-NOTES.md), `modes.json` (environment + engine arguments of the four modes and the prefix-caching opt-in), `README.md`, `RELEASE-NOTES.md`, `BENCHMARKS.md` (depth table,
per-category decode, host conditions), `THIRD_PARTY_NOTICES.md`, `runtime.lock.json` and `checkpoint.lock.json` (what this release was built from and tested with).

## Limitations (draft)
- 98,304-token window in decode mode (MTP pool); 200,000 in prefill-long mode with MTP off; vision not validated; prefix caching is an
  opt-in flag (official numbers measured without it). The prompt rows run W8A8; accuracy numbers are in the model card's table. The `-nopf` modes use
  exact-arithmetic prefill, about 7 % slower.
