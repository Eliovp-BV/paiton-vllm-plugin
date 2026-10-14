# Qwen3.8-Flash-Next release notes

No custom engine was built: regular vLLM with the Paiton plugin and our own kernels. We keep tuning, we keep optimizing.
Pages: [overview](README.md) · [3-bit](README-3bit.md) · [4-bit](README-4bit.md).

## Release 20261014-r2 (hotfix for both images)

### Fixes
- **Stability and correctness fixes** for prefix caching with several requests at once and for long prompts, in every mode
  (r2 images). Same speed as r1.
- **Thinking and tool calls.** Every mode now passes vLLM's reasoning and tool-call parsers: the reasoning comes back in
  `message.reasoning`, the answer in `content`, tool calls in `tool_calls` (works with the r1 images too).
- **3-bit memory caps tuned.** With `--prefix-caching`, `--mode decode` and `--mode decode-nopf` cap the memory pool at 94 % (was 95 %): in our
  test with 8 concurrent requests plus long prompts the cards stayed at or below 31.4 GiB (was up to 31.7 GiB), at 170.2 / 493.0 tok/s
  with 1 / 8 requests. The prefill-long modes, which need the room for 131K-token prompts, and every run without prefix caching keep
  95 %. No need to edit `PYTORCH_ALLOC_CONF` by hand any more (works with the r1 images too).
- **12 October: prefix caching.** `--prefix-caching` (3-bit modes) now sets 2,048-token cache blocks. A repeated conversation then
  gives exactly the same output as a fresh one, and peak VRAM was about 1 GiB lower per card in our 65K-token test.
- **13 October: 4-bit VRAM.** `--mode 4bit` and `--mode 4bit-cache` cap the memory pool at 92.5 %, so a stream of long prompts stays
  at about 30.6 GiB per card (was up to 31.3 GiB), at the same speed.

### New
- **128 GB hosts for the 4-bit modes:** `--ple-host pageable` (opt-in, `--mode 4bit` and `--mode 4bit-cache`, r2 4-bit image) keeps
  the per-layer embedding tables in pageable memory: about 47 GiB pinned instead of 96 GiB, so the GTT pools fit the default limit of
  a 128 GB host. Same output; speed within about 1 % of the default.

### Images
- 3-bit: `qwen38-flashnext-rocm10-vllm029-20261010-r2`; 4-bit: `qwen38-flashnext-rocm10-vllm029-20261011-r2` (the launcher pins both
  by digest). The r1 images stay available; the fixes above need r2.

## Release 20261011

### 4-bit (new): the accuracy release
- **Weights** `EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4`, modes `--mode 4bit` (98,304-token window, MTP depth 3, INT6 trunk) and
  `--mode 4bit-200k` (200,000-token window, MTP off, 8-bit trunk). Per layer the 410 most-used experts are INT4 (g128) and the 102
  least-used use the 3-bit codebook format; about 20.8 GiB of experts per GPU stay in VRAM, the rest are served from pinned system
  memory (expert host tier); fp8 KV cache (about 305,000 tokens in `4bit`).
- **Accuracy** (as served) against the 16-bit model:
  - GSM8K 96.8 vs 96.9 %, HumanEval 96.3 vs 96.3 %, MMLU-Pro 83.7 vs 84.7 %. The MMLU-Pro gap is not significant (p = 0.16) and comes
    from replies at the 16,384-token limit; uncapped: 91.1 vs 91.0 %.
  - Perplexity +0.87 % on wikitext, +0.38 % on unseen text (4096).
  - Long-document NLL within 0.16 % of bf16; needles 40/40 identical to bf16.
- **Speed** on the shipped image: decode 189.8 tok/s (median of 3 runs) / @8 420.7 / prefill @64K 6,991 (an exact 64,000-token
  prompt 7,162) / TTFT 121 ms / update p99 17.7 ms.
- **Cache mode for agents** (opt-in, `--mode 4bit-cache`): prefix caching plus an 8 GiB RAM prefix tier; repeated long prompts start
  in about 1 s instead of 6-16 s (64K 12.3 -> 1.1 s), identical output from the tier. It needs about 108 GiB of free GTT, else the tier
  stays off (with a warning) and prefix caching runs in VRAM only. The cost: about 22 % lower multi-user throughput than `--mode 4bit`
  (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. `--prefix-caching`
  on `--mode 4bit` / `4bit-200k` is refused (not measured there).
- **Host memory**: about 96 GiB pinned (55 GiB host tables + 41 GiB experts) in GTT-backed host pools when the free GTT holds them
  (about 98 GiB: a 256 GB host), with an automatic fallback to ROCm's user-pointer memory and a warning.
- **Image**: `qwen38-flashnext-rocm10-vllm029-20261011-r1` (the launcher picks it for the 4-bit modes).

### 3-bit (launcher update)
- Same image (`qwen38-flashnext-rocm10-vllm029-20261010-r1`), weights, modes, speed and accuracy as the 10 Oct release.
- Launcher: a short settle wait before it reports READY, the peer-to-peer and GTT preflight checks, and the opt-in `--host-pools gtt`
  (removes the occasional memory pauses under load, costs ~1 GiB VRAM in decode); the published user-pointer pools stay the default.
- **Agents and coding tools: add `--prefix-caching`.** An agent resends the whole conversation every turn; with the prefix cache the
  shared part is reused instead of prefilled again: a repeated 64K-token prefix reaches its first token in 0.35 s instead of 9.6 s,
  with identical output. The speed tables are measured without it (the benchmark never repeats a prefix).
- **Known issue:** in very long agentic sessions (beyond ~30K tokens) the 3-bit can end replies early. Use `--prefix-caching`, and
  prefer the 4-bit for long coding sessions; a fix is planned for the next 3-bit release.

### Both
- The server listens on all interfaces (0.0.0.0) by default; `--host 127.0.0.1` keeps it local. API key: `PAITON_API_KEY=<key>` in
  the environment (or `--api-key`, visible in `ps`); it is handed to Docker by variable name, never in the container's command line.
- The launcher warns when the KV cache is small relative to the window with prefix caching on, and its READY line explains prompt vs
  generation throughput in the server log (README Troubleshooting).
- `--dry-run` reports host RAM, the GTT limit and what it chooses, and warns when the GPUs have no peer-to-peer link.

## Next release
- SSD KV-cache tier for the 4-bit cache mode.

## Release 20261010-r1 (previous: 3-bit)

### Stack (image rc8)
- Image: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1` @ `sha256:c7e76bc0d7d9f8a3d575b940d23f9b5219eada13b1960e10f3aaa996b5ad9191`.
- Plugin commit 8582563; runtime bundles tp-collective-m27h-981f701e, moe-grouped-71dad0d2, fn-mixers-9f92291e (scorer layout 3);
  W8A16 drafter view (MTP depth 3). Prompt rows run W8A8; accuracy numbers: the model card's table.
- Official numbers, measured on the shipped image through the launcher: decode 216.3 weighted / 565.1 @8 (48/48; ladder 204.6 / 315.1 / 449.2 / 565.1),
  TTFT 117 ms, update p99 13.9 ms; prefill @64K 8,161 (exact 64,000 tokens 8,212 / 8,204; depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).

### Two prefill paths (both in the image; the launcher mode selects)
- `--mode decode` / `--mode prefill-long` (default): native GDN prefill path; prefill @64K 8,161 tok/s on the shipped image (depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).
- `--mode decode-nopf` / `--mode prefill-long-nopf`: exact-arithmetic prefill, about 7 % slower (the two native-prefill flags off). Our long-document control
  found the native path is not bit-reproducible run to run while the exact path reproduces itself; prefill
  @64K on the exact path: 7,607 tok/s (development-stack measurement, not re-run on the image; exact 64,000 tokens 7,673; depth rows 7,575 / 7,785 /
  7,762 / 7,711 / 7,607 / 7,422), about 7 % below the default path. Decode rows are identical in both.
- `--prefix-caching` (opt-in, the 3-bit modes; the 4-bit modes use `--mode 4bit-cache` and refuse the flag): align-mode prefix caching of the recurrent state per 2,048-token block; 64K repeat 9.6 s -> 0.35 s TTFT, identical output.

### Long context
- `decode`: 98,304-token window with speculative decoding (MTP depth 3); `prefill-long`: 200,000-token window, speculative decoding off.
- Tested end to end at 200K: needles 40/40 at 200,000 tokens (identical to bf16); prefix caching opt-in with byte-identical hits (128K repeat
  20 s -> 0.41 s TTFT); prefill measured to 128K; single-stream decode in the 200K mode (prefill-long, speculation off):
  105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K); 200K with speculation does not fit the stock pool.

### Host memory (3-bit)
- The host-resident tables (about 55 GiB over both GPUs) are allocated as driver-owned GTT memory (`HSA_USERPTR_FOR_PAGED_MEM=0`, set by the
  launcher) when the GPUs' GTT size holds them; the default GTT size is half the system RAM, so this needs about 120 GiB of RAM or more.
  ROCm's default user-pointer memory can be moved when the kernel compacts host memory, which paused the GPU queues briefly under load
  (2.7 s in total over a 3-minute run at 8 concurrent requests; 0.07 s with GTT memory, speed unchanged). On smaller hosts the launcher keeps
  the default and says so; `--host-pools userptr` keeps it everywhere.

## Reference (both releases)

### Host memory (4-bit)
- `--mode 4bit` / `--mode 4bit-200k` keep part of the experts in pinned system memory. Together with the host tables this pins about
  96 GiB (about 55 GiB of tables plus about 41 GiB of experts, both GPUs together; measured 95.5 GiB in the decode mode, 93.0 GiB in the
  200K mode). The launcher checks the available RAM first (pinned memory plus 16 GiB for the system).
- The benchmarked configuration places all of it in GTT memory (`HSA_USERPTR_FOR_PAGED_MEM=0`), which is charged to one GPU's GTT. The
  usable GTT is limited per GPU and by the kernel's TTM page limit, both half the RAM by default, so **the benchmarked 4-bit
  configuration needs about 98 GiB of free GTT: in practice a host with 256 GB of RAM** (a 200 GB host has about 93 GiB).
- On hosts with less GTT the launcher keeps ROCm's default user-pointer memory and prints a warning: the model starts and runs, but
  slower (186.0 vs 191.9 tok/s single-stream, 388.9 vs 423.1 tok/s at 8 requests in our validation run) and the GPUs can pause for
  several seconds when the kernel compacts host memory. Raising the GTT limit (`amdgpu.gttsize`, `ttm.pages_limit`) is a system setting
  the host's administrator may choose; the launcher never changes it.
- `--dry-run` prints the host RAM, the GTT limit and what it would choose, and warns when the two GPUs have no peer-to-peer link.

### Checkpoint repo (3-bit)
- The 3-bit modes load the W8 drafter file `mtp/mtp-draft-e4m3.safetensors` shipped in EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 (verified by
  SHA256SUMS together with the other files on the first start).
- The 3-bit modes use the bf16 KV cache (fp8 KV measured there, not adopted); the 4-bit modes use an fp8 (E4M3) KV cache.

### Launcher behaviour
- The checkpoint repos (EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 and -W4A8-Paiton-RDNA4) are Paiton container exports: a
  descriptor `config.json`, `manifest.json`, `layers/`, `common/`, `mtp/`, `SHA256SUMS` (the 4-bit repo adds `trunk-int6/` and
  `expert-format-map.json`). They carry no base-model config or tokenizer.
- `launch-flashnext.py` builds the vLLM runtime view itself, once, next to the weights (`<weights>-view`): the base model's
  config + tokenizer files at the pinned revision `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` (downloaded
  with `hf download`, non-safetensors files only), `language_model_only` + the Paiton `quantization_config` block, and links to
  the checkpoint files as the container sees them (`/models/ck/...`). The view is mounted read-only at `/models/view`; the server
  loads `--model/--tokenizer /models/view`.
- This reproduces, file for file, the view the official numbers were measured with (the benchmark harness built it with the
  same tool chain); the only omitted entry is an unused alternative drafter export (the decode mode uses the int2 draft head in
  `mtp/`). Bit-neutral: same files, same engine arguments, same environment as the measured modes (`modes.json`).
- Re-running the launcher reuses an existing complete view; delete `<weights>-view` to rebuild it.

### Runtime bundles
- The three Flash Next runtime bundles ship with symbol tables stripped (as every bundle of the base image); their manifests and
  SHA256SUMS are pinned to the shipped files. Stripping removes `.symtab/.strtab` only: `.text`, `.rodata` and `.dynsym` are
  byte-identical to the benchmarked libraries.
### Prerequisites
- Docker with GPU device access, python3 >= 3.10, `pip install huggingface_hub` (downloads go through the Python API; the `hf` command is only a
  fallback), two AMD GPUs visible, disk for the weights (about 116 GB for the 3-bit, 133 GB for the 4-bit). `launch-flashnext.py --dry-run` prints what is missing.

### How the numbers were taken
BetterBench 0.6.0 standard profile on this machine against our OpenAI-compatible server; prefill with exact bf16 activations (no compressed
wire).
