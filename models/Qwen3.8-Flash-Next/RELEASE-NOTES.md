# Qwen3.8-Flash-Next release notes (DRAFT, local only)

First release of this model on Paiton; more will follow (the RAM/SSD cache tier next, then the 4-bit container; we keep tuning, we keep
optimizing). No custom engine was built: regular vLLM with the Paiton plugin and our own kernels.

## Stack (image rc8)
- Image: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1` @ `sha256:c7e76bc0d7d9f8a3d575b940d23f9b5219eada13b1960e10f3aaa996b5ad9191`.
- Plugin commit 8582563; runtime bundles tp-collective-m27h-981f701e, moe-grouped-71dad0d2, fn-mixers-9f92291e (scorer layout 3);
  W8A16 drafter view (MTP depth 3). Prompt rows run W8A8; accuracy numbers: the model card's table.
- Official numbers, measured on the shipped image through the launcher: decode 216.3 weighted / 565.1 @8 (48/48; ladder 204.6 / 315.1 / 449.2 / 565.1),
  TTFT 117 ms, update p99 13.9 ms; prefill @64K 8,161 (exact 64,000 tokens 8,212 / 8,204; depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).

## Two prefill paths (both in the image; the launcher mode selects)
- `--mode decode` / `--mode prefill-long` (default): native GDN prefill path; prefill @64K 8,161 tok/s on the shipped image (depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).
- `--mode decode-nopf` / `--mode prefill-long-nopf`: exact-arithmetic prefill, about 7 % slower (the two native-prefill flags off). Our long-document control
  found the native path is not bit-reproducible run to run while the exact path reproduces itself; prefill
  @64K on the exact path: 7,607 tok/s (development-stack measurement, not re-run on the image; exact 64,000 tokens 7,673; depth rows 7,575 / 7,785 /
  7,762 / 7,711 / 7,607 / 7,422), about 7 % below the default path. Decode rows are identical in both.
- `--prefix-caching` (opt-in, any mode): align-mode prefix caching of the recurrent state per 2,048-token block; 64K repeat 9.6 s -> 0.35 s TTFT, identical output.

## Long context
- `decode`: 98,304-token window with speculative decoding (MTP depth 3); `prefill-long`: 200,000-token window, speculative decoding off.
- Tested end to end at 200K: needles 40/40 at 200,000 tokens (identical to bf16); prefix caching opt-in with byte-identical hits (128K repeat
  20 s -> 0.41 s TTFT); prefill measured to 128K; single-stream decode in the 200K mode (prefill-long, speculation off):
  105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K); 200K with speculation does not fit the stock pool.

## How the numbers were taken
BetterBench 0.6.0 standard profile on this machine against our OpenAI-compatible server; prefill with exact bf16 activations (no compressed
wire).

## Next release
- RAM and SSD KV-cache tier (prefix-cache offload outside the VRAM, as on the 27B release) and the 4-bit container.

## Checkpoint repo requirement
- All rc8 modes load the W8 drafter file `mtp/mtp-draft-e4m3.safetensors` (102,479,504 bytes, sha256 706ced76...). It is NOT yet in
  EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4: the upload (file + SHA256SUMS line) needs the user's approval; the launcher refuses an rc8
  mode when the file is missing from the local checkpoint copy.

## Prerequisites
- Docker with GPU device access, python3 >= 3.10, `pip install huggingface_hub` (downloads go through the Python API; the `hf` command is only a
  fallback), two AMD GPUs visible, ~116 GB of disk for the weights. `launch-flashnext.py --dry-run` prints what is missing.

## Launcher behaviour
- The checkpoint repo (EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4) is the Paiton container export: a descriptor `config.json`,
  `manifest.json`, `layers/`, `common/`, `mtp/`, `SHA256SUMS`. It carries no base-model config or tokenizer.
- `launch-flashnext.py` builds the vLLM runtime view itself, once, next to the weights (`<weights>-view`): the base model's
  config + tokenizer files at the pinned revision `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` (downloaded
  with `hf download`, non-safetensors files only), `language_model_only` + the Paiton `quantization_config` block, and links to
  the checkpoint files as the container sees them (`/models/ck/...`). The view is mounted read-only at `/models/view`; the server
  loads `--model/--tokenizer /models/view`.
- This reproduces, file for file, the view the official numbers were measured with (the benchmark harness built it with the
  same tool chain); the only omitted entry is an unused alternative drafter export (the decode mode uses the int2 draft head in
  `mtp/`). Bit-neutral: same files, same engine arguments, same environment as the measured modes (`modes.json`).
- Re-running the launcher reuses an existing complete view; delete `<weights>-view` to rebuild it.

## Runtime bundles
- The three Flash Next runtime bundles ship with symbol tables stripped (as every bundle of the base image); their manifests and
  SHA256SUMS are pinned to the shipped files. Stripping removes `.symtab/.strtab` only: `.text`, `.rodata` and `.dynsym` are
  byte-identical to the benchmarked libraries.
