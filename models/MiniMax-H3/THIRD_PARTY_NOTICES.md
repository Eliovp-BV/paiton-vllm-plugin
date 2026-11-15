# Third-party notices and local assembly (v1.1.0 Ref2VA)

The published GHCR image contains independently compiled Paiton artifacts, public Python bindings
and a pinned ROCm base. It does **not** contain ComfyUI core, the ComfyUI frontend, AIMDO, model
weights or private compiler source. `launch.sh` builds a second **local** image from pinned upstream
ComfyUI sources for GUI parity; that combined image is not a publication artifact.

| Component | License/source |
|---|---|
| H3 base and derivative weights | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE), retained in LICENSES |
| Qwen3-VL-32B encoder ancestry | [Original Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb) declares Apache-2.0; see the provenance note below |
| ComfyUI core | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/tree/efa6c8f804bff78b46a0fd458ebd2e47bba07a30), full source in the local image |
| ComfyUI frontend | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc), matching source/build files in the local image |
| comfy-aimdo 0.5.2 | [GPL-3.0](https://github.com/Comfy-Org/comfy-aimdo), installed locally only |
| comfy-kitchen | [Apache-2.0](https://github.com/Comfy-Org/comfy-kitchen/tree/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668) |
| Paiton Python bindings, runtime and launch recipes | Apache-2.0, LICENSE and NOTICE |
| AITemplate-derived artifact host runtime | Apache-2.0, Meta Platforms notices |
| Triton-generated code | MIT, retained notice |
| Torch, ROCm and inherited dependencies | Notices retained in the pinned public base image |

## H3 community licence

H3's community licence has **territory and commercial-scale restrictions**; quantization, Turbo
adapters and Ref2VA reference conditioning do not remove them. The grant excludes the European
Union, the United Kingdom, the Republic of Korea and the United States (Section I.5). Distribution
must include the Agreement and the NOTICE text; modified files must be marked; commercial products
above US$20M/year need prior written authorisation. Obtain applicable H3 rights before downloading
or using its derivatives. This package does not establish general authorisation to redistribute
converted H3 weights, and it does not relicense them.

## Encoder provenance

The Comfy encoder identifies the cybermotaz NVFP4 conversion as its source. That publisher's pinned
card labels its terms "Qwen License" and links to a missing LICENSE file, while the original Qwen
card declares Apache-2.0. The discrepancy is retained for redistribution review; no new licence
grant is implied. The selected bytes are pinned and downloaded from the publisher; this package does
not redistribute them.

## ComfyUI GPL interaction

The local engine loads the Paiton artifact into a process that imports GPL-3.0 ComfyUI code. Keeping
the compiler source private does not by itself establish that distribution of this combination meets
GPL corresponding-source obligations. The published image therefore excludes ComfyUI and AIMDO; the
combined local image is private use and must not be redistributed. Redistributors of a combined
image must independently satisfy the GPL, obtain an exception, or use an engine whose licence
permits the intended binary distribution. Supplying only ComfyUI source does not settle the linked
artifact's corresponding-source obligations. The independently distributed artifact exports tensor
kernel entry points and contains no ComfyUI code.

## comfy-kitchen derivation

The H3 fusion artifact contains a modified D128 INT8 attention kernel and required fragment helpers
derived from comfy-kitchen v0.2.33 (e9ea99cf2f0af1d0c49c04690d4153a91c2b8668), Copyright (c) 2025
Comfy Org., Apache License 2.0. Changes are listed in `MODIFICATIONS.md`; the original attention
quantization and online softmax arithmetic are retained.

## v1.2 native image (`native/Dockerfile`)

The v1.2 image holds the compiled Paiton libraries and runner, the Python front end in `native/paiton_h3v/` and the
packages below. It holds no model weights, no ComfyUI, no PyTorch, no Triton and no compiler source. It contains no
comfy-kitchen or Triton-derived code.

| Component | License and source |
|---|---|
| MiniMax H3 Ref2VA weights converted for Paiton (FP8 transformer with the Turbo LoRA merged in, text encoder, VAEs, tokenizer) | MiniMax H3 Community License, copy in `LICENSES/MiniMax-H3-Community.txt`; downloaded on first use from [EliovpAI/MiniMax-H3-Ref2VA-FP8-Paiton-RDNA4](https://huggingface.co/EliovpAI/MiniMax-H3-Ref2VA-FP8-Paiton-RDNA4), which lists the modified files |
| Paiton libraries, runner, front end and scripts | Apache-2.0, `LICENSE` and `NOTICE` |
| ROCm 10.0 runtime (`rocm-sdk-core` wheel) | AMD license files inside the wheel (`/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core`) |
| numpy 2.4.2, Pillow 12.1.0, tokenizers 0.23.2 | BSD-3-Clause (numpy, with 0BSD, MIT, Zlib and CC0-1.0 parts), MIT-CMU (Pillow), Apache-2.0 (tokenizers); license files in each package's `dist-info` |
| FFmpeg 6.1.1 (Ubuntu package `ffmpeg`, with libx264) | GPL-2.0-or-later as built by Ubuntu; the runner calls the `ffmpeg` and `ffprobe` programs and does not link to them. Source: `apt-get source ffmpeg` on Ubuntu 24.04; license files under `/usr/share/doc` |
| Ubuntu 24.04 base image | Package licenses under `/usr/share/doc` |
