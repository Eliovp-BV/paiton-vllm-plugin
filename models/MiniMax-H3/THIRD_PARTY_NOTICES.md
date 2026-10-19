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
