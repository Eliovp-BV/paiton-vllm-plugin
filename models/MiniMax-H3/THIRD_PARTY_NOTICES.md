# Third-party notices and local assembly

The GHCR candidate contains independently compiled Paiton artifacts, public
Python bindings and an existing pinned ROCm base. New H3 dependencies,
including GPL-3.0 AIMDO and codec wheels, are installed only during local assembly. It does not contain ComfyUI core,
the ComfyUI frontend, model weights or private compiler source. `launch.sh`
pulls that image and builds a second image locally from pinned upstream
ComfyUI sources. This locally assembled image is not a publication artifact.

| Component | License/source |
|---|---|
| H3 base and derivative weights | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE), retained in LICENSES |
| Qwen3-VL-32B encoder ancestry | [Original Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb) declares Apache-2.0; see the quantization provenance note below |
| ComfyUI core | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/tree/efa6c8f804bff78b46a0fd458ebd2e47bba07a30), full source in the local image |
| ComfyUI frontend | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc), matching source/build files in the local image |
| AIMDO memory manager | [comfy-aimdo 0.5.2](https://github.com/Comfy-Org/comfy-aimdo), installed locally | GPL-3.0 |
| comfy-kitchen | [Apache-2.0](https://github.com/Comfy-Org/comfy-kitchen/tree/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668) |
| Paiton Python bindings and launch recipes | Apache-2.0, LICENSE and NOTICE |
| AITemplate-derived artifact host runtime | Apache-2.0, Meta Platforms notices |
| Triton-generated code | MIT, retained notice |
| Torch, ROCm and inherited dependencies | Notices retained in the pinned public base image |

H3's community license has territory and commercial-scale restrictions;
quantization and a Turbo adapter do not remove them. Obtain applicable H3
rights before downloading or using its derivatives. An adapter's Apache-2.0
metadata does not relicense the base model. This package does not establish
general authorization to redistribute converted H3 weights.

The Comfy encoder identifies the cybermotaz NVFP4 conversion as its source.
That [publisher's pinned card](https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct/tree/dfcab1395094f56d3c78ba472158284def0fc278)
labels its terms "Qwen License" and links to a missing LICENSE file, while the
original Qwen card declares Apache-2.0. This metadata discrepancy is retained
for the redistribution review; no new license grant from this package is
implied. The selected Comfy encoder bytes remain pinned and unchanged and are
downloaded directly from its publisher; this package does not redistribute them.

The local engine loads the artifact into a process using GPL-3.0 ComfyUI.
The source-based local assembly follows the distinction between private use
and distribution in the [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#GPLRequireSourcePostedPublic).
This packaging approach does not grant permission to redistribute the combined
local image. Redistributors must independently satisfy the licenses of the
combination; supplying only ComfyUI source does not settle a linked artifact's
corresponding-source obligations. The independently distributed artifact has
no ComfyUI code or linkage dependency; it exports tensor kernel entry points.

The H3 fusion artifact also contains a modified D128 INT8 attention kernel and
required fragment helpers derived from comfy-kitchen v0.2.33
(e9ea99cf2f0af1d0c49c04690d4153a91c2b8668), Copyright (c) 2025 Comfy Org.,
Apache License 2.0. Changes include prepacked K/V fragments, query tiling,
native U8 packing and instruction scheduling. The original attention
quantization and online softmax arithmetic are retained.
