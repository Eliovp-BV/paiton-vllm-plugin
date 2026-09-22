# Third-party notices

## Native CLI execution packages

The native CLI retains released external Qwen runtime adapters and
process-local vLLM overlays. Radiance/libr4d attribution and the maintainer's
reported author permission remain as recorded in the
[Qwen notices](models/Qwen3.8-MXFP4-DFlash2/THIRD_PARTY_NOTICES.md); no blanket
license is granted for all third-party components. The native profile uses
the separately obtained Unsloth NVFP4 checkpoint, not the historical AMD
checkpoint named in that notice. MiniCPM retains its
[model-specific notices](models/MiniCPM5-2B/THIRD_PARTY_NOTICES.md).

The wheel includes these notices and retained license texts under
`paiton_vllm_plugin/notices/`. Compiler source and generated implementation
source are excluded. Existing framework dependencies and upstream-generated
code remain external; new Paiton compiler features do not acquire those
dependencies through this packaging change.

The Paiton RDNA4 runtime and generated `.so` retain code or generated output
from the following projects. Their licenses remain applicable to those
portions.

| Component | Source | License / notice |
| --- | --- | --- |
| AMD Qwen3.8 Qronos checkpoint | `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16` | Apache-2.0 |
| Qwen3.8 base model | `Qwen/Qwen3.8-27B` | Apache-2.0 |
| AITemplate runtime components | `facebookincubator/AITemplate` | Apache-2.0; Meta Platforms notices |
| vLLM runtime and paged attention | `vllm-project/vllm` | Apache-2.0; vLLM contributors |
| Flash Linear Attention-derived GDN stages | `fla-org/flash-linear-attention` | MIT; Copyright 2023-2025 Songlin Yang and Yu Zhang |
| AITER-derived causal convolution | `ROCm/aiter` | MIT; Advanced Micro Devices, Inc. |
| Triton-generated host/device code | `ROCm/triton` | MIT; Philippe Tillet, OpenAI, and contributors |
| Composable Kernel build input | `ROCm/composable_kernel` | MIT; Advanced Micro Devices, Inc. |

Applicable retained headers include copyrights held by Advanced Micro Devices,
Inc., the vLLM team and contributors, Songlin Yang, Yu Zhang, Philippe Tillet,
OpenAI, and Meta Platforms.

The full Apache-2.0 license is in [`LICENSE`](LICENSE). Exact retained MIT
license texts are provided in [`LICENSES/`](LICENSES/).

The compiled `.so` dynamically links ROCm libraries supplied by the runtime
container, including HIP, rocBLAS, hipBLASLt, hipBLAS, rocRAND, and RCCL.

## MiniMax H3 video package

MiniMax H3 derivative weights retain the custom MiniMax H3 Community License.
The H3 attention artifact includes Apache-2.0 code derived from comfy-kitchen.
ComfyUI, its frontend and AIMDO are installed from pinned upstream components
when the user assembles the local video engine. They are not part of the
published H3 artifact image. See the [complete video package notices](models/MiniMax-H3/THIRD_PARTY_NOTICES.md)
for attribution, encoder provenance and the distinction between the distributed
artifacts and the locally assembled engine.

## Qwen-Image-2.1 image package

The balanced MXFP4 checkpoint is downloaded separately from the pinned
`EliovpAI/Qwen_Image-2.1-MXFP4-Paiton-RDNA4` repository. It retains the original
Qwen Research License and attribution. The public package contains the external
Python adapter and allowlisted compiled Paiton runtime binaries; compiler and
generated implementation source remain private. The managed image installs its
pinned PyTorch/ROCm, Diffusers and Transformers dependencies from public upstream
distributions, retaining their notices. See the
[image package notices](models/Qwen-Image-2.1/THIRD_PARTY_NOTICES.md).
