# Third-party notices

## Native CLI execution packages

The local native CLI retains released external Qwen runtime adapters and
process-local vLLM overlays. Model attribution remains as recorded in the
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
| vLLM runtime and paged attention | `vllm-project/vllm` | Apache-2.0; vLLM contributors |

Applicable retained headers include copyrights held by Advanced Micro Devices,
Inc. and the vLLM team and contributors.

The full Apache-2.0 license is in [`LICENSE`](LICENSE).

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
