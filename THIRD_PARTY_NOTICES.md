# Third-party notices

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
