# Third-party notices

| Component | Source | License |
|---|---|---|
| Public Paiton runtime bindings | [Paiton vLLM plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin) | Apache 2.0 |
| AITemplate-derived artifact runtime | [AITemplate](https://github.com/facebookincubator/AITemplate) | Apache 2.0, Meta Platforms notices |
| Triton-generated host and device code | [ROCm Triton](https://github.com/ROCm/triton) | MIT, Philippe Tillet, OpenAI and contributors |
| Diffusers | [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Apache 2.0 |
| SDNQ, separate conversion and stock tools only | [Disty0 SDNQ](https://github.com/Disty0/sdnq) | GPL-3.0-only |
| ComfyUI, separate interface container | [ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/12d5279438bfefc058a269eae805ceab6047777f) | GPL-3.0-only |
| ComfyUI frontend | [ComfyUI frontend](https://github.com/Comfy-Org/ComfyUI_frontend/tree/fe51f984206178c7dcc37bf96f665f4ff5c8d3ca) | GPL-3.0-only |
| Paiton ComfyUI node and local image API | Included in `comfyui/` and `service/` | Apache 2.0 |
| FLUX.2 klein 4B model, downloaded separately | [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0 |

The Dockerfile inherits a pinned public Paiton runtime image. Its installed
packages and retained upstream notices also cover PyTorch, ROCm libraries,
Transformers and the other components of that image. The private compiler
source is not distributed. Model license/provenance caveats are documented in
the README. The Apache license is in `LICENSE`; the retained Triton MIT notice
is in `LICENSES/Triton-MIT.txt`.

The ComfyUI container retains the complete pinned ComfyUI Python source and
upstream license. Its frontend is pinned to 1.49.6, with matching source, build
files and license in `/opt/comfyui-frontend-source`. ComfyUI exchanges PNG images with the
independent local inference services over HTTP. No Paiton artifact is loaded
into the ComfyUI process.
