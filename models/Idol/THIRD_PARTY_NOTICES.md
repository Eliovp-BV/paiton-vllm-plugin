# Licenses and runtime boundaries

The package code (`paiton_idol/`) and the native runtime artifact follow the repository's Apache-2.0 runtime
distribution license (`LICENSE`). The proprietary Paiton compiler and generated implementation source remain private
and are not included or relicensed. Only the allowlisted native runtime binary and its integrity manifest are included.

`artifacts/libidol_denoiser_gfx1201.so` is compiled from HIP/C++. Its required libraries are the AMD HIP runtime,
hipBLASLt, rocBLAS and the C/C++ runtime. It has no PyTorch or Triton binding or kernel dependency.

The rest of the pipeline uses the unmodified idol-sdk (Apache-2.0) on PyTorch (BSD-style), Transformers and Diffusers
(Apache-2.0), tensordict (MIT), pyvers (MIT), orjson (Apache-2.0 or MIT), cloudpickle (BSD-3-Clause), plyfile (GPL-3.0, an idol-sdk dependency, installed unmodified from PyPI; its source is on PyPI) and timm (Apache-2.0,
used once to check the DINOv3 key mapping). These run in the PyTorch runtime inherited from the base image; keep the
notices supplied with the installed dependencies.

Model weights are not included. They download from their upstream repositories under their own licenses:
`speridlabs/idol` (Apache-2.0), `timm/vit_huge_plus_patch16_dinov3.lvd1689m` (DINOv3 License, accepted by the user on
Hugging Face) and `black-forest-labs/FLUX.2-klein-4B` `vae/` (Apache-2.0). Nothing is fetched from FLUX.2-dev.
