# Third-party components

Wan2.2, the selected Comfy checkpoint repack and FastWan FullAttn declare Apache-2.0. Downloads use immutable revisions in `checkpoints.lock.json`; this package ships no model weights. FastWan conversion renames safetensors keys only, retains every tensor payload and preserves the original checkpoint. It needs no calibration and excludes no transformer components.

The FastWan sampler adapts [FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/a943220c115228ade5d57b3bab9a6a87fd600a10) Apache-2.0 code. The modified implementation preserves its three DMD timesteps, training-noise shift and CPU random draw order. Paiton bindings and recipes use the accompanying Apache-2.0 LICENSE. The compiled host runtime includes AITemplate-derived support; Triton-generated code uses the retained MIT notice.

ComfyUI core and frontend and comfy-aimdo use GPL-3.0. `Dockerfile.local` assembles ComfyUI from pinned source on the user's machine. The independently prepared artifact image contains no ComfyUI code; it exposes tensor kernel entry points. Do not publish the combined local image as part of this release. Redistribution of a combined image requires satisfying the licenses of the combination, including applicable corresponding-source obligations; including ComfyUI source alone does not settle those obligations.

Comfy-kitchen uses Apache-2.0. Torch, ROCm and inherited dependencies retain their notices in the pinned base image. See `LICENSES/` for retained texts. This package does not include H3 weights, adapters or its model license.
