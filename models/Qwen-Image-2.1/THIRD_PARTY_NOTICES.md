# Licenses and runtime boundaries

The model weights and upstream model configuration are derivatives of
[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), revision
`790c92633540aa0cb11d9abf19eb46d861714758`. Their Qwen Research License is copied
in `LICENSE`, with the required attribution in `NOTICE`. The grant covers
noncommercial research and evaluation. Commercial use requires a separate
upstream license. **Built with Qwen.**

The Paiton plugin Python adapter and compiled runtime artifact follow the
repository's Apache-2.0 runtime distribution license, copied in
`LICENSES/Apache-2.0.txt`. The proprietary Paiton compiler and generated
implementation source remain private and are not included or relicensed.
Only the allowlisted native runtime binaries and their ABI/integrity manifests
are included in this package.

The native weight reconstruction and BF16 region binaries are compiled from HIP/C++. Their required
libraries are the AMD HIP runtime, ROCm libraries and the C/C++ runtime. They have
no PyTorch or Triton binding or kernel dependency. The complete image pipeline
uses the external Diffusers, Transformers and PyTorch runtimes; the entire
pipeline is not a framework-independent compiled model.

AMD Quark performs offline weight conversion. Quark is not required to load
these packed weights on the R9700. Quantization uses the public OCP MXFP4
E2M1/E8M0 format; it does not use NVFP4 scales or an INT4 integer interpretation.
Activation arithmetic remains BF16. Matrix products use the external BF16
runtime after reconstructing the current weight matrix.

Diffusers, Transformers and Accelerate carry their upstream Apache-2.0 notices;
AMD Quark includes MIT-licensed kernel source. PyTorch carries its upstream BSD-style notices.
Keep notices supplied with installed dependencies. LPIPS and CLIP metric
dependencies are used for evaluation, not bundled as inference components.
