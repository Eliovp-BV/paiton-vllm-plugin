# License and provenance

- Qwen3-Coder-30B-A3B-Instruct: Qwen, Apache-2.0. Official model/config examined
  at revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`.
- Quantized checkpoint: cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit,
  revision `4bd30395b72ea6045edd04806c4fea448d4467b3`, Apache-2.0 as declared
  by its publisher. The quantizer does not identify the exact upstream weight
  revision. No weight conversion is performed by this package.
- Paiton runtime integration and benchmark scripts: repository Apache-2.0
  license. Only compiled Paiton artifacts are included in the separate local
  overlay; private compiler source is excluded.
- The compiled runtime includes AITemplate-derived components (Apache-2.0),
  and links against the ROCm libraries in the base image. Existing repository
  third-party notices and license texts remain applicable.
- AMD SMI runtime package and `libamd_smi.so`: Advanced Micro Devices, MIT.
  Its complete license is retained in the overlay's `amdsmi/LICENSE`.
- Base image is pinned to
  `ghcr.io/eliovp/paiton-vllm-plugin@sha256:c56baf54aca1ad229829c1de26e8792806608e65ee9d06ee210b79cd49f70bc9`.
  It supplies vLLM, PyTorch, Triton, Transformers and ROCm; their individual
  licenses and notices continue to apply.

No model weights, access tokens or private compiler source are copied into the
container build context. The model is downloaded separately into persistent
user storage from its pinned public Hugging Face revision.
