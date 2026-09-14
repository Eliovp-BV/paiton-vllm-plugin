# NEO mixed-GGUF native runtime notices

Paiton runtime artifacts and the public plugin are distributed under the root
Apache-2.0 license. The proprietary compiler and generated implementation source
are not distributed. The rights-holder authorization for generated runtime
binaries recorded in the existing Qwen3.8 release process applies.

The GGUF format and quantization algorithms are based on ggml/llama.cpp revision
`434ddbbc0e30522e897670681e503b797c12b7c1`. Preserve the ggml authors' MIT notice
in [LICENSES/ggml-MIT.txt](LICENSES/ggml-MIT.txt). The compiled attention path uses
AMD rocWMMA from the pinned ROCm 7.14 build; its MIT notice is included in
[LICENSES/rocWMMA-MIT.txt](LICENSES/rocWMMA-MIT.txt). Existing root LICENSES retain
notices for inherited runtime dependencies, including Composable Kernel, AITER,
flash-linear-attention and Triton. These inherited packages belong to the
external vLLM stack; the new Paiton model is compiled from native HIP/C++ without
Torch or Triton imports or generated Triton kernels.

The native vision encoder follows the Qwen3-VL/Qwen3.5 architecture and the
selected GGUF projector contract. Its HIP positional interpolation, layer
normalization, rotary attention and merger replace the external vision model;
vLLM and Transformers provide image preprocessing and API integration under
their existing Apache-2.0 notices. No Triton vision kernel is compiled or
embedded into the Paiton artifact.

The runtime reuses the pinned public vLLM/ROCm image documented in the Dockerfile.
Its notices remain under `/licenses`. The NEO image removes the unrelated Qronos
payload and Transformers' test-only `testing_utils` fixture, which embeds a
sandbox CI token. No production model weights or tokenizer values are altered
by this cleanup. Runtime dependency licenses also remain with their installed
packages; the release's dependency SBOM records their exact versions.

Model weights are downloaded separately from the author's pinned repository.
The GGUF and author-source repositories declare Apache-2.0. Exact repository
revisions, file hashes, tokenizer provenance and the selected mixed tensor
formats are recorded in `checkpoint.lock.json`. Paiton preserves the selected
GGUF packed weight values; it does not claim authorship of the fine-tune or
redistribute it as a new quantization. Neither the base Qwen checkpoint nor the
AMD Qronos fine-tune substitutes for these weights. MTP tensors remain in the
upstream file but are not loaded into the qualified target-only model.
