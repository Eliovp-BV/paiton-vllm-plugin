# NEO release checks

This release adds a unique NEO image in the established
`ghcr.io/eliovp/paiton-vllm-plugin` namespace. It preserves existing tags and
packages. The original Qronos checklist contains historical company-repository
migration, Qronos Hugging Face mirroring and blog-specific checks; those actions
are not part of this new model release. The applicable artifact, licensing,
provenance, privacy and serving checks are retained below. No organization-wide
security policy is changed by this release.

- Exact author GGUF, source tokenizer/template, processor and projector revisions
  and hashes are pinned. The model repositories declare Apache-2.0; complete
  Apache-2.0 and retained MIT notices are included.
- The existing rights-holder authorization covers distribution of generated
  runtime binaries. Compiler source, generated implementation source and private
  headers remain private.
- Artifact source is frozen at private tag
  `neo-q4km-image-rdna4-artifact-v1.0.0`, commit
  `0b7ca75aec37b4049dc7c13027774e9efd84794d`. Packaged plugin source is frozen at
  `neo-q4km-image-rdna4-runtime-v1.0.0`, commit
  `a693d6f844553a94b715880a999e802211a2b28e`. Later documentation and benchmark
  harness commits do not change the packaged Python modules.
- Both native artifacts build with Torch and Triton imports blocked. The vision
  builder also blocks NumPy and the external GGUF Python package. The standalone
  vision C++ process verifies changed-input graph replay without Python or those
  frameworks. Existing compiler utilities and the external vLLM stack retain
  their pre-existing dependencies; this is not a claim that the whole historical
  compiler repository has no dependencies.
- The runtime archive contains 23 allowlisted payload files plus its manifest,
  SPDX document and checksums. A clean extraction passes strict archive
  verification. Its Ed25519 signature verifies with the accompanying public key;
  this establishes local integrity, not independently attested publisher identity.
- An [unsigned in-toto build record](provenance.intoto.json) binds the native
  outputs to the pinned inputs. It does not claim a SLSA assurance level or
  independent attestation. The bundle timestamp uses a fixed source-date epoch,
  not a measured wall-clock creation time.
- The final image dependency SBOM records 797 inherited packages. All four OCI
  layers were inspected. New layers contain no credential findings, compiler
  source, private headers or build caches. Retained base-layer cryptographic
  parser/self-test markers and symbol strings were reviewed. The upstream
  Transformers test-only sandbox-token fixture is absent from every layer.
- The image contains no model weights. Startup verifies payload inventory,
  architecture, native ABI, runtime versions, and full GGUF/projector hashes.
  The existing pinned vLLM/ROCm runtime is reused without host-library injection.
- Text and image qualification, arithmetic differences, measured regressions,
  concurrency/context limits, and unfinished MTP are documented in the model
  guide and comparison report. Prefix caching and video remain disabled.

The final publication and immutable-pull results are recorded in
[paiton-release.json](paiton-release.json). A candidate digest alone is not proof
of publication.

The published immutable digest was pulled successfully. A fresh container with
only the named weight-cache volume and `HF_HUB_OFFLINE=1` passed startup,
text generation, JPEG and all five PNG fixtures. Its repeated 11-task suite
retained the same 10/11 result as the reference. The supplemental native mRoPE
numeric and graph test is committed at `965c10c060c3d1a9e837d23629880c08863533d6`.
