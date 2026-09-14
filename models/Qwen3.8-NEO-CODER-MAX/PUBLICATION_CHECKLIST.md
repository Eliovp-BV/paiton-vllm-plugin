# NEO native GGUF v1.1.0 publication checks

This checklist applies to this model's runtime-only release. Source/artifact
pins and the immutable digest are in [paiton-release.json](paiton-release.json).
The [machine-readable record](publication-checks.json) binds the checks to the
published image. Historical experimental profiles remain explicitly identified.

- [x] Exact author GGUF and projector revisions, lengths, hashes and mixed tensor formats retained; tokenizer, template and stop behavior preserved.
- [x] Qualified gfx1201 ABI, 8K context, TP1, one active sequence, image limits and activation arithmetic documented. MTP, video and prefix caching disabled.
- [x] Native HIP/C++ artifacts built and executed independently of Torch/Triton; the existing external vLLM stack remains separate.
- [x] Compiler and plugin source revisions pinned. Compiler source and its source tag remain private; the public runtime-source tag identifies the image build inputs.
- [x] Runtime-binary redistribution authorization and Apache-2.0/MIT notices retained. Weights are downloaded separately from pinned upstream files.
- [x] Explicit artifact allowlist, payload hashes, artifact SPDX and 797-package runtime dependency SBOM verified.
- [x] Runtime archive hash and Ed25519 signature verified. The bundled public key provides an integrity check, not independent publisher identity assurance.
- [x] Final image filesystem and every layer match the fully inspected qualified candidate. Only version/source labels differ. Inherited parser/test/symbol scan findings were reviewed; no private compiler source, generated implementation source, private headers, credentials or build cache is published.
- [x] Native numerical, full-logit, held-out quality, state/graph and matched performance qualification passed within the documented scope. Rejected arithmetic experiments are excluded.
- [x] Final local image started from a new empty cache and passed text, PNG and JPEG generation after downloading the exact pinned files.
- [x] New uniquely versioned GHCR tag was confirmed absent before push; existing release digest was verified unchanged before and after publication.
- [x] Published immutable digest pulled and started with only a read-only cache volume, offline weight loading and no compiler/plugin checkouts.
- [x] Published runtime loaded the expected native language/helper/vision artifacts and pinned HIP library; API features and limits passed repeated functional verification.
- [x] Model launcher, documentation, release metadata and rollback image select the verified digest. Eleven focused plugin tests pass.

Repository publication uses an exact-head pull request. Current branch
protection, rulesets, check runs, commit statuses and reviews are inspected
immediately before merge. A failed mandatory gate stops the merge; publication
scripts do not change repository protection or bypass required approvals.

The runtime archive is the byte-identical qualified rc4 bundle, promoted under
the stable release asset name. Its internal bundle ID and deterministic timestamp
are retained. Image labels distinguish the stable runtime-source revision from
the original qualification revision. Release integrity/provenance files describe
this promotion without claiming a new binary build or independent attestation.
