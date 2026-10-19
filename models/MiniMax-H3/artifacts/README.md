# Qualified gfx1201 artifact

`minimax_h3_projections_gfx1201.so` ships **only inside the published container image**. It is not
checked into git, the release archive or the Hugging Face package. This directory keeps the
compiler-emitted manifest (ABI, contract, target, size and SHA-256) and this note.

`verify.py` checks the manifest and `release.lock.json`, and with `--image <ref>` checks the image's
`dev.paiton.artifact.sha256` label against the locked hash. The image digest is pinned in
`release.lock.json`; the manifest remains the runtime contract.
