#!/usr/bin/env bash
# Build a deterministic release archive + SHA256SUMS for the H3 v1.1.0 package.
# CPU-only. Does not push or upload. Run from the package directory:
#   bash make_release_archive.sh /path/to/output
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out_dir="${1:?usage: make_release_archive.sh OUTPUT_DIR}"
tag="paiton-minimax-h3-ref2va-rdna4-v1.1.0"
archive="$out_dir/$tag.tar.gz"

[[ -f "$package_dir/release.lock.json" ]] || {
  printf 'release.lock.json is required before packaging\n' >&2
  exit 1
}
[[ -s "$package_dir/artifacts/minimax_h3_projections_gfx1201.manifest.json" ]] || {
  printf 'manifest is missing: artifacts/minimax_h3_projections_gfx1201.manifest.json\n' >&2
  exit 1
}

mkdir -p "$out_dir"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Publish only the reviewed files. Weights, caches and notes never enter the archive.
cp -a \
  README.md BENCHMARKS.md release.lock.json checkpoints.lock.json requirements.lock.txt \
  LICENSE NOTICE MODIFICATIONS.md THIRD_PARTY_NOTICES.md verify.py \
  Dockerfile Dockerfile.local .dockerignore \
  "$tmp/"
cp -a artifacts LICENSES results tests paiton_video "$tmp/"
# Image-only distribution: never ship the compiled .so in the archive.
rm -f "$tmp/artifacts/"*.so
[[ -d tests-release ]] && cp -a tests-release "$tmp/"
[[ -d paiton_h3 ]] && cp -a paiton_h3 "$tmp/"
[[ -d comfyui ]] && cp -a comfyui "$tmp/"

# Deterministic tar: sorted names, zero mtime/owner, no extended attributes.
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    --exclude='__pycache__' --exclude='*.pyc' -C "$tmp" -cf - . | gzip -n > "$archive"

# Hash every member and the archive itself.
(
  cd "$tmp"
  find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | while read -r file; do
    sha256sum "$file"
  done
) > "$out_dir/SHA256SUMS"

(
  cd "$out_dir"
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)

printf 'archive: %s\n' "$archive"
printf 'sums:    %s/SHA256SUMS\n' "$out_dir"
printf 'next: extract into an empty dir and run verify.py --require-artifact --require-digest\n'
