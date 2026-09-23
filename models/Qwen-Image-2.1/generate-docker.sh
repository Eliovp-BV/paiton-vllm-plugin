#!/usr/bin/env bash
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.0.3}"
cache="${PAITON_CACHE:-paiton-qwen-image21-cache}"
output="${PAITON_OUTPUT_DIR:-$PWD/outputs}"
mkdir -p "$output"
output="$(cd -- "$output" && pwd)"
exec docker run --rm --device /dev/kfd --device /dev/dri --ipc=host \
  -v "$cache:/cache" -v "$output:/outputs" "$image" generate "$@"
