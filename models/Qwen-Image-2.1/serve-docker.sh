#!/usr/bin/env bash
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:qwen-image21-mxfp4-rdna4-v1.1.0}"
cache="${PAITON_CACHE:-paiton-qwen-image21-cache}"
port="${PAITON_PORT:-8191}"
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
bind="${PAITON_BIND:-0.0.0.0}"
name="${PAITON_CONTAINER:-paiton-qwen-image21}"
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri \
  --ipc=host -p "$bind:$port:8191" -v "$cache:/cache" "$image" "$@"
