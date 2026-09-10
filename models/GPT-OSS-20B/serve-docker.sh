#!/usr/bin/env bash
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:gpt-oss-20b-mxfp4-rdna4-v1.0.0}"
cache="${PAITON_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/paiton-gpt-oss}"
port="${PAITON_PORT:-8020}"
name="${PAITON_CONTAINER:-paiton-gpt-oss}"
mkdir -p "$cache"
extra=()
if [[ -n "${PAITON_HF_CACHE:-}" ]]; then
  extra+=(-v "$PAITON_HF_CACHE:/models/cache/huggingface:ro")
fi
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri \
  --ipc=host -p "$port:8020" -v "$cache:/models/cache" "${extra[@]}" "$image" "$@"
