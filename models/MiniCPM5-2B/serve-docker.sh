#!/usr/bin/env bash
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0}"
cache="${PAITON_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/paiton-minicpm5-2b}"
port="${PAITON_PORT:-8036}"
name="${PAITON_CONTAINER:-paiton-minicpm5-2b}"
mkdir -p "$cache"
extra=()
if [[ -n "${PAITON_HF_CACHE:-}" ]]; then
  extra+=(-v "$PAITON_HF_CACHE:/models/cache/huggingface:ro")
fi
# GPU_MAX_HW_QUEUES=1 keeps RDNA4 decode in its fast mode (decode is about 4x slower without it).
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri -e GPU_MAX_HW_QUEUES=1 \
  --ipc=host -p "$port:8036" -v "$cache:/models/cache" "${extra[@]}" "$image" "$@"
