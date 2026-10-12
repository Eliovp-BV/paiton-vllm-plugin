#!/usr/bin/env bash
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minicpm5-2b-w4a16-rdna4-v1.0.0}"
cache="${PAITON_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/paiton-minicpm5-2b}"
port="${PAITON_PORT:-8036}"
# The API listens on every interface (0.0.0.0); PAITON_BIND=127.0.0.1 keeps it on this machine.
bind="${PAITON_BIND:-0.0.0.0}"
name="${PAITON_CONTAINER:-paiton-minicpm5-2b}"
mkdir -p "$cache"
extra=()
if [[ -n "${PAITON_HF_CACHE:-}" ]]; then
  extra+=(-v "$PAITON_HF_CACHE:/models/cache/huggingface:ro")
fi
# Opt-in: PAITON_API_KEY makes vLLM require that key on /v1 requests (passed by variable name, not as a literal argument).
if [[ -n "${PAITON_API_KEY:-}" ]]; then
  export VLLM_API_KEY="$PAITON_API_KEY"
  extra+=(-e VLLM_API_KEY)
fi
# GPU_MAX_HW_QUEUES=1 keeps RDNA4 decode in its fast mode (decode is about 4x slower without it).
exec docker run --rm --name "$name" --device /dev/kfd --device /dev/dri -e GPU_MAX_HW_QUEUES=1 \
  --ipc=host -p "$bind:$port:8036" -v "$cache:/models/cache" "${extra[@]}" "$image" "$@"
