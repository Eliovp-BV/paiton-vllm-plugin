#!/usr/bin/env bash
set -euo pipefail
image=${PAITON_NEO_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin@sha256:2a8fed46bc19fe8ca7164139c66f78e80f841120d42112b2968ba060c319fdbe}
port=${PAITON_NEO_PORT:-8000}
cache=${PAITON_NEO_CACHE:-paiton-qwen38-neo-cache}
exec docker run --rm --name paiton-qwen38-neo \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  -p "127.0.0.1:${port}:8000" \
  --mount "type=volume,src=${cache},dst=/models/cache" \
  "$image" "$@"
