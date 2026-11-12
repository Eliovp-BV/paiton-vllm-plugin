#!/usr/bin/env bash
# Start the MiniMax-H3-Image server. The first start downloads the 48 GB FP8 weights (see README.md).
#   PAITON_DETACH=1 ./launch.sh      # run in the background; follow with: docker logs -f paiton-minimax-h3-image
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-image-rdna4-v0.1.0@sha256:aadaf6d32efab3edbb77e8b5794fbc81e00e6ac0c9d4c15b859cdf80de7eb8e5}"
cache="${PAITON_CACHE:-paiton-minimax-h3-image-cache}"   # a Docker volume name or an absolute host path
port="${PAITON_PORT:-30000}"
bind="${PAITON_BIND:-127.0.0.1}"                          # 0.0.0.0 serves other machines too
name="${PAITON_CONTAINER:-paiton-minimax-h3-image}"
devices=(--device /dev/kfd)
if [ -n "${PAITON_RENDER:-}" ]; then devices+=(--device "$PAITON_RENDER"); else devices+=(--device /dev/dri); fi
detach=(); [ "${PAITON_DETACH:-0}" = 1 ] && detach=(-d)
env=(); [ -n "${HF_TOKEN:-}" ] && env=(-e HF_TOKEN)
exec docker run --rm "${detach[@]}" --name "$name" "${devices[@]}" --group-add video --ipc=host \
  "${env[@]}" -p "$bind:$port:30000" -v "$cache:/cache" "$image" "$@"
