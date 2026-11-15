#!/usr/bin/env bash
# Start the LTX-2.5 server. The first start downloads the 80 GB weights (see README.md).
#   PAITON_DETACH=1 ./launch.sh      # run in the background; follow with: docker logs -f paiton-ltx-2.5
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:ltx-2.5-rdna4-v0.1.0}"
cache="${PAITON_CACHE:-paiton-ltx-2.5-cache}"            # a Docker volume name or an absolute host path
port="${PAITON_PORT:-30000}"
bind="${PAITON_BIND:-127.0.0.1}"                          # 0.0.0.0 serves other machines too
name="${PAITON_CONTAINER:-paiton-ltx-2.5}"
devices=(--device /dev/kfd)
if [ -n "${PAITON_RENDER:-}" ]; then devices+=(--device "$PAITON_RENDER"); else devices+=(--device /dev/dri); fi
detach=(); [ "${PAITON_DETACH:-0}" = 1 ] && detach=(-d)
env=(); [ -n "${HF_TOKEN:-}" ] && env=(-e HF_TOKEN)
exec docker run --rm "${detach[@]}" --name "$name" "${devices[@]}" --group-add video --ipc=host \
  "${env[@]}" -p "$bind:$port:30000" -v "$cache:/cache" "$image" "$@"
