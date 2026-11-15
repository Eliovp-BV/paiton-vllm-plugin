#!/usr/bin/env bash
# Run MiniMax H3 Ref2VA v1.2 (native) in its container. The first run downloads the weights (see ../README.md).
#   ./launch.sh http [--preset turbo8]                                   # HTTP API on 127.0.0.1:30000 (see ../README.md)
#   ./launch.sh generate --prompt "..." --ref photo.png --out clip.mp4   # files inside the current directory
#   ./launch.sh serve < requests.jsonl                                   # weights load once, one JSON request per line
#   ./launch.sh prepare [--preset turbo8]                                # download and check the weights, then exit
# PAITON_PORT / PAITON_BIND (0.0.0.0 serves other machines too) / PAITON_DETACH=1 apply to http.
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.2.0@sha256:c656184d9d21240544e5b772daa5f2fc0de89fe95463498e34ada645465e287b}"
cache="${PAITON_CACHE:-$HOME/.cache/paiton/minimax-h3-v12}"   # absolute host path for the downloaded weights
[ -e /dev/kfd ] || { echo "ROCm /dev/kfd is required" >&2; exit 1; }
mkdir -p "$cache"
devices=(--device /dev/kfd)
if [ -n "${PAITON_RENDER:-}" ]; then devices+=(--device "$PAITON_RENDER"); else devices+=(--device /dev/dri); fi
env=(); [ -n "${HF_TOKEN:-}" ] && env=(-e HF_TOKEN)
mode=(-i)
if [ "${1:-}" = http ]; then
  mode=(-p "${PAITON_BIND:-127.0.0.1}:${PAITON_PORT:-30000}:30000")
  [ "${PAITON_DETACH:-0}" = 1 ] && mode+=(-d)
fi
exec docker run --rm "${mode[@]}" --name "${PAITON_CONTAINER:-paiton-minimax-h3-v12}" "${devices[@]}" \
  --group-add "$(stat -c '%g' /dev/kfd)" --group-add video --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${env[@]}" -v "$cache:/cache" -v "$PWD:/work" "$image" "$@"
