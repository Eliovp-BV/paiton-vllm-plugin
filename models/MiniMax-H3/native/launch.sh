#!/usr/bin/env bash
# Run MiniMax H3 Ref2VA v1.2 (native) in its container. The first run downloads the weights (see ../README.md).
#   ./launch.sh generate --prompt "..." --ref photo.png --out clip.mp4   # files inside the current directory
#   ./launch.sh serve < requests.jsonl                                   # weights load once, one JSON request per line
#   ./launch.sh prepare [--preset turbo8]                                # download and check the weights, then exit
set -euo pipefail
image="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:minimax-h3-rdna4-v1.2.0}"
cache="${PAITON_CACHE:-$HOME/.cache/paiton/minimax-h3-v12}"   # absolute host path for the downloaded weights
[ -e /dev/kfd ] || { echo "ROCm /dev/kfd is required" >&2; exit 1; }
mkdir -p "$cache"
devices=(--device /dev/kfd)
if [ -n "${PAITON_RENDER:-}" ]; then devices+=(--device "$PAITON_RENDER"); else devices+=(--device /dev/dri); fi
env=(); [ -n "${HF_TOKEN:-}" ] && env=(-e HF_TOKEN)
exec docker run --rm -i --name "${PAITON_CONTAINER:-paiton-minimax-h3-v12}" "${devices[@]}" \
  --group-add "$(stat -c '%g' /dev/kfd)" --group-add video --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${env[@]}" -v "$cache:/cache" -v "$PWD:/work" "$image" "$@"
