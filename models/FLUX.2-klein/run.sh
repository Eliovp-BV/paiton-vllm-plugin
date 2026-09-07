#!/usr/bin/env bash
set -euo pipefail
image_name="${PAITON_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-klein-rdna4-v1.0.0}"
output_dir="${PAITON_OUTPUTS:-$PWD/outputs}"
cache_mount="${PAITON_CACHE:-paiton-flux2-cache}"
render_gid="$(stat -c '%g' /dev/kfd)"
if [[ "${1:-}" == "download" ]]; then
  image_name="${PAITON_TOOLS_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-tools-rdna4-v1.0.0}"
  shift
  set -- convert "$@"
elif [[ "${1:-}" == "benchmark" ]]; then
  prior_arg=""
  for current_arg in "$@"; do
    if [[ "$current_arg" == "--backend=stock" || ( "$prior_arg" == "--backend" && "$current_arg" == "stock" ) ]]; then
      image_name="${PAITON_TOOLS_IMAGE:-ghcr.io/eliovp/paiton-vllm-plugin:flux2-tools-rdna4-v1.0.0}"
    fi
    prior_arg="$current_arg"
  done
fi
mkdir -p "$output_dir"
port_args=()
[[ "${1:-}" != serve ]] || port_args=(-p 127.0.0.1:7860:7860)
exec docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add "$render_gid" \
  --shm-size=2g \
  "${port_args[@]}" \
  -v "$cache_mount:/models" \
  -v "$output_dir:/outputs" \
  "$image_name" "$@"
