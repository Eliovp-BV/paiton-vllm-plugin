#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${PAITON_H3_IMAGE:-paiton-minimax-h3:fused-v4-review}"
data_dir="${PAITON_H3_DATA:-$PWD/paiton-h3-data}"
outputs="${PAITON_H3_OUTPUTS:-$PWD/paiton-videos}"
mkdir -p "$data_dir" "$outputs"
data_dir="$(realpath "$data_dir")"
outputs="$(realpath "$outputs")"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -t "$image" "$package_dir"
fi
[[ -e /dev/kfd ]] || { printf 'ROCm /dev/kfd is required.\n' >&2; exit 1; }
exec docker run --rm --init --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" \
  --shm-size 2g -v "$data_dir:/data" -v "$outputs:/outputs" "$image" "$@"
