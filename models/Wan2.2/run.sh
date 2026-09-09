#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$package_dir/image.sh"
prepare_wan_image
image="$PAITON_WAN_IMAGE"
data_dir="${PAITON_WAN_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/paiton/wan22}"
outputs="${PAITON_WAN_OUTPUTS:-$HOME/paiton-videos}"
mkdir -p "$data_dir/input" "$outputs"
data_dir="$(realpath "$data_dir")"
outputs="$(realpath "$outputs")"
[[ -e /dev/kfd ]] || { printf 'ROCm /dev/kfd is required.\n' >&2; exit 1; }
exec docker run --rm --init --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --user "$(id -u):$(id -g)" \
  --shm-size 2g -v "$data_dir:/data" -v "$outputs:/outputs" "$image" "$@"
