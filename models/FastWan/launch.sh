#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PAITON_WAN_PRESET=fast
export PAITON_WAN_DOWNLOAD_PRESET="${PAITON_WAN_DOWNLOAD_PRESET:-fast}"
exec "$package_dir/../Wan2.2/launch.sh" "$@"
