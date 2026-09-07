#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker build -f "$package_dir/Dockerfile.tools" -t "${PAITON_TOOLS_IMAGE:-ghcr.io/eliovp/paiton-flux2-tools:rdna4-v1.0.0}" "$package_dir"
docker build -f "$package_dir/Dockerfile" -t "${PAITON_IMAGE:-ghcr.io/eliovp/paiton-flux2-klein:rdna4-v1.0.0}" "$package_dir"
