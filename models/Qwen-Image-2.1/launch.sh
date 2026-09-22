#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PAITON_PYTHON:-python3}" "$package_dir/run.py" "$@"
