#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The default: decode mode (98,304-token window, speculative decoding on). Options you pass come later and override these.
exec python3 "$script_dir/launch-flashnext.py" --mode decode "$@"
