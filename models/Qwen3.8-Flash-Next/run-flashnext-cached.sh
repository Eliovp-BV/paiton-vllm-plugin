#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Decode mode with the prefix-caching opt-in (align mode, 2,048-token blocks; byte-identical hits). Options you pass come later.
exec python3 "$script_dir/launch-flashnext.py" --mode decode --prefix-caching --name paiton-flashnext-cached "$@"
