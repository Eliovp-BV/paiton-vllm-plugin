#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 4-bit experts with the expert host tier (about 41 GiB of pinned RAM; the launcher checks it first). Options you pass come later and override these.
exec python3 "$script_dir/launch-flashnext.py" --mode 4bit --name paiton-flashnext-4bit "$@"
